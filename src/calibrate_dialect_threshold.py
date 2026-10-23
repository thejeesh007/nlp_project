"""
Post-hoc per-dialect threshold calibration for an already-trained model.

train.py calibrates dialect_thresholds automatically, but only for models it
trains itself. The production model (models/distilbert-toxicity, full 127K
rows via Colab GPU) predates that code, so this script does the same
calibration -- inference only, no retraining -- against the val split and
writes the result into results/metrics.json's dialect_thresholds field.
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models", "distilbert-toxicity"))
METRICS_PATH = os.environ.get("METRICS_PATH", os.path.join(BASE_DIR, "results", "metrics.json"))
MAX_LENGTH = 128

AAE_MARKERS = [
    r"\bfinna\b", r"\btryna\b", r"\bdeadass\b", r"\bno cap\b", r"\bon god\b",
    r"\bain'?t\b", r"\byo\b", r"\bmornin'?\b", r"\bactin'?\b", r"\btrippin'?\b",
    r"\btrash\b", r"\bwe be\b", r"\bshe be\b", r"\bhe be\b", r"\bthey be\b",
]
AAE_MARKER_RE = re.compile("|".join(AAE_MARKERS), re.IGNORECASE)


def find_best_threshold(labels, probs_toxic):
    best_threshold, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        preds = (probs_toxic >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, round(float(t), 2)
    return best_threshold, best_f1


def main():
    if not os.path.isdir(MODEL_DIR):
        raise SystemExit(f"No model found at {MODEL_DIR}")

    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    print(f"Loaded val split: {len(val_df)} rows")

    is_marked = val_df["comment_text"].str.contains(AAE_MARKER_RE).to_numpy()
    n_marked = int(is_marked.sum())
    print(f"AAE-marked rows in val: {n_marked} / {len(val_df)}")

    ds = Dataset.from_pandas(
        val_df[["comment_text", "toxic"]].rename(columns={"toxic": "label"}), preserve_index=False
    )
    ds = ds.map(lambda b: tokenizer(b["comment_text"], truncation=True, max_length=MAX_LENGTH), batched=True)
    ds = ds.remove_columns(["comment_text"])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/calib_scratch", per_device_eval_batch_size=64, report_to=[]
        ),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    print("Running inference on val set...")
    out = trainer.predict(ds)
    probs_toxic = F.softmax(torch.tensor(out.predictions), dim=-1)[:, 1].numpy()
    labels = out.label_ids

    global_t, global_f1 = find_best_threshold(labels, probs_toxic)
    print(f"Global threshold: {global_t} (F1={global_f1:.4f})")

    dialect_thresholds = {"marked_threshold": global_t, "unmarked_threshold": global_t, "n_marked": n_marked}
    if n_marked >= 20 and (len(is_marked) - n_marked) >= 20:
        t_marked, f1_marked = find_best_threshold(labels[is_marked], probs_toxic[is_marked])
        t_unmarked, f1_unmarked = find_best_threshold(labels[~is_marked], probs_toxic[~is_marked])
        # The fairness audit established the bias only runs one direction:
        # AAE-marked text scores higher than SAE-equivalent text for the same
        # meaning. An independent per-group F1 search can pick t_marked below
        # t_unmarked purely from noise in the ~92 marked val rows, which would
        # make marked text *more* likely to be flagged toxic -- the opposite
        # of a correction. Clamp so the marked threshold can never go below
        # the unmarked one.
        if t_marked < t_unmarked:
            print(f"Marked threshold {t_marked} < unmarked {t_unmarked} -- clamping to {t_unmarked} "
                  f"(an unclamped lower marked threshold would worsen the known bias, not fix it)")
            t_marked = t_unmarked
            preds_marked = (probs_toxic[is_marked] >= t_marked).astype(int)
            _, _, f1_marked, _ = precision_recall_fscore_support(
                labels[is_marked], preds_marked, average="binary", zero_division=0
            )
        dialect_thresholds = {
            "marked_threshold": t_marked,
            "unmarked_threshold": t_unmarked,
            "n_marked": n_marked,
            "n_unmarked": len(is_marked) - n_marked,
        }
        print(f"Marked (n={n_marked}): threshold={t_marked} (F1={f1_marked:.4f})")
        print(f"Unmarked (n={len(is_marked)-n_marked}): threshold={t_unmarked} (F1={f1_unmarked:.4f})")
    else:
        print(f"Only {n_marked} marked rows -- too few to calibrate separately, using global threshold for both")

    if not os.path.isfile(METRICS_PATH):
        raise SystemExit(f"No metrics.json found at {METRICS_PATH} to update")
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    metrics["dialect_thresholds"] = dialect_thresholds
    metrics["dialect_thresholds_calibrated_posthoc"] = True
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nUpdated {METRICS_PATH} with dialect_thresholds: {dialect_thresholds}")


if __name__ == "__main__":
    main()
