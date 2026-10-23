"""
Milestone 2: fine-tune distilbert-base-uncased for binary toxic/non-toxic
classification on the Jigsaw data prepared by data_prep.py.

CPU-only, free-tier friendly: subsamples the (heavily imbalanced, 127K-row)
train split down to a smaller, class-balanced-ish subset so a full run
finishes in minutes rather than hours. Bump SAMPLE_TRAIN/SAMPLE_EVAL (or set
them to None) to train on more data once you have GPU access.
"""

import json
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
RESULTS_DIR = os.path.join(BASE_DIR, "results")
# Overridable so a side-experiment (e.g. testing data augmentation on an 8K
# subsample) can run without clobbering the current production model/metrics.
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models", "distilbert-toxicity"))
METRICS_PATH = os.environ.get("METRICS_PATH", os.path.join(RESULTS_DIR, "metrics.json"))

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128


def _env_int_or_none(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    if val.lower() in ("none", "full", "all"):
        return None
    return int(val)


# Subsample sizes for CPU-feasible training. None = use the full split.
# Override via env vars for a GPU run (e.g. in Colab) without editing code:
#   SAMPLE_TRAIN=none SAMPLE_EVAL=none EPOCHS=3 python src/train.py
SAMPLE_TRAIN = _env_int_or_none("SAMPLE_TRAIN", 8000)
SAMPLE_EVAL = _env_int_or_none("SAMPLE_EVAL", 1000)
EPOCHS = int(os.environ.get("EPOCHS", 3))
SEED = 42

# Same AAE lexical/syntactic markers as app.py's fairness note -- used here
# to calibrate a separate decision threshold for text that looks
# AAE-marked, since the fairness audit found the model over-penalizes it
# at any single global threshold (see Milestone 4 / fairness_audit.py).
AAE_MARKERS = [
    r"\bfinna\b", r"\btryna\b", r"\bdeadass\b", r"\bno cap\b", r"\bon god\b",
    r"\bain'?t\b", r"\byo\b", r"\bmornin'?\b", r"\bactin'?\b", r"\btrippin'?\b",
    r"\btrash\b", r"\bwe be\b", r"\bshe be\b", r"\bhe be\b", r"\bthey be\b",
]
AAE_MARKER_RE = re.compile("|".join(AAE_MARKERS), re.IGNORECASE)


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"))


def stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df

    # Targeted augmentation rows (see data_prep.py) are a tiny fraction of
    # the full non-toxic pool -- a plain stratified sample would drop nearly
    # all of them by chance. Keep every augmented row, then stratified-fill
    # the remainder from the rest so the subsample still hits size n.
    if "augmented" in df.columns and df["augmented"].sum() > 0:
        forced = df[df["augmented"] == 1]
        rest = df[df["augmented"] == 0]
        remaining_n = max(n - len(forced), 0)
        if remaining_n > 0 and remaining_n < len(rest):
            rest_sample, _ = train_test_split(
                rest, train_size=remaining_n, stratify=rest["toxic"], random_state=seed
            )
        else:
            rest_sample = rest
        return pd.concat([forced, rest_sample], ignore_index=True)

    sub, _ = train_test_split(
        df, train_size=n, stratify=df["toxic"], random_state=seed
    )
    return sub


def tokenize_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    ds = Dataset.from_pandas(
        df[["comment_text", "toxic"]].rename(columns={"toxic": "label"}),
        preserve_index=False,
    )

    def _tok(batch):
        return tokenizer(
            batch["comment_text"], truncation=True, max_length=MAX_LENGTH
        )

    ds = ds.map(_tok, batched=True)
    ds = ds.remove_columns(["comment_text"])
    return ds


class WeightedTrainer(Trainer):
    """Trainer with a class-weighted loss, to push recall up on the minority
    (toxic) class instead of treating both classes as equally easy to get
    wrong."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = F.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def find_best_threshold(labels, probs_toxic):
    """Scan thresholds on validation probabilities to maximize F1 on the
    toxic class, instead of assuming the default 0.5 cutoff is optimal."""
    best_threshold, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        preds = (probs_toxic >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_threshold = f1, round(float(t), 2)
    return best_threshold, best_f1


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def plot_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["non-toxic", "toxic"])
    ax.set_yticklabels(["non-toxic", "toxic"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved confusion matrix plot to {out_path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading processed splits...")
    train_df = stratified_subsample(load_split("train"), SAMPLE_TRAIN, SEED)
    val_df = stratified_subsample(load_split("val"), SAMPLE_EVAL, SEED)
    test_df = stratified_subsample(load_split("test"), SAMPLE_EVAL, SEED)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "non-toxic", 1: "toxic"},
        label2id={"non-toxic": 0, "toxic": 1},
    )

    train_ds = tokenize_dataset(train_df, tokenizer)
    val_ds = tokenize_dataset(val_df, tokenizer)
    test_ds = tokenize_dataset(test_df, tokenizer)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Inverse-frequency class weights so the loss stops treating a missed
    # toxic comment (~1 in 9 rows) the same as a missed non-toxic one.
    n_total = len(train_df)
    n_toxic = int(train_df["toxic"].sum())
    n_nontoxic = n_total - n_toxic
    class_weights = torch.tensor(
        [n_total / (2 * n_nontoxic), n_total / (2 * n_toxic)], dtype=torch.float
    )
    print(f"Class weights [non-toxic, toxic]: {class_weights.tolist()}")

    args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "train_checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        report_to=[],
        seed=SEED,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("Training...")
    trainer.train()

    print("Tuning decision threshold on validation set...")
    val_output = trainer.predict(val_ds)
    val_probs_toxic = F.softmax(torch.tensor(val_output.predictions), dim=-1)[:, 1].numpy()
    threshold, val_f1_at_threshold = find_best_threshold(val_output.label_ids, val_probs_toxic)
    print(f"Best threshold: {threshold} (val F1={val_f1_at_threshold:.4f}, vs 0.5 default)")

    print("Calibrating per-dialect thresholds on validation set...")
    val_is_marked = val_df["comment_text"].str.contains(AAE_MARKER_RE).to_numpy()
    n_marked = int(val_is_marked.sum())
    dialect_thresholds = {"marked_threshold": threshold, "unmarked_threshold": threshold, "n_marked": n_marked}
    if n_marked >= 20 and (len(val_is_marked) - n_marked) >= 20:
        # Only split the threshold if there's enough val data on both sides
        # to tune reliably -- with too few marked examples the "optimal"
        # threshold is just noise.
        t_marked, f1_marked = find_best_threshold(
            val_output.label_ids[val_is_marked], val_probs_toxic[val_is_marked]
        )
        t_unmarked, f1_unmarked = find_best_threshold(
            val_output.label_ids[~val_is_marked], val_probs_toxic[~val_is_marked]
        )
        # The fairness audit established the bias only runs one direction:
        # AAE-marked text scores higher than SAE-equivalent text for the same
        # meaning. An independent per-group F1 search can pick t_marked below
        # t_unmarked purely from noise in a small marked val sample, which
        # would make marked text *more* likely to be flagged toxic -- the
        # opposite of a correction. Clamp so it can never go below unmarked.
        if t_marked < t_unmarked:
            print(f"  marked threshold {t_marked} < unmarked {t_unmarked} -- clamping to "
                  f"{t_unmarked} (a lower marked threshold would worsen the known bias)")
            t_marked = t_unmarked
            preds_marked = (val_probs_toxic[val_is_marked] >= t_marked).astype(int)
            _, _, f1_marked, _ = precision_recall_fscore_support(
                val_output.label_ids[val_is_marked], preds_marked, average="binary", zero_division=0
            )
        dialect_thresholds = {
            "marked_threshold": t_marked,
            "unmarked_threshold": t_unmarked,
            "n_marked": n_marked,
            "n_unmarked": len(val_is_marked) - n_marked,
        }
        print(f"  marked (n={n_marked}): threshold={t_marked} (val F1={f1_marked:.4f})")
        print(f"  unmarked (n={len(val_is_marked)-n_marked}): threshold={t_unmarked} (val F1={f1_unmarked:.4f})")
    else:
        print(f"  only {n_marked} AAE-marked rows in this val sample -- too few to "
              f"calibrate separately, falling back to the single global threshold")

    print("Evaluating on test set...")
    test_output = trainer.predict(test_ds)
    labels = test_output.label_ids
    test_probs_toxic = F.softmax(torch.tensor(test_output.predictions), dim=-1)[:, 1].numpy()

    preds_default = (test_probs_toxic >= 0.5).astype(int)
    preds_tuned = (test_probs_toxic >= threshold).astype(int)

    def metrics_for(preds):
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        return acc, precision, recall, f1

    acc_d, prec_d, rec_d, f1_d = metrics_for(preds_default)
    acc_t, prec_t, rec_t, f1_t = metrics_for(preds_tuned)
    cm = confusion_matrix(labels, preds_tuned)

    metrics = {
        "threshold_used_for_confusion_matrix": threshold,
        "default_threshold_0.5": {
            "test_accuracy": acc_d, "test_precision": prec_d, "test_recall": rec_d, "test_f1": f1_d,
        },
        "tuned_threshold": {
            "threshold": threshold,
            "test_accuracy": acc_t, "test_precision": prec_t, "test_recall": rec_t, "test_f1": f1_t,
        },
        "confusion_matrix": cm.tolist(),
        "class_weights": class_weights.tolist(),
        "dialect_thresholds": dialect_thresholds,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }
    print(json.dumps(metrics, indent=2))

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")

    cm_path = METRICS_PATH.replace("metrics", "confusion_matrix").replace(".json", ".png")
    plot_confusion_matrix(cm, cm_path)

    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Saved model + tokenizer to {MODEL_DIR}")


if __name__ == "__main__":
    main()
