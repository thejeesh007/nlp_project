"""
Feature add: multi-label severity classifier.

The binary toxic/non-toxic model (train.py) collapses Jigsaw's 5 sub-labels
(severe_toxic/obscene/threat/insult/identity_hate) into one union label --
good for a clean toxic/non-toxic decision, but it throws away *why* a
comment was flagged. This trains a second, small DistilBERT head that
predicts all 5 sub-labels independently (multi-label, sigmoid + BCE, not
softmax) so the demo can show a category breakdown alongside the toxic
verdict. Deliberately CPU-feasible (small subsample) since this is a
secondary/explanatory signal, not the primary classifier -- the binary
model, trained on the full 127K rows on GPU, remains the source of truth
for the toxic/non-toxic decision itself.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "jigsaw_toxic_comments.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "distilbert-multilabel")
METRICS_PATH = os.path.join(BASE_DIR, "results", "metrics_multilabel.json")

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
LABEL_COLS = ["severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def _env_int_or_none(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    if val.lower() in ("none", "full", "all"):
        return None
    return int(val)


# CPU default stays small (8K/1K) for the demo-only feature add. Override for
# a GPU run (e.g. in Colab) to use the full label-balanced pool and actually
# fix the rare categories (severe_toxic/threat/identity_hate), which had too
# few positive examples at 3K rows to learn at all (F1=0.0 each):
#   SAMPLE_TRAIN=none SAMPLE_EVAL=none EPOCHS=3 python src/train_multilabel.py
SAMPLE_TRAIN = _env_int_or_none("SAMPLE_TRAIN", 8000)
SAMPLE_EVAL = _env_int_or_none("SAMPLE_EVAL", 1000)
EPOCHS = int(os.environ.get("EPOCHS", 3))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
SEED = 42


def load_and_prep():
    df = pd.read_csv(RAW_PATH)
    df = df.dropna(subset=["comment_text"]).copy()
    df["comment_text"] = df["comment_text"].astype(str).str.strip()
    df = df[df["comment_text"].str.len() > 0]
    for col in LABEL_COLS:
        df[col] = df[col].fillna(0).astype(int)
    df = df.drop_duplicates(subset=["comment_text"])
    # Only comments with at least one sub-label are informative for this
    # head -- the overwhelming majority of rows are all-zero, and keeping
    # them at full scale would just teach the model to always predict zero.
    # Keep a modest all-zero slice too so it doesn't over-fire on clean text.
    has_any = df[LABEL_COLS].max(axis=1) == 1
    pos_df = df[has_any]
    neg_df = df[~has_any].sample(n=min(len(pos_df), (~has_any).sum()), random_state=SEED)
    return pd.concat([pos_df, neg_df], ignore_index=True)


def tokenize_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    labels = df[LABEL_COLS].to_numpy(dtype=np.float32)
    ds = Dataset.from_dict({"comment_text": df["comment_text"].tolist(), "labels": labels.tolist()})

    def _tok(batch):
        return tokenizer(batch["comment_text"], truncation=True, max_length=MAX_LENGTH)

    ds = ds.map(_tok, batched=True)
    ds = ds.remove_columns(["comment_text"])
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    return {"f1_micro": f1_micro, "f1_macro": f1_macro}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    print("Loading + preparing multi-label data...")
    df = load_and_prep()
    print(f"Label-balanced pool: {len(df)} rows")

    train_df, temp_df = train_test_split(df, test_size=0.25, random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)

    if SAMPLE_TRAIN and SAMPLE_TRAIN < len(train_df):
        train_df = train_df.sample(n=SAMPLE_TRAIN, random_state=SEED)
    if SAMPLE_EVAL and SAMPLE_EVAL < len(val_df):
        val_df = val_df.sample(n=SAMPLE_EVAL, random_state=SEED)
    if SAMPLE_EVAL and SAMPLE_EVAL < len(test_df):
        test_df = test_df.sample(n=SAMPLE_EVAL, random_state=SEED)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
        id2label={i: c for i, c in enumerate(LABEL_COLS)},
        label2id={c: i for i, c in enumerate(LABEL_COLS)},
    )

    train_ds = tokenize_dataset(train_df, tokenizer)
    val_ds = tokenize_dataset(val_df, tokenizer)
    test_ds = tokenize_dataset(test_df, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, "results", "train_checkpoints_multilabel"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        report_to=[],
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("Training multi-label head...")
    trainer.train()

    print("Evaluating on test set...")
    test_output = trainer.predict(test_ds)
    probs = 1 / (1 + np.exp(-test_output.predictions))
    preds = (probs >= 0.5).astype(int)
    labels = test_output.label_ids

    per_label = {}
    for i, col in enumerate(LABEL_COLS):
        p, r, f1, _ = precision_recall_fscore_support(
            labels[:, i], preds[:, i], average="binary", zero_division=0
        )
        per_label[col] = {"precision": p, "recall": r, "f1": f1, "support": int(labels[:, i].sum())}
        print(f"  {col:15s}: P={p:.3f} R={r:.3f} F1={f1:.3f} (support={int(labels[:, i].sum())})")

    metrics = {
        "label_cols": LABEL_COLS,
        "per_label": per_label,
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }
    print(json.dumps(metrics, indent=2))

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")

    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Saved model + tokenizer to {MODEL_DIR}")


if __name__ == "__main__":
    main()
