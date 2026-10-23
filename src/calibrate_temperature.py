"""
Post-hoc confidence calibration for the production binary model.

Addresses a specific review comment: seeing 100% confidence on some inputs
can look like overfitting. It usually isn't -- modern fine-tuned
transformers are routinely overconfident even when their actual accuracy is
good (Guo et al., "On Calibration of Modern Neural Networks", ICML 2017).
The right way to check is Expected Calibration Error (ECE): bucket
predictions by confidence, and compare each bucket's average confidence to
its *actual* accuracy. A well-calibrated model's 90%-confidence bucket
should be right about 90% of the time; an overconfident model's 90%-bucket
is right less often than that -- which is a calibration problem, not
necessarily an overfitting one (overfitting shows up as a train/test
accuracy gap, which this script also reports, not as high confidence by
itself).

Fix: temperature scaling. Divide the model's logits by a single scalar T
(fit on the validation set to minimize negative log-likelihood) before
softmax. This only rescales confidence -- it doesn't change which class
wins, so classification decisions are preserved once the threshold is
re-tuned on the calibrated scores.

Inference-only (no retraining). Run after any retrain of the production
model (or point MODEL_DIR/METRICS_PATH at another model via env vars).
Writes T and before/after ECE into METRICS_PATH's "calibration" field,
re-tunes "tuned_threshold" and "dialect_thresholds" against the calibrated
scores, and saves a reliability-diagram plot.
"""

import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
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
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models", "distilbert-toxicity"))
METRICS_PATH = os.environ.get("METRICS_PATH", os.path.join(RESULTS_DIR, "metrics.json"))
MAX_LENGTH = 128
N_BINS = 10


def _env_int_or_none(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    if val.lower() in ("none", "full", "all"):
        return None
    return int(val)


# Full val/test (~16K rows each) plus a 5K train sample is reliable but can
# take the better part of an hour on CPU -- override down for a faster,
# still-representative run (e.g. in an environment where a long-running
# background job risks being killed by a reboot/sleep before it finishes).
SAMPLE_TRAIN = _env_int_or_none("SAMPLE_TRAIN", 5000)
SAMPLE_VAL = _env_int_or_none("SAMPLE_VAL", None)
SAMPLE_TEST = _env_int_or_none("SAMPLE_TEST", None)

AAE_MARKERS = [
    r"\bfinna\b", r"\btryna\b", r"\bdeadass\b", r"\bno cap\b", r"\bon god\b",
    r"\bain'?t\b", r"\byo\b", r"\bmornin'?\b", r"\bactin'?\b", r"\btrippin'?\b",
    r"\btrash\b", r"\bwe be\b", r"\bshe be\b", r"\bhe be\b", r"\bthey be\b",
]
AAE_MARKER_RE = re.compile("|".join(AAE_MARKERS), re.IGNORECASE)


def get_logits(model, tokenizer, df: pd.DataFrame) -> np.ndarray:
    ds = Dataset.from_pandas(
        df[["comment_text", "toxic"]].rename(columns={"toxic": "label"}), preserve_index=False
    )
    ds = ds.map(lambda b: tokenizer(b["comment_text"], truncation=True, max_length=MAX_LENGTH), batched=True)
    ds = ds.remove_columns(["comment_text"])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/tmp/calib_scratch", per_device_eval_batch_size=64, report_to=[]),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    out = trainer.predict(ds)
    return out.predictions, out.label_ids


def softmax_toxic_prob(logits: np.ndarray, temperature: float) -> np.ndarray:
    t = torch.tensor(logits) / temperature
    return F.softmax(t, dim=-1)[:, 1].numpy()


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Grid search over T minimizing NLL on validation logits -- simple and
    robust; NLL(T) is well-behaved (unimodal in practice) for this scale."""
    labels_t = torch.tensor(labels, dtype=torch.long)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    best_t, best_nll = 1.0, float("inf")
    for t in np.arange(0.5, 6.01, 0.02):
        scaled = logits_t / t
        nll = F.cross_entropy(scaled, labels_t).item()
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return round(best_t, 2)


def expected_calibration_error(labels: np.ndarray, probs_toxic: np.ndarray, n_bins: int = N_BINS):
    """Standard ECE: bucket by the model's confidence in its own argmax
    decision, compare each bucket's average confidence to its actual
    accuracy. Returns (ece, bin_table) where bin_table has per-bin
    confidence/accuracy/count for the reliability diagram."""
    preds = (probs_toxic >= 0.5).astype(int)
    confidences = np.where(preds == 1, probs_toxic, 1 - probs_toxic)
    correct = (preds == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": 0, "avg_confidence": None, "accuracy": None})
            continue
        bin_conf = float(confidences[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": count, "avg_confidence": bin_conf, "accuracy": bin_acc})
    return ece, rows


def find_best_threshold(labels, probs_toxic):
    best_threshold, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        preds = (probs_toxic >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, round(float(t), 2)
    return best_threshold, best_f1


def plot_reliability(before_rows, after_rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, rows, title in [(axes[0], before_rows, "Before calibration"), (axes[1], after_rows, "After calibration")]:
        xs = [float(r["bin"].split("-")[0]) + 0.05 for r in rows if r["count"] > 0]
        confs = [r["avg_confidence"] for r in rows if r["count"] > 0]
        accs = [r["accuracy"] for r in rows if r["count"] > 0]
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax.bar(xs, accs, width=0.08, alpha=0.6, label="Actual accuracy", color="#2F7D63")
        ax.scatter(xs, confs, color="#B65A2E", zorder=3, label="Avg. confidence")
        ax.set_xlabel("Confidence bucket")
        ax.set_ylabel("Accuracy / confidence")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved reliability diagram to {out_path}")


def main():
    if not os.path.isdir(MODEL_DIR):
        raise SystemExit(f"No model found at {MODEL_DIR}")

    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    if SAMPLE_TRAIN is not None and SAMPLE_TRAIN < len(train_df):
        train_df = train_df.sample(n=SAMPLE_TRAIN, random_state=42)
    if SAMPLE_VAL is not None and SAMPLE_VAL < len(val_df):
        val_df = val_df.sample(n=SAMPLE_VAL, random_state=42)
    if SAMPLE_TEST is not None and SAMPLE_TEST < len(test_df):
        test_df = test_df.sample(n=SAMPLE_TEST, random_state=42)
    print(f"Running inference on train={len(train_df)}, val={len(val_df)}, test={len(test_df)} rows...")
    train_logits, train_labels = get_logits(model, tokenizer, train_df)
    val_logits, val_labels = get_logits(model, tokenizer, val_df)
    test_logits, test_labels = get_logits(model, tokenizer, test_df)

    # Train-vs-test accuracy gap: the actual, direct test for overfitting
    # (a model that memorized the training set does well there and worse on
    # held-out data; confidence level on either set doesn't tell you this).
    train_preds = np.argmax(train_logits, axis=-1)
    test_preds_raw = np.argmax(test_logits, axis=-1)
    train_acc = accuracy_score(train_labels, train_preds)
    test_acc = accuracy_score(test_labels, test_preds_raw)
    print(f"\nTrain accuracy (5K sample): {train_acc:.4f}")
    print(f"Test accuracy:              {test_acc:.4f}")
    print(f"Gap: {train_acc - test_acc:+.4f} (a large positive gap would indicate overfitting; "
          f"this is the actual diagnostic, not raw confidence)")

    print("\nFitting temperature on validation set...")
    temperature = fit_temperature(val_logits, val_labels)
    print(f"Fitted temperature T = {temperature}")

    val_probs_raw = softmax_toxic_prob(val_logits, 1.0)
    val_probs_cal = softmax_toxic_prob(val_logits, temperature)
    ece_before, bins_before = expected_calibration_error(val_labels, val_probs_raw)
    ece_after, bins_after = expected_calibration_error(val_labels, val_probs_cal)
    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration:  {ece_after:.4f}")

    nll_before = F.cross_entropy(torch.tensor(val_logits), torch.tensor(val_labels, dtype=torch.long)).item()
    nll_after = F.cross_entropy(
        torch.tensor(val_logits) / temperature, torch.tensor(val_labels, dtype=torch.long)
    ).item()
    print(f"Val NLL before calibration: {nll_before:.4f}")
    print(f"Val NLL after calibration:  {nll_after:.4f}")

    roc_auc = roc_auc_score(test_labels, softmax_toxic_prob(test_logits, 1.0))
    print(f"Test ROC-AUC (threshold-independent, unaffected by calibration/temperature): {roc_auc:.4f}")

    plot_path = os.path.join(RESULTS_DIR, "calibration_plot.png")
    plot_reliability(bins_before, bins_after, plot_path)

    print("\nRe-tuning decision threshold on calibrated validation probabilities...")
    threshold, val_f1 = find_best_threshold(val_labels, val_probs_cal)
    print(f"New threshold (calibrated): {threshold} (val F1={val_f1:.4f})")

    val_is_marked = val_df["comment_text"].str.contains(AAE_MARKER_RE).to_numpy()
    n_marked = int(val_is_marked.sum())
    dialect_thresholds = {"marked_threshold": threshold, "unmarked_threshold": threshold, "n_marked": n_marked}
    if n_marked >= 20 and (len(val_is_marked) - n_marked) >= 20:
        t_marked, _ = find_best_threshold(val_labels[val_is_marked], val_probs_cal[val_is_marked])
        t_unmarked, _ = find_best_threshold(val_labels[~val_is_marked], val_probs_cal[~val_is_marked])
        # Same floor fix as calibrate_dialect_threshold.py: the known bias
        # only ever inflates marked-text scores, so the marked threshold must
        # never end up lower than the unmarked one.
        if t_marked < t_unmarked:
            print(f"  marked threshold {t_marked} < unmarked {t_unmarked} -- clamping to {t_unmarked}")
            t_marked = t_unmarked
        dialect_thresholds = {
            "marked_threshold": t_marked,
            "unmarked_threshold": t_unmarked,
            "n_marked": n_marked,
            "n_unmarked": len(val_is_marked) - n_marked,
        }
        print(f"  marked (n={n_marked}): threshold={t_marked}")
        print(f"  unmarked (n={len(val_is_marked)-n_marked}): threshold={t_unmarked}")

    print("\nEvaluating on test set with calibrated probabilities + new threshold...")
    test_probs_cal = softmax_toxic_prob(test_logits, temperature)
    test_preds_cal = (test_probs_cal >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds_cal, average="binary", zero_division=0
    )
    test_acc_cal = accuracy_score(test_labels, test_preds_cal)
    print(f"Calibrated test metrics: acc={test_acc_cal:.4f} P={precision:.4f} R={recall:.4f} F1={f1:.4f}")

    if not os.path.isfile(METRICS_PATH):
        raise SystemExit(f"No metrics.json found at {METRICS_PATH} to update")
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    metrics["calibration"] = {
        "temperature": temperature,
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "val_nll_before": round(nll_before, 4),
        "val_nll_after": round(nll_after, 4),
        "test_roc_auc": round(float(roc_auc), 4),
        "train_accuracy_sample": round(float(train_acc), 4),
        "train_sample_n": len(train_df),
        "test_accuracy_raw": round(float(test_acc), 4),
        "train_test_accuracy_gap": round(float(train_acc - test_acc), 4),
        "val_n": len(val_df),
        "test_n": len(test_df),
    }
    metrics["tuned_threshold"] = {
        "threshold": threshold,
        "test_accuracy": round(float(test_acc_cal), 4),
        "test_precision": round(float(precision), 4),
        "test_recall": round(float(recall), 4),
        "test_f1": round(float(f1), 4),
    }
    metrics["dialect_thresholds"] = dialect_thresholds

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nUpdated {METRICS_PATH}")


if __name__ == "__main__":
    main()
