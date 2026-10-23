"""
Milestone 4: dialect fairness audit.

Runs the fine-tuned model on hand-built pairs of semantically equivalent
sentences -- one written in Standard American English (SAE), one in
African-American English (AAE) -- and measures whether the model's toxicity
score depends on dialect rather than meaning.

The pairs (data/fairness/aae_sae_pairs.csv) are original, hand-written for
this project (not scraped from Sap et al., ACL 2019, whose exact dataset
we deliberately avoid reproducing) and span three categories:
  - neutral: everyday, non-toxic statements (25 pairs)
  - mild: mildly critical/annoyed but not hateful (15 pairs)
  - toxic: genuinely hostile/insulting content (10 pairs)

The neutral and mild categories are the ones that matter most: if the
model scores AAE phrasing of ordinary, non-hateful content as more toxic
than the SAE equivalent, that's dialect bias, not correct classification.
"""

import json
import os
import re

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from calibrated_pipeline import build_calibrated_pipeline

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PAIRS_PATH = os.path.join(BASE_DIR, "data", "fairness", "aae_sae_pairs.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
# Overridable so this can be pointed at a side-experiment model/metrics
# (e.g. testing data augmentation) instead of always the production one.
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models", "distilbert-toxicity"))
METRICS_PATH = os.environ.get("METRICS_PATH", os.path.join(RESULTS_DIR, "metrics.json"))

# Same AAE marker list as train.py/app.py -- used here to test whether the
# per-dialect calibrated thresholds (mitigation #2) actually reduce the flip
# disparity found below with a single global threshold.
AAE_MARKERS = [
    r"\bfinna\b", r"\btryna\b", r"\bdeadass\b", r"\bno cap\b", r"\bon god\b",
    r"\bain'?t\b", r"\byo\b", r"\bmornin'?\b", r"\bactin'?\b", r"\btrippin'?\b",
    r"\btrash\b", r"\bwe be\b", r"\bshe be\b", r"\bhe be\b", r"\bthey be\b",
]
AAE_MARKER_RE = re.compile("|".join(AAE_MARKERS), re.IGNORECASE)


def load_metrics():
    if not os.path.isfile(METRICS_PATH):
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


_metrics = load_metrics()
# Use the F1-optimal threshold train.py found on the validation set for
# whichever model is currently saved, instead of assuming 0.5 -- a
# class-weighted model's raw probabilities aren't calibrated the same way.
THRESHOLD = _metrics.get("tuned_threshold", {}).get("threshold", 0.5)
_dt = _metrics.get("dialect_thresholds", {})
MARKED_THRESHOLD = _dt.get("marked_threshold", THRESHOLD)
UNMARKED_THRESHOLD = _dt.get("unmarked_threshold", THRESHOLD)


def dialect_threshold_for(text: str) -> float:
    return MARKED_THRESHOLD if AAE_MARKER_RE.search(text) else UNMARKED_THRESHOLD


def build_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    # Same temperature-scaling as app.py (src/calibrated_pipeline.py) so the
    # audit's measured score gaps reflect calibrated, not raw-overconfident,
    # probabilities -- consistent with what a user of the demo actually sees.
    return build_calibrated_pipeline(
        model, tokenizer, METRICS_PATH,
        top_k=None, device=-1, truncation=True, max_length=128,
    )


def toxic_scores(clf, texts):
    preds = clf(texts, batch_size=16)
    scores = []
    for p in preds:
        toxic_entry = next(x for x in p if x["label"] == "toxic")
        scores.append(toxic_entry["score"])
    return scores


def main():
    if not os.path.isdir(MODEL_DIR):
        raise SystemExit(f"No fine-tuned model found at {MODEL_DIR}. Run src/train.py first.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(PAIRS_PATH)
    print(f"Loaded {len(df)} AAE/SAE pairs ({dict(df['category'].value_counts())})")
    print(f"Model: {MODEL_DIR}")
    print(f"Using classification threshold: {THRESHOLD} (from {METRICS_PATH} tuned_threshold)")
    if MARKED_THRESHOLD != UNMARKED_THRESHOLD:
        print(f"Per-dialect thresholds available: marked={MARKED_THRESHOLD}, unmarked={UNMARKED_THRESHOLD}")

    clf = build_pipeline()

    df["sae_score"] = toxic_scores(clf, df["sae_text"].tolist())
    df["aae_score"] = toxic_scores(clf, df["aae_text"].tolist())
    df["gap"] = df["aae_score"] - df["sae_score"]
    df["sae_pred"] = (df["sae_score"] >= THRESHOLD).astype(int)
    df["aae_pred"] = (df["aae_score"] >= THRESHOLD).astype(int)
    df["flipped"] = df["sae_pred"] != df["aae_pred"]

    # Same predictions, but using each sentence's own per-dialect calibrated
    # threshold instead of one global cutoff -- tests whether mitigation #2
    # (per-dialect threshold calibration) actually reduces the flip
    # disparity found above, as opposed to just the raw score gap (which no
    # threshold choice can fix -- only retraining/augmentation can).
    df["sae_threshold_dialect"] = df["sae_text"].map(dialect_threshold_for)
    df["aae_threshold_dialect"] = df["aae_text"].map(dialect_threshold_for)
    df["sae_pred_dialect"] = (df["sae_score"] >= df["sae_threshold_dialect"]).astype(int)
    df["aae_pred_dialect"] = (df["aae_score"] >= df["aae_threshold_dialect"]).astype(int)
    df["flipped_dialect"] = df["sae_pred_dialect"] != df["aae_pred_dialect"]

    out_path = os.environ.get("FAIRNESS_RESULTS_PATH", os.path.join(RESULTS_DIR, "fairness_results.csv"))
    df.to_csv(out_path, index=False)
    print(f"Saved per-pair results to {out_path}")

    summary = {"overall": {}, "by_category": {}}

    summary["overall"] = {
        "n_pairs": len(df),
        "mean_gap": round(df["gap"].mean(), 4),
        "mean_abs_gap": round(df["gap"].abs().mean(), 4),
        "n_flipped": int(df["flipped"].sum()),
        "pct_flipped": round(100 * df["flipped"].mean(), 1),
        "n_flipped_sae_nontoxic_aae_toxic": int(((df["sae_pred"] == 0) & (df["aae_pred"] == 1)).sum()),
        "n_flipped_sae_toxic_aae_nontoxic": int(((df["sae_pred"] == 1) & (df["aae_pred"] == 0)).sum()),
    }

    print("\n=== Overall ===")
    for k, v in summary["overall"].items():
        print(f"  {k}: {v}")

    print("\n=== By category ===")
    for cat, group in df.groupby("category"):
        cat_summary = {
            "n_pairs": len(group),
            "mean_gap": round(group["gap"].mean(), 4),
            "mean_sae_score": round(group["sae_score"].mean(), 4),
            "mean_aae_score": round(group["aae_score"].mean(), 4),
            "n_flipped": int(group["flipped"].sum()),
            "pct_flipped": round(100 * group["flipped"].mean(), 1),
        }
        summary["by_category"][cat] = cat_summary
        print(f"  {cat} (n={cat_summary['n_pairs']}): "
              f"mean gap={cat_summary['mean_gap']:+.4f}  "
              f"(SAE avg={cat_summary['mean_sae_score']:.3f}, AAE avg={cat_summary['mean_aae_score']:.3f})  "
              f"flipped={cat_summary['n_flipped']}/{cat_summary['n_pairs']} ({cat_summary['pct_flipped']}%)")

    if MARKED_THRESHOLD != UNMARKED_THRESHOLD:
        n_flipped_dialect = int(df["flipped_dialect"].sum())
        summary["per_dialect_calibration"] = {
            "marked_threshold": MARKED_THRESHOLD,
            "unmarked_threshold": UNMARKED_THRESHOLD,
            "n_flipped_global_threshold": summary["overall"]["n_flipped"],
            "n_flipped_dialect_calibrated": n_flipped_dialect,
            "pct_flipped_dialect_calibrated": round(100 * df["flipped_dialect"].mean(), 1),
        }
        print("\n=== Per-dialect threshold calibration: does it help? ===")
        print(f"  global threshold ({THRESHOLD}): {summary['overall']['n_flipped']}/{len(df)} pairs flipped")
        print(f"  per-dialect thresholds (marked={MARKED_THRESHOLD}, unmarked={UNMARKED_THRESHOLD}): "
              f"{n_flipped_dialect}/{len(df)} pairs flipped")
        print("  Note: this only affects which pairs cross into a different label -- it does not "
              "change the underlying score gap itself (that requires retraining/augmentation).")

    # Worst offenders: neutral/mild pairs where AAE scored highest above SAE
    non_toxic_df = df[df["intended_toxic"] == 0].copy()
    worst = non_toxic_df.sort_values("gap", ascending=False).head(5)
    summary["worst_false_positive_gaps"] = worst[
        ["id", "category", "sae_text", "aae_text", "sae_score", "aae_score", "gap"]
    ].to_dict(orient="records")

    print("\n=== Largest AAE-scored-more-toxic gaps among non-toxic pairs ===")
    for _, row in worst.iterrows():
        print(f"  [{row['category']}] gap={row['gap']:+.3f}  "
              f"SAE={row['sae_score']:.3f} '{row['sae_text']}'  |  "
              f"AAE={row['aae_score']:.3f} '{row['aae_text']}'")

    summary_path = os.environ.get("FAIRNESS_SUMMARY_PATH", os.path.join(RESULTS_DIR, "fairness_summary.json"))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
