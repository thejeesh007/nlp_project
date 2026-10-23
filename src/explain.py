"""
Milestone 3: SHAP explainability.

1. Picks a mix of correctly/incorrectly classified test examples and saves
   per-example SHAP word-importance visualizations (HTML) to
   results/shap_examples/.
2. Cross-checks SHAP explanations against HateXplain's human-annotated
   rationales (word spans humans flagged as the reason a post is hateful)
   to see whether the model is "right for the right reasons."

Requires a fine-tuned model at models/distilbert-toxicity (run train.py first).
"""

import json
import os

import numpy as np
import pandas as pd
import shap
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "distilbert-toxicity")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SHAP_DIR = os.path.join(RESULTS_DIR, "shap_examples")

N_TEST_EXAMPLES = 15
N_HATEXPLAIN_EXAMPLES = 15
TOP_K = 5  # top-k SHAP tokens used for the human-rationale agreement check
SEED = 42


def build_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        device=-1,
        truncation=True,
        max_length=128,
    )


def pick_test_examples(clf) -> pd.DataFrame:
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    sample = test_df.sample(n=min(300, len(test_df)), random_state=SEED)

    preds = clf(sample["comment_text"].tolist(), batch_size=16)
    pred_labels = [
        1 if max(p, key=lambda x: x["score"])["label"] == "toxic" else 0
        for p in preds
    ]
    sample = sample.copy()
    sample["pred"] = pred_labels
    sample["correct"] = sample["pred"] == sample["toxic"]

    correct = sample[sample["correct"]]
    incorrect = sample[~sample["correct"]]

    n_incorrect = min(len(incorrect), N_TEST_EXAMPLES // 3 or 1)
    n_correct = N_TEST_EXAMPLES - n_incorrect

    chosen = pd.concat(
        [
            correct.sample(n=min(n_correct, len(correct)), random_state=SEED),
            incorrect.sample(n=n_incorrect, random_state=SEED) if n_incorrect else incorrect,
        ]
    )
    return chosen.reset_index(drop=True)


def explain_and_save(explainer, texts, labels, prefix):
    shap_values = explainer(texts)
    os.makedirs(SHAP_DIR, exist_ok=True)

    for i, text in enumerate(texts):
        html = shap.plots.text(shap_values[i, :, "toxic"], display=False)
        out_path = os.path.join(SHAP_DIR, f"{prefix}_{i:02d}.html")
        with open(out_path, "w") as f:
            f.write(f"<h3>{labels[i]}</h3>\n{html}")
    print(f"Saved {len(texts)} SHAP visualizations to {SHAP_DIR} (prefix={prefix})")
    return shap_values


def load_hatexplain_sample():
    ds = load_dataset("NochnoyRitzar/hatexplain_cleaned", split="train")
    df = ds.to_pandas()
    df = df[df["major_label"] == "Hate Speech"]
    df = df[df["rationales"].map(len) > 0]
    df = df.sample(n=min(N_HATEXPLAIN_EXAMPLES, len(df)), random_state=SEED)
    return df.reset_index(drop=True)


def majority_rationale_tokens(row) -> set:
    rationales = np.array(row["rationales"])  # (n_annotators, n_tokens)
    vote = rationales.mean(axis=0) >= 0.5
    tokens = row["post_tokens"]
    return {tokens[i] for i, flagged in enumerate(vote) if flagged}


def top_shap_words(tokenizer, text: str, sv) -> set:
    # SHAP's masker reports WordPiece pieces with no "##" markers (e.g. "bean",
    # "ers" for "beaners"), so comparing raw sv.data tokens against whole
    # human-rationale words under-counts every slur/rare word that gets split
    # into subwords. Re-tokenize with offsets to regroup pieces into whole
    # words before ranking by summed |SHAP value|. sv.data also carries a
    # leading/trailing empty-string entry for the [CLS]/[SEP] boundary that
    # the tokenizer's own word_ids() doesn't include -- drop those first.
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    word_ids = enc.word_ids()
    offsets = enc["offset_mapping"]

    real = [(tok, val) for tok, val in zip(sv.data, sv.values) if tok.strip()]

    if len(real) != len(word_ids):
        # Fallback: alignment assumption broke (e.g. truncation) -- rank raw pieces.
        order = np.argsort(-np.abs(sv.values))[:TOP_K]
        return {sv.data[i].strip().lower() for i in order if sv.data[i].strip()}

    word_spans = {}
    word_scores = {}
    for (_, val), wid, (start, end) in zip(real, word_ids, offsets):
        if wid is None:
            continue
        span = word_spans.setdefault(wid, [start, end])
        span[0] = min(span[0], start)
        span[1] = max(span[1], end)
        word_scores[wid] = word_scores.get(wid, 0.0) + abs(val)

    ranked = sorted(word_scores.items(), key=lambda kv: -kv[1])[:TOP_K]
    return {text[word_spans[wid][0]:word_spans[wid][1]].lower() for wid, _ in ranked}


def cross_check_with_hatexplain(explainer, tokenizer):
    df = load_hatexplain_sample()
    texts = df["post_text"].tolist()

    shap_values = explain_and_save(
        explainer, texts, [f"HateXplain gold: Hate Speech" for _ in texts], "hatexplain"
    )

    rows = []
    total_overlap = 0.0
    for i, row in df.iterrows():
        rationale_tokens = {t.lower() for t in majority_rationale_tokens(row)}
        shap_tokens = top_shap_words(tokenizer, row["post_text"], shap_values[i, :, "toxic"])
        if not rationale_tokens:
            continue
        overlap = len(rationale_tokens & shap_tokens) / min(len(rationale_tokens), TOP_K)
        total_overlap += overlap
        rows.append(
            {
                "text": row["post_text"],
                "human_rationale_tokens": sorted(rationale_tokens),
                f"top_{TOP_K}_shap_words": sorted(shap_tokens),
                "overlap_ratio": round(overlap, 2),
            }
        )

    agreement_rate = total_overlap / len(rows) if rows else 0.0
    summary = {
        "n_examples": len(rows),
        "top_k": TOP_K,
        "mean_overlap_ratio": round(agreement_rate, 3),
        "examples": rows,
    }
    out_path = os.path.join(RESULTS_DIR, "shap_hatexplain_agreement.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nHateXplain agreement (mean top-{TOP_K} SHAP/human-rationale overlap): "
          f"{agreement_rate:.1%} over {len(rows)} examples")
    print(f"Saved details to {out_path}")


def main():
    if not os.path.isdir(MODEL_DIR):
        raise SystemExit(f"No fine-tuned model found at {MODEL_DIR}. Run src/train.py first.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SHAP_DIR, exist_ok=True)

    clf = build_pipeline()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    explainer = shap.Explainer(clf)

    print("Selecting test examples (mix of correct/incorrect predictions)...")
    examples = pick_test_examples(clf)
    labels = [
        f"gold={'toxic' if r.toxic else 'non-toxic'} pred={'toxic' if r.pred else 'non-toxic'}"
        for r in examples.itertuples()
    ]
    explain_and_save(explainer, examples["comment_text"].tolist(), labels, "test")

    print("\nCross-checking against HateXplain human rationales...")
    cross_check_with_hatexplain(explainer, tokenizer)


if __name__ == "__main__":
    main()
