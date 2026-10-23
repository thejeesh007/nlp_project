"""
Milestone 5: Gradio demo.

Text in -> toxicity label + confidence, SHAP-highlighted tokens, and an
optional fairness note when the input contains AAE markers that Milestone
4's audit found the model over-penalizes (see results/fairness_summary.json
and README.md "Milestone 4").
"""

import csv
import datetime
import json
import os
import re

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.calibrated_pipeline import build_calibrated_pipeline

BASE_DIR = os.path.dirname(__file__)
# Overridable so the demo can point at a validated fix experiment instead
# of the production model, e.g.:
#   MODEL_DIR=models/distilbert-toxicity-8k-augmented-v2-experiment \
#   METRICS_PATH=results/metrics_8k_augmented_v2_experiment.json python app.py
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models", "distilbert-toxicity"))
METRICS_PATH = os.environ.get("METRICS_PATH", os.path.join(BASE_DIR, "results", "metrics.json"))

if not os.path.isdir(MODEL_DIR):
    raise SystemExit(f"No fine-tuned model found at {MODEL_DIR}. Run src/train.py first.")

# Use the F1-optimal threshold train.py found on the validation set rather
# than assuming 0.5 -- a class-weighted model's raw probabilities aren't
# calibrated the same way a plain cross-entropy model's are.
THRESHOLD = 0.5
MARKED_THRESHOLD = 0.5
UNMARKED_THRESHOLD = 0.5
if os.path.isfile(METRICS_PATH):
    with open(METRICS_PATH) as f:
        _metrics = json.load(f)
    THRESHOLD = _metrics.get("tuned_threshold", {}).get("threshold", 0.5)
    # Per-dialect calibration (Milestone 4 mitigation #2): fairness_audit.py
    # found the model over-penalizes AAE-marked text at any single global
    # threshold, so train.py calibrates a separate F1-optimal threshold for
    # marked vs. unmarked validation text. Using that here, instead of just
    # warning about the bias after the fact, is an actual mitigation rather
    # than a note.
    _dt = _metrics.get("dialect_thresholds", {})
    MARKED_THRESHOLD = _dt.get("marked_threshold", THRESHOLD)
    UNMARKED_THRESHOLD = _dt.get("unmarked_threshold", THRESHOLD)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
# Temperature-scaled so displayed confidence isn't raw-softmax overconfident
# (see src/calibrate_temperature.py) -- falls back to an untouched pipeline
# (temperature=1.0) if METRICS_PATH has no "calibration" field yet.
clf = build_calibrated_pipeline(
    model, tokenizer, METRICS_PATH,
    top_k=None, device=-1, truncation=True, max_length=128,
)
explainer = shap.Explainer(clf)

# Optional secondary feature: category breakdown (severe_toxic/obscene/
# threat/insult/identity_hate) from src/train_multilabel.py. This is a
# small, CPU-trained explanatory head -- the binary model above remains the
# source of truth for the toxic/non-toxic decision itself. Loaded only if
# present so the demo still runs fine without it.
MULTILABEL_DIR = os.path.join(BASE_DIR, "models", "distilbert-multilabel")
multilabel_clf = None
# Check for a saved weights file, not just the directory -- train_multilabel.py
# creates the directory immediately at start, well before training finishes and
# the model is actually saved, so a bare isdir() check can pick up an empty,
# in-progress directory and crash trying to load a tokenizer from it.
if os.path.isfile(os.path.join(MULTILABEL_DIR, "config.json")):
    _ml_tokenizer = AutoTokenizer.from_pretrained(MULTILABEL_DIR)
    _ml_model = AutoModelForSequenceClassification.from_pretrained(MULTILABEL_DIR)
    multilabel_clf = pipeline(
        "text-classification",
        model=_ml_model,
        tokenizer=_ml_tokenizer,
        top_k=None,
        device=-1,
        truncation=True,
        max_length=128,
    )

# AAE lexical/syntactic markers Milestone 4 found trigger the largest
# false-positive score gaps (habitual "be", "finna"/"tryna", slang like
# "trash"/"deadass"/"no cap"/"on god", multiple negation with "ain't").
FAIRNESS_MARKERS = [
    r"\bfinna\b", r"\btryna\b", r"\bdeadass\b", r"\bno cap\b", r"\bon god\b",
    r"\bain'?t\b", r"\byo\b", r"\bmornin'?\b", r"\bactin'?\b", r"\btrippin'?\b",
    r"\btrash\b", r"\bwe be\b", r"\bshe be\b", r"\bhe be\b", r"\bthey be\b",
]
FAIRNESS_RE = re.compile("|".join(FAIRNESS_MARKERS), re.IGNORECASE)

# Separate, non-dialect bug found while testing this demo directly: words
# that are BOTH ordinary literal nouns AND common English insults ("that's
# trash" / "you're garbage" / "what junk") carry a toxic prior so strong
# from training data that even a purely literal use ("trash bag", "garbage
# bag", "junk drawer", "dumpster") gets flagged toxic with ~99% confidence,
# in any dialect. Verified this is specific to insult-homographs, not
# "garbage concepts" generally: "waste bin", "recycling bin", "compost bin",
# and "litter box" (true synonyms that aren't also common insults) do NOT
# trigger it. A generic "is the model over-relying on one word" detector
# based on SHAP-value concentration was tried and rejected -- it doesn't
# reliably separate spurious calls (e.g. "trash bag") from correct ones
# (e.g. "shut up" has similarly concentrated SHAP mass but is genuinely
# toxic), so this is a curated list, not an automatic detector.
LEXICAL_BIAS_MARKERS = [r"\btrash\b", r"\bgarbage\b", r"\bjunk\b", r"\bdumpster\b"]
LEXICAL_BIAS_RE = re.compile("|".join(LEXICAL_BIAS_MARKERS), re.IGNORECASE)


def category_breakdown(text: str) -> str:
    """Category breakdown from the multi-label head, if loaded. Shows which
    Jigsaw sub-labels (severe_toxic/obscene/threat/insult/identity_hate)
    scored above 0.5, since the binary model only says toxic-or-not and
    throws away *why*."""
    if multilabel_clf is None:
        return ""
    preds = multilabel_clf([text])[0]
    hits = sorted(
        (p for p in preds if p["score"] >= 0.5), key=lambda p: -p["score"]
    )
    if not hits:
        return ""
    parts = ", ".join(f"{p['label'].replace('_', ' ')} ({p['score']:.0%})" for p in hits)
    return f"**Category breakdown:** {parts}"


def classify_text(text: str):
    """Label + score + bias/borderline notes, no SHAP -- shared by the
    single-text path (which adds SHAP highlighting on top) and the batch CSV
    path (which skips SHAP entirely since per-row SHAP over a whole file
    would be too slow for a live demo)."""
    preds = clf([text])[0]
    toxic_score = next(p["score"] for p in preds if p["label"] == "toxic")

    is_marked = bool(FAIRNESS_RE.search(text))
    active_threshold = MARKED_THRESHOLD if is_marked else UNMARKED_THRESHOLD
    label = "TOXIC" if toxic_score >= active_threshold else "NON-TOXIC"
    # Confidence in the predicted class, not always the raw toxic-class
    # score -- a NON-TOXIC call with toxic_score=0.0002 is ~100% confident
    # it's non-toxic, not "confidence: 0.0%" (which reads backwards, like
    # the model doubts its own verdict).
    confidence = toxic_score if label == "TOXIC" else (1 - toxic_score)
    label_str = f"{label}  (confidence: {confidence:.1%})"

    notes = []
    if is_marked:
        matched = FAIRNESS_RE.search(text).group(0)
        notes.append(
            f"### ⚠️ Dialect-bias flag\n"
            f"Found the word **\"{matched}\"** — that's slang common in African-American "
            f"English. The model tends to rate this kind of slang as more toxic than "
            f"a Standard English sentence that means the exact same thing. So this "
            f"TOXIC result might just be the slang, not the actual message."
        )

    if toxic_score >= 0.5:
        word_match = LEXICAL_BIAS_RE.search(text)
        if word_match:
            word = word_match.group(0)
            notes.append(
                f"### ⚠️ Word-bias flag\n"
                f"Found the word **\"{word}\"** — it's both a normal object word "
                f"(like a trash bag) and a common insult. The model leans toward "
                f"\"insult\" no matter how the word is actually used, so this TOXIC "
                f"result could be wrong if the word was meant literally."
            )

    # Near the threshold AND not already near-certain -- otherwise, with a
    # high threshold (ours sits at 0.85-0.95), a 99-100% score can be
    # "within 0.1" of the cutoff while being nowhere near actually
    # ambiguous, which used to show a nonsensical "low-confidence" note on
    # a maximally confident result.
    near_threshold = abs(toxic_score - active_threshold) <= 0.1
    not_already_certain = 0.02 < toxic_score < 0.98
    if near_threshold and not_already_certain:
        notes.append(
            f"### ℹ️ Borderline call\n"
            f"This score ({toxic_score:.0%}) is close to the cutoff line, so the "
            f"model isn't very confident either way — worth a second look."
        )

    fairness_note = "\n\n---\n\n".join(notes)
    category_note = category_breakdown(text)
    return label, label_str, toxic_score, fairness_note, category_note


def predict(text):
    if not text or not text.strip():
        return "—", "", [], "", ""

    label, label_str, toxic_score, fairness_note, category_note = classify_text(text)

    sv = explainer([text])[0, :, "toxic"]
    values = np.array(sv.values)
    max_abs = np.abs(values).max() or 1.0
    # Small contributions (<15% of the strongest token) are left unlabeled so
    # the highlight only calls out words that actually drove the prediction.
    highlighted = []
    for tok, val in zip(sv.data, values):
        if not tok.strip():
            continue
        rel = val / max_abs
        if rel > 0.15:
            highlighted.append((str(tok), "pos"))
        elif rel < -0.15:
            highlighted.append((str(tok), "neg"))
        else:
            highlighted.append((str(tok), None))

    return label_str, f"{toxic_score:.4f}", highlighted, fairness_note, category_note


def batch_predict(file):
    """Batch CSV path: label + score + notes for every row, no SHAP (would
    be too slow per-row over a whole file for a live demo)."""
    if file is None:
        return None, None
    path = file.name if hasattr(file, "name") else file
    df = pd.read_csv(path)
    text_col = next((c for c in ("text", "comment_text", "comment") if c in df.columns), df.columns[0])

    labels, scores, notes_list, categories = [], [], [], []
    for text in df[text_col].astype(str):
        label, _, toxic_score, fairness_note, category_note = classify_text(text)
        labels.append(label)
        scores.append(round(float(toxic_score), 4))
        # Collapse the markdown headline notes ("### ⚠️ Dialect-bias flag") down
        # to short tags for a CSV cell -- the full markdown formatting is for
        # the single-text UI, not a spreadsheet column.
        flags = re.findall(r"### .*? (.+)", fairness_note)
        notes_list.append(" | ".join(flags))
        categories.append(category_note.replace("**Category breakdown:** ", ""))

    out = pd.DataFrame({
        "text": df[text_col],
        "label": labels,
        "score": scores,
        "bias_notes": notes_list,
        "categories": categories,
    })
    return out, batch_summary_plot(out)


def batch_summary_plot(df: pd.DataFrame):
    """Aggregate view for a batch run -- the row-by-row table answers 'what
    happened to this one comment', this answers 'what does this whole file
    look like', which is what batch mode is actually for (triaging a real
    comment dump, not looking up one row at a time)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))

    counts = df["label"].value_counts()
    ax1.bar(counts.index, counts.values, color=["#B65A2E" if l == "TOXIC" else "#2F7D63" for l in counts.index])
    ax1.set_title(f"Label split (n={len(df)})")
    ax1.set_ylabel("Rows")
    for i, v in enumerate(counts.values):
        ax1.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    cat_counts = {}
    for cats in df["categories"]:
        if not cats:
            continue
        for part in cats.split(", "):
            name = part.split(" (")[0].strip()
            if name:
                cat_counts[name] = cat_counts.get(name, 0) + 1
    if cat_counts:
        names = sorted(cat_counts, key=cat_counts.get, reverse=True)
        ax2.bar(names, [cat_counts[n] for n in names], color="#2E4A5E")
        ax2.set_title("Category tags (≥50% score)")
        ax2.tick_params(axis="x", rotation=30)
    else:
        ax2.text(0.5, 0.5, "No category tags in this batch", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])

    fig.tight_layout()
    return fig


FEEDBACK_PATH = os.path.join(BASE_DIR, "results", "feedback.csv")


def log_feedback(state, is_correct):
    """Active-learning data collection: every demo interaction a user rates
    becomes a labeled row for future retraining, instead of a throwaway
    prediction. Appends to results/feedback.csv rather than blocking on any
    external service, so this works offline and costs nothing to run."""
    if not state or not state.get("text"):
        return "Nothing to rate yet -- classify some text first."
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    is_new = not os.path.isfile(FEEDBACK_PATH)
    with open(FEEDBACK_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "text", "predicted_label", "score", "user_says_correct"])
        w.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            state["text"],
            state["label"],
            state["score"],
            "yes" if is_correct else "no",
        ])
    verdict = "correct" if is_correct else "incorrect"
    return f"Thanks -- logged as **{verdict}** for future retraining."


with gr.Blocks(title="ToxLens") as demo:
    gr.Markdown(
        "# ToxLens\n"
        "Toxicity classifier (DistilBERT fine-tuned on Jigsaw) with SHAP "
        "word-importance explanations and bias warnings (dialect + word-level)."
    )

    with gr.Tab("Single text"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text to classify",
                    placeholder="Type or paste a sentence...",
                    lines=4,
                )
                fairness_toggle = gr.Checkbox(
                    label="Show bias notes (dialect-bias + word-level-bias warnings)",
                    value=True,
                )
                submit_btn = gr.Button("Classify", variant="primary")

            with gr.Column():
                label_out = gr.Textbox(label="Prediction")
                # Bias warnings placed immediately under the prediction, above
                # the score/SHAP boxes, so they can't be scrolled past or
                # cropped out of a screenshot -- learned the hard way that
                # putting this at the bottom of the column meant people missed
                # it entirely.
                fairness_out = gr.Markdown()
                category_out = gr.Markdown()
                score_out = gr.Textbox(label="Raw toxic score")
                highlight_out = gr.HighlightedText(
                    label="SHAP word importance (red = pushes toward toxic, blue = pushes toward non-toxic)",
                    color_map={"pos": "red", "neg": "blue"},
                )
                with gr.Row():
                    correct_btn = gr.Button("👍 Correct")
                    incorrect_btn = gr.Button("👎 Incorrect")
                feedback_out = gr.Markdown()

        last_prediction = gr.State({})

        def run(text, show_fairness):
            label, score, highlighted, note, category_note = predict(text)
            state = {"text": text, "label": label.split(" ")[0], "score": score}
            return label, (note if show_fairness else ""), category_note, score, highlighted, state, ""

        submit_btn.click(
            run,
            inputs=[text_input, fairness_toggle],
            outputs=[label_out, fairness_out, category_out, score_out, highlight_out, last_prediction, feedback_out],
        )
        text_input.submit(
            run,
            inputs=[text_input, fairness_toggle],
            outputs=[label_out, fairness_out, category_out, score_out, highlight_out, last_prediction, feedback_out],
        )
        correct_btn.click(lambda s: log_feedback(s, True), inputs=[last_prediction], outputs=[feedback_out])
        incorrect_btn.click(lambda s: log_feedback(s, False), inputs=[last_prediction], outputs=[feedback_out])

        gr.Examples(
            examples=[
                ["Traffic was trash this mornin'.", True],
                ["The traffic was terrible this morning.", True],
                ["You are such an idiot, I cannot believe you did that.", True],
                ["We should hang out sometime soon.", True],
                ["trash bag", True],
                ["garbage bag", True],
                ["recycling bin", True],
            ],
            inputs=[text_input, fairness_toggle],
        )

    with gr.Tab("Batch (CSV)"):
        gr.Markdown(
            "Upload a CSV with a `text` (or `comment_text`/`comment`) column -- "
            "every row gets a label, score, bias notes, and category breakdown. "
            "No SHAP highlighting in batch mode (too slow per-row over a whole file)."
        )
        batch_file = gr.File(label="CSV file", file_types=[".csv"])
        batch_btn = gr.Button("Classify file", variant="primary")
        batch_summary_out = gr.Plot(label="Batch summary")
        batch_out = gr.Dataframe(label="Results", wrap=True)

        batch_btn.click(batch_predict, inputs=[batch_file], outputs=[batch_out, batch_summary_out])


if __name__ == "__main__":
    demo.launch()
