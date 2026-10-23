# Toxicity/Hate Speech Detector with Explainability and Fairness Auditing

NLP system that classifies text as toxic/non-toxic using a fine-tuned
transformer, explains its predictions with SHAP, and audits it for
dialect-based fairness bias (African-American English vs Standard American
English).

## Status

- [x] Milestone 1 — Environment & data setup
- [x] Milestone 2 — Fine-tune classifier
- [x] Milestone 3 — Explainability layer
- [x] Milestone 4 — Fairness audit
- [x] Milestone 5 — Demo interface
- [x] Milestone 6 — Final report
- [x] Post-review improvement: targeted data augmentation (validated, −59%
      bias) + per-dialect threshold calibration (tested against production,
      null result — see below) + insult-homograph bug found and fixed
- [x] Both fixes merged into the actual production model via full 127K-row
      GPU retrain on Colab (not just an isolated experiment) — see
      "Both fixes merged into the actual production model" below
- [x] Feature: multi-label category breakdown, all 5 categories working
      (retrained on the full label pool via GPU — see "Improvements"
      section 6)
- [x] Feature: CSV batch-upload tab in the demo
- [x] Second GPU round: fairness audit expanded 50→100 pairs (more
      reliable), rare multi-label categories fixed; `trash`-as-slang gap
      partially improved (`take out the trash` fixed) but the core
      held-out case remains an honest, unresolved gap — see "Improvements"
      section 6

## Improvements: testing the recommendations, not just listing them

Two of the mitigations recommended in `FINAL_REPORT.md` were implemented
and tested as an isolated experiment — an 8K-row CPU subsample matched to
the original (pre-full-data) 8K baseline, so the effect of augmentation
alone could be measured without the confound of also changing dataset
size. Model at `models/distilbert-toxicity-8k-augmented-experiment/`,
metrics at `results/metrics_8k_augmented_experiment.json`. **The
production model (`models/distilbert-toxicity/`, full 127K rows) is
untouched by this experiment.**

### 1. Targeted data augmentation — validated, and it works well

`data/augmentation/aae_nontoxic_augmentation.csv` adds 40 hand-written
non-toxic sentences using the exact slang/intensifiers (`trash`,
`deadass`, `on god`, `finna`, etc.) that the fairness audit found driving
nearly all of the remaining bias. `data_prep.py` merges these into the
train split only (never val/test, so evaluation stays honest), and
they're deliberately different sentences from the 50 `fairness_audit.py`
eval pairs — testing whether the model generalizes "these words aren't
inherently toxic," not whether it memorizes the eval set.

**Result, same 50-pair audit, before vs. after (both 8K-row CPU models):**

| Metric | Original 8K | Augmented 8K (+40 rows) |
|---|---|---|
| Mean gap | +0.123 | **+0.051** (−59%) |
| Mean abs gap | 0.124 | **0.065** (−47%) |
| Flip rate | 8.0% | **4.0%** (−50%) |
| F1 (toxic) | 79.2% | 79.6% (no accuracy cost) |

40 extra sentences — 0.5% of the training set — cut the bias roughly in
half at zero cost to accuracy, and this is *more* sample-efficient than
what full-data GPU training alone achieved without augmentation: 16x more
data (127K rows) got the gap to +0.058; 40 targeted rows on just 8K got
it to +0.051.

**But it didn't fix things uniformly**, which is itself an informative
result: `deadass` examples resolved almost completely (worst-case gap
+0.996 → +0.027), but `trash` barely moved (+0.995 → +0.993) despite 7
augmentation sentences using that exact word. `trash` likely carries a
much stronger toxic prior from the full 127K-row corpus (it's common in
genuinely toxic *SAE* comments too, e.g. "this is trash"), so a handful
of non-toxic counter-examples couldn't outweigh it. **Practical
implication:** augmentation needs to be sized per-word based on how
entrenched that word's toxic association already is, not applied evenly.

**Recommended next step** (not yet done — needs another GPU round-trip):
merge this augmentation set into the full 127K-row training data and
retrain on Colab, ideally with more `trash`-specific examples than the
other markers, to combine full-data training's recall gains with
augmentation's fairness gains.

### 2. Per-dialect threshold calibration — tested against production, result: doesn't help

`train.py` calibrates a separate F1-optimal decision threshold for
AAE-marked vs. unmarked validation text, falling back to one global
threshold when there's too little marked validation data (< 20 examples
on either side). `app.py` and `fairness_audit.py` both apply whichever
threshold matches the input text's own marker presence.

At the 8K-row scale, val only had 6 marked examples — too few to
calibrate. So `src/calibrate_dialect_threshold.py` was written to run
this calibration **post-hoc against the production full-data model**
(inference only on its ~16K-row val split, no retraining needed — a
few minutes on CPU):

```bash
python src/calibrate_dialect_threshold.py
```

**Result: 92 marked examples this time — enough to calibrate — and the
answer is the two thresholds came out identical (0.95 for both).** Per-
dialect threshold calibration is a real, honest **null result** at this
scale: it doesn't change any prediction, because the F1-optimal cutoff
for marked and unmarked text happens to coincide once there's enough
data. The score *gap* the fairness audit measures is a property of the
model's raw probabilities, not its decision threshold — no threshold
choice, uniform or per-dialect, can fix that. Only retraining
(augmentation, more/better data) moves the actual bias, which is
exactly what section 1 above demonstrates.

### 3. A second bug, found by using the demo directly: insult-homograph bias

Typing `trash bag` into `app.py` returns **TOXIC at 99.9%** — clearly
wrong, and initially looked like another instance of the dialect bias
above. Testing showed it's broader and not dialect-specific at all:

```
'trash bag'      -> TOXIC 99.9%    'blue sky'       -> NON-TOXIC 0.0%
'garbage bag'    -> TOXIC 99.3%    'coffee cup'     -> NON-TOXIC 0.0%
'junk drawer'    -> TOXIC 98.9%    'red car'        -> NON-TOXIC 0.1%
'dumpster'       -> TOXIC 98.7%    'waste bin'      -> NON-TOXIC 10.1%
                                    'recycling bin'  -> NON-TOXIC 0.0%
                                    'litter box'     -> NON-TOXIC 0.0%
```

Not a general "short phrase" problem (other 2-word phrases classify
fine) and not a "garbage concept" problem (`waste`/`recycling`/`litter`
are unaffected). The precise pattern: **words that are simultaneously
ordinary nouns and common English insults** (`trash`, `garbage`,
`junk`, `dumpster` — as in "you're trash," "what garbage," "that's
junk," "you dumpster fire") carry such a strong toxic prior from
training data that even a purely literal use gets flagged, regardless
of dialect. This is a general **insult-homograph** bias — model
architecture is contextual (DistilBERT uses attention over surrounding
words), but with only 2 words of input there's barely any context to
work with, so the model falls back on the lexical shortcut it learned.

**A generic detector was tried and rejected, for honesty's sake:**
scoring how much of the SHAP attribution concentrates on a single token
seemed like a natural general-purpose "model might be relying on one
word" signal. It doesn't discriminate reliably — `"shut up"` (correctly
toxic) has *higher* SHAP concentration (0.986) than `"trash bag"`
(0.947), and `"go away"` (correctly non-toxic) also scores high
(0.883). There's no clean statistical signature separating spurious
calls from correct ones this way, so rather than ship something that
looks principled but doesn't actually work, `app.py` uses a **curated
word list** (`trash`, `garbage`, `junk`, `dumpster`) instead — narrower,
but honest about what it catches.

`app.py` now shows a separate "Word-bias note" (independent of the
dialect note) whenever one of these words appears and the model scores
the input toxic, including cases with no AAE marker at all (e.g.
`garbage bag`, which the dialect-only note used to miss entirely). A
warning explains the problem; it doesn't fix the underlying wrong
classification.

### 4. Actually fixing the insult-homograph bug — validated

A note isn't a fix. Diagnosis: every existing `trash` augmentation
example used the word *figuratively* ("this update is trash" = bad
quality) — the model had never once seen `trash`/`garbage`/`junk`/
`dumpster` used in their ordinary, literal sense, so it had no way to
learn the distinction at all.

`data/augmentation/literal_object_augmentation.csv` adds 20 sentences
using these words literally ("I need to take out the trash bag before
it starts smelling," "The junk drawer is full of old batteries," etc.),
all labeled non-toxic. Combined with the existing dialect augmentation
and retrained as another isolated 8K-row CPU experiment
(`models/distilbert-toxicity-8k-augmented-v2-experiment/`):

| Phrase | Before | After | Result |
|---|---|---|---|
| `trash bag` | 99.9% | **3.7%** | ✅ Fixed |
| `garbage bag` | 99.3% | **2.9%** | ✅ Fixed |
| `junk drawer` | 98.9% | **0.9%** | ✅ Fixed |
| `take out the trash` | 100.0% | 93.7% | ⚠️ Improved, not fixed |
| `dumpster` (bare word) | 98.7% | 99.3% | ❌ Unchanged |
| `"You are a piece of trash."` | 99.98% | 99.9% | ✅ Still correctly toxic |
| `"You are garbage."` | 99.97% | 99.9% | ✅ Still correctly toxic |

**3 of 5 literal cases genuinely fixed, and — critically — both insult
controls stayed correctly toxic**, so this isn't just suppressing the
word; the model learned the actual distinction for the phrasings it saw
examples of. The two that didn't resolve make sense: `dumpster` alone
as a single word has no surrounding context to disambiguate with, and
`take out the trash` doesn't closely match any of the 20 augmentation
sentences' phrasing — 20 examples teach a pattern, not every phrasing
of it.

**Side effect on dialect fairness:** this combined-augmentation model's
AAE/SAE gap measured +0.070 (vs. +0.051 for the dialect-only
augmentation experiment, vs. +0.123 baseline) — slightly worse than the
dialect-only version, still clearly better than baseline. Likely
run-to-run variance from combining two small augmentation sets on an
8K subsample rather than a real tradeoff, but reported as measured
rather than assumed.

**Not yet done:** this fix, like #1, is validated only on the isolated
8K-row CPU experiment. Bringing it into the production model needs the
same GPU round-trip on the full 127K-row dataset.

### 5. Both fixes merged into the actual production model — done

Both augmentation sets (dialect + literal-object, 60 rows total) were
merged into `data/processed/train.csv` and the **full 127K-row model was
retrained on a Colab T4 GPU** (`SAMPLE_TRAIN=none SAMPLE_EVAL=none
EPOCHS=3 python src/train.py`, ~29 minutes), replacing
`models/distilbert-toxicity/` — this is no longer an isolated experiment,
it's what the live demo runs.

**Verified on the production model itself:**

| Phrase | Result |
|---|---|
| `garbage bag` | ✅ NON-TOXIC (3.0%) |
| `trash bag` | ✅ NON-TOXIC (0.4%) |
| `junk drawer` | ✅ NON-TOXIC (1.5%) |
| `dumpster` | ✅ NON-TOXIC (27.9%) — below threshold, unlike the 8K experiment |
| `"You are a piece of trash."` | ✅ TOXIC (100.0%) — still correctly caught |
| `"Traffic was trash this mornin'."` | ❌ TOXIC (98.5%) — known remaining gap, see below |

Test-set metrics: F1 82.9% at the tuned threshold (0.40), vs. 83.75% for
the pre-augmentation production model (0.95 threshold) — a small (~0.9pt)
F1 cost from the 60 added rows, traded for fixing 4 of the 6 test phrases
above outright and improving the 5th (`dumpster`, no longer above
threshold, vs. still flagged in the 8K experiment — full-data training
plus augmentation combined better than either alone).

*(This table reflects the model right after this first merge. A second
GPU round — section 6 below — added more `trash`-slang augmentation and
retrained again: `dumpster` regressed back to toxic (95.4%) while `take
out the trash` newly fixed, and `"Traffic was trash this mornin'."`
remains unfixed at 99.4%. F1 held at 83.7%/83.4%, matching this
baseline. Current production numbers are in section 6.)*

**A real bug caught in the per-dialect threshold calibration**, not just
in the model: `train.py`'s dialect-threshold search picked
`marked_threshold=0.10` vs. `unmarked_threshold=0.40` on this run — a
*lower* bar for AAE-marked text, which is backwards. The known bias runs
one direction only (marked text scores too high), so a correction should
never lower that bar; doing so actively worsens the bias instead of
fixing it, which was confirmed live: `"Traffic was trash this mornin'."`
jumped to 98.5% TOXIC under the miscalibrated threshold. Root cause: with
only 92 marked validation rows, an independent per-group F1 search is
noisy enough to pick a threshold in the wrong direction. Fixed in both
`train.py` and `src/calibrate_dialect_threshold.py` by clamping
`marked_threshold = max(marked_threshold, unmarked_threshold)` — after
the fix, recalibration correctly falls back to 0.40/0.40 (a null result,
per section 2, but a *safe* one instead of a harmful one).

**Known remaining gap, honestly unfixed:** `"Traffic was trash this
mornin'."` (AAE slang, non-toxic intent) still scores 98.5% toxic. No
threshold fix can address this — the raw model score is too extreme
(threshold would need to exceed 0.985, which would suppress recall on
genuinely toxic text elsewhere). Fixing this needs more targeted
augmentation covering `trash`-as-criticism slang specifically (see
section 1's note that `trash` resolved far less than other markers even
with dedicated augmentation rows) — a data problem, not a threshold or
architecture one.

### 6. A second attempt at the `trash`-slang gap, the rare multi-label categories, and a bigger audit set

Three more things were tried in a second GPU round, all in one Colab
session on top of the already-merged production model:

**`data/augmentation/trash_slang_augmentation.csv`** adds 25 more
non-toxic sentences specifically using bare `trash` as casual criticism
(`"This wifi trash ngl."`, `"That test was straight trash today."`,
etc.) — short, unpunctuated constructions closer to the held-out audit
phrasing than the first round's full "This X is trash, reason." pattern.
Merged into `data_prep.py` (85 augmentation rows total now) and retrained
full-data on GPU again.

**Result: genuinely mixed, reported honestly rather than rounded up to
a win.**

| Phrase | Before this round | After |
|---|---|---|
| `take out the trash` | 93.7% (improved, not fixed) | **0.5%** ✅ newly fixed |
| `garbage bag` / `trash bag` / `junk drawer` | non-toxic | still non-toxic, held |
| `dumpster` (bare word) | 27.9% (non-toxic) | **95.4%** ❌ regressed back to toxic |
| `"Traffic was trash this mornin'."` | 98.5% toxic | **99.4% toxic — still unfixed** |

`take out the trash` flipped to fixed, but `dumpster` alone flipped
*back* to toxic, and the specific held-out case this round targeted
didn't move at all despite 25 dedicated sentences. This is likely
run-to-run variance from a still-small, per-word augmentation set — a
handful of sentences can tip one bare-word case while another loses
whatever thin margin it had, especially for a single word with zero
surrounding context to disambiguate with. **Test F1 held steady at
83.7% (default threshold) / 83.4% (tuned)** — essentially matching the
pre-augmentation full-GPU baseline (83.75%), so this round cost
noticeably less accuracy than the first augmentation round did. Model
and metrics are current production (`models/distilbert-toxicity`,
`results/metrics.json`).

**The multi-label rare-category gap — fixed.** `train_multilabel.py` was
rerun on GPU against the full ~21K-row label-balanced pool (`SAMPLE_TRAIN=
none SAMPLE_EVAL=none EPOCHS=3`) instead of the 3K CPU subsample:

| Category | F1 (3K CPU) | F1 (21K GPU) | Test support |
|---|---|---|---|
| obscene | 0.84 | **0.92** | 1,074 |
| insult | 0.80 | **0.87** | 986 |
| identity_hate | 0.00 | **0.69** | 190 |
| threat | 0.00 | **0.60** | 54 |
| severe_toxic | 0.00 | **0.42** | 183 |

All 5 categories now produce usable predictions — the fix here was
simply more data (5-9x more positive examples per rare category), not a
different approach. `severe_toxic` at 0.42 is the weakest, consistent
with it being the most subjective/overlapping label in the original
Jigsaw annotation scheme.

**The fairness audit set — expanded 50→100 pairs** (`data/fairness/
aae_sae_pairs.csv`, now 40 neutral / 30 mild / 30 toxic, up from 25/15/10),
specifically tripling the `toxic` category since its earlier reversed
mean gap was measured on only 10 pairs. New sentences are original, not
duplicates of the 50 already held out from training. Rerun via
`fairness_audit.py` against the new production model:

| Metric | 50-pair audit | 100-pair audit |
|---|---|---|
| Mean gap (all) | +0.058 | **+0.049** |
| Flip rate | 10.0% | **7.0%** |
| Toxic-category mean gap | −0.022 (n=10) | **−0.063 (n=30)** |
| Toxic-category flips | 1/10 (wrong direction) | 1/30 (wrong direction) |

The `toxic` category's reversed sign — AAE scoring *lower* than SAE on
average for genuinely hostile content — held up with 3x the data and a
lower flip rate, so this looks like a real, stable pattern rather than
n=10 noise: the bias found in this project is specifically an
over-flagging of neutral/mild AAE content, not an general dialect effect
on toxic content. All 7 flips in the 100-pair set are still in the
harmful direction (SAE non-toxic → AAE toxic); none flip the other way.

## Feature: category breakdown (multi-label severity)

`src/train_multilabel.py` trains a second, small DistilBERT head that
predicts Jigsaw's 5 sub-labels independently (`severe_toxic`, `obscene`,
`threat`, `insult`, `identity_hate` — sigmoid + BCE, not softmax) so the
demo can show *why* something was flagged, not just that it was. This is
a secondary, explanatory signal — the binary model above (trained on the
full 127K rows) remains the sole source of truth for the toxic/non-toxic
decision itself; the multi-label head never affects it.

First trained CPU-only on a small, label-balanced 3,000-row subsample —
`obscene`/`insult` worked (F1 0.84/0.80) but `severe_toxic`/`threat`/
`identity_hate` scored F1=0.0 with only 11-41 positive test examples,
too few to learn at all. **Since fixed** by rerunning on the full ~21K-row
label-balanced pool via Colab GPU (see "Improvements" section 6 above):

| Category | F1 (3K, CPU) | F1 (21K, GPU — current) | Support (test) |
|---|---|---|---|
| obscene | 0.84 | **0.92** | 1,074 |
| insult | 0.80 | **0.87** | 986 |
| identity_hate | 0.00 | **0.69** | 190 |
| threat | 0.00 | **0.60** | 54 |
| severe_toxic | 0.00 | **0.42** | 183 |

All 5 categories now produce usable predictions in the demo. The fix was
simply more training data (5-9x more positive examples per rare
category) — same architecture, same approach, just no longer starved of
examples for the rare labels.

`app.py` loads this model if present (checked via a saved `config.json`
file, not just directory existence, since `train_multilabel.py` creates
its output directory before training finishes) and shows a "Category
breakdown" note under the prediction whenever any category scores ≥50%.

## Feature: CSV batch upload

The demo's "Batch (CSV)" tab accepts a CSV with a `text` (or
`comment_text`/`comment`) column and returns label, score, bias notes,
and category breakdown for every row. Deliberately skips SHAP
highlighting in batch mode — per-row SHAP over a whole file would be far
too slow for a live demo, so batch mode trades explainability detail for
throughput; single-text mode keeps full SHAP highlighting.

A summary chart accompanies the per-row table: a label-split bar (toxic
vs. non-toxic counts) and a category-tag bar (how often each Jigsaw
sub-label fired across the file), so batch mode answers "what does this
whole file look like" rather than only "what happened to this one row" —
the point of triaging a real comment dump rather than looking things up
one at a time.

## Feature: feedback logging (active-learning data collection)

Every single-text prediction gets a 👍/👎 pair under the result. Clicking
either appends a row (`timestamp`, `text`, `predicted_label`, `score`,
`user_says_correct`) to `results/feedback.csv` — a real disagreement log
rather than a throwaway interaction. Nothing currently consumes this file
automatically (retraining on it would be a natural next step, not
attempted here); it's a data-collection mechanism, not a closed loop yet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Milestone 1: Data

Dataset: [`tasksource/jigsaw_toxicity`](https://huggingface.co/datasets/tasksource/jigsaw_toxicity)
on HuggingFace — a mirror of the Kaggle "Toxic Comment Classification
Challenge" train set (`comment_text` + 6 binary labels: `toxic`,
`severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`),
apache-2.0 licensed, no auth required.

We use only the labeled train set (159,571 rows) and build our own
80/10/10 split — the original Kaggle `test.csv` is mostly unlabeled
(`-1`) placeholders for the competition leaderboard, so it isn't usable
for evaluation.

**Binary label**: a comment is `toxic` if the original `toxic` column is 1
OR any of the other 5 sub-labels (`severe_toxic`, `obscene`, `threat`,
`insult`, `identity_hate`) is 1. The raw `toxic` column alone misses 929
rows that annotators tagged as obscene/insult/threat/identity_hate but
left `toxic=0` — a labeling inconsistency in the source data. Taking the
union folds those back in, giving a single clean toxic/non-toxic target
(the other 5 columns are dropped from the processed data).

```bash
python src/data_prep.py
```

This downloads the dataset, drops nulls/empty/duplicate rows, keeps casing
and punctuation intact (needed later for SHAP legibility), and writes:

- `data/raw/jigsaw_toxic_comments.csv` — full raw pull
- `data/processed/{train,val,test}.csv` — cleaned 80/10/10 stratified split

Current stats (after cleaning + union labeling, 159,374 rows):

| Split | Rows | Toxic | Non-toxic | Avg. length (words) |
|-------|------|-------|-----------|----------------------|
| train | 127,499 | 10.2% | 89.8% | 67.3 |
| val   | 15,937  | 10.2% | 89.8% | 67.5 |
| test  | 15,938  | 10.2% | 89.8% | 66.9 |

Class imbalance (~1:9 toxic:non-toxic) will matter for Milestone 2 —
worth tracking precision/recall separately, not just accuracy.

## Milestone 2: Fine-tuned classifier

Fine-tuned `distilbert-base-uncased` for 3 epochs. Two versions exist in
this project's history, kept for the record:

1. **8K-row CPU subsample** (`results/metrics_8k_cpu_backup.json`,
   `models/distilbert-toxicity-8k-cpu-backup/`) — the original pass, run
   locally with no GPU available.
2. **Full 127K-row dataset, Google Colab T4 GPU** (current
   `results/metrics.json`, `models/distilbert-toxicity/`) — same
   `train.py`, run with `SAMPLE_TRAIN=none SAMPLE_EVAL=none EPOCHS=3
   python src/train.py` on free GPU compute. **This is the model
   everything below (Milestones 3-5) is now measured against.**

```bash
python src/train.py
```

Both use the same two improvements in `train.py`:
- **Class-weighted loss** (`WeightedTrainer`) — inverse-frequency weights
  so a missed toxic comment (~1 in 9 rows) costs the loss more than a
  missed non-toxic one.
- **Threshold tuning** — after training, scans thresholds on the
  validation set and picks the one that maximizes F1, instead of assuming
  the default 0.5 cutoff is correct.

Test-set results (15,938 held-out rows, 1,619 actually toxic):

| Metric | 8K CPU (tuned, t=0.9) | **127K GPU (default, t=0.5)** | 127K GPU (tuned, t=0.95) |
|--------|---|---|---|
| Accuracy | 95.9% | **96.7%** | 96.7% |
| Precision | 82.1% | **82.4%** | 84.6% |
| Recall | 76.5% | **85.2%** | 83.0% |
| F1 | 79.2% | **83.8%** | 83.8% |

**Full-data training was the real fix.** Recall — the metric that class
weighting alone couldn't move on the 8K subsample (stuck at 76.5%) —
jumped to 85.2% simply from seeing 16x more training data. At this scale,
threshold tuning barely matters: the automatic search still picks 0.95
(it's technically the F1-optimal point on the validation set), but it's
statistically tied with the simpler default 0.5 (83.75% vs 83.81% F1) —
unlike the 8K model, where tuning was a meaningful, deliberate win.
`src/fairness_audit.py` and `app.py` both read whatever threshold
`train.py` found in `results/metrics.json` (currently 0.95), so that's
what Milestones 3-5 below are actually measured at.

Confusion matrix at 0.95 (rows = actual, cols = predicted, n=15,938):

| | Predicted non-toxic | Predicted toxic |
|---|---|---|
| **Actual non-toxic** | 14,074 | 245 |
| **Actual toxic** | 276 | 1,343 |

Accuracy alone would be misleading here — a model that always predicts
"non-toxic" would still score ~90% given the class imbalance.

Model + tokenizer saved to `models/distilbert-toxicity`; metrics
(both thresholds) saved to `results/metrics.json`.

## Milestone 3: Explainability (SHAP)

```bash
python src/explain.py
```

Generates SHAP word-importance visualizations for 15 test examples (mix
of correct/incorrect predictions) and cross-checks against HateXplain's
human-annotated rationales to see whether the model's top tokens agree
with what a person would flag. Outputs to `results/shap_examples/` and
`results/shap_hatexplain_agreement.json`.

**Result: 81.7% agreement** (12-13 of 15 examples), measured against the
current full-data GPU model — unchanged from the 8K CPU model's number
(coincidence; the original unweighted model measured 83.3%, a small
expected shift). Across 15 HateXplain examples, the human-flagged toxic
word appeared in the model's top-5 SHAP words in 12-13 of them (slurs
like `nigger`, `beaners`, `muzzies`, `dyke` were consistently identified
as the top driver of the toxic prediction).

Note: DistilBERT's WordPiece tokenizer splits slurs into subwords
(`beaners` → `bean` + `##ers`), and SHAP reports per-piece scores. The
comparison merges pieces back into whole words (via tokenizer offset
mapping) before ranking — comparing raw subword pieces against
whole-word human rationales instead gave a misleadingly low 13.3%, which
was a bug in the evaluation, not a finding about the model.

## Milestone 4: Fairness audit (AAE vs SAE)

```bash
python src/fairness_audit.py
```

50 hand-written sentence pairs (`data/fairness/aae_sae_pairs.csv`) —
each pair says the same thing once in Standard American English (SAE)
and once in African-American English (AAE), spanning three categories:
25 neutral/everyday statements, 15 mildly critical (but not hateful)
statements, and 10 genuinely hostile statements. Written for this
project rather than sourced from Sap et al. (ACL 2019) directly, to
avoid reproducing their dataset without checking its license, but
following the same methodology: hold meaning constant, vary dialect,
see if the model's score changes.

**Result: the model is still dialect-biased, but full-data training
roughly halved the average gap** compared to the 8K CPU model — and, for
the first time, produced one counter-example.

| Category | n | Mean score gap (AAE − SAE) | SAE avg score | AAE avg score | Flip rate |
|---|---|---|---|---|---|
| neutral | 25 | **+0.080** | 0.001 | 0.081 | 8.0% |
| mild | 15 | **+0.073** | 0.133 | 0.206 | 13.3% |
| toxic | 10 | **−0.022** | 0.994 | 0.972 | 10.0% |
| **all** | **50** | **+0.058** | — | — | **10.0%** |

(For comparison, the 8K CPU model measured +0.123 mean gap, 8.0% flip
rate — full-data training cut the average bias roughly in half.)

Of the 5 pairs that flipped classification, **4 flipped from
SAE=non-toxic to AAE=toxic, and — new this run — 1 flipped the other
way**: `"Nobody even likes you, stop pretending they do."` (SAE, scored
0.973, toxic) vs. `"Ain't nobody even like you, stop actin' like they
do."` (AAE, scored 0.748, non-toxic). So the honest claim is **4-out-of-5
flips penalize AAE, not a clean 100%** — more data made the bias less
absolute, though it didn't eliminate it, and the worst-case gaps below
are still severe.

Worst individual cases (completely benign sentences, huge score gap —
these barely moved from the 8K model):

| SAE (score) | AAE (score) | Gap |
|---|---|---|
| "I do not understand this homework at all." (0.000) | "I don't understand this homework at all, deadass." (0.999) | +0.999 |
| "I cannot stand how slow this computer is." (0.001) | "I can't stand how slow this computer is, deadass." (0.999) | +0.998 |
| "The traffic was terrible this morning." (0.000) | "Traffic was trash this mornin'." (0.996) | +0.995 |
| "This customer service is terrible." (0.052) | "This customer service trash." (0.999) | +0.947 |

**Why this happens:** words like `trash` (AAE slang for "bad") and
intensifiers like `deadass`/`on god` almost never appear in Jigsaw's
mostly-SAE training data except alongside genuinely toxic content, so
the model learned a spurious correlation between AAE slang/syntax and
toxicity rather than learning what toxicity actually is. More training
data reduced this correlation's overall strength but the worst-case
words above are apparently still under-represented in benign contexts
even at 127K rows.

*(This table is Milestone 4's original 50-pair finding, kept as the
historical record. The audit set was later expanded to 100 pairs and
rerun against the augmentation-fixed production model — see
"Improvements" section 6 above for the current numbers: mean gap
+0.058→+0.049, flip rate 10.0%→7.0%, and the `toxic` category's reversed
sign held up with 3x the data.)*

**Recommendation for mitigation** (Milestone 6 conclusion): augment
training data with non-toxic examples specifically containing `trash`,
`deadass`, and similar AAE slang/intensifiers — the pattern above (a
handful of specific words driving nearly all of the remaining bias)
suggests targeted augmentation could close most of this gap without
needing a much larger dataset, and/or calibrate the decision threshold
per-dialect rather than one global cutoff for every input.

Full per-pair results: `results/fairness_results.csv`. Summary stats:
`results/fairness_summary.json`.

## Milestone 5: Demo (Gradio)

```bash
python app.py
```

Launches a local Gradio app: text box in; toxicity label + confidence
score, SHAP word-importance highlighting (red = pushes toward toxic,
blue = pushes toward non-toxic), and an optional fairness-note toggle
that warns when the input contains AAE markers (`finna`, `deadass`,
`trash`, `ain't`, habitual `be`, etc.) flagged by Milestone 4.

Verified end-to-end (re-checked against the full-data GPU model,
threshold 0.95): feeding the demo "Traffic was trash this mornin'."
returns TOXIC at 99.6% confidence with the fairness note active, while
"The traffic was terrible this morning." (same meaning, SAE) returns
NON-TOXIC at 0.0% — reproducing the Milestone 4 bias finding live.

## Milestone 6: Final report

See `FINAL_REPORT.md` for the consolidated write-up: dataset, model,
metrics, explainability findings, fairness findings, and conclusion with
mitigation recommendations.
# nlp_project
