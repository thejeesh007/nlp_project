"""
Milestone 1: load the Jigsaw Toxic Comment Classification dataset, clean it,
split it, and print basic stats.

Dataset source: HuggingFace `tasksource/jigsaw_toxicity` — a mirror of the
Kaggle "Toxic Comment Classification Challenge" train set (same columns:
comment_text + toxic/severe_toxic/obscene/threat/insult/identity_hate),
apache-2.0 licensed, no auth required. The Kaggle test.csv is excluded
deliberately: most of its rows are unlabeled (-1) placeholders for the
original competition leaderboard, so we build our own train/val/test split
from the labeled train set instead.

Binary label: a comment counts as `toxic` if the original `toxic` column is
1 OR any of the other 5 Jigsaw sub-labels (severe_toxic/obscene/threat/
insult/identity_hate) is 1. The raw `toxic` column alone misses ~733 rows
that annotators tagged as obscene/insult/threat/identity_hate but left
toxic=0 -- a labeling inconsistency in the source data. Taking the union
folds those back in and gives a single, clean toxic/non-toxic split.
"""

import os
import re

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
AUGMENTATION_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "augmentation")
# Each file is a targeted, hand-written set of non-toxic sentences meant to
# correct one specific spurious correlation found via the fairness audit or
# by using the demo directly. Add new files here as more are found.
AUGMENTATION_FILES = [
    "aae_nontoxic_augmentation.csv",       # AAE dialect markers (trash/deadass/finna/etc as slang)
    "literal_object_augmentation.csv",     # literal (non-insult) trash/garbage/junk/dumpster
    "trash_slang_augmentation.csv",        # "trash"-as-criticism-slang, round 2
    "aae_scaled_augmentation.csv",         # round 3: programmatic template x topic generation
                                            # (src/generate_aae_augmentation.py) -- 350 structurally
                                            # diverse examples instead of a few dozen hand-typed
                                            # ones, after round 2's narrower set didn't move the
                                            # held-out audit case (see README "Improvements")
]
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    # Collapse repeated whitespace/newlines but keep casing and punctuation
    # intact -- explainability (SHAP) needs the original surface form.
    return WHITESPACE_RE.sub(" ", text).strip()


def load_raw() -> pd.DataFrame:
    ds = load_dataset("tasksource/jigsaw_toxicity", split="train")
    df = ds.to_pandas()
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    df = df.dropna(subset=["comment_text"]).copy()
    df["comment_text"] = df["comment_text"].astype(str).map(normalize_text)
    df = df[df["comment_text"].str.len() > 0]

    for col in LABEL_COLS:
        df[col] = df[col].fillna(0).astype(int)

    df = df.drop_duplicates(subset=["comment_text"])

    after = len(df)
    print(f"Dropped {before - after} rows (nulls/empty/duplicates); {after} remain")
    return df


def to_binary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_before = int(df["toxic"].sum())
    df["toxic"] = df[LABEL_COLS].max(axis=1)
    n_after = int(df["toxic"].sum())
    print(
        f"Union label: toxic count {n_before} -> {n_after} "
        f"(+{n_after - n_before} rows recovered from the other 5 sub-labels)"
    )
    return df[["id", "comment_text", "toxic"]]


def load_augmentation() -> pd.DataFrame:
    # Hand-written non-toxic sentences targeting specific spurious
    # correlations found via the fairness audit or by using the demo
    # directly (see AUGMENTATION_FILES above). Added only to the train
    # split (never val/test) so evaluation stays honest, and deliberately
    # distinct from fairness_audit.py's eval sentences and from app.py's
    # example phrases so this tests generalization, not memorization.
    frames = []
    for fname in AUGMENTATION_FILES:
        path = os.path.join(AUGMENTATION_DIR, fname)
        if not os.path.isfile(path):
            continue
        aug = pd.read_csv(path)
        aug["augmented"] = 1
        frames.append(aug[["id", "comment_text", "toxic", "augmented"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["id", "comment_text", "toxic", "augmented"]
    )


def split(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df, test_size=0.20, stratify=df["toxic"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["toxic"], random_state=42
    )
    return train_df, val_df, test_df


def print_stats(name: str, df: pd.DataFrame):
    n = len(df)
    n_toxic = int(df["toxic"].sum())
    avg_len = df["comment_text"].str.split().map(len).mean()
    print(f"\n--- {name} ({n} rows) ---")
    print(f"  toxic:     {n_toxic:6d} ({100 * n_toxic / n:.1f}%)")
    print(f"  non-toxic: {n - n_toxic:6d} ({100 * (n - n_toxic) / n:.1f}%)")
    print(f"  avg sentence length: {avg_len:.1f} words")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading tasksource/jigsaw_toxicity from HuggingFace...")
    raw_df = load_raw()
    raw_path = os.path.join(RAW_DIR, "jigsaw_toxic_comments.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path} ({len(raw_df)} rows)")

    df = clean(raw_df)

    print("\n=== Overall label breakdown (all 6 Jigsaw labels, before merging) ===")
    for col in LABEL_COLS:
        n_pos = int(df[col].sum())
        print(f"  {col:15s}: {n_pos:6d} ({100 * n_pos / len(df):.2f}%)")

    df = to_binary(df)

    train_df, val_df, test_df = split(df)
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["augmented"] = 0
    val_df["augmented"] = 0
    test_df["augmented"] = 0

    aug_df = load_augmentation()
    if len(aug_df):
        train_df = pd.concat([train_df, aug_df], ignore_index=True)
        print(f"\nAdded {len(aug_df)} targeted non-toxic augmentation rows to train "
              f"(from {len(AUGMENTATION_FILES)} augmentation files)")

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = os.path.join(PROCESSED_DIR, f"{name}.csv")
        split_df.to_csv(out_path, index=False)
        print_stats(name, split_df)
        print(f"  saved to {out_path}")


if __name__ == "__main__":
    main()
