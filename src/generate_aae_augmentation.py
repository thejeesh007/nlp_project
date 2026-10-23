"""
Round-3 dialect-bias augmentation: programmatic, template-based generation
instead of hand-typing more individual sentences.

Rounds 1 and 2 (see README "Improvements" sections 1 and 6) hand-wrote 40 and
then 25 more non-toxic AAE/slang sentences. Round 2 specifically targeted
`trash`-as-criticism slang and still didn't move the held-out audit case
("Traffic was trash this mornin'.") -- the likely reason is structural
narrowness: ~25 sentences almost all shaped like "This X is trash, reason."
teach the model one phrasing pattern, not the general pattern that "trash"
can mean "bad" as a casual adjective in many sentence shapes.

This script instead cross-products a list of everyday topics against many
*structurally distinct* non-toxic sentence templates -- short exclamations,
"been trash" habitual-aspect forms, mid-sentence slang insertions, combined
with other AAE markers -- to generate a few hundred examples covering far
more surface variation per concept than hand-writing individual sentences
would, in the time available. It is still a small, targeted set relative to
the 127K-row training set (not a substitute for real AAVE corpus data), and
is reported as such.

Output: data/augmentation/aae_scaled_augmentation.csv
"""

import csv
import itertools
import os
import random

import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_PATH = os.path.join(BASE_DIR, "data", "augmentation", "aae_scaled_augmentation.csv")
AUDIT_PATH = os.path.join(BASE_DIR, "data", "fairness", "aae_sae_pairs.csv")

SEED = 42
TARGET_COUNT = 350

TOPICS = [
    "traffic", "wifi", "the weather", "this test", "homework", "the game last night",
    "this app", "the printer", "the gym", "the bus schedule", "parking here",
    "the signal out here", "my commute", "the service here", "the food here",
    "this class schedule", "the lighting in here", "the audio on that call",
    "my grades this semester", "my sleep schedule", "this subscription",
    "the landlord", "the coach", "my internet", "this keyboard", "the charger",
    "my headphones", "this printer at work", "the customer service line",
    "this movie", "that show's new season", "the concert sound", "my phone battery",
    "this laptop", "the heating in here", "the AC in here", "my commute today",
    "this group chat", "the group project", "my group project partner",
    "this recipe", "the line at the store", "my new shoes", "this jacket",
    "the vending machine", "my roommate's cooking", "the elevator here",
    "this textbook", "the syllabus", "office hours this week", "the dorm wifi",
    "this update", "the new menu", "my fantasy team this week",
]

# Each template takes {topic}. Deliberately varied sentence shapes so the
# model sees "trash" used as a casual adjective across many structures, not
# one repeated pattern.
TRASH_TEMPLATES = [
    "{topic} was trash today.",
    "{topic} trash ngl.",
    "{topic} been trash all week.",
    "ngl {topic} kinda trash right now.",
    "deadass {topic} was trash this time.",
    "{topic} trash fr fr.",
    "not gonna lie, {topic} trash today.",
    "{topic} been trash lately, not gonna hold you.",
    "{topic} was straight trash this mornin'.",
    "{topic} trash, no cap.",
    "honestly {topic} trash right now.",
    "{topic} been real trash this week.",
    "{topic} trash today, that's just facts.",
    "lowkey {topic} trash rn.",
    "{topic} was mad trash today.",
]

# Broader, non-"trash" AAE markers across the same topics -- the remaining
# rounds leaned hard on "trash" specifically; this widens general dialect
# coverage so the fix isn't a one-word patch.
GENERAL_AAE_TEMPLATES = [
    "{topic} be actin' up again.",
    "I'm finna deal with {topic} later.",
    "I'm tryna figure out {topic} right now.",
    "{topic} ain't been right lately.",
    "{topic} be like that sometimes, deadass.",
    "I been dealin' with {topic} all week, on god.",
    "{topic} always actin' different, I swear.",
    "{topic} finna be a problem again, watch.",
    "{topic} be trippin' fr.",
    "I ain't even worried about {topic} no more.",
]


def render(templates, topics, label_prefix):
    rows = []
    for i, (topic, template) in enumerate(itertools.product(topics, templates)):
        text = template.format(topic=topic)
        text = text[0].upper() + text[1:]
        rows.append((f"{label_prefix}_{i}", text))
    return rows


def main():
    random.seed(SEED)

    trash_rows = render(TRASH_TEMPLATES, TOPICS, "trashgen")
    general_rows = render(GENERAL_AAE_TEMPLATES, TOPICS, "aaegen")
    all_rows = trash_rows + general_rows

    # Dedupe (template x topic can coincidentally repeat phrasing) and shuffle
    seen = set()
    deduped = []
    for rid, text in all_rows:
        if text not in seen:
            seen.add(text)
            deduped.append((rid, text))
    random.shuffle(deduped)

    # Never overlap with the held-out fairness audit set -- training on the
    # exact sentences the audit evaluates would test memorization, not
    # generalization (same principle as rounds 1 and 2).
    audit_df = pd.read_csv(AUDIT_PATH)
    audit_texts = set(audit_df["sae_text"].str.strip()) | set(audit_df["aae_text"].str.strip())
    deduped = [(rid, text) for rid, text in deduped if text not in audit_texts]

    # Keep roughly balanced between the two template families, capped at
    # TARGET_COUNT total -- enough to meaningfully widen coverage without
    # dominating the ~127K-row training set.
    trash_final = [r for r in deduped if r[0].startswith("trashgen")][: TARGET_COUNT // 2]
    general_final = [r for r in deduped if r[0].startswith("aaegen")][: TARGET_COUNT // 2]
    final_rows = trash_final + general_final
    random.shuffle(final_rows)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "comment_text", "toxic"])
        for rid, text in final_rows:
            w.writerow([rid, text, 0])

    print(f"Generated {len(final_rows)} rows "
          f"({len(trash_final)} trash-template, {len(general_final)} general-AAE-template)")
    print(f"Saved to {OUT_PATH}")
    print("\nSample:")
    for rid, text in final_rows[:10]:
        print(f"  {text}")


if __name__ == "__main__":
    main()
