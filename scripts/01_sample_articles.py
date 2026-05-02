"""
Step 1: Sample 50 articles from CNN/DailyMail test split.

Sampling strategy:
- 35 standard articles (no challenging flags)
- 15 articles tagged with at least one challenging type:
    numeric_heavy  : >5 distinct numeric tokens  (hallucination trap)
    multi_entity   : >10 capitalised mid-sentence tokens (entity confusion)
    long_article   : >700 words                 (coverage challenge)
    contrastive    : >=3 contrast markers        (nuanced summarisation)

NOTE: facebook/bart-large-cnn was trained on CNN/DailyMail.
      This gives BART an in-domain advantage over GPT-3.5.
      Document this asymmetry in your paper.
"""

import json
import random
import re
from pathlib import Path

from datasets import load_dataset

SEED = 42
N_TOTAL = 50
N_CHALLENGING = 15
MIN_WORDS = 200

# Use formal contrast markers only — "but" and "while" are too common in news.
CONTRAST_MARKERS = ["however", "nevertheless", "whereas", "although", "despite", "notwithstanding"]


def count_numeric_tokens(text: str) -> int:
    return len(re.findall(r"\b\d[\d,./]*\b", text))


def count_mid_sentence_caps(text: str) -> int:
    """Count words that are capitalised but do not follow a sentence boundary."""
    tokens = text.split()
    count = 0
    for i, tok in enumerate(tokens):
        if i == 0:
            continue
        clean = tok.strip(".,!?;:'\"()")
        prev = tokens[i - 1]
        if clean and clean[0].isupper() and prev[-1] not in ".!?":
            count += 1
    return count


def count_contrast_markers(text: str) -> int:
    lower = text.lower()
    return sum(lower.count(m) for m in CONTRAST_MARKERS)


def word_count(text: str) -> int:
    return len(text.split())


def classify(article: str) -> list[str]:
    """
    Thresholds calibrated against CNN/DM test distribution.
    Targets roughly 25–35% of articles as 'challenging'.
    """
    tags = []
    if count_numeric_tokens(article) > 15:
        tags.append("numeric_heavy")
    if count_mid_sentence_caps(article) > 30:
        tags.append("multi_entity")
    if word_count(article) > 850:
        tags.append("long_article")
    if count_contrast_markers(article) >= 4:
        tags.append("contrastive")
    return tags


def main() -> None:
    random.seed(SEED)
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    print("Loading CNN/DailyMail 3.0.0 test split …")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

    challenging: list[dict] = []
    standard: list[dict] = []

    print(f"Classifying {len(dataset)} examples …")
    for i, ex in enumerate(dataset):
        article = ex["article"]
        if word_count(article) < MIN_WORDS:
            continue
        tags = classify(article)
        record = {
            "original_index": i,
            "original_id": ex["id"],
            "article": article,
            "reference_summary": ex["highlights"],
            "word_count": word_count(article),
            "challenging_types": tags,
        }
        if tags:
            challenging.append(record)
        else:
            standard.append(record)

    print(f"  Challenging pool : {len(challenging)}")
    print(f"  Standard pool    : {len(standard)}")

    # Take up to N_CHALLENGING from the challenging pool.
    # If the standard pool is too small to fill the remainder, draw extra from challenging.
    n_c = min(N_CHALLENGING, len(challenging))
    n_s = N_TOTAL - n_c
    if len(standard) < n_s:
        print(f"  WARNING: standard pool too small ({len(standard)}); "
              f"drawing extra from challenging to reach {N_TOTAL}.")
        n_c = N_TOTAL - len(standard)
        n_s = len(standard)

    sampled_c = random.sample(challenging, n_c)
    sampled_s = random.sample(standard, n_s)
    sampled = sampled_c + sampled_s
    random.shuffle(sampled)

    for i, rec in enumerate(sampled):
        rec["example_id"] = f"ex_{i + 1:03d}"

    out_path = Path("data/raw/sampled_articles.jsonl")
    with open(out_path, "w") as f:
        for rec in sampled:
            f.write(json.dumps(rec) + "\n")

    n_c = sum(1 for r in sampled if r["challenging_types"])
    type_counts: dict[str, int] = {}
    for r in sampled:
        for t in r["challenging_types"]:
            type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\nSaved {len(sampled)} articles → {out_path}")
    print(f"  Challenging : {n_c}")
    print(f"  Standard    : {len(sampled) - n_c}")
    print(f"  Type counts : {type_counts}")


if __name__ == "__main__":
    main()
