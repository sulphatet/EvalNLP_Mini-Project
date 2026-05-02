"""
Step 4: Merge all outputs into a single dataset and produce annotation artefacts.

Outputs
-------
data/final/dataset.jsonl
    Full dataset with both summaries, metadata, and blinded A/B labels.

data/final/system_map.json
    Maps example_id → which system is A and which is B.
    Keep this file private until all annotation is complete.

annotation/annotation_sheet.csv
    Pre-filled sheet for annotators. Contains article excerpts and
    A/B-labelled summaries. Does NOT reveal which system is which.

Blinding
--------
For each example, BART and Cohere are randomly assigned to label A or B.
The assignment uses a fixed seed so it is reproducible.
Reveal the system_map only AFTER all annotators have submitted.
"""

import csv
import json
import random
from pathlib import Path

SEED = 42
EXCERPT_WORDS = 120  # words shown to annotators in the CSV


def load_jsonl(path: Path) -> dict[str, dict]:
    records = {}
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        records[rec["example_id"]] = rec
    return records


def excerpt(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " […]"


def main() -> None:
    random.seed(SEED)
    Path("data/final").mkdir(parents=True, exist_ok=True)
    Path("annotation").mkdir(parents=True, exist_ok=True)

    # ── Load sources ──────────────────────────────────────────────────────────
    articles_path = Path("data/raw/sampled_articles.jsonl")
    bart_path = Path("data/outputs/bart_summaries.jsonl")
    cohere_path = Path("data/outputs/cohere_summaries.jsonl")

    missing = [p for p in [articles_path, bart_path, cohere_path] if not p.exists()]
    if missing:
        for p in missing:
            print(f"MISSING: {p}")
        raise FileNotFoundError("Run scripts 01–03 before this step.")

    articles = load_jsonl(articles_path)
    bart = load_jsonl(bart_path)
    cohere = load_jsonl(cohere_path)

    missing_bart = set(articles) - set(bart)
    missing_cohere = set(articles) - set(cohere)
    if missing_bart:
        raise ValueError(f"BART summaries missing for: {missing_bart}")
    if missing_cohere:
        raise ValueError(f"Cohere summaries missing for: {missing_cohere}")

    # ── Build blinded A/B assignment ──────────────────────────────────────────
    example_ids = sorted(articles.keys())
    system_map: dict[str, dict[str, str]] = {}
    for eid in example_ids:
        if random.random() < 0.5:
            system_map[eid] = {"A": "bart", "B": "cohere"}
        else:
            system_map[eid] = {"A": "cohere", "B": "bart"}

    # ── Build dataset.jsonl ───────────────────────────────────────────────────
    dataset = []
    for eid in example_ids:
        art = articles[eid]
        sm = system_map[eid]
        bart_sum = bart[eid]["bart_summary"]
        cohere_sum = cohere[eid]["cohere_summary"]

        summary_A = bart_sum if sm["A"] == "bart" else cohere_sum
        summary_B = bart_sum if sm["B"] == "bart" else cohere_sum

        record = {
            "example_id": eid,
            "article": art["article"],
            "reference_summary": art["reference_summary"],
            "word_count": art["word_count"],
            "challenging_types": art["challenging_types"],
            "bart_summary": bart_sum,
            "cohere_summary": cohere_sum,
            "label_A": sm["A"],
            "label_B": sm["B"],
            "summary_A": summary_A,
            "summary_B": summary_B,
        }
        dataset.append(record)

    ds_path = Path("data/final/dataset.jsonl")
    with open(ds_path, "w") as f:
        for rec in dataset:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved dataset → {ds_path}  ({len(dataset)} examples)")

    # ── Save system map (keep private until annotation is complete) ───────────
    map_path = Path("data/final/system_map.json")
    with open(map_path, "w") as f:
        json.dump(system_map, f, indent=2)
    print(f"Saved system map → {map_path}  ← DO NOT share with annotators")

    # ── Build annotation sheet ────────────────────────────────────────────────
    fieldnames = [
        "example_id",
        "is_challenging",
        "challenging_types",
        "article_excerpt",
        "summary_A",
        "summary_B",
        # Annotator fills these in:
        "annotator_name",
        "faithfulness_A",
        "faithfulness_B",
        "coverage_A",
        "coverage_B",
        "fluency_A",
        "fluency_B",
        "overall_preference",   # A / B / tie
        "notes",
    ]

    sheet_path = Path("annotation/annotation_sheet.csv")
    with open(sheet_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in dataset:
            writer.writerow(
                {
                    "example_id": rec["example_id"],
                    "is_challenging": "yes" if rec["challenging_types"] else "no",
                    "challenging_types": "; ".join(rec["challenging_types"]) if rec["challenging_types"] else "",
                    "article_excerpt": excerpt(rec["article"], EXCERPT_WORDS),
                    "summary_A": rec["summary_A"],
                    "summary_B": rec["summary_B"],
                    "annotator_name": "",
                    "faithfulness_A": "",
                    "faithfulness_B": "",
                    "coverage_A": "",
                    "coverage_B": "",
                    "fluency_A": "",
                    "fluency_B": "",
                    "overall_preference": "",
                    "notes": "",
                }
            )
    print(f"Saved annotation sheet → {sheet_path}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    n_challenging = sum(1 for r in dataset if r["challenging_types"])
    type_counts: dict[str, int] = {}
    for r in dataset:
        for t in r["challenging_types"]:
            type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\nDataset summary")
    print(f"  Total examples    : {len(dataset)}")
    print(f"  Challenging       : {n_challenging}")
    print(f"  Standard          : {len(dataset) - n_challenging}")
    print(f"  Type breakdown    : {type_counts}")
    print(f"  A=BART, B=Cohere  : {sum(1 for v in system_map.values() if v['A'] == 'bart')}")
    print(f"  A=Cohere, B=BART  : {sum(1 for v in system_map.values() if v['A'] == 'cohere')}")


if __name__ == "__main__":
    main()
