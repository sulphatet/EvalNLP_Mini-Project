"""
Step 5: Parse Google Form annotation responses → tidy format + gold standard + agreement.

Input  : "News Summarisation Evaluation - LLM as a Judge (Responses) - Form Responses 1.csv"
         annotation/annotation_sheet.csv   (row order = column-block order in the form)
         data/final/system_map.json

Outputs:
  data/final/annotations_tidy.csv    one row per (example_id, annotator)
  data/final/gold_scores.jsonl       mean scores + majority preference per example
  data/final/agreement_report.json   Krippendorff's Alpha per dimension
"""

import csv
import json
import re
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd

RESPONSES_CSV = (
    "News Summarisation Evaluation - LLM as a Judge (Responses) - Form Responses 1.csv"
)

# 8 fields per example block, in this order (matches Google Form column sequence)
BLOCK_FIELDS = [
    "faithfulness_A", "coverage_A", "fluency_A",
    "faithfulness_B", "coverage_B", "fluency_B",
    "overall_preference", "notes",
]
SCORE_FIELDS = ["faithfulness_A", "coverage_A", "fluency_A",
                "faithfulness_B", "coverage_B", "fluency_B"]
BLOCK_SIZE = len(BLOCK_FIELDS)
METADATA_COLS = 2  # Timestamp, Annotator Name


def extract_score(value: str) -> int | None:
    m = re.match(r"^(\d)", str(value).strip())
    return int(m.group(1)) if m else None


def extract_preference(value: str) -> str | None:
    v = str(value).strip()
    if v.startswith("A"):
        return "A"
    if v.startswith("B"):
        return "B"
    if v.lower().startswith("t"):  # Tie / tie
        return "tie"
    return None


def load_example_order() -> list[str]:
    """Read annotation_sheet.csv to get the ordered example_id list."""
    with open("annotation/annotation_sheet.csv") as f:
        reader = csv.DictReader(f)
        return [row["example_id"] for row in reader]


def main() -> None:
    Path("data/final").mkdir(parents=True, exist_ok=True)

    responses_path = Path(RESPONSES_CSV)
    if not responses_path.exists():
        raise FileNotFoundError(
            f"Cannot find: {RESPONSES_CSV}\n"
            "Place it in the project root directory."
        )

    example_ids = load_example_order()
    assert len(example_ids) == 50, f"Expected 50 examples, got {len(example_ids)}"

    system_map: dict[str, dict] = json.loads(
        Path("data/final/system_map.json").read_text()
    )

    # ── Spot-check alignment (print 4 examples for manual verification) ───────
    dataset = {
        json.loads(l)["example_id"]: json.loads(l)
        for l in Path("data/final/dataset.jsonl").read_text().splitlines()
    }
    print("=== Column-block alignment spot-check ===")
    print("Verify these example IDs match the questions in the Google Form:\n")
    for idx in [0, 1, 24, 49]:
        eid = example_ids[idx]
        excerpt = " ".join(dataset[eid]["article"].split()[:15])
        print(f"  Block {idx:2d} → {eid}: {excerpt}…")
    print()

    # ── Parse responses CSV ────────────────────────────────────────────────────
    df = pd.read_csv(responses_path, header=0)
    expected_cols = METADATA_COLS + len(example_ids) * BLOCK_SIZE
    assert len(df.columns) == expected_cols, (
        f"Expected {expected_cols} columns, got {len(df.columns)}. "
        "Check that the CSV has not been modified."
    )

    rows = []
    for _, row in df.iterrows():
        annotator = str(row.iloc[1]).strip()
        for ex_idx, eid in enumerate(example_ids):
            offset = METADATA_COLS + ex_idx * BLOCK_SIZE
            block = row.iloc[offset : offset + BLOCK_SIZE].values

            rows.append(
                {
                    "example_id": eid,
                    "annotator": annotator,
                    "faithfulness_A": extract_score(block[0]),
                    "coverage_A": extract_score(block[1]),
                    "fluency_A": extract_score(block[2]),
                    "faithfulness_B": extract_score(block[3]),
                    "coverage_B": extract_score(block[4]),
                    "fluency_B": extract_score(block[5]),
                    "overall_preference": extract_preference(block[6]),
                    "notes": str(block[7]).strip() if str(block[7]).strip() not in ("nan", "") else "",
                }
            )

    tidy = pd.DataFrame(rows)

    # Validate: no nulls in score fields
    null_scores = tidy[SCORE_FIELDS].isnull().sum()
    if null_scores.any():
        print("WARNING: null scores detected:")
        print(null_scores[null_scores > 0])
    else:
        print(f"Parsed {len(tidy)} annotation rows (no missing scores).")

    tidy.to_csv("data/final/annotations_tidy.csv", index=False)
    print("Saved → data/final/annotations_tidy.csv")

    # ── Krippendorff's Alpha ───────────────────────────────────────────────────
    annotators = tidy["annotator"].unique().tolist()
    agreement: dict[str, float] = {}

    for dim in SCORE_FIELDS:
        pivot = (
            tidy.pivot(index="annotator", columns="example_id", values=dim)
            .reindex(columns=example_ids)  # ensure consistent column order
        )
        alpha = krippendorff.alpha(
            pivot.values.astype(float), level_of_measurement="ordinal"
        )
        agreement[dim] = round(float(alpha), 4)

    # Pairwise preference agreement (% of pairs that agreed)
    pref_matrix = (
        tidy.pivot(index="annotator", columns="example_id", values="overall_preference")
        .reindex(columns=example_ids)
    )
    pairs = [(annotators[i], annotators[j]) for i in range(len(annotators)) for j in range(i+1, len(annotators))]
    pref_agreements = []
    for a1, a2 in pairs:
        row1 = pref_matrix.loc[a1]
        row2 = pref_matrix.loc[a2]
        agree = (row1 == row2).sum()
        pref_agreements.append(round(agree / len(example_ids), 4))

    agreement["preference_pairwise_agreement"] = {
        f"{a1}_vs_{a2}": v for (a1, a2), v in zip(pairs, pref_agreements)
    }
    agreement["preference_mean_agreement"] = round(float(np.mean(pref_agreements)), 4)

    Path("data/final/agreement_report.json").write_text(
        json.dumps(agreement, indent=2)
    )
    print("\nKrippendorff's Alpha:")
    for dim in SCORE_FIELDS:
        print(f"  {dim:<22}: {agreement[dim]}")
    print(f"  Preference mean agreement: {agreement['preference_mean_agreement']}")
    print("Saved → data/final/agreement_report.json")

    # ── Gold standard ──────────────────────────────────────────────────────────
    gold_records = []
    for eid in example_ids:
        ex = tidy[tidy["example_id"] == eid]
        sm = system_map[eid]

        rec: dict = {
            "example_id": eid,
            "system_A": sm["A"],
            "system_B": sm["B"],
        }

        # Mean scores per dimension
        for dim in SCORE_FIELDS:
            rec[f"gold_{dim}"] = round(float(ex[dim].mean()), 3)

        # Majority-vote preference
        prefs = ex["overall_preference"].tolist()
        pref_counts = {
            "A": prefs.count("A"),
            "B": prefs.count("B"),
            "tie": prefs.count("tie"),
        }
        rec["gold_preference"] = max(pref_counts, key=pref_counts.get)
        rec["pref_counts"] = pref_counts

        # System-level scores (map A/B labels → bart/cohere)
        bart_label = "A" if sm["A"] == "bart" else "B"
        cohere_label = "B" if bart_label == "A" else "A"
        for dim_base in ["faithfulness", "coverage", "fluency"]:
            rec[f"gold_{dim_base}_bart"] = rec[f"gold_{dim_base}_{bart_label}"]
            rec[f"gold_{dim_base}_cohere"] = rec[f"gold_{dim_base}_{cohere_label}"]

        # Winning system
        if rec["gold_preference"] == "tie":
            rec["gold_winner"] = "tie"
        else:
            rec["gold_winner"] = sm[rec["gold_preference"]]  # "bart" or "cohere"

        gold_records.append(rec)

    with open("data/final/gold_scores.jsonl", "w") as f:
        for r in gold_records:
            f.write(json.dumps(r) + "\n")
    print("Saved → data/final/gold_scores.jsonl")

    # Summary
    winners = [r["gold_winner"] for r in gold_records]
    print(f"\nGold standard summary:")
    print(f"  BART preferred  : {winners.count('bart')}")
    print(f"  Cohere preferred: {winners.count('cohere')}")
    print(f"  Tie             : {winners.count('tie')}")


if __name__ == "__main__":
    main()
