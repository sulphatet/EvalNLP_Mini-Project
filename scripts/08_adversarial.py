"""
Step 8: Adversarial example generation and LLM judge evaluation (Part 5).

WORKFLOW (two-phase):
  Phase 1 (--generate): Use Cohere to create adversarial candidates.
                        Output: data/adversarial/candidates.jsonl
                        YOU MUST REVIEW THIS FILE before Phase 2.
                        For each example, verify the adversarial_summary is
                        genuinely adversarial and the ground_truth_better label
                        is correct. Set human_verified=true when satisfied.

  Phase 2 (--judge):   Run LLM judge on verified examples only.
                        Output: data/adversarial/judge_scores.jsonl

Academic integrity: Do not run Phase 2 until you have read and verified
every adversarial example. The ground truth is defined by construction,
so incorrect adversarial examples corrupt the analysis.

Usage:
  python scripts/08_adversarial.py --generate
  # ... review data/adversarial/candidates.jsonl ...
  python scripts/08_adversarial.py --judge
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

MODEL = "command-a-03-2025"
SLEEP = 0.5

# ── Example selection ──────────────────────────────────────────────────────────
# Manually chosen: mix of challenging and standard, varied topics.
# Selecting from articles with specific verifiable facts (names, numbers, places).
SELECTED_IDS = [
    # Type 1 — fluent but incorrect (5): articles with specific verifiable facts
    "ex_001",  # Opera singer / 80 kingfisher feathers / 120,000 yuan
    "ex_002",  # Ryanair vs Aer Lingus Twitter feud
    "ex_025",  # Paul Gregory fined £795 for blue badge misuse
    "ex_028",  # Tesla Elon Musk $3,000 home battery
    "ex_043",  # Adam Peaty breaks 100m breaststroke world record
    # Type 2 — correct but poorly written (3): degrade fluency while preserving facts
    "ex_009",  # Three friends frightened by (check article)
    "ex_013",  # Hero pilot pulled plane out of terrifying dive
    "ex_030",  # American woman and spider
    # Type 3 — paraphrase vs meaning shift (2): subtle semantic change
    "ex_029",  # Burglars who gutted a family home — swap outcome/actor
    "ex_047",  # Driver deliberately blocked a sneaky motorist — swap who blocked whom
]

ADVERSARIAL_TYPES = {
    "ex_001": "fluent_incorrect",
    "ex_002": "fluent_incorrect",
    "ex_025": "fluent_incorrect",
    "ex_028": "fluent_incorrect",
    "ex_043": "fluent_incorrect",
    "ex_009": "correct_awkward",
    "ex_013": "correct_awkward",
    "ex_030": "correct_awkward",
    "ex_029": "meaning_shift",
    "ex_047": "meaning_shift",
}

# ── Prompts ────────────────────────────────────────────────────────────────────

PROMPT_FLUENT_INCORRECT = """\
You are a text editor. Take the following news article summary and rewrite it \
to be fluent and well-written, but introduce exactly one subtle factual error. \
The error should change a specific detail (a number, name, place, or event outcome) \
to something plausible-sounding but wrong according to the article below.

Article (for reference — do not summarise it, only use it to know what is true):
{article}

Original summary:
{summary}

Rules:
- Keep the same length and writing style
- Change exactly one fact to something plausible but incorrect
- Do not add new sentences or remove content
- Return only the modified summary, nothing else"""

PROMPT_CORRECT_AWKWARD = """\
You are a text editor. Take the following news article summary and rewrite it \
to preserve every fact accurately but make the writing noticeably worse: \
awkward phrasing, passive voice, run-on sentences, and clunky transitions. \
The content must remain completely faithful — only the style should degrade.

Original summary:
{summary}

Rules:
- Keep all facts identical
- Make the writing clearly harder to read but still comprehensible
- Aim for a fluency score of 2/5 (frequent errors, impedes reading)
- Return only the rewritten summary, nothing else"""

PROMPT_MEANING_SHIFT = """\
You are a text editor. Take the following news article summary and create a \
version that looks like a paraphrase but contains a subtle meaning shift — \
change who performed an action, add or remove a negation, or alter a key \
qualifier (e.g. change "guilty" to "not guilty", or swap the subject and object \
of a sentence).

Original summary:
{summary}

Rules:
- The change must affect the core meaning, not just style
- The modified version should still read naturally
- Keep length approximately the same
- Return only the modified summary, nothing else"""

PROMPTS = {
    "fluent_incorrect": PROMPT_FLUENT_INCORRECT,
    "correct_awkward":  PROMPT_CORRECT_AWKWARD,
    "meaning_shift":    PROMPT_MEANING_SHIFT,
}

# ── Judge prompts (reuse from script 06) ──────────────────────────────────────
JUDGE_PAIRWISE = """\
Two summaries of the same news article are shown below. Which is better overall \
in terms of faithfulness, coverage, and fluency?

Summary Original:
{original}

Summary Modified:
{modified}

Respond with exactly one word: Original, Modified, or tie."""

JUDGE_PAIRWISE_SWAPPED = """\
Two summaries of the same news article are shown below. Which is better overall \
in terms of faithfulness, coverage, and fluency?

Summary Modified:
{modified}

Summary Original:
{original}

Respond with exactly one word: Modified, Original, or tie."""

JUDGE_WITH_REF = """\
Article:
{article}

Two summaries of the article above are shown below. Which is better overall \
in terms of faithfulness, coverage, and fluency?

Summary Original:
{original}

Summary Modified:
{modified}

Respond with exactly one word: Original, Modified, or tie."""


# ── Phase 1: Generate ─────────────────────────────────────────────────────────

def generate(client, dataset):
    Path("data/adversarial").mkdir(parents=True, exist_ok=True)
    out_path = Path("data/adversarial/candidates.jsonl")

    completed = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            completed.add(json.loads(line)["example_id"])

    with open(out_path, "a") as f:
        for eid in SELECTED_IDS:
            if eid not in dataset:
                print(f"  SKIP {eid} — not found in dataset")
                continue
            if eid in completed:
                print(f"  SKIP {eid} — already generated")
                continue

            rec = dataset[eid]
            adv_type = ADVERSARIAL_TYPES[eid]
            # Use the Cohere summary as the base (it's better quality)
            base_summary = rec["cohere_summary"]
            article_excerpt = " ".join(rec["article"].split()[:400])

            template = PROMPTS[adv_type]
            if adv_type == "fluent_incorrect":
                prompt = template.format(article=article_excerpt, summary=base_summary)
            else:
                prompt = template.format(summary=base_summary)

            try:
                response = client.chat(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,   # slight randomness to get diverse adversarials
                    max_tokens=300,
                )
                adv_summary = response.message.content[0].text.strip()
            except Exception as e:
                print(f"  ERROR {eid}: {e}")
                continue

            record = {
                "example_id":         eid,
                "adversarial_type":   adv_type,
                "article":            rec["article"],
                "original_summary":   base_summary,
                "adversarial_summary": adv_summary,
                "ground_truth_better": "original",  # by construction
                "human_verified":     False,         # SET TO true AFTER YOU REVIEW
                "notes":              "",
            }
            f.write(json.dumps(record) + "\n")
            print(f"  Generated {eid} ({adv_type})")
            time.sleep(SLEEP)

    print(f"\nSaved candidates → {out_path}")
    print("\n" + "=" * 60)
    print("ACTION REQUIRED — review candidates.jsonl before running --judge")
    print("For each record:")
    print("  1. Read the original_summary and adversarial_summary")
    print("  2. Verify the adversarial_type description is accurate")
    print("  3. Confirm ground_truth_better='original' is correct")
    print("  4. Set human_verified=true")
    print("  5. Add notes about what was changed in the 'notes' field")
    print("=" * 60)


# ── Phase 2: Judge ─────────────────────────────────────────────────────────────

def judge(client):
    candidates_path = Path("data/adversarial/candidates.jsonl")
    if not candidates_path.exists():
        raise FileNotFoundError("Run --generate first.")

    candidates = [json.loads(l) for l in candidates_path.read_text().splitlines()]
    verified = [c for c in candidates if c.get("human_verified")]

    if not verified:
        print("ERROR: No examples have human_verified=true.")
        print("Review candidates.jsonl and set human_verified=true for each confirmed example.")
        sys.exit(1)

    unverified = [c for c in candidates if not c.get("human_verified")]
    if unverified:
        print(f"WARNING: {len(unverified)} examples not yet verified — skipping them.")

    out_path = Path("data/adversarial/judge_scores.jsonl")
    completed = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            completed.add((r["example_id"], r["variant"]))

    judge_variants = [
        ("original_first",  JUDGE_PAIRWISE,        False),
        ("modified_first",  JUDGE_PAIRWISE_SWAPPED, False),
        ("with_ref",        JUDGE_WITH_REF,          True),
    ]

    with open(out_path, "a") as f:
        for cand in verified:
            eid = cand["example_id"]
            for variant_name, template, uses_ref in judge_variants:
                if (eid, variant_name) in completed:
                    continue

                kwargs = {
                    "original": cand["original_summary"],
                    "modified": cand["adversarial_summary"],
                }
                if uses_ref:
                    kwargs["article"] = " ".join(cand["article"].split()[:400])
                prompt = template.format(**kwargs)

                try:
                    resp = client.chat(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=20,
                    )
                    raw = resp.message.content[0].text.strip()
                    parse_error = False
                except Exception as e:
                    raw = f"API_ERROR: {e}"
                    parse_error = True

                # Parse preference — "Original" or "Modified" or "tie"
                pref = None
                if not parse_error:
                    t = raw.lower()
                    if "original" in t:
                        pref = "original"
                    elif "modified" in t:
                        pref = "modified"
                    elif "tie" in t:
                        pref = "tie"
                    else:
                        parse_error = True

                result = {
                    "example_id":        eid,
                    "adversarial_type":  cand["adversarial_type"],
                    "variant":           variant_name,
                    "preference":        pref,
                    "ground_truth":      cand["ground_truth_better"],
                    "correct":           (pref == cand["ground_truth_better"]),
                    "raw_response":      raw,
                    "parse_error":       parse_error,
                }
                f.write(json.dumps(result) + "\n")
                print(f"  {eid} / {variant_name}: {pref}  "
                      f"({'✓' if result['correct'] else '✗'})")
                time.sleep(SLEEP)

    # Summary
    scores = [json.loads(l) for l in out_path.read_text().splitlines()]
    print(f"\n{'='*60}")
    print("Adversarial judge results:")
    for adv_type in ["fluent_incorrect", "correct_awkward", "meaning_shift"]:
        subset = [r for r in scores if r["adversarial_type"] == adv_type and not r["parse_error"]]
        if not subset:
            continue
        correct = sum(1 for r in subset if r["correct"])
        print(f"  {adv_type:<20}: {correct}/{len(subset)} correct "
              f"({correct/len(subset):.0%} LLM got it right)")
    print(f"Saved → {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--judge",    action="store_true")
    args = parser.parse_args()

    if not args.generate and not args.judge:
        parser.print_help()
        sys.exit(1)

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        print("ERROR: COHERE_API_KEY not set.")
        sys.exit(1)

    try:
        import cohere
    except ImportError:
        print("ERROR: pip install cohere")
        sys.exit(1)

    client = cohere.ClientV2(api_key=api_key)

    if args.generate:
        dataset = {
            json.loads(l)["example_id"]: json.loads(l)
            for l in Path("data/final/dataset.jsonl").read_text().splitlines()
        }
        generate(client, dataset)

    if args.judge:
        judge(client)


if __name__ == "__main__":
    main()
