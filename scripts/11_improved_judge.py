"""
Step 11: Improved LLM judge prompts + CoT reruns with extended token budget.

Two tasks:
  (A) New pairwise variants with faithfulness-first, anti-length-bias prompts.
      Targets the self-preference failure identified in Part 6:
      LLM picks Cohere on close calls purely because Cohere is longer.
      New variants: pair_v2_no_ref, pair_v2_with_ref

  (B) Re-run the 26 CoT calls that hit the 300-token limit before writing
      the FINAL: marker. New variants: cot_pair_no_ref_AB_v2, cot_pair_no_ref_BA_v2
      with max_tokens=700.

All new records are appended to data/final/llm_judge_scores.jsonl (new variant
names, so no overwrites). Agreement statistics are reported at the end.

Usage:
  python scripts/11_improved_judge.py           # run both tasks
  python scripts/11_improved_judge.py --pairwise # only new pairwise variants
  python scripts/11_improved_judge.py --cot      # only CoT reruns
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

MODEL = "command-a-03-2025"
SLEEP = 0.5
MAX_ARTICLE_WORDS = 500

# ── Improved pairwise prompts ──────────────────────────────────────────────────
# Design rationale:
#   - Explicit faithfulness-first priority ordering
#   - Hard instruction against length/verbosity preference
#   - Explicit "tie" guidance to prevent forced-choice on near-equal summaries

PAIR_V2_NO_REF = """\
Two summaries of the same news article are shown below.

Evaluate them in this priority order:
1. Faithfulness (most important): Does the summary contain ONLY information from the article? \
Penalise any claim that seems invented, speculative, or changes meaning — even subtle ones.
2. Coverage (second): Does it convey all key events, outcomes, and named entities?
3. Fluency (tiebreaker only): Grammar and readability. \
Do NOT prefer a summary just because it is longer or more detailed — \
length is not a quality signal.

If the summaries are roughly equal across all three dimensions, respond "tie".

Summary A:
{summary_A}

Summary B:
{summary_B}

Respond with exactly one word: A, B, or tie."""

PAIR_V2_WITH_REF = """\
Article:
{article}

Two summaries of the article above are shown below.

Evaluate them in this priority order:
1. Faithfulness (most important): Compare each claim against the article. \
Penalise anything not supported — even a paraphrase that subtly shifts meaning.
2. Coverage (second): Does it capture all key events, outcomes, and named entities in the article?
3. Fluency (tiebreaker only): Grammar and readability. \
Do NOT prefer a summary just because it is longer — length is not quality.

If the summaries are roughly equal across all three dimensions, respond "tie".

Summary A:
{summary_A}

Summary B:
{summary_B}

Respond with exactly one word: A, B, or tie."""

# ── Improved CoT prompt — same logic, extended reasoning space ─────────────────
# Only change from original: more explicit FINAL: instruction to ensure it appears
# before the token limit. max_tokens bumped to 700 at call time.

COT_PAIR_V2 = """\
Two summaries of the same news article are shown below.

Summary {label_first}:
{summary_first}

Summary {label_second}:
{summary_second}

Think step by step:
- Assess faithfulness for each summary (are all claims supported by the source article?).
- Assess coverage for each summary (are all key facts present?).
- Assess fluency for each summary (is it grammatical and readable?).
- Weigh the dimensions: faithfulness matters most, fluency is a tiebreaker only.
- Do not prefer a summary just because it is longer.

After your analysis, write your final answer on the LAST line in exactly this format:
FINAL: {label_first}, FINAL: {label_second}, or FINAL: tie"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def truncate(text, max_words):
    return " ".join(text.split()[:max_words])


def parse_pairwise(text, lf, ls):
    t = text.strip()
    if t in (lf, ls, "tie", "Tie"):
        return t.lower() if t in ("tie", "Tie") else t
    m = re.search(r"FINAL:\s*(" + lf + "|" + ls + r"|tie)", t, re.IGNORECASE)
    if m:
        v = m.group(1)
        return "tie" if v.lower() == "tie" else v.upper()
    for token in reversed(t.split()):
        clean = token.strip(".,!?\"'*_#")
        if clean == lf:
            return lf
        if clean == ls:
            return ls
        if clean.lower() == "tie":
            return "tie"
    return None


def llm_pref_to_system(pref, rec):
    if pref in ("tie", None):
        return pref
    return {"A": rec["system_A"], "B": rec["system_B"]}.get(pref)


# ── Task A: improved pairwise ──────────────────────────────────────────────────

def run_pairwise_v2(client, dataset, gold, out_file, completed):
    print("\n── Task A: pair_v2_no_ref + pair_v2_with_ref ──")
    new_variants = [
        ("pair_v2_no_ref",   PAIR_V2_NO_REF,   False),
        ("pair_v2_with_ref", PAIR_V2_WITH_REF, True),
    ]
    ran = 0
    for rec in dataset:
        eid = rec["example_id"]
        article = truncate(rec["article"], MAX_ARTICLE_WORDS)
        for variant_name, template, uses_ref in new_variants:
            # Both AB and BA orderings
            for order, lf, sf, ls, ss in [
                ("AB", "A", rec["summary_A"], "B", rec["summary_B"]),
                ("BA", "B", rec["summary_B"], "A", rec["summary_A"]),
            ]:
                full_variant = f"{variant_name}_{order}"
                key = (eid, full_variant, None)
                if key in completed:
                    continue

                kwargs = {"summary_A": rec["summary_A"], "summary_B": rec["summary_B"]}
                if uses_ref:
                    kwargs["article"] = article
                # Reformat template to use lf/ls ordering
                prompt = template.replace("Summary A:\n{summary_A}", f"Summary {lf}:\n{sf}")
                prompt = prompt.replace("Summary B:\n{summary_B}", f"Summary {ls}:\n{ss}")
                # Fix the response instruction line to match actual labels
                prompt = prompt.replace(
                    "Respond with exactly one word: A, B, or tie.",
                    f"Respond with exactly one word: {lf}, {ls}, or tie."
                )

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

                pref = None if parse_error else parse_pairwise(raw, lf, ls)
                if pref is None and not parse_error:
                    parse_error = True

                result = {
                    "example_id": eid,
                    "mode": "pairwise",
                    "variant": full_variant,
                    "order": order,
                    "reference": uses_ref,
                    "raw_response": raw,
                    "parse_error": parse_error,
                    "preference": pref,
                    "reasoning": None,
                    "target": None,
                    "faithfulness": None,
                    "coverage": None,
                    "fluency": None,
                }
                out_file.write(json.dumps(result) + "\n")
                out_file.flush()
                completed.add(key)
                ran += 1
                time.sleep(SLEEP)

    print(f"  Ran {ran} calls")
    return ran


# ── Task B: CoT rerun with extended tokens ────────────────────────────────────

def run_cot_v2(client, dataset_map, gold, out_file, completed, original_scores):
    print("\n── Task B: CoT v2 (max_tokens=700) for previously failed examples ──")

    # Find which (eid, variant) pairs failed
    failed_pairs = set()
    for r in original_scores:
        if r["mode"] == "pairwise_cot" and r.get("parse_error"):
            # Map original variant to v2 name
            v2 = r["variant"] + "_v2"
            failed_pairs.add((r["example_id"], v2, r.get("order", "AB")))

    ran = 0
    for eid, v2_variant, order in sorted(failed_pairs):
        key = (eid, v2_variant, None)
        if key in completed:
            continue

        rec = dataset_map[eid]
        lf, ls = order[0], order[1]
        sf = rec["summary_A"] if lf == "A" else rec["summary_B"]
        ss = rec["summary_B"] if lf == "A" else rec["summary_A"]

        prompt = COT_PAIR_V2.format(
            label_first=lf, summary_first=sf,
            label_second=ls, summary_second=ss,
        )

        try:
            resp = client.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=700,
            )
            raw = resp.message.content[0].text.strip()
            parse_error = False
        except Exception as e:
            raw = f"API_ERROR: {e}"
            parse_error = True

        pref = None if parse_error else parse_pairwise(raw, lf, ls)
        if pref is None and not parse_error:
            parse_error = True

        # Extract reasoning (text before FINAL:)
        reasoning = None
        if not parse_error:
            split = re.split(r"FINAL:", raw, flags=re.IGNORECASE)
            if len(split) > 1:
                reasoning = split[0].strip()

        result = {
            "example_id": eid,
            "mode": "pairwise_cot",
            "variant": v2_variant,
            "order": order,
            "reference": False,
            "raw_response": raw,
            "parse_error": parse_error,
            "preference": pref,
            "reasoning": reasoning,
            "target": None,
            "faithfulness": None,
            "coverage": None,
            "fluency": None,
        }
        out_file.write(json.dumps(result) + "\n")
        out_file.flush()
        completed.add(key)
        ran += 1
        status = "✓" if not parse_error else "✗"
        print(f"  {status} {eid} / {v2_variant}: {pref}")
        time.sleep(SLEEP)

    print(f"  Ran {ran} calls")
    return ran


# ── Reporting ──────────────────────────────────────────────────────────────────

def report(gold):
    scores = [json.loads(l) for l in Path("data/final/llm_judge_scores.jsonl").read_text().splitlines()]

    print(f"\n{'='*65}")
    print("PAIRWISE AGREEMENT WITH HUMAN GOLD (AB ordering only)")
    print(f"{'='*65}")
    pairwise_variants = [
        "pair_no_ref_AB",
        "pair_with_ref_AB",
        "pair_v2_no_ref_AB",
        "pair_v2_with_ref_AB",
    ]
    for v in pairwise_variants:
        rows = [r for r in scores if r["variant"] == v and not r.get("parse_error")]
        if not rows:
            continue
        correct = 0
        ties_correct = 0
        for r in rows:
            g = gold[r["example_id"]]
            pref_sys = llm_pref_to_system(r["preference"], g)
            human = g["gold_winner"]
            if pref_sys == human:
                correct += 1
            elif pref_sys == "tie" and g["pref_counts"].get("tie", 0) > 0:
                ties_correct += 1
        pct = correct / len(rows)
        print(f"  {v:<28}: {correct}/{len(rows)}  ({pct:.0%})")

    print(f"\n{'='*65}")
    print("CoT PARSE SUCCESS RATE")
    print(f"{'='*65}")
    cot_variants = [
        ("cot_pair_no_ref_AB",    "original"),
        ("cot_pair_no_ref_BA",    "original"),
        ("cot_pair_no_ref_AB_v2", "v2 (700 tok)"),
        ("cot_pair_no_ref_BA_v2", "v2 (700 tok)"),
    ]
    for v, label in cot_variants:
        rows = [r for r in scores if r["variant"] == v]
        if not rows:
            continue
        ok = sum(1 for r in rows if not r.get("parse_error"))
        print(f"  {v:<30}: {ok}/{len(rows)} parsed  ({ok/len(rows):.0%})  [{label}]")

    # CoT v2 agreement
    print(f"\n{'='*65}")
    print("CoT v2 AGREEMENT WITH HUMAN GOLD (among parsed only)")
    print(f"{'='*65}")
    for v in ("cot_pair_no_ref_AB_v2",):
        rows = [r for r in scores if r["variant"] == v and not r.get("parse_error")]
        if not rows:
            continue
        correct = sum(1 for r in rows
                      if llm_pref_to_system(r["preference"], gold[r["example_id"]]) ==
                         gold[r["example_id"]]["gold_winner"])
        print(f"  {v}: {correct}/{len(rows)}  ({correct/len(rows):.0%})")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise", action="store_true", help="Run new pairwise variants only")
    parser.add_argument("--cot",      action="store_true", help="Run CoT reruns only")
    args = parser.parse_args()
    run_all = not args.pairwise and not args.cot

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        print("ERROR: COHERE_API_KEY not set.")
        sys.exit(1)
    import cohere
    client = cohere.ClientV2(api_key=api_key)

    dataset = [json.loads(l) for l in Path("data/final/dataset.jsonl").read_text().splitlines()]
    dataset_map = {r["example_id"]: r for r in dataset}
    gold = {r["example_id"]: r for r in [
        json.loads(l) for l in Path("data/final/gold_scores.jsonl").read_text().splitlines()
    ]}

    out_path = Path("data/final/llm_judge_scores.jsonl")
    original_scores = [json.loads(l) for l in out_path.read_text().splitlines()]

    completed = set()
    for r in original_scores:
        completed.add((r["example_id"], r["variant"], r.get("target")))
    print(f"Existing records: {len(original_scores)}  |  completed keys: {len(completed)}")

    with open(out_path, "a") as out_file:
        if run_all or args.pairwise:
            run_pairwise_v2(client, dataset, gold, out_file, completed)
        if run_all or args.cot:
            run_cot_v2(client, dataset_map, gold, out_file, completed, original_scores)

    report(gold)


if __name__ == "__main__":
    main()
