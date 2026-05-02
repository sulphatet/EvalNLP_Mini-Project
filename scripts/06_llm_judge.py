"""
Step 6: LLM-as-Judge evaluation pipeline (Parts 3 + 8).

Runs Cohere command-a-03-2025 as judge across 10 prompt variants covering:
  - Direct scoring        (Parts 3.1)
  - Pairwise comparison   (Parts 3.2) + A/B swap for position bias
  - Rubric-based          (Parts 3.3)
  - Chain-of-thought      (Part 8 extension)

Prompt sensitivity axes:
  - Prompt wording  : direct vs rubric framing
  - Candidate order : AB vs BA (pairwise only)
  - Reference       : with vs without source article

Input  : data/final/dataset.jsonl
Output : data/final/llm_judge_scores.jsonl  (~700 rows)

Record all prompt templates verbatim in your paper appendix.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

MODEL = "command-a-03-2025"
MAX_ARTICLE_WORDS = 500
SLEEP_BETWEEN_CALLS = 0.5  # seconds — respects Cohere free-tier rate limits

# ── Prompt templates ──────────────────────────────────────────────────────────

DIRECT_NO_REF = """\
Rate the quality of the following news article summary on three dimensions.

Summary:
{summary}

Respond with valid JSON only:
{{"faithfulness": <1-5>, "coverage": <1-5>, "fluency": <1-5>}}

Faithfulness: 5=all claims supported by article, 1=major hallucinations
Coverage: 5=all key facts present, 1=misses main point
Fluency: 5=perfect grammar, 1=incomprehensible"""

DIRECT_WITH_REF = """\
Article:
{article}

Rate the quality of the following summary of the article above on three dimensions.

Summary:
{summary}

Respond with valid JSON only:
{{"faithfulness": <1-5>, "coverage": <1-5>, "fluency": <1-5>}}

Faithfulness: 5=all claims supported by article, 1=major hallucinations
Coverage: 5=all key facts present, 1=misses main point
Fluency: 5=perfect grammar, 1=incomprehensible"""

PAIR_NO_REF = """\
Two summaries of the same news article are shown below. Which is better overall?

Summary {label_first}:
{summary_first}

Summary {label_second}:
{summary_second}

Respond with exactly one word: {label_first}, {label_second}, or tie."""

PAIR_WITH_REF = """\
Article:
{article}

Two summaries of the article above are shown below. Which is better overall?

Summary {label_first}:
{summary_first}

Summary {label_second}:
{summary_second}

Respond with exactly one word: {label_first}, {label_second}, or tie."""

RUBRIC_NO_REF = """\
You are an expert evaluator. Score the following summary using this rubric:

Faithfulness (1-5): Does it contain only information from the source?
  5=all claims supported, 4=minor imprecision, 3=one factual error, 2=significant hallucination, 1=contradicts source
Coverage (1-5): Does it capture the key information?
  5=all key facts, 4=minor omissions, 3=key point present but detail missing, 2=significant omissions, 1=misses main point
Fluency (1-5): Is it grammatical and readable?
  5=perfect, 4=minor issues, 3=errors but understandable, 2=frequent errors, 1=incomprehensible

Summary:
{summary}

Respond with valid JSON only:
{{"faithfulness": <1-5>, "coverage": <1-5>, "fluency": <1-5>, "reasoning": "<one sentence>"}}"""

RUBRIC_WITH_REF = """\
You are an expert evaluator. Score the following summary of the article below using this rubric:

Article:
{article}

Faithfulness (1-5): Does it contain only information from the article above?
  5=all claims supported, 4=minor imprecision, 3=one factual error, 2=significant hallucination, 1=contradicts source
Coverage (1-5): Does it capture the key information from the article?
  5=all key facts, 4=minor omissions, 3=key point present but detail missing, 2=significant omissions, 1=misses main point
Fluency (1-5): Is it grammatical and readable?
  5=perfect, 4=minor issues, 3=errors but understandable, 2=frequent errors, 1=incomprehensible

Summary:
{summary}

Respond with valid JSON only:
{{"faithfulness": <1-5>, "coverage": <1-5>, "fluency": <1-5>, "reasoning": "<one sentence>"}}"""

COT_PAIR_NO_REF = """\
Two summaries of the same news article are shown below.

Summary {label_first}:
{summary_first}

Summary {label_second}:
{summary_second}

Think step by step: consider faithfulness, coverage, and fluency for each summary. Then state your final answer.

End your response with exactly one of: FINAL: {label_first}, FINAL: {label_second}, or FINAL: tie"""


# ── Parsing helpers ───────────────────────────────────────────────────────────

def truncate(text: str, max_words: int) -> str:
    return " ".join(text.split()[:max_words])


def parse_scores(text: str) -> tuple[int | None, int | None, int | None, str | None]:
    """Return (faithfulness, coverage, fluency, reasoning) from LLM response."""
    # Try full JSON parse first
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            f = int(obj.get("faithfulness")) if obj.get("faithfulness") is not None else None
            c = int(obj.get("coverage")) if obj.get("coverage") is not None else None
            fl = int(obj.get("fluency")) if obj.get("fluency") is not None else None
            r = str(obj.get("reasoning", "")).strip() or None
            if all(v is not None for v in [f, c, fl]):
                return f, c, fl, r
        except (json.JSONDecodeError, ValueError):
            pass

    # Regex fallback for individual fields
    def extract_field(name: str) -> int | None:
        m = re.search(rf'"{name}"\s*:\s*([1-5])', text)
        return int(m.group(1)) if m else None

    f = extract_field("faithfulness")
    c = extract_field("coverage")
    fl = extract_field("fluency")
    r_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', text)
    r = r_match.group(1) if r_match else None
    return f, c, fl, r


def parse_preference(text: str, label_first: str, label_second: str) -> str | None:
    """Extract preference from pairwise response."""
    t = text.strip()
    # Exact single-word response
    if t in (label_first, label_second, "tie", "Tie"):
        return t.lower() if t in ("tie", "Tie") else t
    # Look for FINAL: marker (CoT)
    m = re.search(r"FINAL:\s*(" + label_first + "|" + label_second + r"|tie)", t, re.IGNORECASE)
    if m:
        val = m.group(1)
        return "tie" if val.lower() == "tie" else val.upper()
    # Last resort: find standalone A/B/tie
    for token in reversed(t.split()):
        clean = token.strip(".,!?\"'")
        if clean == label_first:
            return label_first
        if clean == label_second:
            return label_second
        if clean.lower() == "tie":
            return "tie"
    return None


# ── Variant definitions ───────────────────────────────────────────────────────

def build_variants(rec: dict) -> list[dict]:
    """Return list of call specs for one dataset record."""
    article = truncate(rec["article"], MAX_ARTICLE_WORDS)
    summary_A = rec["summary_A"]
    summary_B = rec["summary_B"]
    variants = []

    # Direct scoring — evaluate each summary separately
    for target, summary in [("A", summary_A), ("B", summary_B)]:
        variants.append({
            "mode": "direct_scoring", "variant": "direct_no_ref",
            "target": target, "reference": False,
            "prompt": DIRECT_NO_REF.format(summary=summary),
        })
        variants.append({
            "mode": "direct_scoring", "variant": "direct_with_ref",
            "target": target, "reference": True,
            "prompt": DIRECT_WITH_REF.format(article=article, summary=summary),
        })

    # Pairwise — AB and BA orderings, with/without reference
    for ref, template in [(False, PAIR_NO_REF), (True, PAIR_WITH_REF)]:
        ref_tag = "with_ref" if ref else "no_ref"
        for order in [("AB", "A", summary_A, "B", summary_B),
                       ("BA", "B", summary_B, "A", summary_A)]:
            tag, lf, sf, ls, ss = order
            kwargs = dict(label_first=lf, summary_first=sf,
                          label_second=ls, summary_second=ss)
            if ref:
                kwargs["article"] = article
            variants.append({
                "mode": "pairwise", "variant": f"pair_{ref_tag}_{tag}",
                "order": tag, "reference": ref,
                "prompt": template.format(**kwargs),
            })

    # Rubric-based — evaluate each summary separately
    for target, summary in [("A", summary_A), ("B", summary_B)]:
        variants.append({
            "mode": "rubric_based", "variant": "rubric_no_ref",
            "target": target, "reference": False,
            "prompt": RUBRIC_NO_REF.format(summary=summary),
        })
        variants.append({
            "mode": "rubric_based", "variant": "rubric_with_ref",
            "target": target, "reference": True,
            "prompt": RUBRIC_WITH_REF.format(article=article, summary=summary),
        })

    # CoT pairwise — AB and BA, no reference
    for order in [("AB", "A", summary_A, "B", summary_B),
                   ("BA", "B", summary_B, "A", summary_A)]:
        tag, lf, sf, ls, ss = order
        variants.append({
            "mode": "pairwise_cot", "variant": f"cot_pair_no_ref_{tag}",
            "order": tag, "reference": False,
            "prompt": COT_PAIR_NO_REF.format(
                label_first=lf, summary_first=sf,
                label_second=ls, summary_second=ss,
            ),
        })

    return variants


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        print("ERROR: COHERE_API_KEY environment variable is not set.")
        sys.exit(1)

    try:
        import cohere
    except ImportError:
        print("ERROR: pip install cohere")
        sys.exit(1)

    dataset_path = Path("data/final/dataset.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError("Run scripts 01–04 first.")

    dataset = [json.loads(l) for l in dataset_path.read_text().splitlines()]

    out_path = Path("data/final/llm_judge_scores.jsonl")
    Path("data/final").mkdir(parents=True, exist_ok=True)

    # Resume: track already-completed (example_id, variant) pairs
    completed: set[tuple[str, str, str | None]] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            completed.add((r["example_id"], r["variant"], r.get("target")))
        print(f"Resuming — {len(completed)} calls already done.")

    client = cohere.ClientV2(api_key=api_key)

    total_variants = sum(len(build_variants(rec)) for rec in dataset)
    remaining = total_variants - len(completed)
    print(f"Total variants: {total_variants}  |  To run: {remaining}")

    with open(out_path, "a") as out_file:
        for rec in tqdm(dataset, desc="Examples"):
            eid = rec["example_id"]
            for spec in build_variants(rec):
                target = spec.get("target")
                key = (eid, spec["variant"], target)
                if key in completed:
                    continue

                try:
                    response = client.chat(
                        model=MODEL,
                        messages=[{"role": "user", "content": spec["prompt"]}],
                        temperature=0.0,
                        max_tokens=300,
                    )
                    raw = response.message.content[0].text.strip()
                    parse_error = False
                except Exception as e:
                    raw = f"API_ERROR: {e}"
                    parse_error = True

                result: dict = {
                    "example_id": eid,
                    "mode": spec["mode"],
                    "variant": spec["variant"],
                    "reference": spec["reference"],
                    "raw_response": raw,
                    "parse_error": parse_error,
                }

                if spec["mode"] in ("direct_scoring", "rubric_based"):
                    result["target"] = target
                    if not parse_error:
                        f, c, fl, reasoning = parse_scores(raw)
                        result.update({
                            "faithfulness": f, "coverage": c, "fluency": fl,
                            "reasoning": reasoning,
                        })
                        if any(v is None for v in [f, c, fl]):
                            result["parse_error"] = True
                    else:
                        result.update({"faithfulness": None, "coverage": None,
                                       "fluency": None, "reasoning": None})

                else:  # pairwise or pairwise_cot
                    result["order"] = spec["order"]
                    label_first = spec["order"][0]   # "A" or "B"
                    label_second = spec["order"][1]
                    if not parse_error:
                        pref = parse_preference(raw, label_first, label_second)
                        result["preference"] = pref
                        if pref is None:
                            result["parse_error"] = True
                        result["reasoning"] = None
                        # For CoT, try to extract reasoning (everything before FINAL:)
                        if spec["mode"] == "pairwise_cot":
                            cot_split = re.split(r"FINAL:", raw, flags=re.IGNORECASE)
                            if len(cot_split) > 1:
                                result["reasoning"] = cot_split[0].strip()
                    else:
                        result["preference"] = None
                        result["reasoning"] = None

                out_file.write(json.dumps(result) + "\n")
                out_file.flush()
                completed.add(key)
                time.sleep(SLEEP_BETWEEN_CALLS)

    # Final report
    scores = [json.loads(l) for l in out_path.read_text().splitlines()]
    n_errors = sum(1 for r in scores if r["parse_error"])
    print(f"\nDone. Total rows: {len(scores)}  |  Parse errors: {n_errors}")
    if n_errors:
        print("Rows with parse errors:")
        for r in scores:
            if r["parse_error"]:
                print(f"  {r['example_id']} / {r['variant']} — {r['raw_response'][:80]}")
    print(f"Saved → {out_path}")
    print(f"Model: {MODEL}  temperature=0.0")
    print("Record all prompt templates in your paper appendix.")


if __name__ == "__main__":
    main()
