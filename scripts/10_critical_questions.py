"""
Step 10: Critical questions analysis (Part 7).

Answers the five assignment questions with numbers drawn from
Parts 3–6 of the pipeline. Produces data/final/critical_questions.json
and prints a formatted report for inclusion in the paper.

Questions:
  Q1. How reliable is LLM-as-judge compared to human annotation?
  Q2. What are the main failure modes?
  Q3. Does providing the source article improve reliability?
  Q4. Is there evidence of position bias or self-preference bias?
  Q5. Does chain-of-thought reasoning improve reliability?
"""

import json
from pathlib import Path


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


gold     = {r["example_id"]: r for r in load_jsonl("data/final/gold_scores.jsonl")}
judge    = load_jsonl("data/final/llm_judge_scores.jsonl")
adv      = load_jsonl("data/adversarial/judge_scores.jsonl")
taxonomy = json.loads(Path("data/final/failure_taxonomy.json").read_text())
meta     = json.loads(Path("data/final/meta_eval_report.json").read_text())
agree    = json.loads(Path("data/final/agreement_report.json").read_text())

# ── Q1: Reliability ────────────────────────────────────────────────────────────

# Direct scoring correlation (best variant = direct_no_ref for faithfulness/coverage)
corr = meta["correlation"]
best_faith_rho = corr["direct_no_ref"]["faithfulness"]["spearman_rho"]
best_cov_rho   = corr["direct_no_ref"]["coverage"]["spearman_rho"]
# Fluency: direct_no_ref has higher rho than with_ref (fluency is surface-level)
best_flu_rho   = corr["direct_no_ref"]["fluency"]["spearman_rho"]

# Rubric with ref: coverage rho is highest (0.59)
rubric_cov_rho = corr["rubric_with_ref"]["coverage"]["spearman_rho"]

# Pairwise agreement
pair_agree = meta["pairwise_agreement"]
pairwise_rates = {k: v["agreement"] for k, v in pair_agree.items()}
best_pairwise = max(pairwise_rates.values())
worst_pairwise = min(pairwise_rates.values())

# Krippendorff alpha (human baseline)
human_alpha = agree.get("krippendorff_alpha", {})

q1 = {
    "direct_scoring_spearman": {
        "faithfulness": best_faith_rho,
        "coverage": best_cov_rho,
        "fluency": best_flu_rho,
    },
    "rubric_with_ref_coverage_spearman": rubric_cov_rho,
    "pairwise_agreement_range": [worst_pairwise, best_pairwise],
    "human_inter_annotator_alpha": human_alpha,
    "interpretation": (
        "LLM judge shows moderate correlation with humans on faithfulness "
        f"(ρ={best_faith_rho}) and coverage (ρ={best_cov_rho}), but weak on "
        f"fluency (ρ={best_flu_rho}) without a reference. Pairwise mode "
        f"achieves {best_pairwise:.0%} agreement with human preference — "
        "comparable to inter-annotator agreement (see α values above)."
    ),
}

# ── Q2: Failure modes ──────────────────────────────────────────────────────────

cat_counts = taxonomy["category_counts"]
n_ex = taxonomy["summary"]["n_examples"]

# Dominant failure mode is CoT collapse
f6 = cat_counts["F6_cot_collapse"]
f2 = cat_counts["F2_faithfulness_blind"]
f5 = cat_counts["F5_self_pref_overreach"]
f3 = cat_counts["F3_position_driven"]
f4 = cat_counts["F4_reference_reversal"]

# Adversarial: fluent_incorrect is the hardest
adv_acc = taxonomy["adversarial_accuracy"]

q2 = {
    "category_counts_over_50_examples": cat_counts,
    "dominant_failure": "F6_cot_collapse",
    "cot_parse_failure_rate": meta["cot_analysis"]["parse_failure_rate"],
    "adversarial_accuracy": adv_acc,
    "position_driven_fluent_incorrect": {
        "original_first_accuracy": round(
            adv_acc["fluent_incorrect"]["original_first_correct"] /
            adv_acc["fluent_incorrect"]["original_first_n"], 3),
        "modified_first_accuracy": round(
            adv_acc["fluent_incorrect"]["modified_first_correct"] /
            adv_acc["fluent_incorrect"]["modified_first_n"], 3),
    },
    "interpretation": (
        f"The dominant failure mode is CoT collapse: {f6}/50 examples where "
        f"CoT either fails to parse ({meta['cot_analysis']['n_parse_errors']} cases) "
        f"or contradicts the non-CoT verdict. "
        f"Faithfulness blindness (F2) affects {f2}/50 examples — the LLM assigns "
        f"faithfulness=5 to summaries humans rated ≤4. "
        f"Self-preference overreach (F5) occurs in {f5}/50 cases. "
        "Adversarial testing reveals a critical position-driven failure: the judge "
        "is 100% accurate on fluent+incorrect summaries when the correct summary "
        "appears first, but 0% accurate when the incorrect summary leads."
    ),
}

# ── Q3: Reference effect ───────────────────────────────────────────────────────

ref_effect = meta["reference_effect"]
# Direct: faithfulness delta when ref added
faith_delta = ref_effect["faithfulness"]["mean_delta"]
cov_delta   = ref_effect["coverage"]["mean_delta"]

# Pairwise agreement: no_ref vs with_ref
no_ref_agree  = pair_agree["pair_no_ref_AB"]["agreement"]
with_ref_agree = pair_agree["pair_with_ref_AB"]["agreement"]

# Fluency correlation drops dramatically with reference
flu_no_ref  = corr["direct_no_ref"]["fluency"]["spearman_rho"]
flu_with_ref = corr["direct_with_ref"]["fluency"]["spearman_rho"]

q3 = {
    "pairwise_agreement_no_ref": no_ref_agree,
    "pairwise_agreement_with_ref": with_ref_agree,
    "direct_faithfulness_score_delta_with_ref": faith_delta,
    "direct_coverage_delta_with_ref": cov_delta,
    "fluency_spearman_no_ref": flu_no_ref,
    "fluency_spearman_with_ref": flu_with_ref,
    "position_bias_no_ref_conflict_rate": meta["position_bias"]["no_ref"]["conflict_rate"],
    "position_bias_with_ref_conflict_rate": meta["position_bias"]["with_ref"]["conflict_rate"],
    "interpretation": (
        f"Adding the source article does not uniformly improve reliability. "
        f"Pairwise agreement stays flat ({no_ref_agree:.0%} → {with_ref_agree:.0%}). "
        f"Faithfulness scores rise slightly (+{faith_delta:.2f}) but this inflates "
        f"scores rather than improving calibration. Most strikingly, fluency "
        f"correlation collapses when the article is present "
        f"(ρ={flu_no_ref} → ρ={flu_with_ref:.4f}), suggesting the model "
        f"attends to the article at the expense of the summary's surface quality. "
        f"The one clear benefit: position bias (conflict rate) drops from "
        f"{meta['position_bias']['no_ref']['conflict_rate']:.0%} to "
        f"{meta['position_bias']['with_ref']['conflict_rate']:.0%} with a reference."
    ),
}

# ── Q4: Position bias and self-preference ─────────────────────────────────────

pos_bias = meta["position_bias"]
self_pref = meta["self_preference"]

q4 = {
    "position_bias": {
        "no_ref_conflict_rate": pos_bias["no_ref"]["conflict_rate"],
        "with_ref_conflict_rate": pos_bias["with_ref"]["conflict_rate"],
        "no_ref_first_position_pref_rate": pos_bias["no_ref"]["first_position_pref_rate"],
    },
    "self_preference": {
        "human_cohere_preference_rate": self_pref["human_cohere_preference_rate"],
        "pairwise_no_ref_cohere_rate": self_pref["pair_no_ref_AB"]["llm_cohere_rate"],
        "pairwise_with_ref_cohere_rate": self_pref["pair_with_ref_AB"]["llm_cohere_rate"],
        "rubric_no_ref_cohere_rate": self_pref["rubric_no_ref"]["llm_cohere_rate"],
        "pairwise_vs_human_delta": self_pref["pair_no_ref_AB"]["delta"],
    },
    "adversarial_position_evidence": {
        "fluent_incorrect_orig_first_acc": round(
            adv_acc["fluent_incorrect"]["original_first_correct"] /
            adv_acc["fluent_incorrect"]["original_first_n"], 3),
        "fluent_incorrect_mod_first_acc": round(
            adv_acc["fluent_incorrect"]["modified_first_correct"] /
            adv_acc["fluent_incorrect"]["modified_first_n"], 3),
    },
    "interpretation": (
        f"Position bias is detectable but mild in standard pairwise evaluation "
        f"(no_ref conflict rate: {pos_bias['no_ref']['conflict_rate']:.0%}). "
        f"It is eliminated entirely with a reference "
        f"({pos_bias['with_ref']['conflict_rate']:.0%} conflict rate). "
        f"Self-preference is the stronger bias: the LLM (Cohere) selects its own "
        f"output in pairwise mode {self_pref['pair_no_ref_AB']['llm_cohere_rate']:.0%} "
        f"of the time vs {self_pref['human_cohere_preference_rate']:.0%} for humans "
        f"(+{self_pref['pair_no_ref_AB']['delta']:.0%} gap). "
        f"Rubric mode attenuates this: only "
        f"{self_pref['rubric_no_ref']['llm_cohere_rate']:.0%} Cohere preference. "
        "Adversarial tests provide the clearest position evidence: when the "
        "hallucinated summary is placed first, the judge picks it 5/5 times, "
        "a complete reversal from when it is placed second (0/5 correct)."
    ),
}

# ── Q5: CoT effect ────────────────────────────────────────────────────────────

cot = meta["cot_analysis"]

q5 = {
    "cot_parse_failure_rate": cot["parse_failure_rate"],
    "cot_pairwise_agreement_with_human": cot["cot_pairwise_agreement_with_human"],
    "std_pairwise_agreement_no_ref": pair_agree["pair_no_ref_AB"]["agreement"],
    "cot_vs_std_conflict_rate": cot["cot_conflict_rate"],
    "std_conflict_rate_for_comparison": cot["std_conflict_rate"],
    "conflict_rate_delta": cot["conflict_rate_delta_cot_minus_std"],
    "cot_valid_pairs": cot["cot_n_valid_pairs"],
    "std_valid_pairs": cot["std_n_valid_pairs"],
    "interpretation": (
        f"Chain-of-thought reasoning degrades reliability in our setting. "
        f"26% of CoT calls fail to parse (max_tokens exhausted before FINAL: marker). "
        f"Among valid CoT responses, human agreement is "
        f"{cot['cot_pairwise_agreement_with_human']:.1%} vs "
        f"{pair_agree['pair_no_ref_AB']['agreement']:.0%} for standard pairwise — "
        f"marginally lower. Critically, the CoT conflict rate is "
        f"{cot['cot_conflict_rate']:.0%} vs {cot['std_conflict_rate']:.0%} for standard "
        f"(Δ={cot['conflict_rate_delta_cot_minus_std']:+.0%}), meaning CoT reasoning "
        "introduces internal inconsistency rather than resolving it. "
        "This contradicts findings in Wei et al. (2022) and may reflect that CoT "
        "benefits depend on sufficient reasoning space — our 300-token budget "
        "is too tight for meaningful deliberation."
    ),
}

# ── Compile and save ───────────────────────────────────────────────────────────

report = {
    "Q1_reliability":      q1,
    "Q2_failure_modes":    q2,
    "Q3_reference_effect": q3,
    "Q4_bias":             q4,
    "Q5_cot":              q5,
}

out = Path("data/final/critical_questions.json")
out.write_text(json.dumps(report, indent=2))
print(f"Saved → {out}")

# ── Print formatted report ─────────────────────────────────────────────────────

SEPARATOR = "=" * 70

print(f"\n{SEPARATOR}")
print("CRITICAL QUESTIONS — SUMMARY REPORT")
print(SEPARATOR)

sections = [
    ("Q1", "How reliable is LLM-as-judge vs human annotation?", q1),
    ("Q2", "What are the main failure modes?", q2),
    ("Q3", "Does providing the source article improve reliability?", q3),
    ("Q4", "Is there evidence of position or self-preference bias?", q4),
    ("Q5", "Does chain-of-thought reasoning improve reliability?", q5),
]

for qnum, title, data in sections:
    print(f"\n{qnum}: {title}")
    print("-" * 60)
    print(data["interpretation"])

print(f"\n{SEPARATOR}")
