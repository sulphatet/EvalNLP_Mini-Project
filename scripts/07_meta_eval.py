"""
Step 7: Meta-evaluation — correlations, bias analysis, mode consistency (Parts 4 + 8).

Input  : data/final/llm_judge_scores.jsonl
         data/final/gold_scores.jsonl
         data/final/dataset.jsonl
         data/final/system_map.json

Output : data/final/meta_eval_report.json
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau

# ── Loaders ────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]

all_scores  = load_jsonl("data/final/llm_judge_scores.jsonl")
gold_list   = load_jsonl("data/final/gold_scores.jsonl")
dataset_list = load_jsonl("data/final/dataset.jsonl")

gold       = {r["example_id"]: r for r in gold_list}
dataset    = {r["example_id"]: r for r in dataset_list}
system_map = json.loads(Path("data/final/system_map.json").read_text())
example_ids = sorted(gold.keys())

DIMS = ["faithfulness", "coverage", "fluency"]
TARGETS = ["A", "B"]
report = {}

# ── Helpers ────────────────────────────────────────────────────────────────────

def scoring_index(mode, variant):
    """Return dict keyed (example_id, target) → row, for non-errored rows."""
    return {
        (r["example_id"], r["target"]): r
        for r in all_scores
        if r["mode"] == mode and r["variant"] == variant and not r["parse_error"]
    }

def pairwise_index(mode, variant):
    return {
        r["example_id"]: r
        for r in all_scores
        if r["mode"] == mode and r["variant"] == variant and not r["parse_error"]
    }

def llm_winner(eid, pref):
    """Map A/B preference label → bart/cohere system name."""
    if pref == "tie":
        return "tie"
    return system_map[eid][pref]


# ══════════════════════════════════════════════════════════════════════════════
# 1. CORRELATION: LLM scores vs human gold
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. CORRELATION — LLM scores vs human gold")
print("=" * 60)

scoring_variants = [
    ("direct_scoring", "direct_no_ref"),
    ("direct_scoring", "direct_with_ref"),
    ("rubric_based",   "rubric_no_ref"),
    ("rubric_based",   "rubric_with_ref"),
]

corr_results = {}
for mode, variant in scoring_variants:
    idx = scoring_index(mode, variant)
    vc = {}
    for dim in DIMS:
        llm_vals, human_vals = [], []
        for eid in example_ids:
            for t in TARGETS:
                k = (eid, t)
                if k in idx and idx[k].get(dim) is not None:
                    llm_vals.append(idx[k][dim])
                    human_vals.append(gold[eid][f"gold_{dim}_{t}"])
        rho, p_s = spearmanr(llm_vals, human_vals)
        tau, p_k = kendalltau(llm_vals, human_vals)
        vc[dim] = {
            "spearman_rho": round(float(rho), 4),
            "spearman_p":   round(float(p_s), 4),
            "kendall_tau":  round(float(tau), 4),
            "kendall_p":    round(float(p_k), 4),
            "n": len(llm_vals),
        }
        print(f"  {variant:<28}  {dim:<13}  rho={rho:+.3f} (p={p_s:.3f})  tau={tau:+.3f}")
    corr_results[variant] = vc

report["correlation"] = corr_results


# ══════════════════════════════════════════════════════════════════════════════
# 2. PAIRWISE AGREEMENT: LLM winner vs human gold winner
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. PAIRWISE AGREEMENT — LLM winner vs human gold")
print("=" * 60)

pair_variants = [
    ("pairwise", "pair_no_ref_AB"),
    ("pairwise", "pair_no_ref_BA"),
    ("pairwise", "pair_with_ref_AB"),
    ("pairwise", "pair_with_ref_BA"),
]

pairwise_agreement = {}
for mode, variant in pair_variants:
    idx = pairwise_index(mode, variant)
    agree = total = 0
    for eid in example_ids:
        if eid not in idx:
            continue
        lw = llm_winner(eid, idx[eid]["preference"])
        hw = gold[eid]["gold_winner"]
        if lw == hw:
            agree += 1
        total += 1
    rate = round(agree / total, 4) if total else 0
    pairwise_agreement[variant] = {"agreement": rate, "n": total, "correct": agree}
    print(f"  {variant:<30}  agreement={rate:.3f}  ({agree}/{total})")

report["pairwise_agreement"] = pairwise_agreement


# ══════════════════════════════════════════════════════════════════════════════
# 3. POSITION BIAS: conflict rate when A/B order is swapped
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. POSITION BIAS — conflict rate (AB vs BA)")
print("=" * 60)

position_bias = {}
for ref_tag in ["no_ref", "with_ref"]:
    ab = pairwise_index("pairwise", f"pair_{ref_tag}_AB")
    ba = pairwise_index("pairwise", f"pair_{ref_tag}_BA")

    conflicts = first_wins_ab = first_wins_ba = decisive = 0

    for eid in example_ids:
        if eid not in ab or eid not in ba:
            continue
        p_ab = ab[eid]["preference"]
        p_ba = ba[eid]["preference"]

        if p_ab != "tie" and p_ba != "tie":
            if p_ab != p_ba:
                conflicts += 1
            decisive += 1

        if p_ab == "A":   first_wins_ab += 1   # A is shown first in AB
        if p_ba == "B":   first_wins_ba += 1   # B is shown first in BA

    n = len(example_ids)
    conflict_rate      = round(conflicts / decisive, 4) if decisive else 0
    # First-position preference rate: average of (chose first in AB) and (chose first in BA)
    first_pos_rate     = round((first_wins_ab + first_wins_ba) / (2 * n), 4)

    position_bias[ref_tag] = {
        "conflict_rate":             conflict_rate,
        "first_position_pref_rate":  first_pos_rate,
        "n_decisive_pairs":          decisive,
    }
    print(f"  {ref_tag:<10}  conflict_rate={conflict_rate:.3f}  "
          f"first_pos_pref={first_pos_rate:.3f}  (n={decisive})")

report["position_bias"] = position_bias


# ══════════════════════════════════════════════════════════════════════════════
# 4. CoT ANALYSIS (Part 8)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. CoT vs STANDARD PAIRWISE (Part 8)")
print("=" * 60)

cot_ab  = pairwise_index("pairwise_cot", "cot_pair_no_ref_AB")
cot_ba  = pairwise_index("pairwise_cot", "cot_pair_no_ref_BA")
std_ab  = pairwise_index("pairwise", "pair_no_ref_AB")
std_ba  = pairwise_index("pairwise", "pair_no_ref_BA")

n_total_cot = sum(1 for r in all_scores if r["mode"] == "pairwise_cot")
n_errors_cot = sum(1 for r in all_scores if r["mode"] == "pairwise_cot" and r["parse_error"])

cot_valid = set(cot_ab) & set(cot_ba)
std_valid = set(std_ab) & set(std_ba)

def conflict_rate(ab_idx, ba_idx, valid_ids):
    c = n = 0
    for eid in valid_ids:
        p_ab = ab_idx[eid]["preference"]
        p_ba = ba_idx[eid]["preference"]
        if p_ab != "tie" and p_ba != "tie":
            if p_ab != p_ba:
                c += 1
            n += 1
    return round(c / n, 4) if n else 0, n

cot_cr, cot_n  = conflict_rate(cot_ab, cot_ba, cot_valid)
std_cr, std_n  = conflict_rate(std_ab, std_ba, std_valid)

# CoT agreement with human gold (on valid AB examples)
cot_agree = sum(
    1 for eid in cot_ab
    if llm_winner(eid, cot_ab[eid]["preference"]) == gold[eid]["gold_winner"]
)

cot_results = {
    "n_total_cot_calls":            n_total_cot,
    "n_parse_errors":               n_errors_cot,
    "parse_failure_rate":           round(n_errors_cot / n_total_cot, 4),
    "cot_conflict_rate":            cot_cr,
    "std_conflict_rate":            std_cr,
    "conflict_rate_delta_cot_minus_std": round(cot_cr - std_cr, 4),
    "cot_n_valid_pairs":            len(cot_valid),
    "std_n_valid_pairs":            std_n,
    "cot_pairwise_agreement_with_human": round(cot_agree / len(cot_ab), 4) if cot_ab else 0,
}
print(f"  Parse failure rate:          {n_errors_cot}/{n_total_cot} = {n_errors_cot/n_total_cot:.3f}")
print(f"  Standard conflict rate:      {std_cr:.3f}  (n={std_n})")
print(f"  CoT conflict rate:           {cot_cr:.3f}  (n={cot_n} valid pairs)")
print(f"  Delta (CoT - std):           {cot_cr - std_cr:+.3f}")
print(f"  CoT agreement with human:    {cot_results['cot_pairwise_agreement_with_human']:.3f}")

report["cot_analysis"] = cot_results


# ══════════════════════════════════════════════════════════════════════════════
# 5. REFERENCE EFFECT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. REFERENCE EFFECT — score change when article provided")
print("=" * 60)

no_ref  = scoring_index("direct_scoring", "direct_no_ref")
with_ref = scoring_index("direct_scoring", "direct_with_ref")

ref_effect = {}
for dim in DIMS:
    deltas = []
    for eid in example_ids:
        for t in TARGETS:
            k = (eid, t)
            if k in no_ref and k in with_ref:
                d_nr = no_ref[k].get(dim)
                d_wr = with_ref[k].get(dim)
                if d_nr is not None and d_wr is not None:
                    deltas.append(d_wr - d_nr)
    mean_d = round(float(np.mean(deltas)), 4)
    std_d  = round(float(np.std(deltas)), 4)
    pct_decreased = round(sum(1 for d in deltas if d < 0) / len(deltas), 4)
    ref_effect[dim] = {
        "mean_delta": mean_d,
        "std_delta": std_d,
        "pct_score_decreased": pct_decreased,
        "n": len(deltas),
    }
    print(f"  {dim:<13}  mean Δ={mean_d:+.3f}  "
          f"decreased in {pct_decreased:.0%} of cases")

report["reference_effect"] = ref_effect


# ══════════════════════════════════════════════════════════════════════════════
# 6. VERBOSITY BIAS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. VERBOSITY BIAS — length vs score correlation")
print("=" * 60)

direct_nr = scoring_index("direct_scoring", "direct_no_ref")

lengths, llm_totals, human_totals = [], [], []
for eid in example_ids:
    rec = dataset[eid]
    for t in TARGETS:
        k = (eid, t)
        if k not in direct_nr:
            continue
        summary_text = rec["summary_A"] if t == "A" else rec["summary_B"]
        wc = len(summary_text.split())
        row = direct_nr[k]
        if any(row.get(d) is None for d in DIMS):
            continue
        llm_total   = sum(row[d] for d in DIMS)
        human_total = sum(gold[eid][f"gold_{d}_{t}"] for d in DIMS)
        lengths.append(wc)
        llm_totals.append(llm_total)
        human_totals.append(human_total)

rho_llm,   p_llm   = spearmanr(lengths, llm_totals)
rho_human, p_human = spearmanr(lengths, human_totals)

verbosity = {
    "llm_judge_length_vs_total_score":   {"rho": round(float(rho_llm),   4), "p": round(float(p_llm),   4)},
    "human_gold_length_vs_total_score":  {"rho": round(float(rho_human), 4), "p": round(float(p_human), 4)},
    "n": len(lengths),
}
print(f"  LLM judge (no ref): length vs total score  rho={rho_llm:+.3f}  (p={p_llm:.3f})")
print(f"  Human gold:         length vs total score  rho={rho_human:+.3f}  (p={p_human:.3f})")

report["verbosity_bias"] = verbosity


# ══════════════════════════════════════════════════════════════════════════════
# 7. SELF-PREFERENCE: LLM preference for Cohere vs human preference
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. SELF-PREFERENCE — Cohere judging Cohere")
print("=" * 60)

human_cohere_rate = round(
    sum(1 for r in gold_list if r["gold_winner"] == "cohere") / len(gold_list), 4
)

self_pref = {"human_cohere_preference_rate": human_cohere_rate}

for label, mode, variant in [
    ("pair_no_ref_AB",   "pairwise", "pair_no_ref_AB"),
    ("pair_with_ref_AB", "pairwise", "pair_with_ref_AB"),
    ("rubric_no_ref",    "rubric_based", "rubric_no_ref"),
]:
    if mode == "rubric_based":
        idx = scoring_index(mode, variant)
        # Convert per-summary scores to winners
        cohere_wins = tie_count = 0
        for eid in example_ids:
            sm = system_map[eid]
            a_key = (eid, "A")
            b_key = (eid, "B")
            if a_key not in idx or b_key not in idx:
                continue
            total_A = sum(idx[a_key].get(d, 0) or 0 for d in DIMS)
            total_B = sum(idx[b_key].get(d, 0) or 0 for d in DIMS)
            if total_A > total_B:
                winner = sm["A"]
            elif total_B > total_A:
                winner = sm["B"]
            else:
                winner = "tie"
            if winner == "cohere":
                cohere_wins += 1
        rate = round(cohere_wins / len(example_ids), 4)
    else:
        idx = pairwise_index(mode, variant)
        decisive = [eid for eid in example_ids if eid in idx and idx[eid]["preference"] != "tie"]
        cohere_wins = sum(1 for eid in decisive if llm_winner(eid, idx[eid]["preference"]) == "cohere")
        rate = round(cohere_wins / len(decisive), 4) if decisive else 0

    self_pref[label] = {"llm_cohere_rate": rate, "human_cohere_rate": human_cohere_rate,
                         "delta": round(rate - human_cohere_rate, 4)}
    print(f"  {label:<28}  LLM={rate:.3f}  human={human_cohere_rate:.3f}  "
          f"Δ={rate - human_cohere_rate:+.3f}")

report["self_preference"] = self_pref


# ══════════════════════════════════════════════════════════════════════════════
# 8. MODE CONSISTENCY: do direct, rubric, and pairwise agree?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. MODE CONSISTENCY — direct vs rubric (no ref)")
print("=" * 60)

d_nr = scoring_index("direct_scoring", "direct_no_ref")
r_nr = scoring_index("rubric_based",   "rubric_no_ref")

mode_consistency = {}
for dim in DIMS:
    d_vals, r_vals = [], []
    for eid in example_ids:
        for t in TARGETS:
            k = (eid, t)
            if k in d_nr and k in r_nr:
                dv = d_nr[k].get(dim)
                rv = r_nr[k].get(dim)
                if dv is not None and rv is not None:
                    d_vals.append(dv)
                    r_vals.append(rv)
    rho, p = spearmanr(d_vals, r_vals)
    exact_agree = sum(1 for a, b in zip(d_vals, r_vals) if a == b) / len(d_vals)
    mode_consistency[dim] = {
        "direct_vs_rubric_spearman": round(float(rho), 4),
        "direct_vs_rubric_p": round(float(p), 4),
        "exact_agreement_rate": round(exact_agree, 4),
    }
    print(f"  {dim:<13}  direct vs rubric  rho={rho:+.3f}  exact_agree={exact_agree:.3f}")

report["mode_consistency"] = mode_consistency


# ── Save ───────────────────────────────────────────────────────────────────────
Path("data/final/meta_eval_report.json").write_text(json.dumps(report, indent=2))
print("\n" + "=" * 60)
print(f"Saved → data/final/meta_eval_report.json")


if __name__ == "__main__":
    pass
