"""
Step 9: Failure analysis — taxonomy of LLM judge failure modes (Part 6).

Identifies cases where the LLM judge disagrees with human gold standard,
categorises them into a failure taxonomy, and produces supporting statistics.

Failure categories:
  F1  fluency_dominance    — LLM agrees with human on preference but score gap
                             is driven by fluency, not faithfulness/coverage.
  F2  faithfulness_blind   — LLM rates a summary high on faithfulness that
                             humans rated lower (≥1 point gap, no article ref).
  F3  position_driven      — LLM preference flips between AB and BA orderings
                             (cross-referenced with meta_eval position_bias).
  F4  reference_reversal   — LLM preference changes direction when article is
                             added (no_ref vs with_ref variant pair).
  F5  self_pref_overreach  — Pairwise mode picks Cohere but human gold picks
                             BART; rubric mode does not share this preference.
  F6  cot_collapse         — CoT variant either fails to parse (token exhaust)
                             or contradicts the non-CoT pairwise verdict.

Output:
  data/final/failure_taxonomy.json — per-example failure flags + summary counts
"""

import json
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]

gold   = {r["example_id"]: r for r in load_jsonl("data/final/gold_scores.jsonl")}
scores = load_jsonl("data/final/llm_judge_scores.jsonl")
adv    = load_jsonl("data/adversarial/judge_scores.jsonl")

# Index judge scores
by_id_variant = {}  # (example_id, variant, target) -> record   (direct/rubric)
pairwise_by_id_variant = {}  # (example_id, variant) -> record  (pairwise)
for r in scores:
    if r["mode"] in ("direct_scoring", "rubric_based"):
        key = (r["example_id"], r["variant"], r["target"])
        by_id_variant[key] = r
    else:
        key = (r["example_id"], r["variant"])
        pairwise_by_id_variant[key] = r

# ── Helpers ────────────────────────────────────────────────────────────────────

def llm_pref_to_system(pref_letter, g):
    """Convert A/B preference to bart/cohere using system_map in gold record."""
    if pref_letter in ("tie", None):
        return pref_letter
    sys_map = {"A": g["system_A"], "B": g["system_B"]}
    return sys_map.get(pref_letter)


def get_pairwise_pref_system(example_id, variant):
    rec = pairwise_by_id_variant.get((example_id, variant))
    if rec is None or rec.get("parse_error") or rec.get("preference") is None:
        return None
    g = gold[example_id]
    return llm_pref_to_system(rec["preference"], g)


def get_direct_score(example_id, variant, target, dim):
    rec = by_id_variant.get((example_id, variant, target))
    if rec is None or rec.get("parse_error"):
        return None
    return rec.get(dim)

# ── Build per-example failure flags ───────────────────────────────────────────

examples = list(gold.keys())
taxonomy = []

for eid in examples:
    g = gold[eid]
    human_winner = g["gold_winner"]   # "bart" or "cohere"
    flags = []

    # ── F1: fluency_dominance ─────────────────────────────────────────────────
    # LLM agrees with human preference BUT the LLM's own score gap on fluency
    # is larger than its gap on faithfulness+coverage combined.
    for variant in ("direct_no_ref", "direct_with_ref"):
        fa = get_direct_score(eid, variant, "A", "faithfulness")
        ca = get_direct_score(eid, variant, "A", "coverage")
        fla = get_direct_score(eid, variant, "A", "fluency")
        fb = get_direct_score(eid, variant, "B", "faithfulness")
        cb = get_direct_score(eid, variant, "B", "coverage")
        flb = get_direct_score(eid, variant, "B", "fluency")
        if None in (fa, ca, fla, fb, cb, flb):
            continue
        fluency_gap = abs(fla - flb)
        quality_gap = abs((fa + ca) / 2 - (fb + cb) / 2)
        if fluency_gap > quality_gap + 0.5:  # fluency driving the verdict
            flags.append({"category": "F1_fluency_dominance", "variant": variant,
                          "fluency_gap": round(fluency_gap, 3),
                          "quality_gap": round(quality_gap, 3)})
            break

    # ── F2: faithfulness_blind ────────────────────────────────────────────────
    # LLM faithfulness score for a system is ≥1 higher than human gold,
    # measured on the cohere summary (most likely to have hallucinations).
    for variant in ("direct_no_ref", "rubric_no_ref"):
        # Determine which target letter = cohere for this example
        coh_target = "A" if g["system_A"] == "cohere" else "B"
        llm_faith = get_direct_score(eid, variant, coh_target, "faithfulness")
        human_faith = g["gold_faithfulness_cohere"]
        if llm_faith is None or human_faith is None:
            continue
        if llm_faith - human_faith >= 1.0:
            flags.append({"category": "F2_faithfulness_blind", "variant": variant,
                          "llm_faithfulness": llm_faith,
                          "human_faithfulness": round(human_faith, 3),
                          "gap": round(llm_faith - human_faith, 3)})
            break

    # ── F3: position_driven ───────────────────────────────────────────────────
    # Pairwise preference flips between AB and BA orderings (no_ref).
    pref_ab = get_pairwise_pref_system(eid, "pair_no_ref_AB")
    pref_ba = get_pairwise_pref_system(eid, "pair_no_ref_BA")
    if pref_ab and pref_ba and pref_ab != "tie" and pref_ba != "tie":
        if pref_ab != pref_ba:
            flags.append({"category": "F3_position_driven",
                          "pair_no_ref_AB": pref_ab,
                          "pair_no_ref_BA": pref_ba})

    # ── F4: reference_reversal ────────────────────────────────────────────────
    # Adding the article flips the pairwise preference direction.
    pref_no_ref  = get_pairwise_pref_system(eid, "pair_no_ref_AB")
    pref_with_ref = get_pairwise_pref_system(eid, "pair_with_ref_AB")
    if (pref_no_ref and pref_with_ref
            and pref_no_ref != "tie" and pref_with_ref != "tie"
            and pref_no_ref != pref_with_ref):
        flags.append({"category": "F4_reference_reversal",
                      "no_ref_pref": pref_no_ref,
                      "with_ref_pref": pref_with_ref})

    # ── F5: self_pref_overreach ───────────────────────────────────────────────
    # Pairwise picks Cohere but human gold picked BART, AND
    # rubric mode does NOT prefer Cohere (gives BART higher total score).
    pair_sys = get_pairwise_pref_system(eid, "pair_no_ref_AB")
    if pair_sys == "cohere" and human_winner == "bart":
        # Check rubric: total score cohere vs bart
        coh_t = "A" if g["system_A"] == "cohere" else "B"
        bar_t = "B" if coh_t == "A" else "A"
        rubric_coh_f  = get_direct_score(eid, "rubric_no_ref", coh_t, "faithfulness") or 0
        rubric_coh_c  = get_direct_score(eid, "rubric_no_ref", coh_t, "coverage") or 0
        rubric_coh_fl = get_direct_score(eid, "rubric_no_ref", coh_t, "fluency") or 0
        rubric_bar_f  = get_direct_score(eid, "rubric_no_ref", bar_t, "faithfulness") or 0
        rubric_bar_c  = get_direct_score(eid, "rubric_no_ref", bar_t, "coverage") or 0
        rubric_bar_fl = get_direct_score(eid, "rubric_no_ref", bar_t, "fluency") or 0
        rubric_coh_total = rubric_coh_f + rubric_coh_c + rubric_coh_fl
        rubric_bar_total = rubric_bar_f + rubric_bar_c + rubric_bar_fl
        rubric_winner = "cohere" if rubric_coh_total > rubric_bar_total else "bart"
        if rubric_winner == "bart":
            flags.append({"category": "F5_self_pref_overreach",
                          "pair_pick": "cohere",
                          "rubric_pick": "bart",
                          "human_pick": "bart"})

    # ── F6: cot_collapse ─────────────────────────────────────────────────────
    # CoT variant fails to parse OR contradicts standard pairwise verdict.
    cot_ab = pairwise_by_id_variant.get((eid, "cot_pair_no_ref_AB"))
    std_ab = pairwise_by_id_variant.get((eid, "pair_no_ref_AB"))
    if cot_ab:
        if cot_ab.get("parse_error"):
            flags.append({"category": "F6_cot_collapse", "reason": "parse_error"})
        elif std_ab and not std_ab.get("parse_error"):
            cot_sys = llm_pref_to_system(cot_ab["preference"], g)
            std_sys = llm_pref_to_system(std_ab["preference"], g)
            if cot_sys and std_sys and cot_sys != "tie" and std_sys != "tie" and cot_sys != std_sys:
                flags.append({"category": "F6_cot_collapse", "reason": "contradicts_standard",
                              "cot_pick": cot_sys, "std_pick": std_sys})

    taxonomy.append({
        "example_id": eid,
        "human_winner": human_winner,
        "flags": flags,
        "n_flags": len(flags),
    })

# ── Count failures per category ────────────────────────────────────────────────

counts = {f"F{i}": 0 for i in range(1, 7)}
cat_labels = {
    "F1": "F1_fluency_dominance",
    "F2": "F2_faithfulness_blind",
    "F3": "F3_position_driven",
    "F4": "F4_reference_reversal",
    "F5": "F5_self_pref_overreach",
    "F6": "F6_cot_collapse",
}
examples_with_any_flag = 0
for entry in taxonomy:
    cats = set(f["category"][:2] for f in entry["flags"])
    if cats:
        examples_with_any_flag += 1
    for c in cats:
        counts[c] += 1

# ── Adversarial failure summary ────────────────────────────────────────────────

adv_by_type = {}
for r in adv:
    t = r["adversarial_type"]
    adv_by_type.setdefault(t, []).append(r)

adv_summary = {}
for t, rows in adv_by_type.items():
    ok = [r for r in rows if not r.get("parse_error")]
    if not ok:
        continue
    total = len(ok)
    correct = sum(1 for r in ok if r["correct"])
    # Position bias for fluent_incorrect: original_first vs modified_first
    if t == "fluent_incorrect":
        orig_first = [r for r in ok if r["variant"] == "original_first"]
        mod_first  = [r for r in ok if r["variant"] == "modified_first"]
        adv_summary[t] = {
            "total_evals": total,
            "correct": correct,
            "accuracy": round(correct / total, 3),
            "original_first_correct": sum(1 for r in orig_first if r["correct"]),
            "original_first_n": len(orig_first),
            "modified_first_correct": sum(1 for r in mod_first if r["correct"]),
            "modified_first_n": len(mod_first),
        }
    else:
        adv_summary[t] = {
            "total_evals": total,
            "correct": correct,
            "accuracy": round(correct / total, 3),
        }

# ── Compose report ─────────────────────────────────────────────────────────────

report = {
    "summary": {
        "n_examples": len(examples),
        "n_with_any_failure_flag": examples_with_any_flag,
        "pct_with_any_failure": round(examples_with_any_flag / len(examples), 3),
    },
    "category_counts": {
        cat_labels[k]: v for k, v in counts.items()
    },
    "adversarial_accuracy": adv_summary,
    "per_example": taxonomy,
}

out = Path("data/final/failure_taxonomy.json")
out.write_text(json.dumps(report, indent=2))
print(f"Saved → {out}")

# ── Print summary ──────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("FAILURE TAXONOMY SUMMARY")
print(f"{'='*60}")
print(f"Examples with ≥1 failure flag : {examples_with_any_flag}/{len(examples)} "
      f"({examples_with_any_flag/len(examples):.0%})")
print()
print("Per-category counts (50 examples, multiple flags possible):")
for k, label in cat_labels.items():
    n = counts[k]
    print(f"  {label:<28}: {n:2d}  ({n/len(examples):.0%} of examples)")

print()
print("Adversarial accuracy by type:")
for t, s in adv_summary.items():
    line = f"  {t:<20}: {s['correct']}/{s['total_evals']} ({s['accuracy']:.0%})"
    if t == "fluent_incorrect":
        of_n = s["original_first_n"]
        of_c = s["original_first_correct"]
        mf_n = s["modified_first_n"]
        mf_c = s["modified_first_correct"]
        line += f"  [orig_first={of_c}/{of_n}, mod_first={mf_c}/{mf_n}]"
    print(line)
