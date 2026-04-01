"""
subsumption_matcher.py
----------------------
Stage 3: Subsumption detection via asymmetric MNLI entailment.

After AML (Stage 1) and equivalence MNLI (Stage 2) leave entities unmatched,
this stage asks a different question: is one entity a semantic abstraction of
several others?  Asymmetric entailment is the signal:

    entail(U → E) high  :  U's description is implied by / fits under E
    entail(E → U) low   :  E is more general — it does not narrow down to U

When multiple residual entities U₁, U₂, … from model A each satisfy this
criterion against the same entity E in model B, E is identified as an
abstraction that *subsumes* the group.

Both directions are detected:
  • "large_abstracts_small"  — one entity in the larger model subsumes
                               several fine-grained entities in the smaller
  • "small_abstracts_large"  — one abstract entity in the smaller model
                               subsumes several entities in the larger model
                               (happens when the smaller model is high-level)

Output
------
  outputs/subsumption/<Domain>/<pair>_subsumption.json   per-pair groups
  outputs/alignment_summary_subsumption.csv              combined summary

Usage
-----
    cd ontology_matching
    python subsumption_matcher.py                    # all domains
    python subsumption_matcher.py --domain Automobile
    python subsumption_matcher.py --fwd 0.50 --gap 0.15
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

# Re-use helpers from the equivalence stage
from mnli_matcher import (
    NLIScorer,
    _build_description,
    _build_json_index,
    _build_parent_map,
    _vmodel_names,
    _ancestors,
    _entity_map,
    INPUTS_DIR,
    MODEL_NAME,
)

REPORTS_DIR       = os.path.join(os.path.dirname(__file__), "outputs", "reports")
MNLI_DIR          = os.path.join(os.path.dirname(__file__), "outputs", "mnli")
SUBSUMPTION_DIR   = os.path.join(os.path.dirname(__file__), "outputs", "subsumption")
OUT_CSV           = os.path.join(os.path.dirname(__file__), "outputs",
                                 "alignment_summary_subsumption.csv")

DEFAULT_FWD   = 0.50   # min forward entailment to count as subsumption candidate
DEFAULT_GAP   = 0.15   # min (forward - reverse) asymmetry


# --------------------------------------------------------------------------- #
# Stage-2 residual helper                                                      #
# --------------------------------------------------------------------------- #

def _stage2_matched(domain: str, pair_stem: str) -> set[str]:
    """Return entity names already recovered by Stage-2 MNLI equivalence."""
    path = os.path.join(MNLI_DIR, domain, f"{pair_stem}_mnli.json")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {m["smaller_entity"] for m in data.get("new_matches", [])}


# --------------------------------------------------------------------------- #
# Core subsumption detection                                                   #
# --------------------------------------------------------------------------- #

def _detect_subsumptions(
    residual_names: list[str],
    residual_emap: dict[str, list[dict]],
    residual_pmap: dict[str, list[str]],
    candidate_names: list[str],
    candidate_emap: dict[str, list[dict]],
    candidate_pmap: dict[str, list[str]],
    scorer: NLIScorer,
    fwd_thresh: float,
    gap_thresh: float,
    direction_label: str,
) -> list[dict]:
    """
    Find subsumption groups: one candidate entity that abstractly covers
    multiple residual entities.

    Returns a list of group dicts, one per abstract entity that subsumes ≥ 1
    residual entity (single-entity subsumptions are included but flagged).
    """
    # abstract_entity → list of (residual_name, fwd_score, asymmetry)
    groups: dict[str, list[tuple[str, float, float]]] = defaultdict(list)

    total_pairs = len(residual_names) * len(candidate_names)
    print(f"    [{direction_label}] {len(residual_names)} residual × "
          f"{len(candidate_names)} candidates = {total_pairs} pairs")

    for u_name in residual_names:
        desc_u = _build_description(u_name, residual_emap.get(u_name, []),
                                    _ancestors(u_name, residual_pmap))
        for e_name in candidate_names:
            desc_e = _build_description(e_name, candidate_emap.get(e_name, []),
                                        _ancestors(e_name, candidate_pmap))
            fwd = scorer.entailment_prob(desc_u, desc_e)
            rev = scorer.entailment_prob(desc_e, desc_u)
            asymmetry = fwd - rev
            if fwd >= fwd_thresh and asymmetry >= gap_thresh:
                groups[e_name].append((u_name, round(fwd, 4), round(asymmetry, 4)))

    result = []
    for abstract_name, members in groups.items():
        members.sort(key=lambda x: x[1], reverse=True)
        result.append({
            "abstract_entity":   abstract_name,
            "subsumed_entities": [m[0] for m in members],
            "fwd_scores":        [m[1] for m in members],
            "asymmetry_scores":  [m[2] for m in members],
            "group_size":        len(members),
            "is_group":          len(members) > 1,   # True = 1:many, False = 1:1 subsumption
            "direction":         direction_label,
        })

    # Sort by group size desc, then by mean forward score
    result.sort(
        key=lambda g: (g["group_size"], sum(g["fwd_scores"]) / len(g["fwd_scores"])),
        reverse=True,
    )
    return result


# --------------------------------------------------------------------------- #
# Per-pair driver                                                              #
# --------------------------------------------------------------------------- #

def _process_pair(
    report: dict,
    json_index: dict,
    scorer: NLIScorer,
    fwd_thresh: float,
    gap_thresh: float,
    domain: str,
    pair_stem: str,
) -> dict:
    meta   = report["metadata"]
    name_a = meta["model_a"]
    name_b = meta["model_b"]
    sa     = report["summary"]["model_a"]
    sb     = report["summary"]["model_b"]
    total_a = sa["matched"] + sa["ambiguous"] + sa["missing"]
    total_b = sb["matched"] + sb["ambiguous"] + sb["missing"]

    if total_a <= total_b:
        ref_side, ref_name, other_name = "model_a", name_a, name_b
        ref_total = total_a
    else:
        ref_side, ref_name, other_name = "model_b", name_b, name_a
        ref_total = total_b
    other_side = "model_b" if ref_side == "model_a" else "model_a"

    # Stage-1 missing entities from the smaller model
    missing_ref = [e["name"] for e in report[ref_side]["entities"]
                   if e["status"] == "missing"]

    # Subtract Stage-2 equivalence matches
    already_matched = _stage2_matched(domain, pair_stem)
    residual = [e for e in missing_ref if e not in already_matched]

    # All entities in the larger model
    all_larger = [e["name"] for e in report[other_side]["entities"]]

    # All entities in the smaller model (for reverse direction)
    all_smaller = [e["name"] for e in report[ref_side]["entities"]]

    # Entity → attribute maps and parent maps
    ref_data   = json_index.get((domain, ref_name),   (None, {}))[1]
    other_data = json_index.get((domain, other_name), (None, {}))[1]
    ref_emap   = _entity_map(ref_data)
    other_emap = _entity_map(other_data)
    ref_pmap   = _build_parent_map(ref_emap)
    other_pmap = _build_parent_map(other_emap)

    print(f"  [{domain}] {ref_name[:35]} | {len(residual)} residual entities")

    # Direction 1: residual small-model entities subsumed by larger-model entities
    fwd_groups = _detect_subsumptions(
        residual_names   = residual,
        residual_emap    = ref_emap,
        residual_pmap    = ref_pmap,
        candidate_names  = all_larger,
        candidate_emap   = other_emap,
        candidate_pmap   = other_pmap,
        scorer           = scorer,
        fwd_thresh       = fwd_thresh,
        gap_thresh       = gap_thresh,
        direction_label  = "large_abstracts_small",
    )

    # Direction 2: residual small-model entities that abstract over larger-model entities
    rev_groups = _detect_subsumptions(
        residual_names   = residual,
        residual_emap    = ref_emap,
        residual_pmap    = ref_pmap,
        candidate_names  = all_larger,
        candidate_emap   = other_emap,
        candidate_pmap   = other_pmap,
        scorer           = scorer,
        fwd_thresh       = fwd_thresh,
        gap_thresh       = gap_thresh,
        direction_label  = "small_abstracts_large",
    ) if residual else []
    # For direction 2, we invert: the abstract entity is in the residual list
    # and it subsumes entities in the larger model.
    rev_groups_inv = _detect_subsumptions(
        residual_names   = all_larger,
        residual_emap    = other_emap,
        residual_pmap    = other_pmap,
        candidate_names  = residual,
        candidate_emap   = ref_emap,
        candidate_pmap   = ref_pmap,
        scorer           = scorer,
        fwd_thresh       = fwd_thresh,
        gap_thresh       = gap_thresh,
        direction_label  = "small_abstracts_large",
    ) if residual else []

    all_groups = fwd_groups + rev_groups_inv

    # Count how many residual entities are now covered
    covered = set()
    for g in fwd_groups:
        covered.update(g["subsumed_entities"])
    for g in rev_groups_inv:
        covered.add(g["abstract_entity"])   # the abstract entity in larger is covered

    # Residual entities covered by subsumption (in the smaller model)
    covered_smaller = {e for e in residual if e in covered}

    aml_matched      = report["summary"][ref_side]["matched"]
    stage2_recovered = len(already_matched)
    stage3_covered   = len(covered_smaller)
    total_accounted  = aml_matched + stage2_recovered + stage3_covered
    combined_cov     = round(100 * total_accounted / ref_total, 1) if ref_total else 0
    still_missing    = [e for e in residual if e not in covered_smaller]

    return {
        "pair":              f"{ref_name}_vs_{other_name}",
        "domain":            domain,
        "smaller_model":     ref_name,
        "larger_model":      other_name,
        "smaller_total":     ref_total,
        "aml_matched":       aml_matched,
        "stage2_recovered":  stage2_recovered,
        "stage3_covered":    stage3_covered,
        "combined_coverage": combined_cov,
        "fwd_thresh":        fwd_thresh,
        "gap_thresh":        gap_thresh,
        "subsumption_groups": all_groups,
        "still_missing":     still_missing,
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def run(
    fwd_thresh: float = DEFAULT_FWD,
    gap_thresh: float = DEFAULT_GAP,
    domain_filter: str | None = None,
    all_models: bool = False,
) -> None:
    os.makedirs(SUBSUMPTION_DIR, exist_ok=True)
    json_index = _build_json_index()
    vmodels    = _vmodel_names(json_index)   # derived from input file names
    scorer     = NLIScorer()

    report_files = sorted(glob.glob(
        os.path.join(REPORTS_DIR, "**", "*.json"), recursive=True
    ))

    all_results: list[dict] = []

    for jf in report_files:
        domain = os.path.basename(os.path.dirname(jf))
        if domain_filter and domain.lower() != domain_filter.lower():
            continue

        with open(jf, encoding="utf-8") as f:
            report = json.load(f)

        meta   = report["metadata"]
        name_a = meta["model_a"]
        name_b = meta["model_b"]

        if not all_models and (name_a not in vmodels or name_b not in vmodels):
            continue

        pair_stem = os.path.splitext(os.path.basename(jf))[0]
        print(f"\n{'='*60}")
        print(f"Pair: {pair_stem}")

        result = _process_pair(
            report, json_index, scorer,
            fwd_thresh, gap_thresh, domain, pair_stem,
        )
        all_results.append(result)

        # Save per-pair JSON
        out_path = os.path.join(SUBSUMPTION_DIR, domain, f"{pair_stem}_subsumption.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Print summary table
    print()
    print(f"{'Domain':<14} {'Smaller model':<44} "
          f"{'AML':>5} {'S2':>4} {'S3':>4} {'Combined':>10} {'Still missing':>14}")
    print("-" * 100)
    for r in all_results:
        aml_cov = round(100 * r["aml_matched"] / r["smaller_total"], 1) if r["smaller_total"] else 0
        print(
            f"{r['domain']:<14} {r['smaller_model'][:43]:<44} "
            f"{aml_cov:>4}% {r['stage2_recovered']:>4}  {r['stage3_covered']:>4}  "
            f"{r['combined_coverage']:>8}%  {len(r['still_missing']):>8}"
        )

    # Write CSV
    csv_rows = [
        {k: v for k, v in r.items() if k not in ("subsumption_groups", "still_missing")}
        for r in all_results
    ]
    if csv_rows:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nCSV  -> {OUT_CSV}")
        print(f"JSONs -> {SUBSUMPTION_DIR}")

    # Print subsumption groups
    print("\n=== Subsumption Groups (size >= 2) ===")
    for r in all_results:
        groups = [g for g in r["subsumption_groups"] if g["is_group"]]
        if not groups:
            continue
        print(f"\n[{r['domain']}] {r['smaller_model'][:50]}")
        for g in groups:
            arrow = "<-" if g["direction"] == "large_abstracts_small" else "->"
            print(f"  {g['abstract_entity']} {arrow} {g['subsumed_entities']}  "
                  f"(fwd: {g['fwd_scores']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fwd",    type=float, default=DEFAULT_FWD,
                        help="Min forward entailment score (default 0.50)")
    parser.add_argument("--gap",    type=float, default=DEFAULT_GAP,
                        help="Min asymmetry gap fwd-rev (default 0.15)")
    parser.add_argument("--domain", type=str,   default=None,
                        help="Run only for one domain (e.g. Automobile)")
    parser.add_argument("--all-models", action="store_true",
                        help="Include non-versioned models (Component Network etc.)")
    args = parser.parse_args()
    run(fwd_thresh=args.fwd, gap_thresh=args.gap, domain_filter=args.domain,
        all_models=args.all_models)
