"""
child_matcher.py
----------------
Stage 4: Child-composition enriched MNLI for entities still unmatched after
Stage 3 (subsumption).

By Stage 4, every entity that could be matched via:
  - lexical similarity         (Stage 1 / AML)
  - semantic name equivalence  (Stage 2 / MNLI + ancestors)
  - subsumption abstraction    (Stage 3 / asymmetric entailment + ancestors)
has already been resolved.

The remaining entities are the hardest cases.  They often have sparse or
generic own-attribute sets and are not obviously named.  Their most
discriminative signal is frequently their *composition* — what sub-entities
they contain and what those sub-entities measure.

This stage enriches each description with:
  1. Ancestor chain         (part of: X, Y)           — same as S2/S3
  2. Child component summary (containing: A [a1,a2]; B [b1]) — new in S4

Inputs (all read at runtime, nothing hardcoded):
  outputs/subsumption/<domain>/*_subsumption.json  — still_missing lists
  inputs/.../<domain>/*.json                        — entity + attribute data

Output:
  outputs/child/<domain>/<pair>_child.json   per-pair matches
  outputs/alignment_summary_child.csv        combined summary

Usage
-----
    cd ontology_matching
    python child_matcher.py                    # all domains
    python child_matcher.py --domain Automobile
    python child_matcher.py --threshold 0.55
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

from mnli_matcher import (
    NLIScorer,
    _ancestors,
    _build_description,
    _build_json_index,
    _build_parent_map,
    _entity_map,
    _split_camel,
    _vmodel_names,
    INPUTS_DIR,
    MODEL_NAME,
)

SUBSUMPTION_DIR = os.path.join(os.path.dirname(__file__), "outputs", "subsumption")
CHILD_DIR       = os.path.join(os.path.dirname(__file__), "outputs", "child")
OUT_CSV         = os.path.join(os.path.dirname(__file__), "outputs",
                               "alignment_summary_child.csv")

DEFAULT_THRESHOLD  = 0.55   # higher than S2: richer descriptions, need tighter gate
MAX_CHILDREN       = 3      # entity-type children to expand per entity
MAX_ATTRS_PER_CHILD = 3     # observable attributes per child


# --------------------------------------------------------------------------- #
# Child-enriched description builder                                           #
# --------------------------------------------------------------------------- #

def _child_description(
    ent_name: str,
    emap: dict[str, list[dict]],
    pmap: dict[str, list[str]],
) -> str:
    """
    Build a description that combines:
      - ancestor chain (part of: ...)
      - child component summary (containing: A [obs1, obs2]; B [obs3])
      - own observable attributes

    Only observable-type (non-entity) attributes are listed inline for
    children, avoiding recursive expansion.
    """
    attrs        = emap.get(ent_name, [])
    entity_names = set(emap.keys())
    ancs         = _ancestors(ent_name, pmap)

    # --- own observable attributes (non-entity types) ---
    own_obs = [a for a in attrs if a.get("type", "") not in entity_names]

    # --- child component summaries ---
    child_parts: list[str] = []
    for a in attrs:
        child_type = a.get("type", "")
        if child_type not in entity_names:
            continue
        if len(child_parts) >= MAX_CHILDREN:
            break
        child_obs_names = [
            _split_camel(ca.get("name", ""))
            for ca in emap.get(child_type, [])
            if ca.get("type", "") not in entity_names
        ][:MAX_ATTRS_PER_CHILD]
        label = _split_camel(child_type)
        if child_obs_names:
            child_parts.append(f"{label} [{', '.join(child_obs_names)}]")
        else:
            child_parts.append(label)

    # --- assemble ---
    name_text = _split_camel(ent_name)

    ancestor_text = ""
    if ancs:
        ancestor_text = " (part of: " + ", ".join(_split_camel(a) for a in ancs) + ")"

    child_text = ""
    if child_parts:
        child_text = " containing: " + "; ".join(child_parts) + ";"

    if not own_obs:
        return f"A {name_text}{ancestor_text} entity{child_text}."

    # Reuse _build_description's attribute formatting for own obs attrs
    return _build_description(ent_name, own_obs, ancs) \
           .replace(f"A {name_text}", f"A {name_text}{child_text}", 1) \
           if not child_text else \
           f"A {name_text}{ancestor_text} entity{child_text} " \
           + _build_description(ent_name, own_obs).split(" entity ")[-1]


# --------------------------------------------------------------------------- #
# Per-pair driver                                                               #
# --------------------------------------------------------------------------- #

def _process_pair(
    sub_result: dict,
    json_index: dict,
    scorer: NLIScorer,
    threshold: float,
) -> dict:
    domain       = sub_result["domain"]
    smaller_name = sub_result["smaller_model"]
    larger_name  = sub_result["larger_model"]
    still_missing: list[str] = sub_result.get("still_missing", [])

    if not still_missing:
        return {
            "pair":          sub_result["pair"],
            "domain":        domain,
            "smaller_model": smaller_name,
            "larger_model":  larger_name,
            "smaller_total": sub_result["smaller_total"],
            "stage4_recovered": 0,
            "new_matches":   [],
            "still_missing": [],
        }

    ref_data   = json_index.get((domain, smaller_name), (None, {}))[1]
    other_data = json_index.get((domain, larger_name),  (None, {}))[1]
    ref_emap   = _entity_map(ref_data)
    other_emap = _entity_map(other_data)
    ref_pmap   = _build_parent_map(ref_emap)
    other_pmap = _build_parent_map(other_emap)

    all_larger = list(other_emap.keys())

    print(f"  [{domain}] {smaller_name[:40]} | {len(still_missing)} residual "
          f"x {len(all_larger)} candidates = {len(still_missing)*len(all_larger)} pairs")

    # Score all (still_missing × all_larger) pairs with child-enriched descriptions
    pairs = []
    for u in still_missing:
        desc_u = _child_description(u, ref_emap, ref_pmap)
        for e in all_larger:
            desc_e = _child_description(e, other_emap, other_pmap)
            pairs.append((u, desc_u, e, desc_e))

    scored = scorer.batch_score(pairs)

    used_smaller: set[str] = set()
    used_larger:  set[str] = set()
    new_matches = []
    for name_u, name_e, score in scored:
        if score < threshold:
            break
        if name_u in used_smaller or name_e in used_larger:
            continue
        new_matches.append({
            "smaller_entity": name_u,
            "larger_entity":  name_e,
            "child_score":    round(score, 4),
            "desc_smaller":   _child_description(name_u, ref_emap, ref_pmap),
            "desc_larger":    _child_description(name_e, other_emap, other_pmap),
        })
        used_smaller.add(name_u)
        used_larger.add(name_e)

    matched_names = {m["smaller_entity"] for m in new_matches}
    remaining     = [e for e in still_missing if e not in matched_names]

    return {
        "pair":              sub_result["pair"],
        "domain":            domain,
        "smaller_model":     smaller_name,
        "larger_model":      larger_name,
        "smaller_total":     sub_result["smaller_total"],
        "aml_matched":       sub_result["aml_matched"],
        "stage2_recovered":  sub_result["stage2_recovered"],
        "stage3_covered":    sub_result["stage3_covered"],
        "stage4_recovered":  len(new_matches),
        "combined_coverage": round(
            100 * (sub_result["aml_matched"]
                   + sub_result["stage2_recovered"]
                   + sub_result["stage3_covered"]
                   + len(new_matches))
            / sub_result["smaller_total"], 1
        ) if sub_result["smaller_total"] else 0,
        "threshold":         threshold,
        "new_matches":       new_matches,
        "still_missing":     remaining,
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def run(
    threshold: float = DEFAULT_THRESHOLD,
    domain_filter: str | None = None,
    all_models: bool = False,
) -> None:
    os.makedirs(CHILD_DIR, exist_ok=True)
    json_index = _build_json_index()
    vmodels    = _vmodel_names(json_index)
    scorer     = NLIScorer()

    sub_files = sorted(glob.glob(
        os.path.join(SUBSUMPTION_DIR, "**", "*.json"), recursive=True
    ))

    all_results: list[dict] = []

    for jf in sub_files:
        domain = os.path.basename(os.path.dirname(jf))
        if domain_filter and domain.lower() != domain_filter.lower():
            continue

        with open(jf, encoding="utf-8") as f:
            sub_result = json.load(f)

        # Only process v-model pairs unless --all-models is set
        if not all_models and (sub_result.get("smaller_model", "") not in vmodels or
                sub_result.get("larger_model", "") not in vmodels):
            continue

        print(f"\n{'='*60}")
        print(f"Pair: {sub_result['pair']}")

        result = _process_pair(sub_result, json_index, scorer, threshold)
        all_results.append(result)

        out_path = os.path.join(CHILD_DIR, domain,
                                f"{sub_result['pair']}_child.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Summary table
    print()
    print(f"{'Domain':<14} {'Smaller model':<44} "
          f"{'AML':>5} {'S2':>4} {'S3':>4} {'S4':>4} {'Combined':>10} {'Still missing':>14}")
    print("-" * 108)
    for r in all_results:
        aml_cov = round(100 * r["aml_matched"] / r["smaller_total"], 1) \
                  if r.get("smaller_total") else 0
        print(
            f"{r['domain']:<14} {r['smaller_model'][:43]:<44} "
            f"{aml_cov:>4}% {r['stage2_recovered']:>4}  {r['stage3_covered']:>4}  "
            f"{r['stage4_recovered']:>4}  {r['combined_coverage']:>8}%  "
            f"{len(r['still_missing']):>8}"
        )

    # CSV
    csv_rows = [
        {k: v for k, v in r.items() if k not in ("new_matches", "still_missing")}
        for r in all_results
    ]
    if csv_rows:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nCSV  -> {OUT_CSV}")
        print(f"JSONs -> {CHILD_DIR}")

    # New matches
    print("\n=== Stage 4 New Matches ===")
    for r in all_results:
        if not r["new_matches"]:
            continue
        print(f"\n[{r['domain']}] {r['smaller_model'][:50]}")
        for m in r["new_matches"]:
            print(f"  {m['smaller_entity']} -> {m['larger_entity']}  "
                  f"(score: {m['child_score']})")
            print(f"    S: {m['desc_smaller']}")
            print(f"    L: {m['desc_larger']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Min mutual entailment score (default 0.55)")
    parser.add_argument("--domain", type=str, default=None,
                        help="Run only for one domain (e.g. Automobile)")
    parser.add_argument("--all-models", action="store_true",
                        help="Include non-versioned models (Component Network etc.)")
    args = parser.parse_args()
    run(threshold=args.threshold, domain_filter=args.domain, all_models=args.all_models)
