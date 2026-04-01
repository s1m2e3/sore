"""
association_matcher.py
----------------------
Stage 5: Association-enriched MNLI for entities still unmatched after Stage 4.

Design
------
Stages 1-4 build entity descriptions solely from an entity's own observable
attributes and its composition hierarchy (ancestor chain + child components).
For entities that have sparse own attributes but rich structural roles —
Frame, ECU, TorqueConverter, TimingChain — this produces near-empty
descriptions that MNLI cannot distinguish.

Stage 5 enriches descriptions with each entity's explicit associations:

    "A frame entity.
     Relations: supports entity measuring Mass, Force (via Position);
                attached to entity measuring OperationalState (via Position);
                connected to entity measuring Torque (via AngularVelocity)."

The key invariant: partner entities are NEVER named — only their observable
type signatures appear. Observable types are drawn from a shared cross-model
vocabulary (Torque, Temperature, OperationalState, ...) and are therefore
model-name-agnostic. This means V1's 'EngineBlock' and V3's 'PropulsionUnit'
can match even though their names share no tokens, provided their relational
observable type fingerprints are similar.

If a JSON model has no associations (empty list), the description falls back
to the standard attribute-only description from mnli_matcher.

Verb extraction
---------------
Association names encode both participants and a relationship verb, e.g.:
  EngineToTransmissionCoupling  participants=[Engine, Transmission]
  RodLinksPistonToCrankshaft    participants=[ConnectingRod, Piston, Crankshaft]

The verb is estimated by:
  1. camelCase-splitting the association name into tokens
  2. camelCase-splitting each participant name into tokens
  3. Removing participant tokens from the association tokens
  4. The remainder is the verb phrase

Inputs (read at runtime)
------------------------
  outputs/child/<domain>/*_child.json          Stage 4 still_missing lists
  outputs/subsumption/<domain>/*_subsumption.json  fallback if no child file
  inputs/.../<domain>/*.json                   JSON models with associations

Outputs
-------
  outputs/association/<domain>/<pair>_assoc.json   per-pair results
  outputs/alignment_summary_association.csv        combined summary

Usage
-----
    cd ontology_matching
    python association_matcher.py                    # all domains
    python association_matcher.py --domain Automobile
    python association_matcher.py --threshold 0.45
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from typing import Any

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

CHILD_DIR       = os.path.join(os.path.dirname(__file__), "outputs", "child")
SUBSUMPTION_DIR = os.path.join(os.path.dirname(__file__), "outputs", "subsumption")
ASSOC_DIR       = os.path.join(os.path.dirname(__file__), "outputs", "association")
OUT_CSV         = os.path.join(os.path.dirname(__file__), "outputs",
                               "alignment_summary_association.csv")

DEFAULT_THRESHOLD  = 0.45
MAX_ASSOCS         = 4   # max associations to include per entity
MAX_PARTNER_TYPES  = 3   # max observable types from a partner entity
MAX_ASSOC_TYPES    = 2   # max observable types on the association itself


# --------------------------------------------------------------------------- #
# Verb extraction                                                              #
# --------------------------------------------------------------------------- #

def _camel_tokens(name: str) -> list[str]:
    """Split a camelCase/PascalCase name into lowercase tokens."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [t.lower() for t in s.split() if t]


def _extract_verb(assoc_name: str, participants: list[str]) -> str:
    """
    Extract the relationship verb phrase from an association name by removing
    the camelCase tokens of all participant names.

    'EngineToTransmissionCoupling', ['Engine','Transmission']  ->  'to coupling'
    'RodLinksPistonToCrankshaft',   ['ConnectingRod','Piston','Crankshaft']
                                                               ->  'links to'
    Falls back to the full split name if nothing remains.
    """
    assoc_tokens = _camel_tokens(assoc_name)
    part_tokens: set[str] = set()
    for p in participants:
        for t in _camel_tokens(p):
            part_tokens.add(t)

    verb_tokens = [t for t in assoc_tokens if t not in part_tokens]
    if verb_tokens:
        return " ".join(verb_tokens)
    # fallback: whole association name split
    return " ".join(assoc_tokens)


# --------------------------------------------------------------------------- #
# Association index                                                            #
# --------------------------------------------------------------------------- #

def _build_assoc_index(
    json_data: dict,
    emap: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """
    Build a per-entity association index.

    Returns: entity_name -> list of {
        "verb":             str,          relationship phrase
        "partner_obs":      list[str],    observable types of the partner entity
        "assoc_obs":        list[str],    observable types on the association
    }

    Entities not present in emap are ignored.
    Associations with no participants in emap contribute nothing.
    For associations with N > 2 participants, each participant is indexed
    against a "partners" entry that merges all other participants' signatures.
    """
    entity_names = set(emap.keys())
    index: dict[str, list[dict]] = {name: [] for name in entity_names}
    assocs = json_data.get("associations", [])

    for a in assocs:
        assoc_name   = a.get("associationName", "") or a.get("name", "")
        participants = a.get("associationParticipants", []) or a.get("participants", [])
        assoc_attrs  = a.get("associationAttributes", []) or a.get("attributes", [])

        # Observable types carried on the association itself
        assoc_obs = [
            x.get("type", "") for x in assoc_attrs
            if x.get("type", "") and x.get("type", "") not in entity_names
        ][:MAX_ASSOC_TYPES]

        # Only process participants that are in our entity map
        known = [p for p in participants if p in entity_names]
        if not known:
            continue

        verb = _extract_verb(assoc_name, participants)

        for focal in known:
            # All other known participants are "partners"
            partners = [p for p in known if p != focal]
            if not partners:
                # Self-association or single participant — still record assoc_obs
                index[focal].append({
                    "verb":        verb,
                    "partner_obs": [],
                    "assoc_obs":   assoc_obs,
                })
                continue

            # Merge observable types from all partners
            partner_obs: list[str] = []
            seen: set[str] = set()
            for p in partners:
                for attr in emap.get(p, []):
                    t = attr.get("type", "")
                    if t and t not in entity_names and t not in seen:
                        seen.add(t)
                        partner_obs.append(t)
                        if len(partner_obs) >= MAX_PARTNER_TYPES:
                            break
                if len(partner_obs) >= MAX_PARTNER_TYPES:
                    break

            index[focal].append({
                "verb":        verb,
                "partner_obs": partner_obs,
                "assoc_obs":   assoc_obs,
            })

    return index


# --------------------------------------------------------------------------- #
# Association-enriched description                                             #
# --------------------------------------------------------------------------- #

def _assoc_description(
    ent_name: str,
    emap: dict[str, list[dict]],
    assoc_index: dict[str, list[dict]],
    pmap: dict[str, list[str]],
) -> str:
    """
    Build a natural-language description that includes:
      1. Ancestor chain context (part of: ...)
      2. Own observable attributes
      3. Association relations (verb + partner obs types + assoc obs types)

    Falls back to _build_description if no associations are indexed.
    """
    assoc_list = assoc_index.get(ent_name, [])

    # Base description (handles own attrs + ancestor chain)
    base = _build_description(ent_name, emap.get(ent_name, []), _ancestors(ent_name, pmap))

    if not assoc_list:
        return base

    rel_parts: list[str] = []
    for entry in assoc_list[:MAX_ASSOCS]:
        verb       = entry["verb"]
        p_obs      = entry["partner_obs"]
        a_obs      = entry["assoc_obs"]

        # Build the relation phrase
        obs_parts: list[str] = []
        if p_obs:
            obs_parts.append("entity measuring " + ", ".join(_split_camel(t) for t in p_obs))
        if a_obs:
            obs_parts.append("via " + ", ".join(_split_camel(t) for t in a_obs))

        if obs_parts:
            rel_parts.append(f"{verb} {' '.join(obs_parts)}")
        else:
            rel_parts.append(verb)

    if not rel_parts:
        return base

    relation_text = " Relations: " + "; ".join(rel_parts) + "."

    # Append to base (strip trailing period first)
    return base.rstrip(".") + "." + relation_text


# --------------------------------------------------------------------------- #
# Per-pair driver                                                              #
# --------------------------------------------------------------------------- #

def _load_prev_result(domain: str, stem: str) -> dict | None:
    """
    Load the most recent stage output for this pair.
    Preference order: child (S4) > subsumption (S3).
    Returns None if neither exists.
    """
    child_path = os.path.join(CHILD_DIR, domain, f"{stem}_child.json")
    if os.path.exists(child_path):
        with open(child_path, encoding="utf-8") as f:
            return json.load(f)

    sub_path = os.path.join(SUBSUMPTION_DIR, domain, f"{stem}_subsumption.json")
    if os.path.exists(sub_path):
        with open(sub_path, encoding="utf-8") as f:
            return json.load(f)

    return None


def _process_pair(
    prev: dict,
    json_index: dict,
    scorer: NLIScorer,
    threshold: float,
) -> dict:
    domain       = prev["domain"]
    smaller_name = prev["smaller_model"]
    larger_name  = prev["larger_model"]
    still_missing: list[str] = prev.get("still_missing", [])

    base_result = {
        "pair":              prev["pair"],
        "domain":            domain,
        "smaller_model":     smaller_name,
        "larger_model":      larger_name,
        "smaller_total":     prev["smaller_total"],
        "aml_matched":       prev["aml_matched"],
        "stage2_recovered":  prev.get("stage2_recovered",  0),
        "stage3_covered":    prev.get("stage3_covered",    0),
        "stage4_recovered":  prev.get("stage4_recovered",  0),
        "stage5_recovered":  0,
        "combined_coverage": prev.get("combined_coverage", 0),
        "threshold":         threshold,
        "new_matches":       [],
        "still_missing":     still_missing,
    }

    if not still_missing:
        return base_result

    ref_data   = json_index.get((domain, smaller_name), (None, {}))[1]
    other_data = json_index.get((domain, larger_name),  (None, {}))[1]

    if not ref_data or not other_data:
        print(f"  WARNING: missing JSON data for {smaller_name} or {larger_name}")
        return base_result

    ref_emap   = _entity_map(ref_data)
    other_emap = _entity_map(other_data)
    ref_pmap   = _build_parent_map(ref_emap)
    other_pmap = _build_parent_map(other_emap)

    ref_assoc   = _build_assoc_index(ref_data,   ref_emap)
    other_assoc = _build_assoc_index(other_data, other_emap)

    has_ref_assoc   = any(v for v in ref_assoc.values())
    has_other_assoc = any(v for v in other_assoc.values())
    print(f"  [{domain}] {smaller_name[:40]}")
    print(f"    {len(still_missing)} residual entities  |  "
          f"smaller assocs={'yes' if has_ref_assoc else 'NO'}  "
          f"larger assocs={'yes' if has_other_assoc else 'NO'}")

    all_larger = list(other_emap.keys())

    # Build descriptions for residual entities (smaller model)
    pairs: list[tuple[str, str, str, str]] = []
    for u in still_missing:
        desc_u = _assoc_description(u, ref_emap, ref_assoc, ref_pmap)
        for e in all_larger:
            desc_e = _assoc_description(e, other_emap, other_assoc, other_pmap)
            pairs.append((u, desc_u, e, desc_e))

    print(f"    Scoring {len(pairs)} pairs...", flush=True)
    scored = scorer.batch_score(pairs)

    used_smaller: set[str] = set()
    used_larger:  set[str] = set()
    new_matches: list[dict] = []

    for name_u, name_e, score in scored:
        if score < threshold:
            break
        if name_u in used_smaller or name_e in used_larger:
            continue
        new_matches.append({
            "smaller_entity": name_u,
            "larger_entity":  name_e,
            "assoc_score":    round(score, 4),
            "desc_smaller":   _assoc_description(name_u, ref_emap, ref_assoc, ref_pmap),
            "desc_larger":    _assoc_description(name_e, other_emap, other_assoc, other_pmap),
        })
        used_smaller.add(name_u)
        used_larger.add(name_e)

    matched_names = {m["smaller_entity"] for m in new_matches}
    remaining     = [e for e in still_missing if e not in matched_names]

    total         = prev["smaller_total"]
    total_matched = (prev["aml_matched"]
                     + prev.get("stage2_recovered", 0)
                     + prev.get("stage3_covered",   0)
                     + prev.get("stage4_recovered", 0)
                     + len(new_matches))
    combined_cov  = round(100 * total_matched / total, 1) if total else 0

    base_result.update({
        "stage5_recovered":  len(new_matches),
        "combined_coverage": combined_cov,
        "new_matches":       new_matches,
        "still_missing":     remaining,
    })
    return base_result


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def run(
    threshold: float = DEFAULT_THRESHOLD,
    domain_filter: str | None = None,
    all_models: bool = False,
) -> None:
    os.makedirs(ASSOC_DIR, exist_ok=True)
    json_index = _build_json_index()
    vmodels    = _vmodel_names(json_index)
    scorer     = NLIScorer()

    # Collect all pair stems by scanning child + subsumption dirs
    stems_by_domain: dict[str, list[str]] = {}
    for directory in [CHILD_DIR, SUBSUMPTION_DIR]:
        for jf in sorted(glob.glob(os.path.join(directory, "**", "*.json"), recursive=True)):
            domain = os.path.basename(os.path.dirname(jf))
            stem   = re.sub(r"_(child|subsumption)$", "",
                            os.path.splitext(os.path.basename(jf))[0])
            stems_by_domain.setdefault(domain, [])
            if stem not in stems_by_domain[domain]:
                stems_by_domain[domain].append(stem)

    all_results: list[dict] = []

    for domain, stems in sorted(stems_by_domain.items()):
        if domain_filter and domain.lower() != domain_filter.lower():
            continue

        for stem in stems:
            prev = _load_prev_result(domain, stem)
            if prev is None:
                continue

            # Only process versioned-model pairs unless --all-models is set
            if not all_models and (prev.get("smaller_model", "") not in vmodels or
                    prev.get("larger_model", "") not in vmodels):
                continue

            print(f"\n{'='*60}")
            print(f"Pair: {stem}")

            result = _process_pair(prev, json_index, scorer, threshold)
            all_results.append(result)

            out_path = os.path.join(ASSOC_DIR, domain, f"{stem}_assoc.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    # Summary table
    print()
    print(f"{'Domain':<14} {'Smaller model':<44} "
          f"{'AML':>5} {'S2':>4} {'S3':>4} {'S4':>4} {'S5':>4} {'Combined':>10} {'Still':>6}")
    print("-" * 105)
    for r in all_results:
        t = r["smaller_total"]
        aml_pct = round(100 * r["aml_matched"] / t, 1) if t else 0
        print(
            f"{r['domain']:<14} {r['smaller_model'][:43]:<44} "
            f"{aml_pct:>4}% "
            f"{r['stage2_recovered']:>4}  "
            f"{r['stage3_covered']:>4}  "
            f"{r['stage4_recovered']:>4}  "
            f"{r['stage5_recovered']:>4}  "
            f"{r['combined_coverage']:>8}%  "
            f"{len(r['still_missing']):>5}"
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
        print(f"JSONs -> {ASSOC_DIR}")

    # Print new matches
    print("\n=== Stage 5 New Matches ===")
    for r in all_results:
        if not r["new_matches"]:
            continue
        print(f"\n[{r['domain']}] {r['smaller_model'][:55]}")
        for m in r["new_matches"]:
            print(f"  {m['smaller_entity']} -> {m['larger_entity']}  (score: {m['assoc_score']})")
            print(f"    S: {m['desc_smaller'][:120]}")
            print(f"    L: {m['desc_larger'][:120]}")

    # Print remaining unmatched
    print("\n=== Still Unmatched After S5 ===")
    for r in all_results:
        if not r["still_missing"]:
            continue
        print(f"[{r['domain']}] {r['smaller_model'][:55]}: {r['still_missing']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Min mutual entailment score (default 0.45)")
    parser.add_argument("--domain", type=str, default=None,
                        help="Run only for one domain (e.g. Automobile)")
    parser.add_argument("--all-models", action="store_true",
                        help="Include non-versioned models (Component Network etc.)")
    args = parser.parse_args()
    run(threshold=args.threshold, domain_filter=args.domain, all_models=args.all_models)
