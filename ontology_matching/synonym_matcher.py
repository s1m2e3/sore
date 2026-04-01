"""
synonym_matcher.py
------------------
Stage 6: Dense-embedding synonym / semantic-similarity matching using SBERT.

Encodes each entity's name-first description as a dense vector with a
sentence-transformer model (paraphrase-MiniLM-L6-v2), then finds the
best cosine-similarity match for each still-unmatched smaller entity.

Why this stage complements S2–S5
---------------------------------
MNLI (cross-encoder) scores *entailment*, not similarity; it struggles when
entity descriptions are sparse or when two entities share a name-root across
naming conventions (EngineBlock / EngineAssembly / Engine).  SBERT (bi-encoder)
learns a shared semantic embedding space: synonymous names cluster together
even when descriptions differ, making it ideal for:

  • Pure synonyms:      Vehicle ≈ Automobile,  GearBox ≈ Transmission
  • Granularity hops:  EngineBlock ≈ EngineAssembly ≈ Engine
  • Naming-convention gaps that lexical AML misses

Description format
------------------
  "<EntityName>: <observable attribute types>. <association relation phrases>"

The entity name is placed first so the encoder's attention gives it maximum
weight — critical for synonym detection.  Association phrases reuse the same
enrichment logic as Stage 5 so that relational context is captured when names
alone are ambiguous.

Inputs (read at runtime)
------------------------
  outputs/association/<domain>/*_assoc.json    Stage 5 still_missing (preferred)
  outputs/child/<domain>/*_child.json          fallback
  outputs/subsumption/<domain>/*_subsumption.json  fallback
  outputs/mnli/<domain>/*_mnli.json            last-resort fallback
  inputs/.../<domain>/*.json                   JSON models

Outputs
-------
  outputs/synonym/<domain>/<pair>_synonym.json  per-pair results
  outputs/alignment_summary_synonym.csv          combined summary

Usage
-----
    cd ontology_matching

    python synonym_matcher.py --domain Automobile
    python synonym_matcher.py --domain Automobile --all-models
    python synonym_matcher.py --threshold 0.72
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from typing import Any

import numpy as np
import torch

# ── Point sentence-transformers at the local models cache ─────────────────── #
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(_MODELS_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", _MODELS_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _MODELS_DIR)
# ─────────────────────────────────────────────────────────────────────────── #

from mnli_matcher import (
    _ancestors,
    _build_description,
    _build_json_index,
    _build_parent_map,
    _entity_map,
    _split_camel,
    _vmodel_names,
    INPUTS_DIR,
)
from association_matcher import (
    _build_assoc_index,
    _assoc_description,
    MAX_ASSOCS,
)

ASSOC_DIR       = os.path.join(os.path.dirname(__file__), "outputs", "association")
CHILD_DIR       = os.path.join(os.path.dirname(__file__), "outputs", "child")
SUBSUMPTION_DIR = os.path.join(os.path.dirname(__file__), "outputs", "subsumption")
MNLI_DIR        = os.path.join(os.path.dirname(__file__), "outputs", "mnli")
SYNONYM_DIR     = os.path.join(os.path.dirname(__file__), "outputs", "synonym")
OUT_CSV         = os.path.join(os.path.dirname(__file__), "outputs",
                               "alignment_summary_synonym.csv")

SBERT_MODEL     = "paraphrase-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.72

# --------------------------------------------------------------------------- #
# SBERT encoder                                                               #
# --------------------------------------------------------------------------- #

class SBERTEncoder:
    """Thin wrapper around sentence-transformers for cosine-similarity search."""

    def __init__(self, model_name: str = SBERT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading SBERT model '{model_name}' on {device} …")
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings (shape: N × dim)."""
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine = dot product after L2-norm
            show_progress_bar=False,
        )

    def top_matches(
        self,
        query_texts: list[str],
        corpus_texts: list[str],
        k: int = 1,
    ) -> list[list[tuple[int, float]]]:
        """
        For each query return up to k (corpus_index, cosine_score) pairs,
        sorted descending by score.
        """
        if not query_texts or not corpus_texts:
            return [[] for _ in query_texts]
        q_emb = self.encode(query_texts)   # Q × dim
        c_emb = self.encode(corpus_texts)  # C × dim
        sims  = q_emb @ c_emb.T            # Q × C
        results = []
        for row in sims:
            top = sorted(enumerate(row.tolist()), key=lambda x: -x[1])[:k]
            results.append(top)
        return results


# --------------------------------------------------------------------------- #
# Name-first SBERT description                                                #
# --------------------------------------------------------------------------- #

def _sbert_description(
    ent_name: str,
    emap: dict,
    assoc_index: dict,
    pmap: dict,
) -> str:
    """
    Build a description where the entity name comes first (maximises SBERT
    attention on the name token for synonym detection), followed by observable
    attribute types and association relations.
    """
    # Observable attribute types (excluding other entity references)
    entity_names = set(emap.keys())
    obs_types = [
        a.get("type", "") for a in emap.get(ent_name, [])
        if a.get("type", "") and a.get("type", "") not in entity_names
    ]

    parts: list[str] = []

    # 1. Name as the leading token
    readable_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", ent_name)
    readable_name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", readable_name)
    parts.append(readable_name)

    # 2. Observable attributes
    if obs_types:
        parts.append("measures " + ", ".join(_split_camel(t) for t in obs_types[:5]))

    # 3. Ancestor context
    ancs = _ancestors(ent_name, pmap)
    if ancs:
        parts.append("part of " + ", ".join(
            re.sub(r"([a-z])([A-Z])", r"\1 \2", a) for a in ancs[:2]
        ))

    # 4. Association relations (verb + observable partner types)
    for entry in assoc_index.get(ent_name, [])[:MAX_ASSOCS]:
        verb  = entry["verb"]
        p_obs = entry["partner_obs"]
        a_obs = entry["assoc_obs"]
        obs_p = []
        if p_obs:
            obs_p.append("entity measuring " + ", ".join(_split_camel(t) for t in p_obs))
        if a_obs:
            obs_p.append("via " + ", ".join(_split_camel(t) for t in a_obs))
        if obs_p:
            parts.append(verb + " " + " ".join(obs_p))
        else:
            parts.append(verb)

    return ". ".join(parts)


# --------------------------------------------------------------------------- #
# Load previous stage output                                                  #
# --------------------------------------------------------------------------- #

def _load_prev_result(domain: str, stem: str) -> dict | None:
    """
    Load the most recent pipeline stage result.
    Preference: S5 assoc > S4 child > S3 subsumption > S2 mnli.
    """
    for directory, suffix in [
        (ASSOC_DIR,       "assoc"),
        (CHILD_DIR,       "child"),
        (SUBSUMPTION_DIR, "subsumption"),
        (MNLI_DIR,        "mnli"),
    ]:
        p = os.path.join(directory, domain, f"{stem}_{suffix}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return None


# --------------------------------------------------------------------------- #
# Per-pair driver                                                              #
# --------------------------------------------------------------------------- #

def _process_pair(
    prev: dict,
    json_index: dict,
    encoder: SBERTEncoder,
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
        "stage5_recovered":  prev.get("stage5_recovered",  0),
        "stage6_recovered":  0,
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

    print(f"  [{domain}] {smaller_name[:50]}")
    print(f"    {len(still_missing)} residual entities to encode", flush=True)

    all_larger = list(other_emap.keys())

    # Build name-first SBERT descriptions
    query_descs  = [_sbert_description(u, ref_emap, ref_assoc, ref_pmap)
                    for u in still_missing]
    corpus_descs = [_sbert_description(e, other_emap, other_assoc, other_pmap)
                    for e in all_larger]

    top_matches = encoder.top_matches(query_descs, corpus_descs, k=1)

    used_larger: set[str] = set()
    scored: list[tuple[str, str, float, str, str]] = []
    for u, matches in zip(still_missing, top_matches):
        if not matches:
            continue
        idx, score = matches[0]
        e = all_larger[idx]
        scored.append((u, e, score, query_descs[still_missing.index(u)], corpus_descs[idx]))

    # Sort descending, greedy 1-to-1 assignment
    scored.sort(key=lambda x: -x[2])

    used_smaller: set[str] = set()
    new_matches: list[dict] = []
    for u, e, score, desc_u, desc_e in scored:
        if score < threshold:
            continue
        if u in used_smaller or e in used_larger:
            continue
        new_matches.append({
            "smaller_entity": u,
            "larger_entity":  e,
            "sbert_score":    round(float(score), 4),
            "desc_smaller":   desc_u,
            "desc_larger":    desc_e,
        })
        used_smaller.add(u)
        used_larger.add(e)

    matched_names = {m["smaller_entity"] for m in new_matches}
    remaining     = [e for e in still_missing if e not in matched_names]

    total         = prev["smaller_total"]
    total_matched = (prev["aml_matched"]
                     + prev.get("stage2_recovered", 0)
                     + prev.get("stage3_covered",   0)
                     + prev.get("stage4_recovered", 0)
                     + prev.get("stage5_recovered", 0)
                     + len(new_matches))
    combined_cov  = round(100 * total_matched / total, 1) if total else 0

    base_result.update({
        "stage6_recovered":  len(new_matches),
        "combined_coverage": combined_cov,
        "new_matches":       new_matches,
        "still_missing":     remaining,
    })
    return base_result


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def run(
    threshold: float = DEFAULT_THRESHOLD,
    domain_filter: str | None = None,
    all_models: bool = False,
) -> None:
    os.makedirs(SYNONYM_DIR, exist_ok=True)
    json_index = _build_json_index()
    vmodels    = _vmodel_names(json_index)
    encoder    = SBERTEncoder(SBERT_MODEL)

    # Collect all pair stems from previous stage outputs
    stems_by_domain: dict[str, list[str]] = {}
    for directory, suffix in [
        (ASSOC_DIR,       "assoc"),
        (CHILD_DIR,       "child"),
        (SUBSUMPTION_DIR, "subsumption"),
        (MNLI_DIR,        "mnli"),
    ]:
        for jf in sorted(glob.glob(os.path.join(directory, "**", "*.json"),
                                   recursive=True)):
            domain = os.path.basename(os.path.dirname(jf))
            stem   = re.sub(rf"_{suffix}$", "",
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

            if not all_models and (
                prev.get("smaller_model", "") not in vmodels or
                prev.get("larger_model", "") not in vmodels
            ):
                continue

            print(f"\n{'='*60}")
            print(f"Pair: {stem}")

            result = _process_pair(prev, json_index, encoder, threshold)
            all_results.append(result)

            out_path = os.path.join(SYNONYM_DIR, domain, f"{stem}_synonym.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    # Summary table
    print()
    print(f"{'Domain':<14} {'Smaller model':<44} "
          f"{'AML':>5} {'S2':>4} {'S3':>4} {'S4':>4} {'S5':>4} {'S6':>4} "
          f"{'Combined':>10} {'Still':>6}")
    print("-" * 110)
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
            f"{r['stage6_recovered']:>4}  "
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
        print(f"JSONs -> {SYNONYM_DIR}")

    # Print new matches
    print("\n=== Stage 6 New Matches ===")
    for r in all_results:
        if not r["new_matches"]:
            continue
        print(f"\n[{r['domain']}] {r['smaller_model'][:55]}")
        for m in r["new_matches"]:
            print(f"  {m['smaller_entity']} -> {m['larger_entity']}  "
                  f"(sbert: {m['sbert_score']})")
            print(f"    S: {m['desc_smaller'][:120]}")
            print(f"    L: {m['desc_larger'][:120]}")

    print("\n=== Still Unmatched After S6 ===")
    for r in all_results:
        if not r["still_missing"]:
            continue
        print(f"[{r['domain']}] {r['smaller_model'][:55]}: {r['still_missing']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 6: SBERT dense-embedding synonym matching."
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Min cosine similarity score (default 0.72)")
    parser.add_argument("--domain", type=str, default=None,
                        help="Run only for one domain (e.g. Automobile)")
    parser.add_argument("--all-models", action="store_true",
                        help="Include non-versioned models (Component Network etc.)")
    args = parser.parse_args()
    run(threshold=args.threshold, domain_filter=args.domain,
        all_models=args.all_models)
