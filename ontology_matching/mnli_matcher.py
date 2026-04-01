"""
mnli_matcher.py
---------------
Semantic layer on top of AML: uses a cross-encoder NLI model to recover
matches for entities that AML left unmapped (lexical similarity failed).

Design
------
For every pair (smaller model, larger model):
  1. Identify unmapped entities in the smaller model (status == "missing")
  2. Identify candidate entities in the larger model that are not already
     matched (status in the larger side == "missing" or "ambiguous" — i.e.
     available to absorb a new match)
  3. Build a natural-language description for each entity from its JSON
     (name tokens + attribute names + observable types)
  4. Run cross-encoder NLI in both directions:
         score(U, E) = min(entail(U→E), entail(E→U))
     Mutual entailment ≈ semantic equivalence; asymmetric entailment catches
     subsumption (e.g. "ECU" entails "ElectricalSystem") but scores lower.
  5. Greedy 1:1 assignment above a configurable threshold
  6. Write results as a supplementary JSON and update the coverage CSV

Model
-----
  cross-encoder/nli-deberta-v3-base  (~370 MB, GPU-accelerated, 512-token context)
  Significantly better NLI accuracy than MiniLM; fits in 4 GB VRAM.

Usage
-----
    cd ontology_matching
    python mnli_matcher.py                    # all domains
    python mnli_matcher.py --threshold 0.4    # lower threshold
    python mnli_matcher.py --domain Automobile
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from itertools import product
from typing import Any

# ── Local model cache — avoids re-downloading from HuggingFace every run ──── #
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(_MODELS_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", _MODELS_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_MODELS_DIR, "hub"))
# ─────────────────────────────────────────────────────────────────────────── #

REPORTS_DIR  = os.path.join(os.path.dirname(__file__), "outputs", "reports")
INPUTS_DIR   = os.path.join(
    os.path.dirname(__file__), "inputs",
    "CONceptual_ExtractionCategory_Examples",
    "CONceptual_ExtractionCategory_Examples",
)
MNLI_DIR     = os.path.join(os.path.dirname(__file__), "outputs", "mnli")
OUT_CSV      = os.path.join(os.path.dirname(__file__), "outputs", "alignment_summary_mnli.csv")
MODEL_NAME   = "cross-encoder/nli-deberta-v3-base"
MAX_LENGTH   = 512          # DeBERTa-v3-base supports up to 512 tokens
DEFAULT_THRESHOLD = 0.45


# --------------------------------------------------------------------------- #
# Text helpers                                                                 #
# --------------------------------------------------------------------------- #

def _split_camel(name: str) -> str:
    """'BrakeRotor' → 'brake rotor', 'ECU' → 'ECU' (all-caps kept)."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return s.lower().strip()


def _build_description(
    ent_name: str,
    attrs: list[dict],
    ancestors: list[str] | None = None,
) -> str:
    """
    Build a natural-language description of an entity for NLI input.

    Includes ancestor chain context ("part of: X, Y") when provided.

    Example output:
      "A cylinder block (part of: engine) entity with attributes:
       block id (Identifier), bore diameter (Distance), cylinder count (Count)."
    """
    name_text = _split_camel(ent_name)

    ancestor_text = ""
    if ancestors:
        ancestor_text = " (part of: " + ", ".join(_split_camel(a) for a in ancestors) + ")"

    if not attrs:
        return f"A {name_text}{ancestor_text} entity."

    attr_parts = []
    seen_types: set[str] = set()
    for a in attrs:
        attr_label = _split_camel(a.get("name", ""))
        attr_type  = a.get("type", "")
        if attr_type and attr_type not in seen_types:
            seen_types.add(attr_type)
            attr_parts.append(f"{attr_label} ({_split_camel(attr_type)})")
        else:
            attr_parts.append(attr_label)

    attr_text = ", ".join(attr_parts[:12])
    return f"A {name_text}{ancestor_text} entity with attributes: {attr_text}."


def _child_summary(
    ent_name: str,
    emap: dict[str, list[dict]],
    max_children: int = 3,
    max_attrs_per_child: int = 3,
) -> list[str]:
    """
    Summarise the observable attributes of depth-1 entity-type children.

    Only attributes whose type is itself an entity in the same model are
    expanded; their own observable (non-entity) attributes are listed inline.
    This gives abstract composite entities richer descriptions without
    requiring changes to the scoring logic.

    Returns a list of strings like "cylinder block [bore diameter, cylinder count]".
    """
    attrs       = emap.get(ent_name, [])
    entity_names = set(emap.keys())
    summaries: list[str] = []

    for a in attrs:
        child_type = a.get("type", "")
        if child_type not in entity_names:
            continue                        # observable-type attribute, skip
        if len(summaries) >= max_children:
            break

        child_obs = [
            _split_camel(ca.get("name", ""))
            for ca in emap.get(child_type, [])
            if ca.get("type", "") not in entity_names   # only observable attrs
        ][:max_attrs_per_child]

        label = _split_camel(child_type)
        if child_obs:
            summaries.append(f"{label} [{', '.join(child_obs)}]")
        else:
            summaries.append(label)

    return summaries


# --------------------------------------------------------------------------- #
# JSON index                                                                   #
# --------------------------------------------------------------------------- #

def _build_json_index() -> dict[tuple[str, str], tuple[str, dict]]:
    """(domain, modelName) -> (json_path, json_data)"""
    index: dict[tuple[str, str], tuple[str, dict]] = {}
    for jf in sorted(glob.glob(os.path.join(INPUTS_DIR, "**", "*.json"), recursive=True)):
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        domain = os.path.basename(os.path.dirname(jf))
        name   = data.get("modelName", "")
        index[(domain, name)] = (jf, data)
    return index


def _vmodel_names(json_index: dict) -> set[str]:
    """
    Derive the set of versioned-model names (V1/V2/V3) by inspecting input
    file names — no hardcoded domain or version strings.

    V-model files are named like  *_model_v<digit>*.json.
    Variation files are named like *_variation_*.json and are excluded.
    Everything is read from the json_index built by _build_json_index().
    """
    result: set[str] = set()
    for (_, model_name), (jf, _) in json_index.items():
        fname = os.path.basename(jf).lower()
        if re.search(r"_model_v\d", fname):
            result.add(model_name)
    return result


def _entity_map(json_data: dict) -> dict[str, list[dict]]:
    """entity_name -> list of attribute dicts"""
    result: dict[str, list[dict]] = {}
    for ent in json_data.get("entities", []):
        name  = ent.get("entityName") or ent.get("name", "")
        attrs = ent.get("entityAttributes") or ent.get("attributes", [])
        result[name] = attrs
    return result


def _build_parent_map(emap: dict[str, list[dict]]) -> dict[str, list[str]]:
    """
    Build a reverse containment index: entity_name -> list of entity names
    that directly contain it (i.e. have an attribute whose type == entity_name).
    """
    parent_map: dict[str, list[str]] = {name: [] for name in emap}
    for container, attrs in emap.items():
        for attr in attrs:
            child = attr.get("type", "")
            if child in parent_map and container not in parent_map[child]:
                parent_map[child].append(container)
    return parent_map


def _ancestors(
    ent_name: str,
    parent_map: dict[str, list[str]],
    depth: int = 2,
) -> list[str]:
    """
    Return all direct parents up to *depth* levels via BFS (no duplicates).
    Root entities — those with no container of their own — are excluded.
    Including roots adds noise: they appear in every ontology and inflate
    entailment scores against the top-level entity in the matched model.
    """
    seen: set[str] = set()
    result: list[str] = []
    current = list(parent_map.get(ent_name, []))
    for _ in range(depth):
        next_level: list[str] = []
        for p in current:
            if p not in seen:
                # Root entities have an empty parent list — skip them.
                if not parent_map.get(p):
                    continue
                seen.add(p)
                result.append(p)
                next_level.extend(parent_map.get(p, []))
        current = next_level
        if not current:
            break
    return result


# --------------------------------------------------------------------------- #
# NLI scorer                                                                   #
# --------------------------------------------------------------------------- #

class NLIScorer:
    """Wraps a cross-encoder NLI model; caches results for (premise, hypothesis) pairs."""

    ENTAIL_IDX = 0    # label order for MiniLM NLI: entailment=0, neutral=1, contradiction=2

    def __init__(self, model_name: str = MODEL_NAME):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MNLI] Device: {self._device}  "
              f"({'GPU: ' + torch.cuda.get_device_name(0) if self._device.type == 'cuda' else 'CPU'})")
        print(f"[MNLI] Loading model: {model_name}")
        self._tok   = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, low_cpu_mem_usage=True
        )
        self._model.to(self._device)
        self._model.eval()
        self._torch = torch
        self._cache: dict[tuple[str, str], float] = {}
        # Detect label order from model config
        id2label = self._model.config.id2label
        for idx, lbl in id2label.items():
            if "entail" in lbl.lower():
                self.ENTAIL_IDX = int(idx)
                break
        print(f"[MNLI] Entailment label index: {self.ENTAIL_IDX}  labels: {id2label}")

    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        key = (premise, hypothesis)
        if key in self._cache:
            return self._cache[key]
        import torch
        inputs = self._tok(premise, hypothesis,
                           return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        score = float(probs[self.ENTAIL_IDX])
        self._cache[key] = score
        return score

    def mutual_entailment(self, desc_a: str, desc_b: str) -> float:
        """min(A→B, B→A): high only when both directions entail."""
        return min(
            self.entailment_prob(desc_a, desc_b),
            self.entailment_prob(desc_b, desc_a),
        )

    def batch_score(
        self,
        pairs: list[tuple[str, str, str, str]],  # (name_a, desc_a, name_b, desc_b)
    ) -> list[tuple[str, str, float]]:
        """Score all pairs; return (name_a, name_b, score) sorted descending."""
        results = []
        for name_a, desc_a, name_b, desc_b in pairs:
            score = self.mutual_entailment(desc_a, desc_b)
            results.append((name_a, name_b, score))
        results.sort(key=lambda x: x[2], reverse=True)
        return results


# --------------------------------------------------------------------------- #
# Per-pair matching                                                            #
# --------------------------------------------------------------------------- #

def _match_pair(
    report: dict,
    json_index: dict,
    scorer: NLIScorer,
    threshold: float,
    domain: str,
) -> dict:
    """
    Run MNLI matching for one model pair and return a supplementary result dict.
    """
    meta     = report["metadata"]
    name_a   = meta["model_a"]
    name_b   = meta["model_b"]
    sa       = report["summary"]["model_a"]
    sb       = report["summary"]["model_b"]
    total_a  = sa["matched"] + sa["ambiguous"] + sa["missing"]
    total_b  = sb["matched"] + sb["ambiguous"] + sb["missing"]

    # Identify the smaller model
    if total_a <= total_b:
        ref_side, ref_name, other_name = "model_a", name_a, name_b
        ref_total = total_a
    else:
        ref_side, ref_name, other_name = "model_b", name_b, name_a
        ref_total = total_b

    # Missing entities in the smaller model
    missing_ref = [e["name"] for e in report[ref_side]["entities"]
                   if e["status"] == "missing"]
    if not missing_ref:
        return {"pair": f"{ref_name}_vs_{other_name}", "new_matches": [],
                "aml_coverage": round(100*report["summary"][ref_side]["matched"]/ref_total,1)}

    # All entities in the larger model
    other_side = "model_b" if ref_side == "model_a" else "model_a"
    other_side_data = report[other_side]
    larger_entities = [e["name"] for e in other_side_data["entities"]]

    # Build entity descriptions from JSON
    ref_data   = json_index.get((domain, ref_name),   (None, {}))[1]
    other_data = json_index.get((domain, other_name), (None, {}))[1]
    ref_emap   = _entity_map(ref_data)
    other_emap = _entity_map(other_data)
    ref_pmap   = _build_parent_map(ref_emap)
    other_pmap = _build_parent_map(other_emap)

    # Build candidate pairs: missing × all larger entities
    pairs = []
    for u in missing_ref:
        desc_u = _build_description(u, ref_emap.get(u, []),
                                    _ancestors(u, ref_pmap))
        for e in larger_entities:
            desc_e = _build_description(e, other_emap.get(e, []),
                                        _ancestors(e, other_pmap))
            pairs.append((u, desc_u, e, desc_e))

    print(f"  [{domain}] {ref_name[:35]} | {len(missing_ref)} unmapped × "
          f"{len(larger_entities)} larger = {len(pairs)} pairs")

    print(f"    Scoring {len(pairs)} pairs...", flush=True)
    scored = scorer.batch_score(pairs)

    # Greedy 1:1 assignment above threshold
    used_ref:   set[str] = set()
    used_other: set[str] = set()
    new_matches = []
    for name_u, name_e, score in scored:
        if score < threshold:
            break
        if name_u in used_ref or name_e in used_other:
            continue
        new_matches.append({
            "smaller_entity": name_u,
            "larger_entity":  name_e,
            "mnli_score":     round(score, 4),
            "desc_smaller":   _build_description(name_u, ref_emap.get(name_u, []),
                                                 _ancestors(name_u, ref_pmap)),
            "desc_larger":    _build_description(name_e, other_emap.get(name_e, []),
                                                 _ancestors(name_e, other_pmap)),
        })
        used_ref.add(name_u)
        used_other.add(name_e)

    aml_matched  = report["summary"][ref_side]["matched"]
    aml_coverage = round(100 * aml_matched / ref_total, 1) if ref_total else 0
    combined_cov = round(100 * (aml_matched + len(new_matches)) / ref_total, 1) if ref_total else 0

    return {
        "pair":            f"{ref_name}_vs_{other_name}",
        "domain":          domain,
        "smaller_model":   ref_name,
        "larger_model":    other_name,
        "smaller_total":   ref_total,
        "aml_matched":     aml_matched,
        "mnli_recovered":  len(new_matches),
        "aml_coverage":    aml_coverage,
        "combined_coverage": combined_cov,
        "threshold":       threshold,
        "new_matches":     new_matches,
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def run(threshold: float = DEFAULT_THRESHOLD, domain_filter: str | None = None,
        all_models: bool = False) -> None:
    os.makedirs(MNLI_DIR, exist_ok=True)
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

        # Only process pairs where both models are versioned (V1/V2/V3) models,
        # not variation models. The vmodels set is derived from input file names.
        if not all_models and (name_a not in vmodels or name_b not in vmodels):
            continue

        result = _match_pair(report, json_index, scorer, threshold, domain)
        all_results.append(result)

        # Save per-pair JSON
        safe_stem = os.path.splitext(os.path.basename(jf))[0]
        out_path  = os.path.join(MNLI_DIR, domain, f"{safe_stem}_mnli.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Print summary table
    print()
    print(f"{'Domain':<14} {'Smaller model':<44} {'AML':>6} {'MNLI+':>7} {'Combined':>10}")
    print("-" * 86)
    for r in all_results:
        if "aml_coverage" not in r:
            continue
        print(f"{r['domain']:<14} {r['smaller_model'][:43]:<44} "
              f"{r['aml_coverage']:>5}% {r['mnli_recovered']:>6}  "
              f"{r.get('combined_coverage', r['aml_coverage']):>8}%")

    # Write combined CSV
    import csv
    csv_rows = [
        {k: v for k, v in r.items() if k != "new_matches"}
        for r in all_results if "aml_coverage" in r
    ]
    if csv_rows:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nCSV -> {OUT_CSV}")
        print(f"Per-pair JSONs -> {MNLI_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Minimum mutual entailment score (default 0.45)")
    parser.add_argument("--domain", type=str, default=None,
                        help="Run only for one domain (e.g. Automobile)")
    parser.add_argument("--all-models", action="store_true",
                        help="Include non-versioned models (Component Network etc.)")
    args = parser.parse_args()
    run(threshold=args.threshold, domain_filter=args.domain, all_models=args.all_models)
