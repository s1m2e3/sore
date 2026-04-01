"""
run_crosstype_comparisons.py
-----------------------------
Runs S1 (AML) and generates report JSONs for all cross-type pairs that the
main pipeline skips — specifically V-model vs Network/Variation model pairs.

For every domain, compares every V-model against every non-V-model (and vice
versa), producing the same report format as generate_alignment_reports.py so
that structural_matcher.py (S2) can consume them unchanged.

Usage
-----
    cd ontology_matching
    .venv/Scripts/python.exe run_crosstype_comparisons.py
    .venv/Scripts/python.exe run_crosstype_comparisons.py --domain Automobile
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

from aml_matcher import AMLMatcher

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(
    BASE_DIR, "inputs",
    "CONceptual_ExtractionCategory_Examples",
    "CONceptual_ExtractionCategory_Examples",
)
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
AML_DIR     = os.path.join(BASE_DIR, "AML")

matcher = AMLMatcher()


def _safe(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip())
    return re.sub(r"_+", "_", s).strip("_") or "unknown"


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _model_name(data: dict) -> str:
    return data.get("modelName") or data.get("topicDomain") or "Unknown"


def _is_v_model(path: str) -> bool:
    """True if the JSON filename contains a version marker like _v1, _v2, _v3."""
    base = os.path.basename(path).lower()
    return bool(re.search(r"_v[123][_.]", base))


def _build_report(
    domain: str,
    json_path_a: str,
    json_path_b: str,
    data_a: dict,
    data_b: dict,
    cells_ab: list[dict],   # [{entity_a, entity_b, score}]
    cells_ba: list[dict],
) -> dict:
    """
    Produce a report JSON in the same format as generate_alignment_reports.py.
    """
    model_name_a = _model_name(data_a)
    model_name_b = _model_name(data_b)

    # Build match sets (symmetric confirmed matches only)
    ab_map = {c["entity_a"]: c["entity_b"] for c in cells_ab}
    ba_map = {c["entity_a"]: c["entity_b"] for c in cells_ba}

    def classify_side(data: dict, forward: dict, backward: dict):
        """
        forward  = {entity_this -> entity_other}  (A->B direction)
        backward = {entity_other -> entity_this}  (B->A direction, inverted)
        """
        entities_out = []
        for ent in data.get("entities", []):
            name = ent.get("entityName") or ent.get("name", "")
            if name in forward:
                target = forward[name]
                # Confirmed if B->A also maps target back to name
                if backward.get(target) == name:
                    status = "matched"
                else:
                    status = "ambiguous"
                entities_out.append({"name": name, "status": status, "matched_to": target})
            else:
                entities_out.append({"name": name, "status": "missing", "matched_to": None})
        return entities_out

    # Invert ba_map for confirming A-side
    inv_ba = {v: k for k, v in ba_map.items()}
    # Invert ab_map for confirming B-side
    inv_ab = {v: k for k, v in ab_map.items()}

    ents_a = classify_side(data_a, ab_map, inv_ba)
    ents_b = classify_side(data_b, ba_map, inv_ab)

    def counts(ents):
        matched   = sum(1 for e in ents if e["status"] == "matched")
        ambiguous = sum(1 for e in ents if e["status"] == "ambiguous")
        missing   = sum(1 for e in ents if e["status"] == "missing")
        return {"matched": matched, "ambiguous": ambiguous, "missing": missing}

    rel_a = os.path.relpath(json_path_a, BASE_DIR).replace("\\", "/")
    rel_b = os.path.relpath(json_path_b, BASE_DIR).replace("\\", "/")

    stem_ab = f"{_safe(model_name_a)}_vs_{_safe(model_name_b)}"
    stem_ba = f"{_safe(model_name_b)}_vs_{_safe(model_name_a)}"
    rdf_ab  = os.path.join("AML", domain, f"{stem_ab}.rdf")
    rdf_ba  = os.path.join("AML", domain, f"{stem_ba}.rdf")

    return {
        "metadata": {
            "domain":      domain,
            "model_a":     model_name_a,
            "model_b":     model_name_b,
            "json_a":      rel_a,
            "json_b":      rel_b,
            "alignment_ab": rdf_ab,
            "alignment_ba": rdf_ba,
        },
        "summary": {
            "model_a": counts(ents_a),
            "model_b": counts(ents_b),
        },
        "model_a": {"entities": ents_a},
        "model_b": {"entities": ents_b},
    }


def run_domain(domain: str) -> int:
    domain_dir = os.path.join(INPUTS_DIR, domain)
    all_jsons  = sorted(glob.glob(os.path.join(domain_dir, "*.json")))
    if not all_jsons:
        return 0

    v_jsons   = [p for p in all_jsons if _is_v_model(p)]
    net_jsons = [p for p in all_jsons if not _is_v_model(p)]

    if not v_jsons or not net_jsons:
        return 0

    out_dir = os.path.join(REPORTS_DIR, domain)
    os.makedirs(out_dir, exist_ok=True)

    written = 0
    for vp in v_jsons:
        for np_ in net_jsons:
            data_v = _load(vp)
            data_n = _load(np_)
            name_v = _model_name(data_v)
            name_n = _model_name(data_n)

            stem = f"{_safe(name_v)}_vs_{_safe(name_n)}"
            out_path = os.path.join(out_dir, f"{stem}.json")

            # Skip if already exists
            if os.path.exists(out_path):
                print(f"  [skip] {stem[:70]}")
                continue

            # Run AML both directions
            align_vn = matcher.match(data_v, data_n)
            align_nv = matcher.match(data_n, data_v)

            cells_vn = [{"entity_a": c.entity_a, "entity_b": c.entity_b, "score": c.score}
                        for c in align_vn.cells]
            cells_nv = [{"entity_a": c.entity_a, "entity_b": c.entity_b, "score": c.score}
                        for c in align_nv.cells]

            report = _build_report(domain, vp, np_, data_v, data_n, cells_vn, cells_nv)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            ma_counts = report["summary"]["model_a"]
            print(f"  {stem[:70]}  matched={ma_counts['matched']}  missing={ma_counts['missing']}")
            written += 1

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", help="Limit to one domain")
    args = parser.parse_args()

    domains = (
        [args.domain] if args.domain
        else sorted(d for d in os.listdir(INPUTS_DIR)
                    if os.path.isdir(os.path.join(INPUTS_DIR, d)))
    )

    total = 0
    for domain in domains:
        print(f"\n=== {domain} ===")
        total += run_domain(domain)

    print(f"\n=== Done: {total} new cross-type reports written ===")


if __name__ == "__main__":
    main()
