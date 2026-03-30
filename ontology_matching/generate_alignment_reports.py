"""
generate_alignment_reports.py
------------------------------
Recursively discovers all AML alignment RDF files under AML/<Domain>/,
ensures the reverse direction (B->A) exists (running AML if needed), then
classifies every entity and attribute in both models as:

  matched   - appears in both A->B AND B->A with a mutually consistent mapping
  ambiguous - appears in A->B but B->A does not confirm (absent or contradicts)
  missing   - not matched in either direction

Output layout
-------------
  outputs/
    reports/
      <Domain>/
        <ModelA>_vs_<ModelB>.json

JSON structure
--------------
  {
    "metadata": { domain, model_a, model_b, json_a, json_b,
                  alignment_ab, alignment_ba },
    "summary":  { "model_a": {matched, ambiguous, missing},
                  "model_b": {matched, ambiguous, missing} },
    "model_a":  { "entities": [{name, status, matched_to},...],
                  "attributes": [{entity, name, status, matched_to},...] },
    "model_b":  { ... }
  }

Usage
-----
    cd ontology_matching
    python generate_alignment_reports.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR    = os.path.join(
    BASE_DIR, "inputs",
    "CONceptual_ExtractionCategory_Examples",
    "CONceptual_ExtractionCategory_Examples",
)
AML_OUT_DIR   = os.path.join(BASE_DIR, "AML")
REPORTS_DIR   = os.path.join(BASE_DIR, "outputs", "reports")
OWL_DIR       = os.path.join(BASE_DIR, "outputs")
AML_DIR       = os.path.join(BASE_DIR, "tools", "AML", "AML_v3.2")
AML_JAR       = os.path.join(AML_DIR, "AgreementMakerLight.jar")
JAVA_EXE      = r"C:\Program Files\Microsoft\jdk-11.0.30.7-hotspot\bin\java.exe"
INSTANCE_BASE = "http://example.org/concept_instances/"

_ALIGN_NS = "http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
_RDF_NS   = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _safe(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip())
    return re.sub(r"_+", "_", s).strip("_") or "unknown"


def _owl_path(domain: str, model_name: str) -> str:
    return os.path.join(OWL_DIR, domain, f"{_safe(model_name)}.owl")


def _split_vs(stem: str, valid_safe_names: set[str]) -> tuple[str, str] | None:
    """
    Split a filename stem on '_vs_' and return (safe_a, safe_b) only when
    both halves are recognised safe model names.  Tries every occurrence of
    '_vs_' so names that happen to contain 'vs' are handled correctly.
    """
    idx = 0
    while True:
        pos = stem.find("_vs_", idx)
        if pos == -1:
            return None
        safe_a, safe_b = stem[:pos], stem[pos + 4:]
        if safe_a in valid_safe_names and safe_b in valid_safe_names:
            return safe_a, safe_b
        idx = pos + 1


# --------------------------------------------------------------------------- #
# AML subprocess                                                               #
# --------------------------------------------------------------------------- #

def _run_aml(owl_src: str, owl_tgt: str, out_rdf: str) -> bool:
    """Invoke AML as a Java subprocess. Returns True on success."""
    os.makedirs(os.path.dirname(out_rdf), exist_ok=True)
    cmd = [JAVA_EXE, "-jar", AML_JAR, "-s", owl_src, "-t", owl_tgt, "-o", out_rdf, "-a"]
    r = subprocess.run(cmd, cwd=AML_DIR, capture_output=True, text=True, timeout=300)
    return r.returncode == 0


# --------------------------------------------------------------------------- #
# RDF parsing                                                                  #
# --------------------------------------------------------------------------- #

def _parse_alignment(rdf_path: str) -> dict[str, str]:
    """
    Parse an AML Alignment API RDF/XML file.
    Returns {entity1_iri: entity2_iri} keeping the highest-scoring match per
    entity1 when AML produces multiple candidates for the same source IRI.
    Returns an empty dict if the file is absent or unparseable.
    """
    if not os.path.isfile(rdf_path):
        return {}
    try:
        root = ET.parse(rdf_path).getroot()
    except ET.ParseError:
        return {}

    # best_score[iri1] -> (score, iri2) so we keep only the top-scoring target
    best_score: dict[str, tuple[float, str]] = {}

    # AML may omit the alignment namespace on inner elements (bare tags with
    # default namespace declared on the root).  Try both namespaced and bare.
    def _find(cell, tag: str):
        node = cell.find(f"{{{_ALIGN_NS}}}{tag}")
        if node is None:
            node = cell.find(tag)
        return node

    def _iter_cells(root):
        cells = list(root.iter(f"{{{_ALIGN_NS}}}Cell"))
        if not cells:
            cells = list(root.iter("Cell"))
        return cells

    for cell in _iter_cells(root):
        e1   = _find(cell, "entity1")
        e2   = _find(cell, "entity2")
        meas = _find(cell, "measure")
        if e1 is None or e2 is None:
            continue
        iri1 = e1.get(f"{{{_RDF_NS}}}resource") or e1.get("rdf:resource", "")
        iri2 = e2.get(f"{{{_RDF_NS}}}resource") or e2.get("rdf:resource", "")
        if not iri1 or not iri2:
            continue
        try:
            score = float(meas.text) if meas is not None and meas.text else 0.0
        except ValueError:
            score = 0.0
        prev = best_score.get(iri1)
        if prev is None or score > prev[0]:
            best_score[iri1] = (score, iri2)

    return {iri1: iri2 for iri1, (_, iri2) in best_score.items()}


# --------------------------------------------------------------------------- #
# IRI construction                                                             #
# --------------------------------------------------------------------------- #

def _element_iris(json_data: dict) -> dict[str, dict]:
    """
    Build full_iri -> info for every entity, attribute, and observable type
    in json_data.  Observable types are included because AML matches them as
    OWL individuals and their IRIs appear in the alignment output.

    info keys
    ---------
    kind       : "entity" | "attribute" | "observable"
    name       : original element name
    entity     : parent entity name (attributes only)
    """
    model_name = json_data.get("modelName", "UnknownModel")
    prefix     = _safe(model_name)
    base       = f"{INSTANCE_BASE}{prefix}/"
    result: dict[str, dict] = {}

    # Observable types (Type-A schema: top-level "observables" list)
    for obs_name in json_data.get("observables", []):
        obs_iri = f"{base}{prefix}_obs_{_safe(obs_name)}"
        result[obs_iri] = {"kind": "observable", "name": obs_name}

    # Associations
    for assoc in json_data.get("associations", []):
        assoc_name = assoc.get("associationName") or assoc.get("name", "")
        assoc_iri  = f"{base}{prefix}_assoc_{_safe(assoc_name)}"
        result[assoc_iri] = {"kind": "association", "name": assoc_name}

    for ent in json_data.get("entities", []):
        ent_name = ent.get("entityName") or ent.get("name", "")
        ent_iri  = f"{base}{prefix}_ent_{_safe(ent_name)}"
        result[ent_iri] = {"kind": "entity", "name": ent_name}

        for attr in (ent.get("entityAttributes") or ent.get("attributes", [])):
            attr_name = attr.get("name", "")
            attr_iri  = f"{base}{prefix}_attr_{_safe(ent_name)}_{_safe(attr_name)}"
            result[attr_iri] = {"kind": "attribute", "name": attr_name, "entity": ent_name}

    return result


# --------------------------------------------------------------------------- #
# Classification                                                               #
# --------------------------------------------------------------------------- #

def _classify_side(
    json_data: dict,
    iri_map_other: dict,
    forward: dict[str, str],   # self_iri  -> other_iri  (e.g. A->B)
    backward: dict[str, str],  # other_iri -> self_iri   (e.g. B->A, used to confirm)
) -> dict:
    """
    Classify every entity and attribute in json_data.

    forward  : the alignment going *from* this model *to* the other
    backward : the alignment going *from* the other model *to* this one;
               a match is "confirmed" when backward[forward[iri]] == iri
    """
    model_name = json_data.get("modelName", "UnknownModel")
    prefix     = _safe(model_name)
    base       = f"{INSTANCE_BASE}{prefix}/"

    entities:   list[dict] = []
    attributes: list[dict] = []

    for ent in json_data.get("entities", []):
        ent_name = ent.get("entityName") or ent.get("name", "")
        ent_iri  = f"{base}{prefix}_ent_{_safe(ent_name)}"

        if ent_iri in forward:
            other_iri  = forward[ent_iri]
            confirmed  = backward.get(other_iri) == ent_iri
            status     = "matched" if confirmed else "ambiguous"
            other_info = iri_map_other.get(other_iri, {})
            matched_to = other_info.get("name")
        else:
            status     = "missing"
            matched_to = None

        entities.append({"name": ent_name, "status": status, "matched_to": matched_to})

        for attr in (ent.get("entityAttributes") or ent.get("attributes", [])):
            attr_name = attr.get("name", "")
            attr_iri  = f"{base}{prefix}_attr_{_safe(ent_name)}_{_safe(attr_name)}"

            if attr_iri in forward:
                other_iri   = forward[attr_iri]
                confirmed   = backward.get(other_iri) == attr_iri
                attr_status = "matched" if confirmed else "ambiguous"
                other_info  = iri_map_other.get(other_iri, {})
                if other_info:
                    attr_matched_to = (
                        f"{other_info.get('entity', '')}.{other_info.get('name', '')}"
                    )
                else:
                    attr_matched_to = None
            else:
                attr_status     = "missing"
                attr_matched_to = None

            attributes.append({
                "entity":     ent_name,
                "name":       attr_name,
                "status":     attr_status,
                "matched_to": attr_matched_to,
            })

    return {"entities": entities, "attributes": attributes}


def _build_report(
    json_a: dict,
    json_b: dict,
    ab: dict[str, str],   # A->B alignment
    ba: dict[str, str],   # B->A alignment
    meta: dict,
) -> dict:
    iri_a = _element_iris(json_a)
    iri_b = _element_iris(json_b)

    # Model A perspective: forward=A->B, backward=B->A (confirms A's matches)
    model_a = _classify_side(json_a, iri_b, forward=ab, backward=ba)
    # Model B perspective: forward=B->A, backward=A->B (confirms B's matches)
    model_b = _classify_side(json_b, iri_a, forward=ba, backward=ab)

    def _counts(side: dict) -> dict:
        c: dict[str, int] = {"matched": 0, "ambiguous": 0, "missing": 0}
        for item in side["entities"] + side["attributes"]:
            c[item["status"]] += 1
        return c

    return {
        "metadata": meta,
        "summary":  {"model_a": _counts(model_a), "model_b": _counts(model_b)},
        "model_a":  model_a,
        "model_b":  model_b,
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def _build_json_index() -> dict[tuple[str, str], tuple[str, dict]]:
    """
    Scan INPUTS_DIR recursively and return:
        (domain, safe_model_name) -> (json_path, json_data)
    """
    index: dict[tuple[str, str], tuple[str, dict]] = {}
    for jf in sorted(glob.glob(os.path.join(INPUTS_DIR, "**", "*.json"), recursive=True)):
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        domain     = os.path.basename(os.path.dirname(jf))
        safe       = _safe(data.get("modelName", ""))
        index[(domain, safe)] = (jf, data)
    return index


def run_all() -> None:
    # Index all input JSONs so we can look them up by domain + safe model name
    json_index = _build_json_index()

    # Discover all RDF files produced by AML (any depth under AML_OUT_DIR)
    all_rdfs = sorted(glob.glob(os.path.join(AML_OUT_DIR, "**", "*.rdf"), recursive=True))
    if not all_rdfs:
        print(f"[WARN] No RDF files found under {AML_OUT_DIR}")
        return

    # Group by domain directory
    rdfs_by_domain: dict[str, list[str]] = {}
    for rdf in all_rdfs:
        domain = os.path.basename(os.path.dirname(rdf))
        rdfs_by_domain.setdefault(domain, []).append(rdf)

    total = 0
    processed: set[tuple[str, str, str]] = set()   # (domain, safe_a, safe_b) canonical

    for domain in sorted(rdfs_by_domain):
        print(f"\n=== {domain} ===")
        rep_domain_dir = os.path.join(REPORTS_DIR, domain)
        os.makedirs(rep_domain_dir, exist_ok=True)
        aml_domain_dir = os.path.join(AML_OUT_DIR, domain)

        # Valid safe names for this domain (for _vs_ split validation)
        valid_safe = {k[1] for k in json_index if k[0] == domain}

        for rdf_path in sorted(rdfs_by_domain[domain]):
            stem  = os.path.splitext(os.path.basename(rdf_path))[0]
            split = _split_vs(stem, valid_safe)
            if split is None:
                print(f"  [SKIP] Cannot parse pair from filename: {stem}")
                continue

            safe_a, safe_b = split

            # Canonical key: alphabetical order to deduplicate A_vs_B and B_vs_A
            canon = (domain, *sorted([safe_a, safe_b]))
            if canon in processed:
                continue
            processed.add(canon)

            entry_a = json_index.get((domain, safe_a))
            entry_b = json_index.get((domain, safe_b))
            if entry_a is None or entry_b is None:
                missing = safe_a if entry_a is None else safe_b
                print(f"  [WARN] No JSON found for safe name '{missing}' in domain '{domain}'")
                continue

            json_path_a, data_a = entry_a
            json_path_b, data_b = entry_b
            name_a = data_a.get("modelName", safe_a)
            name_b = data_b.get("modelName", safe_b)

            rdf_ab = os.path.join(aml_domain_dir, f"{safe_a}_vs_{safe_b}.rdf")
            rdf_ba = os.path.join(aml_domain_dir, f"{safe_b}_vs_{safe_a}.rdf")

            # Ensure B->A alignment exists
            if not os.path.isfile(rdf_ba):
                owl_a = _owl_path(domain, name_a)
                owl_b = _owl_path(domain, name_b)
                if (
                    os.path.isfile(AML_JAR)
                    and os.path.isfile(owl_a)
                    and os.path.isfile(owl_b)
                ):
                    label = f"{safe_b[:24]:<24} vs {safe_a[:24]:<24}"
                    print(f"  [B->A] {label} ...", end=" ", flush=True)
                    ok = _run_aml(owl_b, owl_a, rdf_ba)
                    print("ok" if ok else "FAILED")
                else:
                    print(
                        f"  [WARN] Cannot produce B->A for {safe_b[:36]}"
                        " - OWL or JAR missing"
                    )

            ab = _parse_alignment(rdf_ab)
            ba = _parse_alignment(rdf_ba)

            report = _build_report(
                data_a, data_b, ab, ba,
                meta={
                    "domain":       domain,
                    "model_a":      name_a,
                    "model_b":      name_b,
                    "json_a":       os.path.relpath(json_path_a, BASE_DIR).replace("\\", "/"),
                    "json_b":       os.path.relpath(json_path_b, BASE_DIR).replace("\\", "/"),
                    "alignment_ab": (
                        os.path.relpath(rdf_ab, BASE_DIR).replace("\\", "/")
                        if os.path.isfile(rdf_ab) else None
                    ),
                    "alignment_ba": (
                        os.path.relpath(rdf_ba, BASE_DIR).replace("\\", "/")
                        if os.path.isfile(rdf_ba) else None
                    ),
                },
            )

            out_path = os.path.join(rep_domain_dir, f"{safe_a}_vs_{safe_b}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            sa = report["summary"]["model_a"]
            sb = report["summary"]["model_b"]
            label = f"{safe_a[:24]:<24} vs {safe_b[:24]:<24}"
            print(
                f"  {label}  "
                f"A[{sa['matched']}M {sa['ambiguous']}A {sa['missing']}X]  "
                f"B[{sb['matched']}M {sb['ambiguous']}A {sb['missing']}X]"
            )
            total += 1

    print(f"\n=== Done: {total} reports written to {REPORTS_DIR} ===")


if __name__ == "__main__":
    run_all()
