"""
aml_runner.py
-------------
Python wrapper for the actual AgreementMakerLight (AML) v3.2 JAR.

AML is a Java-based OAEI ontology matching system.
Source: https://github.com/AgreementMakerLight/AML-Project
JAR:    enriched_ontology_matching/tools/AML/AML_v3.2/AgreementMakerLight.jar

IMPORTANT: AML must be invoked from its own directory because it reads
           store/config.ini and store/StopList.txt at runtime using
           relative paths.

Workflow:
  1. Reuse json_to_tbox_owl() from logmap_runner to build OWL TBox files.
  2. Run AgreementMakerLight.jar -s <owl_a> -t <owl_b> -o <output.rdf> -a
     in automatic match mode, with cwd set to the AML folder.
  3. Parse the OAEI Alignment API RDF output.
  4. Return sorted mapping dicts identical in shape to LogMapRunner output.

Usage
-----
    from aml_runner import AMLRunner

    runner = AMLRunner()
    mappings = runner.match_test_file("test_auto_v1_v2.json")
    for m in mappings:
        print(m["entity_a"], "<->", m["entity_b"], m["confidence"])

CLI
---
    venv/Scripts/python.exe enriched_ontology_matching/aml_runner.py \\
        enriched_ontology_matching/test_auto_v1_v2.json
"""

from __future__ import annotations

import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DIR    = Path(__file__).parent
AML_DIR = _DIR / "tools" / "AML_v3.2"
AML_JAR = AML_DIR / "AgreementMakerLight.jar"

# ---------------------------------------------------------------------------
# Reuse OWL TBox generation from logmap_runner
# ---------------------------------------------------------------------------
from logmap_runner import json_to_tbox_owl, _entities_from_json, _safe_local

# ---------------------------------------------------------------------------
# AML invocation
# ---------------------------------------------------------------------------

def run_aml(
    owl_a: Path,
    owl_b: Path,
    output_rdf: Path,
    timeout: int = 300,
) -> Path:
    """
    Invoke AML in automatic match mode (-a).

    AML is run with cwd=AML_DIR so it can find store/config.ini and
    store/StopList.txt via relative paths.

    Parameters
    ----------
    owl_a, owl_b : absolute paths to the two OWL TBox files
    output_rdf   : where AML should write the OAEI alignment RDF
    timeout      : subprocess timeout in seconds (AML can be slow on larger ontologies)

    Returns
    -------
    Path to the output alignment RDF (same as output_rdf).
    """
    if not AML_JAR.exists():
        raise FileNotFoundError(
            f"AML JAR not found at {AML_JAR}\n"
            "Expected: enriched_ontology_matching/tools/AML/AML_v3.2/AgreementMakerLight.jar"
        )

    output_rdf.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "java", "-jar", str(AML_JAR),
        "-s", str(owl_a.resolve()),
        "-t", str(owl_b.resolve()),
        "-o", str(output_rdf.resolve()),
        "-a",   # automatic match mode
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(AML_DIR),   # REQUIRED: AML reads store/ using relative paths
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"AML exited with code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout[-1000:]}\n"
            f"STDERR:\n{result.stderr[-1000:]}"
        )

    if not output_rdf.exists() or output_rdf.stat().st_size == 0:
        raise FileNotFoundError(
            f"AML ran but produced no output at {output_rdf}.\n"
            f"STDOUT:\n{result.stdout[-800:]}"
        )

    return output_rdf


# ---------------------------------------------------------------------------
# OAEI Alignment API parser
# ---------------------------------------------------------------------------

# AML uses the OAEI namespace WITHOUT a trailing '#'
_ALIGN_NS = "http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
_RDF_NS   = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def _local_name(iri: str) -> str:
    """Extract the local fragment from an IRI (after # or last /)."""
    if "#" in iri:
        return iri.split("#")[-1]
    return iri.rstrip("/").split("/")[-1]


def parse_alignment(rdf_path: Path) -> list[dict]:
    """
    Parse an OAEI Alignment API RDF file produced by AML and return
    mapping dicts sorted by descending confidence.

    Each dict has keys: entity_a, entity_b, iri_a, iri_b, confidence, relation.
    """
    tree = ET.parse(rdf_path)
    root = tree.getroot()

    mappings: list[dict] = []
    for cell in root.iter(f"{{{_ALIGN_NS}}}Cell"):
        e1      = cell.find(f"{{{_ALIGN_NS}}}entity1")
        e2      = cell.find(f"{{{_ALIGN_NS}}}entity2")
        measure = cell.find(f"{{{_ALIGN_NS}}}measure")
        rel_el  = cell.find(f"{{{_ALIGN_NS}}}relation")

        if e1 is None or e2 is None:
            continue

        iri_a = e1.get(f"{{{_RDF_NS}}}resource", "")
        iri_b = e2.get(f"{{{_RDF_NS}}}resource", "")
        conf  = float(measure.text.strip()) if (measure is not None and measure.text) else 0.0
        rel   = rel_el.text.strip() if (rel_el is not None and rel_el.text) else "="

        mappings.append({
            "entity_a":   _local_name(iri_a),
            "entity_b":   _local_name(iri_b),
            "iri_a":      iri_a,
            "iri_b":      iri_b,
            "confidence": conf,
            "relation":   rel,
        })

    return sorted(mappings, key=lambda m: m["confidence"], reverse=True)


# ---------------------------------------------------------------------------
# High-level AMLRunner API
# ---------------------------------------------------------------------------

class AMLRunner:
    """
    Thin Python wrapper around the AML v3.2 JAR.

    Parameters
    ----------
    output_base : base directory for generated OWL files and AML output.
                  Defaults to enriched_ontology_matching/outputs/aml/
    """

    def __init__(self, output_base: Path | None = None):
        self.output_base = output_base or (_DIR / "outputs" / "aml")

    def match_jsons(
        self,
        json_a: dict,
        json_b: dict,
        name_a: str = "ontology_a",
        name_b: str = "ontology_b",
    ) -> list[dict]:
        """
        Match two conceptual-model JSON dicts using AML automatic mode.

        Returns list of mapping dicts sorted by descending confidence.
        """
        run_dir = self.output_base / f"{_safe_local(name_a)}_vs_{_safe_local(name_b)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        owl_a      = run_dir / "ontology_a.owl"
        owl_b      = run_dir / "ontology_b.owl"
        output_rdf = run_dir / "aml_alignment.rdf"

        json_to_tbox_owl(json_a, owl_a)
        json_to_tbox_owl(json_b, owl_b)

        print(f"  [AML] Ontology A: {len(_entities_from_json(json_a))} classes -> {owl_a.name}")
        print(f"  [AML] Ontology B: {len(_entities_from_json(json_b))} classes -> {owl_b.name}")
        print(f"  [AML] Running JAR ...")

        run_aml(owl_a, owl_b, output_rdf)
        mappings = parse_alignment(output_rdf)

        print(f"  [AML] Done -- {len(mappings)} class mappings found.")
        return mappings

    def match_test_file(self, test_json_path: str | Path) -> list[dict]:
        """
        Run AML on a test JSON containing "json_a" and "json_b" keys
        (as produced by root_comparator.py).
        """
        path = Path(test_json_path)
        data = json.loads(path.read_text(encoding="utf-8"))

        json_a = data.get("json_a", data)
        json_b = data.get("json_b", {})

        name_a = _safe_local(json_a.get("modelName", path.stem + "_a"))
        name_b = _safe_local(json_b.get("modelName", path.stem + "_b"))

        return self.match_jsons(json_a, json_b, name_a=name_a, name_b=name_b)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Run AML v3.2 on a test JSON file (json_a + json_b structure)."
    )
    parser.add_argument(
        "test_json",
        help="Path to test JSON containing 'json_a' and 'json_b' keys.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum confidence score to display (default: show all).",
    )
    args = parser.parse_args()

    runner = AMLRunner()
    print(f"\nRunning AML on: {args.test_json}\n")

    try:
        mappings = runner.match_test_file(args.test_json)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    shown = [m for m in mappings if m["confidence"] >= args.threshold]
    print(f"\n{'='*85}")
    print(f"AML results -- {len(shown)} mappings (threshold={args.threshold:.2f})")
    print(f"{'='*85}")
    print(f"{'Entity A':<38} {'Entity B':<38} {'Conf':>5}  Rel")
    print(f"{'-'*85}")
    for m in shown:
        print(f"{m['entity_a']:<38} {m['entity_b']:<38} {m['confidence']:>5.3f}  {m['relation']}")
    print(f"{'='*85}")
