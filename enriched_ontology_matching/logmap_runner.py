"""
logmap_runner.py
----------------
Python wrapper for the actual LogMap 4.0 ontology matching JAR.

LogMap 4.0 (July 2021) — Java-based, OAEI award-winning matcher.
Source: https://github.com/ernestojimenezruiz/logmap-matcher
JAR:    enriched_ontology_matching/tools/logmap/logmap-matcher-4.0.jar

Workflow:
  1. Convert each conceptual-model JSON to a minimal OWL TBox
     (entities and associations become owl:Class with rdfs:label).
  2. Run logmap-matcher-4.0.jar MATCHER via subprocess.
  3. Parse the OAEI Alignment API output (logmap_mappings.rdf).
  4. Return a sorted list of mapping dicts.

Usage
-----
    from logmap_runner import LogMapRunner

    runner = LogMapRunner()
    mappings = runner.match_test_file("test_auto_v1_v2.json")
    for m in mappings:
        print(m["entity_a"], "<->", m["entity_b"], m["confidence"])

CLI
---
    # From repo root, using the project venv:
    venv/Scripts/python.exe enriched_ontology_matching/logmap_runner.py \\
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
_DIR       = Path(__file__).parent
LOGMAP_JAR = _DIR / "tools" / "logmap" / "logmap-matcher-4.0.jar"


# ---------------------------------------------------------------------------
# OWL TBox generation
# ---------------------------------------------------------------------------

def _safe_local(name: str) -> str:
    """Sanitise a string for use as an OWL IRI local fragment."""
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip())
    return re.sub(r"_+", "_", s).strip("_") or "Unknown"


def _entities_from_json(json_data: dict) -> list[str]:
    """
    Extract all entity and association names from a conceptual-model JSON.
    Supports both Type-A (entityName) and Type-B (name) schemas.
    """
    names: list[str] = []
    for ent in json_data.get("entities", []):
        name = ent.get("entityName") or ent.get("name", "")
        if name:
            names.append(name)
    for assoc in json_data.get("associations", []):
        name = assoc.get("associationName") or assoc.get("name", "")
        if name:
            names.append(name)
    return names


def json_to_tbox_owl(json_data: dict, output_path: Path) -> None:
    """
    Write a minimal OWL TBox file where every entity / association in
    json_data becomes an owl:Class with an rdfs:label.

    The ontology IRI is set to the file:/// URI of output_path so that
    LogMap's OWL API can resolve the file when saving alignment output.
    Class IRIs are anchored to that same file URI.

    Parameters
    ----------
    json_data   : conceptual-model JSON dict (Type-A or Type-B schema)
    output_path : where to write the .owl file (must be absolute)
    """
    # Use the file's own URI as ontology IRI — required so LogMap's OWL API
    # can save the alignment output without "URI is not hierarchical" errors.
    iri_base = output_path.resolve().as_uri()  # e.g. file:///C:/...ontology_a.owl

    lines = [
        '<?xml version="1.0"?>',
        '<rdf:RDF',
        '  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
        '  xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"',
        '  xmlns:owl="http://www.w3.org/2002/07/owl#"',
        f'  xml:base="{iri_base}">',
        f'  <owl:Ontology rdf:about="{iri_base}"/>',
    ]
    for name in _entities_from_json(json_data):
        local = _safe_local(name)
        # Escape XML special chars in the label
        label = name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines += [
            f'  <owl:Class rdf:about="{iri_base}#{local}">',
            f'    <rdfs:label xml:lang="en">{label}</rdfs:label>',
            f'  </owl:Class>',
        ]
    lines.append('</rdf:RDF>')
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# LogMap invocation
# ---------------------------------------------------------------------------

def _file_iri(path: Path) -> str:
    """
    Convert an absolute path to a file:/// IRI suitable for LogMap.
    Uses Python's Path.as_uri() which handles Windows drive letters correctly.
    """
    return path.resolve().as_uri()


def run_logmap(
    owl_a: Path,
    owl_b: Path,
    output_dir: Path,
    classify: bool = False,
    timeout: int = 180,
) -> Path:
    """
    Invoke the LogMap 4.0 JAR in MATCHER mode.

    Parameters
    ----------
    owl_a, owl_b : paths to the two OWL TBox files
    output_dir   : directory where LogMap writes output files
    classify     : run OWL reasoner on merged ontology (slow; default False)
    timeout      : subprocess timeout in seconds

    Returns
    -------
    Path to the primary class-mapping output file (logmap_mappings.rdf).

    Raises
    ------
    FileNotFoundError : if JAR is missing or no .rdf output is produced
    RuntimeError      : if LogMap exits with a non-zero return code
    """
    if not LOGMAP_JAR.exists():
        raise FileNotFoundError(
            f"LogMap JAR not found at {LOGMAP_JAR}\n"
            "Run the installation script or re-download from:\n"
            "https://github.com/ernestojimenezruiz/logmap-matcher/releases"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # LogMap expects a trailing slash on the output directory path
    out_str = str(output_dir.resolve()).replace("\\", "/") + "/"

    cmd = [
        "java", "-jar", str(LOGMAP_JAR),
        "MATCHER",
        _file_iri(owl_a),
        _file_iri(owl_b),
        out_str,
        str(classify).lower(),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    # LogMap writes a lot to stdout; only raise on non-zero exit
    if result.returncode != 0:
        raise RuntimeError(
            f"LogMap exited with code {result.returncode}.\n"
            f"STDOUT (last 1000 chars):\n{result.stdout[-1000:]}\n"
            f"STDERR (last 1000 chars):\n{result.stderr[-1000:]}"
        )

    # Prefer the TSV file — it is plain text and never truncated, unlike the
    # RDF output which LogMap sometimes leaves with an incomplete final tag.
    tsv_file = output_dir / "logmap2_mappings.tsv"
    if not tsv_file.exists() or tsv_file.stat().st_size == 0:
        # Fall back to the RDF file
        tsv_file = output_dir / "logmap2_mappings.rdf"
        if not tsv_file.exists():
            rdf_candidates = sorted(output_dir.glob("*.rdf"))
            if rdf_candidates:
                tsv_file = rdf_candidates[0]
            else:
                raise FileNotFoundError(
                    f"LogMap ran but produced no output in {output_dir}.\n"
                    f"LogMap STDOUT:\n{result.stdout[-800:]}"
                )

    return tsv_file


# ---------------------------------------------------------------------------
# Alignment parsers
# ---------------------------------------------------------------------------

_ALIGN_NS = "http://knowledgeweb.semanticweb.org/heterogeneity/alignment#"
_RDF_NS   = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

# LogMap wraps ontology IRIs in Java's Optional.toString() in its TSV/module
# output, producing "Optional.of(<iri>)". Strip that wrapper here.
_OPTIONAL_RE = re.compile(r"^Optional\.of\((.+)\)$")


def _local_name(iri: str) -> str:
    """
    Extract the local fragment from an IRI (after # or last /).
    Also strips any LogMap Java Optional.of(...) wrapper.
    """
    # Unwrap Java Optional.of(...) if present
    m = _OPTIONAL_RE.match(iri)
    if m:
        iri = m.group(1)
    if "#" in iri:
        return iri.split("#")[-1]
    return iri.rstrip("/").split("/")[-1]


def _parse_tsv(tsv_path: Path) -> list[dict]:
    """
    Parse a LogMap TSV mapping file.
    Format: IRI_A \\t IRI_B \\t confidence  (one mapping per line)
    """
    mappings: list[dict] = []
    for line in tsv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        iri_a, iri_b, conf_str = parts[0], parts[1], parts[2]
        try:
            conf = float(conf_str)
        except ValueError:
            continue
        mappings.append({
            "entity_a":   _local_name(iri_a),
            "entity_b":   _local_name(iri_b),
            "iri_a":      iri_a,
            "iri_b":      iri_b,
            "confidence": conf,
            "relation":   "=",
        })
    return mappings


def _parse_rdf(rdf_path: Path) -> list[dict]:
    """
    Parse an OAEI Alignment API RDF file.
    Falls back gracefully if the XML is truncated.
    """
    content = rdf_path.read_text(encoding="utf-8", errors="replace")
    # Attempt to close a truncated XML file by appending closing tags
    if not content.rstrip().endswith("</rdf:RDF>"):
        content = content.rstrip()
        # Close any open Cell and map tags, then close the root
        if "<Cell" in content and "</Cell>" not in content.split("<Cell")[-1]:
            content += "</Cell></map></Alignment></rdf:RDF>"
        elif "<map>" in content and "</map>" not in content.split("<map>")[-1]:
            content += "</map></Alignment></rdf:RDF>"
        else:
            content += "</Alignment></rdf:RDF>"

    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []

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

    return mappings


def parse_alignment(path: Path) -> list[dict]:
    """
    Parse a LogMap output file (TSV preferred; RDF fallback).
    Returns mapping dicts sorted by descending confidence.

    Each dict has keys: entity_a, entity_b, iri_a, iri_b, confidence, relation.
    """
    if path.suffix == ".tsv":
        mappings = _parse_tsv(path)
    else:
        mappings = _parse_rdf(path)
    return sorted(mappings, key=lambda m: m["confidence"], reverse=True)


def list_output_files(output_dir: Path) -> dict[str, Path]:
    """
    Return a dict of all LogMap output files (TSV + RDF) in output_dir.
    """
    return {f.name: f for f in sorted(output_dir.glob("logmap*.*"))}


# ---------------------------------------------------------------------------
# High-level LogMapRunner API
# ---------------------------------------------------------------------------

class LogMapRunner:
    """
    Thin Python wrapper around the LogMap 4.0 JAR.

    Converts conceptual-model JSON dicts to minimal OWL TBox files,
    runs LogMap, and returns the parsed OAEI alignment mappings.

    Parameters
    ----------
    output_base : base directory for generated OWL files and LogMap output.
                  Defaults to enriched_ontology_matching/outputs/logmap/
    """

    def __init__(self, output_base: Path | None = None):
        self.output_base = output_base or (_DIR / "outputs" / "logmap")

    def match_jsons(
        self,
        json_a: dict,
        json_b: dict,
        name_a: str = "ontology_a",
        name_b: str = "ontology_b",
        classify: bool = False,
    ) -> list[dict]:
        """
        Match two conceptual-model JSON dicts using LogMap.

        Parameters
        ----------
        json_a, json_b : conceptual-model JSON dicts
        name_a, name_b : human-readable names used to build IRIs and output paths
        classify        : pass True to enable OWL reasoning (slower)

        Returns
        -------
        List of mapping dicts sorted by descending confidence.
        """
        run_dir = self.output_base / f"{_safe_local(name_a)}_vs_{_safe_local(name_b)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        owl_a = run_dir / "ontology_a.owl"
        owl_b = run_dir / "ontology_b.owl"

        json_to_tbox_owl(json_a, owl_a)
        json_to_tbox_owl(json_b, owl_b)

        print(f"  [LogMap] Ontology A: {len(_entities_from_json(json_a))} classes -> {owl_a.name}")
        print(f"  [LogMap] Ontology B: {len(_entities_from_json(json_b))} classes -> {owl_b.name}")
        print(f"  [LogMap] Running JAR ...")

        alignment_rdf = run_logmap(owl_a, owl_b, run_dir, classify=classify)
        mappings = parse_alignment(alignment_rdf)

        print(f"  [LogMap] Done -- {len(mappings)} class mappings found.")
        all_rdf = list_output_files(run_dir)
        if len(all_rdf) > 1:
            print(f"  [LogMap] All output files: {', '.join(all_rdf)}")

        return mappings

    def match_test_file(
        self,
        test_json_path: str | Path,
        classify: bool = False,
    ) -> list[dict]:
        """
        Run LogMap on a test JSON file that contains "json_a" and "json_b" keys
        (as produced by root_comparator.py).

        Parameters
        ----------
        test_json_path : path to the test JSON file
        classify        : pass True to enable OWL reasoning (slower)
        """
        path = Path(test_json_path)
        data = json.loads(path.read_text(encoding="utf-8"))

        json_a = data.get("json_a", data)
        json_b = data.get("json_b", {})

        name_a = _safe_local(json_a.get("modelName", path.stem + "_a"))
        name_b = _safe_local(json_b.get("modelName", path.stem + "_b"))

        return self.match_jsons(json_a, json_b, name_a=name_a, name_b=name_b, classify=classify)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Run LogMap 4.0 on a test JSON file (json_a + json_b structure)."
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
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Enable OWL classification with mappings (slower, more precise).",
    )
    args = parser.parse_args()

    runner = LogMapRunner()
    print(f"\nRunning LogMap on: {args.test_json}\n")

    try:
        mappings = runner.match_test_file(args.test_json, classify=args.classify)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    shown = [m for m in mappings if m["confidence"] >= args.threshold]
    print(f"\n{'='*85}")
    print(f"LogMap results -- {len(shown)} mappings (threshold={args.threshold:.2f})")
    print(f"{'='*85}")
    print(f"{'Entity A':<38} {'Entity B':<38} {'Conf':>5}  Rel")
    print(f"{'-'*85}")
    for m in shown:
        print(f"{m['entity_a']:<38} {m['entity_b']:<38} {m['confidence']:>5.3f}  {m['relation']}")
    print(f"{'='*85}")
