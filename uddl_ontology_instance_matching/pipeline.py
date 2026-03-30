"""
pipeline.py
-----------
End-to-end UDDL message comparison pipeline.

Takes two UDDL-layered JSON files as input (e.g. the human reference and the
LLM-generated version derived from the same ICD) and outputs alignment metrics.

Usage
-----
  python pipeline.py \\
      --msg-a  inputs/uddl_layered_aircraft_message_1.json \\
      --msg-b  inputs/uddl_layered_aircraft_message_2.json \\
      [--ground-truth  inputs/ground_truth_aircraft.json] \\
      [--no-mnli] \\
      [--out-dir  outputs/] \\
      [--java     "C:/Program Files/Microsoft/jdk-11.0.30.7-hotspot/bin/java"] \\
      [--limes    tools/limes.jar]

The pipeline performs:
  Step 1 – Build/load UDDL metamodel OWL             (base_ontology.py)
  Step 2 – Parse each JSON -> OWL instance graph     (json_to_uddl_owl.py)
  Step 3 – Convert OWL to N-Triples                  (rdflib)
  Step 4 – Generate LIMES XML config dynamically     (no hardcoded paths)
  Step 5 – Run LIMES                                 (Java 11+)
  Step 6 – Run MNLI semantic matcher + evaluate      (evaluate_matching.py)
  Step 7 – Write CSV results
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

from rdflib import Graph


# ── Step 1 helpers ────────────────────────────────────────────────────────────

def ensure_metamodel(metamodel_path: str) -> None:
    if not os.path.exists(metamodel_path):
        print("[Step 1] Building UDDL metamodel OWL ...")
        from base_ontology import build_uddl_metamodel
        build_uddl_metamodel()
    else:
        print(f"[Step 1] Metamodel already exists: {metamodel_path}")


# ── Step 2 helpers ────────────────────────────────────────────────────────────

def json_to_owl(json_path: str, out_owl: str, metamodel_path: str) -> None:
    print(f"[Step 2] Parsing {json_path} -> {out_owl}")
    from json_to_uddl_owl import UDDLJsonToOWL
    converter = UDDLJsonToOWL(metamodel_path=metamodel_path)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    converter.parse(data, output_path=out_owl)


# ── Step 3 helpers ────────────────────────────────────────────────────────────

def owl_to_nt(owl_path: str, nt_path: str) -> None:
    print(f"[Step 3] Converting {owl_path} -> {nt_path}")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = Graph()
        g.parse(owl_path, format="xml")
        g.serialize(destination=nt_path, format="nt")


def owl_to_ttl(owl_path: str, ttl_path: str) -> None:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = Graph()
        g.parse(owl_path, format="xml")
        g.serialize(destination=ttl_path, format="turtle")


# ── Step 4: LIMES config generator ───────────────────────────────────────────

def generate_limes_config(
    src_nt: str,
    tgt_nt: str,
    restriction_class: str,
    property_path: str,
    accept_out: str,
    review_out: str,
    accept_threshold: float = 0.5,
    review_threshold: float = 0.2,
    metric: str = "trigrams",
    metamodel_ns: str = "http://example.org/uddl_metamodel.owl#",
) -> str:
    """
    Generate a LIMES XML configuration string.
    All paths are passed as parameters -- nothing is hardcoded.

    Parameters
    ----------
    src_nt            : absolute path to source N-Triples file
    tgt_nt            : absolute path to target N-Triples file
    restriction_class : local name of OWL class to match (e.g. ConceptualObservable)
    property_path     : property local name used for comparison (e.g. hasName)
    accept_out        : path for high-confidence alignment output
    review_out        : path for candidate alignment output
    metric            : LIMES metric function (trigrams, jaccard, cosine, ...)
    """
    # Normalise Windows paths to forward slashes for LIMES
    def fwd(p: str) -> str:
        return p.replace("\\", "/")

    src_nt    = fwd(os.path.abspath(src_nt))
    tgt_nt    = fwd(os.path.abspath(tgt_nt))
    accept_out = fwd(os.path.abspath(accept_out))
    review_out = fwd(os.path.abspath(review_out))

    prop_label = f"uddl:{property_path}"
    prop_expr  = f"uddl:{property_path} AS nolang->lowercase"
    metric_expr = (
        f"{metric}(x.{prop_label}, y.{prop_label})"
    )

    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE LIMES SYSTEM "limes.dtd">
        <LIMES>

          <PREFIX>
            <NAMESPACE>http://www.w3.org/1999/02/22-rdf-syntax-ns#</NAMESPACE>
            <LABEL>rdf</LABEL>
          </PREFIX>
          <PREFIX>
            <NAMESPACE>http://www.w3.org/2002/07/owl#</NAMESPACE>
            <LABEL>owl</LABEL>
          </PREFIX>
          <PREFIX>
            <NAMESPACE>{metamodel_ns}</NAMESPACE>
            <LABEL>uddl</LABEL>
          </PREFIX>

          <SOURCE>
            <ID>src</ID>
            <ENDPOINT>{src_nt}</ENDPOINT>
            <VAR>?x</VAR>
            <PAGESIZE>-1</PAGESIZE>
            <RESTRICTION>?x rdf:type uddl:{restriction_class}</RESTRICTION>
            <PROPERTY>{prop_expr}</PROPERTY>
            <TYPE>NT</TYPE>
          </SOURCE>

          <TARGET>
            <ID>tgt</ID>
            <ENDPOINT>{tgt_nt}</ENDPOINT>
            <VAR>?y</VAR>
            <PAGESIZE>-1</PAGESIZE>
            <RESTRICTION>?y rdf:type uddl:{restriction_class}</RESTRICTION>
            <PROPERTY>{prop_expr}</PROPERTY>
            <TYPE>NT</TYPE>
          </TARGET>

          <METRIC>{metric_expr}</METRIC>

          <ACCEPTANCE>
            <THRESHOLD>{accept_threshold}</THRESHOLD>
            <FILE>{accept_out}</FILE>
            <RELATION>owl:sameAs</RELATION>
          </ACCEPTANCE>

          <REVIEW>
            <THRESHOLD>{review_threshold}</THRESHOLD>
            <FILE>{review_out}</FILE>
            <RELATION>owl:sameAs</RELATION>
          </REVIEW>

          <GRANULARITY>2</GRANULARITY>
          <OUTPUT>NT</OUTPUT>

        </LIMES>
    """)


# ── Step 5: Run LIMES ─────────────────────────────────────────────────────────

JAVA_CANDIDATES = [
    "C:/Program Files/Microsoft/jdk-11.0.30.7-hotspot/bin/java",
    "C:/Program Files/Eclipse Adoptium/jdk-11.0.13.8-hotspot/bin/java",
    "java",   # system PATH fallback
]

LIMES_CANDIDATES = [
    "tools/limes.jar",
    "limes/limes.jar",
]


def find_java() -> Optional[str]:
    for candidate in JAVA_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return str(path)
    # fallback: check PATH
    try:
        result = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, timeout=5
        )
        out = result.stderr + result.stdout
        if re.search(r"version \"(11|17|21)", out):
            return "java"
    except Exception:
        pass
    return None


def run_limes(
    config_path: str,
    java_exe: Optional[str] = None,
    limes_jar: Optional[str] = None,
) -> bool:
    """Run LIMES with the given config. Returns True on success."""
    java  = java_exe  or find_java()
    jar   = limes_jar or next((c for c in LIMES_CANDIDATES if os.path.exists(c)), None)

    if not java:
        print("[Step 5] WARNING: Java 11+ not found. Skipping LIMES.")
        return False
    if not jar:
        print("[Step 5] WARNING: limes.jar not found. Skipping LIMES.")
        return False

    print(f"[Step 5] Running LIMES: {java} -jar {jar} {config_path}")
    result = subprocess.run(
        [java, "-jar", jar, config_path],
        capture_output=True, text=True, timeout=120
    )
    # LIMES always exits 0 even on errors; check for meaningful output
    info_lines = [l for l in result.stderr.split("\n") if "INFO" in l and "Size" in l]
    if info_lines:
        print("  " + "\n  ".join(info_lines))
    if result.returncode != 0:
        print(f"  [LIMES exit {result.returncode}]")
        return False
    return True


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    msg_a: str,
    msg_b: str,
    ground_truth: Optional[str] = None,
    no_mnli: bool = False,
    out_dir: str = "outputs",
    java: Optional[str] = None,
    limes_jar: Optional[str] = None,
    metamodel_path: str = "outputs/uddl_metamodel.owl",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("limes_configs", exist_ok=True)

    # Derive output file names from input stems (no hardcoding)
    stem_a = Path(msg_a).stem   # e.g. "uddl_layered_aircraft_message_1"
    stem_b = Path(msg_b).stem

    owl_a = os.path.join(out_dir, f"{stem_a}.owl")
    owl_b = os.path.join(out_dir, f"{stem_b}.owl")
    nt_a  = os.path.join(out_dir, f"{stem_a}.nt")
    nt_b  = os.path.join(out_dir, f"{stem_b}.nt")
    ttl_a = os.path.join(out_dir, f"{stem_a}.ttl")
    ttl_b = os.path.join(out_dir, f"{stem_b}.ttl")

    limes_cfg    = f"limes_configs/limes_obs_{stem_a}_vs_{stem_b}.xml"
    limes_accept = os.path.join(out_dir, f"limes_obs_accept_{stem_a}_vs_{stem_b}.nt")
    limes_review = os.path.join(out_dir, f"limes_obs_review_{stem_a}_vs_{stem_b}.nt")
    out_csv      = os.path.join(out_dir, f"results_{stem_a}_vs_{stem_b}.csv")

    # Step 1: metamodel
    ensure_metamodel(metamodel_path)

    # Step 2: JSON -> OWL
    json_to_owl(msg_a, owl_a, metamodel_path)
    json_to_owl(msg_b, owl_b, metamodel_path)

    # Step 3: OWL -> NT + TTL
    owl_to_nt(owl_a, nt_a)
    owl_to_nt(owl_b, nt_b)
    owl_to_ttl(owl_a, ttl_a)
    owl_to_ttl(owl_b, ttl_b)

    # Step 4: generate LIMES config
    print(f"[Step 4] Writing LIMES config -> {limes_cfg}")
    config_xml = generate_limes_config(
        src_nt            = nt_a,
        tgt_nt            = nt_b,
        restriction_class = "ConceptualObservable",
        property_path     = "hasName",
        accept_out        = limes_accept,
        review_out        = limes_review,
        accept_threshold  = 0.5,
        review_threshold  = 0.2,
        metric            = "trigrams",
    )
    with open(limes_cfg, "w", encoding="utf-8") as f:
        f.write(config_xml)

    # Step 5: run LIMES (soft failure)
    # Clear stale cache so LIMES re-reads the NT files
    import shutil
    if os.path.exists("cache"):
        shutil.rmtree("cache")
    run_limes(limes_cfg, java_exe=java, limes_jar=limes_jar)

    # Step 6 + 7: evaluate
    print("[Step 6] Running matchers and evaluation ...")
    from evaluate_matching import run_evaluation
    run_evaluation(
        src_ttl           = ttl_a,
        tgt_ttl           = ttl_b,
        limes_accept      = limes_accept,
        limes_review      = limes_review,
        ground_truth_path = ground_truth,
        run_mnli          = not no_mnli,
        out_csv           = out_csv,
    )
    print(f"\n[Done] Results -> {out_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="UDDL message comparison pipeline (ICD1 vs ICD2)"
    )
    p.add_argument("--msg-a",        required=True, help="UDDL JSON file A (reference)")
    p.add_argument("--msg-b",        required=True, help="UDDL JSON file B (LLM-generated)")
    p.add_argument("--ground-truth", default=None,  help="Ground-truth JSON (optional)")
    p.add_argument("--no-mnli",      action="store_true")
    p.add_argument("--out-dir",      default="outputs")
    p.add_argument("--java",         default=None,
                   help="Path to java executable (Java 11+)")
    p.add_argument("--limes",        default=None,
                   help="Path to limes.jar")
    p.add_argument("--metamodel",    default="outputs/uddl_metamodel.owl")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        msg_a        = args.msg_a,
        msg_b        = args.msg_b,
        ground_truth = args.ground_truth,
        no_mnli      = args.no_mnli,
        out_dir      = args.out_dir,
        java         = args.java,
        limes_jar    = args.limes,
        metamodel_path = args.metamodel,
    )
