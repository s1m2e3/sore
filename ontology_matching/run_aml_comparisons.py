"""
run_aml_comparisons.py
----------------------
Runs pairwise ontology alignment using actual AgreementMakerLight (AML) v3.2
for all domains in inputs/CONceptual_ExtractionCategory_Examples/.

AML is invoked as a Java subprocess from tools/AML/AML_v3.2/ (so its
store/ directory is found automatically).

Pairs compared per domain
-------------------------
  v-models   : v1 vs v2, v1 vs v3, v2 vs v3  (3 pairs)
  variations : var1 vs var2, var1 vs var3, var2 vs var3  (3 pairs)
  Total      : 6 pairs x 6 domains = 36 alignment files

Output layout
-------------
  AML/
    <Domain>/
      <ModelA>_vs_<ModelB>.rdf   (AML Alignment API RDF/XML output)

Usage
-----
    cd ontology_matching
    python run_aml_comparisons.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from itertools import combinations

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR  = os.path.join(
    BASE_DIR, "inputs",
    "CONceptual_ExtractionCategory_Examples",
    "CONceptual_ExtractionCategory_Examples",
)
OWL_DIR     = os.path.join(BASE_DIR, "outputs")
AML_OUT_DIR = os.path.join(BASE_DIR, "AML")
AML_DIR     = os.path.join(BASE_DIR, "tools", "AML", "AML_v3.2")
AML_JAR     = os.path.join(AML_DIR, "AgreementMakerLight.jar")
JAVA_EXE    = r"C:\Program Files\Microsoft\jdk-11.0.30.7-hotspot\bin\java.exe"


def _safe(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip())
    return re.sub(r"_+", "_", s).strip("_") or "unknown"


def _owl_path(domain: str, model_name: str) -> str:
    return os.path.join(OWL_DIR, domain, f"{_safe(model_name)}.owl")


def _collect_domain_files(domain_dir: str) -> tuple[list[str], list[str]]:
    all_json  = sorted(glob.glob(os.path.join(domain_dir, "*.json")))
    v_files   = [f for f in all_json if "variation" not in os.path.basename(f)]
    var_files = [f for f in all_json if "variation"     in os.path.basename(f)]
    return v_files, var_files


def run_aml(owl_a: str, owl_b: str, out_rdf: str) -> tuple[bool, str]:
    """
    Invoke AML as a Java subprocess.

    AML must be run from its own directory so it can find store/.
    Returns (success, stderr_snippet).
    """
    os.makedirs(os.path.dirname(out_rdf), exist_ok=True)
    cmd = [
        JAVA_EXE, "-jar", AML_JAR,
        "-s", owl_a,
        "-t", owl_b,
        "-o", out_rdf,
        "-a",                    # automatic matching mode
    ]
    result = subprocess.run(
        cmd,
        cwd=AML_DIR,             # must run from AML dir so store/ is found
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        return False, (result.stderr or result.stdout)[-400:]
    return True, ""


def run_all() -> None:
    if not os.path.isfile(AML_JAR):
        print(f"[ERROR] AML JAR not found at: {AML_JAR}")
        return

    total_ok, total_fail = 0, 0

    domain_dirs = sorted([
        d for d in glob.glob(os.path.join(INPUTS_DIR, "*"))
        if os.path.isdir(d)
    ])

    for domain_dir in domain_dirs:
        domain = os.path.basename(domain_dir)
        v_files, var_files = _collect_domain_files(domain_dir)
        aml_domain_dir = os.path.join(AML_OUT_DIR, domain)
        os.makedirs(aml_domain_dir, exist_ok=True)

        print(f"\n=== {domain} ===")

        for group_label, group in [("v-models", v_files), ("variations", var_files)]:
            pairs = list(combinations(group, 2))
            if not pairs:
                continue
            print(f"  [{group_label}]  {len(pairs)} pairs")

            for path_a, path_b in pairs:
                with open(path_a, encoding="utf-8") as f:
                    data_a = json.load(f)
                with open(path_b, encoding="utf-8") as f:
                    data_b = json.load(f)

                name_a  = data_a.get("modelName", os.path.basename(path_a))
                name_b  = data_b.get("modelName", os.path.basename(path_b))
                owl_a   = _owl_path(domain, name_a)
                owl_b   = _owl_path(domain, name_b)
                out_rdf = os.path.join(
                    aml_domain_dir,
                    f"{_safe(name_a)}_vs_{_safe(name_b)}.rdf",
                )

                label = f"{_safe(name_a)[:28]:<28} vs {_safe(name_b)[:28]:<28}"
                print(f"    {label} ...", end=" ", flush=True)

                ok, err = run_aml(owl_a, owl_b, out_rdf)
                if ok:
                    # Count Cell elements in the produced RDF
                    try:
                        with open(out_rdf, encoding="utf-8") as f:
                            n = f.read().count("<Cell>")
                    except Exception:
                        n = -1
                    print(f"{n:5d} correspondences")
                    total_ok += 1
                else:
                    print(f"FAILED\n      {err}")
                    total_fail += 1

    print(f"\n=== Done: {total_ok} alignments written, {total_fail} failed ===")
    print(f"    Output: {AML_OUT_DIR}")


if __name__ == "__main__":
    run_all()
