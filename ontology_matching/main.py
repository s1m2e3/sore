"""
main.py
-------
Builds the concept_metamodel.owl and converts every JSON in
inputs/CONceptual_ExtractionCategory_Examples/ to a conforming OWL file.

Usage
-----
    cd ontology_matching
    python main.py
"""

from __future__ import annotations

import glob
import json
import os

from base_ontology import build_concept_metamodel
from json_to_concept_owl import ConceptJsonToOWL

BASE_DIR   = os.path.dirname(__file__)
INPUTS_DIR = os.path.join(
    BASE_DIR,
    "inputs",
    "CONceptual_ExtractionCategory_Examples",
    "CONceptual_ExtractionCategory_Examples",
)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def main() -> None:
    # Step 1 – build/save the shared metamodel; keep the live ontology object
    print("=== Step 1: Build conceptual metamodel ===")
    meta_onto = build_concept_metamodel()

    # Step 2 – convert every JSON (pass the live ontology object to avoid
    # Windows file URI issues when re-loading from disk)
    print("\n=== Step 2: Convert all JSON files to OWL ===")
    converter = ConceptJsonToOWL(meta_onto=meta_onto, output_dir=OUTPUT_DIR)

    json_files = sorted(glob.glob(os.path.join(INPUTS_DIR, "**", "*.json"), recursive=True))
    if not json_files:
        print(f"No JSON files found under {INPUTS_DIR}")
        return

    success, failed = 0, []
    for jf in json_files:
        # Domain name = parent folder of the JSON (e.g. Automobile, Coffee …)
        domain = os.path.basename(os.path.dirname(jf))
        domain_out_dir = os.path.join(OUTPUT_DIR, domain)
        os.makedirs(domain_out_dir, exist_ok=True)
        try:
            converter.convert_file(jf, output_dir=domain_out_dir)
            success += 1
        except Exception as exc:
            print(f"  [ERROR] {os.path.basename(jf)}: {exc}")
            failed.append(jf)

    print(f"\n=== Done: {success} converted, {len(failed)} failed ===")
    if failed:
        for f in failed:
            print(f"  FAILED: {f}")


if __name__ == "__main__":
    main()
