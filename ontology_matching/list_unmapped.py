"""
list_unmapped.py
----------------
For each pair, reads the JSON report and prints every entity and attribute
in the SMALLER model that has status 'missing' (not found in the larger model).

Output: outputs/unmapped_summary.csv  (one row per unmapped element)

Usage:
    cd ontology_matching
    python list_unmapped.py
"""

from __future__ import annotations

import csv
import glob
import json
import os

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "reports")
OUT_CSV     = os.path.join(os.path.dirname(__file__), "outputs", "unmapped_summary.csv")


def run() -> None:
    rows = []

    for jf in sorted(glob.glob(os.path.join(REPORTS_DIR, "**", "*.json"), recursive=True)):
        with open(jf, encoding="utf-8") as f:
            d = json.load(f)

        meta = d["metadata"]
        sa   = d["summary"]["model_a"]
        sb   = d["summary"]["model_b"]
        total_a = sa["matched"] + sa["ambiguous"] + sa["missing"]
        total_b = sb["matched"] + sb["ambiguous"] + sb["missing"]

        # Pick the smaller model side
        if total_a <= total_b:
            ref_side   = "model_a"
            ref_model  = meta["model_a"]
            other_model= meta["model_b"]
            ref_total  = total_a
        else:
            ref_side   = "model_b"
            ref_model  = meta["model_b"]
            other_model= meta["model_a"]
            ref_total  = total_b

        side_data = d[ref_side]

        missing_ents  = [e for e in side_data["entities"]   if e["status"] == "missing"]
        missing_attrs = [a for a in side_data["attributes"] if a["status"] == "missing"]

        cov = round(100 * d["summary"][ref_side]["matched"] / ref_total, 1) if ref_total else 0

        for e in missing_ents:
            rows.append({
                "domain":       meta["domain"],
                "smaller_model":ref_model,
                "larger_model": other_model,
                "coverage_pct": cov,
                "element_kind": "entity",
                "entity":       e["name"],
                "attribute":    "",
            })

        for a in missing_attrs:
            rows.append({
                "domain":       meta["domain"],
                "smaller_model":ref_model,
                "larger_model": other_model,
                "coverage_pct": cov,
                "element_kind": "attribute",
                "entity":       a["entity"],
                "attribute":    a["name"],
            })

    if not rows:
        print("No unmapped elements found.")
        return

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"Written {len(rows)} unmapped elements -> {OUT_CSV}")
    print()

    # Console summary grouped by domain + pair
    from itertools import groupby
    key = lambda r: (r["domain"], r["smaller_model"][:40], r["larger_model"][:40], r["coverage_pct"])
    sorted_rows = sorted(rows, key=key)

    for (domain, sm, lm, cov), group in groupby(sorted_rows, key=key):
        items = list(group)
        ents  = [i for i in items if i["element_kind"] == "entity"]
        attrs = [i for i in items if i["element_kind"] == "attribute"]
        print(f"[{domain}]  {sm}  ->  {lm}  (coverage {cov}%)")
        if ents:
            print(f"  Unmapped entities ({len(ents)}):")
            for e in ents:
                print(f"    - {e['entity']}")
        if attrs:
            print(f"  Unmapped attributes ({len(attrs)}):")
            for a in attrs:
                print(f"    - {a['entity']}.{a['attribute']}")
        print()


if __name__ == "__main__":
    run()
