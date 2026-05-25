"""
summary_to_csv.py
-----------------
Convert one or more domain summary JSON files
(enriched_ontology_matching/summaries/*_summary.json)
into flat CSV files.

Usage
-----
# All summaries -> one CSV per file in summaries/
python scripts/summary_to_csv.py

# Specific domain(s)
python scripts/summary_to_csv.py --domains Automobile Hospital

# All summaries -> one combined CSV
python scripts/summary_to_csv.py --combine

# Custom output directory
python scripts/summary_to_csv.py --out-dir my_csvs/
"""

import argparse
import csv
import json
from pathlib import Path

_REPO_ROOT   = Path(__file__).resolve().parent.parent
_SUMMARIES   = _REPO_ROOT / "enriched_ontology_matching" / "summaries"
_METRICS     = ["cosine_avg", "wup", "lin_ic", "coherence_sym"]

FIELDNAMES = [
    "domain", "ont_a", "ont_b",
    "distance", "composite", "n_entity_pairs",
    "cosine_avg", "wup", "lin_ic", "coherence_sym",
    "cosine_avg_weight", "wup_weight", "lin_ic_weight", "coherence_sym_weight",
]


def load_summary(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as fh:
        return json.load(fh)


def summary_to_rows(data: dict) -> list[dict]:
    domain  = data["domain"]
    weights = data.get("metric_weights", {})
    rows = []
    for pair in data.get("pairs", []):
        row = {
            "domain":        domain,
            "ont_a":         pair["ont_a"],
            "ont_b":         pair["ont_b"],
            "distance":      pair["distance"],
            "composite":     pair["composite"],
            "n_entity_pairs": pair["n_entity_pairs"],
        }
        for m in _METRICS:
            row[m]              = pair.get("metrics", {}).get(m, {}).get("mean", "")
            row[f"{m}_weight"]  = pair.get("metrics", {}).get(m, {}).get("weight",
                                    weights.get(m, ""))
        rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert domain summary JSON(s) to CSV.")
    parser.add_argument(
        "--domains", nargs="*", default=None,
        metavar="DOMAIN",
        help="Domains to convert (default: all *.json in summaries/).",
    )
    parser.add_argument(
        "--summaries-dir", default=str(_SUMMARIES), metavar="PATH",
        help=f"Directory containing *_summary.json files (default: {_SUMMARIES}).",
    )
    parser.add_argument(
        "--out-dir", default=None, metavar="PATH",
        help="Output directory for CSV files (default: same as summaries dir).",
    )
    parser.add_argument(
        "--combine", action="store_true",
        help="Write all domains into a single combined CSV instead of one per domain.",
    )
    args = parser.parse_args()

    summaries_dir = Path(args.summaries_dir)
    out_dir       = Path(args.out_dir) if args.out_dir else summaries_dir

    if args.domains:
        json_files = [summaries_dir / f"{d.lower()}_summary.json" for d in args.domains]
        missing = [f for f in json_files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Summary JSON(s) not found: {missing}")
    else:
        json_files = sorted(summaries_dir.glob("*_summary.json"))
        if not json_files:
            raise FileNotFoundError(f"No *_summary.json files found in {summaries_dir}")

    print(f"Converting {len(json_files)} summary file(s)...")

    if args.combine:
        all_rows = []
        for jf in json_files:
            data = load_summary(jf)
            all_rows.extend(summary_to_rows(data))
        write_csv(all_rows, out_dir / "all_domains_summary.csv")
    else:
        for jf in json_files:
            data = load_summary(jf)
            rows = summary_to_rows(data)
            out_path = out_dir / jf.name.replace(".json", ".csv")
            write_csv(rows, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
