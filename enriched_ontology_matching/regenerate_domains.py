"""
regenerate_domains.py
---------------------
Force-regenerate the full enriched-ontology-matching pipeline for one or more
domains, then produce a domain summary JSON for each.

Each domain runs ALL six pipeline stages (L0-L6 + merge) from scratch —
existing outputs are overwritten (no --skip-existing flag is passed).

GPU/CUDA is used automatically by semantic_encoder.py (L4) and
containment_closure.py (L5) if torch detects a CUDA-capable device.

Usage
-----
# Regenerate all 5 non-Automobile domains (typical first run)
python scripts/regenerate_domains.py

# Regenerate ALL 6 domains (including Automobile)
python scripts/regenerate_domains.py --domains all

# Regenerate specific domain(s)
python scripts/regenerate_domains.py --domains Hospital University

# Regenerate then also convert each summary JSON to CSV
python scripts/regenerate_domains.py --to-csv

# Regenerate then build the combined 3-D ontology map HTML
python scripts/regenerate_domains.py --all-domains --ontology-map
"""

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT   = Path(__file__).resolve().parent.parent
_PYTHON      = sys.executable
_RUNNER      = _REPO_ROOT / "enriched_ontology_matching" / "run_all_pairs.py"
_COMPARE     = _REPO_ROOT / "enriched_ontology_matching" / "compare_stage.py"
_INPUTS_DIR  = _REPO_ROOT / "enriched_ontology_matching" / "inputs"
_SUMMARIES   = _REPO_ROOT / "enriched_ontology_matching" / "summaries"
_SUMMARY_CSV = _REPO_ROOT / "enriched_ontology_matching" / "summary_to_csv.py"

ALL_DOMAINS       = ["Automobile", "Coffee", "Homebrewing", "Hospital", "SmartHome", "University"]
DEFAULT_DOMAINS   = ["Coffee", "Homebrewing", "Hospital", "SmartHome", "University"]


def _print_gpu_status() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[GPU] CUDA available — {name}")
        else:
            print("[GPU] CUDA not available — running on CPU")
    except ImportError:
        print("[GPU] torch not importable from this interpreter — GPU status unknown")


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        print(f"[ERROR] '{label}' exited with code {result.returncode}")
        return False
    return True


def regenerate_domain(domain: str, matcher: str = "both") -> bool:
    """Run the full pipeline for one domain (no --skip-existing = force overwrite)."""
    ok = _run(
        [
            _PYTHON, str(_RUNNER),
            "--inputs-dir", str(_INPUTS_DIR),
            "--domains", domain,
            "--matcher", matcher,
        ],
        f"Pipeline — {domain}",
    )
    if not ok:
        return False

    # Generate domain summary JSON
    ok = _run(
        [_PYTHON, str(_COMPARE), "--domain-summary", domain],
        f"Summary JSON — {domain}",
    )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Force-regenerate the pipeline for selected domains."
    )
    parser.add_argument(
        "--domains", nargs="*", default=None,
        metavar="DOMAIN",
        help=(
            f"Domains to regenerate. Use 'all' for all six, or list names. "
            f"Default: {DEFAULT_DOMAINS}."
        ),
    )
    parser.add_argument(
        "--matcher", choices=["aml", "logmap", "both"], default="both",
        help="Which structural matcher(s) to run (default: both).",
    )
    parser.add_argument(
        "--to-csv", action="store_true",
        help="After regeneration, convert each summary JSON to CSV via summary_to_csv.py.",
    )
    parser.add_argument(
        "--combine-csv", action="store_true",
        help="With --to-csv: merge all domains into one all_domains_summary.csv.",
    )
    parser.add_argument(
        "--ontology-map", action="store_true",
        help="After regeneration, rebuild the interactive 3-D ontology map HTML.",
    )
    args = parser.parse_args()

    # --- Resolve domain list --------------------------------------------------
    if args.domains is None:
        domains = DEFAULT_DOMAINS
    elif len(args.domains) == 1 and args.domains[0].lower() == "all":
        domains = ALL_DOMAINS
    else:
        unknown = [d for d in args.domains if d not in ALL_DOMAINS]
        if unknown:
            print(f"[ERROR] Unknown domain(s): {unknown}")
            print(f"        Known: {ALL_DOMAINS}")
            sys.exit(1)
        domains = args.domains

    # --- GPU check ------------------------------------------------------------
    _print_gpu_status()
    print(f"\nDomains to regenerate: {domains}")
    print(f"Matcher: {args.matcher}")

    # --- Pipeline per domain --------------------------------------------------
    failed = []
    for domain in domains:
        ok = regenerate_domain(domain, matcher=args.matcher)
        if not ok:
            failed.append(domain)

    # --- Optional: CSV export ------------------------------------------------
    if args.to_csv:
        csv_cmd = [_PYTHON, str(_SUMMARY_CSV)]
        if args.combine_csv:
            csv_cmd.append("--combine")
        _run(csv_cmd, "Convert summary JSONs -> CSV")

    # --- Optional: rebuild 3-D map -------------------------------------------
    if args.ontology_map:
        _run([_PYTHON, str(_COMPARE)], "Rebuild ontology map HTML")

    # --- Summary -------------------------------------------------------------
    print("\n" + "="*60)
    if failed:
        print(f"[DONE WITH ERRORS] Failed domains: {failed}")
        sys.exit(1)
    else:
        print(f"[DONE] All {len(domains)} domain(s) regenerated successfully.")
        print(f"       Summary JSONs -> {_SUMMARIES}/")


if __name__ == "__main__":
    main()
