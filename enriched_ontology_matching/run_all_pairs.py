"""
run_all_pairs.py
----------------
Runs the enriched ontology matching pipeline for ALL possible within-domain
pairs across every domain in CONceptual_ExtractionCategory_Examples.

For each domain (Automobile, Coffee, Homebrewing, Hospital, SmartHome,
University) with 6 models, this generates C(6,2)=15 pairs, running the
full 2-layer AML+LogMap+WordNet+ConceptNet pipeline on each.

Total: 6 domains × 15 pairs = 90 pairs.

Outputs
-------
  enriched_ontology_matching/pairs/<domain_key>.json        — test JSON per pair
  enriched_ontology_matching/outputs/enriched/<stem>.csv    — per-pair CSV
  enriched_ontology_matching/outputs/enriched/all_domains_combined.csv
  enriched_ontology_matching/outputs/all_domains_results.md

Usage
-----
  # From repo root, using the project venv:
  ontology_matching/.venv/Scripts/python.exe ^
      enriched_ontology_matching/run_all_pairs.py

  # Skip already-computed pairs (re-use existing CSVs):
  ontology_matching/.venv/Scripts/python.exe ^
      enriched_ontology_matching/run_all_pairs.py --skip-existing

  # Run only specific domains:
  ontology_matching/.venv/Scripts/python.exe ^
      enriched_ontology_matching/run_all_pairs.py --domains Automobile Hospital

  # Only AML or only LogMap:
  ontology_matching/.venv/Scripts/python.exe ^
      enriched_ontology_matching/run_all_pairs.py --matcher aml
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

# Make enriched_ontology_matching importable
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = _DIR.parent
_EXAMPLES  = (
    _DIR / "inputs"
    / "CONceptual_ExtractionCategory_Examples"
    / "CONceptual_ExtractionCategory_Examples"
)
_PAIRS_DIR  = _DIR / "pairs"
_ENRICHED   = _DIR / "outputs" / "enriched"
_NBR_DIR    = _DIR / "outputs" / "neighbourhood"
_LIN_IC_DIR = _DIR / "outputs" / "lin_ic"
_EMB_DIR    = _DIR / "outputs" / "embeddings"
_MERGED_DIR = _DIR / "outputs" / "merged"
_COMBINED   = _ENRICHED / "all_domains_combined.csv"
_MD_OUT     = _DIR / "outputs" / "all_domains_results.md"

# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------
DOMAINS = ["Automobile", "Coffee", "Homebrewing", "Hospital", "SmartHome", "University"]

_DOMAIN_SHORT: dict[str, str] = {
    "Automobile":  "auto",
    "Coffee":      "coffee",
    "Homebrewing": "brew",
    "Hospital":    "hosp",
    "SmartHome":   "smarthome",
    "University":  "univ",
}


def _model_short(json_path: Path) -> str:
    """
    Derive a 2-5 char short key from the JSON file stem.

    Variation models  → Net1 / Net2 / Net3
    V-models          → V1 / V2 / V3
    """
    stem = json_path.stem.lower()
    for i, tag in enumerate(("variation_1", "variation_2", "variation_3"), start=1):
        if tag in stem:
            return f"Net{i}"
    for v in ("_v1", "_v2", "_v3"):
        if stem.endswith(v):
            return v[1:].upper()   # "_v1" → "V1"
    return stem[:8]


def discover_models(domain: str) -> list[Path]:
    """Return all JSON model files for a domain, sorted."""
    return sorted((_EXAMPLES / domain).glob("*.json"))


# ---------------------------------------------------------------------------
# Combined CSV field order  (adds 'domain' column to per-pair CSV fields)
# ---------------------------------------------------------------------------
_COMBINED_FIELDS = [
    "domain",
    "entity_a", "entity_b", "source", "matcher_conf", "layer",
    "token_a", "token_b",
    "wup_score", "max_wup", "avg_wup", "wn_relation", "wn_hops",
    "cn_relations", "cn_label",
    "gloss_hit", "semantic_label", "layer2_type",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_pair(
    a: Path, b: Path,
    domain_key: str,
    skip_existing: bool,
    use_aml: bool,
    use_logmap: bool,
    run_pipeline,
    nbr_analysis,
    run_lin_ic,
    run_embedding_stage,
    merge_pair,
    _safe_local,
) -> tuple[str, Path] | None:
    """
    Run the full 5-step pipeline for a single model pair (a, b).

    Returns (domain_key, enriched_csv_path) on success, None on failure.
    """
    data_a = json.loads(a.read_text(encoding="utf-8"))
    data_b = json.loads(b.read_text(encoding="utf-8"))

    name_a = data_a.get("modelName", a.stem)
    name_b = data_b.get("modelName", b.stem)

    stem     = f"{_safe_local(name_a)}_vs_{_safe_local(name_b)}"
    pair_csv = _ENRICHED / f"{stem}.csv"

    nbr_csv    = _NBR_DIR    / f"{domain_key}_coherence.csv"
    lin_ic_csv = _LIN_IC_DIR / f"{domain_key}_lin_ic.csv"
    emb_csv    = _EMB_DIR    / f"{domain_key}_emb.csv"
    merged_csv = _MERGED_DIR / f"{stem}_metrics.csv"

    # ── Step 1: Enriched pipeline ────────────────────────────────────────
    if skip_existing and pair_csv.exists():
        print(f"  [SKIP] {domain_key}  (enriched CSV exists)")
        csv_path = pair_csv
    else:
        pair_json = _PAIRS_DIR / f"{domain_key}.json"
        pair_json.write_text(
            json.dumps({"json_a": data_a, "json_b": data_b}, indent=2),
            encoding="utf-8",
        )
        print(f"\n  [PAIR] {domain_key}")
        print(f"    A: {name_a}  ({len(data_a.get('entities', []))} ents, "
              f"{len(data_a.get('associations', []))} assocs)")
        print(f"    B: {name_b}  ({len(data_b.get('entities', []))} ents, "
              f"{len(data_b.get('associations', []))} assocs)")
        try:
            csv_path = run_pipeline(pair_json, use_aml=use_aml, use_logmap=use_logmap)
        except Exception as exc:
            print(f"  [ERROR] {domain_key}: {exc}")
            return None

    # ── Step 2: Neighbourhood coherence ─────────────────────────────────
    if skip_existing and nbr_csv.exists():
        print(f"    [SKIP] Coherence for {domain_key}")
    else:
        try:
            nbr_analysis(json_a=a, json_b=b, matches_csv=csv_path,
                         out_csv=nbr_csv, threshold=0.5)
        except Exception as exc:
            print(f"    [WARN] Coherence failed for {domain_key}: {exc}")

    # ── Step 3: Lin-IC ───────────────────────────────────────────────────
    if skip_existing and lin_ic_csv.exists():
        print(f"    [SKIP] Lin-IC for {domain_key}")
    else:
        try:
            run_lin_ic(csv_path, lin_ic_csv)
        except Exception as exc:
            print(f"    [WARN] Lin-IC failed for {domain_key}: {exc}")

    # ── Step 4: Sentence embeddings ──────────────────────────────────────
    if skip_existing and emb_csv.exists():
        print(f"    [SKIP] Embeddings for {domain_key}")
    else:
        try:
            run_embedding_stage(csv_path, a, b, emb_csv)
        except Exception as exc:
            print(f"    [WARN] Embedding failed for {domain_key}: {exc}")

    # ── Step 5: Merge ────────────────────────────────────────────────────
    if skip_existing and merged_csv.exists():
        print(f"    [SKIP] Merge for {domain_key}")
    else:
        try:
            merge_pair(
                enriched_csv = csv_path,
                nbr_csv      = nbr_csv      if nbr_csv.exists()     else None,
                lin_ic_csv   = lin_ic_csv   if lin_ic_csv.exists()  else None,
                emb_csv      = emb_csv      if emb_csv.exists()     else None,
                out_csv      = merged_csv,
            )
        except Exception as exc:
            print(f"    [WARN] Merge failed for {domain_key}: {exc}")

    return (domain_key, csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run enriched pipeline on model pairs (within-domain and/or cross-domain)."
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip pairs whose per-pair CSV already exists (use cached results).",
    )
    parser.add_argument(
        "--domains", nargs="*", default=None,
        help="Limit to specific domains (e.g. --domains Automobile Hospital).",
    )
    parser.add_argument(
        "--matcher", choices=["aml", "logmap", "both"], default="both",
        help="Which structural matcher(s) to run (default: both).",
    )
    parser.add_argument(
        "--cross-domain", action="store_true",
        help="Also run cross-domain pairs (AML+LogMap across all selected domain pairs).",
    )
    args = parser.parse_args()

    from enriched_matcher import run_pipeline
    from logmap_runner import _safe_local
    from neighbourhood_coherence import run_analysis as nbr_analysis
    from lin_ic_stage import run_lin_ic
    from semantic_encoder import run_embedding_stage
    from merge_stage import merge_pair

    _PAIRS_DIR.mkdir(exist_ok=True)
    _ENRICHED.mkdir(parents=True, exist_ok=True)
    _NBR_DIR.mkdir(parents=True, exist_ok=True)
    _LIN_IC_DIR.mkdir(parents=True, exist_ok=True)
    _EMB_DIR.mkdir(parents=True, exist_ok=True)
    _MERGED_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_kwargs = dict(
        skip_existing      = args.skip_existing,
        use_aml            = args.matcher in ("aml",    "both"),
        use_logmap         = args.matcher in ("logmap", "both"),
        run_pipeline       = run_pipeline,
        nbr_analysis       = nbr_analysis,
        run_lin_ic         = run_lin_ic,
        run_embedding_stage= run_embedding_stage,
        merge_pair         = merge_pair,
        _safe_local        = _safe_local,
    )

    domains = args.domains if args.domains else DOMAINS
    all_pair_csvs: list[tuple[str, Path]] = []

    # ── Within-domain pairs ────────────────────────────────────────────────
    for domain in domains:
        if domain not in DOMAINS:
            print(f"[WARN] Unknown domain '{domain}'. Valid: {DOMAINS}")
            continue

        models = discover_models(domain)
        if len(models) < 2:
            print(f"[WARN] {domain}: only {len(models)} model(s) found, skipping.")
            continue

        d_short = _DOMAIN_SHORT.get(domain, domain.lower())
        pairs   = list(itertools.combinations(models, 2))
        print(f"\n{'='*60}")
        print(f"Domain: {domain}  ({len(models)} models -> {len(pairs)} pairs)")
        print(f"{'='*60}")

        for a, b in pairs:
            short_a    = _model_short(a)
            short_b    = _model_short(b)
            domain_key = f"{d_short}_{short_a}_{short_b}"
            result = _run_pair(a, b, domain_key, **pipeline_kwargs)
            if result:
                all_pair_csvs.append(result)

    # ── Cross-domain pairs ─────────────────────────────────────────────────
    if args.cross_domain:
        valid_domains = [d for d in domains if d in DOMAINS]
        cross_pairs_total = sum(
            len(discover_models(da)) * len(discover_models(db))
            for da, db in itertools.combinations(valid_domains, 2)
        )
        print(f"\n{'='*60}")
        print(f"Cross-domain pairs ({cross_pairs_total} total)")
        print(f"{'='*60}")

        for domain_a, domain_b in itertools.combinations(valid_domains, 2):
            models_a = discover_models(domain_a)
            models_b = discover_models(domain_b)
            d_short_a = _DOMAIN_SHORT.get(domain_a, domain_a.lower())
            d_short_b = _DOMAIN_SHORT.get(domain_b, domain_b.lower())

            print(f"\n  {domain_a} x {domain_b}: "
                  f"{len(models_a)} x {len(models_b)} = {len(models_a)*len(models_b)} pairs")

            for a, b in itertools.product(models_a, models_b):
                short_a    = _model_short(a)
                short_b    = _model_short(b)
                domain_key = f"{d_short_a}_{short_a}_vs_{d_short_b}_{short_b}"
                result = _run_pair(a, b, domain_key, **pipeline_kwargs)
                if result:
                    all_pair_csvs.append(result)

    if not all_pair_csvs:
        print("\n[WARN] No pairs processed. Check domain names and input files.")
        return

    # ── Combine all per-pair CSVs ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Combine] {len(all_pair_csvs)} pair CSVs -> {_COMBINED}")
    total_rows = 0
    with open(_COMBINED, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COMBINED_FIELDS, extrasaction="ignore")
        w.writeheader()
        for domain_key, csv_path in all_pair_csvs:
            if not csv_path.exists():
                print(f"  [WARN] Missing CSV for {domain_key}: {csv_path}")
                continue
            with open(csv_path, newline="", encoding="utf-8") as rf:
                for row in csv.DictReader(rf):
                    row["domain"] = domain_key
                    w.writerow({k: row.get(k, "") for k in _COMBINED_FIELDS})
                    total_rows += 1
    print(f"[Combine] {total_rows} rows written.")

    # ── Generate MD report ─────────────────────────────────────────────────
    print(f"\n[Report] Generating {_MD_OUT} ...")
    from generate_report import generate
    generate(_COMBINED, _MD_OUT, nbr_dir=_NBR_DIR, lin_ic_dir=_LIN_IC_DIR,
             emb_dir=_EMB_DIR)
    print("[Done]")


if __name__ == "__main__":
    main()
