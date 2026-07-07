"""
run_all_pairs.py
----------------
Runs the enriched ontology matching pipeline for ALL possible within-domain
pairs across every domain in enriched_ontology_matching/inputs/.

For each domain (Automobile, Coffee, Homebrewing, Hospital, SmartHome,
University) with 6 models, this generates C(6,2)=15 pairs, running the
full pipeline on each.

Total: 6 domains × 15 pairs = 90 pairs.

Pipeline stages per pair
------------------------
  Step 1 — Enriched matching (AML + LogMap + WN + CN) → per-pair CSV
  Step 2 — Sentence embeddings (cosine_avg) — name-based fallback signal
  Step 3 — Attribute-type embeddings (type_embed_sim) — primary matching signal
  Step 4 — Merge enriched + embeddings + type-embeddings → metrics CSV

Outputs
-------
  enriched_ontology_matching/pairs/<domain_key>.json        — test JSON per pair
  enriched_ontology_matching/outputs/enriched/<stem>.csv    — per-pair CSV
  enriched_ontology_matching/outputs/enriched/all_domains_combined.csv

Usage
-----
  # From repo root, using the project venv:
  .venv/Scripts/python.exe enriched_ontology_matching/run_all_pairs.py --cross-domain

  # Skip already-computed pairs (re-use existing CSVs):
  .venv/Scripts/python.exe enriched_ontology_matching/run_all_pairs.py --cross-domain --skip-existing

  # Run only specific domains:
  .venv/Scripts/python.exe enriched_ontology_matching/run_all_pairs.py --domains Automobile Hospital

  # Only AML or only LogMap:
  .venv/Scripts/python.exe enriched_ontology_matching/run_all_pairs.py --matcher aml
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
_EXAMPLES  = _DIR / "inputs"
_PAIRS_DIR  = _DIR / "pairs"
_ENRICHED   = _DIR / "outputs" / "enriched"
_EMB_DIR    = _DIR / "outputs" / "embeddings"
_TYPE_DIR   = _DIR / "outputs" / "type_embed"
_MERGED_DIR = _DIR / "outputs" / "merged"
_WL_DIR     = _DIR / "outputs" / "wl"
_AD_DIR     = _DIR / "outputs" / "attr_dist"
_COMBINED   = _ENRICHED / "all_domains_combined.csv"

# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------
DOMAINS = ["Automobile", "Automobile_Synthetic", "Coffee", "Homebrewing", "Hospital", "SmartHome", "University"]

_DOMAIN_SHORT: dict[str, str] = {
    "Automobile":          "auto",
    "Automobile_Synthetic": "autosyn",
    "Coffee":              "coffee",
    "Homebrewing":         "brew",
    "Hospital":            "hosp",
    "SmartHome":           "smarthome",
    "University":          "univ",
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


def discover_models(domain: str, inputs_dir: Path = _EXAMPLES) -> list[Path]:
    """Return all JSON model files for a domain directory, sorted by name.

    Raises FileNotFoundError if the domain subdirectory does not exist.
    Raises ValueError if the directory exists but contains no JSON files.
    """
    domain_dir = inputs_dir / domain
    if not domain_dir.is_dir():
        raise FileNotFoundError(
            f"Domain directory not found: {domain_dir}\n"
            f"Expected structure: {inputs_dir}/<Domain>/*.json\n"
            f"Available domains: {[d.name for d in inputs_dir.iterdir() if d.is_dir()] if inputs_dir.is_dir() else '(inputs-dir not found)'}"
        )
    models = sorted(domain_dir.glob("*.json"))
    if not models:
        raise ValueError(
            f"No JSON files found in {domain_dir}.\n"
            "Each domain directory must contain at least 2 model JSON files."
        )
    return models


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
    run_embedding_stage,
    run_type_embed_stage,
    merge_pair,
    run_wl_stage,
    run_attr_dist_stage,
    _safe_local,
) -> tuple[str, Path] | None:
    """
    Run the full 6-step pipeline for a single model pair (a, b).

    Steps:
      1. Enriched matching (AML + LogMap + WN + CN) → per-pair CSV
      2. Sentence embeddings (cosine_avg) — name-based, used as matching fallback
      3. Attribute-type embeddings (type_embed_sim) — primary matching signal
      4. Merge enriched + embeddings + type-embeddings → metrics CSV
      5. WL kernel + shape metrics → wl CSV
      6. Attribute reach distribution → attr_dist CSV

    Returns (domain_key, enriched_csv_path) on success, None on failure.
    """
    data_a = json.loads(a.read_text(encoding="utf-8"))
    data_b = json.loads(b.read_text(encoding="utf-8"))

    name_a = data_a.get("modelName", a.stem)
    name_b = data_b.get("modelName", b.stem)

    stem     = f"{_safe_local(name_a)}_vs_{_safe_local(name_b)}"
    pair_csv = _ENRICHED / f"{stem}.csv"

    emb_csv    = _EMB_DIR    / f"{domain_key}_emb.csv"
    type_csv   = _TYPE_DIR   / f"{domain_key}_type_emb.csv"
    merged_csv = _MERGED_DIR / f"{stem}_metrics.csv"
    wl_csv     = _WL_DIR     / f"{stem}_metrics_wl.csv"
    ad_csv     = _AD_DIR     / f"{stem}_metrics_attr_dist.csv"

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

    # ── Step 2: Sentence embeddings ──────────────────────────────────────
    if skip_existing and emb_csv.exists():
        print(f"    [SKIP] Embeddings for {domain_key}")
    else:
        try:
            run_embedding_stage(csv_path, a, b, emb_csv)
        except Exception as exc:
            print(f"    [WARN] Embedding failed for {domain_key}: {exc}")

    # ── Step 3: Attribute-type embeddings ────────────────────────────────
    if skip_existing and type_csv.exists():
        print(f"    [SKIP] Type-embed for {domain_key}")
    else:
        try:
            run_type_embed_stage(csv_path, a, b, type_csv)
        except Exception as exc:
            print(f"    [WARN] Type-embed failed for {domain_key}: {exc}")

    # ── Step 4: Merge ────────────────────────────────────────────────────
    if skip_existing and merged_csv.exists():
        print(f"    [SKIP] Merge for {domain_key}")
    else:
        try:
            merge_pair(
                enriched_csv = csv_path,
                emb_csv      = emb_csv if emb_csv.exists() else None,
                type_csv     = type_csv if type_csv.exists() else None,
                out_csv      = merged_csv,
            )
        except Exception as exc:
            print(f"    [WARN] Merge failed for {domain_key}: {exc}")

    # ── Step 5: WL kernel + shape metrics ────────────────────────────────
    if skip_existing and wl_csv.exists():
        print(f"    [SKIP] WL for {domain_key}")
    else:
        try:
            data_a = json.loads(a.read_text(encoding="utf-8"))
            data_b = json.loads(b.read_text(encoding="utf-8"))
            run_wl_stage(data_a, data_b, merged_csv, wl_csv)
        except Exception as exc:
            print(f"    [WARN] WL failed for {domain_key}: {exc}")

    # ── Step 6: Attribute reach distribution ─────────────────────────────
    if skip_existing and ad_csv.exists():
        print(f"    [SKIP] AttrDist for {domain_key}")
    else:
        try:
            data_a = json.loads(a.read_text(encoding="utf-8"))
            data_b = json.loads(b.read_text(encoding="utf-8"))
            run_attr_dist_stage(data_a, data_b, ad_csv)
        except Exception as exc:
            print(f"    [WARN] AttrDist failed for {domain_key}: {exc}")

    return (domain_key, csv_path)


def main() -> None:
    """Entry point for the end-to-end enriched ontology matching pipeline.

    For each requested domain, discovers model JSONs under --inputs-dir/<Domain>/,
    runs all within-domain pairs through the 3-step pipeline (enriched matching →
    embeddings → merge), and combines results into a single CSV.

    Use --inputs-dir to point at any directory that contains domain subdirectories
    with JSON model files. Defaults to the built-in inputs/ directory.
    """
    parser = argparse.ArgumentParser(
        description="Run enriched pipeline on model pairs (within-domain and/or cross-domain)."
    )
    parser.add_argument(
        "--inputs-dir", default=str(_EXAMPLES), metavar="PATH",
        help=(
            "Root directory containing domain subdirectories with model JSON files. "
            "Expected layout: <inputs-dir>/<Domain>/*.json  "
            f"(default: {_EXAMPLES})"
        ),
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip pairs whose per-pair CSV already exists (use cached results).",
    )
    parser.add_argument(
        "--domains", nargs="*", default=None,
        help=(
            "Limit to specific domains (e.g. --domains Automobile Hospital). "
            f"Known domains: {DOMAINS}. "
            "Each domain must have a matching subdirectory under --inputs-dir."
        ),
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

    inputs_dir = Path(args.inputs_dir)
    if not inputs_dir.is_dir():
        raise FileNotFoundError(
            f"--inputs-dir not found: {inputs_dir}\n"
            "Create the directory and add domain subdirectories containing model JSON files."
        )

    from enriched_matcher import run_pipeline
    from logmap_runner import _safe_local
    from semantic_encoder import run_embedding_stage
    from merge_stage import merge_pair
    from wl_kernel_matcher import run_wl_stage
    from attribute_reach import run_attr_dist_stage, run_type_embed_stage

    _PAIRS_DIR.mkdir(exist_ok=True)
    _ENRICHED.mkdir(parents=True, exist_ok=True)
    _EMB_DIR.mkdir(parents=True, exist_ok=True)
    _TYPE_DIR.mkdir(parents=True, exist_ok=True)
    _MERGED_DIR.mkdir(parents=True, exist_ok=True)
    _WL_DIR.mkdir(parents=True, exist_ok=True)
    _AD_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_kwargs = dict(
        skip_existing         = args.skip_existing,
        use_aml               = args.matcher in ("aml",    "both"),
        use_logmap            = args.matcher in ("logmap", "both"),
        run_pipeline          = run_pipeline,
        run_embedding_stage   = run_embedding_stage,
        run_type_embed_stage  = run_type_embed_stage,
        merge_pair            = merge_pair,
        run_wl_stage          = run_wl_stage,
        run_attr_dist_stage   = run_attr_dist_stage,
        _safe_local           = _safe_local,
    )

    domains = args.domains if args.domains else DOMAINS
    for d in domains:
        if d not in DOMAINS:
            raise ValueError(
                f"Unknown domain '{d}'. "
                f"Valid options are: {DOMAINS}"
            )

    all_pair_csvs: list[tuple[str, Path]] = []

    # ── Within-domain pairs ────────────────────────────────────────────────
    for domain in domains:
        try:
            models = discover_models(domain, inputs_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[SKIP] {domain}: {exc}")
            continue

        if len(models) < 2:
            print(f"[WARN] {domain}: only {len(models)} model(s) found — need at least 2.")
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
            models_a = discover_models(domain_a, inputs_dir)
            models_b = discover_models(domain_b, inputs_dir)
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
    print("[Done]")


if __name__ == "__main__":
    main()
