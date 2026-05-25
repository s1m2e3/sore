"""
merge_stage.py
--------------
Step 5 — Metrics-only per-pair CSV

Joins the four per-pair stage CSVs (enriched, neighbourhood, lin_ic,
embeddings) into a single flat CSV keyed on (entity_a, entity_b).

Output columns (metrics only — no intermediate derivation fields):
  entity_a, entity_b        — pair identity
  matched                   — 1 if AML or LogMap found this pair, else 0
  wup                       — (max_wup + avg_wup) / 2: blends best-token signal with full average
  lin_ic                    — Lin Information Content similarity (0–1)
  coherence_sym             — symmetric neighbourhood coherence (0–1)
  cosine_avg                — token-average embedding cosine similarity

Usage (standalone):
    venv/Scripts/python.exe enriched_ontology_matching/merge_stage.py \\
        --enriched  enriched_ontology_matching/outputs/enriched/<stem>.csv \\
        --nbr       enriched_ontology_matching/outputs/neighbourhood/<key>_coherence.csv \\
        --lin-ic    enriched_ontology_matching/outputs/lin_ic/<key>_lin_ic.csv \\
        --emb       enriched_ontology_matching/outputs/embeddings/<key>_emb.csv \\
        --out       enriched_ontology_matching/outputs/merged/<key>_metrics.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

MERGED_FIELDS = [
    "entity_a", "entity_b",
    "matched",
    "wup",
    "lin_ic",
    "coherence_sym",
    "verb_coherence",
    "attr_reach_sim",
    "cosine_avg",
    "gnn_sim",
    "entailment_a_covers_b",
    "entailment_b_covers_a",
    "entailment_f1",
]


def _read_csv(path: Path) -> list[dict]:
    """Read a CSV file and return its rows as dicts. Returns [] if path is None or missing."""
    if not path or not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _index(rows: list[dict]) -> dict[tuple, dict]:
    """Build a (entity_a, entity_b) → row dict for fast keyed lookup."""
    return {(r["entity_a"], r["entity_b"]): r for r in rows}


def merge_pair(
    enriched_csv: Path,
    nbr_csv: Path | None,
    lin_ic_csv: Path | None,
    emb_csv: Path | None,
    out_csv: Path,
    closure_csv: Path | None = None,
    gnn_csv: Path | None = None,
) -> list[dict]:
    """Join per-pair stage CSVs into a single metrics-only flat CSV.

    enriched_csv is required (it defines the entity pairs). The neighbourhood,
    lin_ic, emb, and closure CSVs are optional — missing files are silently
    skipped and their metric columns will be empty in the output.

    Raises FileNotFoundError if enriched_csv does not exist.
    """
    if not enriched_csv.exists():
        raise FileNotFoundError(
            f"Enriched CSV not found: {enriched_csv}\n"
            "Run enriched_matcher.py for this pair first."
        )
    base_rows = _read_csv(enriched_csv)
    nbr_idx     = _index(_read_csv(nbr_csv)     if nbr_csv     else [])
    lin_idx     = _index(_read_csv(lin_ic_csv)  if lin_ic_csv  else [])
    emb_idx     = _index(_read_csv(emb_csv)     if emb_csv     else [])
    closure_idx = _index(_read_csv(closure_csv) if closure_csv else [])
    gnn_idx     = _index(_read_csv(gnn_csv)     if gnn_csv     else [])

    merged: list[dict] = []
    for row in base_rows:
        key  = (row.get("entity_a", ""), row.get("entity_b", ""))
        src  = row.get("source", "")

        nbr     = nbr_idx.get(key, {})
        lin     = lin_idx.get(key, {})
        emb     = emb_idx.get(key, {})
        closure = closure_idx.get(key, {})
        gnn     = gnn_idx.get(key, {})

        # New format: (max_wup + avg_wup) / 2.
        # Old format fallback: use wup_score (equivalent to per-token max).
        raw_max   = row.get("max_wup", "").strip()
        raw_avg   = row.get("avg_wup", "").strip()
        raw_score = row.get("wup_score", "").strip()
        try:
            wup_val = round((float(raw_max) + float(raw_avg)) / 2, 4)
        except (ValueError, TypeError):
            try:
                wup_val = round(float(raw_max or raw_avg or raw_score), 4)
            except (ValueError, TypeError):
                wup_val = ""

        merged.append({
            "entity_a":              key[0],
            "entity_b":              key[1],
            "matched":               1 if src in ("AML", "LogMap", "Both") else 0,
            "wup":                   wup_val,
            "lin_ic":                lin.get("lin_ic", ""),
            "coherence_sym":         nbr.get("coherence_sym", ""),
            "verb_coherence":        nbr.get("verb_coherence", ""),
            "attr_reach_sim":        nbr.get("attr_reach_sim", ""),
            "cosine_avg":            emb.get("cosine_avg", ""),
            "gnn_sim":               gnn.get("gnn_sim", ""),
            "entailment_a_covers_b": closure.get("entailment_a_covers_b", ""),
            "entailment_b_covers_a": closure.get("entailment_b_covers_a", ""),
            "entailment_f1":         closure.get("entailment_f1", ""),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MERGED_FIELDS)
        w.writeheader()
        w.writerows(merged)

    print(f"[Merge] {len(merged)} rows -> {out_csv}")
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge all per-pair stage CSVs into a metrics-only flat CSV."
    )
    parser.add_argument("--enriched",  required=True, help="Enriched CSV (base)")
    parser.add_argument("--nbr",       default=None,  help="Neighbourhood coherence CSV")
    parser.add_argument("--lin-ic",    default=None,  help="Lin-IC CSV")
    parser.add_argument("--emb",       default=None,  help="Embedding cosine CSV")
    parser.add_argument("--closure",   default=None,  help="Containment closure CSV")
    parser.add_argument("--gnn",       default=None,  help="GNN similarity CSV")
    parser.add_argument("--out",       required=True, help="Output merged CSV path")
    args = parser.parse_args()

    merge_pair(
        enriched_csv = Path(args.enriched),
        nbr_csv      = Path(args.nbr)      if args.nbr      else None,
        lin_ic_csv   = Path(getattr(args, "lin_ic")) if args.lin_ic else None,
        emb_csv      = Path(args.emb)      if args.emb      else None,
        closure_csv  = Path(args.closure)  if args.closure  else None,
        gnn_csv      = Path(args.gnn)      if args.gnn      else None,
        out_csv      = Path(args.out),
    )
