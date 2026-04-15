"""
merge_stage.py
--------------
Step 5 — Unified per-pair CSV

Joins the four per-pair output CSVs (enriched, neighbourhood, lin_ic,
embeddings) into a single flat CSV keyed on (entity_a, entity_b).

All metrics side by side per entity pair:
  Structural  : source, matcher_conf, layer, semantic_label, layer2_type
  WN / CN     : wup_score, max_wup, avg_wup, wn_relation, cn_label, ...
  L0 Coherence: coherence_sym, n_nbrs_a, n_nbrs_b, verb_coherence, ...
  L3 Lin-IC   : lin_ic, max_lin_ic, avg_lin_ic, lcs, ic_lcs, token_lin_details
  L4 Embedding: cosine_whole, cosine_sum, cosine_prod, cosine_avg, tokens_a, tokens_b

Usage (standalone):
    venv/Scripts/python.exe enriched_ontology_matching/merge_stage.py \\
        --enriched  enriched_ontology_matching/outputs/enriched/<stem>.csv \\
        --nbr       enriched_ontology_matching/outputs/neighbourhood/<key>_coherence.csv \\
        --lin-ic    enriched_ontology_matching/outputs/lin_ic/<key>_lin_ic.csv \\
        --emb       enriched_ontology_matching/outputs/embeddings/<key>_emb.csv \\
        --out       enriched_ontology_matching/outputs/merged/<key>_full.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Merged field order
# ---------------------------------------------------------------------------
MERGED_FIELDS = [
    # Identity
    "entity_a", "entity_b",
    # Structural
    "source", "matcher_conf", "layer", "semantic_label", "layer2_type",
    # WN / CN lexical
    "token_a", "token_b",
    "wup_score", "max_wup", "avg_wup",
    "wn_relation", "wn_hops", "cn_relations", "cn_label", "gloss_hit",
    # L0 — Neighbourhood coherence
    "n_nbrs_a", "n_nbrs_b",
    "coherence_a2b", "coherence_b2a", "coherence_sym",
    "avg_best_a2b", "avg_best_b2a",
    "verb_coherence", "best_pairs",
    # L3 — Lin-IC
    "lin_ic", "max_lin_ic", "avg_lin_ic",
    "lcs", "ic_lcs", "token_lin_details",
    # L4 — Sentence embedding
    "cosine_whole", "cosine_sum", "cosine_prod", "cosine_avg",
    "tokens_a", "tokens_b",
]


def _read_csv(path: Path) -> list[dict]:
    if not path or not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _index(rows: list[dict]) -> dict[tuple, dict]:
    """Index rows by (entity_a, entity_b) — last row wins on duplicates."""
    return {(r["entity_a"], r["entity_b"]): r for r in rows}


def merge_pair(
    enriched_csv: Path,
    nbr_csv: Path | None,
    lin_ic_csv: Path | None,
    emb_csv: Path | None,
    out_csv: Path,
) -> list[dict]:
    """
    Left-join neighbourhood, lin_ic, and embedding rows onto the enriched
    CSV (base).  Missing secondary rows produce empty strings for those fields.

    Returns the merged rows.
    """
    base_rows = _read_csv(enriched_csv)
    nbr_idx   = _index(_read_csv(nbr_csv)    if nbr_csv    else [])
    lin_idx   = _index(_read_csv(lin_ic_csv) if lin_ic_csv else [])
    emb_idx   = _index(_read_csv(emb_csv)    if emb_csv    else [])

    merged: list[dict] = []
    for row in base_rows:
        key = (row.get("entity_a", ""), row.get("entity_b", ""))

        nbr = nbr_idx.get(key, {})
        lin = lin_idx.get(key, {})
        emb = emb_idx.get(key, {})

        out: dict = {}
        for field in MERGED_FIELDS:
            # Prefer enriched (base) for shared fields; fall back to secondary
            if field in row:
                out[field] = row[field]
            elif field in nbr:
                out[field] = nbr[field]
            elif field in lin:
                out[field] = lin[field]
            elif field in emb:
                out[field] = emb[field]
            else:
                out[field] = ""

        merged.append(out)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MERGED_FIELDS, extrasaction="ignore")
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
        description="Merge all per-pair stage CSVs into one flat CSV."
    )
    parser.add_argument("--enriched",  required=True, help="Enriched CSV (base)")
    parser.add_argument("--nbr",       default=None,  help="Neighbourhood coherence CSV")
    parser.add_argument("--lin-ic",    default=None,  help="Lin-IC CSV")
    parser.add_argument("--emb",       default=None,  help="Embedding cosine CSV")
    parser.add_argument("--out",       required=True, help="Output merged CSV path")
    args = parser.parse_args()

    merge_pair(
        enriched_csv = Path(args.enriched),
        nbr_csv      = Path(args.nbr)    if args.nbr    else None,
        lin_ic_csv   = Path(getattr(args, "lin_ic"))  if args.lin_ic else None,
        emb_csv      = Path(args.emb)    if args.emb    else None,
        out_csv      = Path(args.out),
    )
