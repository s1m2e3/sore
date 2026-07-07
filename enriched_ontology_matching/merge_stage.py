"""
merge_stage.py
--------------
Merge step — Metrics-only per-pair CSV

Joins the enriched-matcher CSV, the sentence-embedding CSV, and the
attribute-type-embedding CSV into a single flat CSV keyed on (entity_a, entity_b).

Output columns:
  entity_a, entity_b  — pair identity
  matched             — 1 if AML or LogMap confirmed this pair, else 0
  wup                 — (max_wup + avg_wup) / 2: blends best-token signal with full average
  cosine_avg          — name-based sentence-embedding cosine similarity (rescaled to [0, 1])
  type_embed_sim      — attribute-type-based embedding cosine similarity (attribute_reach.py);
                         empty when neither entity has any observable-type evidence
  primary_sim         — the signal actually used for entity-matching decisions:
                         type_embed_sim when it meets TYPE_FALLBACK_THRESHOLD (types are the
                         intended matching signal), else cosine_avg as a name-based fallback
                         for entities with no attribute-type evidence or a weak type signal.

Usage (standalone):
    venv/Scripts/python.exe enriched_ontology_matching/merge_stage.py \\
        --enriched  enriched_ontology_matching/outputs/enriched/<stem>.csv \\
        --emb       enriched_ontology_matching/outputs/embeddings/<key>_emb.csv \\
        --type      enriched_ontology_matching/outputs/type_embed/<key>_type_emb.csv \\
        --out       enriched_ontology_matching/outputs/merged/<key>_metrics.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

MERGED_FIELDS = [
    "entity_a", "entity_b",
    "matched",
    "wup",
    "cosine_avg",
    "type_embed_sim",
    "primary_sim",
]

# type_embed_sim at or above this counts as confident type evidence and wins
# outright over the name-based cosine_avg. Below it (or absent), fall back to
# cosine_avg — the model has too little attribute-type signal to trust it.
TYPE_FALLBACK_THRESHOLD = 0.5


def _to_float(v) -> float | None:
    """Parse a CSV cell to float; return None for blank/invalid."""
    s = str(v).strip() if v is not None else ""
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


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
    emb_csv: Path | None,
    out_csv: Path,
    type_csv: Path | None = None,
) -> list[dict]:
    """Join enriched-matcher, sentence-embedding, and type-embedding CSVs into a
    metrics-only flat CSV.

    enriched_csv is required (it defines the entity pairs). emb_csv and type_csv
    are optional — if missing, cosine_avg / type_embed_sim will be empty in the
    output. primary_sim is computed from whichever of the two is available,
    preferring type_embed_sim (see TYPE_FALLBACK_THRESHOLD).

    Raises FileNotFoundError if enriched_csv does not exist.
    """
    if not enriched_csv.exists():
        raise FileNotFoundError(
            f"Enriched CSV not found: {enriched_csv}\n"
            "Run enriched_matcher.py for this pair first."
        )
    base_rows = _read_csv(enriched_csv)
    emb_idx   = _index(_read_csv(emb_csv) if emb_csv else [])
    type_idx  = _index(_read_csv(type_csv) if type_csv else [])

    merged: list[dict] = []
    for row in base_rows:
        key = (row.get("entity_a", ""), row.get("entity_b", ""))
        src = row.get("source", "")
        emb = emb_idx.get(key, {})
        typ = type_idx.get(key, {})

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

        name_sim = _to_float(emb.get("cosine_avg"))
        type_sim = _to_float(typ.get("type_embed_sim"))

        # Types are the primary matching signal; naming is only a fallback
        # for entities with no/weak attribute-type evidence.
        if type_sim is not None and type_sim >= TYPE_FALLBACK_THRESHOLD:
            primary_sim = type_sim
        elif name_sim is not None:
            primary_sim = name_sim
        else:
            primary_sim = type_sim if type_sim is not None else ""

        merged.append({
            "entity_a":       key[0],
            "entity_b":       key[1],
            "matched":        1 if src in ("AML", "LogMap", "Both") else 0,
            "wup":            wup_val,
            "cosine_avg":     emb.get("cosine_avg", ""),
            "type_embed_sim": typ.get("type_embed_sim", ""),
            "primary_sim":    primary_sim,
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
        description="Merge enriched-matcher and embedding CSVs into a metrics-only flat CSV."
    )
    parser.add_argument("--enriched", required=True, help="Enriched CSV (base)")
    parser.add_argument("--emb",      default=None,  help="Name embedding cosine CSV")
    parser.add_argument("--type",     default=None,  help="Attribute-type embedding CSV")
    parser.add_argument("--out",      required=True, help="Output merged CSV path")
    args = parser.parse_args()

    merge_pair(
        enriched_csv = Path(args.enriched),
        emb_csv      = Path(args.emb) if args.emb else None,
        out_csv      = Path(args.out),
        type_csv     = Path(args.type) if args.type else None,
    )
