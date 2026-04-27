"""
neighbourhood_coherence.py
--------------------------
Computes semantically-weighted neighbourhood coherence scores for ontology
match pairs.

For a matched pair (ea, eb):
  1. Collect nbrs_a = all entities adjacent to ea in ontology A
     Collect nbrs_b = all entities adjacent to eb in ontology B
  2. For each na in nbrs_a, compute:
         score(na, nb) = sqrt(wup(na, nb) * cosine(na, nb))   [geometric mean]
     and take the max over all nb in nbrs_b.
  3. coherence_a2b = mean of those per-neighbour max scores   [0–1, continuous]
  4. Compute symmetrically coherence_b2a.
  5. coherence_sym = (coherence_a2b + coherence_b2a) / 2

The geometric mean of WUP and cosine acts as a gate:
  - High WUP + high cosine → confirmed semantic match → full credit
  - High WUP + low cosine  → WordNet topology accident (device.n.01 etc.) → killed
  - Low WUP + high cosine  → distributional similarity without WN path → partial credit

All coherence scores are continuous [0–1]; no binary threshold is used in
the final output.

Works for any domain — handles both JSON association formats:
  V-model format : associationName  / associationParticipants
  Network format : name             / participants

Usage
-----
  # From repo root:
  python enriched_ontology_matching/neighbourhood_coherence.py \\
      --pair enriched_ontology_matching/pairs/auto_V1_V2.json \\
      --matches-csv enriched_ontology_matching/outputs/enriched/<stem>.csv \\
      --out-csv     enriched_ontology_matching/outputs/neighbourhood/auto_V1_V2_coherence.csv

  # Or with separate model JSONs:
  python enriched_ontology_matching/neighbourhood_coherence.py \\
      --a model_a.json --b model_b.json ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Optional

import numpy as np

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

from root_comparator import split_camel, _cn_load_csv, _CN_CSV_DEFAULT
from enriched_matcher import characterise_entity_pair

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CN_CSV = _CN_CSV_DEFAULT

_VERB_TOKENS = {
    "to", "in", "on", "at", "with", "of", "from", "by", "via", "through",
    "into", "onto", "between", "within", "across", "along",
    "supports", "support", "drives", "drive", "powers", "power",
    "lubricates", "lubricate", "contains", "contain", "connects", "connect",
    "mounts", "mount", "links", "link", "couples", "couple", "charges",
    "charge", "feeds", "feed", "engages", "engage", "modulates", "modulate",
    "actuates", "actuate", "controls", "control", "monitors", "monitor",
    "supplies", "supply", "regulates", "regulate", "circulates", "circulate",
    "pressurises", "pressurize", "synchronises", "synchronize", "transfers",
    "transfer", "converts", "convert", "packages", "package", "adjacent",
    "composed", "packaged", "interacts", "interact", "manages", "manage",
    "requests", "request", "provides", "provide", "uses", "use", "has",
    "have", "is", "are",
}

# ---------------------------------------------------------------------------
# Lazy sentence-transformer model
# ---------------------------------------------------------------------------
_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        try:
            from sentence_transformers import SentenceTransformer
            _ENCODER = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            print("[Encoder] paraphrase-MiniLM-L6-v2 loaded.")
        except ImportError:
            print("[Encoder] sentence-transformers not installed — cosine fallback to WUP.")
            _ENCODER = False
    return _ENCODER


def _build_emb_cache(names: list[str]) -> dict[str, np.ndarray]:
    """
    Batch-encode a list of entity names into unit-normalised vectors.
    CamelCase names are split into readable tokens before encoding.
    Returns {} if the encoder is unavailable.
    """
    model = _get_encoder()
    if not model:
        return {}
    readable = [" ".join(split_camel(n)) or n for n in names]
    vecs = model.encode(readable, normalize_embeddings=True, show_progress_bar=False)
    return {name: vec for name, vec in zip(names, vecs)}


# ---------------------------------------------------------------------------
# Association format helpers
# ---------------------------------------------------------------------------

def _assoc_name(assoc: dict) -> str:
    return assoc.get("associationName") or assoc.get("name") or ""


def _assoc_participants(assoc: dict) -> list[str]:
    return assoc.get("associationParticipants") or assoc.get("participants") or []


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def build_adjacency(data: dict) -> dict[str, set[str]]:
    """entity → set[neighbour_entity] (undirected)."""
    adj: dict[str, set[str]] = defaultdict(set)
    for assoc in data.get("associations", []):
        parts = _assoc_participants(assoc)
        for p in parts:
            for q in parts:
                if q != p:
                    adj[p].add(q)
    return dict(adj)


def build_edge_verbs(data: dict) -> dict[str, set[str]]:
    """entity → set[verb_token] from association names."""
    verbs: dict[str, set[str]] = defaultdict(set)
    for assoc in data.get("associations", []):
        name  = _assoc_name(assoc)
        parts = _assoc_participants(assoc)
        v = _extract_verb_tokens(name)
        for p in parts:
            verbs[p].update(v)
    return dict(verbs)


def _extract_verb_tokens(assoc_name: str) -> list[str]:
    tokens = [t.lower() for t in split_camel(assoc_name)]
    return [t for t in tokens if t in _VERB_TOKENS]


# ---------------------------------------------------------------------------
# Semantic scoring
# ---------------------------------------------------------------------------

def _semantic_score(na: str, nb: str, emb_cache: dict) -> float:
    """
    Geometric mean of WUP and cosine similarity for a neighbour entity pair.

    sqrt(wup * cosine) requires BOTH signals to be high:
      - WordNet topology accident (high WUP, low cosine) → score stays low
      - Embedding-only match (low WUP, high cosine) → partial credit
      - Confirmed semantic match (high WUP, high cosine) → full credit
    """
    r   = characterise_entity_pair(na, nb)
    wup = float(r.get("max_wup", r.get("wup_score", 0.0)))

    va = emb_cache.get(na)
    vb = emb_cache.get(nb)
    if va is not None and vb is not None:
        cos = max(0.0, float(np.dot(va, vb)))
    else:
        cos = wup  # graceful fallback when encoder unavailable

    return sqrt(wup * cos)


# ---------------------------------------------------------------------------
# Coherence scorer
# ---------------------------------------------------------------------------

def neighbourhood_coherence(
    ea: str,
    eb: str,
    adj_a: dict[str, set[str]],
    adj_b: dict[str, set[str]],
    emb_cache: dict,
) -> dict:
    """
    Compute semantic neighbourhood coherence for a matched entity pair (ea, eb).

    coherence_a2b — mean over na in nbrs_A of max_nb(semantic_score(na, nb))
    coherence_b2a — mean over nb in nbrs_B of max_na(semantic_score(nb, na))
    coherence_sym — (coherence_a2b + coherence_b2a) / 2   [0–1, continuous]

    Both directions use sqrt(wup * cosine) per neighbour pair so that WUP
    inflation from shared WordNet ancestors is killed by low cosine.
    """
    nbrs_a = adj_a.get(ea, set())
    nbrs_b = adj_b.get(eb, set())

    result = {
        "entity_a": ea, "entity_b": eb,
        "n_nbrs_a": len(nbrs_a), "n_nbrs_b": len(nbrs_b),
        "coherence_a2b": 0.0, "coherence_b2a": 0.0, "coherence_sym": 0.0,
        "avg_best_a2b": 0.0, "avg_best_b2a": 0.0,
        "best_pairs": [],
    }

    if not nbrs_a or not nbrs_b:
        return result

    # A → B
    scores_a2b: list[float] = []
    all_pairs: list[tuple[float, str, str]] = []
    for na in nbrs_a:
        best = 0.0
        best_nb = ""
        for nb in nbrs_b:
            s = _semantic_score(na, nb, emb_cache)
            if s > best:
                best = s
                best_nb = nb
        scores_a2b.append(best)
        all_pairs.append((best, na, best_nb))

    # B → A
    scores_b2a: list[float] = []
    for nb in nbrs_b:
        best = 0.0
        for na in nbrs_a:
            s = _semantic_score(nb, na, emb_cache)
            if s > best:
                best = s
        scores_b2a.append(best)

    avg_a2b = sum(scores_a2b) / len(scores_a2b)
    avg_b2a = sum(scores_b2a) / len(scores_b2a)
    all_pairs.sort(reverse=True)

    result.update({
        "coherence_a2b": round(avg_a2b, 4),
        "coherence_b2a": round(avg_b2a, 4),
        "coherence_sym": round((avg_a2b + avg_b2a) / 2, 4),
        "avg_best_a2b":  round(avg_a2b, 4),
        "avg_best_b2a":  round(avg_b2a, 4),
        "best_pairs":    [(round(s, 3), na, nb) for s, na, nb in all_pairs[:5]],
    })
    return result


def verb_coherence(
    ea: str,
    eb: str,
    verbs_a: dict[str, set[str]],
    verbs_b: dict[str, set[str]],
) -> float:
    """Jaccard overlap of verb/relation types used by ea and eb."""
    va = verbs_a.get(ea, set())
    vb = verbs_b.get(eb, set())
    if not va and not vb:
        return 0.0
    return round(len(va & vb) / len(va | vb), 4)


# ---------------------------------------------------------------------------
# Match loader / fallback
# ---------------------------------------------------------------------------

def _load_matches_from_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _compute_matches_simple(data_a: dict, data_b: dict, name_a: str, name_b: str) -> list[dict]:
    from aml_runner import AMLRunner
    print("[Matches] No CSV provided — running AML to get L1 matches ...")
    aml_maps = AMLRunner().match_jsons(data_a, data_b, name_a=name_a, name_b=name_b)
    from enriched_matcher import layer1_annotate
    return layer1_annotate(aml_maps, source_label="AML")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    json_a: Path | str | None = None,
    json_b: Path | str | None = None,
    pair_json: Path | str | None = None,
    matches_csv: Optional[Path] = None,
    out_csv: Optional[Path] = None,
    threshold: float = 0.5,    # kept for summary printout only
) -> list[dict]:
    """
    Full semantic neighbourhood coherence analysis for an ontology pair.

    Supply either:
      - pair_json : a combined pair JSON with "json_a" and "json_b" keys, OR
      - json_a + json_b : paths to the two individual model JSON files

    Parameters
    ----------
    matches_csv : pre-computed enriched per-pair CSV (L1+L2 matches)
    out_csv     : path to write results CSV
    threshold   : display-only; does NOT affect coherence_sym value
    """
    # ── Load model data ──────────────────────────────────────────────────────
    if pair_json is not None:
        p = Path(pair_json)
        if not p.exists():
            raise FileNotFoundError(
                f"Pair JSON not found: {p}\n"
                "Expected a file with 'json_a' and 'json_b' keys."
            )
        combined = json.loads(p.read_text(encoding="utf-8"))
        data_a = combined.get("json_a", combined)
        data_b = combined.get("json_b", {})
        name_a = data_a.get("modelName", p.stem + "_a")
        name_b = data_b.get("modelName", p.stem + "_b")
    elif json_a is not None and json_b is not None:
        path_a = Path(json_a)
        path_b = Path(json_b)
        for label, p in [("json_a", path_a), ("json_b", path_b)]:
            if not p.exists():
                raise FileNotFoundError(f"{label} not found: {p}")
        data_a = json.loads(path_a.read_text(encoding="utf-8"))
        data_b = json.loads(path_b.read_text(encoding="utf-8"))
        name_a = data_a.get("modelName", path_a.stem)
        name_b = data_b.get("modelName", path_b.stem)
    else:
        raise ValueError("Supply either pair_json or both json_a and json_b.")

    print(f"\n{'='*70}")
    print(f"Neighbourhood Coherence Analysis (semantic-weighted)")
    print(f"  A: {name_a}")
    print(f"  B: {name_b}")
    print(f"{'='*70}")

    # ── Build graphs ─────────────────────────────────────────────────────────
    adj_a   = build_adjacency(data_a)
    adj_b   = build_adjacency(data_b)
    verbs_a = build_edge_verbs(data_a)
    verbs_b = build_edge_verbs(data_b)

    print(f"  Graph A: {len(data_a.get('entities',[]))} entities, "
          f"{len(adj_a)} with edges")
    print(f"  Graph B: {len(data_b.get('entities',[]))} entities, "
          f"{len(adj_b)} with edges")

    # ── Load or compute matches ───────────────────────────────────────────────
    if matches_csv and Path(matches_csv).exists():
        match_rows = _load_matches_from_csv(Path(matches_csv))
        print(f"  Loaded {len(match_rows)} pairs from {Path(matches_csv).name}")
    else:
        match_rows = _compute_matches_simple(data_a, data_b, name_a, name_b)

    if not match_rows:
        print("  [WARN] No matches found.")
        return []

    # ── Pre-load ConceptNet ───────────────────────────────────────────────────
    print("[CN] Pre-loading ConceptNet CSV ...")
    _cn_load_csv(_CN_CSV)

    # ── Build embedding cache for all neighbour entities ──────────────────────
    # Collect every entity name that will appear in any neighbour comparison.
    all_neighbor_names: set[str] = set()
    for row in match_rows:
        ea, eb = row["entity_a"], row["entity_b"]
        all_neighbor_names.update(adj_a.get(ea, set()))
        all_neighbor_names.update(adj_b.get(eb, set()))

    print(f"[Encoder] Building embedding cache for {len(all_neighbor_names)} neighbour entities ...")
    emb_cache = _build_emb_cache(list(all_neighbor_names))

    # ── Compute coherence for each matched pair ───────────────────────────────
    print(f"\n[Coherence] Analysing {len(match_rows)} pairs ...\n")
    results = []
    for row in match_rows:
        ea  = row["entity_a"]
        eb  = row["entity_b"]
        sem = row.get("semantic_label", "")
        wup = row.get("wup_score", "")

        coh  = neighbourhood_coherence(ea, eb, adj_a, adj_b, emb_cache)
        vcoh = verb_coherence(ea, eb, verbs_a, verbs_b)

        results.append({
            "entity_a":       ea,
            "entity_b":       eb,
            "semantic_label": sem,
            "wup_score":      wup,
            "max_wup":        row.get("max_wup", wup),
            "avg_wup":        row.get("avg_wup", wup),
            "n_nbrs_a":       coh["n_nbrs_a"],
            "n_nbrs_b":       coh["n_nbrs_b"],
            "coherence_a2b":  coh["coherence_a2b"],
            "coherence_b2a":  coh["coherence_b2a"],
            "coherence_sym":  coh["coherence_sym"],
            "avg_best_a2b":   coh["avg_best_a2b"],
            "avg_best_b2a":   coh["avg_best_b2a"],
            "verb_coherence": vcoh,
            "best_pairs":     " | ".join(
                f"{nb}<->{nc}({s})" for s, nb, nc in coh["best_pairs"]
            ),
        })

    results.sort(key=lambda r: r["coherence_sym"], reverse=True)
    _print_results(results, name_a, name_b, threshold)

    if out_csv:
        _write_results_csv(results, Path(out_csv))

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_COL_FIELDS = [
    "entity_a", "entity_b", "semantic_label", "wup_score", "max_wup", "avg_wup",
    "n_nbrs_a", "n_nbrs_b",
    "coherence_a2b", "coherence_b2a", "coherence_sym",
    "avg_best_a2b", "avg_best_b2a", "verb_coherence", "best_pairs",
]


def _print_results(results: list[dict], name_a: str, name_b: str, threshold: float) -> None:
    w = 26
    print(f"\n{'Entity A':<{w}} {'Entity B':<{w}} {'Sem':<14} "
          f"{'WUP':>5} {'Coh_A2B':>8} {'Coh_B2A':>8} "
          f"{'Coh_sym':>8} {'VerbCoh':>8}  Best neighbour pairs")
    print("-" * 160)
    for r in results:
        flag = " *" if r["coherence_sym"] >= threshold else ""
        print(
            f"{r['entity_a']:<{w}} {r['entity_b']:<{w}} "
            f"{r['semantic_label']:<14} "
            f"{str(r['wup_score']):>5} "
            f"{r['coherence_a2b']:>8.3f} {r['coherence_b2a']:>8.3f} "
            f"{r['coherence_sym']:>8.3f} {r['verb_coherence']:>8.3f}"
            f"  {r['best_pairs'][:70]}{flag}"
        )
    high = sum(1 for r in results if r["coherence_sym"] >= threshold)
    print(f"\n  Total pairs: {len(results)}  |  "
          f"coherence_sym >= {threshold}: {high}")


def _write_results_csv(results: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COL_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\n[CSV] Written {len(results)} rows -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic neighbourhood coherence for an ontology pair."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pair", dest="pair_json",
                       help="Combined pair JSON with json_a / json_b keys.")
    group.add_argument("--a",   dest="json_a",
                       help="Path to ontology A JSON (use with --b).")
    parser.add_argument("--b", dest="json_b",
                        help="Path to ontology B JSON (use with --a).")
    parser.add_argument("--matches-csv", default=None,
                        help="Pre-computed enriched per-pair CSV (L1+L2 matches).")
    parser.add_argument("--out-csv", default=None,
                        help="Output CSV path for coherence results.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Display threshold for summary (default 0.5). "
                             "Does NOT affect coherence_sym value.")
    args = parser.parse_args()

    if args.json_a and not args.json_b:
        parser.error("--a requires --b")

    run_analysis(
        json_a      = args.json_a,
        json_b      = args.json_b,
        pair_json   = args.pair_json,
        matches_csv = args.matches_csv,
        out_csv     = args.out_csv,
        threshold   = args.threshold,
    )
