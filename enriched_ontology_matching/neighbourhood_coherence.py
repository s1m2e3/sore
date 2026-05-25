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
from model_normalizer import load_inventory, normalize_model
from attribute_reach import (
    collect_obs_vocab, attribute_reach, attribute_reach_similarity,
    _direct_attrs,
)

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
# Sentence-transformer model — reuse semantic_encoder's singleton to avoid
# loading two copies of the same paraphrase-MiniLM-L6-v2 into VRAM.
# ---------------------------------------------------------------------------

def _get_encoder():
    try:
        from semantic_encoder import _get_encoder as _se_get_encoder, DEFAULT_MODEL
        return _se_get_encoder(DEFAULT_MODEL)
    except Exception:
        return None


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
    """
    entity → set[canonical relation type] from every association it participates in.

    Requires data to have been passed through normalize_model() first so every
    association already carries a ``canonical`` field.  Inverse-expressed
    associations (A→B vs B←A) both resolve to the same canonical token, making
    verb_coherence a Jaccard overlap of canonical relation types rather than raw
    verb strings — inverse pairs score 1.0 instead of 0.0.
    """
    verbs: dict[str, set[str]] = defaultdict(set)
    for assoc in data.get("associations", []):
        canonical = assoc.get("canonical", "RelatedTo")
        parts     = _assoc_participants(assoc)
        for p in parts:
            verbs[p].add(canonical)
    return dict(verbs)


def _extract_verb_tokens(assoc_name: str) -> list[str]:
    tokens = [t.lower() for t in split_camel(assoc_name)]
    return [t for t in tokens if t in _VERB_TOKENS]


# ---------------------------------------------------------------------------
# Semantic scoring
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Coherence scorer — vectorised matrix approach
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

    coherence_a2b — mean over na in nbrs_A of max_nb(score(na, nb))
    coherence_b2a — mean over nb in nbrs_B of max_na(score(nb, na))
    coherence_sym — (coherence_a2b + coherence_b2a) / 2   [0–1, continuous]

    score(na, nb) = sqrt(wup * cosine).  The geometric mean kills WUP inflation
    from shared WordNet ancestors when cosine is low.

    Vectorised: builds a (|A| × |B|) WUP matrix via cached characterise_entity_pair
    calls, then computes the cosine matrix in one BLAS matrix-multiply, applies
    sqrt element-wise, and extracts row/column max with numpy — replacing the
    previous 2 × |A| × |B| Python loops.
    """
    nbrs_a = sorted(adj_a.get(ea, set()))
    nbrs_b = sorted(adj_b.get(eb, set()))

    result = {
        "entity_a": ea, "entity_b": eb,
        "n_nbrs_a": len(nbrs_a), "n_nbrs_b": len(nbrs_b),
        "coherence_a2b": 0.0, "coherence_b2a": 0.0, "coherence_sym": 0.0,
        "avg_best_a2b": 0.0, "avg_best_b2a": 0.0,
        "best_pairs": [],
    }

    if not nbrs_a or not nbrs_b:
        return result

    na, nb = len(nbrs_a), len(nbrs_b)

    # ── WUP matrix  (|A| × |B|) ─────────────────────────────────────────────
    # characterise_entity_pair is LRU-cached so this is O(1) per call after
    # the first encounter; the Python loop is unavoidable but fast.
    wup_mat = np.empty((na, nb), dtype=np.float32)
    for i, na_name in enumerate(nbrs_a):
        for j, nb_name in enumerate(nbrs_b):
            r = characterise_entity_pair(na_name, nb_name)
            wup_mat[i, j] = float(r.get("max_wup", r.get("wup_score", 0.0)))

    # ── Cosine matrix  (|A| × |B|) — single BLAS call ───────────────────────
    sample = next(iter(emb_cache.values()), None)
    if sample is not None:
        d = sample.shape[0]
        zero = np.zeros(d, dtype=np.float32)
        vecs_a = np.stack([emb_cache.get(n, zero) for n in nbrs_a]).astype(np.float32)
        vecs_b = np.stack([emb_cache.get(n, zero) for n in nbrs_b]).astype(np.float32)
        cos_mat = np.maximum(0.0, vecs_a @ vecs_b.T)  # (na, nb)

        # Entities without embeddings: fall back to WUP as the cosine proxy
        missing_a = np.array([n not in emb_cache for n in nbrs_a])
        missing_b = np.array([n not in emb_cache for n in nbrs_b])
        fallback   = missing_a[:, None] | missing_b[None, :]
        cos_mat[fallback] = wup_mat[fallback]
    else:
        cos_mat = wup_mat.copy()   # no encoder available: score = WUP

    # ── Score matrix: sqrt(wup × cos) — fully vectorised ────────────────────
    score_mat = np.sqrt(wup_mat * cos_mat)  # (na, nb)

    # ── Directional coherence: row-max → A→B, col-max → B→A ─────────────────
    scores_a2b = score_mat.max(axis=1)   # (na,)
    scores_b2a = score_mat.max(axis=0)   # (nb,)
    best_nb_idx = score_mat.argmax(axis=1)

    avg_a2b = float(scores_a2b.mean())
    avg_b2a = float(scores_b2a.mean())

    # Top-5 best pairs for the report column
    flat = score_mat.ravel()
    top_k = min(5, flat.size)
    top_idx = np.argpartition(flat, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
    best_pairs = [
        (round(float(flat[idx]), 3),
         nbrs_a[int(idx) // nb],
         nbrs_b[int(idx) % nb])
        for idx in top_idx
    ]

    result.update({
        "coherence_a2b": round(avg_a2b, 4),
        "coherence_b2a": round(avg_b2a, 4),
        "coherence_sym": round((avg_a2b + avg_b2a) / 2, 4),
        "avg_best_a2b":  round(avg_a2b, 4),
        "avg_best_b2a":  round(avg_b2a, 4),
        "best_pairs":    best_pairs,
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

    # ── Normalize: canonical associations + composition → PartOf ─────────────
    inventory = load_inventory()
    data_a = normalize_model(data_a, inventory)
    data_b = normalize_model(data_b, inventory)

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

    # ── Attribute reach ───────────────────────────────────────────────────────
    print("[AttrReach] Building shared observable vocabulary ...")
    obs_vocab = collect_obs_vocab(data_a, data_b)
    print(f"[AttrReach] Vocabulary: {len(obs_vocab)} observable types. "
          f"Computing 2-hop reach ...")
    reach_a = attribute_reach(data_a, obs_vocab, K=2)
    reach_b = attribute_reach(data_b, obs_vocab, K=2)

    ent_names_a = {e.get("entityName") or e.get("name","") for e in data_a.get("entities",[])}
    ent_names_b = {e.get("entityName") or e.get("name","") for e in data_b.get("entities",[])}
    direct_a = _direct_attrs(data_a, ent_names_a)
    direct_b = _direct_attrs(data_b, ent_names_b)

    # ── Build embedding cache for all neighbour entities ──────────────────────
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

        coh      = neighbourhood_coherence(ea, eb, adj_a, adj_b, emb_cache)
        vcoh     = verb_coherence(ea, eb, verbs_a, verbs_b)
        attr_sim = attribute_reach_similarity(ea, eb, reach_a, reach_b)

        # Leaf-leaf pairs (no association neighbours on either side) carry no
        # neighbourhood signal — averaging them as 0.0 artificially deflates
        # coherence for V-models whose attribute entities are structural leaves.
        # Use "" so downstream averaging (merge_stage) skips them entirely.
        leaf_pair = coh["n_nbrs_a"] == 0 and coh["n_nbrs_b"] == 0

        results.append({
            "entity_a":       ea,
            "entity_b":       eb,
            "semantic_label": sem,
            "wup_score":      wup,
            "max_wup":        row.get("max_wup", wup),
            "avg_wup":        row.get("avg_wup", wup),
            "n_nbrs_a":       coh["n_nbrs_a"],
            "n_nbrs_b":       coh["n_nbrs_b"],
            "coherence_a2b":  "" if leaf_pair else coh["coherence_a2b"],
            "coherence_b2a":  "" if leaf_pair else coh["coherence_b2a"],
            "coherence_sym":  "" if leaf_pair else coh["coherence_sym"],
            "avg_best_a2b":   "" if leaf_pair else coh["avg_best_a2b"],
            "avg_best_b2a":   "" if leaf_pair else coh["avg_best_b2a"],
            "verb_coherence": vcoh,
            "attr_reach_sim": attr_sim,
            "n_obs_a":        len(direct_a.get(ea, {})),
            "n_obs_b":        len(direct_b.get(eb, {})),
            "n_reach_a":      len(reach_a.get(ea, {})),
            "n_reach_b":      len(reach_b.get(eb, {})),
            "best_pairs":     "" if leaf_pair else " | ".join(
                f"{nb}<->{nc}({s})" for s, nb, nc in coh["best_pairs"]
            ),
        })

    # Sort: non-empty coherence_sym first (descending), then leaf pairs last
    def _sort_key(r):
        v = r["coherence_sym"]
        return (1, 0.0) if v == "" else (0, -float(v))
    results.sort(key=_sort_key)
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
    "avg_best_a2b", "avg_best_b2a", "verb_coherence",
    "attr_reach_sim", "n_obs_a", "n_obs_b", "n_reach_a", "n_reach_b",
    "best_pairs",
]


def _print_results(results: list[dict], name_a: str, name_b: str, threshold: float) -> None:
    w = 26
    print(f"\n{'Entity A':<{w}} {'Entity B':<{w}} {'Sem':<14} "
          f"{'WUP':>5} {'Coh_sym':>8} {'VerbCoh':>8} {'AttrSim':>8}  Best neighbour pairs")
    print("-" * 160)
    for r in results:
        coh_val = r["coherence_sym"]
        coh_str = f"{float(coh_val):>8.3f}" if coh_val != "" else f"{'(leaf)':>8}"
        flag    = " *" if (coh_val != "" and float(coh_val) >= threshold) else ""
        print(
            f"{r['entity_a']:<{w}} {r['entity_b']:<{w}} "
            f"{r['semantic_label']:<14} "
            f"{str(r['wup_score']):>5} "
            f"{coh_str} {r['verb_coherence']:>8.3f} "
            f"{r['attr_reach_sim']:>8.4f}"
            f"  {str(r['best_pairs'])[:60]}{flag}"
        )
    high = sum(1 for r in results if r["coherence_sym"] != "" and float(r["coherence_sym"]) >= threshold)
    connected = sum(1 for r in results if r["coherence_sym"] != "")
    print(f"\n  Total pairs: {len(results)}  |  connected (non-leaf): {connected}  |  "
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
