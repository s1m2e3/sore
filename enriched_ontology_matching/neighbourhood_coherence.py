"""
neighbourhood_coherence.py
--------------------------
Computes neighbourhood coherence scores for ontology match pairs using
WordNet + ConceptNet semantic comparison of each entity's local graph context.

For a matched pair (ea, eb):
  1. Collect nbrs_a = all entities adjacent to ea in ontology A
     Collect nbrs_b = all entities adjacent to eb in ontology B
  2. For each na in nbrs_a, find its best-matching neighbour in nbrs_b by max_wup.
  3. coherence_a2b = mean of those best scores (how well A's context is explained by B)
  4. Compute symmetrically coherence_b2a.
  5. coherence_sym = (a2b + b2a) / 2  — final symmetric score [0, 1]

Also compares the association verb tokens (the relationship types used within each
ontology) to give a separate relational-type coherence signal.

Works for any domain — handles both JSON association formats:
  V-model format : associationName  / associationParticipants
  Network format : name             / participants

Usage
-----
  # From repo root:
  ontology_matching/.venv/Scripts/python.exe ^
      enriched_ontology_matching/neighbourhood_coherence.py ^
      --a  inputs/.../Automobile/automobile_model_v1.json ^
      --b  inputs/.../Automobile/automobile_model_v2.json ^
      [--matches-csv enriched_ontology_matching/outputs/enriched/<stem>.csv] ^
      [--out-csv     enriched_ontology_matching/outputs/neighbourhood/auto_v1_v2.csv] ^
      [--threshold   0.5]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

from root_comparator import split_camel, _cn_load_csv, _CN_CSV_DEFAULT
from enriched_matcher import characterise_entity_pair

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CN_CSV = _CN_CSV_DEFAULT

# Tokens that indicate a relationship verb in association names
_VERB_TOKENS = {
    # prepositions / structural
    "to", "in", "on", "at", "with", "of", "from", "by", "via", "through",
    "into", "onto", "between", "within", "across", "along",
    # common verb stems
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

_STOP_ENTITY_TOKENS = {
    "system", "model", "type", "base", "item", "unit", "data", "info",
    "module", "assembly", "component", "part", "sub",
}


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
    """
    Build entity → set[neighbour_entity] map from any JSON format.
    Edges are undirected (each participant is connected to all others).
    """
    adj: dict[str, set[str]] = defaultdict(set)
    for assoc in data.get("associations", []):
        parts = _assoc_participants(assoc)
        for i, p in enumerate(parts):
            for q in parts:
                if q != p:
                    adj[p].add(q)
    return dict(adj)


def build_edge_verbs(data: dict) -> dict[str, set[str]]:
    """
    Build entity → set[verb_token] map: what relation types does each entity
    participate in?  Verbs are extracted from CamelCase association names.
    """
    verbs: dict[str, set[str]] = defaultdict(set)
    for assoc in data.get("associations", []):
        name  = _assoc_name(assoc)
        parts = _assoc_participants(assoc)
        v = _extract_verb_tokens(name)
        for p in parts:
            verbs[p].update(v)
    return dict(verbs)


def _extract_verb_tokens(assoc_name: str) -> list[str]:
    """
    Extract the verb / preposition tokens from a CamelCase association name.
    e.g. "EngineToTransmissionCoupling"   → ["to"]
         "PistonAssemblyDrivesCrankshaft" → ["drives"]
         "BodyFrameSupportsEngineBlock"   → ["supports"]
    """
    tokens = [t.lower() for t in split_camel(assoc_name)]
    return [t for t in tokens if t in _VERB_TOKENS]


# ---------------------------------------------------------------------------
# Coherence scorer
# ---------------------------------------------------------------------------

def _best_match_score(na: str, nbrs_b: set[str]) -> float:
    """Max max_wup between entity na and any entity in nbrs_b."""
    if not nbrs_b:
        return 0.0
    best = 0.0
    for nb in nbrs_b:
        r = characterise_entity_pair(na, nb)
        score = r.get("max_wup", r.get("wup_score", 0.0))
        if score > best:
            best = score
        if best >= 1.0:
            break
    return best


def neighbourhood_coherence(
    ea: str,
    eb: str,
    adj_a: dict[str, set[str]],
    adj_b: dict[str, set[str]],
    threshold: float = 0.5,
) -> dict:
    """
    Compute symmetric neighbourhood coherence for a matched entity pair (ea, eb).

    Returns a dict with:
      coherence_a2b  — fraction of ea's neighbours that find a match ≥ threshold in B
      coherence_b2a  — fraction of eb's neighbours that find a match ≥ threshold in A
      coherence_sym  — symmetric average of the above
      avg_best_a2b   — mean best-match score (continuous, not thresholded)
      avg_best_b2a   — mean best-match score from B→A
      n_nbrs_a, n_nbrs_b
      best_pairs     — top-5 (na, nb, score) triples
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

    # A → B direction
    scores_a2b = []
    all_pairs: list[tuple[float, str, str]] = []
    for na in nbrs_a:
        best_score = 0.0
        best_nb = ""
        for nb in nbrs_b:
            r = characterise_entity_pair(na, nb)
            s = float(r.get("max_wup", r.get("wup_score", 0.0)))
            if s > best_score:
                best_score = s
                best_nb = nb
        scores_a2b.append(best_score)
        all_pairs.append((best_score, na, best_nb))

    # B → A direction
    scores_b2a = []
    for nb in nbrs_b:
        best_score = 0.0
        for na in nbrs_a:
            r = characterise_entity_pair(nb, na)
            s = float(r.get("max_wup", r.get("wup_score", 0.0)))
            if s > best_score:
                best_score = s
        scores_b2a.append(best_score)

    avg_a2b = sum(scores_a2b) / len(scores_a2b)
    avg_b2a = sum(scores_b2a) / len(scores_b2a)
    coh_a2b = sum(1 for s in scores_a2b if s >= threshold) / len(scores_a2b)
    coh_b2a = sum(1 for s in scores_b2a if s >= threshold) / len(scores_b2a)

    all_pairs.sort(reverse=True)

    result.update({
        "coherence_a2b": round(coh_a2b, 4),
        "coherence_b2a": round(coh_b2a, 4),
        "coherence_sym": round((coh_a2b + coh_b2a) / 2, 4),
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
    """
    Jaccard overlap of the verb/relation types used by ea (in A) and eb (in B).
    """
    va = verbs_a.get(ea, set())
    vb = verbs_b.get(eb, set())
    if not va and not vb:
        return 0.0
    union = va | vb
    inter = va & vb
    return round(len(inter) / len(union), 4)


# ---------------------------------------------------------------------------
# Match loader / fallback
# ---------------------------------------------------------------------------

def _load_matches_from_csv(csv_path: Path) -> list[dict]:
    """Load L1 matches from an existing enriched per-pair CSV."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("layer") == "1":
                rows.append(r)
    return rows


def _compute_matches_simple(
    data_a: dict, data_b: dict,
    name_a: str, name_b: str,
) -> list[dict]:
    """
    Fallback: run AML to get L1 matches if no CSV is available.
    """
    from aml_runner import AMLRunner
    print("[Matches] No CSV provided — running AML to get L1 matches ...")
    aml_maps = AMLRunner().match_jsons(data_a, data_b, name_a=name_a, name_b=name_b)
    from enriched_matcher import layer1_annotate
    return layer1_annotate(aml_maps, source_label="AML")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    json_a: Path | str,
    json_b: Path | str,
    matches_csv: Optional[Path] = None,
    out_csv: Optional[Path] = None,
    threshold: float = 0.5,
) -> list[dict]:
    """
    Full neighbourhood coherence analysis for any two ontology JSON files.

    Parameters
    ----------
    json_a, json_b  : paths to the two model JSON files
    matches_csv     : optional path to a pre-computed enriched per-pair CSV
    out_csv         : optional path to write results CSV
    threshold       : WUP threshold for counting a neighbour as "covered"

    Returns
    -------
    List of result dicts sorted by coherence_sym descending.
    """
    path_a = Path(json_a)
    path_b = Path(json_b)

    data_a = json.loads(path_a.read_text(encoding="utf-8"))
    data_b = json.loads(path_b.read_text(encoding="utf-8"))

    name_a = data_a.get("modelName", path_a.stem)
    name_b = data_b.get("modelName", path_b.stem)

    print(f"\n{'='*70}")
    print(f"Neighbourhood Coherence Analysis")
    print(f"  A: {name_a}")
    print(f"  B: {name_b}")
    print(f"{'='*70}")

    # Build graphs
    adj_a   = build_adjacency(data_a)
    adj_b   = build_adjacency(data_b)
    verbs_a = build_edge_verbs(data_a)
    verbs_b = build_edge_verbs(data_b)

    print(f"  Graph A: {len(data_a.get('entities',[]))} entities, "
          f"{len(adj_a)} with edges, "
          f"{sum(len(v) for v in adj_a.values())} edge-endpoints")
    print(f"  Graph B: {len(data_b.get('entities',[]))} entities, "
          f"{len(adj_b)} with edges, "
          f"{sum(len(v) for v in adj_b.values())} edge-endpoints")

    # Load or compute matches
    if matches_csv and Path(matches_csv).exists():
        match_rows = _load_matches_from_csv(Path(matches_csv))
        print(f"  Loaded {len(match_rows)} L1 matches from {Path(matches_csv).name}")
    else:
        match_rows = _compute_matches_simple(data_a, data_b, name_a, name_b)
        print(f"  Computed {len(match_rows)} L1 matches via AML")

    if not match_rows:
        print("  [WARN] No matches found.")
        return []

    # Pre-load ConceptNet once
    print(f"\n[CN] Pre-loading ConceptNet CSV ...")
    _cn_load_csv(_CN_CSV)

    # Compute coherence for each matched pair
    print(f"\n[Coherence] Analysing {len(match_rows)} pairs ...\n")
    results = []
    for row in match_rows:
        ea = row["entity_a"]
        eb = row["entity_b"]
        sem = row.get("semantic_label", "")
        wup = row.get("wup_score", "")
        max_wup = row.get("max_wup", wup)
        avg_wup = row.get("avg_wup", wup)

        coh = neighbourhood_coherence(ea, eb, adj_a, adj_b, threshold=threshold)
        vcoh = verb_coherence(ea, eb, verbs_a, verbs_b)

        results.append({
            "entity_a":       ea,
            "entity_b":       eb,
            "semantic_label": sem,
            "wup_score":      wup,
            "max_wup":        max_wup,
            "avg_wup":        avg_wup,
            "n_nbrs_a":       coh["n_nbrs_a"],
            "n_nbrs_b":       coh["n_nbrs_b"],
            "coherence_a2b":  coh["coherence_a2b"],
            "coherence_b2a":  coh["coherence_b2a"],
            "coherence_sym":  coh["coherence_sym"],
            "avg_best_a2b":   coh["avg_best_a2b"],
            "avg_best_b2a":   coh["avg_best_b2a"],
            "verb_coherence": vcoh,
            "best_pairs":     " | ".join(f"{nb}<->{nc}({s})"
                                         for s, nb, nc in coh["best_pairs"]),
        })

    results.sort(key=lambda r: (r["coherence_sym"], r["avg_best_a2b"]), reverse=True)

    # Print ranked table
    _print_results(results, name_a, name_b, threshold)

    # Write CSV
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
          f"{'WUP':>5} {'MaxW':>5} {'Coh_A2B':>8} {'Coh_B2A':>8} "
          f"{'Coh_sym':>8} {'VerbCoh':>8}  Best neighbour pairs")
    print("-" * 160)

    for r in results:
        flag = " *" if r["coherence_sym"] >= threshold else (
               " ?" if r["n_nbrs_a"] == 0 or r["n_nbrs_b"] == 0 else "")
        print(
            f"{r['entity_a']:<{w}} {r['entity_b']:<{w}} "
            f"{r['semantic_label']:<14} "
            f"{str(r['wup_score']):>5} {str(r['max_wup']):>5} "
            f"{r['coherence_a2b']:>8.3f} {r['coherence_b2a']:>8.3f} "
            f"{r['coherence_sym']:>8.3f} {r['verb_coherence']:>8.3f}"
            f"  {r['best_pairs'][:70]}{flag}"
        )

    # Summary
    high  = [r for r in results if r["coherence_sym"] >= threshold]
    iso_a = [r for r in results if r["n_nbrs_a"] == 0]
    iso_b = [r for r in results if r["n_nbrs_b"] == 0]
    low   = [r for r in results if r["coherence_sym"] < threshold
             and r["n_nbrs_a"] > 0 and r["n_nbrs_b"] > 0]

    print(f"\n{'-'*70}")
    print(f"  Total L1 matches: {len(results)}")
    print(f"  Neighbourhood-supported (sym >= {threshold}): {len(high)}  *")
    print(f"  Low coherence (both have neighbours):        {len(low)}")
    print(f"  Isolated in A (no edges):                   {len(iso_a)}")
    print(f"  Isolated in B (no edges):                   {len(iso_b)}")

    if low:
        print(f"\n  [!] Low-coherence matches (name match but neighbourhood disagrees):")
        for r in sorted(low, key=lambda x: x["coherence_sym"]):
            print(f"    {r['entity_a']:<26} <-> {r['entity_b']:<26}  "
                  f"coh={r['coherence_sym']:.3f}  "
                  f"A-nbrs={set(r['best_pairs'].split(' | ')[0].split('<->')[0]) if r['best_pairs'] else '{}'}")


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
        description="Neighbourhood coherence analysis for any ontology pair."
    )
    parser.add_argument("--a", required=True, help="Path to ontology A JSON")
    parser.add_argument("--b", required=True, help="Path to ontology B JSON")
    parser.add_argument("--matches-csv", default=None,
                        help="Pre-computed enriched per-pair CSV (L1 matches).")
    parser.add_argument("--out-csv", default=None,
                        help="Output CSV path for coherence results.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="WUP threshold to count a neighbour as 'covered' (default 0.5).")
    args = parser.parse_args()

    run_analysis(
        json_a       = args.a,
        json_b       = args.b,
        matches_csv  = args.matches_csv,
        out_csv      = args.out_csv,
        threshold    = args.threshold,
    )
