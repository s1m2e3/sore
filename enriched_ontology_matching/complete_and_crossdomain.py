"""
complete_and_crossdomain.py
---------------------------
1. Completes the enriched pipeline (embeddings + re-merge) for
   hospital/university pairs whose enriched CSV already exists.
2. For every other pair (cross-domain, missing within-hospital) runs a
   pure-semantic pipeline: cosine similarity of entity-name embeddings
   finds candidate matches.
3. Regenerates outputs/ontology_map.html via compare_stage.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PAIRS_DIR   = ROOT / "pairs"
ENRICHED    = ROOT / "outputs" / "enriched"
NBR_DIR     = ROOT / "outputs" / "neighbourhood"
EMB_DIR     = ROOT / "outputs" / "embeddings"
MERGED_DIR  = ROOT / "outputs" / "merged"

for d in (PAIRS_DIR, ENRICHED, NBR_DIR, EMB_DIR, MERGED_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model extraction
# ---------------------------------------------------------------------------

def extract_models() -> dict[str, dict]:
    """Return {modelName: full_json_dict} from all known source files."""
    # Individual model files (standalone JSON, not pair-wrapped)
    models_dir = ROOT / "models"
    individual_files = list(models_dir.rglob("*.json")) if models_dir.exists() else []

    # Pair/test files (json_a / json_b wrapped)
    pair_files = list(ROOT.glob("test_*.json")) + list(PAIRS_DIR.glob("*.json"))

    models: dict[str, dict] = {}

    # Load standalone model files first (highest priority)
    for f in individual_files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Standalone: the file IS the model (not wrapped)
        name = d.get("modelName", "")
        if name and name not in models:
            models[name] = d

    # Load from pair files (lower priority — don't overwrite standalone)
    for f in pair_files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for side in ("json_a", "json_b"):
            m = d.get(side, {})
            name = m.get("modelName", "")
            if name and name not in models:
                models[name] = m
    return models


def safe_stem(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def entity_names(model: dict) -> list[str]:
    result = []
    for e in model.get("entities", []):
        name = e.get("entityName") or e.get("name") or ""
        if name:
            result.append(name)
    return result


def adjacency(model: dict) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {}
    for assoc in model.get("associations", []):
        src = assoc.get("source", "")
        tgt = assoc.get("target", "")
        if src and tgt:
            adj.setdefault(src, set()).add(tgt)
            adj.setdefault(tgt, set()).add(src)
    return adj


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

_EMB_MODEL = None

def get_emb_model():
    global _EMB_MODEL
    if _EMB_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMB_MODEL = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    return _EMB_MODEL


def embed_names(names: list[str]):
    import numpy as np
    model = get_emb_model()
    # readable form: CamelCase -> spaces
    import re
    readable = [re.sub(r"(?<=[a-z])(?=[A-Z])", " ", n) for n in names]
    embs = model.encode(readable, normalize_embeddings=True)
    return embs  # shape (N, D)


# ---------------------------------------------------------------------------
# Stage runners (import from existing modules)
# ---------------------------------------------------------------------------

def run_embeddings_on_pairs(pairs: list[tuple[str, str]], stem: str) -> Path:
    """Run sentence-embedding cosine for each pair and save CSV."""
    import numpy as np

    EMB_FIELDS = ["entity_a", "entity_b", "cosine_whole", "cosine_avg"]
    model = get_emb_model()

    import re
    def readable(n): return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", n)

    all_names = list({n for pair in pairs for n in pair})
    embs_arr  = model.encode([readable(n) for n in all_names],
                              normalize_embeddings=True)
    emb_map   = dict(zip(all_names, embs_arr))

    rows = []
    for ea, eb in pairs:
        va, vb = emb_map.get(ea), emb_map.get(eb)
        if va is not None and vb is not None:
            c = (float(np.dot(va, vb)) + 1.0) / 2.0
        else:
            c = 0.5
        rows.append({"entity_a": ea, "entity_b": eb,
                     "cosine_whole": round(c, 4), "cosine_avg": round(c, 4)})

    out = EMB_DIR / f"{stem}_emb.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=EMB_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out


# ---------------------------------------------------------------------------
# Complete existing enriched CSVs (hospital / university)
# ---------------------------------------------------------------------------

def complete_existing_pairs():
    from merge_stage import merge_pair

    stems_done = set()
    for enriched_csv in ENRICHED.glob("*.csv"):
        if enriched_csv.stem == "all_domains_combined":
            continue
        stem = enriched_csv.stem
        merged_out = MERGED_DIR / f"{stem}_metrics.csv"

        # Check if already fully enriched
        if merged_out.exists():
            rows = list(csv.DictReader(open(merged_out, encoding="utf-8")))
            if rows and rows[0].get("cosine_avg", "").strip():
                print(f"  [skip] already complete: {stem[:60]}")
                stems_done.add(stem)
                continue

        # Read candidate pairs from enriched CSV
        enriched_rows = list(csv.DictReader(open(enriched_csv, encoding="utf-8")))
        pairs = [(r["entity_a"], r["entity_b"]) for r in enriched_rows]
        if not pairs:
            continue

        print(f"  [complete] {stem[:70]}")

        emb_csv = EMB_DIR / f"{stem}_emb.csv"

        if not emb_csv.exists():
            emb_csv = run_embeddings_on_pairs(pairs, stem)

        merge_pair(
            enriched_csv=enriched_csv,
            nbr_csv=NBR_DIR / f"{stem}_coherence.csv" if (NBR_DIR / f"{stem}_coherence.csv").exists() else None,
            emb_csv=emb_csv,
            out_csv=merged_out,
        )
        stems_done.add(stem)
    return stems_done


# ---------------------------------------------------------------------------
# Pure-semantic pipeline for new / cross-domain pairs
# ---------------------------------------------------------------------------

TOP_K = 15   # fixed number of top pairs to keep per direction (no threshold cutoff)

NBR_FIELDS = ["entity_a", "entity_b", "coherence_a2b", "coherence_b2a",
              "coherence_sym", "n_nbrs_a", "n_nbrs_b"]


def _is_complete(merged_out: Path) -> bool:
    """Return True only if the merged CSV has cosine AND avg_wup populated."""
    if not merged_out.exists():
        return False
    try:
        rows = list(csv.DictReader(open(merged_out, encoding="utf-8")))
        return bool(rows
                    and rows[0].get("cosine_avg", "").strip()
                    and rows[0].get("avg_wup", "").strip())
    except Exception:
        return False


def _run_wup_on_pairs(pairs: list[tuple[str, str]]) -> dict[tuple, float]:
    """Compute avg_wup via WordNet for each (entity_a, entity_b) pair."""
    from enriched_matcher import characterise_entity_pair
    wup_map: dict[tuple, float] = {}
    for ea, eb in pairs:
        try:
            r = characterise_entity_pair(ea, eb)
            wup_map[(ea, eb)] = float(r.get("avg_wup", 0) or 0)
        except Exception:
            wup_map[(ea, eb)] = 0.0
    return wup_map


def _run_coherence_on_pairs(
    pairs: list[tuple[str, str]],
    model_a: dict,
    model_b: dict,
    stem: str,
) -> Path:
    """Compute neighbourhood coherence for each pair and write a CSV."""
    from neighbourhood_coherence import (
        build_adjacency, _build_emb_cache, neighbourhood_coherence,
    )

    adj_a = build_adjacency(model_a)
    adj_b = build_adjacency(model_b)

    # Build embedding cache for all neighbour entities
    all_nbr_names: set[str] = set()
    for ea, eb in pairs:
        all_nbr_names.update(adj_a.get(ea, set()))
        all_nbr_names.update(adj_b.get(eb, set()))
    emb_cache = _build_emb_cache(list(all_nbr_names))

    rows = []
    for ea, eb in pairs:
        coh = neighbourhood_coherence(ea, eb, adj_a, adj_b, emb_cache)
        rows.append({
            "entity_a":      ea,
            "entity_b":      eb,
            "coherence_a2b": coh["coherence_a2b"],
            "coherence_b2a": coh["coherence_b2a"],
            "coherence_sym": coh["coherence_sym"],
            "n_nbrs_a":      coh["n_nbrs_a"],
            "n_nbrs_b":      coh["n_nbrs_b"],
        })

    out = NBR_DIR / f"{stem}_coherence.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=NBR_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out


def semantic_pair(name_a: str, model_a: dict,
                  name_b: str, model_b: dict,
                  stem: str) -> Path:
    """
    Match entities using cosine similarity then enrich with WUP + coherence.
    Takes TOP_K best matches per direction regardless of score.
    """
    import numpy as np

    merged_out = MERGED_DIR / f"{stem}_metrics.csv"
    if _is_complete(merged_out):
        print(f"  [skip] {stem[:70]}")
        return merged_out

    ents_a = entity_names(model_a)
    ents_b = entity_names(model_b)

    print(f"  [semantic] {name_a[:28]} vs {name_b[:28]} "
          f"({len(ents_a)} x {len(ents_b)})")

    emb_a = embed_names(ents_a)
    emb_b = embed_names(ents_b)
    sim   = emb_a @ emb_b.T       # (|A|, |B|)

    seen: dict[tuple, float] = {}

    k_a = min(TOP_K, len(ents_b))
    for i, ea in enumerate(ents_a):
        top_js = np.argsort(sim[i])[::-1][:k_a]
        for j in top_js:
            key = (ea, ents_b[j])
            sc  = float(sim[i, j])
            if key not in seen or sc > seen[key]:
                seen[key] = sc

    k_b = min(TOP_K, len(ents_a))
    for j, eb in enumerate(ents_b):
        top_is = np.argsort(sim[:, j])[::-1][:k_b]
        for i in top_is:
            key = (ents_a[i], eb)
            sc  = float(sim[i, j])
            if key not in seen or sc > seen[key]:
                seen[key] = sc

    top = sorted(seen.items(), key=lambda x: -x[1])[:TOP_K * 2]
    pairs = [k for k, _ in top]

    # --- WUP ---
    print(f"    [wup] computing for {len(pairs)} pairs ...")
    wup_map = _run_wup_on_pairs(pairs)

    # --- Neighbourhood coherence ---
    print(f"    [coherence] computing for {len(pairs)} pairs ...")
    nbr_csv = _run_coherence_on_pairs(pairs, model_a, model_b, stem)
    nbr_idx = {(r["entity_a"], r["entity_b"]): r
               for r in csv.DictReader(open(nbr_csv, encoding="utf-8"))}

    # --- Enriched CSV (now includes avg_wup) ---
    enriched_out = ENRICHED / f"{stem}.csv"
    ENR_FIELDS = ["entity_a", "entity_b", "source", "matcher_conf",
                  "layer", "token_a", "token_b", "wup_score", "avg_wup",
                  "wn_relation", "wn_hops", "cn_relations", "cn_label",
                  "gloss_hit", "semantic_label", "layer2_type"]
    with open(enriched_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ENR_FIELDS)
        w.writeheader()
        for (ea, eb), sc in top:
            wup = wup_map.get((ea, eb), 0.0)
            w.writerow({
                "entity_a": ea, "entity_b": eb,
                "source": "Semantic", "matcher_conf": round(sc, 4),
                "layer": 1, "token_a": ea.lower(), "token_b": eb.lower(),
                "wup_score": round(wup, 4), "avg_wup": round(wup, 4),
                "wn_relation": "", "wn_hops": "",
                "cn_relations": "", "cn_label": "", "gloss_hit": "",
                "semantic_label": "Semantic", "layer2_type": "",
            })

    emb_csv = run_embeddings_on_pairs(pairs, stem)

    emb_idx = {(r["entity_a"], r["entity_b"]): r
               for r in csv.DictReader(open(emb_csv, encoding="utf-8"))}

    FIELDS = ["entity_a", "entity_b", "matched", "avg_wup",
              "coherence_sym", "cosine_avg"]
    merged_rows = []
    for (ea, eb), sc in top:
        key = (ea, eb)
        emb = emb_idx.get(key, {})
        nbr = nbr_idx.get(key, {})
        merged_rows.append({
            "entity_a":      ea,
            "entity_b":      eb,
            "matched":       0,
            "avg_wup":       round(wup_map.get(key, 0.0), 4),
            "coherence_sym": nbr.get("coherence_sym", ""),
            "cosine_avg":    emb.get("cosine_avg", round((sc + 1.0) / 2.0, 4)),
        })

    with open(merged_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(merged_rows)

    print(f"    {len(merged_rows)} pairs -> {merged_out.name}")
    return merged_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Step 1: Complete existing hospital/university enriched CSVs")
    print("=" * 65)
    complete_existing_pairs()

    print()
    print("=" * 65)
    print("Step 2: Load all individual ontology models")
    print("=" * 65)
    models = extract_models()
    names  = sorted(models.keys())
    print(f"  Found {len(names)} models:")
    for n in names:
        print(f"    {len(entity_names(models[n])):4d} entities | {n[:65]}")

    print()
    print("=" * 65)
    print("Step 3: Run semantic pipeline on ALL pairs")
    print("=" * 65)

    import re as _re

    def _norm(s: str) -> str:
        return _re.sub(r"_+", "_", s).strip("_").lower()

    # Build coverage map: frozenset({normA, normB}) -> True if fully enriched
    covered: set = set()
    for csv_path in MERGED_DIR.glob("*_metrics.csv"):
        stem_existing = csv_path.stem.replace("_metrics", "")
        try:
            idx = stem_existing.index("_vs_")
            na = _norm(stem_existing[:idx])
            nb = _norm(stem_existing[idx + 4:])
        except ValueError:
            continue
        if _is_complete(csv_path):
            covered.add(frozenset([na, nb]))

    all_pairs = list(itertools.combinations(names, 2))
    total = len(all_pairs)
    print(f"  Total pairs: {total}  |  Already covered: {len(covered)}")

    for idx_p, (name_a, name_b) in enumerate(all_pairs):
        key = frozenset([_norm(safe_stem(name_a)), _norm(safe_stem(name_b))])
        if key in covered:
            print(f"  [{idx_p+1}/{total}] skip  {name_a[:28]} vs {name_b[:28]}")
            continue

        stem_a = safe_stem(name_a)
        stem_b = safe_stem(name_b)
        stem   = f"{stem_a}_vs_{stem_b}"
        semantic_pair(name_a, models[name_a], name_b, models[name_b], stem)
        covered.add(key)

    print()
    print("=" * 65)
    print("Step 4: Regenerate ontology_map.html")
    print("=" * 65)
    import compare_stage
    compare_stage.main()


if __name__ == "__main__":
    main()
