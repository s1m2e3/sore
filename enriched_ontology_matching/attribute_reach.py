"""
attribute_reach.py
------------------
Attribute-reach similarity: how similar are two entities based on the
observable/measurable types they carry directly or inherit through structural
associations up to K hops?

Why this matters
----------------
V-models declare typed attributes on entities (Engine.horsepower: Power).
Net-models express the same domain via explicit associations and carry no
attributes.  Raw attribute comparison fails for V↔Net pairs: Net entities
have empty attribute vectors, so similarity is always zero regardless of
actual semantic relatedness.

Two-tier reach
--------------
Tier 1 — Structural propagation (applies to both model types)
    BFS through PartOf / Connects / UsedFor associations accumulates the
    observable types of connected entities, weighted by hop distance and
    edge canonical type.  Weight for a path = product of edge weights along
    it; the MAX across all paths to the same type is kept (no inflation).

    Canonical edge weights (tunable via hop_weights param):
        PartOf / HasA / MadeOf → 0.6  (composition: attributes of parts highly relevant)
        Connects / UsedFor      → 0.4  (functional coupling: partially relevant)
        CapableOf / Causes / IsA→ 0.25 (causal/role: weaker)
        RelatedTo / AtLocation  → skip (too vague to propagate)

Tier 2 — Name-based imputation (Net-model fallback)
    For entities whose reach vector is STILL empty after structural propagation
    (no attributed entity is reachable within K hops), the entity name is
    embedded via sentence-transformer and compared against each observable type
    in the shared vocabulary.  Types above a cosine threshold are imputed as
    soft attributes at base weight 0.3 (lower confidence, reflecting that this
    is inference from name semantics, not a declared attribute).

    The shared vocabulary is collected directly from the JSON models:
      - data["observables"]  — explicitly declared observable types
      - entityAttribute.type — attribute types across all entities in both models
    Nothing is invented; every type is traceable to a JSON declaration.

Similarity
----------
Weighted Jaccard between two reach signatures (continuous [0, 1]):
    sim(A, B) = Σ min(w_a, w_b) / Σ max(w_a, w_b)  for all observable types

This degrades gracefully: when both sides have rich signatures the score is
informative; when one side is imputation-only the weights are lower (0.3×),
dragging the score down — correctly reflecting lower evidential confidence.

Usage (standalone)
------------------
    python enriched_ontology_matching/attribute_reach.py \\
        --pair enriched_ontology_matching/pairs/auto_V1_vs_brew_Net1.json \\
        --matches-csv enriched_ontology_matching/outputs/enriched/auto_V1_vs_brew_Net1.csv \\
        --hops 2

Usage (from other modules)
--------------------------
    from attribute_reach import attribute_reach, attribute_reach_similarity, collect_obs_vocab

    vocab  = collect_obs_vocab(data_a, data_b)
    reach_a = attribute_reach(data_a, obs_vocab=vocab, K=2)
    reach_b = attribute_reach(data_b, obs_vocab=vocab, K=2)
    sim = attribute_reach_similarity("Engine", "Motor", reach_a, reach_b)
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

import numpy as np

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

from model_normalizer import load_inventory, normalize_model

# ── Default hop weights by canonical relation type ────────────────────────────

_DEFAULT_HOP_WEIGHTS: dict[str, float] = {
    # Strong structural / compositional
    "PartOf":         0.6,
    "HasA":           0.6,
    "MadeOf":         0.5,
    # Functional coupling
    "Connects":       0.4,
    "UsedFor":        0.4,
    # Causal / role / capability
    "CapableOf":      0.25,
    "Causes":         0.25,
    "IsA":            0.25,
    "ReceivesAction": 0.25,   # receiver is subject to the action's physics
    "HasPrerequisite":0.20,   # functional dependency: B's observables flow to A
    # Spatial co-location: things sharing a location share environmental observables
    # (temperature, pressure, vibration).  27% of inventory — too large to skip.
    "AtLocation":     0.30,
    # RelatedTo is the catch-all (0.7% of inventory, uncategorised) — excluded
}

# ── Sentence encoder ──────────────────────────────────────────────────────────

_ENCODER       = None
_ENCODER_MODEL = None
_DEFAULT_MODEL = "paraphrase-MiniLM-L6-v2"


def _get_encoder(model_name: str = _DEFAULT_MODEL):
    global _ENCODER, _ENCODER_MODEL
    if _ENCODER is None or _ENCODER_MODEL != model_name:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _ENCODER       = SentenceTransformer(model_name, device=device)
            _ENCODER_MODEL = model_name
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "pip install sentence-transformers"
            )
    return _ENCODER


_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _readable(name: str) -> str:
    return _CAMEL_RE.sub(" ", name).lower()


# ── Shared observable vocabulary ──────────────────────────────────────────────

def collect_obs_vocab(data_a: dict, data_b: dict) -> list[str]:
    """
    Collect the shared observable-type vocabulary from both model JSONs.

    Sources (all traceable to JSON declarations):
      1. data["observables"]          — explicitly declared observable types
      2. entityAttribute.type values  — attribute types from entity definitions

    Entity names are excluded so that compositional references (Engine.engine:
    Engine) are not counted as observables.
    """
    entity_names: set[str] = set()
    for data in (data_a, data_b):
        for e in data.get("entities", []):
            n = e.get("entityName") or e.get("name", "")
            if n:
                entity_names.add(n)

    vocab: set[str] = set()
    for data in (data_a, data_b):
        vocab.update(data.get("observables", []))
        for ent in data.get("entities", []):
            for attr in (ent.get("entityAttributes") or ent.get("attributes") or []):
                t = attr.get("type", "")
                if t and t not in entity_names:
                    vocab.add(t)

    vocab.discard("")
    return sorted(vocab)


# ── Direct attribute collection ───────────────────────────────────────────────

def _direct_attrs(data: dict, entity_names: set[str]) -> dict[str, dict[str, float]]:
    """entity_name → {observable_type: 1.0} for directly declared attributes."""
    result: dict[str, dict[str, float]] = {}
    for ent in data.get("entities", []):
        name = ent.get("entityName") or ent.get("name", "")
        if not name:
            continue
        attrs: dict[str, float] = {}
        for attr in (ent.get("entityAttributes") or ent.get("attributes") or []):
            t = attr.get("type", "")
            if t and t not in entity_names:   # exclude entity-type references
                attrs[t] = 1.0
        result[name] = attrs
    return result


# ── Hop adjacency ─────────────────────────────────────────────────────────────

def _hop_adj(
    data: dict,
    hop_weights: dict[str, float],
) -> dict[str, list[tuple[str, float]]]:
    """
    entity → [(neighbor_entity, edge_weight)] for propagation-eligible edges.

    Requires data to have been passed through normalize_model() so every
    association carries a ``canonical`` field.  Associations whose canonical
    is not in hop_weights (or has weight 0) are silently skipped.
    """
    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for assoc in data.get("associations", []):
        canonical = assoc.get("canonical", "RelatedTo")
        w = hop_weights.get(canonical, 0.0)
        if w <= 0.0:
            continue
        parts = assoc.get("associationParticipants") or assoc.get("participants") or []
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                adj[parts[i]].append((parts[j], w))
                adj[parts[j]].append((parts[i], w))
    return dict(adj)


# ── Name-based attribute imputation ──────────────────────────────────────────

def _impute_from_name(
    entity_name:  str,
    obs_vocab:    list[str],
    encoder,
    threshold:    float = 0.30,
    base_weight:  float = 0.30,
) -> dict[str, float]:
    """
    For entities with no reachable attributes, infer a soft attribute signature
    by comparing the entity name embedding against each observable type embedding.

    Returns {observable_type: cosine_sim * base_weight} for types above threshold.
    The base_weight (default 0.3) marks these as lower-confidence than declared
    or structurally propagated attributes (weight 1.0 / 0.6 / 0.4).
    """
    if not obs_vocab:
        return {}
    all_texts = [_readable(entity_name)] + [_readable(o) for o in obs_vocab]
    embs = encoder.encode(
        all_texts, normalize_embeddings=True,
        show_progress_bar=False, batch_size=64,
    )
    name_emb = embs[0]
    obs_embs = embs[1:]
    sims     = obs_embs @ name_emb       # cosine similarity (unit vectors)

    return {
        obs: round(float(sim) * base_weight, 4)
        for obs, sim in zip(obs_vocab, sims)
        if sim > threshold
    }


# ── Main reach computation ────────────────────────────────────────────────────

def attribute_reach(
    data:         dict,
    obs_vocab:    list[str],
    K:            int  = 2,
    hop_weights:  dict[str, float] | None = None,
    impute_threshold: float = 0.30,
    impute_base_weight: float = 0.30,
    encoder_model: str = _DEFAULT_MODEL,
) -> dict[str, dict[str, float]]:
    """
    Compute the K-hop attribute reach for every entity in one model JSON.

    Parameters
    ----------
    data          : normalized model JSON (run normalize_model() first).
    obs_vocab     : shared observable type vocabulary (from collect_obs_vocab).
    K             : maximum hop count (1 or 2 recommended).
    hop_weights   : {canonical: multiplier}; defaults to _DEFAULT_HOP_WEIGHTS.
    impute_threshold : cosine threshold for name-based imputation (Tier 2).
    impute_base_weight : confidence weight for imputed attributes (0–1).
    encoder_model : sentence-transformer model name.

    Returns
    -------
    {entity_name: {observable_type: accumulated_weight}}
    Direct attributes have weight 1.0; propagated attributes decay by the
    product of edge weights along the path; imputed attributes are capped at
    impute_base_weight.
    """
    if hop_weights is None:
        hop_weights = _DEFAULT_HOP_WEIGHTS

    entity_names: set[str] = {
        e.get("entityName") or e.get("name", "")
        for e in data.get("entities", [])
    }
    entity_names.discard("")

    direct = _direct_attrs(data, entity_names)
    adj    = _hop_adj(data, hop_weights)

    # Initialise reach with direct attributes
    reach: dict[str, dict[str, float]] = {
        e: dict(direct.get(e, {})) for e in entity_names
    }

    # K-hop BFS propagation — read from previous-hop state, write to new_reach
    for _ in range(K):
        new_reach: dict[str, dict[str, float]] = {e: dict(r) for e, r in reach.items()}
        for entity in entity_names:
            for neighbor, edge_w in adj.get(entity, []):
                for obs_type, nbr_w in reach.get(neighbor, {}).items():
                    propagated = edge_w * nbr_w
                    # Keep the maximum-weight path to each observable type
                    if propagated > new_reach[entity].get(obs_type, 0.0):
                        new_reach[entity][obs_type] = round(propagated, 4)
        reach = new_reach

    # Tier 2: name-based imputation for entities with empty reach
    empty_entities = [e for e in entity_names if not reach.get(e)]
    if empty_entities and obs_vocab:
        enc = _get_encoder(encoder_model)
        print(
            f"[AttrReach] Imputing attributes for {len(empty_entities)} "
            f"entity/entities with no reachable observables ...",
            file=sys.stderr,
        )
        for entity in empty_entities:
            imputed = _impute_from_name(
                entity, obs_vocab, enc, impute_threshold, impute_base_weight,
            )
            if imputed:
                reach[entity] = imputed

    return reach


# ── Similarity ────────────────────────────────────────────────────────────────

def attribute_reach_similarity(
    ea:      str,
    eb:      str,
    reach_a: dict[str, dict[str, float]],
    reach_b: dict[str, dict[str, float]],
) -> float:
    """
    Weighted Jaccard similarity between the attribute reach signatures of ea and eb.

    sim = Σ min(w_a, w_b) / Σ max(w_a, w_b)  for all observable types.

    Returns 0.0 when both signatures are empty (metric not applicable).
    The caller can detect this case by checking that both reach vectors are {}.
    """
    sig_a = reach_a.get(ea, {})
    sig_b = reach_b.get(eb, {})

    if not sig_a and not sig_b:
        return 0.0

    all_types  = set(sig_a) | set(sig_b)
    numerator  = sum(min(sig_a.get(t, 0.0), sig_b.get(t, 0.0)) for t in all_types)
    denominator = sum(max(sig_a.get(t, 0.0), sig_b.get(t, 0.0)) for t in all_types)

    return round(numerator / denominator, 4) if denominator > 0 else 0.0


# ── Standalone analysis ───────────────────────────────────────────────────────

def run_analysis(
    pair_json:   Path | str,
    matches_csv: Path | str | None = None,
    K:           int  = 2,
    out_csv:     Path | str | None = None,
) -> list[dict]:
    """
    Compute attribute-reach similarity for all matched entity pairs in a comparison JSON.

    Parameters
    ----------
    pair_json   : path to a {json_a, json_b} pair file.
    matches_csv : pre-computed enriched CSV with entity_a / entity_b columns.
                  If None, all entity cross-products are evaluated.
    K           : hop depth (1 or 2).
    out_csv     : optional path to write results.

    Returns list of dicts with keys:
        entity_a, entity_b, attr_reach_sim,
        n_obs_a (direct), n_obs_b (direct),
        n_reach_a (total reach), n_reach_b (total reach),
        imputed_a (bool), imputed_b (bool)
    """
    p = Path(pair_json)
    combined = json.loads(p.read_text(encoding="utf-8"))
    raw_a    = combined.get("json_a", combined)
    raw_b    = combined.get("json_b", {})
    name_a   = raw_a.get("modelName", p.stem + "_a")
    name_b   = raw_b.get("modelName", p.stem + "_b")

    inv    = load_inventory()
    data_a = normalize_model(raw_a, inv)
    data_b = normalize_model(raw_b, inv)

    vocab = collect_obs_vocab(data_a, data_b)
    print(f"[AttrReach] Shared observable vocabulary: {len(vocab)} types", file=sys.stderr)
    print(f"[AttrReach] Computing {K}-hop reach for '{name_a}' ...", file=sys.stderr)
    reach_a = attribute_reach(data_a, vocab, K=K)
    print(f"[AttrReach] Computing {K}-hop reach for '{name_b}' ...", file=sys.stderr)
    reach_b = attribute_reach(data_b, vocab, K=K)

    # Determine entity pairs to score
    if matches_csv and Path(matches_csv).exists():
        with open(matches_csv, newline="", encoding="utf-8") as fh:
            pairs = [(r["entity_a"], r["entity_b"]) for r in csv.DictReader(fh)]
        print(f"[AttrReach] Scoring {len(pairs)} matched pairs from CSV.", file=sys.stderr)
    else:
        ents_a = [e.get("entityName") or e.get("name", "")
                  for e in data_a.get("entities", []) if e.get("entityName") or e.get("name")]
        ents_b = [e.get("entityName") or e.get("name", "")
                  for e in data_b.get("entities", []) if e.get("entityName") or e.get("name")]
        pairs = [(ea, eb) for ea in ents_a for eb in ents_b]
        print(f"[AttrReach] Scoring all {len(pairs)} cross-product pairs.", file=sys.stderr)

    # Collect direct-attribute counts for diagnostics
    entity_names_a = {
        e.get("entityName") or e.get("name", "")
        for e in data_a.get("entities", [])
    }
    entity_names_a.discard("")
    direct_a = _direct_attrs(data_a, entity_names_a)

    entity_names_b = {
        e.get("entityName") or e.get("name", "")
        for e in data_b.get("entities", [])
    }
    entity_names_b.discard("")
    direct_b = _direct_attrs(data_b, entity_names_b)

    results = []
    for ea, eb in pairs:
        sim = attribute_reach_similarity(ea, eb, reach_a, reach_b)
        results.append({
            "entity_a":       ea,
            "entity_b":       eb,
            "attr_reach_sim": sim,
            "n_obs_a":        len(direct_a.get(ea, {})),
            "n_obs_b":        len(direct_b.get(eb, {})),
            "n_reach_a":      len(reach_a.get(ea, {})),
            "n_reach_b":      len(reach_b.get(eb, {})),
            "imputed_a":      len(direct_a.get(ea, {})) == 0 and len(reach_a.get(ea, {})) > 0,
            "imputed_b":      len(direct_b.get(eb, {})) == 0 and len(reach_b.get(eb, {})) > 0,
        })

    results.sort(key=lambda r: -r["attr_reach_sim"])

    if out_csv:
        _write_csv(results, Path(out_csv))

    return results


# ── Anonymous attribute reach distribution stage ─────────────────────────────

def run_attr_dist_stage(
    data_a:        dict,
    data_b:        dict,
    out_csv:       Path,
    K:             int   = 2,
    encoder_model: str   = _DEFAULT_MODEL,
) -> float:
    """
    Compute anonymous embedded attribute reach distribution similarity for one
    ontology pair and write a single-row summary CSV.

    Match-independent, name-independent:
      1. Collect union observable-type vocabulary from both normalized models.
      2. Embed each type name with sentence-transformers.
      3. For every entity (anonymous — names never used as keys in output),
         compute its K-hop reach signature via BFS through PartOf/Connects edges.
      4. Sum reach-weighted type embeddings over ALL entities → one dense
         vector per model.
      5. attr_dist_sim = cosine(agg_A, agg_B).

    Imputation is disabled: only declared and structurally propagated attributes
    contribute, keeping the metric purely structural/semantic rather than
    name-dependent.

    Returns attr_dist_sim (float, 0.0 when no vocab exists).
    """
    import re
    import numpy as np

    inv    = load_inventory()
    norm_a = normalize_model(data_a, inv)
    norm_b = normalize_model(data_b, inv)

    vocab = collect_obs_vocab(norm_a, norm_b)
    if not vocab:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=["attr_dist_sim"]).writeheader()
            csv.DictWriter(fh, fieldnames=["attr_dist_sim"]).writerow({"attr_dist_sim": 0.0})
        return 0.0

    # BFS reach with imputation disabled (threshold > 1 never fires)
    reach_a = attribute_reach(norm_a, vocab, K=K, impute_threshold=1.1)
    reach_b = attribute_reach(norm_b, vocab, K=K, impute_threshold=1.1)

    # Embed observable type names
    readable = [" ".join(re.findall(r'[A-Z][a-z]*|[a-z]+', t)) or t for t in vocab]
    enc       = _get_encoder(encoder_model)
    type_embs = enc.encode(readable, normalize_embeddings=True,
                            show_progress_bar=False, batch_size=64)
    type_emb_map = {t: type_embs[i] for i, t in enumerate(vocab)}
    dim = type_embs.shape[1]

    def _agg(reach: dict) -> np.ndarray:
        agg = np.zeros(dim, dtype=np.float64)
        for sig in reach.values():
            for obs_type, weight in sig.items():
                if obs_type in type_emb_map:
                    agg += weight * type_emb_map[obs_type]
        return agg

    agg_a = _agg(reach_a)
    agg_b = _agg(reach_b)
    na, nb = np.linalg.norm(agg_a), np.linalg.norm(agg_b)
    sim = round(float(np.dot(agg_a, agg_b) / (na * nb)), 4) if na > 1e-10 and nb > 1e-10 else 0.0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["attr_dist_sim"])
        w.writeheader()
        w.writerow({"attr_dist_sim": sim})

    print(f"[AttrDist] attr_dist_sim={sim:.4f}  vocab={len(vocab)}  "
          f"entities_a={len(reach_a)}  entities_b={len(reach_b)}", file=sys.stderr)
    return sim


# ── WUP-based anonymous attribute reach distribution ─────────────────────────

def run_attr_wup_stage(
    data_a: dict,
    data_b: dict,
    out_csv: Path,
    K: int = 2,
) -> float:
    """
    WUP-based anonymous attribute reach distribution similarity.

    Replaces BERT embedding cosine with WordNet Wu-Palmer similarity so that
    semantically distinct observable types (Power vs BloodPressure) score near
    zero while genuinely related types (Power vs HorsePower) get partial credit.

    Algorithm:
      1. Collect union observable-type vocab from both normalized models.
      2. Compute K-hop BFS reach for all entities in each model (imputation off).
      3. Aggregate total reach weight per observable type → freq_A, freq_B.
      4. For each type_a: best_wup = max WUP against all types in vocab_B.
         Weighted soft recall A→B = Σ freq_A[t] × best_wup(t) / Σ freq_A[t]
      5. Same symmetrically B→A.
      6. attr_wup_sim = (recall_A→B + recall_B→A) / 2.

    Returns 0.0 when neither model has observable types.
    """
    from root_comparator import best_wup as _best_wup

    inv    = load_inventory()
    norm_a = normalize_model(data_a, inv)
    norm_b = normalize_model(data_b, inv)

    vocab = collect_obs_vocab(norm_a, norm_b)
    if not vocab:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["attr_wup_sim"])
            w.writeheader(); w.writerow({"attr_wup_sim": 0.0})
        return 0.0

    reach_a = attribute_reach(norm_a, vocab, K=K, impute_threshold=1.1)
    reach_b = attribute_reach(norm_b, vocab, K=K, impute_threshold=1.1)

    # Aggregate total reach weight per observable type (anonymous — entity names dropped)
    def _freq(reach: dict) -> dict[str, float]:
        f: dict[str, float] = {}
        for sig in reach.values():
            for t, w in sig.items():
                f[t] = f.get(t, 0.0) + w
        return f

    freq_a = _freq(reach_a)
    freq_b = _freq(reach_b)

    if not freq_a or not freq_b:
        sim = 0.0
    else:
        vocab_a = list(freq_a.keys())
        vocab_b = list(freq_b.keys())

        # Precompute WUP matrix between all type pairs (vocab small: ~10-60 types)
        wup_cache: dict[tuple[str, str], float] = {}
        for ta in vocab_a:
            for tb in vocab_b:
                key = (ta.lower(), tb.lower())
                if key not in wup_cache:
                    if ta.lower() == tb.lower():
                        wup_cache[key] = 1.0
                    else:
                        wup_cache[key] = _best_wup(ta.lower(), tb.lower())[0]

        # Weighted soft recall A→B
        total_a = sum(freq_a.values())
        recall_ab = sum(
            freq_a[ta] * max((wup_cache.get((ta.lower(), tb.lower()), 0.0) for tb in vocab_b), default=0.0)
            for ta in vocab_a
        ) / total_a if total_a > 0 else 0.0

        # Weighted soft recall B→A
        total_b = sum(freq_b.values())
        recall_ba = sum(
            freq_b[tb] * max((wup_cache.get((ta.lower(), tb.lower()), 0.0) for ta in vocab_a), default=0.0)
            for tb in vocab_b
        ) / total_b if total_b > 0 else 0.0

        sim = round((recall_ab + recall_ba) / 2, 4)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["attr_wup_sim"])
        w.writeheader(); w.writerow({"attr_wup_sim": sim})

    print(f"[AttrWUP] attr_wup_sim={sim:.4f}  vocab_a={len(freq_a)}  vocab_b={len(freq_b)}",
          file=sys.stderr)
    return sim


# ── Output helpers ────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "entity_a", "entity_b", "attr_reach_sim",
    "n_obs_a", "n_obs_b", "n_reach_a", "n_reach_b",
    "imputed_a", "imputed_b",
]


def _write_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"[AttrReach] Written {len(results)} rows → {path}", file=sys.stderr)


def _print_results(results: list[dict], top_n: int = 30) -> None:
    w = 30
    print(f"\n{'Entity A':<{w}} {'Entity B':<{w}} {'Sim':>6}  "
          f"{'ObsA':>5} {'ReachA':>7}  {'ObsB':>5} {'ReachB':>7}  Imputed")
    print("-" * 105)
    for r in results[:top_n]:
        imp = ("A" if r["imputed_a"] else "") + ("B" if r["imputed_b"] else "")
        print(
            f"{r['entity_a']:<{w}} {r['entity_b']:<{w}} "
            f"{r['attr_reach_sim']:>6.4f}  "
            f"{r['n_obs_a']:>5} {r['n_reach_a']:>7}  "
            f"{r['n_obs_b']:>5} {r['n_reach_b']:>7}  {imp}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Attribute-reach similarity: entity similarity via typed observable "
            "attributes propagated through structural associations (K hops). "
            "Handles V-models (explicit attributes) and Net-models (no attributes) "
            "via name-based imputation for empty reach vectors."
        )
    )
    ap.add_argument("--pair",        required=True,
                    help="Path to pair JSON with json_a / json_b keys.")
    ap.add_argument("--matches-csv", default=None,
                    help="Pre-computed enriched CSV; if omitted all pairs are scored.")
    ap.add_argument("--hops",        type=int, default=2,
                    help="Maximum hop depth (default: 2).")
    ap.add_argument("--out-csv",     default=None,
                    help="Output CSV path.")
    ap.add_argument("--top",         type=int, default=30,
                    help="Rows to print (default: 30).")
    args = ap.parse_args()

    results = run_analysis(
        pair_json   = args.pair,
        matches_csv = args.matches_csv,
        K           = args.hops,
        out_csv     = args.out_csv,
    )
    _print_results(results, top_n=args.top)
