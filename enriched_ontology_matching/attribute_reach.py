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
Embedding-based cosine similarity between two entities' weighted reach
vectors (continuous [0, 1]), see type_embed_similarity(). Unlike an exact
string-match Jaccard, this captures near-synonym types (Power vs HorsePower)
via embedding distance.

This degrades gracefully: when both sides have rich signatures the score is
informative; when one side is imputation-only the weights are lower (0.3×),
dragging the score down — correctly reflecting lower evidential confidence.

Usage (from other modules)
--------------------------
    from attribute_reach import attribute_reach, type_embed_similarity, collect_obs_vocab, _embed_vocab

    vocab  = collect_obs_vocab(data_a, data_b)
    reach_a = attribute_reach(data_a, obs_vocab=vocab, K=2)
    reach_b = attribute_reach(data_b, obs_vocab=vocab, K=2)
    type_emb_map, dim = _embed_vocab(vocab)
    sim = type_embed_similarity("Engine", "Motor", reach_a, reach_b, type_emb_map, dim)

Pipeline integration (called from run_all_pairs.py / enriched_matcher.py):
    run_type_embed_stage(enriched_csv, json_a, json_b, out_csv)   — per-entity-pair type_embed_sim
    run_attr_dist_stage(data_a, data_b, out_csv)                  — whole-model attr_dist_sim
    layer3_type_backup(entities_a, entities_b, covered_a, covered_b, json_a, json_b)
        — candidate-generation backup for entities with matching types but unrelated names
"""
from __future__ import annotations

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


def _embed_vocab(
    vocab: list[str],
    encoder_model: str = _DEFAULT_MODEL,
) -> tuple[dict[str, np.ndarray], int]:
    """
    Embed each observable-type name in vocab with sentence-transformers.

    Returns ({type_name: unit_vector}, embedding_dim). Shared by
    run_attr_dist_stage (whole-model aggregate) and run_type_embed_stage
    (per-entity-pair) so both score types the same way.
    """
    readable = [" ".join(re.findall(r'[A-Z][a-z]*|[a-z]+', t)) or t for t in vocab]
    enc = _get_encoder(encoder_model)
    embs = enc.encode(readable, normalize_embeddings=True,
                       show_progress_bar=False, batch_size=64)
    return {t: embs[i] for i, t in enumerate(vocab)}, embs.shape[1]


def _weighted_type_vector(
    sig: dict[str, float],
    type_emb_map: dict[str, np.ndarray],
    dim: int,
) -> np.ndarray:
    """Weighted sum of type embeddings for one entity's reach signature (unnormalised)."""
    agg = np.zeros(dim, dtype=np.float64)
    for obs_type, weight in sig.items():
        if obs_type in type_emb_map:
            agg += weight * type_emb_map[obs_type]
    return agg


def type_embed_similarity(
    ea: str,
    eb: str,
    reach_a: dict[str, dict[str, float]],
    reach_b: dict[str, dict[str, float]],
    type_emb_map: dict[str, np.ndarray],
    dim: int,
) -> Optional[float]:
    """
    Embedding-based type similarity between two entities: cosine similarity of
    their weighted attribute-type embedding vectors (direct + K-hop reach +
    name-imputed for attribute-less Net-model entities).

    Unlike an exact string-match weighted Jaccard, this captures near-synonym
    types (e.g. Power vs HorsePower) via embedding distance.

    Returns None when either entity has no observable-type evidence at all
    (zero vector) — the caller should fall back to a different signal (e.g.
    name-based similarity) in that case.
    """
    va = _weighted_type_vector(reach_a.get(ea, {}), type_emb_map, dim)
    vb = _weighted_type_vector(reach_b.get(eb, {}), type_emb_map, dim)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-10 or nb < 1e-10:
        return None
    return round(float(np.dot(va, vb) / (na * nb)), 4)


def run_type_embed_stage(
    enriched_csv: Path,
    json_a: Path,
    json_b: Path,
    out_csv: Path,
    K: int = 2,
    encoder_model: str = _DEFAULT_MODEL,
) -> list[dict]:
    """
    Per-entity-pair embedding-based attribute-type similarity stage.

    Mirrors semantic_encoder.run_embedding_stage's interface (same enriched_csv
    row set, entity_a/entity_b keyed output) but scores each pair on what
    observable TYPES the entities carry (declared + structurally propagated +
    name-imputed) rather than on the entities' own NAMES.

    Raises FileNotFoundError if enriched_csv, json_a, or json_b do not exist.
    """
    for label, p in [("enriched_csv", enriched_csv), ("json_a", json_a), ("json_b", json_b)]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    data_a = json.loads(Path(json_a).read_text(encoding="utf-8"))
    data_b = json.loads(Path(json_b).read_text(encoding="utf-8"))

    inv    = load_inventory()
    norm_a = normalize_model(data_a, inv)
    norm_b = normalize_model(data_b, inv)

    vocab = collect_obs_vocab(norm_a, norm_b)
    print(f"[TypeEmbed] Shared observable vocabulary: {len(vocab)} types", file=sys.stderr)

    with open(enriched_csv, newline="", encoding="utf-8") as fh:
        rows_in = list(csv.DictReader(fh))

    results: list[dict] = []
    if vocab:
        reach_a = attribute_reach(norm_a, vocab, K=K, encoder_model=encoder_model)
        reach_b = attribute_reach(norm_b, vocab, K=K, encoder_model=encoder_model)
        type_emb_map, dim = _embed_vocab(vocab, encoder_model)

        for row in rows_in:
            ea, eb = row.get("entity_a", ""), row.get("entity_b", "")
            sim = type_embed_similarity(ea, eb, reach_a, reach_b, type_emb_map, dim)
            results.append({
                "entity_a": ea,
                "entity_b": eb,
                "type_embed_sim": "" if sim is None else sim,
            })
    else:
        for row in rows_in:
            results.append({
                "entity_a": row.get("entity_a", ""),
                "entity_b": row.get("entity_b", ""),
                "type_embed_sim": "",
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["entity_a", "entity_b", "type_embed_sim"])
        w.writeheader()
        w.writerows(results)

    print(f"[TypeEmbed] Written {len(results)} rows -> {out_csv}", file=sys.stderr)
    return results


def layer3_type_backup(
    entities_a: list[str],
    entities_b: list[str],
    covered_a: set[str],
    covered_b: set[str],
    json_a: dict,
    json_b: dict,
    threshold: float = 0.5,
    top_k: int = 3,
    K: int = 2,
    encoder_model: str = _DEFAULT_MODEL,
) -> list[dict]:
    """
    Candidate-generation backup for orphan entities, using attribute-TYPE
    embedding similarity instead of WordNet WUP. Mirrors
    enriched_matcher.layer3_wup_backup(), which only rescues entities via
    name lexical similarity.

    Why this exists: type_embed_sim (and cosine_avg, wup) are only ever
    computed for entity pairs that already appear as a CANDIDATE row in the
    enriched CSV. AML, LogMap, and layer3_wup_backup all decide candidacy
    purely from entity NAMES. Two entities with identical attribute types but
    unrelated names (e.g. PowerCell vs ReactorUnit) never become a candidate
    pair at all under that scheme — so no amount of type-embedding accuracy
    downstream can rescue them. This layer closes that gap by proposing
    candidates directly from attribute-type overlap for entities that name-based
    layers (AML/LogMap/WUP) left completely unmatched.

    entities_a/entities_b must already be concept-only (no associations) —
    same lists passed to layer3_wup_backup. json_a/json_b must already be
    normalize_model()-ed.

    Returns rows shaped like enriched_matcher._CSV_FIELDS (entity_a, entity_b,
    source="Layer3_Type", matcher_conf, layer, plus WN/CN characterisation for
    schema consistency with the other layers' rows).
    """
    orphans_a = [e for e in entities_a if e not in covered_a]
    orphans_b = [e for e in entities_b if e not in covered_b]
    if not orphans_a and not orphans_b:
        return []

    vocab = collect_obs_vocab(json_a, json_b)
    if not vocab:
        return []

    reach_a = attribute_reach(json_a, vocab, K=K, encoder_model=encoder_model)
    reach_b = attribute_reach(json_b, vocab, K=K, encoder_model=encoder_model)
    type_emb_map, dim = _embed_vocab(vocab, encoder_model)

    from enriched_matcher import characterise_entity_pair

    print(f"  [Layer3-Type] Type backup: {len(orphans_a)} orphan(s) in A, "
          f"{len(orphans_b)} in B (type_embed_sim threshold={threshold}) ...",
          file=sys.stderr)

    added: set[tuple[str, str]] = set()
    rows: list[dict] = []

    for ea in orphans_a:
        scored = [
            (sim, eb) for eb in entities_b
            if (sim := type_embed_similarity(ea, eb, reach_a, reach_b, type_emb_map, dim)) is not None
            and sim >= threshold
        ]
        scored.sort(key=lambda x: -x[0])
        for sim, eb in scored[:top_k]:
            key = (ea, eb)
            if key in added:
                continue
            added.add(key)
            rows.append({
                "entity_a": ea, "entity_b": eb,
                "source": "Layer3_Type", "matcher_conf": "", "layer": 3,
                **characterise_entity_pair(ea, eb),
            })

    for eb in orphans_b:
        scored = [
            (sim, ea) for ea in entities_a
            if (sim := type_embed_similarity(ea, eb, reach_a, reach_b, type_emb_map, dim)) is not None
            and sim >= threshold
        ]
        scored.sort(key=lambda x: -x[0])
        for sim, ea in scored[:top_k]:
            key = (ea, eb)
            if key in added:
                continue
            added.add(key)
            rows.append({
                "entity_a": ea, "entity_b": eb,
                "source": "Layer3_Type", "matcher_conf": "", "layer": 3,
                **characterise_entity_pair(ea, eb),
            })

    print(f"  [Layer3-Type] Added {len(rows)} type-backup pair(s).", file=sys.stderr)
    return rows


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

    type_emb_map, dim = _embed_vocab(vocab, encoder_model)

    def _agg(reach: dict) -> np.ndarray:
        agg = np.zeros(dim, dtype=np.float64)
        for sig in reach.values():
            agg += _weighted_type_vector(sig, type_emb_map, dim)
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
