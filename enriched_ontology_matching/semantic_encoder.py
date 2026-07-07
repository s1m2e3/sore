"""
semantic_encoder.py
--------------------
Sentence-embedding cosine similarity for the enriched ontology matching
pipeline.  Mirrors the approach in ontology_matching/synonym_matcher.py
(paraphrase-MiniLM-L6-v2, L2-normalised embeddings) but adapted for the
two JSON formats used in enriched_ontology_matching (V-model and Network).

Entity representation strategy: an entity's vector is built from the TYPES of
its declared attributes (e.g. Engine.{oilPressure:Pressure, rpm:AngularVelocity}
-> ["Pressure", "AngularVelocity"]), not from its own name or its attributes'
names. Composition references (an attribute whose type is another entity's
name) are excluded — only genuine observable/datatype types count.
Entities with zero declared attributes (e.g. bare Net-model nodes) fall back
to their own CamelCase-split name, since there is no type evidence to use.

API
---
  encode_entities(data)           → {entity_name: np.ndarray}  (unit vectors)
  cosine_sim(name_a, name_b, ...)  → float in [0, 1]
  pair_similarities(pairs, ...)   → list[dict] with 'cosine_sim' column
  run_embedding_stage(enriched_csv, json_a, json_b, out_csv)

Usage (standalone):
  venv/Scripts/python.exe enriched_ontology_matching/semantic_encoder.py \\
      --a  ontology_matching/inputs/.../automobile_model_v1.json \\
      --b  ontology_matching/inputs/.../automobile_model_v2.json \\
      --matches-csv enriched_ontology_matching/outputs/enriched/<stem>.csv \\
      --out-csv     enriched_ontology_matching/outputs/embeddings/<key>_emb.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

from root_comparator import split_camel

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

DEFAULT_MODEL  = "paraphrase-MiniLM-L6-v2"
MAX_NEIGHBOURS = 5     # max neighbours included in description
MAX_VERBS      = 4     # max association verbs per entity

# ---------------------------------------------------------------------------
# Encoder singleton
# ---------------------------------------------------------------------------

_ENCODER       = None
_ENCODER_MODEL = None


def _get_encoder(model_name: str = DEFAULT_MODEL):
    global _ENCODER, _ENCODER_MODEL
    if _ENCODER is None or _ENCODER_MODEL != model_name:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Encoder] Loading {model_name} on {device} ...")
            _ENCODER = SentenceTransformer(model_name, device=device)
            _ENCODER_MODEL = model_name
            print(f"[Encoder] Ready.")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
    return _ENCODER


def _encode_batch(texts: list[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Encode a list of texts, returning L2-normalised unit vectors."""
    enc = _get_encoder(model_name)
    embs = enc.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,   # cosine = dot product after L2-norm
        convert_to_numpy=True,
    )
    return embs   # shape: (N, dim)


# ---------------------------------------------------------------------------
# Association format helpers (handles both V-model and Network formats)
# ---------------------------------------------------------------------------

def _assoc_name(assoc: dict) -> str:
    return assoc.get("associationName") or assoc.get("name") or ""


def _assoc_participants(assoc: dict) -> list[str]:
    return assoc.get("associationParticipants") or assoc.get("participants") or []


# ---------------------------------------------------------------------------
# Entity description builder
# ---------------------------------------------------------------------------

_VERB_TOKENS = {
    "supports", "drives", "powers", "lubricates", "contains", "connects",
    "mounts", "links", "couples", "charges", "feeds", "engages", "modulates",
    "actuates", "controls", "monitors", "supplies", "regulates", "circulates",
    "transfers", "converts", "packages", "interacts", "manages", "requests",
    "provides", "uses", "has",
}

_STOP_TOKENS = {"to", "in", "on", "at", "with", "of", "from", "by", "via",
                "through", "into", "onto", "between", "within", "across",
                "along", "and", "or", "the", "a", "an"}


def _readable(name: str) -> str:
    """CamelCase → 'camel case' readable string."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return s.lower()


def _extract_verbs(assoc_name: str) -> list[str]:
    tokens = [t.lower() for t in split_camel(assoc_name)]
    return [t for t in tokens if t in _VERB_TOKENS]


def build_descriptions(data: dict) -> dict[str, str]:
    """
    Build a name-first description for every entity in a model JSON.

    Returns {entity_name: description_string}.
    """
    entities   = {e["entityName"] for e in data.get("entities", [])}
    assocs     = data.get("associations", [])

    # Build adjacency and verb maps
    neighbours: dict[str, list[str]] = {e: [] for e in entities}
    verbs:      dict[str, list[str]] = {e: [] for e in entities}

    for assoc in assocs:
        parts = _assoc_participants(assoc)
        name  = _assoc_name(assoc)
        vs    = _extract_verbs(name)
        entity_parts = [p for p in parts if p in entities]

        for p in entity_parts:
            # Neighbours are the other participants
            for q in entity_parts:
                if q != p and q not in neighbours[p]:
                    neighbours[p].append(q)
            # Association verbs
            for v in vs:
                if v not in verbs[p]:
                    verbs[p].append(v)

    descriptions: dict[str, str] = {}
    for ent_name in entities:
        parts: list[str] = []

        # 1. Entity name — first, for maximum encoder attention
        parts.append(_readable(ent_name))

        # 2. Association verbs (relational role)
        vs = verbs[ent_name][:MAX_VERBS]
        if vs:
            parts.append("role: " + ", ".join(vs))

        # 3. Direct neighbours (topological context)
        nbrs = neighbours[ent_name][:MAX_NEIGHBOURS]
        if nbrs:
            parts.append("connected to: " + ", ".join(_readable(n) for n in nbrs))

        descriptions[ent_name] = ". ".join(parts)

    return descriptions


# ---------------------------------------------------------------------------
# Multi-representation embeddings, sourced from attribute TYPES
# ---------------------------------------------------------------------------
#
# For entity "Engine" with attribute types [Pressure, AngularVelocity] we
# produce three vectors:
#
#   whole : encode("pressure angular velocity")        — holistic compound meaning
#   sum   : normalize(E_pressure + E_angularvelocity)   — additive type composition
#   prod  : normalize(E_pressure ⊙ E_angularvelocity)   — interactive type composition
#
# Entities with no declared attribute types fall back to their own
# CamelCase-split name (same construction, but tokens come from the name).
# For a single-type/single-token entity all three are identical.
# Cosine similarity is reported for each representation and as an average.
# ---------------------------------------------------------------------------

def _l2_norm(v: np.ndarray) -> np.ndarray:
    """Return unit vector; handles zero-vector by returning zeros."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


class MultiRepEmbedding:
    """Holds three complementary unit-vector representations of one entity."""
    __slots__ = ("whole", "sum", "prod", "tokens")

    def __init__(self, whole: np.ndarray, sum_: np.ndarray,
                 prod: np.ndarray, tokens: list[str]):
        self.whole  = whole   # encode(joined readable tokens)
        self.sum    = sum_    # normalized sum of token embeddings
        self.prod   = prod    # normalized element-wise product of token embeddings
        self.tokens = tokens  # attribute-type tokens used (or name tokens, as fallback)


def _get_entity_tokens(name: str) -> list[str]:
    """CamelCase-split, lower-cased, filtering stop tokens. Name-based fallback
    for entities with no declared attribute types."""
    raw = [t.lower() for t in split_camel(name)]
    filtered = [t for t in raw if t not in _STOP_TOKENS and len(t) > 1]
    return filtered or raw[:1]  # fallback to at least one token


def _get_type_tokens(types: list[str]) -> list[str]:
    """Readable-ize each attribute type (CamelCase -> spaced, lower-cased) as
    ONE token per type — types are not split into sub-words the way entity
    names are, since e.g. "ElectricPotential" is a single concept, not two."""
    return [_readable(t) for t in types if t]


def _attr_type_map(data: dict) -> dict[str, list[str]]:
    """
    entity_name -> list of its direct attribute TYPES.

    Excludes attributes whose type is a composition reference to another
    entity in the same model (e.g. Engine.cylinderBlock: CylinderBlock) —
    only genuine observable/datatype types count, mirroring
    attribute_reach.collect_obs_vocab's entity-type exclusion.
    """
    entity_names = {
        e.get("entityName") or e.get("name", "") for e in data.get("entities", [])
    }
    entity_names.discard("")
    result: dict[str, list[str]] = {}
    for e in data.get("entities", []):
        name = e.get("entityName") or e.get("name", "")
        if not name:
            continue
        result[name] = [
            attr.get("type", "")
            for attr in (e.get("entityAttributes") or e.get("attributes") or [])
            if attr.get("type") and attr.get("type") not in entity_names
        ]
    return result


def encode_entities_multi(
    entity_names: list[str],
    model_name: str = DEFAULT_MODEL,
    attr_types: dict[str, list[str]] | None = None,
) -> dict[str, MultiRepEmbedding]:
    """
    Encode a list of entities with three representations each.

    attr_types, when given, maps entity_name -> its attribute-type list
    (see _attr_type_map); those types become the tokens embedded for that
    entity. Entities absent from attr_types, or with an empty type list,
    fall back to CamelCase-splitting their own name.

    All token encodings are batched in a single GPU call for efficiency.

    Returns {entity_name: MultiRepEmbedding}.
    """
    if not entity_names:
        return {}
    attr_types = attr_types or {}

    # Step 1: collect all texts to encode in one batch
    #   - whole texts (one per entity)
    #   - unique token texts (deduplicated across all entities)
    whole_texts: list[str] = []
    token_sets:  list[list[str]] = []

    for name in entity_names:
        types = attr_types.get(name) or []
        if types:
            tokens = _get_type_tokens(types)
            whole_texts.append(" ".join(tokens))
        else:
            tokens = _get_entity_tokens(name)
            whole_texts.append(_readable(name))
        token_sets.append(tokens)

    unique_tokens = list({t for ts in token_sets for t in ts})

    # Step 2: encode everything in two batches (whole names + tokens)
    all_texts = whole_texts + unique_tokens
    all_embs  = _encode_batch(all_texts, model_name)   # L2-normalised

    whole_embs = all_embs[:len(whole_texts)]
    token_emb_map: dict[str, np.ndarray] = {
        tok: all_embs[len(whole_texts) + i]
        for i, tok in enumerate(unique_tokens)
    }

    # Step 3: build MultiRepEmbedding for each entity
    result: dict[str, MultiRepEmbedding] = {}
    for idx, name in enumerate(entity_names):
        tokens = token_sets[idx]
        tok_vecs = [token_emb_map[t] for t in tokens if t in token_emb_map]

        if len(tok_vecs) == 0:
            tok_vecs = [whole_embs[idx]]

        # Sum representation
        sum_raw = np.sum(tok_vecs, axis=0)
        sum_vec = _l2_norm(sum_raw)

        # Product representation (element-wise; identity for single token)
        prod_raw = tok_vecs[0].copy()
        for v in tok_vecs[1:]:
            prod_raw = prod_raw * v
        prod_vec = _l2_norm(prod_raw)

        result[name] = MultiRepEmbedding(
            whole  = whole_embs[idx],
            sum_   = sum_vec,
            prod   = prod_vec,
            tokens = tokens,
        )

    return result


def cosine_multi(
    emb_a: MultiRepEmbedding,
    emb_b: MultiRepEmbedding,
) -> dict[str, float]:
    """
    Compute cosine similarity under all three representations and their average.

    Returns:
        cosine_whole : similarity of whole-name encodings
        cosine_sum   : similarity of sum-of-tokens encodings
        cosine_prod  : similarity of element-wise-product encodings
        cosine_avg   : mean of the three
    """
    c_whole = float(np.dot(emb_a.whole, emb_b.whole))
    c_sum   = float(np.dot(emb_a.sum,   emb_b.sum))
    c_prod  = float(np.dot(emb_a.prod,  emb_b.prod))
    c_avg   = (c_whole + c_sum + c_prod) / 3
    # Rescale from [-1, 1] to [0, 1]
    c_whole = (c_whole + 1.0) / 2.0
    c_sum   = (c_sum   + 1.0) / 2.0
    c_prod  = (c_prod  + 1.0) / 2.0
    c_avg   = (c_avg   + 1.0) / 2.0
    return {
        "cosine_whole": round(c_whole, 4),
        "cosine_sum":   round(c_sum,   4),
        "cosine_prod":  round(c_prod,  4),
        "cosine_avg":   round(c_avg,   4),
    }


# ---------------------------------------------------------------------------
# Convenience: single-name cosine (no graph context, name only)
# ---------------------------------------------------------------------------

def cosine_from_names(
    name_a: str,
    name_b: str,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, float]:
    """
    Compute all three cosine similarities between two entity names,
    encoded on the fly (no graph context).

    Returns dict with cosine_whole, cosine_sum, cosine_prod, cosine_avg.
    """
    embs = encode_entities_multi([name_a, name_b], model_name)
    return cosine_multi(embs[name_a], embs[name_b])


# ---------------------------------------------------------------------------
# CSV augmentation
# ---------------------------------------------------------------------------

_EMB_FIELDS = [
    "entity_a", "entity_b", "layer", "semantic_label",
    "wup_score", "max_wup", "avg_wup",
    "cosine_whole", "cosine_sum", "cosine_prod", "cosine_avg",
    "tokens_a", "tokens_b",
]


def run_embedding_stage(
    enriched_csv: Path,
    json_a: Path,
    json_b: Path,
    out_csv: Path,
    model_name: str = DEFAULT_MODEL,
) -> list[dict]:
    """Read a per-pair enriched CSV, compute multi-representation cosine
    similarities for every row, write augmented rows to out_csv.

    Each entity's vector is built from its attribute TYPES (see
    _attr_type_map); entities with no declared attributes fall back to their
    own name.

    Parameters
    ----------
    enriched_csv  : per-pair enriched CSV from enriched_matcher.py
    json_a, json_b : original model JSON files (for entity/attribute extraction)
    out_csv       : output CSV path

    Raises FileNotFoundError if enriched_csv, json_a, or json_b do not exist.
    """
    for label, p in [("enriched_csv", enriched_csv), ("json_a", json_a), ("json_b", json_b)]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} not found: {p}")
    data_a = json.loads(Path(json_a).read_text(encoding="utf-8"))
    data_b = json.loads(Path(json_b).read_text(encoding="utf-8"))

    def _entity_name(e: dict) -> str:
        return e.get("entityName") or e.get("name") or ""

    names_a = [_entity_name(e) for e in data_a.get("entities", [])]
    names_b = [_entity_name(e) for e in data_b.get("entities", [])]
    names_a = [n for n in names_a if n]
    names_b = [n for n in names_b if n]

    attr_types_a = _attr_type_map(data_a)
    attr_types_b = _attr_type_map(data_b)

    print(f"[Encoder] Encoding {len(names_a)} entities from A (by attribute type) ...")
    embs_a = encode_entities_multi(names_a, model_name, attr_types=attr_types_a)

    print(f"[Encoder] Encoding {len(names_b)} entities from B (by attribute type) ...")
    embs_b = encode_entities_multi(names_b, model_name, attr_types=attr_types_b)

    # Read enriched CSV
    with open(enriched_csv, newline="", encoding="utf-8") as fh:
        rows_in = list(csv.DictReader(fh))

    print(f"[Encoder] Scoring {len(rows_in)} pairs ...")

    # ── Separate pairs into those with precomputed embeddings vs fallbacks ──
    valid_idx: list[int] = []
    fallback_idx: list[int] = []
    for i, row in enumerate(rows_in):
        ea, eb = row.get("entity_a", ""), row.get("entity_b", "")
        if embs_a.get(ea) is not None and embs_b.get(eb) is not None:
            valid_idx.append(i)
        else:
            fallback_idx.append(i)

    # ── Vectorised scoring for the common case ───────────────────────────────
    # Stack all embedding vectors into matrices and compute all three cosine
    # variants with a single element-wise multiply + sum (row-wise dot product).
    # This replaces N individual cosine_multi() calls with 6 numpy operations.
    cosine_results: dict[int, dict] = {}

    if valid_idx:
        ea_list = [rows_in[i]["entity_a"] for i in valid_idx]
        eb_list = [rows_in[i]["entity_b"] for i in valid_idx]

        whole_a = np.stack([embs_a[ea].whole for ea in ea_list])   # (N, d)
        whole_b = np.stack([embs_b[eb].whole for eb in eb_list])
        sum_a   = np.stack([embs_a[ea].sum   for ea in ea_list])
        sum_b   = np.stack([embs_b[eb].sum   for eb in eb_list])
        prod_a  = np.stack([embs_a[ea].prod  for ea in ea_list])
        prod_b  = np.stack([embs_b[eb].prod  for eb in eb_list])

        # Row-wise dot products via element-wise multiply + sum (= batch cosine)
        c_whole = (whole_a * whole_b).sum(axis=1)
        c_sum   = (sum_a   * sum_b  ).sum(axis=1)
        c_prod  = (prod_a  * prod_b ).sum(axis=1)
        c_avg   = (c_whole + c_sum + c_prod) / 3.0
        # Rescale from [-1, 1] to [0, 1]
        c_whole = (c_whole + 1.0) / 2.0
        c_sum   = (c_sum   + 1.0) / 2.0
        c_prod  = (c_prod  + 1.0) / 2.0
        c_avg   = (c_avg   + 1.0) / 2.0

        for k, i in enumerate(valid_idx):
            cosine_results[i] = {
                "cosine_whole": round(float(c_whole[k]), 4),
                "cosine_sum":   round(float(c_sum[k]),   4),
                "cosine_prod":  round(float(c_prod[k]),  4),
                "cosine_avg":   round(float(c_avg[k]),   4),
            }

    # ── Per-row fallback for entities not in the pre-encoded set ────────────
    _fallback_cache: dict[str, "MultiRepEmbedding"] = {}
    for i in fallback_idx:
        ea = rows_in[i].get("entity_a", "")
        eb = rows_in[i].get("entity_b", "")
        missing = {n for n in (ea, eb)
                   if n and n not in embs_a and n not in embs_b
                   and n not in _fallback_cache}
        if missing:
            _fallback_cache.update(encode_entities_multi(list(missing), model_name))
        all_embs = {**embs_a, **embs_b, **_fallback_cache}
        ma, mb = all_embs.get(ea), all_embs.get(eb)
        cosine_results[i] = (
            cosine_multi(ma, mb) if (ma is not None and mb is not None)
            else {"cosine_whole": 0.5, "cosine_sum": 0.5,
                  "cosine_prod": 0.5,  "cosine_avg": 0.5}
        )

    # ── Assemble output rows ─────────────────────────────────────────────────
    results: list[dict] = []
    for i, row in enumerate(rows_in):
        ea = row.get("entity_a", "")
        eb = row.get("entity_b", "")
        sims = cosine_results.get(i, {
            "cosine_whole": 0.5, "cosine_sum": 0.5,
            "cosine_prod": 0.5,  "cosine_avg": 0.5,
        })
        results.append({
            "entity_a":       ea,
            "entity_b":       eb,
            "layer":          row.get("layer", ""),
            "semantic_label": row.get("semantic_label", ""),
            "wup_score":      row.get("wup_score", ""),
            "max_wup":        row.get("max_wup", ""),
            "avg_wup":        row.get("avg_wup", ""),
            **sims,
            "tokens_a": "/".join(embs_a[ea].tokens) if ea in embs_a else ea.lower(),
            "tokens_b": "/".join(embs_b[eb].tokens) if eb in embs_b else eb.lower(),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_EMB_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print(f"[Encoder] Written {len(results)} rows -> {out_csv}")
    return results




# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Embedding cosine similarity stage for ontology match pairs."
    )
    parser.add_argument("--a", required=True,
                        help="Path to ontology A JSON")
    parser.add_argument("--b", required=True,
                        help="Path to ontology B JSON")
    parser.add_argument("--matches-csv", required=True,
                        help="Per-pair enriched CSV (from enriched_matcher.py)")
    parser.add_argument("--out-csv", required=True,
                        help="Output CSV path with cosine_sim column")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Sentence-transformer model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    run_embedding_stage(
        enriched_csv = Path(args.matches_csv),
        json_a       = Path(args.a),
        json_b       = Path(args.b),
        out_csv      = Path(args.out_csv),
        model_name   = args.model,
    )
