"""
entailment_matcher.py
----------------------
Neural cross-encoder NLI entailment between entity pairs, as a fifth,
opt-in semantic-comparison metric (not part of the core 4-metric composite).

Two independent probes, both using the same cross-encoder machinery:

  Entity-level entailment  (run_entity_entailment_stage)
      Premise/hypothesis are built from entity NAMES ("This is a {name}.").
      Probes the taxonomic question: does being an instance of A imply being
      an instance of B (hypernymy/hyponymy), or do both directions hold
      (synonymy)? This is the lexical-relation question, at the naming level.

  Attribute-level entailment  (run_attribute_entailment_stage)
      Premise/hypothesis are built from each entity's K-hop attribute-type
      reach signature (reusing attribute_reach.py's weighted BFS, the same
      reach computation attr_weighted uses), turned into a sentence like
      "The component has these properties: Temperature, Torque.". Probes
      whether one entity's observable-attribute profile entails the other's,
      independent of what either entity is named.

Both stages score BOTH directions per pair (entailment is not symmetric) and
report entailment_a_covers_b, entailment_b_covers_a, and entailment_f1 =
max(a_covers_b, b_covers_a).

Model
-----
Cross-encoder (NOT a bi-encoder): premise and hypothesis are encoded jointly
in one transformer pass, which is required for entailment scoring, a
bi-encoder only ever produces independent embeddings and cannot represent
this kind of directional, joint-context relationship.

Default: cross-encoder/nli-MiniLM2-L6-H768 (small, CPU-friendly, previously
validated in this repo, see git history commit f6a9867). Swappable via
nli_model_name, e.g. "cross-encoder/nli-deberta-v3-base" for higher accuracy
at higher compute cost (same sentence-transformers CrossEncoder API).

Usage (standalone)
-------------------
    from entailment_matcher import run_entity_entailment_stage, run_attribute_entailment_stage

    run_entity_entailment_stage(json_a, json_b, out_csv=Path("entity_entailment.csv"))
    run_attribute_entailment_stage(json_a, json_b, out_csv=Path("attr_entailment.csv"))

CLI
---
    python enriched_ontology_matching/entailment_matcher.py \\
        --a enriched_ontology_matching/inputs/Automobile/automobile_model_v1.json \\
        --b enriched_ontology_matching/inputs/Automobile/automobile_model_v2.json \\
        --mode entity --out entity_entailment.csv

    ... --mode attribute --out attr_entailment.csv
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

from model_normalizer import load_inventory, normalize_model
from attribute_reach import attribute_reach, collect_obs_vocab

# ---------------------------------------------------------------------------
# Model loading (cross-encoder, cached locally like the other sentence-
# transformer models in this pipeline)
# ---------------------------------------------------------------------------

_MODELS_DIR       = _DIR / "models"
DEFAULT_NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
_NLI_MODEL        = None
_NLI_MODEL_NAME   = None


def _model_local_path(model_name: str) -> Path:
    return _MODELS_DIR / model_name.replace("/", "--")


def _get_nli_model(model_name: str = DEFAULT_NLI_MODEL):
    global _NLI_MODEL, _NLI_MODEL_NAME
    if _NLI_MODEL is None or _NLI_MODEL_NAME != model_name:
        try:
            from sentence_transformers import CrossEncoder
            import torch
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for neural entailment. "
                "Install with: pip install sentence-transformers"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local  = _model_local_path(model_name)
        if local.is_dir():
            print(f"[NLI] Loading {model_name} from {local} ...", file=sys.stderr)
            _NLI_MODEL = CrossEncoder(str(local), device=device)
        else:
            print(f"[NLI] Downloading {model_name} ...", file=sys.stderr)
            _MODELS_DIR.mkdir(parents=True, exist_ok=True)
            _NLI_MODEL = CrossEncoder(model_name, device=device)
            _NLI_MODEL.save(str(local))
            print(f"[NLI] Saved to {local}", file=sys.stderr)
        _NLI_MODEL_NAME = model_name
        print(f"[NLI] Ready ({device}).", file=sys.stderr)
    return _NLI_MODEL


def _get_entailment_idx(model) -> int:
    """Look up which output index the model's config calls 'entailment' —
    NLI models don't agree on label order (some are contradiction/entailment/
    neutral, others entailment/neutral/contradiction)."""
    id2label = getattr(model.model.config, "id2label", {})
    for idx, lbl in id2label.items():
        if lbl.lower() == "entailment":
            return int(idx)
    return 1  # fallback, matches nli-MiniLM2-L6-H768's default ordering


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _score_entailment_batch(
    pairs: list[tuple[str, str]],
    model,
    ent_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score every (premise, hypothesis) pair in both directions.

    `pairs` must already contain BOTH directions interleaved as
    [(a,b), (b,a), (a,b), (b,a), ...] — i.e. 2 entries per logical pair.
    Returns (a_covers_b_arr, b_covers_a_arr), one entry per logical pair.
    """
    if not pairs:
        return np.array([]), np.array([])

    raw = np.array(model.predict(pairs, batch_size=64, show_progress_bar=False))

    if raw.ndim == 1:
        # Binary relevance-style model: apply sigmoid, treat score as entailment prob directly
        ent_probs = 1.0 / (1.0 + np.exp(-raw))
    else:
        probs = _softmax_np(raw)
        ent_probs = probs[:, ent_idx]

    return ent_probs[0::2], ent_probs[1::2]


# ---------------------------------------------------------------------------
# Text construction
# ---------------------------------------------------------------------------

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _readable(name: str) -> str:
    return _CAMEL_RE.sub(" ", name).lower()


def _entity_names(data: dict) -> list[str]:
    return [
        e.get("entityName") or e.get("name", "")
        for e in data.get("entities", [])
        if (e.get("entityName") or e.get("name", ""))
    ]


def _entity_premise(entity_name: str) -> str:
    return f"This is a {_readable(entity_name)}."


def _attr_premise(obs_types) -> str:
    if not obs_types:
        return "The component has no known properties."
    return "The component has these properties: " + ", ".join(sorted(_readable(t) for t in obs_types)) + "."


# ---------------------------------------------------------------------------
# Shared CSV writer
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "entity_a", "entity_b",
    "entailment_a_covers_b",
    "entailment_b_covers_a",
    "entailment_f1",
]


def _write_rows(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Stage 1: entity-level entailment (naming / taxonomic relation probe)
# ---------------------------------------------------------------------------

def run_entity_entailment_stage(
    data_a: dict,
    data_b: dict,
    out_csv: Path,
    nli_model_name: str = DEFAULT_NLI_MODEL,
) -> Path:
    """
    Cross-encoder NLI entailment between every A x B entity pair, using each
    entity's NAME as the premise/hypothesis text. Probes synonymy (both
    directions entail) and hypernymy/hyponymy (only one direction entails).
    """
    entities_a = _entity_names(data_a)
    entities_b = _entity_names(data_b)
    pairs = [(ea, eb) for ea in entities_a for eb in entities_b]

    if not pairs:
        _write_rows([], out_csv)
        return out_csv

    model   = _get_nli_model(nli_model_name)
    ent_idx = _get_entailment_idx(model)

    text_pairs: list[tuple[str, str]] = []
    for ea, eb in pairs:
        pa, pb = _entity_premise(ea), _entity_premise(eb)
        text_pairs.append((pa, pb))
        text_pairs.append((pb, pa))

    print(f"  [EntityEntailment] Scoring {len(pairs)} pairs ({len(text_pairs)} inferences) ...",
          file=sys.stderr)
    a_covers_b_arr, b_covers_a_arr = _score_entailment_batch(text_pairs, model, ent_idx)

    rows = []
    for i, (ea, eb) in enumerate(pairs):
        acb, bca = float(a_covers_b_arr[i]), float(b_covers_a_arr[i])
        rows.append({
            "entity_a": ea, "entity_b": eb,
            "entailment_a_covers_b": round(acb, 4),
            "entailment_b_covers_a": round(bca, 4),
            "entailment_f1": round(max(acb, bca), 4),
        })

    _write_rows(rows, out_csv)
    print(f"  [EntityEntailment] {len(entities_a)} x {len(entities_b)} = {len(rows)} pairs -> {out_csv}",
          file=sys.stderr)
    return out_csv


# ---------------------------------------------------------------------------
# Stage 2: attribute-level entailment (observable-type profile probe)
# ---------------------------------------------------------------------------

def run_attribute_entailment_stage(
    data_a: dict,
    data_b: dict,
    out_csv: Path,
    K: int = 2,
    nli_model_name: str = DEFAULT_NLI_MODEL,
) -> Path:
    """
    Cross-encoder NLI entailment between every A x B entity pair, using each
    entity's K-hop attribute-type reach signature (attribute_reach.py, the
    same reach computation attr_weighted uses) as the premise/hypothesis
    text (a single holistic "has these properties: ..." sentence per side).
    Probes whether one entity's observable-attribute profile entails the
    other's, independent of entity naming.

    A decomposed, per-property atomic variant (testing each of B's
    properties as its own single-fact hypothesis, then averaging) was tried
    and empirically performed worse: independently top-k-capping each side's
    properties by weight discards exactly the overlapping evidence needed to
    detect containment, and even uncapped, per-property averaging washed out
    the clean signal this holistic version gets right (see git history /
    conversation for the validated example: two same-name entities across
    model variants, one with a strict superset of the other's declared
    attribute types, correctly scored ~0.07 / ~0.94 here).

    Imputation is disabled, matching run_attr_dist_stage, so this stays
    purely structural/declared rather than name-dependent.
    """
    inv    = load_inventory()
    norm_a = normalize_model(data_a, inv)
    norm_b = normalize_model(data_b, inv)

    vocab = collect_obs_vocab(norm_a, norm_b)
    entities_a = _entity_names(norm_a)
    entities_b = _entity_names(norm_b)
    pairs = [(ea, eb) for ea in entities_a for eb in entities_b]

    if not pairs or not vocab:
        _write_rows([], out_csv)
        return out_csv

    reach_a = attribute_reach(norm_a, vocab, K=K, impute_threshold=1.1)
    reach_b = attribute_reach(norm_b, vocab, K=K, impute_threshold=1.1)

    model   = _get_nli_model(nli_model_name)
    ent_idx = _get_entailment_idx(model)

    text_cache_a = {ea: _attr_premise(set(reach_a.get(ea, {}))) for ea in entities_a}
    text_cache_b = {eb: _attr_premise(set(reach_b.get(eb, {}))) for eb in entities_b}

    text_pairs: list[tuple[str, str]] = []
    for ea, eb in pairs:
        pa, pb = text_cache_a[ea], text_cache_b[eb]
        text_pairs.append((pa, pb))
        text_pairs.append((pb, pa))

    print(f"  [AttrEntailment] Scoring {len(pairs)} pairs ({len(text_pairs)} inferences) ...",
          file=sys.stderr)
    a_covers_b_arr, b_covers_a_arr = _score_entailment_batch(text_pairs, model, ent_idx)

    rows = []
    for i, (ea, eb) in enumerate(pairs):
        acb, bca = float(a_covers_b_arr[i]), float(b_covers_a_arr[i])
        rows.append({
            "entity_a": ea, "entity_b": eb,
            "entailment_a_covers_b": round(acb, 4),
            "entailment_b_covers_a": round(bca, 4),
            "entailment_f1": round(max(acb, bca), 4),
        })

    _write_rows(rows, out_csv)
    print(f"  [AttrEntailment] {len(entities_a)} x {len(entities_b)} = {len(rows)} pairs -> {out_csv}",
          file=sys.stderr)
    return out_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Cross-encoder NLI entailment, entity-level or attribute-level."
    )
    ap.add_argument("--a", required=True, help="Ontology A JSON")
    ap.add_argument("--b", required=True, help="Ontology B JSON")
    ap.add_argument("--mode", choices=["entity", "attribute"], required=True)
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--model", default=DEFAULT_NLI_MODEL)
    ap.add_argument("--k", type=int, default=2, help="K-hop reach (attribute mode only)")
    args = ap.parse_args()

    data_a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    data_b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    data_a = data_a.get("json_a", data_a)
    data_b = data_b.get("json_b", data_b) if "json_b" in data_b else data_b

    if args.mode == "entity":
        run_entity_entailment_stage(data_a, data_b, Path(args.out), nli_model_name=args.model)
    else:
        run_attribute_entailment_stage(data_a, data_b, Path(args.out), K=args.k, nli_model_name=args.model)
