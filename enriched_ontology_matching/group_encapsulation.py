"""
group_encapsulation.py
------------------------
Step 2 of conceptual-encapsulation (1:n / "complex match") discovery: score
whether a candidate subgroup (subgraph_candidates.py) in one model
corresponds to a single, coarser entity in the OTHER model.

Sketch/prototype: scores one (coarse_entity, candidate_group) pair at a
time. Does not enumerate every orphan x every candidate group across a full
domain, that loop is a straightforward addition once the scoring itself is
validated on known cases, deliberately deferred.

Two independent scores, deliberately reusing infrastructure already
validated elsewhere in this pipeline rather than inventing new machinery
for a problem two other modules already mostly solve:

  score_group_attribute()   — does the UNION of the group's attribute-type
      reach vectors match the coarse entity's own reach vector? Reuses
      attribute_reach.py's embed+WUP kernel (the exact machinery behind
      attr_weighted), aggregated over every group member instead of one
      entity.

  score_group_entity_name()  — does "a system made of {group member names}"
      entail "this is {a coarse entity}"? Reuses entailment_matcher.py's
      cross-encoder NLI model and batch-scoring helper.

Usage (standalone)
-------------------
    from group_encapsulation import score_group_attribute, score_group_entity_name

    attr_scores = score_group_attribute("Engine", group_b, data_a, data_b)
    name_scores = score_group_entity_name("Engine", group_b)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

from model_normalizer import load_inventory, normalize_model
from attribute_reach import (
    attribute_reach, collect_obs_vocab, _embed_vocab, _weighted_type_vector,
    _wup_kernel, _type_weight_vector, _soft_cosine,
)
from entailment_matcher import (
    _get_nli_model, _get_entailment_idx, _score_entailment_batch,
    _entity_premise, _readable, DEFAULT_NLI_MODEL,
)


# ---------------------------------------------------------------------------
# Attribute-level: does the group's UNION reach match the coarse entity's?
# ---------------------------------------------------------------------------

def score_group_attribute(
    entity_a: str,
    group_b,
    data_a: dict,
    data_b: dict,
    K: int = 2,
    encoder_model: str = "paraphrase-MiniLM-L6-v2",
) -> dict:
    """
    Attribute-level encapsulation score between one coarse entity in model A
    and a candidate group of entities in model B.

    Mirrors attribute_reach.run_attr_dist_stage's embed+WUP blend exactly,
    except the "B side" vector is the union of every group member's reach
    instead of a single entity's.

    Returns {embed_sim, wup_sim, encapsulation_sim}; encapsulation_sim is
    min(embed_sim, wup_sim), same rationale as attr_dist_sim — WUP alone
    runs too forgiving for engineering/physics nouns.
    """
    inv    = load_inventory()
    norm_a = normalize_model(data_a, inv)
    norm_b = normalize_model(data_b, inv)

    vocab = collect_obs_vocab(norm_a, norm_b)
    if not vocab:
        return {"embed_sim": 0.0, "wup_sim": 0.0, "encapsulation_sim": 0.0}

    reach_a = attribute_reach(norm_a, vocab, K=K, impute_threshold=1.1)
    reach_b = attribute_reach(norm_b, vocab, K=K, impute_threshold=1.1)

    type_emb_map, dim = _embed_vocab(vocab, encoder_model)

    agg_entity = _weighted_type_vector(reach_a.get(entity_a, {}), type_emb_map, dim)
    agg_group  = np.zeros(dim, dtype=np.float64)
    for member in group_b:
        agg_group += _weighted_type_vector(reach_b.get(member, {}), type_emb_map, dim)

    n_e, n_g = np.linalg.norm(agg_entity), np.linalg.norm(agg_group)
    embed_sim = (
        round(float(np.dot(agg_entity, agg_group) / (n_e * n_g)), 4)
        if n_e > 1e-10 and n_g > 1e-10 else 0.0
    )

    vocab_index = {t: i for i, t in enumerate(vocab)}
    kernel   = _wup_kernel(vocab)
    w_entity = _type_weight_vector({entity_a: reach_a.get(entity_a, {})}, vocab_index)
    w_group  = _type_weight_vector({m: reach_b.get(m, {}) for m in group_b}, vocab_index)
    wup_sim  = _soft_cosine(w_entity, w_group, kernel)

    return {
        "embed_sim": embed_sim,
        "wup_sim": wup_sim,
        "encapsulation_sim": round(min(embed_sim, wup_sim), 4),
    }


# ---------------------------------------------------------------------------
# Entity-name level: does "a system made of {group}" entail "{entity}"?
# ---------------------------------------------------------------------------

def _group_premise(members) -> str:
    names = sorted(_readable(m) for m in members)
    if len(names) == 1:
        return f"This is a system made of {names[0]}."
    return "This is a system made of " + ", ".join(names[:-1]) + f", and {names[-1]}."


def score_group_entity_name(
    entity_a: str,
    group_b,
    nli_model_name: str = DEFAULT_NLI_MODEL,
) -> dict:
    """
    Entity-name-level encapsulation score: cross-encoder NLI entailment
    between a coarse entity's name and a composite description of a
    candidate group's member names.

    Returns {group_covers_entity, entity_covers_group, encapsulation_f1},
    mirroring entailment_matcher.py's a_covers_b/b_covers_a/f1 convention.
    group_covers_entity ("the parts, taken together, constitute the whole")
    is the theoretically meaningful direction; entity_covers_group is
    reported alongside for consistency and diagnostic value.
    """
    model   = _get_nli_model(nli_model_name)
    ent_idx = _get_entailment_idx(model)

    group_text  = _group_premise(group_b)
    entity_text = _entity_premise(entity_a)

    text_pairs = [(group_text, entity_text), (entity_text, group_text)]
    group_covers_arr, entity_covers_arr = _score_entailment_batch(text_pairs, model, ent_idx)

    gce, ecg = float(group_covers_arr[0]), float(entity_covers_arr[0])
    return {
        "group_covers_entity": round(gce, 4),
        "entity_covers_group": round(ecg, 4),
        "encapsulation_f1": round(max(gce, ecg), 4),
    }


# ---------------------------------------------------------------------------
# Abstention: deciding when the top-ranked candidate is NOT a confident match
# ---------------------------------------------------------------------------
#
# Empirically (33-case Automobile-domain evaluation, see conversation), a
# single threshold on the top score alone cannot separate correct from
# incorrect top-1 picks: some genuinely correct matches score very high but
# with a tiny margin over the runner-up (e.g. BodyStructure vs a body/
# suspension group at 0.969, runner-up at 0.958, margin 0.011), which looks
# statistically identical to genuinely wrong picks with an equally tiny
# margin (e.g. HVACSystem, which no candidate correctly represents, at 0.823
# vs 0.812, margin 0.011). A margin-only rule would incorrectly suppress the
# former along with the latter.
#
# Two-tier rule instead:
#   1. If the top score is very high (>= high_confidence), trust it
#      regardless of margin — a near-certain absolute score is itself
#      strong evidence, and this is what protects the correct
#      high-score/low-margin cases above.
#   2. Otherwise, require BOTH a minimum absolute score AND a minimum
#      margin over the runner-up before reporting a match.
#
# This is not perfect (see conversation: it does not catch every case,
# not least because "trust anything above high_confidence" also protects a
# small number of genuinely wrong high-scoring near-misses), but it removes
# the most damaging failure mode observed: confidently wrong answers on
# entities with no true corresponding group at all.

def classify_top_match(
    scored: list[tuple[float, object]],
    min_score: float = 0.5,
    min_margin: float = 0.15,
    high_confidence: float = 0.9,
) -> dict:
    """
    Decide whether the top-ranked (score, group) candidate in `scored`
    (sorted descending by score) should be reported as a match, or whether
    the evidence is too weak / too ambiguous and the result should be an
    abstention instead.

    Returns {"decision": "match" | "no_match", "reason": str, "top": (score,
    group) or None, "margin": float}.
    """
    if not scored:
        return {"decision": "no_match", "reason": "no_candidates", "top": None, "margin": 0.0}

    top_score, top_group = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    margin = top_score - second_score

    if top_score >= high_confidence:
        return {"decision": "match", "reason": "high_confidence", "top": scored[0], "margin": margin}
    if top_score < min_score:
        return {"decision": "no_match", "reason": "low_score", "top": scored[0], "margin": margin}
    if margin < min_margin:
        return {"decision": "no_match", "reason": "ambiguous_margin", "top": scored[0], "margin": margin}
    return {"decision": "match", "reason": "ok", "top": scored[0], "margin": margin}


# ---------------------------------------------------------------------------
# Pipeline-scale stage: every entity vs every candidate group, both directions
# ---------------------------------------------------------------------------
#
# score_group_attribute()/score_group_entity_name() above are single-pair
# building blocks: correct, but each call recomputes the attribute-type
# vocabulary, K-hop reach, embedding table, and WUP kernel from scratch, and
# issues its own tiny NLI batch. That's fine for the CLI/ad-hoc case (one
# entity, one group) but does not scale to "every entity in A against every
# candidate group in B" — validated at ~200 pairs in the conversation, where
# it took several minutes due to exactly this redundant recomputation.
#
# run_encapsulation_stage() computes the shared per-pair machinery ONCE and
# batches every NLI call for the whole pair into a single model.predict()
# invocation, the same batching strategy entailment_matcher.py's stage
# functions already use.

_ENCAPS_FIELDS = [
    "direction", "entity", "decision", "reason",
    "best_group", "attr_score",
    "name_group_covers_entity", "name_entity_covers_group", "name_f1",
    "margin",
]


def _write_encaps_rows(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_ENCAPS_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _ENCAPS_FIELDS})


def run_encapsulation_stage(
    data_a: dict,
    data_b: dict,
    out_csv: Path,
    K: int = 2,
    encoder_model: str = "paraphrase-MiniLM-L6-v2",
    nli_model_name: str = DEFAULT_NLI_MODEL,
    min_score: float = 0.5,
    min_margin: float = 0.15,
    high_confidence: float = 0.9,
) -> Path:
    """
    Pipeline-scale conceptual-encapsulation stage for one model pair.

    For every entity in each model, tests whether it corresponds to one of
    the OTHER model's candidate subgroups (subgraph_candidates.py's Louvain
    communities), in both directions, then reports only the best-scoring
    candidate per entity after the classify_top_match() abstention gate.
    Most entities have no true group correspondence; reporting every
    candidate would be almost entirely noise (see conversation: a 33-case
    evaluation found this abstention gate raises precision from ~77% to
    ~89.5% at the cost of recall dropping from ~88% to ~68%, an expected
    and, for this use case, worthwhile trade).

    Writes one row per (direction, entity): direction is "a_in_b" (entities
    of A tested against B's candidate groups) or "b_in_a". decision is
    "match" or "no_match"; no_match rows have blank score/group fields.
    """
    from subgraph_candidates import candidate_subgroups

    inv    = load_inventory()
    norm_a = normalize_model(data_a, inv)
    norm_b = normalize_model(data_b, inv)

    entities_a = [e.get("entityName") or e.get("name", "") for e in norm_a.get("entities", [])]
    entities_a = [e for e in entities_a if e]
    entities_b = [e.get("entityName") or e.get("name", "") for e in norm_b.get("entities", [])]
    entities_b = [e for e in entities_b if e]

    communities_a = candidate_subgroups(data_a)["communities"]
    communities_b = candidate_subgroups(data_b)["communities"]

    vocab = collect_obs_vocab(norm_a, norm_b)
    if not vocab or (not communities_a and not communities_b):
        _write_encaps_rows([], out_csv)
        return out_csv

    reach_a = attribute_reach(norm_a, vocab, K=K, impute_threshold=1.1)
    reach_b = attribute_reach(norm_b, vocab, K=K, impute_threshold=1.1)
    type_emb_map, dim = _embed_vocab(vocab, encoder_model)
    vocab_index = {t: i for i, t in enumerate(vocab)}
    kernel = _wup_kernel(vocab)

    model   = _get_nli_model(nli_model_name)
    ent_idx = _get_entailment_idx(model)

    def _attr_score(entity: str, reach_entity_side: dict, group, reach_group_side: dict) -> float:
        agg_entity = _weighted_type_vector(reach_entity_side.get(entity, {}), type_emb_map, dim)
        agg_group  = np.zeros(dim, dtype=np.float64)
        for member in group:
            agg_group += _weighted_type_vector(reach_group_side.get(member, {}), type_emb_map, dim)
        n_e, n_g = np.linalg.norm(agg_entity), np.linalg.norm(agg_group)
        embed_sim = (
            round(float(np.dot(agg_entity, agg_group) / (n_e * n_g)), 4)
            if n_e > 1e-10 and n_g > 1e-10 else 0.0
        )
        w_entity = _type_weight_vector({entity: reach_entity_side.get(entity, {})}, vocab_index)
        w_group  = _type_weight_vector({m: reach_group_side.get(m, {}) for m in group}, vocab_index)
        wup_sim  = _soft_cosine(w_entity, w_group, kernel)
        return round(min(embed_sim, wup_sim), 4)

    def _direction(entities, groups, reach_entity_side, reach_group_side, direction_label) -> list[dict]:
        if not entities or not groups:
            return []

        text_pairs: list[tuple[str, str]] = []
        index: list[tuple[str, int]] = []
        for entity in entities:
            entity_text = _entity_premise(entity)
            for gi, group in enumerate(groups):
                group_text = _group_premise(group)
                text_pairs.append((group_text, entity_text))
                text_pairs.append((entity_text, group_text))
                index.append((entity, gi))

        group_covers_arr, entity_covers_arr = _score_entailment_batch(text_pairs, model, ent_idx)

        per_entity: dict[str, list[tuple[float, float, float, int]]] = {}
        for k, (entity, gi) in enumerate(index):
            gce, ecg = float(group_covers_arr[k]), float(entity_covers_arr[k])
            per_entity.setdefault(entity, []).append((max(gce, ecg), gce, ecg, gi))

        out_rows: list[dict] = []
        for entity, scored in per_entity.items():
            scored.sort(key=lambda x: -x[0])
            ranked = [(s[0], s[3]) for s in scored]
            decision = classify_top_match(
                ranked, min_score=min_score, min_margin=min_margin, high_confidence=high_confidence
            )
            if decision["decision"] != "match":
                out_rows.append({
                    "direction": direction_label, "entity": entity,
                    "decision": "no_match", "reason": decision["reason"],
                    "margin": round(decision["margin"], 4),
                })
                continue

            top_f1, top_gi = decision["top"]
            group = groups[top_gi]
            gce = next(s[1] for s in scored if s[3] == top_gi)
            ecg = next(s[2] for s in scored if s[3] == top_gi)
            out_rows.append({
                "direction": direction_label, "entity": entity,
                "decision": "match", "reason": decision["reason"],
                "best_group": "|".join(sorted(group)),
                "attr_score": _attr_score(entity, reach_entity_side, group, reach_group_side),
                "name_group_covers_entity": round(gce, 4),
                "name_entity_covers_group": round(ecg, 4),
                "name_f1": round(top_f1, 4),
                "margin": round(decision["margin"], 4),
            })
        return out_rows

    rows = []
    rows += _direction(entities_a, communities_b, reach_a, reach_b, "a_in_b")
    rows += _direction(entities_b, communities_a, reach_b, reach_a, "b_in_a")

    _write_encaps_rows(rows, out_csv)
    n_match = sum(1 for r in rows if r["decision"] == "match")
    print(
        f"  [Encapsulation] {len(entities_a)}+{len(entities_b)} entities, "
        f"{len(communities_a)}+{len(communities_b)} candidate groups -> "
        f"{n_match}/{len(rows)} confident matches -> {out_csv}",
        file=sys.stderr,
    )
    return out_csv


# ---------------------------------------------------------------------------
# CLI — score one (entity, group) pair for manual inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Score whether a candidate group encapsulates a coarse entity."
    )
    ap.add_argument("--a", required=True, help="Model A JSON (contains the coarse entity)")
    ap.add_argument("--b", required=True, help="Model B JSON (contains the candidate group)")
    ap.add_argument("--entity", required=True, help="Coarse entity name in model A")
    ap.add_argument("--group", required=True, nargs="+", help="Candidate group member names in model B")
    args = ap.parse_args()

    data_a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    data_b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    group_b = set(args.group)

    attr = score_group_attribute(args.entity, group_b, data_a, data_b)
    name = score_group_entity_name(args.entity, group_b)

    print(f"Entity: {args.entity}")
    print(f"Group:  {sorted(group_b)}")
    print(f"\nAttribute-level: {attr}")
    print(f"Entity-name level: {name}")
