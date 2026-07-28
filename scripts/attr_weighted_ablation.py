#!/usr/bin/env python3
"""
scripts/attr_weighted_ablation.py
==================================
2x2 ablation for attr_weighted (attr_dist_sim, attribute_reach.py):

    aggregation:  {per-type bag-of-embeddings (production), joint-list embedding (experimental)}
    encoder:      {paraphrase-MiniLM-L6-v2 (production default), a candidate encoder}

Run over all 105 pairs of the 15-model Automobile_Synthetic factorial
(3 vocab tiers SAME/SYN/ALT x 5 topologies DEEP/WIDE/HUB/BIP/GRID), scored
against the hand-authored oracle in
enriched_ontology_matching/inputs/Automobile_Synthetic/metric_predictions.csv.

Each condition (aggregation x encoder) loads exactly ONE sentence-transformer
model, so it must run in its own process — loading two different transformer
architectures in one Python process on this machine reliably OOMs the GPU or
segfaults on CPU. `--condition` runs one condition and writes its per-pair
results to a JSON file; `--aggregate` (no torch import needed) reads all four
JSON files back and prints the comparison report.

Usage:
    python scripts/attr_weighted_ablation.py --condition bag   --encoder paraphrase-MiniLM-L6-v2 --out /tmp/bag_prod.json
    python scripts/attr_weighted_ablation.py --condition bag   --encoder all-mpnet-base-v2       --out /tmp/bag_cand.json
    python scripts/attr_weighted_ablation.py --condition joint --encoder paraphrase-MiniLM-L6-v2 --out /tmp/joint_prod.json
    python scripts/attr_weighted_ablation.py --condition joint --encoder all-mpnet-base-v2        --out /tmp/joint_cand.json
    python scripts/attr_weighted_ablation.py --aggregate /tmp/bag_prod.json /tmp/bag_cand.json /tmp/joint_prod.json /tmp/joint_cand.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).resolve().parent.parent
_EOM  = _ROOT / "enriched_ontology_matching"
sys.path.insert(0, str(_EOM))

INPUTS = _EOM / "inputs" / "Automobile_Synthetic"
PRED_CSV = INPUTS / "metric_predictions.csv"


def _json_path(model_name: str) -> Path:
    # "SAME_DEEP" -> auto_same_deep.json
    return INPUTS / f"auto_{model_name.lower()}.json"


def _vocab_tier(model_name: str) -> str:
    return model_name.split("_")[0]  # SAME / SYN / ALT


def _load_pred_rows() -> list[dict]:
    return list(csv.DictReader(open(PRED_CSV, newline="", encoding="utf-8")))


def run_bag_of_embeddings(data_a: dict, data_b: dict, K: int, encoder_model: str) -> dict:
    """Production aggregation (attribute_reach.run_attr_dist_stage's logic,
    inlined without the CSV side-effect) — embed each attribute type
    independently, weight-sum per entity and across entities, THEN cosine."""
    import numpy as np
    from attribute_reach import (
        load_inventory, normalize_model, collect_obs_vocab, attribute_reach,
        _embed_vocab, _weighted_type_vector, _wup_kernel, _type_weight_vector,
        _soft_cosine,
    )

    inv    = load_inventory()
    norm_a = normalize_model(data_a, inv)
    norm_b = normalize_model(data_b, inv)

    vocab = collect_obs_vocab(norm_a, norm_b)
    if not vocab:
        return {"embed_sim": 0.0, "wup_sim": 0.0, "attr_dist_sim": 0.0}

    reach_a = attribute_reach(norm_a, vocab, K=K, impute_threshold=1.1)
    reach_b = attribute_reach(norm_b, vocab, K=K, impute_threshold=1.1)

    type_emb_map, dim = _embed_vocab(vocab, encoder_model)

    def _agg(reach):
        agg = np.zeros(dim, dtype=np.float64)
        for sig in reach.values():
            agg += _weighted_type_vector(sig, type_emb_map, dim)
        return agg

    agg_a, agg_b = _agg(reach_a), _agg(reach_b)
    na, nb = np.linalg.norm(agg_a), np.linalg.norm(agg_b)
    embed_sim = round(float(np.dot(agg_a, agg_b) / (na * nb)), 4) if na > 1e-10 and nb > 1e-10 else 0.0

    vocab_index = {t: i for i, t in enumerate(vocab)}
    wup_kernel  = _wup_kernel(vocab)
    wvec_a = _type_weight_vector(reach_a, vocab_index)
    wvec_b = _type_weight_vector(reach_b, vocab_index)
    wup_sim = _soft_cosine(wvec_a, wvec_b, wup_kernel)

    return {
        "embed_sim": embed_sim,
        "wup_sim": wup_sim,
        "attr_dist_sim": round(min(embed_sim, wup_sim), 4),
    }


def run_one_condition(aggregation: str, encoder_model: str, K: int = 2) -> list[dict]:
    rows = _load_pred_rows()
    print(f"[Ablation] {len(rows)} pairs, aggregation={aggregation}, encoder={encoder_model}",
          file=sys.stderr)

    if aggregation == "bag":
        fn = run_bag_of_embeddings
    elif aggregation == "joint":
        from attribute_reach import run_attr_dist_stage_joint

        def fn(a, b, K, enc):
            return run_attr_dist_stage_joint(a, b, K=K, encoder_model=enc)
    else:
        raise ValueError(f"unknown aggregation: {aggregation}")

    json_cache: dict[str, dict] = {}

    def _load(model_name: str) -> dict:
        if model_name not in json_cache:
            json_cache[model_name] = json.loads(_json_path(model_name).read_text(encoding="utf-8"))
        return json_cache[model_name]

    out = []
    for i, row in enumerate(rows):
        model_a, model_b = row["model_a"], row["model_b"]
        r = fn(_load(model_a), _load(model_b), K, encoder_model)
        r["model_a"] = model_a
        r["model_b"] = model_b
        r["attr_weighted_pred"] = float(row["attr_weighted_pred"])
        out.append(r)
        if (i + 1) % 30 == 0:
            print(f"[Ablation] {i + 1}/{len(rows)} pairs done", file=sys.stderr)
    return out


def aggregate(paths: list[Path]) -> None:
    from scipy.stats import spearmanr

    conditions: dict[str, list[dict]] = {}
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        conditions[data["label"]] = data["rows"]

    any_rows = next(iter(conditions.values()))
    pred_vals = [r["attr_weighted_pred"] for r in any_rows]
    tier_pair = [
        "same" if sorted([_vocab_tier(r["model_a"]), _vocab_tier(r["model_b"])])[0]
        == sorted([_vocab_tier(r["model_a"]), _vocab_tier(r["model_b"])])[1]
        else "-".join(sorted([_vocab_tier(r["model_a"]), _vocab_tier(r["model_b"])]))
        for r in any_rows
    ]

    print("\n=== Spearman rank correlation vs attr_weighted_pred (n=%d) ===" % len(any_rows))
    print(f"{'condition':<28} {'embed_sim rho':>14} {'attr_dist_sim rho':>18}")
    for name, rs in conditions.items():
        embed_vals = [r["embed_sim"] for r in rs]
        dist_vals  = [r["attr_dist_sim"] for r in rs]
        rho_embed  = spearmanr(embed_vals, pred_vals).correlation
        rho_dist   = spearmanr(dist_vals, pred_vals).correlation
        print(f"{name:<28} {rho_embed:>14.4f} {rho_dist:>18.4f}")

    print("\n=== Group means by vocab-tier relationship (attr_dist_sim) ===")
    tiers_order = ["same", "SAME-SYN", "ALT-SYN", "ALT-SAME"]
    present_tiers = [t for t in tiers_order if t in tier_pair] + \
        sorted(set(tier_pair) - set(tiers_order))
    header = f"{'condition':<28} " + " ".join(f"{t:>10}" for t in present_tiers)
    print(header)
    for name, rs in conditions.items():
        dist_vals = [r["attr_dist_sim"] for r in rs]
        line = f"{name:<28} "
        for t in present_tiers:
            vals = [v for v, tp in zip(dist_vals, tier_pair) if tp == t]
            line += f"{mean(vals):>10.4f} " if vals else f"{'--':>10} "
        print(line)

    line = f"{'oracle (attr_weighted_pred)':<28} "
    for t in present_tiers:
        vals = [v for v, tp in zip(pred_vals, tier_pair) if tp == t]
        line += f"{mean(vals):>10.4f} " if vals else f"{'--':>10} "
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["bag", "joint"])
    ap.add_argument("--encoder")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--aggregate", nargs="+", type=Path)
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.aggregate)
        return

    if not (args.condition and args.encoder and args.out):
        ap.error("--condition, --encoder and --out are required unless --aggregate is used")

    rows = run_one_condition(args.condition, args.encoder, args.k)
    label = f"{args.condition}+{args.encoder}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"label": label, "rows": rows}, indent=2), encoding="utf-8")
    print(f"[Ablation] wrote {len(rows)} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
