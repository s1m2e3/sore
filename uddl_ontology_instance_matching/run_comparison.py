"""
run_comparison.py
-----------------
Runs the full MNLI + LIMES comparison between two UDDL messages
without requiring any ground truth.

MNLI strategy (three passes):
  1. obs_A  x  obs_B   -- observable vs observable (same concept, different label?)
  2. ent_A  x  obs_B   -- does entity A entail any observable in B?
  3. ent_B  x  obs_A   -- does entity B entail any observable in A?

For each pass, every (src, tgt) pair is scored and pairs above the
entailment threshold are reported with their score.
LIMES lexical results (from pre-run NT files) are shown alongside.

Usage
-----
  python run_comparison.py \\
      --ttl-a  outputs/uddl_layered_aircraft_message_1.ttl \\
      --ttl-b  outputs/uddl_layered_aircraft_message_2.ttl \\
      [--limes-accept  outputs/limes_obs_accept_*.nt] \\
      [--limes-review  outputs/limes_obs_review_*.nt] \\
      [--threshold     0.35] \\
      [--out-csv       outputs/mnli_comparison.csv]
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

from rdflib import Graph, RDF, URIRef

META   = "http://example.org/uddl_metamodel.owl#"
OWL_SA = "http://www.w3.org/2002/07/owl#sameAs"

# ── RDF loaders ───────────────────────────────────────────────────────────────

def load_labels(ttl_path: str, class_name: str) -> Dict[str, str]:
    """Return {local_id: hasName} for every instance of the given class."""
    g = Graph()
    g.parse(ttl_path, format="turtle")
    out: Dict[str, str] = {}
    for s, _, _ in g.triples((None, RDF.type, URIRef(META + class_name))):
        names = list(g.objects(s, URIRef(META + "hasName")))
        if names:
            out[str(s).split("#")[-1]] = str(names[0])
    return out


def parse_limes_nt(path: str) -> List[Tuple[str, str]]:
    pairs = []
    if not path or not os.path.exists(path):
        return pairs
    with open(path, encoding="utf-8") as f:
        for line in f:
            iris = re.findall(r"<([^>]+)>", line.strip())
            if len(iris) >= 3:   # subject, predicate, object
                pairs.append((iris[0].split("#")[-1], iris[2].split("#")[-1]))
    return pairs


# ── MNLI ─────────────────────────────────────────────────────────────────────

_model = None
_tok   = None

def load_mnli(model_name: str = "facebook/bart-large-mnli") -> None:
    global _model, _tok
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(f"Loading {model_name} ...")
        _tok   = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _model.eval()


def entailment_score(premise: str, hypothesis: str) -> float:
    import torch, torch.nn.functional as F
    load_mnli()
    inp = _tok(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        probs = F.softmax(_model(**inp).logits[0], dim=-1)
    labels = [v.lower() for v in _model.config.id2label.values()]
    return float(probs[labels.index("entailment")])


def split_camel(name: str) -> str:
    words = re.sub(r"([A-Z][a-z]+)", r" \1", name).strip().lower()
    return re.sub(r"\s+", " ", words)


def symmetric_entailment(label_a: str, label_b: str) -> float:
    """Max of both entailment directions."""
    a, b = split_camel(label_a), split_camel(label_b)
    return max(
        entailment_score(f"This is about {a}.", f"This is about {b}."),
        entailment_score(f"This is about {b}.", f"This is about {a}."),
    )


def cross_entailment(src_label: str, tgt_label: str) -> Tuple[float, float]:
    """Return (src->tgt score, tgt->src score) separately."""
    a, b = split_camel(src_label), split_camel(tgt_label)
    fwd = entailment_score(f"This is about {a}.", f"This is about {b}.")
    rev = entailment_score(f"This is about {b}.", f"This is about {a}.")
    return round(fwd, 4), round(rev, 4)


# ── Reporting helpers ─────────────────────────────────────────────────────────

def all_pairs_scored(
    src: Dict[str, str],
    tgt: Dict[str, str],
    threshold: float,
    symmetric: bool = True,
) -> List[Tuple[float, str, str, str, str]]:
    """Return (score, src_id, src_name, tgt_id, tgt_name) for all pairs >= threshold."""
    rows = []
    for si, sl in src.items():
        for ti, tl in tgt.items():
            if symmetric:
                score = symmetric_entailment(sl, tl)
            else:
                score, _ = cross_entailment(sl, tl)
            if score >= threshold:
                rows.append((score, si, sl, ti, tl))
    rows.sort(reverse=True)
    return rows


def print_section(title: str, rows: List[Tuple], limes_pairs: List[Tuple[str, str]]) -> None:
    limes_set = set(limes_pairs)
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    if not rows:
        print("  (no pairs above threshold)")
        return
    print(f"  {'Score':>7}  {'LIMES':>5}  Source label                    ->  Target label")
    print(f"  {'-' * 65}")
    for score, si, sl, ti, tl in rows:
        limes_tag = " [L]" if (si, ti) in limes_set else "    "
        print(f"  {score:.4f}{limes_tag}  {sl:35s} ->  {tl}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    ttl_a: str,
    ttl_b: str,
    limes_accept: str = "",
    limes_review: str = "",
    threshold: float  = 0.35,
    out_csv: str      = "",
) -> None:
    print(f"\nLoading message A: {ttl_a}")
    obs_a  = load_labels(ttl_a, "ConceptualObservable")
    ent_a  = load_labels(ttl_a, "ConceptualEntity")

    print(f"Loading message B: {ttl_b}")
    obs_b  = load_labels(ttl_b, "ConceptualObservable")
    ent_b  = load_labels(ttl_b, "ConceptualEntity")

    print(f"\n  Msg A observables ({len(obs_a)}): {list(obs_a.values())}")
    print(f"  Msg A entities    ({len(ent_a)}): {list(ent_a.values())}")
    print(f"  Msg B observables ({len(obs_b)}): {list(obs_b.values())}")
    print(f"  Msg B entities    ({len(ent_b)}): {list(ent_b.values())}")
    print(f"\nEntailment threshold: {threshold}")

    # Collect LIMES pairs for annotation
    limes_pairs = parse_limes_nt(limes_accept) + parse_limes_nt(limes_review)

    load_mnli()

    # ── Pass 1: Observable A x Observable B ──────────────────────────────────
    print("\n[Pass 1] Observable(A) x Observable(B)  -- symmetric entailment")
    p1 = all_pairs_scored(obs_a, obs_b, threshold, symmetric=True)
    print_section("Pass 1: Observable(A) <-> Observable(B)", p1, limes_pairs)

    # ── Pass 2: Entity A x Observable B ──────────────────────────────────────
    print("\n[Pass 2] Entity(A) -> Observable(B)  -- forward entailment")
    p2 = all_pairs_scored(ent_a, obs_b, threshold, symmetric=False)
    print_section("Pass 2: Entity(A) -> Observable(B)", p2, limes_pairs)

    # ── Pass 3: Entity B x Observable A ──────────────────────────────────────
    print("\n[Pass 3] Entity(B) -> Observable(A)  -- forward entailment")
    p3 = all_pairs_scored(ent_b, obs_a, threshold, symmetric=False)
    print_section("Pass 3: Entity(B) -> Observable(A)", p3, limes_pairs)

    # ── Pass 4: Entity A x Entity B ──────────────────────────────────────────
    print("\n[Pass 4] Entity(A) <-> Entity(B)  -- symmetric entailment")
    p4 = all_pairs_scored(ent_a, ent_b, threshold, symmetric=True)
    print_section("Pass 4: Entity(A) <-> Entity(B)", p4, limes_pairs)

    # ── LIMES summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  LIMES Lexical Matches (for reference)")
    print(f"{'=' * 70}")
    if limes_pairs:
        all_obs = {**obs_a, **obs_b, **ent_a, **ent_b}
        for si, ti in limes_pairs:
            sl = obs_a.get(si) or ent_a.get(si) or si
            tl = obs_b.get(ti) or ent_b.get(ti) or ti
            print(f"  {sl:35s} ->  {tl}")
    else:
        print("  (no LIMES NT files provided or empty)")

    # ── CSV output ────────────────────────────────────────────────────────────
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
        limes_set = set(limes_pairs)
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("pass,score,src_id,src_name,tgt_id,tgt_name,limes_match\n")
            for pass_name, rows in [
                ("obs_A_x_obs_B", p1),
                ("ent_A_x_obs_B", p2),
                ("ent_B_x_obs_A", p3),
                ("ent_A_x_ent_B", p4),
            ]:
                for score, si, sl, ti, tl in rows:
                    limes = (si, ti) in limes_set
                    f.write(f"{pass_name},{score},{si},{sl},{ti},{tl},{limes}\n")
        print(f"\nSaved CSV -> {out_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--ttl-a",       required=True)
    p.add_argument("--ttl-b",       required=True)
    p.add_argument("--limes-accept", default="")
    p.add_argument("--limes-review", default="")
    p.add_argument("--threshold",   type=float, default=0.35)
    p.add_argument("--out-csv",     default="")
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    run(
        ttl_a        = a.ttl_a,
        ttl_b        = a.ttl_b,
        limes_accept = a.limes_accept,
        limes_review = a.limes_review,
        threshold    = a.threshold,
        out_csv      = a.out_csv,
    )
