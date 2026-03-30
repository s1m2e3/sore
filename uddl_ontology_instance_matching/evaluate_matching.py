"""
evaluate_matching.py
--------------------
Evaluates UDDL message matching quality at the ConceptualObservable level.

Fully parameterised -- no hardcoded instance IDs, file paths, or ontology IRIs.
All inputs are derived from the supplied Turtle files at runtime.

Usage
-----
  python evaluate_matching.py \\
      --src  outputs/msg_A.ttl \\
      --tgt  outputs/msg_B.ttl \\
      [--limes-accept  outputs/limes_obs_accept.nt] \\
      [--limes-review  outputs/limes_obs_review.nt] \\
      [--ground-truth  ground_truth.json] \\
      [--no-mnli] \\
      [--out-csv       outputs/matching_results.csv]

Ground-truth JSON format (optional; omit for unsupervised alignment output):
  [
    {"src": "obs_altitude_001_obs", "tgt": "obs_height_002_obs"},
    ...
  ]
  OR specify by observable NAME instead of local IRI:
  [
    {"src_name": "Altitude", "tgt_name": "HeightAboveMeanSeaLevel"},
    ...
  ]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from rdflib import Graph, RDF, URIRef

# ── Schema constants (metamodel IRI only -- no instance-specific values) ──────
META   = "http://example.org/uddl_metamodel.owl#"
OWL_SA = "http://www.w3.org/2002/07/owl#sameAs"


# ── RDF helpers ───────────────────────────────────────────────────────────────

def load_observable_labels(ttl_path: str) -> Dict[str, str]:
    """Return {local_id: hasName_label} for all ConceptualObservable instances
    in the given Turtle file. Fully derived from the file -- no hardcoding."""
    g = Graph()
    g.parse(ttl_path, format="turtle")
    result: Dict[str, str] = {}
    for s, _, _ in g.triples((None, RDF.type, URIRef(META + "ConceptualObservable"))):
        names = list(g.objects(s, URIRef(META + "hasName")))
        if names:
            local = str(s).split("#")[-1]
            result[local] = str(names[0])
    return result


def load_measurement_labels(ttl_path: str) -> Dict[str, Dict[str, str]]:
    """Return {local_id: {value_type, unit, measurement_system}} for all
    LogicalMeasurement instances. Derived entirely from the Turtle file."""
    g = Graph()
    g.parse(ttl_path, format="turtle")
    result: Dict[str, Dict[str, str]] = {}
    for s, _, _ in g.triples((None, RDF.type, URIRef(META + "LogicalMeasurement"))):
        local = str(s).split("#")[-1]
        info: Dict[str, str] = {}
        # Follow object properties to named individuals for value_type and unit
        for vt in g.objects(s, URIRef(META + "measurementValueType")):
            vtn = list(g.objects(vt, URIRef(META + "valueTypeName")))
            if vtn:
                info["value_type"] = str(vtn[0])
        for u in g.objects(s, URIRef(META + "measurementUnit")):
            sym = list(g.objects(u, URIRef(META + "symbol")))
            if sym:
                info["unit"] = str(sym[0])
        for ms in g.objects(s, URIRef(META + "measurementSystem")):
            msn = list(g.objects(ms, URIRef(META + "measurementSystemName")))
            if msn:
                info["measurement_system"] = str(msn[0])
        names = list(g.objects(s, URIRef(META + "hasName")))
        if names:
            info["name"] = str(names[0])
        result[local] = info
    return result


def parse_limes_nt(path: str) -> Set[Tuple[str, str]]:
    """Read a LIMES NT alignment file; return (src_local_id, tgt_local_id) pairs."""
    pairs: Set[Tuple[str, str]] = set()
    if not os.path.exists(path):
        return pairs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            iris = re.findall(r"<([^>]+)>", line)
            if len(iris) >= 3:   # subject, predicate, object
                pairs.add((iris[0].split("#")[-1], iris[2].split("#")[-1]))
    return pairs


# ── String similarity helpers ─────────────────────────────────────────────────

def split_camel(name: str) -> str:
    """'TrueAirspeed' -> 'true airspeed'"""
    words = re.sub(r"([A-Z][a-z]+)", r" \1", name).strip().lower()
    return re.sub(r"\s+", " ", words)


def trigram_set(text: str) -> Set[str]:
    t = text.lower()
    return {t[i:i+3] for i in range(len(t) - 2)} if len(t) >= 3 else set(t)


def trigram_jaccard(a: str, b: str) -> float:
    sa, sb = trigram_set(a), trigram_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def split_trigram_jaccard(a: str, b: str) -> float:
    return trigram_jaccard(split_camel(a), split_camel(b))


# ── Alignment helpers ─────────────────────────────────────────────────────────

def greedy_align(
    src_labels: Dict[str, str],
    tgt_labels: Dict[str, str],
    score_fn,
    threshold: float = 0.0,
) -> Dict[Tuple[str, str], float]:
    """Greedy 1-to-1 alignment by descending score.
    Returns {(src_id, tgt_id): score} for all accepted pairs."""
    scores = [
        (score_fn(sl, tl), si, ti)
        for si, sl in src_labels.items()
        for ti, tl in tgt_labels.items()
    ]
    scores.sort(reverse=True)
    used_src: Set[str] = set()
    used_tgt: Set[str] = set()
    pairs: Dict[Tuple[str, str], float] = {}
    for score, si, ti in scores:
        if score < threshold:
            continue
        if si not in used_src and ti not in used_tgt:
            pairs[(si, ti)] = round(score, 4)
            used_src.add(si)
            used_tgt.add(ti)
    return pairs


def prf1(
    predicted,   # Set[Tuple] or Dict[Tuple, float]
    ground_truth: Set[Tuple[str, str]],
) -> Tuple[float, float, float]:
    pred_set = set(predicted)  # works for both Set and Dict.keys()
    tp = len(pred_set & ground_truth)
    p  = tp / len(pred_set)     if pred_set     else 0.0
    r  = tp / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * p * r / (p + r)   if (p + r) > 0  else 0.0
    return round(p, 3), round(r, 3), round(f1, 3)


# ── MNLI semantic scorer ──────────────────────────────────────────────────────

_mnli_model = None
_mnli_tok   = None


def _load_mnli(model_name: str = "facebook/bart-large-mnli") -> None:
    global _mnli_model, _mnli_tok
    if _mnli_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(f"  Loading {model_name} ...")
        _mnli_tok   = AutoTokenizer.from_pretrained(model_name)
        _mnli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _mnli_model.eval()


def mnli_entailment(premise: str, hypothesis: str) -> float:
    import torch
    import torch.nn.functional as F
    _load_mnli()
    inputs = _mnli_tok(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        probs = F.softmax(_mnli_model(**inputs).logits[0], dim=-1)
    labels = [v.lower() for v in _mnli_model.config.id2label.values()]
    return float(probs[labels.index("entailment")])


def mnli_score(label_a: str, label_b: str) -> float:
    """Symmetric MNLI entailment score between two concept labels."""
    a = split_camel(label_a)
    b = split_camel(label_b)
    return max(
        mnli_entailment(f"This observable is about {a}.",
                        f"This observable is about {b}."),
        mnli_entailment(f"This observable is about {b}.",
                        f"This observable is about {a}."),
    )


# ── Ground-truth loader ───────────────────────────────────────────────────────

def load_ground_truth(
    path: Optional[str],
    src_labels: Dict[str, str],
    tgt_labels: Dict[str, str],
) -> Optional[Set[Tuple[str, str]]]:
    """Load ground truth from a JSON file.

    Accepts two formats:
      [{"src": "local_id_a", "tgt": "local_id_b"}, ...]
      [{"src_name": "Altitude", "tgt_name": "HeightAboveMeanSeaLevel"}, ...]
    Returns None if path is not provided.
    """
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    gt: Set[Tuple[str, str]] = set()
    # Build reverse name->id lookups
    name_to_src = {v: k for k, v in src_labels.items()}
    name_to_tgt = {v: k for k, v in tgt_labels.items()}
    for rec in records:
        if "src" in rec and "tgt" in rec:
            gt.add((rec["src"], rec["tgt"]))
        elif "src_name" in rec and "tgt_name" in rec:
            s = name_to_src.get(rec["src_name"])
            t = name_to_tgt.get(rec["tgt_name"])
            if s and t:
                gt.add((s, t))
    return gt


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation(
    src_ttl: str,
    tgt_ttl: str,
    limes_accept: str  = "",
    limes_review: str  = "",
    ground_truth_path: Optional[str] = None,
    run_mnli: bool     = True,
    out_csv: str       = "",
) -> Dict:
    """
    Run all matchers and return a results dict.
    All file paths are parameters -- nothing is hardcoded.
    """
    print(f"Loading src: {src_ttl}")
    src_labels = load_observable_labels(src_ttl)
    print(f"Loading tgt: {tgt_ttl}")
    tgt_labels = load_observable_labels(tgt_ttl)

    print(f"  Src observables ({len(src_labels)}): {list(src_labels.values())}")
    print(f"  Tgt observables ({len(tgt_labels)}): {list(tgt_labels.values())}")
    print()

    ground_truth = load_ground_truth(ground_truth_path, src_labels, tgt_labels)

    # results: method -> {(src_id, tgt_id): score}
    results: Dict[str, Dict[Tuple[str, str], float]] = {}

    # 1. LIMES lexical output (scores not available from NT file, set to 1.0 for accepted)
    if limes_accept or limes_review:
        limes_raw = parse_limes_nt(limes_accept) | parse_limes_nt(limes_review)
        limes_valid = [(s, t) for s, t in limes_raw
                       if s in src_labels and t in tgt_labels]
        used_s: Set[str] = set()
        used_t: Set[str] = set()
        limes_1to1: Dict[Tuple[str, str], float] = {}
        for s, t in sorted(limes_valid):
            if s not in used_s and t not in used_t:
                limes_1to1[(s, t)] = 1.0   # LIMES NT doesn't carry scores
                used_s.add(s)
                used_t.add(t)
        results["LIMES trigrams (lexical)"] = limes_1to1

    # 2. Trigram Jaccard (Python)
    results["Trigram Jaccard (Python)"] = greedy_align(
        src_labels, tgt_labels,
        lambda a, b: trigram_jaccard(a, b),
        threshold=0.2,
    )

    # 3. Split-CamelCase trigrams
    results["Split-CamelCase trigrams"] = greedy_align(
        src_labels, tgt_labels,
        split_trigram_jaccard,
        threshold=0.2,
    )

    # 4. MNLI semantic (independent of ground truth -- runs regardless)
    if run_mnli:
        print("Running MNLI semantic matching ...")
        results["MNLI entailment (semantic)"] = greedy_align(
            src_labels, tgt_labels,
            mnli_score,
            threshold=0.3,
        )

    # ── Print alignment details
    print("\n" + "=" * 70)
    print("ALIGNMENT DETAILS" + (" vs GROUND TRUTH" if ground_truth else ""))
    print("=" * 70)
    for method, scored_pairs in results.items():
        print(f"\n-- {method} --")
        for (s, t), score in sorted(scored_pairs.items(),
                                    key=lambda x: src_labels.get(x[0][0], x[0][0])):
            sl = src_labels.get(s, s)
            tl = tgt_labels.get(t, t)
            score_str = f"[{score:.3f}]"
            if ground_truth is not None:
                tag = "OK" if (s, t) in ground_truth else "XX"
                print(f"  {tag} {score_str:8s} {sl:40s} <-> {tl}")
            else:
                print(f"  {score_str:8s} {sl:40s} <-> {tl}")
        if ground_truth is not None:
            missed = ground_truth - set(scored_pairs)
            if missed:
                print(f"  [missed {len(missed)} ground-truth pairs]")
                for s, t in sorted(missed):
                    print(f"    - {src_labels.get(s, s):40s} <-> {tgt_labels.get(t, t)}")

    # ── Summary table
    if ground_truth is not None:
        print("\n" + "=" * 70)
        print(f"{'Method':<35} {'P':>6} {'R':>6} {'F1':>6}  Pairs")
        print("-" * 70)
        for method, scored_pairs in results.items():
            p, r, f1 = prf1(scored_pairs, ground_truth)
            print(f"{method:<35} {p:>6.3f} {r:>6.3f} {f1:>6.3f}  {len(scored_pairs)}")
        print("-" * 70)
        print(f"Ground truth: {len(ground_truth)} pairs\n")
    else:
        print("\n" + "=" * 70)
        print(f"{'Method':<35} {'Score':>7}  Src label -> Tgt label")
        print("-" * 70)
        for method, scored_pairs in results.items():
            print(f"\n  {method} ({len(scored_pairs)} pairs):")
            for (s, t), score in sorted(scored_pairs.items(), reverse=True,
                                        key=lambda x: x[1]):
                sl = src_labels.get(s, s)
                tl = tgt_labels.get(t, t)
                print(f"    {score:.3f}  {sl} -> {tl}")
        print()

    # ── Save CSV
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("method,src_id,src_name,tgt_id,tgt_name,score,correct\n")
            for method, scored_pairs in results.items():
                for (s, t), score in sorted(scored_pairs.items()):
                    correct = (s, t) in ground_truth if ground_truth is not None else ""
                    f.write(f'"{method}",{s},{src_labels.get(s, "")},'
                            f'{t},{tgt_labels.get(t, "")},{score},{correct}\n')
        print(f"Saved CSV -> {out_csv}")

    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate UDDL message matching")
    p.add_argument("--src",          required=True,  help="Source Turtle file (reference/human)")
    p.add_argument("--tgt",          required=True,  help="Target Turtle file (LLM-generated)")
    p.add_argument("--limes-accept", default="",     help="LIMES accept NT output file")
    p.add_argument("--limes-review", default="",     help="LIMES review NT output file")
    p.add_argument("--ground-truth", default=None,   help="Ground-truth JSON file (optional)")
    p.add_argument("--no-mnli",      action="store_true", help="Skip MNLI semantic matching")
    p.add_argument("--out-csv",      default="",     help="Path for CSV results output")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        src_ttl           = args.src,
        tgt_ttl           = args.tgt,
        limes_accept      = args.limes_accept,
        limes_review      = args.limes_review,
        ground_truth_path = args.ground_truth,
        run_mnli          = not args.no_mnli,
        out_csv           = args.out_csv,
    )
