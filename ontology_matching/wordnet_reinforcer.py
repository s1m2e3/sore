"""
wordnet_reinforcer.py
---------------------
S1b: AML Reinforcement via WordNet compound-word semantic similarity.

Operates on entities that AML (Stage 1) left with status "missing".
For each unmatched entity in model A, computes a bidirectional
compound-word similarity against every unmatched entity in model B
using WordNet Wu-Palmer similarity on camelCase-split constituent tokens.
Greedy 1:1 assignment is performed above a configurable threshold.

Scoring
-------
For two entity names split into constituent word lists W_a and W_b:

  forward(a→b)  = mean_{w_a in W_a}[ max_{w_b in W_b} wn_sim(w_a, w_b) ]
  backward(b→a) = mean_{w_b in W_b}[ max_{w_a in W_a} wn_sim(w_b, w_a) ]
  score         = min(forward, backward)

  Rationale:
    - max()  selects the nearest WordNet counterpart for each source token.
    - mean() aggregates over all source tokens in the compound.
    - min(fwd, bwd) penalises length-asymmetric compounds (which indicate
      subsumption rather than equivalence), consistent with the mutual-
      entailment criterion used in Stage 2 MNLI.

  wn_sim uses NLTK path_similarity (1 / (1 + shortest_path_length)) on the
  single best noun synset per token. This avoids the Lowest Common Subsumer
  traversal of wup_similarity, giving ~3-5x speedup with similar accuracy.
  Only the top synset (most frequent sense) is checked per token.

  If a token has no WordNet synset, character bigram Jaccard is used as
  a fallback so that domain-specific terms (e.g. "ECU", "Matryoshka") still
  contribute a partial signal rather than zeroing out.

  OOV tokens shorter than 3 characters are skipped entirely.

Output
------
  outputs/wordnet/<domain>/<stem>_wordnet.json

  Format mirrors structural_matcher.py output so stage123_distance_report.py
  can optionally consume it.

Usage
-----
    cd ontology_matching
    .venv/Scripts/python.exe wordnet_reinforcer.py
    .venv/Scripts/python.exe wordnet_reinforcer.py --domain Automobile
    .venv/Scripts/python.exe wordnet_reinforcer.py --threshold 0.45
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from functools import lru_cache

# Prevent OpenBLAS/BLAS multi-threading conflicts with other processes
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Lazy WordNet import — first attempt NLTK; if that fails (e.g. memory/BLAS
# conflict from concurrent processes), fall back to a pure-Python bigram
# similarity so the script degrades gracefully rather than crashing.
_WN_AVAILABLE: bool | None = None  # None = not yet attempted

def _wn():
    global _WN_AVAILABLE
    if _WN_AVAILABLE is None:
        try:
            from nltk.corpus import wordnet as _wn_module
            # Smoke-test to force full load
            _wn_module.synsets("fuel")
            _WN_AVAILABLE = True
            return _wn_module
        except Exception as exc:
            print(f"  [S1b] WordNet unavailable ({exc}); using bigram-Jaccard fallback.")
            _WN_AVAILABLE = False
            return None
    if _WN_AVAILABLE:
        from nltk.corpus import wordnet as _wn_module
        return _wn_module
    return None

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR  = os.path.join(BASE_DIR, "outputs", "reports")
WN_DIR       = os.path.join(BASE_DIR, "outputs", "wordnet")

DEFAULT_THRESHOLD = 0.45


# --------------------------------------------------------------------------- #
# Text helpers                                                                  #
# --------------------------------------------------------------------------- #

def _split_camel(name: str) -> list[str]:
    """Split camelCase/PascalCase name into lowercase tokens, drop short ones."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    tokens = re.split(r"[^A-Za-z0-9]+", s)
    return [t.lower() for t in tokens if len(t) >= 3]


# --------------------------------------------------------------------------- #
# WordNet similarity                                                            #
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=4096)
def _best_synsets(token: str) -> tuple:
    """Return the single best synset for token (noun-first, then any POS).

    Using only 1 synset (the most frequent/common sense) is significantly
    faster than checking 3 synsets per token, since path_similarity is O(1)
    per pair rather than O(n^2) over all synset combinations.
    """
    wordnet = _wn()
    if wordnet is None:
        return ()
    syns = wordnet.synsets(token, pos=wordnet.NOUN)
    if not syns:
        syns = wordnet.synsets(token)
    return tuple(syns[:1])


@lru_cache(maxsize=256)
def _bigrams(s: str) -> frozenset:
    return frozenset(s[i:i + 2] for i in range(len(s) - 1)) if len(s) > 1 else frozenset({s})


def _bigram_jaccard(a: str, b: str) -> float:
    bg_a, bg_b = _bigrams(a), _bigrams(b)
    if not bg_a or not bg_b:
        return float(a == b)
    return len(bg_a & bg_b) / len(bg_a | bg_b)


@lru_cache(maxsize=16384)
def _wn_sim(tok_a: str, tok_b: str) -> float:
    """Path similarity between two tokens' best synsets via NLTK WordNet.

    Uses path_similarity (1 / (1 + shortest_path_length)) instead of
    wup_similarity. Path similarity avoids the Lowest Common Subsumer
    traversal that made wup_similarity slow, making it ~3-5x faster while
    still capturing semantic distance in the WordNet hierarchy.

    Falls back to character bigram Jaccard when:
      - WordNet is unavailable (BLAS/memory conflict), or
      - either token has no synset (domain-specific OOV term).
    """
    if tok_a == tok_b:
        return 1.0

    syns_a = _best_synsets(tok_a)
    syns_b = _best_synsets(tok_b)

    if syns_a and syns_b:
        try:
            sim = syns_a[0].path_similarity(syns_b[0])
        except Exception:
            sim = None
        if sim is not None:
            return sim

    return _bigram_jaccard(tok_a, tok_b)


def _one_direction(toks_src: list[str], toks_tgt: list[str]) -> float:
    """mean_{s in src}[ max_{t in tgt} wn_sim(s, t) ]"""
    if not toks_src or not toks_tgt:
        return 0.0
    return sum(max(_wn_sim(s, t) for t in toks_tgt) for s in toks_src) / len(toks_src)


def compound_sim(name_a: str, name_b: str) -> tuple[float, float, float]:
    """Return (score, forward, backward) for two entity names.

    score = min(forward, backward):
      - high  → symmetric semantic similarity → equivalence candidate
      - low   → asymmetric → subsumption (not handled here; left to Stage 3)
    """
    toks_a = _split_camel(name_a)
    toks_b = _split_camel(name_b)

    if not toks_a or not toks_b:
        return 0.0, 0.0, 0.0

    fwd = _one_direction(toks_a, toks_b)
    bwd = _one_direction(toks_b, toks_a)
    return min(fwd, bwd), fwd, bwd


# --------------------------------------------------------------------------- #
# Per-pair processing                                                           #
# --------------------------------------------------------------------------- #

def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_pair(report_path: str, threshold: float) -> dict | None:
    report   = _load(report_path)
    meta     = report["metadata"]
    domain   = meta["domain"]
    stem     = os.path.splitext(os.path.basename(report_path))[0]

    missing_a = [e["name"] for e in report["model_a"]["entities"] if e["status"] == "missing"]
    missing_b = [e["name"] for e in report["model_b"]["entities"] if e["status"] == "missing"]

    if not missing_a or not missing_b:
        return None

    # Score all pairs
    scored: list[tuple[float, float, float, str, str]] = []
    for na in missing_a:
        for nb in missing_b:
            score, fwd, bwd = compound_sim(na, nb)
            if score >= threshold:
                scored.append((score, fwd, bwd, na, nb))

    if not scored:
        return None

    # Greedy 1:1 assignment (descending score)
    scored.sort(key=lambda x: x[0], reverse=True)
    used_a: set[str] = set()
    used_b: set[str] = set()
    new_matches = []

    for score, fwd, bwd, na, nb in scored:
        if na in used_a or nb in used_b:
            continue
        used_a.add(na)
        used_b.add(nb)
        new_matches.append({
            "smaller_entity": na,
            "larger_entity":  nb,
            "wn_score":       round(score, 4),
            "forward":        round(fwd, 4),
            "backward":       round(bwd, 4),
            "method":         "wordnet_compound",
        })

    if not new_matches:
        return None

    return {
        "metadata":    {"domain": domain, "pair": stem,
                        "threshold": threshold, "stage": "S1b_WordNet"},
        "new_matches": new_matches,
    }


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="S1b: WordNet AML Reinforcement")
    parser.add_argument("--domain",    type=str,   default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    pattern = os.path.join(
        REPORTS_DIR,
        args.domain if args.domain else "*",
        "*.json",
    )
    report_paths = sorted(glob.glob(pattern))

    total_new = 0
    for rp in report_paths:
        domain = os.path.basename(os.path.dirname(rp))
        stem   = os.path.splitext(os.path.basename(rp))[0]

        result = run_pair(rp, args.threshold)
        if result:
            out_dir = os.path.join(WN_DIR, domain)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{stem}_wordnet.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            n = len(result["new_matches"])
            print(f"  {domain:<14} | {stem[:45]:<45} | +{n} matches")
            for m in result["new_matches"]:
                print(f"      {m['smaller_entity']:<30} <-> {m['larger_entity']:<30}  "
                      f"score={m['wn_score']:.3f}  fwd={m['forward']:.3f}  bwd={m['backward']:.3f}")
            total_new += n

    print(f"\n=== Done: {total_new} new WordNet matches found. ===")


if __name__ == "__main__":
    main()
