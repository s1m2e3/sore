"""
lin_ic_stage.py
---------------
Layer 3 — Lin-IC Final Scoring

Computes Lin Information-Content similarity for all L1 matched and L2
discovered entity pairs, using NLTK's WordNet IC (Brown corpus, Resnik
add-1 smoothing) and WordNet taxonomy for Least Common Subsumer (LCS).

Formula:
    Lin(a, b) = 2 * IC(LCS(a, b)) / (IC(a) + IC(b))

For multi-word CamelCase entities, all N×M token combinations are scored
and we report max_lin_ic, avg_lin_ic.  The LCS column shows the WordNet
ancestor that a and b share — a high IC(LCS) means a specific, informative
common ancestor; a low IC(LCS) (< 3) means only a generic root like
"entity.n.01" links them, flagging a weak match.

Output CSV columns:
    entity_a, entity_b, layer, semantic_label,
    wup_score, max_wup, avg_wup,
    lin_ic, max_lin_ic, avg_lin_ic,
    lcs, ic_lcs

Usage (standalone):
    venv/Scripts/python.exe enriched_ontology_matching/lin_ic_stage.py \\
        --csv enriched_ontology_matching/outputs/enriched/<pair>.csv \\
        --out enriched_ontology_matching/outputs/lin_ic/<domain_key>_lin_ic.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

from root_comparator import split_camel

# ---------------------------------------------------------------------------
# Stopword tokens — removed before comparison (too generic to carry IC signal)
# ---------------------------------------------------------------------------
_STOP_TOKENS = {
    "system", "model", "type", "base", "item", "unit", "data", "info",
    "module", "assembly", "component", "part", "sub", "link", "node",
    "entity", "object", "class", "instance", "set", "list",
}


def _filter_tokens(tokens: list[str]) -> list[str]:
    out = [t for t in tokens if len(t) > 2 and t not in _STOP_TOKENS]
    return out or (tokens[:1] if tokens else [""])


# ---------------------------------------------------------------------------
# NLTK IC loader (singleton)
# ---------------------------------------------------------------------------

_IC_DATA = None
_IC_AVAILABLE = False


def _get_ic():
    global _IC_DATA, _IC_AVAILABLE
    if _IC_DATA is not None:
        return _IC_DATA, _IC_AVAILABLE
    try:
        from nltk.corpus import wordnet_ic
        for ic_name in (
            "ic-brown-resnik-add1.dat",
            "ic-brown.dat",
            "ic-semcor-add1.dat",
        ):
            try:
                _IC_DATA = wordnet_ic.ic(ic_name)
                _IC_AVAILABLE = True
                print(f"[Lin-IC] Loaded IC corpus: {ic_name}")
                return _IC_DATA, True
            except Exception:
                continue
        # Try downloading as fallback
        import nltk
        nltk.download("wordnet_ic", quiet=True)
        _IC_DATA = wordnet_ic.ic("ic-brown-resnik-add1.dat")
        _IC_AVAILABLE = True
        print("[Lin-IC] Downloaded and loaded IC corpus.")
    except Exception as e:
        print(f"[Lin-IC] WARNING: IC corpus unavailable ({e}) — Lin-IC will be 0.")
        _IC_DATA = None
        _IC_AVAILABLE = False
    return _IC_DATA, _IC_AVAILABLE


# ---------------------------------------------------------------------------
# Per-token Lin-IC computation
# ---------------------------------------------------------------------------

def lin_ic_token_pair(ta: str, tb: str, ic) -> dict:
    """
    Lin-IC similarity between two tokens.

    Returns dict with:
        lin_ic   : Lin similarity in [0, 1]
        lcs      : WordNet name of Least Common Subsumer (e.g. 'artifact.n.01')
        ic_lcs   : IC(LCS) — specificity of the common ancestor
        ic_a     : IC of best synset for ta
        ic_b     : IC of best synset for tb
    """
    from nltk.corpus import wordnet as wn
    from nltk.corpus.reader.wordnet import information_content

    empty = {"lin_ic": 0.0, "lcs": "", "ic_lcs": 0.0, "ic_a": 0.0, "ic_b": 0.0}

    if ta == tb:
        syns = wn.synsets(ta, pos=wn.NOUN)
        if syns:
            try:
                ic_val = information_content(syns[0], ic)
                return {
                    "lin_ic": 1.0, "lcs": syns[0].name(),
                    "ic_lcs": round(ic_val, 4),
                    "ic_a": round(ic_val, 4), "ic_b": round(ic_val, 4),
                }
            except Exception:
                pass
        return {"lin_ic": 1.0, "lcs": ta, "ic_lcs": 0.0, "ic_a": 0.0, "ic_b": 0.0}

    # Prefer noun synsets; fall back to unrestricted if needed
    syns_a = wn.synsets(ta, pos=wn.NOUN)[:3] or wn.synsets(ta)[:3]
    syns_b = wn.synsets(tb, pos=wn.NOUN)[:3] or wn.synsets(tb)[:3]

    if not syns_a or not syns_b:
        return empty

    best_lin = 0.0
    best_lcs = ""
    best_ic_lcs = 0.0
    best_ic_a = 0.0
    best_ic_b = 0.0

    for sa in syns_a:
        for sb in syns_b:
            if sa.pos() != sb.pos():
                continue
            try:
                lin = sa.lin_similarity(sb, ic)
                if lin is None:
                    continue
                lin = max(0.0, lin)   # clamp floating-point negatives
                if lin > best_lin:
                    best_lin = lin
                    lcs_list = sa.lowest_common_hypernyms(sb, use_min_depth=True)
                    if lcs_list:
                        best_lcs = lcs_list[0].name()
                        try:
                            best_ic_lcs = information_content(lcs_list[0], ic)
                        except Exception:
                            best_ic_lcs = 0.0
                    try:
                        best_ic_a = information_content(sa, ic)
                        best_ic_b = information_content(sb, ic)
                    except Exception:
                        pass
            except Exception:
                continue

    return {
        "lin_ic":  round(best_lin, 4),
        "lcs":     best_lcs,
        "ic_lcs":  round(best_ic_lcs, 4),
        "ic_a":    round(best_ic_a, 4),
        "ic_b":    round(best_ic_b, 4),
    }


# ---------------------------------------------------------------------------
# N×M entity pair aggregation
# ---------------------------------------------------------------------------

def lin_ic_entity_pair(entity_a: str, entity_b: str, ic) -> dict:
    """
    N×M Lin-IC across all CamelCase token combinations.

    Returns:
        lin_ic           — best single-token-pair Lin-IC
        max_lin_ic       — max across all N×M pairs
        avg_lin_ic       — average across all N×M pairs
        lcs              — LCS of the best pair
        ic_lcs           — IC of that LCS
        token_lin_details — semicolon-separated per-token breakdown:
                           "ta/tb:lin_ic(lcs)" for every (ta, tb) pair
    """
    tokens_a = _filter_tokens([t.lower() for t in split_camel(entity_a)])
    tokens_b = _filter_tokens([t.lower() for t in split_camel(entity_b)])

    all_results = []
    token_pairs = []
    for ta in tokens_a:
        for tb in tokens_b:
            r = lin_ic_token_pair(ta, tb, ic)
            all_results.append(r)
            token_pairs.append((ta, tb, r))

    if not all_results:
        return {
            "lin_ic": 0.0, "max_lin_ic": 0.0, "avg_lin_ic": 0.0,
            "lcs": "", "ic_lcs": 0.0, "token_lin_details": "",
        }

    best = max(all_results, key=lambda r: r["lin_ic"])
    lin_vals = [r["lin_ic"] for r in all_results]

    # Build compact per-token detail string: "ta/tb:0.97(motor.n.01)"
    details = "; ".join(
        f"{ta}/{tb}:{r['lin_ic']}({r['lcs']})"
        for ta, tb, r in token_pairs
    )

    return {
        "lin_ic":            round(best["lin_ic"], 4),
        "max_lin_ic":        round(max(lin_vals), 4),
        "avg_lin_ic":        round(sum(lin_vals) / len(lin_vals), 4),
        "lcs":               best["lcs"],
        "ic_lcs":            round(best["ic_lcs"], 4),
        "token_lin_details": details,
    }


# ---------------------------------------------------------------------------
# CSV augmentation
# ---------------------------------------------------------------------------

_LIN_FIELDS = [
    "entity_a", "entity_b", "layer", "semantic_label",
    "wup_score", "max_wup", "avg_wup",
    "lin_ic", "max_lin_ic", "avg_lin_ic",
    "lcs", "ic_lcs",
    "token_lin_details",
]


def run_lin_ic(enriched_csv: Path, out_csv: Path) -> list[dict]:
    """Read a per-pair enriched CSV, compute Lin-IC for every row, write to out_csv.

    Returns the augmented rows. If the NLTK IC corpus is unavailable all lin_ic
    values will be 0.0 (pipeline continues without failing).

    Raises FileNotFoundError if enriched_csv does not exist.
    """
    if not Path(enriched_csv).exists():
        raise FileNotFoundError(
            f"Enriched CSV not found: {enriched_csv}\n"
            "Run enriched_matcher.py for this pair first."
        )
    ic, available = _get_ic()

    with open(enriched_csv, newline="", encoding="utf-8") as fh:
        rows_in = list(csv.DictReader(fh))

    print(f"[Lin-IC] Scoring {len(rows_in)} pairs ...")

    results: list[dict] = []
    for row in rows_in:
        ea = row.get("entity_a", "")
        eb = row.get("entity_b", "")

        if available and ic is not None:
            lin = lin_ic_entity_pair(ea, eb, ic)
        else:
            lin = {
                "lin_ic": 0.0, "max_lin_ic": 0.0, "avg_lin_ic": 0.0,
                "lcs": "", "ic_lcs": 0.0, "token_lin_details": "",
            }

        results.append({
            "entity_a":          ea,
            "entity_b":          eb,
            "layer":             row.get("layer", ""),
            "semantic_label":    row.get("semantic_label", ""),
            "wup_score":         row.get("wup_score", ""),
            "max_wup":           row.get("max_wup", ""),
            "avg_wup":           row.get("avg_wup", ""),
            "lin_ic":            lin["lin_ic"],
            "max_lin_ic":        lin["max_lin_ic"],
            "avg_lin_ic":        lin["avg_lin_ic"],
            "lcs":               lin["lcs"],
            "ic_lcs":            lin["ic_lcs"],
            "token_lin_details": lin.get("token_lin_details", ""),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_LIN_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print(f"[Lin-IC] Written {len(results)} rows -> {out_csv}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Layer 3: Lin-IC scoring for ontology match pairs."
    )
    parser.add_argument("--csv", required=True,
                        help="Per-pair enriched CSV (from enriched_matcher.py)")
    parser.add_argument("--out", required=True,
                        help="Output Lin-IC CSV path")
    args = parser.parse_args()
    run_lin_ic(Path(args.csv), Path(args.out))
