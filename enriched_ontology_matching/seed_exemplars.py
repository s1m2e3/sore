"""
seed_exemplars.py
-----------------
Populates exemplar_phrases in config/canonical_relations.json.

Primary source  — ConceptNet 5 REST API (Speer et al., 2017):
  https://doi.org/10.1609/aaai.v31i1.11164
  API: https://api.conceptnet.io

Automatic fallback — if the API is unreachable, short verb phrases are
  extracted directly from each canonical's definition text.  Those
  definitions are themselves drawn from ConceptNet/RO/SSN literature, so
  the fallback output remains grounded in the same authoritative sources.

In both cases the resulting phrases replace the full definition as the
BERT comparison target in relation_normalizer.py, fixing the phrase-vs-
definition length mismatch that causes systematically low cosine scores.

Usage
-----
    python scripts/seed_exemplars.py            # fetch and save
    python scripts/seed_exemplars.py --dry-run  # print phrases, don't save
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# NLTK / WordNet bootstrap (needed for the definition-extraction fallback)
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.corpus import wordnet as wn

    for _res, _kind in [
        ("wordnet",  "corpora"),
        ("omw-1.4",  "corpora"),
        ("punkt",    "tokenizers"),
        ("punkt_tab","tokenizers"),
    ]:
        try:
            nltk.data.find(f"{_kind}/{_res}")
        except LookupError:
            nltk.download(_res, quiet=True)

    _NLTK = True
except ImportError:
    _NLTK = False

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
_REPO_ROOT  = Path(__file__).resolve().parent.parent
_CANONICAL  = _REPO_ROOT / "config" / "canonical_relations.json"
_API_BASE   = "https://api.conceptnet.io"
_EDGE_LIMIT = 200
_MAX_KEEP   = 25


# ---------------------------------------------------------------------------
# ConceptNet API path
# ---------------------------------------------------------------------------

def _fetch_edges(rel_iri: str) -> list[dict]:
    url = f"{_API_BASE}/query?rel={rel_iri}&limit={_EDGE_LIMIT}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json",
                 "User-Agent": "sore-relation-seeder/1.0"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read()).get("edges", [])


_BRACKETS = re.compile(r"\[\[.*?\]\]")


def _phrase_from_surface(surface_text: str) -> str | None:
    """Extract the verb phrase from '[[X]] is part of [[Y]]' → 'is part of'."""
    cleaned = _BRACKETS.sub("___", surface_text).strip()
    m = re.match(r"^___\s+(.+?)\s+___$", cleaned)
    if not m:
        return None
    phrase = m.group(1).strip().lower()
    words = phrase.split()
    if not (1 <= len(words) <= 6):
        return None
    if not all(re.match(r"[a-z'\-]+$", w) for w in words):
        return None
    return phrase


def _exemplars_from_api(rel: dict) -> list[str] | None:
    """
    Returns a list of phrases fetched from ConceptNet, or None if the API
    failed (caller should fall back to definition extraction).
    Relations using the generic /r/RelatedTo IRI are skipped immediately
    (too broad to yield useful phrases).
    """
    iri = rel.get("conceptnet_iri", "")
    if not iri or iri == "/r/RelatedTo":
        return None

    try:
        edges = _fetch_edges(iri)
    except (urllib.error.URLError, urllib.error.HTTPError, Exception):
        return None   # signal to caller: use fallback

    phrases: list[str] = []
    seen: set[str] = set()
    for edge in edges:
        st = edge.get("surfaceText")
        if not st:
            continue
        phrase = _phrase_from_surface(st)
        if phrase and phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)

    return phrases[:_MAX_KEEP]


# ---------------------------------------------------------------------------
# Definition-extraction fallback
# ---------------------------------------------------------------------------

def _phrases_from_definition(definition: str) -> list[str]:
    """
    Extract short verb phrases directly from a canonical definition string.

    The definition text is itself drawn from ConceptNet/RO/SSN literature,
    so phrases extracted from it remain grounded in the same authoritative
    sources — just restructured into short forms that match the length and
    style of the association names BERT will compare them against.
    """
    text = re.sub(r"\bX\b", "__X__", definition)
    text = re.sub(r"\bY\b", "__Y__", text)

    seen: set[str] = set()
    candidates: list[str] = []

    def _add(raw: str) -> None:
        # Normalise whitespace; strip trailing filler words/prepositions
        p = re.sub(r"\s+", " ", raw).strip().lower()
        p = re.sub(r"\s+(a|an|the|as|its|or)$", "", p)
        words = p.split()
        if p and 1 <= len(words) <= 6 and p not in seen and re.search(r"[a-z]", p):
            seen.add(p)
            candidates.append(p)

    # Pattern A: "__X__ <phrase> __Y__" — the direct X-to-Y verb phrase
    for m in re.finditer(r"__X__\s+(.+?)\s+__Y__", text):
        chunk = m.group(1)
        for raw in re.split(r",\s*(?:or\s+)?|\s+or\s+", chunk):
            raw = raw.split(" that ")[0]   # drop subordinate "that" clause
            _add(raw)

    # Pattern B: "that <phrase> __Y__" — subordinate clause list before Y
    for m in re.finditer(r"\bthat\s+(.+?)\s+__Y__", text):
        chunk = m.group(1)
        for raw in re.split(r",\s*(?:or\s+)?|\s+or\s+", chunk):
            _add(raw)

    # Pattern C: "__X__ is <list>." — passive/unary list (ReceivesAction, etc.)
    for m in re.finditer(r"__X__\s+is\s+([^.]+?)\.", text):
        chunk = m.group(1)
        if "__Y__" in chunk:
            continue   # already handled by A / B
        for raw in re.split(r",\s*(?:or\s+)?|\s+or\s+", chunk):
            raw = raw.strip()
            if raw:
                _add("is " + raw)

    # Pattern D: "__X__ <verblist>" not followed by __Y__ on the same line
    # Catches flipped sentences like "something __X__ possesses, includes, ..."
    for m in re.finditer(
        r"__X__\s+((?:[a-z][a-z\-]+"
        r"(?:\s+(?:on|to|into|with|from|within|in|at|for|up))?"
        r"(?:,\s*)?)+)",
        text,
    ):
        chunk = m.group(1)
        if "__Y__" in chunk:
            continue
        for raw in re.split(r",\s*(?:or\s+)?|\s+or\s+", chunk):
            raw = re.sub(r"\s+as\s+.+$", "", raw)  # drop "as part of ..."
            _add(raw)

    # Verb filter: keep phrases where at least one word has a WordNet verb sense.
    # WordNet lookup is unambiguous unlike POS tagging without context —
    # "fires", "couples", "bridges", "interfaces" all have verb synsets.
    # Copulas ("is", "are") are accepted on their own to preserve "is a",
    # "is used for" etc., but "is" alone is dropped as noise.
    _COPULAS = {"is", "are", "was", "were", "be"}
    if _NLTK:
        result = []
        for p in candidates:
            words = p.split()
            if p in _COPULAS:       # bare copula — useless as a BERT target
                continue
            has_verb = any(wn.synsets(w, pos=wn.VERB) for w in words)
            if not has_verb:
                # Also accept "is/are + past-participle" passives whose head
                # copula is the only verb form but the phrase is meaningful
                has_verb = words[0] in _COPULAS and len(words) >= 2
            if has_verb:
                result.append(p)
        return result

    # No NLTK: drop bare copulas, return everything else
    return [p for p in candidates if p not in _COPULAS]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _exemplars_for(rel: dict) -> tuple[list[str], str]:
    """
    Return (phrases, source) where source is 'conceptnet_api' or 'definition'.
    Always returns a non-empty list (falls back to definition if needed).
    """
    # Try ConceptNet API first
    api_result = _exemplars_from_api(rel)
    if api_result is not None and api_result:
        return api_result, "conceptnet_api"

    # Automatic fallback: extract from definition text
    phrases = _phrases_from_definition(rel.get("definition", ""))
    return phrases[:_MAX_KEEP], "definition"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Seed exemplar phrases into canonical_relations.json "
                    "(ConceptNet API with automatic definition fallback)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print phrases but do not modify canonical_relations.json",
    )
    args = parser.parse_args(argv)

    canonicals: list[dict] = json.loads(_CANONICAL.read_text(encoding="utf-8"))

    sources: dict[str, int] = {"conceptnet_api": 0, "definition": 0}

    for rel in canonicals:
        phrases, source = _exemplars_for(rel)
        sources[source] += 1
        rel["exemplar_phrases"] = phrases
        tag = "[API]" if source == "conceptnet_api" else "[def]"
        print(f"  {tag} {rel['id']:18s}  {len(phrases)} phrases")
        if args.dry_run:
            for p in phrases:
                print(f"        {p}")
        time.sleep(0.3)

    api_count = sources["conceptnet_api"]
    def_count = sources["definition"]
    print(f"\nSources: {api_count} from ConceptNet API, {def_count} from definition fallback")

    if args.dry_run:
        print("(dry-run: canonical_relations.json not modified)")
        return

    _CANONICAL.write_text(
        json.dumps(canonicals, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    total = sum(len(r.get("exemplar_phrases", [])) for r in canonicals)
    print(f"Saved {total} exemplar phrases to {_CANONICAL.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
