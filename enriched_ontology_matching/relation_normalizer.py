"""
relation_normalizer.py
----------------------
Extracts all association names from conceptual-model JSONs, then maps each
one to a canonical relation from config/canonical_relations.json using a
combined WUP + BERT cosine similarity score.

Scoring
-------
  WUP:  content words (nouns/verbs) are extracted from each canonical
        relation's "definition" field via NLTK POS tagging.  No hand-
        curated lemma lists are used.  Max WUP similarity between the
        association tokens and those extracted content words.

  BERT: the entity-stripped relation phrase (e.g. "charges" extracted from
        "AlternatorChargesBattery") is compared against the mean embedding of
        each canonical's exemplar_phrases (short ConceptNet surface forms
        seeded by seed_exemplars.py).  Falls back to the full split phrase
        when no stripped phrase is available, and to the definition when no
        exemplar phrases exist.  When all relation tokens are function words
        (prepositions, adverbs — no WordNet synsets) the WUP weight is dropped
        to 0 so BERT carries the full score; WUP is 0 for such tokens anyway.

  Combined: 0.5 * wup + 0.5 * bert  (equal weight).

Outputs
-------
  config/relation_map.json                            -- association_name → {canonical, score, ...}
  enriched_ontology_matching/association_inventory.csv -- full table with both sub-scores

Usage
-----
    python enriched_ontology_matching/relation_normalizer.py

    # Only flag mappings below a combined score of 0.5:
    python enriched_ontology_matching/relation_normalizer.py --threshold 0.5

    # Equal-weight vs. WUP-heavy:
    python enriched_ontology_matching/relation_normalizer.py --wup-weight 0.7

    # Scan a different input directory:
    python enriched_ontology_matching/relation_normalizer.py --input-dir enriched_ontology_matching/inputs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# NLTK bootstrap
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer as _WNL
except ImportError:
    print("ERROR: nltk not installed. Run: pip install nltk", file=sys.stderr)
    sys.exit(1)

for _res, _kind in [
    ("wordnet",                        "corpora"),
    ("omw-1.4",                        "corpora"),
    ("averaged_perceptron_tagger",     "taggers"),
    ("averaged_perceptron_tagger_eng", "taggers"),
    ("punkt",                          "tokenizers"),
    ("punkt_tab",                      "tokenizers"),
]:
    try:
        nltk.data.find(f"{_kind}/{_res}")
    except LookupError:
        nltk.download(_res, quiet=True)

# ---------------------------------------------------------------------------
# Sentence-Transformers bootstrap
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _BERT_MODEL   = SentenceTransformer("all-MiniLM-L6-v2", device=_DEVICE)
    _BERT_AVAILABLE = True
    print(f"[BERT] Using device: {_DEVICE}", file=sys.stderr)
except ImportError:
    _BERT_AVAILABLE = False
    print("WARNING: sentence-transformers not available; running WUP-only mode.", file=sys.stderr)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT  = Path(__file__).resolve().parent.parent
_CANONICAL  = _REPO_ROOT / "config" / "canonical_relations.json"
_INPUT_DIRS = [
    _REPO_ROOT / "enriched_ontology_matching" / "inputs",
    _REPO_ROOT / "enriched_ontology_matching" / "models",
]
_OUT_MAP        = _REPO_ROOT / "config" / "relation_map.json"
_OUT_CSV        = _REPO_ROOT / "enriched_ontology_matching" / "association_inventory.csv"
_BERT_CACHE     = _REPO_ROOT / "config" / ".bert_embedding_cache.pkl"
_BERT_MODEL_KEY = "all-MiniLM-L6-v2"  # bump this string to invalidate cache on model change

# ---------------------------------------------------------------------------
# PascalCase splitter
# ---------------------------------------------------------------------------
_CAMEL_RE = re.compile(
    r"(?<=[a-z])(?=[A-Z])"
    r"|(?<=[A-Z])(?=[A-Z][a-z])"
    r"|(?<=\D)(?=\d)"
    r"|(?<=\d)(?=\D)"
)


def split_pascal(name: str) -> list[str]:
    """Split PascalCase/camelCase into lowercase tokens, drop single chars."""
    return [t.lower() for t in _CAMEL_RE.split(name) if len(t) > 1]


def entity_tokens_in_name(assoc_name: str, entity_names: list[str]) -> set[str]:
    """
    Find which lowercase tokens of *assoc_name* belong to participant entities.

    Uses two complementary strategies so both exact and abbreviated matches
    are handled:

    1. STRING-LEVEL suffix/prefix matching against the raw association name.
       Tries the full entity name, then progressively shorter right-anchored
       PascalCase segments (suffixes), then left-anchored segments (prefixes).
       This handles multi-word entities like "AutomatedMedicationDispenser"
       finding "Dispenser" in "DispenserInMedRoom".

    2. TOKEN-LEVEL fallback: any split token of the entity name that also
       appears as a split token of the association name is added to the
       exclusion set.  This catches abbreviated cases like "AeropressKit" →
       token "aeropress" → found in "AeropressDispensesToCup".

    The union of both strategies is returned, so no single approach can miss
    a match that the other would catch.
    """
    entity_token_set: set[str] = set()
    remaining = assoc_name

    for entity in sorted(entity_names, key=len, reverse=True):
        raw_segs = _CAMEL_RE.split(entity)
        matched = False

        # Strategy 1a: progressively shorter suffixes (right-anchored)
        for start in range(len(raw_segs)):
            candidate = "".join(raw_segs[start:])
            if len(candidate) < 2:
                break
            if candidate in remaining:
                for tok in split_pascal(candidate):
                    entity_token_set.add(tok)
                remaining = remaining.replace(candidate, "", 1)
                matched = True
                break

        # Strategy 1b: progressively shorter prefixes (left-anchored)
        if not matched:
            for end in range(len(raw_segs), 0, -1):
                candidate = "".join(raw_segs[:end])
                if len(candidate) < 2:
                    break
                if candidate in remaining:
                    for tok in split_pascal(candidate):
                        entity_token_set.add(tok)
                    remaining = remaining.replace(candidate, "", 1)
                    matched = True
                    break

        # Strategy 2: token-level fallback — any shared token qualifies
        if not matched:
            assoc_toks = set(split_pascal(assoc_name))
            for tok in split_pascal(entity):
                if tok in assoc_toks:
                    entity_token_set.add(tok)

    return entity_token_set


# ---------------------------------------------------------------------------
# Content-word extraction from definition text
# ---------------------------------------------------------------------------

def content_words_from_text(text: str) -> list[str]:
    """
    Extract nouns and verbs from a definition string via NLTK POS tagging.
    Returns lowercase lemmas.  Stopwords like 'is', 'be', 'have' are filtered
    so they do not dominate WUP scoring.
    """
    _stopverbs = {"be", "is", "are", "has", "have", "do", "does", "use", "used",
                  "get", "make", "can", "may", "must", "will", "would", "could"}
    tokens = nltk.word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens)
    words: list[str] = []
    for word, tag in tagged:
        if not word.isalpha() or word in _stopverbs:
            continue
        if tag.startswith("VB"):
            words.append(word)
        elif tag.startswith("NN"):
            words.append(word)
    return list(dict.fromkeys(words))  # deduplicated, order preserved


# ---------------------------------------------------------------------------
# WUP scoring
# ---------------------------------------------------------------------------

_lemmatizer = _WNL()


@lru_cache(maxsize=None)
def _synsets_for(word: str) -> tuple:
    """
    Return up to 5 WordNet synsets for a word, trying the verb sense first.

    Inflected forms like 'charges', 'fires', 'bridges' must be lemmatized
    before lookup — WordNet only indexes base forms ('charge', 'fire',
    'bridge').  Without lemmatization, verb-form queries return nothing and
    the code falls through to generic noun synsets, which then match IsA
    content words ("instance", "kind") through shallow hypernym paths and
    produce wrong WUP-dominated mappings.
    """
    verb_lemma = _lemmatizer.lemmatize(word, pos="v")
    noun_lemma = _lemmatizer.lemmatize(word, pos="n")
    syns = (
        wn.synsets(verb_lemma, pos=wn.VERB)
        or wn.synsets(noun_lemma, pos=wn.NOUN)
        or wn.synsets(word)
    )
    return tuple(syns[:5])


@lru_cache(maxsize=None)
def best_wup_pair(word_a: str, word_b: str) -> float:
    """Max WUP similarity across the top-5 synset pairs of two words."""
    syns_a = list(_synsets_for(word_a))
    syns_b = list(_synsets_for(word_b))
    best = 0.0
    for sa in syns_a:
        for sb in syns_b:
            try:
                s = sa.wup_similarity(sb)
                if s and s > best:
                    best = s
            except Exception:
                pass
    return best


def wup_score(assoc_tokens: list[str], content_words: list[str]) -> float:
    """Max WUP between any assoc token and any content word from the definition."""
    if not assoc_tokens or not content_words:
        return 0.0
    return max(
        best_wup_pair(tok, cw)
        for tok in assoc_tokens
        for cw in content_words
    )


# ---------------------------------------------------------------------------
# BERT cosine scoring
# ---------------------------------------------------------------------------

def cosine(a, b) -> float:
    norm = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Disk embedding cache  (keyed by sha256(model_name + ":" + text))
# ---------------------------------------------------------------------------

def _cache_key(text: str) -> str:
    return hashlib.sha256(f"{_BERT_MODEL_KEY}:{text}".encode()).hexdigest()[:20]


def load_bert_cache() -> dict[str, object]:
    if _BERT_CACHE.exists():
        try:
            with _BERT_CACHE.open("rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def save_bert_cache(cache: dict[str, object]) -> None:
    _BERT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with _BERT_CACHE.open("wb") as f:
        pickle.dump(cache, f)


def cached_batch_encode(texts: list[str], cache: dict) -> dict[str, object]:
    """
    Encode only texts not already in the disk cache, update cache in-place,
    and return a phrase → embedding dict for all requested texts.
    """
    if not _BERT_AVAILABLE:
        return {}
    missing = [t for t in texts if _cache_key(t) not in cache]
    if missing:
        new_embs = _BERT_MODEL.encode(missing, batch_size=256, show_progress_bar=False)
        for text, emb in zip(missing, new_embs):
            cache[_cache_key(text)] = emb
    return {t: cache[_cache_key(t)] for t in texts}


def precompute_bert_embeddings(canonicals: list[dict], cache: dict) -> dict[str, object]:
    """
    Build a BERT embedding for each canonical relation.

    If the canonical has exemplar_phrases (seeded from ConceptNet by
    seed_exemplars.py), encode each phrase and take their mean — this keeps
    the comparison target in the same short-phrase embedding space as the
    association names being classified.

    Falls back to the full definition text when no exemplar phrases exist.
    """
    if not _BERT_AVAILABLE:
        return {}

    results: dict[str, object] = {}
    for c in canonicals:
        exemplars = [p for p in c.get("exemplar_phrases", []) if p]
        if exemplars:
            phrase_map = cached_batch_encode(exemplars, cache)
            embs = [phrase_map[p] for p in exemplars if p in phrase_map]
            if embs:
                results[c["id"]] = np.mean(embs, axis=0)
                continue
        # Fallback: encode the definition sentence
        phrase_map = cached_batch_encode([c["definition"]], cache)
        emb = phrase_map.get(c["definition"])
        if emb is not None:
            results[c["id"]] = emb

    return results


def batch_encode_associations(phrases: list[str], cache: dict) -> dict[str, object]:
    """Encode all association phrases in one GPU batch, using disk cache."""
    if not _BERT_AVAILABLE:
        return {}
    return cached_batch_encode(phrases, cache)


def bert_score(assoc_phrase: str, canonical_id: str,
               canonical_embeddings: dict, assoc_embeddings: dict) -> float:
    if not _BERT_AVAILABLE:
        return 0.0
    assoc_emb = assoc_embeddings.get(assoc_phrase)
    canon_emb = canonical_embeddings.get(canonical_id)
    if assoc_emb is None or canon_emb is None:
        return 0.0
    return cosine(assoc_emb, canon_emb)


# ---------------------------------------------------------------------------
# POS classification helper
# ---------------------------------------------------------------------------

_PREP_TAGS   = {"IN", "RP", "TO"}      # prepositions and particles
_ADVERB_TAGS = {"RB", "RBR", "RBS"}   # adverbs

# Spatial prepositions that unambiguously signal a physical-location relation.
# Grounded in the AtLocation definition: "X is physically installed, placed,
# mounted, situated, housed, or located in or at Y."  The prepositions "at"
# and "in" are explicit; "on", "within", "inside", "atop", etc. are implied
# by the same physical-placement meaning.
# Excluded: "to" (directional → UsedFor/Causes), "from", "through", "across",
#           "of" (PartOf), "with" (HasA/UsedWith).
_SPATIAL_PREPS = {
    "on", "upon", "atop",
    "in", "within", "inside",
    "at",
    "beside", "next", "near", "above", "below", "under", "over",
    "alongside", "adjacent",
}


def pos_classify(
    relation_tokens: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    POS-tag *relation_tokens* (entity-stripped) and partition them into
    (verb_tokens, prep_tokens, adverb_tokens).

    Tag map used:
      VB*           → verb  (includes VBD, VBG, VBN, VBP, VBZ)
      NNS + lemma   → verb  (catches 3rd-person forms mis-tagged as NNS,
                             e.g. "charges"→"charge", "fires"→"fire")
      IN / RP / TO  → preposition or particle
      RB / RBR / RBS→ adverb

    Tokens not matching any category (adjectives JJ, other nouns NN…) are
    omitted from all three lists; callers fall back to the full relation_tokens
    list for WUP when all three lists are empty.
    """
    if not relation_tokens:
        return [], [], []
    tagged        = nltk.pos_tag(relation_tokens)
    verb_tokens:  list[str] = []
    prep_tokens:  list[str] = []
    adverb_tokens: list[str] = []
    for word, tag in tagged:
        if tag.startswith("VB"):
            verb_tokens.append(word)
        elif tag == "NNS":
            vl = _lemmatizer.lemmatize(word, pos="v")
            if vl != word and wn.synsets(vl, pos=wn.VERB):
                verb_tokens.append(word)
        elif tag in _PREP_TAGS:
            prep_tokens.append(word)
        elif tag in _ADVERB_TAGS:
            adverb_tokens.append(word)
    return verb_tokens, prep_tokens, adverb_tokens


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(
    name: str,
    canonicals: list[dict],
    cw_index: dict[str, list[str]],
    canonical_embeddings: dict,
    assoc_embeddings: dict,
    wup_weight: float,
    low_threshold: float,
    participants: list[str] | None = None,
    relation_phrase: str | None = None,
    bert_phrase: str | None = None,
) -> dict:
    """
    Map an association name to the best canonical relation.

    Steps:
      1. Split PascalCase into tokens.
      2. Strip tokens that match participant entity names — these are entity
         identifiers embedded in the association name (e.g. "Airlock" and
         "Fermenter" in "AirlockOnFermenter") and carry no relation signal.
         What remains are the pure relation tokens ("on", "charges", "feeds").
      3. POS-classify relation tokens via pos_classify() into verbs,
         prepositions (IN/RP/TO), and adverbs (RB*).  Extend NLTK's verb
         detection to catch 3rd-person forms mis-tagged as NNS.
         WUP uses verb tokens if found, else all relation tokens.
      4. Spatial preposition shortcut: if all relation tokens are spatial
         prepositions from _SPATIAL_PREPS (on, in, at, within, …) they
         unambiguously signal a physical-location relation and are mapped
         directly to AtLocation (score=1.0, method="spatial_prep").
      5. For non-spatial function-word cases (to, through, …): drop WUP
         weight to 0 so BERT carries the full score — WUP is always 0 for
         prepositions/adverbs since they have no WordNet verb/noun synsets.
         Detection uses POS tags, not WordNet (preps like "on" have spurious
         adjective synsets, e.g. on.s.01, that would fool a synset-based test).
      6. For each canonical: WUP against definition content words; BERT cosine
         between bert_phrase and the canonical exemplar-mean embedding.
         bert_phrase = relation_phrase + object entity for prep-only cases,
         giving "on fermenter" instead of bare "on" for better discrimination.
      7. Combined = eff_wup_weight * wup + eff_bert_weight * bert.
      8. Pick the highest; fall back to RelatedTo if below low_threshold.
    """
    tokens       = split_pascal(name) or [name.lower()]
    assoc_phrase = " ".join(tokens)

    # ── Strip entity tokens from participant names ────────────────────────────
    # Match PascalCase suffixes of participant names against the raw association
    # string (not split tokens) so that multi-word entity names like
    # "AutomatedMedicationDispenser" correctly identify "Dispenser" in the name.
    participant_token_set: set[str] = (
        entity_tokens_in_name(name, list(participants)) if participants else set()
    )
    relation_tokens = [t for t in tokens if t not in participant_token_set] or tokens
    if relation_phrase is None:
        relation_phrase = " ".join(relation_tokens)

    # ── POS-classify relation tokens ──────────────────────────────────────────
    verb_tokens, prep_tokens, adverb_tokens = pos_classify(relation_tokens)

    # WUP: prefer verbs; fall back to all relation tokens for prep-only cases.
    active = verb_tokens if verb_tokens else relation_tokens

    # ── Post-preposition refinement ───────────────────────────────────────────
    # Abbreviated entity tokens that survive suffix-matching (e.g. "Med" from
    # "MedRoom" vs entity "MedicationRoom") land after the preposition and are
    # sometimes mistaggged as verbs by NLTK out of context ("med" → VBD).
    # Strip everything after the last preposition when:
    #   • a preposition exists, AND
    #   • participants were provided (so we know entity stripping was attempted), AND
    #   • no verb token appears BEFORE the preposition
    #     (a verb before the preposition is a genuine relation word, not a remnant)
    if prep_tokens and participant_token_set:
        last_prep_idx = max(
            i for i, t in enumerate(relation_tokens) if t in set(prep_tokens)
        )
        verb_before_prep = any(
            i <= last_prep_idx
            for i, t in enumerate(relation_tokens)
            if t in set(verb_tokens)
        )
        if not verb_before_prep:
            refined = relation_tokens[:last_prep_idx + 1]
            if refined != relation_tokens:
                relation_tokens = refined
                relation_phrase  = " ".join(relation_tokens)
                verb_tokens, prep_tokens, adverb_tokens = pos_classify(relation_tokens)
                active = relation_tokens

    # ── Spatial preposition shortcut ──────────────────────────────────────────
    # "on", "in", "at", "within", "inside", "atop", etc. are unambiguous
    # spatial prepositions that directly signal a physical-location relation.
    # Bypassing WUP/BERT avoids false disambiguation against HasPrerequisite
    # ("depends on", "based on" exemplars score higher than AtLocation for the
    # bare word "on" in BERT space).
    function_words_only = (
        not verb_tokens
        and bool(prep_tokens or adverb_tokens)
        and set(relation_tokens) <= set(prep_tokens) | set(adverb_tokens)
    )
    if function_words_only and set(prep_tokens) <= _SPATIAL_PREPS and not adverb_tokens:
        return {
            "canonical":     "AtLocation",
            "score":         1.0,
            "wup":           0.0,
            "bert":          1.0,
            "method":        "spatial_prep",
            "tokens":        tokens,
            "verb_tokens":   verb_tokens,
            "prep_tokens":   prep_tokens,
            "adverb_tokens": adverb_tokens,
            "all_scores":    {},
        }

    # ── Weight adjustment for remaining function-word cases ───────────────────
    # Non-spatial prepositions ("to", "from", "through") and adverbs still have
    # no WordNet synsets — WUP is structurally 0, so drop its weight.
    eff_wup_weight  = 0.0 if function_words_only else wup_weight
    eff_bert_weight = 1.0 - eff_wup_weight

    # BERT input: prefer the pre-computed bert_phrase (passed from main).
    # For prep-only cases main() sets this to "relation_phrase + object_entity"
    # (e.g. "on fermenter") to ground bare prepositions with domain context.
    effective_bert = bert_phrase if (bert_phrase and bert_phrase in assoc_embeddings) else (
        relation_phrase if relation_phrase in assoc_embeddings else assoc_phrase
    )

    scores: dict[str, dict] = {}
    for c in canonicals:
        cid     = c["id"]
        wup     = wup_score(active, cw_index[cid])
        bert    = bert_score(effective_bert, cid, canonical_embeddings, assoc_embeddings)
        combined = eff_wup_weight * wup + eff_bert_weight * bert
        scores[cid] = {"wup": round(wup, 4), "bert": round(bert, 4), "combined": round(combined, 4)}

    best_id = max(scores, key=lambda k: scores[k]["combined"])
    best    = scores[best_id]

    if best["combined"] < low_threshold:
        return {
            "canonical":     "RelatedTo",
            "score":         best["combined"],
            "wup":           best["wup"],
            "bert":          best["bert"],
            "method":        "fallback",
            "tokens":        tokens,
            "verb_tokens":   verb_tokens,
            "prep_tokens":   prep_tokens,
            "adverb_tokens": adverb_tokens,
            "all_scores":    scores,
        }

    return {
        "canonical":     best_id,
        "score":         best["combined"],
        "wup":           best["wup"],
        "bert":          best["bert"],
        "method":        "wup+bert" if _BERT_AVAILABLE else "wup",
        "tokens":        tokens,
        "verb_tokens":   verb_tokens,
        "prep_tokens":   prep_tokens,
        "adverb_tokens": adverb_tokens,
        "all_scores":    scores,
    }


# ---------------------------------------------------------------------------
# JSON scanning
# ---------------------------------------------------------------------------

def iter_json_files(dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        if d.exists():
            files.extend(d.rglob("*.json"))
    return sorted(set(files))


def extract_associations(json_files: list[Path]) -> dict[str, dict]:
    registry: dict[str, dict] = defaultdict(lambda: {
        "participants": set(),
        "sources":      [],
        "count":        0,
    })
    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            continue
        candidates = [data]
        if "json_a" in data:
            candidates.append(data["json_a"])
        if "json_b" in data:
            candidates.append(data["json_b"])
        for model in candidates:
            if not isinstance(model, dict):
                continue
            for assoc in model.get("associations", []):
                name = assoc.get("associationName") or assoc.get("name", "")
                if not name:
                    continue
                parts = tuple(sorted(
                    assoc.get("associationParticipants") or assoc.get("participants") or []
                ))
                entry = registry[name]
                entry["participants"].add(parts)
                entry["sources"].append(str(path.relative_to(_REPO_ROOT)))
                entry["count"] += 1
    return dict(registry)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dirs: list[Path], threshold: float, wup_weight: float) -> None:
    if not _CANONICAL.exists():
        print(f"ERROR: canonical relations file not found: {_CANONICAL}", file=sys.stderr)
        sys.exit(1)
    canonicals: list[dict] = json.loads(_CANONICAL.read_text(encoding="utf-8"))

    # Pre-compute content words and BERT embeddings from definitions
    print("Indexing canonical relation definitions ...")
    cw_index = {c["id"]: content_words_from_text(c["definition"]) for c in canonicals}
    print("  Content words per relation:")
    for cid, cws in cw_index.items():
        print(f"    {cid:<20} {cws}")

    # Load disk embedding cache
    emb_cache = load_bert_cache()
    cache_size_before = len(emb_cache)

    print("\nEncoding canonical definitions with BERT ...")
    bert_embeddings = precompute_bert_embeddings(canonicals, emb_cache)
    if _BERT_AVAILABLE:
        print(f"  Encoded {len(bert_embeddings)} relations with all-MiniLM-L6-v2.")
    else:
        print("  BERT unavailable — using WUP only (weight=1.0).")
        wup_weight = 1.0

    json_files = iter_json_files(input_dirs)
    print(f"\nScanning {len(json_files)} JSON files ...")
    registry = extract_associations(json_files)
    print(f"Found {len(registry)} unique association names.\n")

    all_names = sorted(registry)

    # Pre-compute entity-stripped relation phrases and BERT input phrases.
    # relation_phrase: pure relation tokens, e.g. "on" from "AirlockOnFermenter"
    # bert_phrase: for preposition-only cases, appends the object entity so BERT
    #   gets "on fermenter" instead of bare "on" — the domain entity grounds the
    #   spatial sense and helps separate AtLocation from HasPrerequisite.
    relation_phrases: dict[str, str] = {}
    bert_phrases:     dict[str, str] = {}
    for name in all_names:
        entry = registry[name]
        all_p = sorted({e for parts in entry["participants"] for e in parts})
        toks  = split_pascal(name) or [name.lower()]
        # Use string-level suffix matching so multi-word entity names like
        # "AutomatedMedicationDispenser" find "Dispenser" in the association name.
        excl     = entity_tokens_in_name(name, all_p)
        rel_toks = [t for t in toks if t not in excl] or toks
        rel_phrase = " ".join(rel_toks)

        # Detect preposition-only (spatial shortcut will handle it, but compute
        # bert_phrase anyway for non-spatial prep cases like "to", "from").
        v_t, p_t, a_t = pos_classify(rel_toks)

        # Mirror the post-preposition refinement in classify(): strip tokens
        # after the last preposition when no verb precedes it (those tokens
        # are entity abbreviations, possibly mistaggged as verbs by NLTK).
        if p_t and excl:
            last_pi = max(i for i, t in enumerate(rel_toks) if t in set(p_t))
            verb_before = any(i <= last_pi for i, t in enumerate(rel_toks) if t in set(v_t))
            if not verb_before:
                refined = rel_toks[:last_pi + 1]
                if refined != rel_toks:
                    rel_toks   = refined
                    rel_phrase = " ".join(rel_toks)
                    v_t, p_t, a_t = pos_classify(rel_toks)
        relation_phrases[name] = rel_phrase

        is_func = not v_t and bool(p_t or a_t) and set(rel_toks) <= set(p_t) | set(a_t)

        if is_func:
            # Append object entity (tokens after the last relation token) for context.
            rel_set  = set(rel_toks)
            last_rel = max((i for i, t in enumerate(toks) if t in rel_set), default=-1)
            after    = [toks[i] for i in range(last_rel + 1, len(toks)) if toks[i] in excl]
            bert_phrases[name] = (rel_phrase + " " + " ".join(after)).strip() if after else rel_phrase
        else:
            bert_phrases[name] = rel_phrase  # use stripped phrase for verbs too

    # Batch-encode full phrases, relation phrases, and bert phrases in one pass
    all_phrases   = [" ".join(split_pascal(n) or [n.lower()]) for n in all_names]
    all_to_encode = sorted(
        set(all_phrases) | set(relation_phrases.values()) | set(bert_phrases.values())
    )
    cache_hits = sum(1 for p in all_to_encode if _cache_key(p) in emb_cache)
    print(f"Encoding phrases with BERT "
          f"({cache_hits}/{len(all_to_encode)} cache hits) ...")
    assoc_embeddings = batch_encode_associations(all_to_encode, emb_cache)
    print(f"  Done. {len(emb_cache) - cache_size_before} new embeddings added to cache.")

    relation_map: dict[str, dict] = {}
    rows: list[dict] = []

    for name in all_names:
        entry  = registry[name]
        all_participants = sorted({entity for parts in entry["participants"] for entity in parts})
        result = classify(
            name, canonicals, cw_index, bert_embeddings, assoc_embeddings,
            wup_weight, threshold * 0.6,
            participants=all_participants,
            relation_phrase=relation_phrases[name],
            bert_phrase=bert_phrases[name],
        )
        participants_flat = "; ".join(" + ".join(p) for p in sorted(entry["participants"]))

        relation_map[name] = {
            "canonical":        result["canonical"],
            "score":            round(result["score"], 4),
            "wup":              round(result["wup"], 4),
            "bert":             round(result["bert"], 4),
            "method":           result["method"],
            "verb_tokens":      result["verb_tokens"],
            "prep_tokens":      result["prep_tokens"],
            "adverb_tokens":    result["adverb_tokens"],
            "all_tokens":       result["tokens"],
            "occurrence_count": entry["count"],
        }
        rows.append({
            "association_name": name,
            "canonical":        result["canonical"],
            "score":            round(result["score"], 4),
            "wup":              round(result["wup"], 4),
            "bert":             round(result["bert"], 4),
            "method":           result["method"],
            "verb_tokens":      " | ".join(result["verb_tokens"]),
            "prep_tokens":      " | ".join(result["prep_tokens"]),
            "adverb_tokens":    " | ".join(result["adverb_tokens"]),
            "all_tokens":       " | ".join(result["tokens"]),
            "participants":     participants_flat,
            "occurrence_count": entry["count"],
            "low_confidence":   "YES" if result["score"] < threshold else "",
        })

    # Persist updated embedding cache and WUP stats
    if _BERT_AVAILABLE:
        save_bert_cache(emb_cache)
        print(f"Cache saved: {len(emb_cache)} entries -> {_BERT_CACHE}")
    info = best_wup_pair.cache_info()
    print(f"WUP lru_cache: {info.hits} hits / {info.hits + info.misses} calls "
          f"({100*info.hits/(info.hits+info.misses+1e-9):.0f}% hit rate)")

    # Write outputs
    _OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    _OUT_MAP.write_text(json.dumps(relation_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {_OUT_MAP}")

    _OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "association_name", "canonical", "score", "wup", "bert", "method",
        "verb_tokens", "prep_tokens", "adverb_tokens",
        "all_tokens", "participants", "occurrence_count", "low_confidence",
    ]
    with _OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {_OUT_CSV}")

    # Console summary
    from collections import Counter
    dist = Counter(r["canonical"] for r in rows)
    low  = [r for r in rows if r["low_confidence"] == "YES"]

    print(f"\n{'='*72}")
    print(f"Canonical relation distribution ({len(rows)} associations)")
    print(f"  Scoring: WUP weight={wup_weight:.2f}  BERT weight={1-wup_weight:.2f}")
    print(f"{'='*72}")
    for rel, cnt in dist.most_common():
        print(f"  {rel:<20} {cnt:>4}  {'#'*cnt}")

    if low:
        print(f"\nLow-confidence mappings (combined score < {threshold}):")
        for r in sorted(low, key=lambda x: x["score"]):
            print(f"  {r['association_name']:<42} combined={r['score']:.3f}  wup={r['wup']:.3f}  bert={r['bert']:.3f}")
    else:
        print(f"\nAll {len(rows)} associations mapped with combined score >= {threshold}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize association names to canonical relations via WUP + BERT."
    )
    parser.add_argument("--input-dir", nargs="+", default=None,
                        help="Directories to scan (default: inputs/ and models/).")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Combined score below which a mapping is flagged (default: 0.5).")
    parser.add_argument("--wup-weight", type=float, default=0.5,
                        help="Weight for WUP score in combined metric (default: 0.5).")
    args = parser.parse_args()

    dirs = [Path(d) for d in args.input_dir] if args.input_dir else _INPUT_DIRS
    main(dirs, args.threshold, args.wup_weight)
