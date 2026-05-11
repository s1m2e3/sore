"""
root_comparator.py
------------------
Receives two ontology JSON files, compares their root entities using a
combined WordNet + ConceptNet pipeline, and writes a merged output JSON
containing both inputs plus all intermediate steps needed to relate the roots.

Algorithm
---------
1.  Extract root entity name from each JSON (first entity in 'entities' list).
2.  Split camelCase/PascalCase root names into tokens.
3.  For each token pair (cross-product of tokens_a x tokens_b):
      a. Direct wup_similarity across all synset pairs  → IsA / synonym
      b. Gloss mining (does token_b appear in token_a's synset definition?)
      c. BFS over WordNet (hypernym + meronym/holonym + derivational) combined
         with live ConceptNet edges (all relation types, no pre-specification).
         Agent nouns (applicant, student, …) are normalised to action nouns
         (application, study, …) before BFS.
4.  Select the token pair with the shortest path.
5.  Write output JSON: json_a + json_b + alignment (all steps, methods, CN relations).

Usage
-----
    # From repo root, using the ontology_matching venv:
    .\\ontology_matching\\.venv\\Scripts\\python.exe ^
        enriched_ontology_matching\\root_comparator.py ^
        <json_a_path> <json_b_path> [--output <out.json>]

    # Or from within enriched_ontology_matching/:
    ..\\ontology_matching\\.venv\\Scripts\\python.exe ^
        root_comparator.py <json_a_path> <json_b_path>
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional deps — degrade gracefully if absent
# ---------------------------------------------------------------------------
try:
    from nltk.corpus import wordnet as wn
    wn.synsets("test")          # force load
    _WN_OK = True
except Exception:
    _WN_OK = False
    print("[WARN] WordNet unavailable — WordNet steps will be skipped.")

try:
    import requests
    _REQ_OK = True
except ImportError:
    _REQ_OK = False
    print("[WARN] 'requests' not installed — ConceptNet queries will be skipped.")

# Circuit breaker: after this many consecutive API failures, stop calling CN API
_CN_FAIL_THRESHOLD = 3
_cn_consecutive_failures = 0
_CN_DISABLED = False   # flipped True when circuit opens (API down)

# CSV fallback state
_CN_CSV_INDEX: dict[str, list[tuple[str, str, float]]] = {}  # word → [(rel, other, weight)]
_CN_CSV_LOADED = False   # True once the CSV has been indexed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CN_API        = "https://api.conceptnet.io/query"
CN_TIMEOUT    = 3       # seconds per request
CN_RETRIES    = 2
CN_MIN_WEIGHT = 1.0    # filter noisy crowd-sourced edges

# Default path to the local ConceptNet CSV (override via --cn-csv)
_CN_CSV_DEFAULT = str(
    Path(__file__).resolve().parent.parent
    / "ontology_matching" / "inputs"
    / "conceptnet-assertions-5.7.0.csv" / "assertions.csv"
)

AGENT_SUFFIXES = ("ant", "ent", "er", "or", "ee", "ist", "ar")

# In-process cache: (word_a, word_b) -> list of CN edge dicts
_CN_CACHE: dict[tuple[str, str], list[dict]] = {}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def split_camel(name: str) -> list[str]:
    """Split PascalCase / camelCase into lowercase tokens (min 3 chars)."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    tokens = re.split(r"[^A-Za-z0-9]+", s)
    return [t.lower() for t in tokens if len(t) >= 3]


def is_agent_noun(word: str) -> bool:
    return word.lower().endswith(AGENT_SUFFIXES)


# ---------------------------------------------------------------------------
# WordNet helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2048)
def wn_synsets(word: str) -> tuple:
    if not _WN_OK:
        return ()
    syns = wn.synsets(word, pos=wn.NOUN)
    return tuple(syns)


def get_action_nouns(word: str) -> set[str]:
    """
    Agent noun → action/result nouns, staying within the noun domain.

    Chain: agent_noun.n → DRF verb synset (specific, not all senses of the
    verb word) → DRF noun from that same verb synset.

    Using only the specific verb synset reached by DRF (e.g. apply.v.06 for
    'applicant') avoids the explosion caused by looking up ALL senses of the
    verb word (e.g. apply.v.01 = 'be relevant', apply.v.02 = 'use', …).
    This keeps the normalised nouns semantically tied to the agent role:
      applicant  → application          (not utility, usage, use, …)
      student    → education, study
      supplier   → supply, provision
      teacher    → instruction, teaching
    """
    if not _WN_OK:
        return set()
    action_nouns: set[str] = set()
    original = word.lower().replace("_", " ")
    for syn in wn_synsets(word):
        for lemma in syn.lemmas():
            for drf in lemma.derivationally_related_forms():
                verb_syn = drf.synset()
                if verb_syn.pos() != "v":
                    continue
                # Stay within this specific verb synset — do NOT look up all
                # synsets of the verb word.
                for vlemma in verb_syn.lemmas():
                    for ndrf in vlemma.derivationally_related_forms():
                        if ndrf.synset().pos() == "n":
                            n = ndrf.name().replace("_", " ")
                            if n != original and not is_agent_noun(n):
                                action_nouns.add(n)
    return action_nouns


def wn_edges_from_synset(syn) -> list[tuple[str, str, str]]:
    """
    Returns list of (wn_method, relation_label, neighbor_synset_name).
    wn_method values: hypernym, meronym_part, meronym_member,
                      holonym_part, holonym_member, derivational
    """
    edges = []
    for s in syn.hypernyms():
        edges.append(("hypernym", "wn:IsA", s.name()))
    for s in syn.part_meronyms():
        edges.append(("meronym_part", "wn:PartOf", s.name()))
    for s in syn.member_meronyms():
        edges.append(("meronym_member", "wn:MemberOf", s.name()))
    for s in syn.part_holonyms():
        edges.append(("holonym_part", "wn:IsPartOf", s.name()))
    for s in syn.member_holonyms():
        edges.append(("holonym_member", "wn:IsMemberOf", s.name()))
    for lemma in syn.lemmas():
        for drf in lemma.derivationally_related_forms():
            edges.append(("derivational", "wn:DerivedFrom", drf.synset().name()))
    return edges


def gloss_check(word_a: str, word_b: str) -> Optional[dict]:
    """
    Returns a step dict if word_b (or a lemma of its synsets) appears in
    any synset definition of word_a.  Otherwise None.
    """
    if not _WN_OK:
        return None
    target_terms = {word_b.lower()}
    for syn in wn_synsets(word_b):
        for lemma in syn.lemma_names():
            target_terms.add(lemma.lower().replace("_", " "))

    for syn in wn_synsets(word_a):
        gloss = syn.definition().lower()
        for term in target_terms:
            if term in gloss:
                return {
                    "method": "gloss_mining",
                    "synset": syn.name(),
                    "definition": syn.definition(),
                    "matched_term": term,
                }
    return None


def best_wup(word_a: str, word_b: str) -> tuple[float, str, str]:
    """Best Wu-Palmer similarity across all synset pairs. Returns (score, syn_a, syn_b)."""
    if not _WN_OK:
        return 0.0, "", ""
    best = (0.0, "", "")
    for sa in wn_synsets(word_a):
        for sb in wn_synsets(word_b):
            score = sa.wup_similarity(sb) or 0.0
            if score > best[0]:
                best = (score, sa.name(), sb.name())
    return best


# ---------------------------------------------------------------------------
# ConceptNet helpers
# ---------------------------------------------------------------------------

def _cn_request(url: str, params: dict) -> list[dict]:
    """HTTP GET with retry + circuit breaker. Returns list of edge dicts or []."""
    global _cn_consecutive_failures, _CN_DISABLED
    if not _REQ_OK or _CN_DISABLED:
        return []
    for attempt in range(CN_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=CN_TIMEOUT)
            if r.status_code == 200:
                _cn_consecutive_failures = 0   # reset on success
                return r.json().get("edges", [])
            if r.status_code in (502, 503, 504):
                _cn_consecutive_failures += 1
                if _cn_consecutive_failures >= _CN_FAIL_THRESHOLD:
                    _CN_DISABLED = True
                    print(f"[INFO] ConceptNet API unavailable after "
                          f"{_CN_FAIL_THRESHOLD} failures — disabling for this run.")
                    return []
                time.sleep(0.5)
                continue
            return []
        except Exception:
            _cn_consecutive_failures += 1
            if _cn_consecutive_failures >= _CN_FAIL_THRESHOLD:
                _CN_DISABLED = True
                print(f"[INFO] ConceptNet API unavailable — disabling for this run.")
                return []
            time.sleep(0.5)
    return []


# Pre-compiled patterns for fast CSV field extraction (avoids json.loads per line)
_CN_WEIGHT_RE = re.compile(r'"weight"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
_CN_EN_PREFIX = "/c/en/"


def _cn_load_csv(csv_path: str) -> None:
    """
    Build an in-memory index from the ConceptNet assertions CSV.

    Expected format (tab-separated, 5 columns):
      assertion_uri  relation_uri  start_node  end_node  metadata_json

    Only English-to-English edges (/c/en/…) with weight >= CN_MIN_WEIGHT
    are indexed.  Both directions are stored so lookups are symmetric.

    Index structure:
      _CN_CSV_INDEX[word] = [(relation_label, other_word, weight), ...]
    """
    global _CN_CSV_INDEX, _CN_CSV_LOADED
    if _CN_CSV_LOADED:
        return

    if not Path(csv_path).exists():
        print(f"[WARN] ConceptNet CSV not found at: {csv_path}")
        _CN_CSV_LOADED = True   # mark as attempted so we don't retry
        return

    print(f"[INFO] Loading ConceptNet CSV index from {csv_path} …", flush=True)
    count = 0
    index: dict[str, list[tuple[str, str, float]]] = {}

    # Inline helpers — defined once outside loop for speed
    en_prefix = _CN_EN_PREFIX
    en_prefix_len = len(en_prefix)
    weight_re = _CN_WEIGHT_RE
    min_weight = CN_MIN_WEIGHT

    def _word_from_node(node_id: str) -> str:
        """Extract plain word from /c/en/word or /c/en/word/pos/… URI."""
        if not node_id.startswith(en_prefix):
            return ""
        # Slice past "/c/en/" and take the first path segment
        rest = node_id[en_prefix_len:]
        slash = rest.find("/")
        word = rest if slash == -1 else rest[:slash]
        return word.replace("_", " ").lower() if word else ""

    def _rel_label(rel_uri: str) -> str:
        """Extract label from /r/RelatedTo → RelatedTo."""
        slash = rel_uri.rfind("/")
        return rel_uri[slash + 1:] if slash >= 0 else rel_uri

    try:
        with open(csv_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                # Fast tab-split with early English filter on raw bytes
                tab1 = line.find("\t")
                if tab1 < 0:
                    continue
                tab2 = line.find("\t", tab1 + 1)
                if tab2 < 0:
                    continue
                tab3 = line.find("\t", tab2 + 1)
                if tab3 < 0:
                    continue
                tab4 = line.find("\t", tab3 + 1)

                start_node = line[tab2 + 1:tab3]
                end_node   = line[tab3 + 1:tab4] if tab4 >= 0 else line[tab3 + 1:].rstrip("\n")

                # English-only filter (cheap prefix check before any parsing)
                if not (start_node.startswith(en_prefix) and end_node.startswith(en_prefix)):
                    continue

                rel_uri = line[tab1 + 1:tab2]
                meta    = line[tab4 + 1:].rstrip("\n") if tab4 >= 0 else ""

                # Weight filter — regex is ~4x faster than json.loads for this field
                m = weight_re.search(meta)
                weight = float(m.group(1)) if m else 0.0
                if weight < min_weight:
                    continue

                sw = _word_from_node(start_node)
                ew = _word_from_node(end_node)
                if not sw or not ew or sw == ew:
                    continue

                rel = _rel_label(rel_uri)

                # Forward: sw → ew
                index.setdefault(sw, []).append((rel, ew, weight))
                # Reverse: ew → sw  (so lookups are symmetric)
                index.setdefault(ew, []).append((f"{rel}(inv)", sw, weight))
                count += 1

    except Exception as exc:
        print(f"[WARN] Error reading ConceptNet CSV: {exc}")

    _CN_CSV_INDEX = index
    _CN_CSV_LOADED = True
    print(f"[INFO] ConceptNet CSV index built: {count:,} edges, "
          f"{len(index):,} unique words.", flush=True)


def cn_edges_between(word_a: str, word_b: str,
                     csv_path: Optional[str] = None) -> list[dict]:
    """
    Find ALL ConceptNet edges between word_a and word_b.

    Priority:
      1. JSON API  (?node=A&other=B)  — exact pairwise query
      2. Local CSV index              — built lazily on first CSV request

    Both directions are checked so the result is symmetric.
    Results cached per pair.
    """
    key = (word_a.lower(), word_b.lower())
    if key in _CN_CACHE:
        return _CN_CACHE[key]

    # 1. JSON API
    if not _CN_DISABLED:
        edges = _cn_request(
            CN_API,
            {"node":  f"/c/en/{word_a.lower()}",
             "other": f"/c/en/{word_b.lower()}",
             "limit": 50},
        )
        if edges or not _CN_DISABLED:   # cache even empty result if API replied
            filtered = [e for e in edges if e.get("weight", 0) >= CN_MIN_WEIGHT]
            _CN_CACHE[key] = filtered
            return filtered

    # 2. Local CSV fallback
    _cn_load_csv(csv_path or _CN_CSV_DEFAULT)

    wa = word_a.lower().replace("_", " ")
    wb = word_b.lower().replace("_", " ")
    matched: list[dict] = []

    for rel, other, weight in _CN_CSV_INDEX.get(wa, []):
        if other == wb:
            matched.append({
                "start":  {"@id": f"/c/en/{wa}"},
                "end":    {"@id": f"/c/en/{wb}"},
                "rel":    {"label": rel},
                "weight": weight,
                "source": "csv",
            })

    if not matched:
        for rel, other, weight in _CN_CSV_INDEX.get(wb, []):
            if other == wa:
                matched.append({
                    "start":  {"@id": f"/c/en/{wa}"},
                    "end":    {"@id": f"/c/en/{wb}"},
                    "rel":    {"label": rel},
                    "weight": weight,
                    "source": "csv",
                })

    _CN_CACHE[key] = matched
    return matched




def _cn_neighbor_word(edge: dict, source_word: str) -> tuple[str, str]:
    """
    Given a CN edge and the word we came from, return (relation_type, neighbor_word).
    """
    start_id = edge.get("start", {}).get("@id", "")
    end_id   = edge.get("end",   {}).get("@id", "")
    rel      = edge.get("rel",   {}).get("label", "RelatedTo")

    # Extract plain word from /c/en/word or /c/en/word/pos/...
    def extract_word(node_id: str) -> str:
        parts = node_id.lstrip("/").split("/")
        # parts: ['c', 'en', 'word', ...]
        if len(parts) >= 3:
            return parts[2].replace("_", " ")
        return ""

    start_word = extract_word(start_id)
    end_word   = extract_word(end_id)
    src        = source_word.lower().replace("_", " ")

    if start_word == src:
        return rel, end_word
    elif end_word == src:
        # reverse edge — label it as reversed
        return f"{rel}(inv)", start_word
    else:
        # just return end_word as fallback
        return rel, end_word


# ---------------------------------------------------------------------------
# BFS path finding
# ---------------------------------------------------------------------------

def _node_label(node: str) -> str:
    """Human-readable label for a node (synset name or plain word)."""
    if _WN_OK and "." in node:
        try:
            return f"{node}  \"{wn.synset(node).definition()[:70]}\""
        except Exception:
            pass
    return node


def _build_step(
    step_index: int,
    node: str,
    relation: Optional[str],
    source: Optional[str],
    wn_method: Optional[str],
    cn_relation: Optional[str],
    is_normalization: bool = False,
) -> dict:
    step: dict = {
        "step":               step_index,
        "node":               node,
        "node_type":          "synset" if ("." in node and _WN_OK) else "word",
        "relation":           relation,
        "source":             source,
        "wordnet_method":     wn_method,
        "conceptnet_relation": cn_relation,
        "normalization_step": is_normalization,
    }
    if _WN_OK and "." in node:
        try:
            step["definition"] = wn.synset(node).definition()
        except Exception:
            pass
    return step


def wn_expand(words: list[str], max_depth: int = 8) -> dict[str, list[dict]]:
    """
    Pure WordNet BFS from a list of start words.
    Returns {lemma_word: path_to_reach_it} for every plain-word lemma
    reachable within max_depth WN edges (hypernym/meronym/holonym/deriv).
    The path is the list of step dicts from the start word to this lemma.
    """
    if not _WN_OK:
        return {}

    # node → best (shortest) path found so far
    best_paths: dict[str, list[dict]] = {}
    visited: set[str] = set()
    queue: deque[tuple[str, list[dict]]] = deque()

    for word in words:
        node = word.lower()
        step = _build_step(0, node, None, None, None, None)
        queue.append((node, [step]))
        best_paths[node] = [step]
        for syn in wn_synsets(word):
            snode = syn.name()
            sstep = _build_step(0, snode, "wn:synset_lookup", "WordNet", "synset_lookup", None)
            queue.append((snode, [step, sstep]))

    while queue:
        node, path = queue.popleft()
        if node in visited or len(path) > max_depth:
            continue
        visited.add(node)

        depth = len(path)

        if "." in node:
            try:
                syn = wn.synset(node)
                # Record all lemma words reachable at this synset
                for lemma in syn.lemma_names():
                    w = lemma.lower().replace("_", " ")
                    if w not in best_paths:
                        lstep = _build_step(depth, w, "wn:lemma", "WordNet", "lemma_lookup", None)
                        best_paths[w] = path + [lstep]

                # Expand via WN structural edges
                for wn_method, rel_label, neighbor in wn_edges_from_synset(syn):
                    if neighbor not in visited:
                        step = _build_step(depth, neighbor, rel_label, "WordNet", wn_method, None)
                        queue.append((neighbor, path + [step]))
            except Exception:
                pass
        else:
            for syn in wn_synsets(node):
                sn = syn.name()
                if sn not in visited:
                    step = _build_step(depth, sn, "wn:synset_lookup", "WordNet", "synset_lookup", None)
                    queue.append((sn, path + [step]))

    return best_paths


def bfs(
    start_words: list[str],
    target_word: str,
    max_depth: int = 8,
) -> Optional[list[dict]]:
    """
    Two-phase path finding:

    Phase 1 — WordNet only:
        BFS from start_words through WN edges (hypernym/meronym/holonym/deriv).
        If target_word (or a synset of it) is reached, return the path.

    Phase 2 — WordNet + ConceptNet as relationship checker:
        Expand both start_words and target_word independently via WordNet.
        For each lemma_a in start expansion × lemma_b in target expansion,
        ask ConceptNet: "is there any edge between lemma_a and lemma_b?"
        ConceptNet is used ONLY to confirm/discover a relationship between
        two already-known WN-derived concepts — it never introduces new words
        to follow.  Returns the combined path if a CN bridge is found.
    """
    if not _WN_OK:
        return None

    targets_wn  = {s.name() for s in wn_synsets(target_word)}
    targets_all = targets_wn | {target_word.lower()}

    # ── Phase 1: pure WordNet BFS ────────────────────────────────────────────
    visited: set[str] = set()
    queue: deque[tuple[str, list[dict]]] = deque()

    for word in start_words:
        node = word.lower()
        step = _build_step(0, node, None, None, None, None)
        queue.append((node, [step]))
        for syn in wn_synsets(word):
            snode = syn.name()
            sstep = _build_step(0, snode, "wn:synset_lookup", "WordNet", "synset_lookup", None)
            queue.append((snode, [step, sstep]))

    while queue:
        node, path = queue.popleft()
        if node in visited or len(path) > max_depth:
            continue
        visited.add(node)
        if node in targets_all:
            return path
        depth = len(path)
        if _WN_OK and "." in node:
            try:
                syn = wn.synset(node)
                for wn_method, rel_label, neighbor in wn_edges_from_synset(syn):
                    if neighbor not in visited:
                        step = _build_step(depth, neighbor, rel_label, "WordNet", wn_method, None)
                        queue.append((neighbor, path + [step]))
            except Exception:
                pass
        else:
            for syn in wn_synsets(node):
                sn = syn.name()
                if sn not in visited:
                    step = _build_step(depth, sn, "wn:synset_lookup", "WordNet", "synset_lookup", None)
                    queue.append((sn, path + [step]))

    # ── Phase 2: CN as relationship bridge ───────────────────────────────────
    # Skip only if both the API is down AND the CSV index is empty (or not found)
    if _CN_DISABLED and _CN_CSV_LOADED and not _CN_CSV_INDEX:
        return None
    if not _REQ_OK and not _CN_CSV_LOADED:
        # requests unavailable and CSV not yet tried — attempt CSV load
        pass

    expand_a = wn_expand(start_words, max_depth=max_depth)
    expand_b = wn_expand([target_word], max_depth=max_depth)

    # Check up to 40 most specific (shortest-path) WN-derived lemmas per side.
    # The CSV fallback is a local O(1) dict lookup, so expanding to 40×40 is
    # fine regardless of whether the API is up or down.
    lemmas_a = sorted(expand_a.keys(), key=lambda w: len(expand_a[w]))[:40]
    lemmas_b = sorted(expand_b.keys(), key=lambda w: len(expand_b[w]))[:40]

    def _build_bridge(la: str, lb: str, edges: list[dict]) -> list[dict]:
        edge    = edges[0]
        cn_rel  = edge.get("rel", {}).get("label", "RelatedTo")
        path_a  = expand_a.get(la, [_build_step(0, la, None, None, None, None)])
        path_b  = expand_b.get(lb, [_build_step(0, lb, None, None, None, None)])
        cn_step = _build_step(len(path_a), lb, f"cn:{cn_rel}", "ConceptNet", None, cn_rel)
        full    = path_a + [cn_step]
        for s in reversed(path_b):
            s2 = dict(s); s2["step"] = len(full); full.append(s2)
        return full

    for la in lemmas_a:
        for lb in lemmas_b:
            if la == lb:
                continue
            edges = cn_edges_between(la, lb)
            if edges:
                return _build_bridge(la, lb, edges)

    return None


# ---------------------------------------------------------------------------
# Root extraction
# ---------------------------------------------------------------------------

def extract_root(data: dict) -> str:
    """Return the name of the root (first) entity in the JSON."""
    entities = data.get("entities", [])
    if not entities:
        raise ValueError("No 'entities' key found in JSON.")
    first = entities[0]
    for key in ("entityName", "name"):
        if key in first:
            return first[key]
    raise ValueError(f"Cannot find entity name in first entity: {first}")


# ---------------------------------------------------------------------------
# Normalization: token → set of "meaning nouns"
# ---------------------------------------------------------------------------

def normalize_token(token: str) -> list[str]:
    """
    Convert a raw token into one or more "meaning nouns" — nouns that carry
    independent semantic content.

    - If the token is already a meaning noun (not an agent noun), return [token].
    - If it is an agent noun (-ant/-ent/-er/-or/-ee/-ist), derive its base verbs
      via WordNet and return all non-agent action/result nouns found.
      The original token is kept as a fallback if no action nouns are found.

    Example:
        "automobile" → ["automobile"]           (already a meaning noun)
        "applicant"  → ["application","practice","use",…]
        "student"    → ["study","education","training",…]
    """
    if not is_agent_noun(token):
        return [token.lower()]
    action = get_action_nouns(token)
    if action:
        return sorted(action)   # deterministic order
    return [token.lower()]      # fallback: keep original


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def _compare_normalized_pair(norm_a: str, norm_b: str,
                              orig_a: str, orig_b: str) -> dict:
    """
    Run the full pipeline for one normalized word pair (norm_a vs norm_b).
    orig_a / orig_b are the pre-normalization tokens (for reporting).
    """
    wn_methods_used:    list[str] = []
    cn_relations_found: list[str] = []

    # ── Stage 0: identical ─────────────────────────────────────────────────
    if norm_a == norm_b:
        return {
            "orig_a": orig_a, "orig_b": orig_b,
            "norm_a": norm_a, "norm_b": norm_b,
            "stage": "identical", "path_length": 0,
            "path_steps": [_build_step(0, norm_a, None, None, None, None)],
            "wup_score": 1.0,
            "wordnet_methods_used": [], "conceptnet_relations_found": [],
            "status": "found",
        }

    # ── Stage 1: wup_similarity ────────────────────────────────────────────
    wup_score, syn_a, syn_b = best_wup(norm_a, norm_b)
    if wup_score >= 0.70:
        return {
            "orig_a": orig_a, "orig_b": orig_b,
            "norm_a": norm_a, "norm_b": norm_b,
            "stage": "direct_wup", "path_length": 1,
            "path_steps": [
                _build_step(0, norm_a, None, None, None, None),
                _build_step(1, syn_a, "wn:synset_lookup", "WordNet", "synset_lookup", None),
                _build_step(2, syn_b, "wn:wup_similarity≥0.70", "WordNet", "wup_similarity", None),
                _build_step(3, norm_b, "wn:synset→word", "WordNet", "synset_lookup", None),
            ],
            "wup_score": round(wup_score, 4),
            "synset_a": syn_a, "synset_b": syn_b,
            "wordnet_methods_used": ["synset_lookup", "wup_similarity"],
            "conceptnet_relations_found": [],
            "status": "found",
        }

    # ── Stage 2: gloss mining ──────────────────────────────────────────────
    for w1, w2 in [(norm_a, norm_b), (norm_b, norm_a)]:
        hit = gloss_check(w1, w2)
        if hit:
            return {
                "orig_a": orig_a, "orig_b": orig_b,
                "norm_a": norm_a, "norm_b": norm_b,
                "stage": "gloss_mining", "direction": f"{w1}→{w2}",
                "path_length": 2,
                "path_steps": [
                    _build_step(0, w1, None, None, None, None),
                    _build_step(1, hit["synset"], "wn:synset_lookup", "WordNet", "synset_lookup", None),
                    _build_step(2, w2, "wn:gloss_mention", "WordNet", "gloss_mining", None),
                ],
                "gloss_synset": hit["synset"],
                "gloss_text":   hit["definition"],
                "matched_term": hit["matched_term"],
                "wup_score": round(wup_score, 4),
                "wordnet_methods_used": ["synset_lookup", "gloss_mining"],
                "conceptnet_relations_found": [],
                "status": "found",
            }

    # ── Stage 3: WN BFS + CN relationship check ────────────────────────────
    path = bfs([norm_a], norm_b)
    if path is None:
        path = bfs([norm_b], norm_a)

    if path:
        for step in path:
            if step.get("wordnet_method") and step["wordnet_method"] not in wn_methods_used:
                wn_methods_used.append(step["wordnet_method"])
            if step.get("conceptnet_relation") and step["conceptnet_relation"] not in cn_relations_found:
                cn_relations_found.append(step["conceptnet_relation"])
        return {
            "orig_a": orig_a, "orig_b": orig_b,
            "norm_a": norm_a, "norm_b": norm_b,
            "stage": "bfs", "path_length": len(path) - 1,
            "path_steps": path,
            "wup_score": round(wup_score, 4),
            "wordnet_methods_used": list(set(wn_methods_used)),
            "conceptnet_relations_found": list(set(cn_relations_found)),
            "status": "found",
        }

    return {
        "orig_a": orig_a, "orig_b": orig_b,
        "norm_a": norm_a, "norm_b": norm_b,
        "stage": "not_found", "path_length": 999,
        "path_steps": [], "wup_score": round(wup_score, 4),
        "wordnet_methods_used": [], "conceptnet_relations_found": [],
        "status": "not_found",
    }


def compare_roots(root_a: str, root_b: str) -> dict:
    """
    1. Split both roots into tokens (camelCase → individual words).
    2. Normalize each token upfront: agent nouns → action nouns.
    3. Compare one normalized word from A against one normalized word from B
       at a time (full cross-product).
    4. Return the result with the shortest path.
    """
    raw_tokens_a = split_camel(root_a) or [root_a.lower()]
    raw_tokens_b = split_camel(root_b) or [root_b.lower()]

    # Normalize every token → list of meaning nouns
    # Each entry: (original_token, normalized_word)
    norm_pairs_a = [(t, n) for t in raw_tokens_a for n in normalize_token(t)]
    norm_pairs_b = [(t, n) for t in raw_tokens_b for n in normalize_token(t)]

    print(f"  tokens_a (raw):  {raw_tokens_a}")
    print(f"  tokens_b (raw):  {raw_tokens_b}")
    print(f"  normalized_a:    {[n for _, n in norm_pairs_a]}")
    print(f"  normalized_b:    {[n for _, n in norm_pairs_b]}")

    best: Optional[dict] = None

    for orig_a, norm_a in norm_pairs_a:
        for orig_b, norm_b in norm_pairs_b:
            print(f"  '{norm_a}' (from '{orig_a}') <-> '{norm_b}' (from '{orig_b}') ...",
                  end=" ", flush=True)
            result = _compare_normalized_pair(norm_a, norm_b, orig_a, orig_b)
            print(f"stage={result['stage']}  path_len={result['path_length']}")
            if best is None or result["path_length"] < best["path_length"]:
                best = result

    if best is None:
        best = {
            "status": "not_found", "path_length": 999,
            "path_steps": [], "wordnet_methods_used": [],
            "conceptnet_relations_found": [],
        }

    return {
        "root_a":            root_a,
        "root_b":            root_b,
        "raw_tokens_a":      raw_tokens_a,
        "raw_tokens_b":      raw_tokens_b,
        "normalized_a":      [n for _, n in norm_pairs_a],
        "normalized_b":      [n for _, n in norm_pairs_b],
        "best_orig_pair":    [best.get("orig_a",""), best.get("orig_b","")],
        "best_norm_pair":    [best.get("norm_a",""), best.get("norm_b","")],
        "stage":             best.get("stage"),
        "path_length":       best.get("path_length"),
        "path_steps":        best.get("path_steps", []),
        "wup_score":         best.get("wup_score"),
        "wordnet_methods_used":       best.get("wordnet_methods_used", []),
        "conceptnet_relations_found": best.get("conceptnet_relations_found", []),
        "status":            best.get("status"),
        "all_pairs_tried":   [
            f"{na}(from {oa})<->{nb}(from {ob})"
            for oa, na in norm_pairs_a
            for ob, nb in norm_pairs_b
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _CN_CSV_DEFAULT
    parser = argparse.ArgumentParser(
        description="Compare roots of two ontology JSONs using WordNet + ConceptNet."
    )
    parser.add_argument("json_a", help="Path to first ontology JSON")
    parser.add_argument("json_b", help="Path to second ontology JSON")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON path (default: <stem_a>__<stem_b>__alignment.json)"
    )
    parser.add_argument(
        "--cn-csv", default=_CN_CSV_DEFAULT,
        help="Path to ConceptNet assertions CSV (fallback when API is down)"
    )
    args = parser.parse_args()

    # Override the default CSV path if the user supplied one
    _CN_CSV_DEFAULT = args.cn_csv

    path_a = Path(args.json_a)
    path_b = Path(args.json_b)
    for label, p in [("json_a", path_a), ("json_b", path_b)]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    with open(path_a, encoding="utf-8") as f:
        data_a = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        data_b = json.load(f)

    root_a = extract_root(data_a)
    root_b = extract_root(data_b)

    print(f"\nRoot A: {root_a}  (from {path_a.name})")
    print(f"Root B: {root_b}  (from {path_b.name})")
    print()

    alignment = compare_roots(root_a, root_b)

    output = {
        "json_a":     data_a,
        "json_b":     data_b,
        "alignment":  alignment,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(
            f"{path_a.stem}__{path_b.stem}__alignment.json"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nAlignment status : {alignment['status']}")
    print(f"Best orig pair   : {alignment['best_orig_pair']}")
    print(f"Best norm pair   : {alignment['best_norm_pair']}")
    print(f"Path length      : {alignment['path_length']}")
    print(f"Stage reached    : {alignment['stage']}")
    print(f"WN methods used  : {alignment['wordnet_methods_used']}")
    print(f"CN relations     : {alignment['conceptnet_relations_found']}")
    print(f"\nOutput written   : {out_path}")


if __name__ == "__main__":
    main()
