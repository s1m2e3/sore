"""
enriched_matcher.py
-------------------
3-Layer semantic enrichment pipeline over AML / LogMap matches.

Layer 1 — Characterise
    For every match already found by AML or LogMap, annotate it with the
    semantic relationship type (Synonym, Near-Synonym, Hypernym, Hyponym,
    PartOf, RelatedTo) derived from WordNet + ConceptNet.  This is NOT
    validation — it is characterisation: we want to know *why* two entities
    were matched so the relationship type is explicit in the output.

Layer 2 — Discover
    For entity pairs NOT found by either matcher, run the same token-level
    semantic analysis.  New candidates are split into two groups:
      • Equivalence  (= relation) — synonym or ≤1-hop hypernym/hyponym
      • Subsumption  (⊑ relation) — deeper hypernym chain or PartOf/MemberOf

Layer 3 — WUP Backup
    For any entity that still has ZERO candidate partners after L1+L2 (common
    when the two models operate at different levels of abstraction, e.g. an
    assembly-level model vs a component-level model), sweep all concept entities
    on the other side and take the top-k pairs whose blended WUP score
    (max_wup + avg_wup) / 2 >= threshold.  This ensures every entity has at
    least one representative pair so it contributes to composite scoring.
    source = "Layer3_WUP"; layer = 3.

Output: a single CSV written to enriched_ontology_matching/outputs/enriched/

Usage
-----
    venv/Scripts/python.exe enriched_ontology_matching/enriched_matcher.py \\
        enriched_ontology_matching/test_auto_v1_v2.json

    # run only AML or only LogMap:
    ... --matcher aml
    ... --matcher logmap
"""

from __future__ import annotations

import csv
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Reuse semantic machinery from root_comparator
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from root_comparator import (
    split_camel,
    best_wup,
    cn_edges_between,
    wn_synsets,
    gloss_check,
    _WN_OK,
    _CN_CSV_DEFAULT,
    _cn_load_csv,
)
from model_normalizer import load_inventory as _norm_load_inventory, normalize_model

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
_DIR        = Path(__file__).parent
_CN_CSV     = _CN_CSV_DEFAULT
_EQUIV_RELS = {"Identical", "Synonym", "Near-Synonym", "SimilarTo"}  # → = relation
_SUBSUM_RELS = {"Hypernym", "Hyponym", "PartOf", "MemberOf"}  # → ⊑ relation

# Canonical association types that express composition / containment
_PARTOF_CANONICALS = {"PartOf", "HasA", "MadeOf", "Composition", "hasPart"}


# ---------------------------------------------------------------------------
# Spurious match filters
# ---------------------------------------------------------------------------

def is_same_entity_attribute_match(ea: str, eb: str) -> bool:
    """
    Return True when both entity names are attributes of the SAME base entity
    (i.e., share the same 'EntityName__' prefix).

    Example: SteeringRack__steeringRackID  ↔  SteeringRack__rackType
    Both are attributes of SteeringRack — matching them to each other is wrong.
    """
    if "__" not in ea or "__" not in eb:
        return False
    return ea.split("__")[0].lower() == eb.split("__")[0].lower()

def build_partof_parents(model_json: dict) -> dict[str, str]:
    """
    Return {child_entity: parent_entity} for every PartOf/composition edge in
    a NORMALIZED model JSON (run normalize_model() first so canonical is set).
    """
    parents: dict[str, str] = {}
    for assoc in model_json.get("associations", []):
        if assoc.get("canonical", "") not in _PARTOF_CANONICALS:
            continue
        parts = (assoc.get("associationParticipants")
                 or assoc.get("participants") or [])
        if len(parts) >= 2:
            # Convention: parts[0] = child/part, parts[1] = parent/whole
            parents[parts[0]] = parts[1]
    return parents


def is_parent_name_embedded(ea: str, eb: str,
                             parents_a: dict[str, str],
                             parents_b: dict[str, str]) -> bool:
    """
    Return True when a proposed cross-model match (ea from A, eb from B) is
    spurious because ea's name embeds eb's name (or vice versa) and the two
    entities have a direct PartOf parent-child relationship within one model.

    Pattern caught:
      Model A: EngineBlock  partOf  Engine
      Model B: Engine
      Match proposed: EngineBlock (A) ↔ Engine (B)
      → spurious: EngineBlock is a PART of Engine, not equivalent to it.
    """
    toks_a = set(split_camel(ea))
    toks_b = set(split_camel(eb))

    # Name-embedding check: one token set is a proper subset of the other
    if not (toks_b < toks_a or toks_a < toks_b):
        return False

    # PartOf check: is the match a parent-child pair in either model?
    # Case 1: ea is a child of eb (or something with the same tokens) in model A
    parent_a = parents_a.get(ea, "")
    if parent_a and set(split_camel(parent_a)) == toks_b:
        return True

    # Case 2: eb is a child of ea (or something with the same tokens) in model B
    parent_b = parents_b.get(eb, "")
    if parent_b and set(split_camel(parent_b)) == toks_a:
        return True

    # Case 3: eb appears as a child in model A with parent named like ea
    parent_b_in_a = parents_a.get(eb, "")
    if parent_b_in_a and set(split_camel(parent_b_in_a)) == toks_a:
        return True

    # Case 4: ea appears as a child in model B with parent named like eb
    parent_a_in_b = parents_b.get(ea, "")
    if parent_a_in_b and set(split_camel(parent_a_in_b)) == toks_b:
        return True

    return False

# ConceptNet relations that imply equivalence
_CN_EQUIV = {"Synonym", "SimilarTo"}
# ConceptNet relations that imply directed subsumption (IsA = A is-a B → B is Hypernym)
_CN_ISA   = {"IsA"}
_CN_PART  = {"PartOf", "HasA", "MemberOf"}


# ---------------------------------------------------------------------------
# Layer 1 & 2 core: characterise a pair of tokens semantically
# ---------------------------------------------------------------------------

def _wn_direct_relation(syns_a, syns_b) -> Optional[tuple[str, int]]:
    """
    Check WordNet for a direct (≤2-hop) relation between two synset sets.
    Returns (relation_type, hop_count) or None.

    relation_type values: Synonym | Hypernym | Hyponym | PartOf | MemberOf
    hop_count: 0 = same synset, 1 = direct edge, 2 = two-hop
    """
    names_a = {s.name() for s in syns_a}
    names_b = {s.name() for s in syns_b}

    # Hop 0 — same synset
    if names_a & names_b:
        return "Synonym", 0

    # Hop 1 — direct hypernym / hyponym
    for sb in syns_b:
        for h in sb.hypernyms():
            if h.name() in names_a:
                return "Hypernym", 1   # A is more general than B
    for sa in syns_a:
        for h in sa.hypernyms():
            if h.name() in names_b:
                return "Hyponym", 1    # B is more general than A

    # Hop 1 — meronym / holonym  (PartOf / MemberOf)
    for sa in syns_a:
        mero = sa.part_meronyms() + sa.member_meronyms()
        holo = sa.part_holonyms() + sa.member_holonyms()
        for m in mero:
            if m.name() in names_b:
                return "PartOf", 1
        for m in holo:
            if m.name() in names_b:
                return "PartOf", 1
    for sb in syns_b:
        mero = sb.part_meronyms() + sb.member_meronyms()
        for m in mero:
            if m.name() in names_a:
                return "PartOf", 1

    # Hop 2 — two-hop hypernym
    for sb in syns_b:
        for h1 in sb.hypernyms():
            for h2 in h1.hypernyms():
                if h2.name() in names_a:
                    return "Hypernym", 2
    for sa in syns_a:
        for h1 in sa.hypernyms():
            for h2 in h1.hypernyms():
                if h2.name() in names_b:
                    return "Hyponym", 2

    return None


def _cn_relation_type(cn_edges: list[dict]) -> Optional[str]:
    """
    Distil a list of ConceptNet edges into a single semantic label.

    Priority order (most to least specific):
      Synonym/SimilarTo > IsA > PartOf/HasA/MemberOf >
      Antonym > UsedFor > CapableOf > Causes > HasProperty >
      AtLocation > HasPrerequisite > CausesDesire > MotivatedByGoal >
      HasContext > RelatedTo > (any other actual label)

    Returns the actual ConceptNet relation label — never collapses all
    non-structural relations into a generic "RelatedTo" bucket.
    """
    rels = {e.get("rel", {}).get("label", "") for e in cn_edges}
    # Strip (inv) suffix for classification
    bare = {r.replace("(inv)", "").strip() for r in rels if r}
    if not bare:
        return None
    if bare & _CN_EQUIV:
        return "Synonym"
    if bare & _CN_ISA:
        return "Hypernym"
    if bare & _CN_PART:
        return "PartOf"
    # Return the most specific actual CN label present (not a synthetic "RelatedTo")
    _OTHER_PRIORITY = [
        "Antonym",
        "UsedFor",
        "CapableOf",
        "Causes",
        "HasProperty",
        "AtLocation",
        "HasPrerequisite",
        "CausesDesire",
        "MotivatedByGoal",
        "HasContext",
        "RelatedTo",        # actual CN RelatedTo edge (not a catch-all)
    ]
    for rel in _OTHER_PRIORITY:
        if rel in bare:
            return rel
    return next(iter(bare))   # any remaining label not in the priority list


@lru_cache(maxsize=None)
def characterise_token_pair(ta: str, tb: str) -> dict:
    """
    Full semantic characterisation of a single token pair (ta, tb).

    Returns a dict with keys:
      token_a, token_b,
      wup_score,        — WordNet Wu-Palmer similarity (0–1)
      wn_relation,      — Synonym|Hypernym|Hyponym|PartOf|RelatedTo|None
      wn_hops,          — hop count in WordNet (0/1/2/-1 if not via WN)
      cn_relations,     — comma-separated raw ConceptNet relation labels
      cn_label,         — distilled CN label (Synonym|Hypernym|PartOf|RelatedTo|None)
      gloss_hit,        — True if tb appears in ta's synset definition
      semantic_label,   — final composite label (most specific wins)
      layer2_type,      — "Equivalence" | "Subsumption" | "None"

    Cached: token vocabulary is small (~200/domain), so this eliminates the
    dominant O(|A|×|B|×|tokens|²) WN/CN overhead across Layer 2 and coherence.
    The returned dict must NOT be mutated by callers.
    """
    ta = ta.lower().strip()
    tb = tb.lower().strip()

    result = {
        "token_a":      ta,
        "token_b":      tb,
        "wup_score":    0.0,
        "wn_relation":  "None",
        "wn_hops":      -1,
        "cn_relations": "",
        "cn_label":     "None",
        "gloss_hit":    False,
        "semantic_label": "Unrelated",
        "layer2_type":  "None",
    }

    # ── exact match ────────────────────────────────────────────────────────
    if ta == tb:
        result.update(wup_score=1.0, wn_relation="Identical", wn_hops=0,
                      semantic_label="Identical", layer2_type="Equivalence")
        return result

    # ── WordNet ────────────────────────────────────────────────────────────
    if _WN_OK:
        # Restrict to top-N most-frequent synsets per word to avoid polysemy
        # false positives (e.g. "line" and "air" both appear in tune.n.01
        # but are unrelated in an automotive context).
        _MAX_SYNS = 5
        syns_a = wn_synsets(ta)[:_MAX_SYNS]
        syns_b = wn_synsets(tb)[:_MAX_SYNS]

        # Restricted WUP: same top-5 restriction
        if syns_a and syns_b:
            wup = max(
                (sa.wup_similarity(sb) or 0.0 for sa in syns_a for sb in syns_b),
                default=0.0,
            )
        else:
            wup = 0.0
        result["wup_score"] = round(wup, 4)

        # Lemma-set overlap (different spellings of same concept)
        lemmas_a = {l for s in syns_a for l in s.lemma_names()}
        lemmas_b = {l for s in syns_b for l in s.lemma_names()}
        if lemmas_a & lemmas_b:
            result.update(wn_relation="Synonym", wn_hops=0)
        elif syns_a and syns_b:
            wn_hit = _wn_direct_relation(syns_a, syns_b)
            if wn_hit:
                result["wn_relation"], result["wn_hops"] = wn_hit

        # Near-synonym by high wup when no direct synset path.
        # Threshold 0.92 (not 0.85) to avoid "physical artifact" WN proximity
        # false positives: many unrelated devices (fuse, filter, blower, gear)
        # share device.n.01/instrumentality.n.03 as their LCS at depth ~8,
        # which yields wup ≈ 0.8889 even for semantically unrelated components.
        if result["wn_relation"] == "None" and wup >= 0.92:
            result["wn_relation"] = "Near-Synonym"

        # Gloss check
        g = gloss_check(ta, tb)
        if g:
            result["gloss_hit"] = True

    # ── ConceptNet ─────────────────────────────────────────────────────────
    cn_edges = cn_edges_between(ta, tb, csv_path=_CN_CSV)
    if cn_edges:
        raw_rels = [e.get("rel", {}).get("label", "") for e in cn_edges]
        result["cn_relations"] = ", ".join(sorted(set(raw_rels)))
        result["cn_label"] = _cn_relation_type(cn_edges) or "None"

    # ── Composite semantic label ────────────────────────────────────────────
    # Priority: direct WN synset/lemma > CN > gloss > WN-proximity fallback
    #
    # Label provenance:
    #   "Synonym" / "Near-Synonym" / "Hypernym" / "Hyponym" / "PartOf" / "MemberOf"
    #       — from WordNet synset structure
    #   "RelatedTo" / "UsedFor" / "AtLocation" / "CapableOf" / etc.
    #       — from ConceptNet (cn_label is the raw CN relation)
    #   "GlossOverlap"
    #       — WordNet gloss text overlap; not a CN edge
    #   "WN-Proximate"
    #       — WUP ≥ 0.5 but NO direct WN or CN edge found; purely positional
    #         similarity in the WordNet hierarchy; do NOT interpret as a confirmed
    #         semantic relationship
    wn_rel = result["wn_relation"]
    cn_lbl = result["cn_label"]

    if wn_rel == "Synonym":
        label = "Synonym"
    elif wn_rel == "Near-Synonym" or cn_lbl == "Synonym":
        label = "Near-Synonym"
    elif wn_rel in ("Hypernym", "Hyponym"):
        label = wn_rel
    elif cn_lbl == "Hypernym":
        label = "Hypernym"
    elif wn_rel in ("PartOf", "MemberOf") or cn_lbl == "PartOf":
        label = "PartOf"
    elif cn_lbl not in (None, "None", ""):
        # CN edge takes priority over gloss overlap — a confirmed edge is stronger
        # evidence than shared text in a definition.
        # This may be "RelatedTo" (CN's own generic label), or a specific
        # functional label: UsedFor, CapableOf, AtLocation, Causes, etc.
        label = cn_lbl
    elif result["gloss_hit"]:
        label = "GlossOverlap"   # WordNet gloss text overlap — no CN edge found
    elif result["wup_score"] >= 0.3:
        label = "WN-Proximate"   # WN hierarchy proximity only — no confirmed edge
    else:
        label = "Unrelated"

    result["semantic_label"] = label

    # Layer 2 type
    if label in _EQUIV_RELS:
        result["layer2_type"] = "Equivalence"
    elif label in _SUBSUM_RELS:
        result["layer2_type"] = "Subsumption"
    elif result["cn_label"] not in (None, "None", ""):
        # CN found a direct edge — preserve the actual relation as layer2_type
        # so Layer 2 discovery can emit it rather than silently dropping the pair.
        result["layer2_type"] = result["cn_label"]
    else:
        result["layer2_type"] = "None"

    return result


def _result_from_cn_compound(term_a: str, term_b: str,
                             cn_edges: list[dict]) -> dict:
    """
    Build a minimal characterisation result from CN edges between compound terms.
    WordNet fields are left at defaults since WN doesn't index multi-word compounds.
    """
    raw_rels = [e.get("rel", {}).get("label", "") for e in cn_edges]
    cn_lbl = _cn_relation_type(cn_edges) or "None"

    # Composite label — same priority logic as characterise_token_pair
    if cn_lbl in _CN_EQUIV:
        label = "Synonym"
    elif cn_lbl == "IsA":
        label = "Hypernym"
    elif cn_lbl in _CN_PART:
        label = "PartOf"
    elif cn_lbl not in (None, "None", ""):
        label = cn_lbl
    else:
        label = "Unrelated"

    if label in _EQUIV_RELS:
        l2 = "Equivalence"
    elif label in _SUBSUM_RELS:
        l2 = "Subsumption"
    elif cn_lbl not in (None, "None", ""):
        l2 = cn_lbl
    else:
        l2 = "None"

    return {
        "token_a":        term_a,
        "token_b":        term_b,
        "wup_score":      0.0,
        "max_wup":        0.0,
        "avg_wup":        0.0,
        "wn_relation":    "None",
        "wn_hops":        -1,
        "cn_relations":   ", ".join(sorted(set(raw_rels))),
        "cn_label":       cn_lbl,
        "gloss_hit":      False,
        "semantic_label": label,
        "layer2_type":    l2,
    }


@lru_cache(maxsize=None)
def characterise_entity_pair(entity_a: str, entity_b: str) -> dict:
    """
    Characterise a pair of entity names by comparing all token combinations
    and returning the result from the most semantically informative pair.

    Priority: Synonym > Near-Synonym > Hypernym/Hyponym > PartOf > RelatedTo

    Two passes are performed:
      1. Token-level: split_camel tokens compared via WN + CN (existing logic)
      2. Compound-level: multi-word phrases queried directly in CN (new)
         e.g. "water pump" PartOf "cooling system" would be missed by token pass

    Cached: entity pairs repeat heavily across Layer 2, neighbourhood coherence,
    and multiple within-domain pairs. Cache hits skip all WN/CN work entirely.
    The returned dict must NOT be mutated by callers.
    """
    # ── Exact-name short-circuit ────────────────────────────────────────────
    # If both entity names are identical (case-insensitive) skip all WN/CN
    # lookups — the relationship is trivially "Identical", not merely synonymous.
    if entity_a.lower() == entity_b.lower():
        tok = entity_a.lower()
        return {
            "token_a": tok, "token_b": tok,
            "wup_score": 1.0, "max_wup": 1.0, "avg_wup": 1.0,
            "wn_relation": "Identical", "wn_hops": 0,
            "cn_relations": "", "cn_label": "None",
            "gloss_hit": False,
            "semantic_label": "Identical", "layer2_type": "Equivalence",
        }

    tokens_a = split_camel(entity_a) or [entity_a.lower()]
    tokens_b = split_camel(entity_b) or [entity_b.lower()]

    # Remove very generic stop tokens that pollute matching
    _STOP = {"system", "type", "model", "base", "item", "unit", "data", "info"}
    tokens_a_f = [t for t in tokens_a if t not in _STOP] or tokens_a
    tokens_b_f = [t for t in tokens_b if t not in _STOP] or tokens_b

    _PRIORITY = {
        "Identical": 7,                          # same name — no lookup needed
        "Synonym": 6, "Near-Synonym": 5,
        "Hypernym": 4, "Hyponym": 4,
        "PartOf": 3, "MemberOf": 3,
        # CN-confirmed edges (including CN's own generic RelatedTo)
        "RelatedTo": 2, "UsedFor": 2, "CapableOf": 2, "Causes": 2,
        "HasProperty": 2, "AtLocation": 2, "HasPrerequisite": 2,
        "CausesDesire": 2, "MotivatedByGoal": 2, "HasContext": 2,
        "Antonym": 2,
        # WN-only signals with no confirmed edge
        "GlossOverlap": 2,  # WN gloss text overlap — weaker than CN edge
        "WN-Proximate": 1,  # WUP ≥ 0.5 fallback — positional only, no edge
        "Unrelated": 0,
    }

    best: Optional[dict] = None
    best_pri = 0
    all_wups: list[float] = []  # WUP score for every (token_a, token_b) pair

    # ── Pass 1: token-level (WN + CN) ──────────────────────────────────────
    for ta in tokens_a_f:
        for tb in tokens_b_f:
            r = characterise_token_pair(ta, tb)
            all_wups.append(r["wup_score"])
            pri = _PRIORITY.get(r["semantic_label"], 0)
            if pri > best_pri:
                best_pri = pri
                best = r
        # Synonym/Identical found — no higher priority possible, skip remaining tokens_a
        if best_pri >= 6:
            break

    # ── Pass 2: compound-level CN lookup ───────────────────────────────────
    # Build candidate (term_a, term_b) pairs that use multi-word compound forms.
    # Only attempted when the current best hasn't already hit priority ≥4
    # (Synonym/Near-Synonym/Hypernym already found by token pass).
    #
    # We try BOTH filtered (stop-words removed) AND unfiltered compounds because
    # stop-word removal can break valid CN entries: "CoolingSystem" filters to
    # "cooling" but ConceptNet has the edge for "cooling system" (full phrase).
    if best_pri < 4:
        compound_pairs: set[tuple[str, str]] = set()

        # Build compound strings from both filtered and unfiltered token lists
        for toks_a, toks_b in [
            (tokens_a_f, tokens_b_f),   # stop-words removed
            (tokens_a,   tokens_b),     # unfiltered (preserves "system" etc.)
        ]:
            comp_a = " ".join(toks_a)
            comp_b = " ".join(toks_b)

            if len(toks_a) > 1 and len(toks_b) > 1:
                compound_pairs.add((comp_a, comp_b))
            if len(toks_a) > 1:
                for tb in toks_b:
                    compound_pairs.add((comp_a, tb))
            if len(toks_b) > 1:
                for ta in toks_a:
                    compound_pairs.add((ta, comp_b))

        for term_a, term_b in compound_pairs:
            edges = cn_edges_between(term_a, term_b, csv_path=_CN_CSV)
            if not edges:
                continue
            r = _result_from_cn_compound(term_a, term_b, edges)
            pri = _PRIORITY.get(r["semantic_label"], 0)
            if pri > best_pri:
                best_pri = pri
                best = r

    # ── Compute many-to-many WUP aggregates ────────────────────────────────
    # all_wups holds one entry per (token_a_i, token_b_j) pair; compute
    # max and average across all N×M combinations so both metrics are reported.
    if all_wups:
        max_wup = round(max(all_wups), 4)
        avg_wup = round(sum(all_wups) / len(all_wups), 4)
    else:
        max_wup = avg_wup = 0.0

    if best is None:
        best = characterise_token_pair(tokens_a_f[0], tokens_b_f[0])
        # Single-token fallback: max == avg == wup_score
        max_wup = avg_wup = round(best["wup_score"], 4)

    # Copy before mutating: characterise_token_pair results are LRU-cached
    # and must not be modified in place.
    best = dict(best)
    best["max_wup"] = max_wup
    best["avg_wup"] = avg_wup
    return best


# ---------------------------------------------------------------------------
# Layer 1 — annotate existing matches
# ---------------------------------------------------------------------------

def layer1_annotate(
    matches: list[dict],
    source_label: str,
) -> list[dict]:
    """
    Annotate each match dict with semantic characterisation fields.

    Input match dicts must have: entity_a, entity_b, confidence, relation.
    Returns enriched dicts with additional fields from characterise_entity_pair.
    """
    enriched = []
    for m in matches:
        char = characterise_entity_pair(m["entity_a"], m["entity_b"])
        row = {
            "entity_a":       m["entity_a"],
            "entity_b":       m["entity_b"],
            "source":         source_label,
            "matcher_conf":   round(m.get("confidence", 0.0), 4),
            "layer":          1,
            **char,
        }
        enriched.append(row)
    return enriched


# ---------------------------------------------------------------------------
# Layer 2 — discover new candidates among unmatched entities
# ---------------------------------------------------------------------------

def _head_token(entity_name: str) -> str:
    """
    Return the head (most specific) token of an entity name.
    In English compound nouns the head is the LAST token: 'FuelInjector' -> 'injector'.
    For associations expressed as verb phrases ('OilPumpLubricatesEngine'),
    the head is still extracted but the entity is flagged as an association.
    """
    tokens = split_camel(entity_name)
    return tokens[-1] if tokens else entity_name.lower()


def _is_association(entity_name: str) -> bool:
    """
    Heuristic: an entity whose name contains a verb-like connector
    (To, From, Of, In, On, Has, Is, etc.) in a middle position is
    likely an association / relationship, not a standalone concept.

    NOTE: uses a raw regex split (not split_camel) because split_camel
    drops short words like "To", "In", "On" that are the key connectors.
    """
    # Raw CamelCase split that preserves ALL words, including short ones
    raw_tokens = re.findall(r'[A-Z][a-z]*|[a-z]+', entity_name)
    tokens = [t.lower() for t in raw_tokens]
    # Short names (≤2 tokens) are plain concepts
    if len(tokens) <= 2:
        return False
    # If any middle token looks like a verb/preposition connector
    _CONNECTORS = {"to", "from", "of", "on", "in", "at", "via", "with",
                   "has", "have", "is", "are", "and", "for",
                   "drives", "powers", "feeds", "links", "couples",
                   "connects", "lubricates", "monitors", "assists",
                   "pressures", "charges", "engages", "modulates",
                   "syncs", "taps", "supplies", "aids", "turns",
                   "clamps", "circulates"}
    return any(t in _CONNECTORS for t in tokens[1:-1])


def layer2_discover(
    entities_a: list[str],
    entities_b: list[str],
    already_matched_a: set[str],
    already_matched_b: set[str],
    parents_a: dict[str, str] | None = None,
    parents_b: dict[str, str] | None = None,
) -> list[dict]:
    """
    For every unmatched entity in A × unmatched entity in B, run
    characterise_entity_pair and emit rows for non-trivial semantic links.

    Filters:
    - Skip pairs where both sides are associations (verb-phrase names).
    - Require that the matched token is the HEAD token of at least one
      entity (prevents spurious hits on sub-tokens of association names).

    Emits:
      • layer2_type == Equivalence          → source "Layer2_equivalence"
      • layer2_type == Subsumption          → source "Layer2_subsumption"
      • layer2_type == <CN relation label>  → source "Layer2_<cn_label>" (e.g. "Layer2_UsedFor")
      • layer2_type == None                 → skipped
    """
    unmatched_a = [e for e in entities_a if e not in already_matched_a]
    unmatched_b = [e for e in entities_b if e not in already_matched_b]

    print(f"  [Layer2] Searching {len(unmatched_a)} x {len(unmatched_b)} "
          f"unmatched pairs for semantic links ...")

    # Pre-filter associations at the outer loop level to avoid O(|B|) redundant
    # _is_association(ea) calls inside the inner loop.
    concept_a = [e for e in unmatched_a if not _is_association(e)]
    concept_b = [e for e in unmatched_b if not _is_association(e)]

    _par_a = parents_a or {}
    _par_b = parents_b or {}

    rows = []
    for ea in concept_a:
        for eb in concept_b:
            # Filter spurious matches: same-entity attribute pairs or
            # parent-name-embedded PartOf pairs.
            if (is_same_entity_attribute_match(ea, eb)
                    or is_parent_name_embedded(ea, eb, _par_a, _par_b)):
                continue

            char = characterise_entity_pair(ea, eb)
            l2   = char["layer2_type"]
            if l2 == "None":
                continue

            # Guard: matched token must be the head token of at least one side
            head_a = _head_token(ea)
            head_b = _head_token(eb)
            if char["token_a"] != head_a and char["token_b"] != head_b:
                continue

            row = {
                "entity_a":     ea,
                "entity_b":     eb,
                "source":       f"Layer2_{l2}",
                "matcher_conf": "",
                "layer":        2,
                **char,
            }
            rows.append(row)

    equiv  = sum(1 for r in rows if r["layer2_type"] == "Equivalence")
    subsum = sum(1 for r in rows if r["layer2_type"] == "Subsumption")
    cn_rel = sum(1 for r in rows if r["layer2_type"] not in ("Equivalence", "Subsumption"))
    print(f"  [Layer2] Found {equiv} equivalence + {subsum} subsumption + {cn_rel} CN-relation candidates.")
    return rows


# ---------------------------------------------------------------------------
# Layer 3 — WUP backup for orphan entities
# ---------------------------------------------------------------------------

def layer3_wup_backup(
    entities_a: list[str],
    entities_b: list[str],
    covered_a: set[str],
    covered_b: set[str],
    threshold: float = 0.9,
    top_k: int = 3,
) -> list[dict]:
    """
    For every entity with zero partners after L1+L2, find its top-k WUP
    matches on the other side where max_wup >= threshold.

    Using max_wup (not blended) as the gate: a max_wup >= 0.9 means the
    best token pair is nearly identical in WordNet — typically an exact
    shared root word (fuel↔fuel, brake↔brake).  This avoids spurious
    matches driven by the generic WordNet machine/artifact hierarchy
    inflating blended scores for unrelated concepts.

    entity_a / entity_b orientation is preserved: candidates from A go in
    entity_a; candidates from B go in entity_b.
    """
    orphans_a = [e for e in entities_a if e not in covered_a and not _is_association(e)]
    orphans_b = [e for e in entities_b if e not in covered_b and not _is_association(e)]

    if not orphans_a and not orphans_b:
        return []

    print(f"  [Layer3] WUP backup: {len(orphans_a)} orphan(s) in A, "
          f"{len(orphans_b)} in B (max_wup threshold={threshold}) ...")

    added: set[tuple[str, str]] = set()
    rows: list[dict] = []

    # For each orphan in A, find best WUP partners in ALL of B
    concept_b = [e for e in entities_b if not _is_association(e)]
    for ea in orphans_a:
        scored = []
        for eb in concept_b:
            char = characterise_entity_pair(ea, eb)
            if char["max_wup"] >= threshold:
                scored.append((char["max_wup"], eb, char))
        scored.sort(key=lambda x: -x[0])
        for _, eb, char in scored[:top_k]:
            key = (ea, eb)
            if key in added:
                continue
            added.add(key)
            rows.append({
                "entity_a":     ea,
                "entity_b":     eb,
                "source":       "Layer3_WUP",
                "matcher_conf": "",
                "layer":        3,
                **char,
            })

    # For each orphan in B, find best WUP partners in ALL of A
    concept_a = [e for e in entities_a if not _is_association(e)]
    for eb in orphans_b:
        scored = []
        for ea in concept_a:
            char = characterise_entity_pair(ea, eb)  # always (A, B) orientation
            if char["max_wup"] >= threshold:
                scored.append((char["max_wup"], ea, char))
        scored.sort(key=lambda x: -x[0])
        for _, ea, char in scored[:top_k]:
            key = (ea, eb)
            if key in added:
                continue
            added.add(key)
            rows.append({
                "entity_a":     ea,
                "entity_b":     eb,
                "source":       "Layer3_WUP",
                "matcher_conf": "",
                "layer":        3,
                **char,
            })

    print(f"  [Layer3] Added {len(rows)} WUP-backup pair(s).")
    return rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "entity_a", "entity_b", "source", "matcher_conf", "layer",
    "token_a", "token_b",
    "wup_score", "max_wup", "avg_wup", "wn_relation", "wn_hops",
    "cn_relations", "cn_label",
    "gloss_hit", "semantic_label", "layer2_type",
]


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  [CSV] Written {len(rows)} rows -> {out_path}")


# ---------------------------------------------------------------------------
# Entity extraction helper
# ---------------------------------------------------------------------------

def _extract_entities(json_data: dict) -> list[str]:
    """Extract all entity and association names from a conceptual-model JSON."""
    names = []
    for ent in json_data.get("entities", []):
        n = ent.get("entityName") or ent.get("name", "")
        if n:
            names.append(n)
    for assoc in json_data.get("associations", []):
        n = assoc.get("associationName") or assoc.get("name", "")
        if n:
            names.append(n)
    return names


def _extract_concept_entities(json_data: dict) -> list[str]:
    """
    Extract ONLY the standalone concept entity names (not associations).
    Used for Layer 2 discovery so that associations (verb-phrase names) are
    never proposed as equivalence or subsumption candidates.
    """
    names = []
    for ent in json_data.get("entities", []):
        n = ent.get("entityName") or ent.get("name", "")
        if n:
            names.append(n)
    return names


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    test_json_path: str | Path,
    use_aml: bool = True,
    use_logmap: bool = True,
    cn_csv: Optional[str] = None,
) -> Path:
    """
    Full 3-layer enrichment pipeline on a test JSON file.

    1. Load json_a and json_b from the test file.
    2. Run AML and/or LogMap to get candidate matches.
    3. Layer 1: characterise every match with WordNet + ConceptNet.
    4. Layer 2: discover new equivalence / subsumption candidates.
    5. Layer 3: WUP backup — add top-k WUP pairs for any entity still orphaned.
    6. Write unified CSV.

    Returns the path to the output CSV.
    """
    global _CN_CSV
    if cn_csv:
        _CN_CSV = cn_csv

    path = Path(test_json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Test JSON not found: {path}\n"
            "Expected a file with 'json_a' and 'json_b' keys produced by run_all_pairs.py."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    json_a = data.get("json_a", data)
    json_b = data.get("json_b", {})

    name_a = json_a.get("modelName", path.stem + "_a")
    name_b = json_b.get("modelName", path.stem + "_b")

    # ── Normalize: canonical associations + composition → PartOf ─────────────
    _inv = _norm_load_inventory()
    json_a = normalize_model(json_a, _inv)
    json_b = normalize_model(json_b, _inv)

    print(f"\n{'='*70}")
    print(f"Enriched Matcher: {name_a}")
    print(f"           vs   : {name_b}")
    print(f"{'='*70}")

    # ── Run matchers ────────────────────────────────────────────────────────
    all_matches: list[dict] = []

    if use_aml:
        from aml_runner import AMLRunner
        print("\n[AML] Running ...")
        aml_maps = AMLRunner().match_jsons(json_a, json_b,
                                           name_a=name_a, name_b=name_b)
        all_matches.append(("AML", aml_maps))

    if use_logmap:
        from logmap_runner import LogMapRunner
        print("\n[LogMap] Running ...")
        lm_maps = LogMapRunner().match_jsons(json_a, json_b,
                                             name_a=name_a, name_b=name_b)
        all_matches.append(("LogMap", lm_maps))

    # ── Merge: deduplicate, mark "Both" ─────────────────────────────────────
    # key = (entity_a, entity_b)
    merged: dict[tuple, dict] = {}
    for source, maps in all_matches:
        for m in maps:
            key = (m["entity_a"], m["entity_b"])
            if key in merged:
                merged[key]["source"] = "Both"
                merged[key]["aml_conf"]    = merged[key].get("aml_conf", "")
                merged[key]["logmap_conf"] = merged[key].get("logmap_conf", "")
                merged[key][f"{source.lower()}_conf"] = m["confidence"]
            else:
                merged[key] = {
                    "entity_a":   m["entity_a"],
                    "entity_b":   m["entity_b"],
                    "confidence": m["confidence"],
                    "relation":   m.get("relation", "="),
                    "source":     source,
                }

    merged_list = list(merged.values())
    print(f"\n[Merge] {len(merged_list)} unique matches "
          f"({sum(1 for v in merged.values() if v['source']=='Both')} found by both)")

    # Pre-load ConceptNet CSV once before the per-pair loops
    print(f"\n[CN] Loading ConceptNet CSV (first time only — may take ~30s) ...")
    _cn_load_csv(_CN_CSV)

    # ── Layer 1: characterise ────────────────────────────────────────────────
    print("\n[Layer 1] Characterising matched entity pairs ...")
    layer1_rows = layer1_annotate(merged_list, source_label="")
    # Restore per-match source label
    for row, orig in zip(layer1_rows, merged_list):
        row["source"] = orig["source"]

    # ── Layer 2: discover ────────────────────────────────────────────────────
    # Use only standalone concept entities (not associations) so that
    # verb-phrase relationship names never appear as equivalence candidates.
    print("\n[Layer 2] Discovering new semantic candidates ...")
    entities_a = _extract_concept_entities(json_a)
    entities_b = _extract_concept_entities(json_b)
    matched_a  = {m["entity_a"] for m in merged_list}
    matched_b  = {m["entity_b"] for m in merged_list}

    # Build PartOf parent maps from normalized models for spurious-match filtering
    parents_a = build_partof_parents(json_a)
    parents_b = build_partof_parents(json_b)

    # Filter Layer 1 AML/LogMap matches that are spurious
    filtered_spurious = 0
    filtered_layer1 = []
    for row in layer1_rows:
        ea, eb = row["entity_a"], row["entity_b"]
        if (is_same_entity_attribute_match(ea, eb)
                or is_parent_name_embedded(ea, eb, parents_a, parents_b)):
            filtered_spurious += 1
        else:
            filtered_layer1.append(row)
    if filtered_spurious:
        print(f"  [Filter] Removed {filtered_spurious} spurious L1 match(es).")
    layer1_rows = filtered_layer1

    layer2_rows = layer2_discover(entities_a, entities_b, matched_a, matched_b,
                                  parents_a=parents_a, parents_b=parents_b)

    # ── Layer 3: WUP backup for orphan entities ──────────────────────────────
    covered_a = {r["entity_a"] for r in layer1_rows + layer2_rows}
    covered_b = {r["entity_b"] for r in layer1_rows + layer2_rows}
    layer3_rows = layer3_wup_backup(entities_a, entities_b, covered_a, covered_b)

    # ── Write CSV ────────────────────────────────────────────────────────────
    from logmap_runner import _safe_local
    stem   = f"{_safe_local(name_a)}_vs_{_safe_local(name_b)}"
    out    = _DIR / "outputs" / "enriched" / f"{stem}.csv"
    all_rows = layer1_rows + layer2_rows + layer3_rows
    write_csv(all_rows, out)

    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="2-Layer semantic enrichment over AML/LogMap matches."
    )
    parser.add_argument("test_json",
                        help="Path to test JSON with json_a / json_b keys.")
    parser.add_argument("--matcher", choices=["aml", "logmap", "both"],
                        default="both",
                        help="Which matcher(s) to run (default: both).")
    parser.add_argument("--cn-csv", default=None,
                        help="Override path to ConceptNet assertions CSV.")
    args = parser.parse_args()

    csv_path = run_pipeline(
        args.test_json,
        use_aml    = args.matcher in ("aml",    "both"),
        use_logmap = args.matcher in ("logmap", "both"),
        cn_csv     = args.cn_csv,
    )

    # ── Print summary table from the CSV ────────────────────────────────────
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

    l1 = [r for r in rows if r["layer"] == "1"]
    l2 = [r for r in rows if r["layer"] == "2"]

    print(f"\n{'='*90}")
    print(f"LAYER 1 — {len(l1)} characterised matches")
    print(f"{'='*90}")
    print(f"{'Entity A':<32} {'Entity B':<32} {'Src':<6} {'Conf':>5}  "
          f"{'Semantic':<14} {'WUP':>5}  {'CN'}")
    print("-"*90)
    for r in sorted(l1, key=lambda x: x["semantic_label"]):
        conf = r["matcher_conf"] or ""
        print(f"{r['entity_a']:<32} {r['entity_b']:<32} {r['source']:<6} "
              f"{conf:>5}  {r['semantic_label']:<14} {r['wup_score']:>5}  "
              f"{r['cn_relations'][:25]}")

    if l2:
        equiv  = [r for r in l2 if r["layer2_type"] == "Equivalence"]
        subsum = [r for r in l2 if r["layer2_type"] == "Subsumption"]
        cn_rel = [r for r in l2 if r["layer2_type"] not in ("Equivalence", "Subsumption")]

        if equiv:
            print(f"\n{'='*90}")
            print(f"LAYER 2 — {len(equiv)} new equivalence candidates")
            print(f"{'='*90}")
            print(f"{'Entity A':<32} {'Entity B':<32} {'Semantic':<14} "
                  f"{'WUP':>5}  {'Tokens'}")
            print("-"*90)
            for r in sorted(equiv, key=lambda x: -float(x["wup_score"] or 0)):
                print(f"{r['entity_a']:<32} {r['entity_b']:<32} "
                      f"{r['semantic_label']:<14} {r['wup_score']:>5}  "
                      f"{r['token_a']} / {r['token_b']}")

        if subsum:
            print(f"\n{'='*90}")
            print(f"LAYER 2 — {len(subsum)} subsumption candidates  (A subClassOf B or B subClassOf A)")
            print(f"{'='*90}")
            print(f"{'Entity A':<32} {'Entity B':<32} {'Relation':<14} "
                  f"{'WUP':>5}  {'Tokens'}")
            print("-"*90)
            for r in sorted(subsum,
                            key=lambda x: (-float(x["wup_score"] or 0), x["entity_a"])):
                print(f"{r['entity_a']:<32} {r['entity_b']:<32} "
                      f"{r['wn_relation']:<14} {r['wup_score']:>5}  "
                      f"{r['token_a']} / {r['token_b']}")

        if cn_rel:
            print(f"\n{'='*90}")
            print(f"LAYER 2 — {len(cn_rel)} ConceptNet-backed relation candidates")
            print(f"{'='*90}")
            print(f"{'Entity A':<32} {'Entity B':<32} {'CN Relation':<16} "
                  f"{'WUP':>5}  {'Tokens'}")
            print("-"*90)
            for r in sorted(cn_rel,
                            key=lambda x: (x["layer2_type"], -float(x["wup_score"] or 0))):
                print(f"{r['entity_a']:<32} {r['entity_b']:<32} "
                      f"{r['layer2_type']:<16} {r['wup_score']:>5}  "
                      f"{r['token_a']} / {r['token_b']}")

    print(f"\n[Done] Full results: {csv_path}")
