
"""
structural_matcher.py
---------------------
Stage 2: Structural refinement using semantic-first pairing with topology
         validation.

This stage identifies candidate matches for entities that remain unmapped
after lexical (AML), semantic (MNLI), and structural (Child/Assoc) stages.

Logic
-----
1. Build an undirected topological graph for both ontologies.
2. Treat all JSON relationships (nesting and associations) as topological edges.
3. Pass 1 — Semantic-first with topology gate:
   a. Batch-encode ALL unmatched entities in A and B with SBERT.
   b. Compute full cosine similarity matrix (|unmatched_A| × |unmatched_B|).
   c. For every pair (u_a, u_b) with cos_sim ≥ threshold:
        - Topology gate: u_a must be within BFS distance 2 of some anchor
          anchor_a in G_a, AND u_b must be a direct neighbour of anchor_a's
          matched counterpart anchor_b in G_b.
        - WordNet gate (borderline zone threshold ≤ cos < WN_GATE_UPPER):
          compound_sim(u_a, u_b) ≥ WN_MIN_SCORE.
        - Score = cos_sim × dist_weight (× ROOT_ANCHOR_PENALTY if anchor is
          a synthetic root anchor).
   d. Global greedy assignment: sort all topology-valid candidates by score
      (descending), assign best non-conflicting pair for each entity.
4. Pass 2 (fallback): association-vocabulary Jaccard similarity for entities
   that still have no candidate after Pass 1.

Semantic-first benefits over previous topology-first design
-----------------------------------------------------------
  - Full cosine matrix computed once per pair (batch GPU inference).
  - Topology is a gate, not a generator — topology-adjacent pairs that are
    semantically wrong are suppressed; semantically strong pairs that happen
    to have topology support are always found.
  - Global greedy assignment eliminates sequential first-pick bias.

Scoring
-------
  score(u_a, u_b) = cos_sim × dist_weight [× ROOT_ANCHOR_PENALTY]

  - cos_sim      = cosine similarity of L2-normalised SBERT embeddings after
                   camelCase splitting ("BrakeRotor" → "brake rotor").
  - dist_weight  = 1 / (d_a + 1), where d_a = BFS distance from u_a to
                   the nearest anchor in G_a.
  - ROOT_ANCHOR_PENALTY applied only when the validating anchor is a
                   synthetic root anchor (domain-name match or WordNet root).

Embedding model
---------------
  paraphrase-MiniLM-L6-v2  (already used by synonym_matcher.py in this pipeline)

Usage
-----
    cd ontology_matching
    python structural_matcher.py
    python structural_matcher.py --domain Automobile
    python structural_matcher.py --threshold 0.45

NOTE: This file is Stage 2 in the 8-stage pipeline.  Internal JSON keys
(stage_matched etc.) intentionally retain their original naming; only the
stage label in metadata output has been updated.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import numpy as np
import torch
import networkx as nx
from typing import Any

# Paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR  = os.path.join(
    BASE_DIR, "inputs",
    "CONceptual_ExtractionCategory_Examples",
    "CONceptual_ExtractionCategory_Examples",
)
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
MNLI_DIR    = os.path.join(BASE_DIR, "outputs", "mnli")
SUB_DIR     = os.path.join(BASE_DIR, "outputs", "subsumption")
CHILD_DIR   = os.path.join(BASE_DIR, "outputs", "child")
ASSOC_DIR   = os.path.join(BASE_DIR, "outputs", "association")
SYN_DIR     = os.path.join(BASE_DIR, "outputs", "synonym")
WN_DIR      = os.path.join(BASE_DIR, "outputs", "wordnet")
STRUCT_DIR  = os.path.join(BASE_DIR, "outputs", "structural")

DEFAULT_THRESHOLD = 0.60   # minimum cos_sim for a cosine_struct match to be accepted
SBERT_MODEL       = "paraphrase-MiniLM-L6-v2"

# WordNet validation gate (two-tier, applied to ALL matches):
#
# Tier 1 (0.60 ≤ cos < WN_GATE_UPPER):
#   token-level compound_sim ≥ WN_MIN_SCORE (0.30) — catches false positives that
#   share embedding neighbourhood but are semantically different sub-concepts.
#
# Tier 2 (cos ≥ WN_GATE_UPPER):
#   full-entity synset path_similarity ≥ WN_FULL_MIN (0.15) — only blocks pairs
#   where both entities have WordNet synsets AND those synsets are very distant.
#   "SteeringWheel"↔"Wheel": path_sim(steering_wheel.n.01, wheel.n.01) = 0.143 → blocked.
#   "Automobile"↔"Vehicle":  path_sim(car.n.01, vehicle.n.01)          = 0.200 → passes.
#   When either entity has no compound synset the check is skipped (pass-through).
WN_GATE_UPPER  = 0.75   # cosine threshold separating the two tiers
WN_MIN_SCORE   = 0.30   # Tier 1 minimum: token-level compound_sim
WN_FULL_MIN    = 0.15   # Tier 2 minimum: full-entity synset path_similarity

from wordnet_reinforcer import compound_sim as _wn_compound_sim

def _wn_full_entity_sim(name_a: str, name_b: str) -> float | None:
    """Path similarity between two entity names treated as compound nouns.

    Converts camelCase to underscore_form ("SteeringWheel" → "steering_wheel"),
    looks up the first WordNet NOUN synset for each, and returns path_similarity.
    Returns None if either entity has no compound synset — callers should treat
    None as 'cannot determine; skip the check'.
    """
    from nltk.corpus import wordnet as wn
    def _to_wn_key(name: str) -> str:
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
        return re.sub(r"\s+", "_", s.strip()).lower()

    syns_a = wn.synsets(_to_wn_key(name_a), pos=wn.NOUN)
    syns_b = wn.synsets(_to_wn_key(name_b), pos=wn.NOUN)
    if not syns_a or not syns_b:
        return None
    sim = syns_a[0].path_similarity(syns_b[0])
    return float(sim) if sim is not None else None

# Root-anchor penalty: matches whose only structural prior is the injected
# domain-root anchor carry very weak structural evidence. Discount so genuine
# semantically similar matches still win, but root-only anchors are suppressed.
ROOT_ANCHOR_PENALTY = 0.60  # multiplied into the cosine score (not the old 0.20)


# --------------------------------------------------------------------------- #
# Sentence-transformer encoder (lazy singleton)                               #
# --------------------------------------------------------------------------- #

class _SBERTEncoder:
    """Lazy singleton wrapper around paraphrase-MiniLM-L6-v2."""
    _instance = None

    @classmethod
    def get(cls) -> "_SBERTEncoder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [S2] Loading '{SBERT_MODEL}' on {device} …")
        self._model = SentenceTransformer(SBERT_MODEL, device=device)

    def encode(self, names: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings for a list of entity names.
        Names are camelCase-split before encoding so 'BrakeRotor' → 'brake rotor'.
        """
        texts = [_camel_split(n) for n in names]
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def cosine(self, name_a: str, name_b: str) -> float:
        """Cosine similarity between two entity names (dot product after L2-norm)."""
        embs = self.encode([name_a, name_b])
        return float(np.dot(embs[0], embs[1]))


def _camel_split(name: str) -> str:
    """Split camelCase/PascalCase to lower-cased tokens: 'BrakeRotor' → 'brake rotor'."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return re.sub(r"[^A-Za-z0-9]+", " ", s).strip().lower()

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _safe(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip())
    return re.sub(r"_+", "_", s).strip("_") or "unknown"

def _tokenize(name: str) -> set[str]:
    """Split camelCase/PascalCase name into lowercase tokens."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return {t.lower() for t in re.split(r"[^A-Za-z0-9]+", s) if len(t) > 1}

def _build_assoc_vocab(json_data: dict) -> dict[str, set[str]]:
    """
    Build per-entity vocabulary from association names and partner names.
    Returns {entity_name: set_of_tokens}.
    Handles both JSON schemas (associationName/associationParticipants and name/participants).
    """
    vocab: dict[str, set[str]] = {}
    for assoc in json_data.get("associations", []):
        assoc_name   = assoc.get("associationName", "") or assoc.get("name", "")
        participants = assoc.get("associationParticipants", []) or assoc.get("participants", [])
        name_tokens  = _tokenize(assoc_name)
        for p in participants:
            if p not in vocab:
                vocab[p] = set()
            vocab[p].update(name_tokens)
            for other in participants:
                if other != p:
                    vocab[p].update(_tokenize(other))
    return vocab

def _assoc_token_sim(vocab_a: dict, vocab_b: dict, ea: str, eb: str) -> float:
    """Jaccard similarity between association-vocabulary sets of two entities."""
    ta = vocab_a.get(ea, set())
    tb = vocab_b.get(eb, set())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def _build_undirected_graph(json_data: dict) -> nx.Graph:
    """Build a topological graph from JSON data.

    Associations are treated as first-class nodes, creating a triadic structure:
        participant_A  --  assoc_node  --  participant_B

    This is structurally correct: the association is a concept in its own right,
    not merely a shortcut edge between two entities. Adding it as a node means:
      - Its degree reflects how many entities it relates.
      - Entities connected through associations see the association as a potential
        common ancestor in the graph, enabling Lin-IC backtracking through it.
      - IC values for entities change to reflect their true connectivity pattern.

    Node attribute 'node_type': 'entity' | 'association'
    """
    G = nx.Graph()
    entities = {e.get("entityName") or e.get("name") for e in json_data.get("entities", [])}

    for e in entities:
        G.add_node(e, node_type="entity")

    # 1. Compositions (nesting) — entity-to-entity edges via attribute types
    for ent in json_data.get("entities", []):
        parent = ent.get("entityName") or ent.get("name")
        for attr in (ent.get("entityAttributes") or ent.get("attributes", [])):
            child = attr.get("type")
            if child in entities:
                G.add_edge(parent, child)

    # 2. Associations — added as first-class intermediate nodes
    for assoc in json_data.get("associations", []):
        assoc_name   = assoc.get("associationName", "") or assoc.get("name", "")
        participants = assoc.get("associationParticipants", []) or assoc.get("participants", [])
        known = [p for p in participants if p in entities]
        if not assoc_name or not known:
            continue
        G.add_node(assoc_name, node_type="association")
        for p in known:
            G.add_edge(p, assoc_name)

    return G


def _get_anchors(report: dict, domain: str, stem: str) -> dict[str, str]:
    """
    Collect all matched entities from ALL previous stages (S1-S6).
    Returns {model_a_name: model_b_name}
    """
    anchors = {}
    
    # S1: AML (from the report)
    for ent in report["model_a"]["entities"]:
        if ent["status"] == "matched":
            target = ent.get("matched_to") or ent.get("match")
            if target:
                anchors[ent["name"]] = target

    # Helper to load stage files
    def _add_stage(directory, suffix, s_key, l_key):
        p = os.path.join(directory, domain, f"{stem}_{suffix}.json")
        if os.path.exists(p):
            for m in _load(p).get("new_matches", []):
                anchors[m[s_key]] = m[l_key]

    # S1b WordNet reinforcement — must be loaded before MNLI/child/assoc/synonym
    # so that semantic root pairs (e.g. "Vehicle"↔"Automobile") are anchors for BFS
    _add_stage(WN_DIR, "wordnet", "smaller_entity", "larger_entity")
    _add_stage(MNLI_DIR, "mnli", "smaller_entity", "larger_entity")
    _add_stage(CHILD_DIR, "child", "smaller_entity", "larger_entity")
    _add_stage(ASSOC_DIR, "assoc", "smaller_entity", "larger_entity")
    _add_stage(SYN_DIR, "synonym", "smaller_entity", "larger_entity")
    
    return anchors

def _get_near_anchors(node: str, G: nx.Graph, anchor_set: set[str], max_dist: int = 2) -> list[tuple[str, int]]:
    """Find matched anchors in the neighborhood of a node."""
    if node not in G: return []
    found = []
    visited = {node}
    queue = [(node, 0)]
    while queue:
        curr, dist = queue.pop(0)
        if dist >= max_dist: continue
        for nb in G.neighbors(curr):
            if nb not in visited:
                visited.add(nb)
                if nb in anchor_set:
                    found.append((nb, dist + 1))
                queue.append((nb, dist + 1))
    return found

def _find_root_anchors(
    json_a: dict, json_b: dict, domain: str,
    G_a: nx.Graph, G_b: nx.Graph,
    existing_matched_b: set[str],
) -> dict[str, str]:
    """
    Bootstrap anchors from structural priors, independent of S1 lexical matches.

    Strategy 1 — Domain-root entity:
        The root concept of a domain (e.g. "Automobile" in the Automobile domain)
        almost always appears in every model for that domain and is the natural
        common ancestor of the entire ontology tree.  We inject it as a synthetic
        anchor so that Lin-IC propagation has a starting point even when S1 found
        nothing lexically.

    Strategy 2 — Highest-degree hub (Jaccard):
        Fallback: find the most-connected entity node in A that has a name-similar
        counterpart in B (Jaccard >= 0.5 on camelCase tokens).

    Strategy 3 — WordNet semantic root detection:
        For cross-type pairs (e.g. V-model vs Network model) where lexical names
        diverge ("Vehicle" vs "Automobile", "Chassis" vs "Automobile"), Strategies
        1 and 2 both fail.  Strategy 3 scores the top-3 highest-degree nodes in A
        against all B nodes using WordNet path_similarity on constituent tokens.
        The best pair above 0.35 is injected as the root anchor.
        This ensures BFS propagation always has at least one starting point even
        when root entities are named differently across model types.
    """
    domain_tokens = _tokenize(domain)
    entity_nodes_a = {n for n, d in G_a.nodes(data=True) if d.get("node_type") == "entity"}
    entity_nodes_b = {n for n, d in G_b.nodes(data=True) if d.get("node_type") == "entity"}
    anchors: dict[str, str] = {}

    # Strategy 1: entities whose name tokens overlap with the domain name
    cands_a = [e for e in entity_nodes_a if _tokenize(e) & domain_tokens]
    cands_b = [e for e in entity_nodes_b if _tokenize(e) & domain_tokens]
    for ea in cands_a:
        ta = _tokenize(ea)
        for eb in cands_b:
            if eb in existing_matched_b:
                continue
            tb = _tokenize(eb)
            if ta and tb and len(ta & tb) / len(ta | tb) >= 0.5:
                anchors[ea] = eb

    if anchors:
        return anchors

    # Strategy 2: highest-degree entity hub with name similarity (Jaccard)
    if not entity_nodes_a or not entity_nodes_b:
        return anchors
    top_a = max(entity_nodes_a, key=lambda n: G_a.degree(n))
    ta = _tokenize(top_a)
    best_sim, best_b = 0.0, None
    for eb in entity_nodes_b:
        if eb in existing_matched_b:
            continue
        tb = _tokenize(eb)
        if ta and tb:
            sim = len(ta & tb) / len(ta | tb)
            if sim > best_sim:
                best_sim, best_b = sim, eb
    if best_b and best_sim >= 0.5:
        anchors[top_a] = best_b
        return anchors

    # Strategy 3: WordNet semantic similarity on top-3 highest-degree nodes.
    # Handles cases like "Vehicle"↔"Automobile" or "Chassis"↔"Automobile" where
    # token Jaccard is 0 but the concepts are semantically close/equivalent.
    top3_a = sorted(entity_nodes_a, key=lambda n: G_a.degree(n), reverse=True)[:3]
    best_wn, best_ea, best_eb = 0.0, None, None
    for ea in top3_a:
        for eb in entity_nodes_b:
            if eb in existing_matched_b:
                continue
            wn_score, _, _ = _wn_compound_sim(ea, eb)
            if wn_score > best_wn:
                best_wn, best_ea, best_eb = wn_score, ea, eb
    if best_eb and best_wn >= 0.35:
        anchors[best_ea] = best_eb

    return anchors


def _derive_association_anchors(
    G_a: nx.Graph, G_b: nx.Graph, anchors_ab: dict[str, str]
) -> dict[str, str]:
    """
    Given a set of matched entity anchors, infer matched association nodes.

    An association node in A is matched to an association node in B when the
    entity participants of A's association map (via anchors_ab) to the entity
    participants of B's association — i.e., both endpoints are already matched.

    These association anchors are added to anchors_ab so that the Lin-IC
    propagation in Pass 1 can backtrack through the association node as a
    common ancestor when looking for unmatched neighbour candidates.
    """
    assoc_nodes_a = {n for n, d in G_a.nodes(data=True) if d.get("node_type") == "association"}
    assoc_nodes_b = {n for n, d in G_b.nodes(data=True) if d.get("node_type") == "association"}
    assoc_anchors: dict[str, str] = {}

    for aa in assoc_nodes_a:
        # Entity participants of aa in G_a
        parts_a = {nb for nb in G_a.neighbors(aa)
                   if G_a.nodes[nb].get("node_type") == "entity"}
        # Map them to B via existing entity anchors
        mapped_b = {anchors_ab[p] for p in parts_a if p in anchors_ab}
        if not mapped_b:
            continue
        # Find the association in B whose entity participants best overlap with mapped_b
        best_sim, best_ab = 0.0, None
        for ab in assoc_nodes_b:
            if ab in assoc_anchors.values():
                continue
            parts_b = {nb for nb in G_b.neighbors(ab)
                       if G_b.nodes[nb].get("node_type") == "entity"}
            if not parts_b:
                continue
            sim = len(mapped_b & parts_b) / len(mapped_b | parts_b)
            if sim > best_sim:
                best_sim, best_ab = sim, ab
        if best_ab and best_sim >= 0.5:
            assoc_anchors[aa] = best_ab

    return assoc_anchors


# --------------------------------------------------------------------------- #
# Main Logic                                                                   #
# --------------------------------------------------------------------------- #

def run_pair(domain: str, report_path: str, threshold: float) -> dict | None:
    report = _load(report_path)
    meta   = report["metadata"]
    stem   = os.path.splitext(os.path.basename(report_path))[0]
    
    # Load original JSONs to build graphs
    path_a = os.path.join(BASE_DIR, meta["json_a"])
    path_b = os.path.join(BASE_DIR, meta["json_b"])
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return None
        
    json_a, json_b = _load(path_a), _load(path_b)
    G_a, G_b = _build_undirected_graph(json_a), _build_undirected_graph(json_b)

    # Build association-vocabulary indexes for the fallback pass
    vocab_a = _build_assoc_vocab(json_a)
    vocab_b = _build_assoc_vocab(json_b)

    # All previous matches (S1+) are anchors
    anchors_ab = _get_anchors(report, domain, stem)
    matched_a = set(anchors_ab.keys())
    matched_b = set(anchors_ab.values())

    # Bootstrap: inject domain-root entity as synthetic anchor if not already matched
    root_anchors = _find_root_anchors(json_a, json_b, domain, G_a, G_b, matched_b)
    root_anchor_set: set[str] = set()
    for ea, eb in root_anchors.items():
        if ea not in matched_a and eb not in matched_b:
            anchors_ab[ea] = eb
            matched_a.add(ea)
            matched_b.add(eb)
            root_anchor_set.add(ea)

    # Derive association-node anchors from the entity anchors now in hand.
    assoc_anchors = _derive_association_anchors(G_a, G_b, anchors_ab)
    anchors_ab.update(assoc_anchors)
    matched_a.update(assoc_anchors.keys())
    matched_b.update(assoc_anchors.values())

    # Candidates are entities that are NOT matched yet
    unmatched_a = [e["name"] for e in report["model_a"]["entities"] if e["name"] not in matched_a]
    unmatched_b = [e["name"] for e in report["model_b"]["entities"] if e["name"] not in matched_b]

    # Lazy-load the sentence encoder (shared across all pairs in a run)
    encoder = _SBERTEncoder.get()

    new_matches = []

    # ---------------------------------------------------------------------- #
    # Pass 1: Semantic-first with topology gate.                              #
    #                                                                         #
    # Step A — Batch encode all unmatched entities in one GPU call each.      #
    # Step B — Compute full cosine matrix.                                    #
    # Step C — For each pair above the cosine threshold, check topology:      #
    #           u_a must be within BFS-2 of an anchor in G_a whose            #
    #           counterpart in G_b is within BFS-2 of u_b.                   #
    # Step D — Apply WordNet gate in the borderline cosine zone.              #
    # Step E — Global greedy assignment (sort by score, no first-pick bias).  #
    # ---------------------------------------------------------------------- #

    unmatched_b_set = set(unmatched_b)
    a_in_graph = [u for u in unmatched_a if u in G_a]

    if a_in_graph and unmatched_b:
        # Step A: batch encode
        embs_a = encoder.encode(a_in_graph)           # shape (|A|, dim)
        embs_b = encoder.encode(unmatched_b)           # shape (|B|, dim)
        cos_matrix = embs_a @ embs_b.T                 # shape (|A|, |B|)

        # Pre-compute per-anchor: which unmatched B nodes are within BFS depth 2?
        # Depth 2 (vs depth 1) extends reach through intermediate nodes such as
        # association nodes or matched intermediate entities, catching pairs like
        # Piston↔PistonAssembly that are two hops from the nearest anchor.
        anc_b_nbrs: dict[str, set[str]] = {}
        for anc_b in set(anchors_ab.values()):
            if anc_b not in G_b:
                continue
            reachable: set[str] = set()
            visited = {anc_b}
            queue: list[tuple[str, int]] = [(anc_b, 0)]
            while queue:
                curr, d = queue.pop(0)
                if d >= 2:
                    continue
                for nb in G_b.neighbors(curr):
                    if nb not in visited:
                        visited.add(nb)
                        if nb in unmatched_b_set:
                            reachable.add(nb)
                        queue.append((nb, d + 1))
            anc_b_nbrs[anc_b] = reachable

        # Step B/C/D: collect all topology-valid candidates
        all_candidates: list[dict] = []
        for i, u_a in enumerate(a_in_graph):
            near_a = _get_near_anchors(u_a, G_a, matched_a)
            if not near_a:
                continue

            for j, u_b in enumerate(unmatched_b):
                cos_sim = float(cos_matrix[i, j])
                if cos_sim < threshold:
                    continue

                # Topology gate: find the best (highest dist_weight) anchor that
                # connects u_a → anc_a → anc_b → u_b
                best_dw, best_is_root = 0.0, False
                for anc_a, d_a in near_a:
                    anc_b = anchors_ab.get(anc_a)
                    if anc_b and u_b in anc_b_nbrs.get(anc_b, set()):
                        dw = 1.0 / (d_a + 1)
                        if dw > best_dw:
                            best_dw = dw
                            best_is_root = anc_a in root_anchor_set
                if best_dw == 0.0:
                    continue  # no topology support — discard

                # Step D: Two-tier WordNet gate (applied to ALL candidates).
                #
                # Tier 1 (borderline cosine): token-level compound_sim catches pairs
                # like "Flywheel"↔"Wheel" that share embedding space but differ semantically.
                #
                # Tier 2 (high cosine): full-entity synset path_similarity catches pairs
                # like "SteeringWheel"↔"Wheel" (cos=0.82) where a sub-token match inflates
                # compound_sim but the compound synsets are genuinely distant (0.143 < 0.15).
                # Pairs with no compound synset (e.g. "AxleAssembly") skip Tier 2.
                if cos_sim < WN_GATE_UPPER:
                    wn_score, _, _ = _wn_compound_sim(u_a, u_b)
                    if wn_score < WN_MIN_SCORE:
                        continue
                else:
                    full_sim = _wn_full_entity_sim(u_a, u_b)
                    if full_sim is not None and full_sim < WN_FULL_MIN:
                        continue

                score = cos_sim * best_dw
                if best_is_root:
                    score *= ROOT_ANCHOR_PENALTY

                all_candidates.append({
                    "smaller_entity":     u_a,
                    "larger_entity":      u_b,
                    "struct_score":       round(score, 4),
                    "cos_sim":            round(cos_sim, 4),
                    "lca_is_root_anchor": best_is_root,
                    "method":             "cosine_struct",
                })

        # Step E: global greedy assignment — best score wins, no conflicts
        all_candidates.sort(key=lambda x: x["struct_score"], reverse=True)
        assigned_a: set[str] = set()
        assigned_b: set[str] = set()
        for cand in all_candidates:
            if cand["smaller_entity"] in assigned_a or cand["larger_entity"] in assigned_b:
                continue
            assigned_a.add(cand["smaller_entity"])
            assigned_b.add(cand["larger_entity"])
            new_matches.append(cand)

        # Update unmatched lists for Pass 2
        unmatched_a = [u for u in unmatched_a if u not in assigned_a]
        unmatched_b = [u for u in unmatched_b if u not in assigned_b]

    # Pass 2: Association-vocabulary fallback for entities with no Lin-IC candidates.
    # Applies to all pairs but is especially important for network ontologies where
    # S1 found no lexical anchors, leaving Pass 1 with nothing to propagate from.
    ASSOC_THRESHOLD = max(threshold * 0.8, 0.25)
    for u_a in unmatched_a:
        if not vocab_a.get(u_a):
            continue  # entity has no associations — nothing to compare

        best_score, best_b = 0.0, None
        for u_b in unmatched_b:
            sim = _assoc_token_sim(vocab_a, vocab_b, u_a, u_b)
            if sim > best_score:
                best_score, best_b = sim, u_b

        if best_b and best_score >= ASSOC_THRESHOLD:
            new_matches.append({
                "smaller_entity": u_a,
                "larger_entity": best_b,
                "struct_score": round(best_score, 4),
                "lin_sim": 0.0,
                "anchor": "assoc_vocab",
                "method": "assoc_vocab"
            })
            unmatched_b.remove(best_b)

    if not new_matches:
        return None
        
    return {
        "metadata": {
            "domain": domain,
            "pair": stem,
            "threshold": threshold,
            "stage": "S2_Structural"
        },
        "new_matches": new_matches
    }

def main():
    parser = argparse.ArgumentParser(description="Stage 7: Structural Refinement")
    parser.add_argument("--domain", type=str, help="Filter by domain")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    pattern = os.path.join(REPORTS_DIR, args.domain if args.domain else "*", "*.json")
    report_paths = sorted(glob.glob(pattern))
    
    total_new = 0
    for rp in report_paths:
        domain = os.path.basename(os.path.dirname(rp))
        stem = os.path.splitext(os.path.basename(rp))[0]
        
        result = run_pair(domain, rp, args.threshold)
        if result:
            out_dir = os.path.join(STRUCT_DIR, domain)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{stem}_structural.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            
            n = len(result["new_matches"])
            print(f"  {domain:<12} | {stem[:40]:<40} | +{n} matches")
            total_new += n

    print(f"\n=== Done: {total_new} new structural matches found. ===")

if __name__ == "__main__":
    main()
