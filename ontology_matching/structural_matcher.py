
"""
structural_matcher.py
---------------------
Stage 2: Structural refinement using Topological Lin-Similarity.

This stage identifies candidate matches for entities that remain unmapped 
after lexical (AML), semantic (MNLI), and structural (Child/Assoc) stages.

Logic
-----
1. Build an undirected topological graph for both ontologies (V1 and V2).
2. Treat all JSON relationships (nesting and associations) as topological edges.
3. Calculate Intrinsic Information Content (IC) for all nodes based on degree.
4. For every unmatched entity in Model A:
   a. Find its closest "anchors" (entities already matched in previous stages).
   b. Look at the neighbors of those anchors in Model B.
   c. Calculate Lin Similarity between the unmatched A-entity and the candidate B-neighbor.
   d. Assign matches based on the highest Lin-IC score.

Formula
-------
Lin Similarity = (2 * IC(LCA)) / (IC(Entity_A) + IC(Entity_B))
Where LCA is the shared matched anchor (the common context).

Usage
-----
    cd ontology_matching
    python structural_matcher.py
    python structural_matcher.py --domain Automobile
    python structural_matcher.py --threshold 0.6

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
STRUCT_DIR  = os.path.join(BASE_DIR, "outputs", "structural")

DEFAULT_THRESHOLD = 0.5

# Lin-IC penalty applied when the only common ancestor is the injected domain-root
# anchor.  A root-mediated match means "both entities are in the same domain",
# which is trivially true and carries no discriminative structural signal.
# At 0.2 a raw lin_sim of 1.0 becomes 0.2, which falls below DEFAULT_THRESHOLD,
# effectively filtering pure root-anchor matches unless corroborated elsewhere.
ROOT_ANCHOR_PENALTY = 0.20

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

def _calculate_ic(G: nx.Graph) -> dict[str, float]:
    """Calculate Intrinsic Information Content based on node degree."""
    ic_map = {}
    total_nodes = len(G.nodes)
    if total_nodes <= 1:
        return {n: 1.0 for n in G.nodes}
    
    max_deg = max(dict(G.degree()).values()) if total_nodes > 0 else 1
    for node, deg in G.degree():
        # Generality is proportional to degree; IC is 1 - generality
        ic_map[node] = 1 - (math.log(deg + 1) / math.log(max_deg + 2))
    return ic_map

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

    # Add stages S2, S4, S5, S6 (S3 is subsumption/groups, handled separately if needed)
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

    Strategy 2 — Highest-degree hub:
        Fallback: find the most-connected entity node in A that has a name-similar
        counterpart in B (Jaccard >= 0.5 on camelCase tokens).
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

    # Strategy 2: highest-degree entity hub with name similarity
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
    ic_a, ic_b = _calculate_ic(G_a), _calculate_ic(G_b)

    # Build association-vocabulary indexes for the fallback pass
    vocab_a = _build_assoc_vocab(json_a)
    vocab_b = _build_assoc_vocab(json_b)

    # All previous matches (S1+) are anchors
    anchors_ab = _get_anchors(report, domain, stem)
    matched_a = set(anchors_ab.keys())
    matched_b = set(anchors_ab.values())

    # Bootstrap: inject domain-root entity as synthetic anchor if not already matched
    root_anchors = _find_root_anchors(json_a, json_b, domain, G_a, G_b, matched_b)
    root_anchor_set: set[str] = set()   # track which anchors are root-only injections
    for ea, eb in root_anchors.items():
        if ea not in matched_a and eb not in matched_b:
            anchors_ab[ea] = eb
            matched_a.add(ea)
            matched_b.add(eb)
            root_anchor_set.add(ea)

    # Derive association-node anchors from the entity anchors now in hand.
    # These allow Lin-IC propagation to backtrack through association nodes
    # as common ancestors when searching for unmatched neighbour candidates.
    assoc_anchors = _derive_association_anchors(G_a, G_b, anchors_ab)
    anchors_ab.update(assoc_anchors)
    matched_a.update(assoc_anchors.keys())
    matched_b.update(assoc_anchors.values())

    # Candidates are entities that are NOT matched yet
    unmatched_a = [e["name"] for e in report["model_a"]["entities"] if e["name"] not in matched_a]
    unmatched_b = [e["name"] for e in report["model_b"]["entities"] if e["name"] not in matched_b]

    new_matches = []

    # Pass 1: Lin-IC propagation from existing anchors
    for u_a in list(unmatched_a):
        if u_a not in G_a: continue

        candidates = []
        near_anchors_a = _get_near_anchors(u_a, G_a, matched_a)

        for anc_a, d_a in near_anchors_a:
            anc_b = anchors_ab[anc_a]
            if anc_b not in G_b: continue

            is_root_anchor = anc_a in root_anchor_set

            for u_b in G_b.neighbors(anc_b):
                if u_b in unmatched_b:
                    ic_lca = (ic_a[anc_a] + ic_b[anc_b]) / 2
                    denom = ic_a[u_a] + ic_b[u_b]
                    lin_sim_raw = (2 * ic_lca / denom) if denom > 0 else 0

                    # Penalize matches whose only common ancestor is the injected
                    # domain-root anchor: "both are in the same domain" is not a
                    # structural signal — discount heavily so genuine matches survive.
                    if is_root_anchor:
                        lin_sim = lin_sim_raw * ROOT_ANCHOR_PENALTY
                    else:
                        lin_sim = lin_sim_raw

                    dist_weight = 1.0 / (d_a + 1)
                    score = lin_sim * dist_weight

                    if score >= threshold:
                        candidates.append({
                            "smaller_entity": u_a,
                            "larger_entity": u_b,
                            "struct_score": round(score, 4),
                            "lin_sim": round(lin_sim, 4),
                            "lin_sim_raw": round(lin_sim_raw, 4),
                            "lca_is_root_anchor": is_root_anchor,
                            "anchor": anc_a,
                            "method": "lin_ic"
                        })

        if candidates:
            best = max(candidates, key=lambda x: x["struct_score"])
            new_matches.append(best)
            if best["larger_entity"] in unmatched_b:
                unmatched_b.remove(best["larger_entity"])
            unmatched_a.remove(u_a)

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
