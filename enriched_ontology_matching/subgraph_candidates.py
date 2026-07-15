"""
subgraph_candidates.py
-----------------------
Step 1 of conceptual-encapsulation (1:n / "complex match") discovery:
topology-agnostic candidate subgroup generation within ONE model.

The underlying idea (see conversation): two models can describe the same
concept at different resolutions, one entity in model A may correspond not
to a single entity in model B, but to a *group* of entities in B. This is
what the ontology-matching literature calls complex matching / 1:n
correspondence, and no metric in this pipeline (lexical, WL, shape, attr,
entailment) currently proposes or scores groups, every one of them is
strictly pairwise (one entity vs one entity).

This module is deliberately scope-limited to candidate GENERATION only, no
scoring or classification of whether a candidate group is a genuine
encapsulation match happens here (see the conversation for that proposal:
reusing attribute_reach's embed+WUP kernel on the union of a group's reach
vectors, and the entailment_matcher.py cross-encoder for entity-name-level
"a system made of {group} entails {coarse entity}" testing).

Why community detection, not PartOf-tree walking
-------------------------------------------------
A conceptually coherent subgroup is not necessarily a composition subtree.
It could equally be a ring of Connects edges, a star of UsedFor edges, or
any other topology, the thing that makes a set of entities "belong
together" is that they're densely interconnected relative to the rest of
the model, not that they share a specific relation type. Restricting
candidate generation to PartOf edges would find deep-composition groups and
miss everything else. Louvain community detection over the FULL association
graph (every canonical relation type, undirected, edge-type-agnostic) finds
densely-interconnected clusters regardless of shape, a ring clusters
together exactly as readily as a composition chain, since the algorithm only
looks at edge density, never at what a relation is called.

Two independent, complementary strategies
-------------------------------------------
  community_candidates()  — Louvain community detection: a single
                             non-overlapping partition of the whole model.
  ego_candidates()         — bounded k-hop neighbourhood around every node:
                             overlapping, local candidates. A node's own
                             neighbourhood is still a legitimate candidate
                             group even when the global partition assigns it
                             elsewhere.

Usage (standalone)
-------------------
    from subgraph_candidates import candidate_subgroups

    result = candidate_subgroups(json_a)
    for group in result["communities"]:
        print(sorted(group))

CLI
---
    python enriched_ontology_matching/subgraph_candidates.py \\
        --json enriched_ontology_matching/inputs/Automobile/automobile_model_v1.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

from model_normalizer import load_inventory, normalize_model

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_association_graph(data: dict, canonical_map: dict[str, str] | None = None) -> nx.Graph:
    """
    Undirected graph of every entity in one model, with an edge for every
    association regardless of canonical relation type.

    Deliberately NOT restricted to PartOf/composition edges — a conceptual
    cluster can just as easily be held together by Connects, UsedFor, or any
    other relation. Filtering by relation type would bias candidate
    generation toward tree-shaped (composition) groups only.
    """
    inv  = canonical_map or load_inventory()
    norm = normalize_model(data, inv)

    names: set[str] = {
        e.get("entityName") or e.get("name", "")
        for e in norm.get("entities", [])
    }
    names.discard("")

    G = nx.Graph()
    G.add_nodes_from(names)

    for assoc in norm.get("associations", []):
        parts = assoc.get("associationParticipants") or assoc.get("participants") or []
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                u, v = parts[i], parts[j]
                if u in names and v in names and u != v:
                    G.add_edge(u, v)

    return G


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def community_candidates(
    G: nx.Graph,
    min_size: int = 2,
    max_frac: float = 0.6,
    seed: int = 0,
) -> list[frozenset[str]]:
    """
    Partition G into candidate conceptual subgroups via Louvain community
    detection over the full graph. Topology-agnostic: finds a densely
    interconnected ring cluster exactly as readily as a composition chain,
    since it only looks at edge density, never at relation labels.

    min_size drops singletons (nothing to encapsulate with just one entity).
    max_frac drops the degenerate whole-graph-as-one-group case, not a
    meaningful "subgroup" if it's most of the model.
    """
    n = G.number_of_nodes()
    if n < min_size or G.number_of_edges() == 0:
        return []
    communities = nx.algorithms.community.louvain_communities(G, seed=seed)
    return [
        frozenset(c) for c in communities
        if min_size <= len(c) <= max_frac * n
    ]


def ego_candidates(
    G: nx.Graph,
    k: int = 1,
    min_size: int = 2,
    max_frac: float = 0.6,
) -> list[frozenset[str]]:
    """
    Bounded k-hop neighbourhood around every node — an overlapping, local
    complement to community_candidates()'s single global partition.
    """
    n = G.number_of_nodes()
    if n < min_size:
        return []
    seen: set[frozenset[str]] = set()
    out: list[frozenset[str]] = []
    for node in G.nodes():
        ego = frozenset(nx.ego_graph(G, node, radius=k).nodes())
        if min_size <= len(ego) <= max_frac * n and ego not in seen:
            seen.add(ego)
            out.append(ego)
    return out


def candidate_subgroups(data: dict, k_ego: int = 1) -> dict:
    """
    Full step-1 candidate generation for one model.

    Returns the association graph (callers doing step-2 scoring will want it
    to compute per-group signatures) plus both strategies' candidate groups.
    """
    G = build_association_graph(data)
    return {
        "graph":        G,
        "communities":  community_candidates(G),
        "ego_networks": ego_candidates(G, k=k_ego),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Discover topology-agnostic candidate conceptual subgroups within one model."
    )
    ap.add_argument("--json", required=True, help="Model JSON (or a pair JSON with json_a/json_b)")
    ap.add_argument("--key", default=None, help="If a pair JSON, which side: json_a or json_b (default: json_a)")
    ap.add_argument("--ego-k", type=int, default=1, help="Hop radius for ego-network candidates (default: 1)")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    if "json_a" in data or "json_b" in data:
        data = data.get(args.key or "json_a", data)

    result = candidate_subgroups(data, k_ego=args.ego_k)
    G = result["graph"]
    print(f"Model: {data.get('modelName', args.json)}")
    print(f"Entities: {G.number_of_nodes()}   Associations (edges): {G.number_of_edges()}")

    print(f"\n=== Community candidates ({len(result['communities'])}) ===")
    for i, c in enumerate(sorted(result["communities"], key=len, reverse=True)):
        print(f"  [{i}] size={len(c):<3} {sorted(c)}")

    print(f"\n=== Ego-network candidates, k={args.ego_k} ({len(result['ego_networks'])}) ===")
    for i, c in enumerate(sorted(result["ego_networks"], key=len, reverse=True)):
        print(f"  [{i}] size={len(c):<3} {sorted(c)}")
