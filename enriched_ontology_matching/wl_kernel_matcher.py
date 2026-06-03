"""
wl_kernel_matcher.py
--------------------
Builds anonymized ontology graphs from merged metrics CSVs and computes
Weisfeiler-Lehman (WL) kernel similarity between them.

Anonymization scheme
---------------------
  Entity matches  → shared label "E{i}"  (i ranked by descending cosine_avg)
  Unmatched in A  → "A_{entity_name}"    (unique to graph A)
  Unmatched in B  → "B_{entity_name}"    (unique to graph B)
  Relations       → canonical type from association_inventory.csv (shared vocab)

Because matched entities start with the same label, WL refinement naturally
measures whether their K-hop neighbourhoods are structurally similar.
Unmatched entities never contribute to shared label counts, so they only
dilute the score — they never inflate it.

Match thresholds
-----------------
Derived from within-domain empirical analysis across 3 domains
(Automobile n=1797, Hospital n=2415, Homebrewing n=1838; fresh 2026-06-01).

The unmatched pool contains subsumption candidates (Room→ICURoom, etc.)
with cosine 0.60–0.80.  Thresholds below 0.85 declare these as equivalences,
corrupting the WL graph comparison.

  Primary  : cosine_avg >= 0.90          recall=87.6%  false_pass=0.9%
  Fallback : cosine_avg >= 0.80
             AND wup    >= 0.90          combined recall=88.3%  false_pass=2.2%

Usage
-----
  # Standalone
  python enriched_ontology_matching/wl_kernel_matcher.py \\
      --pair   enriched_ontology_matching/pairs/auto_V1_V2.json \\
      --metrics enriched_ontology_matching/outputs/merged/Automobile_Model_V1_SystemCentric_vs_Automobile_Model_V2_ComponentCentric_metrics.csv

  # Pipeline integration (called from run_all_pairs.py)
  from wl_kernel_matcher import run_wl_stage
  result = run_wl_stage(data_a, data_b, merged_csv, wl_csv)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR        = Path(__file__).resolve().parent
_REPO_ROOT  = _DIR.parent
_INVENTORY  = _DIR / "association_inventory.csv"

# ── Thresholds ────────────────────────────────────────────────────────────────
# Match declaration uses a 4-dimension composite score mirroring compare_stage.py,
# applied at the entity-pair level.  This ensures ALL available signals are used:
#
#   dim_lexical   = avg(cosine_avg, wup)                  — always present
#   dim_coherence = coherence_sym                         — when available
#   dim_graph     = avg(verb_coherence, gnn_sim)          — when available
#   dim_transfer  = avg(attr_reach_sim, entailment_f1)    — when available
#   composite     = mean of available dimensions
#
# Match sources (priority order):
#   1. matched=1  (AML + LogMap)     — always kept regardless of composite
#   2. composite >= COMPOSITE_THRESHOLD  — supplements missed equivalences
#
COMPOSITE_THRESHOLD = 0.65   # generous supplement on top of matched=1
WL_HOPS             = 3


# ── Inventory ─────────────────────────────────────────────────────────────────

def _load_canonical_map(path: Path = _INVENTORY) -> dict[str, str]:
    """Return {assoc_name: canonical} from association_inventory.csv."""
    result: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            result[row["association_name"]] = row["canonical"]
    return result


# ── Match declaration ─────────────────────────────────────────────────────────

def _row_composite(row: dict) -> float | None:
    """
    Compute 4-dimension composite score for one entity-pair row, mirroring
    compare_stage.py so that ALL available signals contribute.

      dim_lexical   = avg(cosine_avg, wup)
      dim_coherence = coherence_sym
      dim_graph     = avg(verb_coherence, gnn_sim)
      dim_transfer  = avg(attr_reach_sim, entailment_f1)
      composite     = mean of available dimensions
    """
    def _sf(v) -> float | None:
        try:
            f = float(v)
            return f if f > 0 else None
        except (TypeError, ValueError):
            return None

    cos = _sf(row.get("cosine_avg"))
    wup = _sf(row.get("wup"))
    coh = _sf(row.get("coherence_sym"))
    vc  = _sf(row.get("verb_coherence"))
    gnn = _sf(row.get("gnn_sim"))
    ar  = _sf(row.get("attr_reach_sim"))
    ef  = _sf(row.get("entailment_f1"))

    dims: list[float] = []

    # dim 1: lexical — always use raw values (include 0)
    def _sfz(v) -> float | None:
        try: return float(v)
        except: return None
    cos0, wup0 = _sfz(row.get("cosine_avg")), _sfz(row.get("wup"))
    lex_parts = [v for v in [cos0, wup0] if v is not None]
    if lex_parts:
        dims.append(sum(lex_parts) / len(lex_parts))

    # dim 2: neighbourhood coherence
    if coh is not None:
        dims.append(coh)

    # dim 3: graph structure
    graph_parts = [v for v in [vc, gnn] if v is not None]
    if graph_parts:
        dims.append(sum(graph_parts) / len(graph_parts))

    # dim 4: transfer / attribute reach
    transfer_parts = [v for v in [ar, ef] if v is not None]
    if transfer_parts:
        dims.append(sum(transfer_parts) / len(transfer_parts))

    return sum(dims) / len(dims) if dims else None


def declare_entity_matches(
    metrics_csv:          Path,
    composite_threshold:  float = COMPOSITE_THRESHOLD,
    data_a:               dict | None = None,
    data_b:               dict | None = None,
) -> list[dict]:
    """
    Read a merged metrics CSV and return declared entity matches.

    Match sources (priority order for 1-to-1 resolution):
      1. matched=1  (AML + LogMap confirmed equivalence) — always included.
      2. composite >= composite_threshold  — catches missed equivalences.

    Spurious parent-name matches are filtered out: if one entity name contains
    the other's tokens AND they have a PartOf parent-child relationship within
    either model, the match is suppressed regardless of matcher confidence.

    Returns list sorted by (matched=1 first, then composite desc).
    Enforces 1-to-1 constraint.
    """
    from enriched_matcher import (build_partof_parents, is_parent_name_embedded,
                                  is_same_entity_attribute_match)
    from model_normalizer import load_inventory, normalize_model

    parents_a: dict[str, str] = {}
    parents_b: dict[str, str] = {}
    if data_a is not None and data_b is not None:
        inv = load_inventory()
        parents_a = build_partof_parents(normalize_model(data_a, inv))
        parents_b = build_partof_parents(normalize_model(data_b, inv))

    candidates: list[dict] = []
    with open(metrics_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            is_matched = row.get("matched") == "1"
            comp       = _row_composite(row)
            is_above   = comp is not None and comp >= composite_threshold

            if is_matched or is_above:
                ea, eb = row["entity_a"], row["entity_b"]
                if (is_same_entity_attribute_match(ea, eb)
                        or is_parent_name_embedded(ea, eb, parents_a, parents_b)):
                    continue
                def _sfz(v):
                    try: return float(v)
                    except: return None
                candidates.append({
                    "entity_a":   ea,
                    "entity_b":   eb,
                    "cosine_avg": _sfz(row.get("cosine_avg")) or 0.0,
                    "wup":        _sfz(row.get("wup")) or 0.0,
                    "composite":  comp or 0.0,
                    "rule":       "matched" if is_matched else "composite",
                    "_sort_key":  (1 if is_matched else 0, comp or 0.0),
                })

    candidates.sort(key=lambda x: (-x["_sort_key"][0], -x["_sort_key"][1]))
    for c in candidates:
        del c["_sort_key"]

    used_a: set[str] = set()
    used_b: set[str] = set()
    matches: list[dict] = []
    for c in candidates:
        ea, eb = c["entity_a"], c["entity_b"]
        if ea not in used_a and eb not in used_b:
            matches.append(c)
            used_a.add(ea)
            used_b.add(eb)

    return matches


# ── Label maps ────────────────────────────────────────────────────────────────

def _build_label_maps(
    matches:    list[dict],
    entities_a: list[str],
    entities_b: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Assign anonymized node labels.

      Matched pair  → shared label "E{i}"   (i = rank in descending confidence)
      Unmatched A   → "A_{name}"
      Unmatched B   → "B_{name}"

    Returns (label_map_a, label_map_b).
    """
    label_a: dict[str, str] = {}
    label_b: dict[str, str] = {}

    for i, m in enumerate(matches):
        shared = f"E{i}"
        label_a[m["entity_a"]] = shared
        label_b[m["entity_b"]] = shared

    for e in entities_a:
        if e not in label_a:
            label_a[e] = f"A_{e}"
    for e in entities_b:
        if e not in label_b:
            label_b[e] = f"B_{e}"

    return label_a, label_b


# ── Edge list builder ─────────────────────────────────────────────────────────

def _build_edge_list(
    model_json:   dict,
    label_map:    dict[str, str],
    canonical_map: dict[str, str],
) -> list[tuple[str, str, str]]:
    """
    Return [(node_label_i, node_label_j, canonical_relation), ...].

    Normalizes the model first (adds canonical annotation + PartOf synthesis
    for composition edges in V-models), then converts every association into
    undirected edges between participant labels.  Edges where either endpoint
    is absent from label_map are silently skipped.
    """
    sys.path.insert(0, str(_DIR))
    from model_normalizer import normalize_model
    model_json = normalize_model(model_json, canonical_map)

    edges: list[tuple[str, str, str]] = []
    for assoc in model_json.get("associations", []):
        canonical    = assoc.get("canonical", "RelatedTo")
        participants = (
            assoc.get("associationParticipants")
            or assoc.get("participants")
            or []
        )
        for a_idx in range(len(participants) - 1):
            for b_idx in range(a_idx + 1, len(participants)):
                ea, eb = participants[a_idx], participants[b_idx]
                if ea in label_map and eb in label_map:
                    li, lj = label_map[ea], label_map[eb]
                    if li != lj:
                        edges.append((li, lj, canonical))
    return edges


def _entity_names(model_json: dict, canonical_map: dict[str, str]) -> list[str]:
    """Extract entity names from a (normalized) model JSON."""
    sys.path.insert(0, str(_DIR))
    from model_normalizer import normalize_model
    model_json = normalize_model(model_json, canonical_map)
    names = []
    for e in model_json.get("entities", []):
        name = e.get("entityName") or e.get("name") or ""
        if name:
            names.append(name)
    return names


# ── WL kernel ─────────────────────────────────────────────────────────────────

def wl_kernel(
    edges_a: list[tuple[str, str, str]],
    nodes_a: list[str],
    edges_b: list[tuple[str, str, str]],
    nodes_b: list[str],
    K: int = WL_HOPS,
) -> float:
    """
    Edge-aware Weisfeiler-Lehman kernel for two labeled graphs.

    Each hop:
      h_v ← md5( h_v + sorted[(h_neighbour, edge_label) for each neighbour] )

    Label frequencies are accumulated across all K+1 snapshots (hop 0 … hop K).
    Similarity = cosine of the two frequency vectors.

    Range: [0, 1].  Returns 0.0 if either graph has no nodes.
    """
    def _h(s: str) -> str:
        return hashlib.md5(s.encode()).hexdigest()[:16]

    def _build_adj(edges: list, nodes: list) -> dict[str, list[tuple[str, str]]]:
        adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
        node_set = set(nodes)
        for u, v, el in edges:
            if u in node_set and v in node_set:
                adj[u].append((v, el))
                adj[v].append((u, el))
        return adj

    def _refine(
        labels: dict[str, str],
        adj:    dict[str, list[tuple[str, str]]],
    ) -> dict[str, str]:
        new_labels: dict[str, str] = {}
        for node, lab in labels.items():
            nbrs = sorted(
                (labels.get(nb, "?"), el) for nb, el in adj[node]
            )
            raw = lab + "|" + "|".join(f"{nl}:{el}" for nl, el in nbrs)
            new_labels[node] = _h(raw)
        return new_labels

    if not nodes_a or not nodes_b:
        return 0.0

    adj_a = _build_adj(edges_a, nodes_a)
    adj_b = _build_adj(edges_b, nodes_b)

    labels_a = {n: _h(n) for n in nodes_a}
    labels_b = {n: _h(n) for n in nodes_b}

    freq_a: Counter = Counter()
    freq_b: Counter = Counter()

    # Collect label frequencies at each snapshot (hop 0 through K)
    for _ in range(K + 1):
        for lab in labels_a.values():
            freq_a[lab] += 1
        for lab in labels_b.values():
            freq_b[lab] += 1
        if _ < K:
            labels_a = _refine(labels_a, adj_a)
            labels_b = _refine(labels_b, adj_b)

    all_keys = set(freq_a) | set(freq_b)
    dot    = sum(freq_a[k] * freq_b[k] for k in all_keys)
    norm_a = sum(v * v for v in freq_a.values()) ** 0.5
    norm_b = sum(v * v for v in freq_b.values()) ** 0.5

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return round(dot / (norm_a * norm_b), 4)


# ── Induced subgraph WL (consistency score) ──────────────────────────────────

def _wl_kernel_with_labels(
    edges_a:   list[tuple[str,str,str]],
    nodes_a:   list[str],
    init_a:    dict[str, str],
    edges_b:   list[tuple[str,str,str]],
    nodes_b:   list[str],
    init_b:    dict[str, str],
    K:         int,
) -> float:
    """
    WL kernel where node IDs (entity names) are kept separate from initial
    labels.  Adjacency uses entity names; initial WL label comes from init_a/b.
    This prevents anonymous 'N'-labelled nodes from being merged in adjacency.
    """
    def _h(s: str) -> str:
        return hashlib.md5(s.encode()).hexdigest()[:16]

    def _build_adj(edges: list, nodes: list) -> dict:
        adj: dict[str, list] = defaultdict(list)
        ns = set(nodes)
        for u, v, el in edges:
            if u in ns and v in ns:
                adj[u].append((v, el)); adj[v].append((u, el))
        return adj

    def _refine(labels: dict, adj: dict) -> dict:
        new: dict = {}
        for node, lab in labels.items():
            nbrs = sorted((labels.get(nb, "?"), el) for nb, el in adj[node])
            new[node] = _h(lab + "|" + "|".join(f"{nl}:{el}" for nl, el in nbrs))
        return new

    if not nodes_a or not nodes_b:
        return 0.0

    adj_a = _build_adj(edges_a, nodes_a)
    adj_b = _build_adj(edges_b, nodes_b)

    # Initialise from explicit label maps instead of hashing node ID
    labels_a = {n: _h(init_a.get(n, "N")) for n in nodes_a}
    labels_b = {n: _h(init_b.get(n, "N")) for n in nodes_b}

    freq_a: Counter = Counter()
    freq_b: Counter = Counter()

    for k in range(K + 1):
        for lab in labels_a.values(): freq_a[lab] += 1
        for lab in labels_b.values(): freq_b[lab] += 1
        if k < K:
            labels_a = _refine(labels_a, adj_a)
            labels_b = _refine(labels_b, adj_b)

    all_keys = set(freq_a) | set(freq_b)
    dot    = sum(freq_a[k] * freq_b[k] for k in all_keys)
    norm_a = sum(v * v for v in freq_a.values()) ** 0.5
    norm_b = sum(v * v for v in freq_b.values()) ** 0.5
    return round(dot / (norm_a * norm_b), 4) if norm_a > 1e-10 and norm_b > 1e-10 else 0.0


def _k_hop_neighbourhood(
    seed_nodes: set[str],
    adj:        dict[str, list[tuple[str, str]]],
    K:          int,
) -> set[str]:
    """Return all nodes reachable within K hops from any seed node."""
    frontier = set(seed_nodes)
    visited  = set(seed_nodes)
    for _ in range(K):
        next_frontier: set[str] = set()
        for node in frontier:
            for nb, _ in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)
        frontier = next_frontier
    return visited


def wl_consistency(
    data_a:       dict,
    data_b:       dict,
    matches:      list[dict],
    canonical_map: dict[str, str],
    K_subgraph:   int = 2,   # wider induced subgraph for more context
    K_wl:         int = 5,   # more hops: induced subgraph is smaller so we need depth
) -> dict:
    """
    Induced-subgraph WL — measures structural consistency of matched entities
    independently of how many unmatched entities exist.

    Steps:
      1. Identify matched entity pairs and their K_subgraph-hop neighbourhoods
         in each graph.
      2. Extract the induced subgraph on those nodes (only edges where both
         endpoints are in the neighbourhood).
      3. Run WL on the two induced subgraphs with matched E{i}/A_/B_ labels.

    Because unmatched bulk nodes are excluded, the score is not diluted by
    low match coverage.  Only meaningful when n_matches >= 2.

    Returns:
      wl_consistency   — WL on induced subgraphs [0,1]; NaN if < 2 matches
      match_coverage   — n_matches / min(n_nodes_a, n_nodes_b)
      induced_frac_a   — fraction of A nodes included in induced subgraph
      induced_frac_b   — fraction of B nodes included in induced subgraph
    """
    entities_a = _entity_names(data_a, canonical_map)
    entities_b = _entity_names(data_b, canonical_map)
    n_a, n_b   = len(entities_a), len(entities_b)
    n_min      = min(n_a, n_b) or 1
    coverage   = round(len(matches) / n_min, 4)

    # Coverage: fraction of graph nodes (not CSV rows) that have a match
    entity_set_a = set(entities_a); entity_set_b = set(entities_b)
    graph_matches = [m for m in matches if m["entity_a"] in entity_set_a and m["entity_b"] in entity_set_b]
    coverage = round(len(graph_matches) / n_min, 4)

    if len(graph_matches) < 2:
        return {
            "wl_consistency": float("nan"),
            "match_coverage": coverage,
            "induced_frac_a": 0.0,
            "induced_frac_b": 0.0,
        }

    label_a, label_b = _build_label_maps(graph_matches, entities_a, entities_b)

    raw_la = {e: e for e in entities_a}
    raw_lb = {e: e for e in entities_b}
    raw_edges_a = _build_edge_list(data_a, raw_la, canonical_map)
    raw_edges_b = _build_edge_list(data_b, raw_lb, canonical_map)

    def _build_adj(edges, nodes):
        adj: dict[str, list] = defaultdict(list)
        ns = set(nodes)
        for u, v, el in edges:
            if u in ns and v in ns:
                adj[u].append((v, el)); adj[v].append((u, el))
        return adj

    adj_a = _build_adj(raw_edges_a, entities_a)
    adj_b = _build_adj(raw_edges_b, entities_b)

    seeds_a = {m["entity_a"] for m in graph_matches}
    seeds_b = {m["entity_b"] for m in graph_matches}

    sub_a = _k_hop_neighbourhood(seeds_a, adj_a, K_subgraph)
    sub_b = _k_hop_neighbourhood(seeds_b, adj_b, K_subgraph)

    # Induce edges (both endpoints must be in subgraph)
    ind_edges_a = [(u, v, el) for u, v, el in raw_edges_a if u in sub_a and v in sub_a]
    ind_edges_b = [(u, v, el) for u, v, el in raw_edges_b if u in sub_b and v in sub_b]

    # Matched entities get shared E{i} as initial WL label.
    # Unmatched induced-subgraph nodes get "N" as initial label — we measure
    # structural consistency around matched entities, not the identity of their
    # specific unmatched neighbours.
    # IMPORTANT: entity names remain as node IDs for adjacency (so "N"-labelled
    # nodes are NOT merged into one supranode); only the initial hash differs.
    def _init_lbl_a(e):
        return label_a[e] if e in label_a and label_a[e].startswith("E") else "N"
    def _init_lbl_b(e):
        return label_b[e] if e in label_b and label_b[e].startswith("E") else "N"

    score = _wl_kernel_with_labels(
        ind_edges_a, list(sub_a), {e: _init_lbl_a(e) for e in sub_a},
        ind_edges_b, list(sub_b), {e: _init_lbl_b(e) for e in sub_b},
        K_wl,
    )

    return {
        "wl_consistency": score,
        "match_coverage": coverage,
        "induced_frac_a": round(len(sub_a) / n_a, 4) if n_a else 0.0,
        "induced_frac_b": round(len(sub_b) / n_b, 4) if n_b else 0.0,
    }


# ── Global shape similarity ───────────────────────────────────────────────────

def _degree_sequence(entities: list[str], edges: list[tuple[str,str,str]]) -> list[int]:
    """Return sorted (descending) degree sequence, counting unique undirected edges."""
    deg: dict[str, int] = {e: 0 for e in entities}
    seen: set[tuple] = set()
    for u, v, _ in edges:
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            if u in deg: deg[u] += 1
            if v in deg: deg[v] += 1
    return sorted(deg.values(), reverse=True)


def _laplacian_eigenvalues(entities: list[str], edges: list[tuple[str,str,str]]) -> np.ndarray:
    """
    Compute sorted eigenvalues of the normalized Laplacian for an undirected graph.
    Eigenvalues lie in [0, 2].  Returns zeros array if graph has < 2 nodes.
    """
    n = len(entities)
    if n < 2:
        return np.zeros(n)
    idx = {e: i for i, e in enumerate(entities)}
    A = np.zeros((n, n))
    seen: set[tuple] = set()
    for u, v, _ in edges:
        i, j = idx.get(u, -1), idx.get(v, -1)
        if i >= 0 and j >= 0 and i != j:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                A[i, j] = A[j, i] = 1.0
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    return np.sort(np.linalg.eigvalsh(L_norm))


def _vec_cosine(a: list | np.ndarray, b: list | np.ndarray) -> float:
    """Cosine similarity between two vectors (padded with zeros to equal length)."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    n = max(len(a), len(b))
    a = np.pad(a, (0, n - len(a))); b = np.pad(b, (0, n - len(b)))
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0.0


def _clustering_coefficients(
    entities: list[str], edges: list[tuple[str,str,str]]
) -> list[float]:
    """
    Sorted (descending) local clustering coefficients for all nodes.

    C(v) = (edges between v's neighbours) / (d*(d-1)/2)

    Nodes with degree < 2 get C=0. High values indicate triangles — common in
    Net-models where components interconnect; low in tree-like V-model hierarchies.
    """
    adj: dict[str, set] = defaultdict(set)
    seen: set[tuple] = set()
    for u, v, _ in edges:
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            adj[u].add(v); adj[v].add(u)

    coeffs = []
    for e in entities:
        nbrs = adj[e]
        d = len(nbrs)
        if d < 2:
            coeffs.append(0.0)
            continue
        triangle_edges = sum(
            1 for n1 in nbrs for n2 in nbrs
            if n1 < n2 and n2 in adj[n1]
        )
        coeffs.append(triangle_edges / (d * (d - 1) / 2))
    return sorted(coeffs, reverse=True)


def _betweenness_centrality(
    entities: list[str], edges: list[tuple[str,str,str]]
) -> list[float]:
    """
    Sorted (descending) normalised betweenness centrality for all nodes.

    Uses Brandes' O(VE) algorithm on the undirected graph.
    Normalised by (n-1)(n-2) so values lie in [0, 1].

    High betweenness = bottleneck / bridge node (common for root entities in
    V-model hierarchies and for connector entities in Net-models).
    """
    adj: dict[str, list] = defaultdict(list)
    seen: set[tuple] = set()
    for u, v, _ in edges:
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            adj[u].append(v); adj[v].append(u)

    betweenness: dict[str, float] = {e: 0.0 for e in entities}

    for s in entities:
        stack: list[str] = []
        pred: dict[str, list] = {e: [] for e in entities}
        sigma: dict[str, float] = {e: 0.0 for e in entities}
        dist: dict[str, int] = {e: -1 for e in entities}
        sigma[s] = 1.0; dist[s] = 0
        queue: deque = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    queue.append(w); dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]; pred[w].append(v)

        delta: dict[str, float] = {e: 0.0 for e in entities}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    n = len(entities)
    if n > 2:
        scale = 1.0 / ((n - 1) * (n - 2))
        for e in entities:
            betweenness[e] *= scale

    return sorted(betweenness.values(), reverse=True)


def graph_shape_sim(
    data_a: dict,
    data_b: dict,
    canonical_map: dict[str, str],
) -> dict:
    """
    Global topology similarity — degree, spectral, clustering, betweenness.

    Returns:
      degree_sim      — cosine of sorted degree sequences (hub/leaf distribution)
      spectral_sim    — cosine of sorted Laplacian eigenvalues (community structure)
      clustering_sim  — cosine of sorted clustering coefficients (triangle density)
      betweenness_sim — cosine of sorted betweenness centralities (bottleneck structure)
      shape_sim       — avg of all four
    """
    ea = _entity_names(data_a, canonical_map)
    eb = _entity_names(data_b, canonical_map)

    edges_a = _build_edge_list(data_a, {e: e for e in ea}, canonical_map)
    edges_b = _build_edge_list(data_b, {e: e for e in eb}, canonical_map)

    deg_sim  = _vec_cosine(_degree_sequence(ea, edges_a),
                           _degree_sequence(eb, edges_b))
    spec_sim = _vec_cosine(_laplacian_eigenvalues(ea, edges_a),
                           _laplacian_eigenvalues(eb, edges_b))
    clust_sim = _vec_cosine(_clustering_coefficients(ea, edges_a),
                            _clustering_coefficients(eb, edges_b))
    bet_sim   = _vec_cosine(_betweenness_centrality(ea, edges_a),
                            _betweenness_centrality(eb, edges_b))

    shape = round((deg_sim + spec_sim + clust_sim + bet_sim) / 4, 4)
    return {
        "degree_sim":      round(deg_sim,   4),
        "spectral_sim":    round(spec_sim,  4),
        "clustering_sim":  round(clust_sim, 4),
        "betweenness_sim": round(bet_sim,   4),
        "shape_sim":       shape,
    }


# (attr_dist_sim lives in attribute_reach.py → run_attr_dist_stage)


# ── Pipeline stage ────────────────────────────────────────────────────────────

def run_wl_stage(
    data_a:               dict,
    data_b:               dict,
    metrics_csv:          Path,
    out_csv:              Path,
    K:                    int   = WL_HOPS,
    composite_threshold:  float = COMPOSITE_THRESHOLD,
    inventory_path:       Path  = _INVENTORY,
) -> dict:
    """
    Full WL kernel stage for one ontology pair.  Returns three scores:

    wl_structural  — all nodes anonymous ("N"), edges keep canonical type.
                     Measures pure relational topology: does this graph have
                     the same PartOf/UsedFor/etc. architecture, regardless of
                     which entities fill those roles?

    wl_matched     — nodes labeled E{i} for matched pairs, A_/B_ for unmatched.
                     Measures structural consistency around declared matches:
                     do matched entities sit in the same relational position?

    wl_composite   — avg(wl_structural, wl_matched).  Combines both signals.

    Steps:
      1. Build raw edge lists from both models (needed for both scores).
      2. Declare entity matches from metrics_csv (for wl_matched only).
      3. Compute wl_structural with uniform "N" node labels.
      4. Compute wl_matched with anonymized E{i}/A_/B_ node labels.
      5. Write single-row result CSV.
    """
    canonical_map = _load_canonical_map(inventory_path)

    entities_a = _entity_names(data_a, canonical_map)
    entities_b = _entity_names(data_b, canonical_map)

    # ── structural score: all nodes anonymous, edges keep canonical type ──────
    # Use entity name as node ID for adjacency lookup, but hash all to same "N"
    anon_label_a = {e: e for e in entities_a}  # identity map for edge building
    anon_label_b = {e: e for e in entities_b}
    raw_edges_a  = _build_edge_list(data_a, anon_label_a, canonical_map)
    raw_edges_b  = _build_edge_list(data_b, anon_label_b, canonical_map)

    # Remap node IDs to "N" for the kernel (anonymous nodes, labeled edges)
    struct_nodes_a = ["N"] * len(entities_a)
    struct_nodes_b = ["N"] * len(entities_b)
    # Keep unique node IDs for adjacency but start with label "N"
    struct_edges_a = raw_edges_a   # edge endpoints still entity names
    struct_edges_b = raw_edges_b

    # Override: build adjacency from raw entity names, initialise labels as "N"
    def _wl_structural(edges, entity_list, K):
        """WL kernel where every node starts with label 'N'."""
        from collections import Counter, defaultdict
        import hashlib
        def _h(s): return hashlib.md5(s.encode()).hexdigest()[:16]
        adj = defaultdict(list)
        node_set = set(entity_list)
        for u, v, el in edges:
            if u in node_set and v in node_set:
                adj[u].append((v, el)); adj[v].append((u, el))
        labels = {n: _h("N") for n in entity_list}
        freq: Counter = Counter()
        for _ in range(K + 1):
            for lab in labels.values(): freq[lab] += 1
            if _ < K:
                new = {}
                for node, lab in labels.items():
                    nbrs = sorted((labels.get(nb, "?"), el) for nb, el in adj[node])
                    new[node] = _h(lab + "|" + "|".join(f"{nl}:{el}" for nl, el in nbrs))
                labels = new
        return freq

    freq_sa = _wl_structural(raw_edges_a, entities_a, K)
    freq_sb = _wl_structural(raw_edges_b, entities_b, K)
    all_s   = set(freq_sa) | set(freq_sb)
    dot_s   = sum(freq_sa[k] * freq_sb[k] for k in all_s)
    norm_sa = sum(v*v for v in freq_sa.values()) ** 0.5
    norm_sb = sum(v*v for v in freq_sb.values()) ** 0.5
    wl_structural = round(dot_s / (norm_sa * norm_sb), 4) if norm_sa > 1e-10 and norm_sb > 1e-10 else 0.0

    # ── matched score: E{i}/A_/B_ labels ─────────────────────────────────────
    matches = declare_entity_matches(metrics_csv, composite_threshold,
                                     data_a=data_a, data_b=data_b)
    label_a, label_b = _build_label_maps(matches, entities_a, entities_b)
    edges_a  = _build_edge_list(data_a, label_a, canonical_map)
    edges_b  = _build_edge_list(data_b, label_b, canonical_map)
    nodes_a  = list(label_a.values())
    nodes_b  = list(label_b.values())
    wl_matched = wl_kernel(edges_a, nodes_a, edges_b, nodes_b, K)

    wl_composite = round((wl_structural + wl_matched) / 2, 4)
    shared       = set(nodes_a) & set(nodes_b)

    # ── Global shape: degree distribution + spectral ──────────────────────────
    shape = graph_shape_sim(data_a, data_b, canonical_map)
    structural_enriched = round((wl_structural + shape["shape_sim"]) / 2, 4)

    # ── Induced subgraph WL: coverage-independent consistency ─────────────────
    consistency = wl_consistency(data_a, data_b, matches, canonical_map)

    result = {
        "wl_structural":      wl_structural,
        "wl_matched":         wl_matched,
        "wl_composite":       wl_composite,
        "degree_sim":         shape["degree_sim"],
        "spectral_sim":       shape["spectral_sim"],
        "clustering_sim":     shape["clustering_sim"],
        "betweenness_sim":    shape["betweenness_sim"],
        "shape_sim":          shape["shape_sim"],
        "structural_enriched": structural_enriched,
        "match_coverage":     consistency["match_coverage"],
        "wl_consistency":     consistency["wl_consistency"],
        "induced_frac_a":     consistency["induced_frac_a"],
        "induced_frac_b":     consistency["induced_frac_b"],
        "n_entity_matches":   len(matches),
        "n_shared_labels":    len(shared),
        "n_nodes_a":          len(entities_a),
        "n_nodes_b":          len(entities_b),
        "n_edges_a":          len(raw_edges_a),
        "n_edges_b":          len(raw_edges_b),
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(result.keys()))
        w.writeheader()
        w.writerow(result)

    print(
        f"[WL]  structural={wl_structural:.4f}  shape={shape['shape_sim']:.4f}  "
        f"enriched={structural_enriched:.4f}  matched={wl_matched:.4f}  "
        f"composite={wl_composite:.4f}",
        file=sys.stderr,
    )
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_model(path: Path, key: str | None) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if key and key in data:
        return data[key]
    if "json_a" in data and key != "json_b":
        return data["json_a"]
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "WL kernel ontology similarity — anonymized graphs + structural matching.\n"
            "Declares entity matches from a merged metrics CSV, builds anonymized\n"
            "graphs, and computes Weisfeiler-Lehman kernel similarity."
        )
    )
    ap.add_argument("--pair",    metavar="JSON",  help="Pair JSON with json_a/json_b keys")
    ap.add_argument("--a",       metavar="JSON",  help="Ontology A JSON (alternative to --pair)")
    ap.add_argument("--b",       metavar="JSON",  help="Ontology B JSON (alternative to --pair)")
    ap.add_argument("--key-a",   default=None)
    ap.add_argument("--key-b",   default=None)
    ap.add_argument("--metrics", required=True,   metavar="CSV",
                    help="Merged metrics CSV with cosine_avg and wup columns")
    ap.add_argument("--out",     default=None,    metavar="CSV",
                    help="Output CSV path (default: stdout summary only)")
    ap.add_argument("--hops",    type=int, default=WL_HOPS)
    ap.add_argument("--composite-threshold", type=float, default=COMPOSITE_THRESHOLD)
    ap.add_argument("--show-matches",    action="store_true",
                    help="Print declared entity matches to stdout")
    args = ap.parse_args()

    if args.pair:
        pair_data = json.loads(Path(args.pair).read_text(encoding="utf-8"))
        data_a = pair_data.get("json_a", pair_data)
        data_b = pair_data.get("json_b", pair_data)
    elif args.a and args.b:
        data_a = _load_model(Path(args.a), args.key_a)
        data_b = _load_model(Path(args.b), args.key_b)
    else:
        ap.error("Provide either --pair or both --a and --b")

    metrics_csv = Path(args.metrics)
    out_csv     = Path(args.out) if args.out else Path("/dev/null")

    if args.show_matches:
        matches = declare_entity_matches(metrics_csv, args.composite_threshold)
        print(f"\nDeclared entity matches ({len(matches)}):")
        for m in matches:
            print(
                f"  [{m['rule']:8s}  cos={m['cosine_avg']:.3f}  wup={m['wup']:.3f}]"
                f"  {m['entity_a']:<35} <->  {m['entity_b']}"
            )

    result = run_wl_stage(
        data_a, data_b, metrics_csv, out_csv,
        K                    = args.hops,
        composite_threshold  = args.composite_threshold,
    )

    print(f"\nWL kernel similarity ({args.hops}-hop): {result['wl_sim']:.4f}")
    print(f"Entity matches  : {result['n_entity_matches']}")
    print(f"Shared labels   : {result['n_shared_labels']}")
    print(f"Graph A         : {result['n_nodes_a']} nodes, {result['n_edges_a']} edges")
    print(f"Graph B         : {result['n_nodes_b']} nodes, {result['n_edges_b']} edges")
