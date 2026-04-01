"""
visualize_alignment.py
----------------------
Interactive HTML alignment visualization (plotly, zoomable).

Layout
------
  Left half  : smaller ontology — spring layout of its own associations
  Right half : larger ontology  — spring layout of its own associations
  Internal edges : drawn as a toggleable trace ("Internal associations")
  Cross edges    : one trace per matching stage — click legend to show/hide

Stages are fully separated: each stage is its own plotly trace, so you can
toggle individual stages on and off by clicking the legend entries.

Output
------
  A single standalone HTML file per pair. Open in any browser.
  Scroll to zoom, drag to pan, hover nodes for entity names.

Usage
-----
    cd ontology_matching

    python visualize_alignment.py --list
    python visualize_alignment.py --domain Automobile --smaller V2 --larger V1
    python visualize_alignment.py --domain Automobile --smaller V1 --larger V3 --matched-only
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import networkx as nx
import plotly.graph_objects as go

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #

_HERE        = os.path.dirname(__file__)
OUTPUTS      = os.path.join(_HERE, "outputs")
REPORTS_DIR  = os.path.join(OUTPUTS, "reports")
MNLI_DIR     = os.path.join(OUTPUTS, "mnli")
SUB_DIR      = os.path.join(OUTPUTS, "subsumption")
CHILD_DIR    = os.path.join(OUTPUTS, "child")
ASSOC_DIR    = os.path.join(OUTPUTS, "association")
SYNONYM_DIR  = os.path.join(OUTPUTS, "synonym")
INPUTS_DIR   = os.path.join(_HERE, "inputs",
                             "CONceptual_ExtractionCategory_Examples",
                             "CONceptual_ExtractionCategory_Examples")
VIZ_DIR      = os.path.join(OUTPUTS, "visualizations")

# --------------------------------------------------------------------------- #
# Style                                                                        #
# --------------------------------------------------------------------------- #

STAGE_ORDER  = ["S1_AML", "S2_MNLI", "S3_Subsumption", "S4_Child", "S5_Association", "S6_Synonym", "S7_Structural"]

STAGE_COLORS = {
    "S1_AML":         "#4682B4",   # steelblue
    "S2_MNLI":        "#2E8B22",   # forestgreen
    "S3_Subsumption": "#FF8C00",   # darkorange
    "S4_Child":       "#9370DB",   # mediumpurple
    "S5_Association": "#DC143C",   # crimson
    "S6_Synonym":     "#20B2AA",   # lightseagreen
    "S7_Structural":  "#708090",   # slategray
}

STAGE_LABELS = {
    "S1_AML":         "S1 — AML (lexical)",
    "S2_MNLI":        "S2 — MNLI equivalence",
    "S3_Subsumption": "S3 — Subsumption",
    "S4_Child":       "S4 — Child composition",
    "S5_Association": "S5 — Association-enriched",
    "S6_Synonym":     "S6 — SBERT synonym",
    "S7_Structural":  "S7 — Structural refinement",
}

INTERNAL_COLOR = "#CCCCCC"
NODE_UNMATCHED = "#E8E8E8"
NODE_MATCHED   = "#FFFFFF"

# --------------------------------------------------------------------------- #
# I/O helpers                                                                  #
# --------------------------------------------------------------------------- #

def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _find_input_json(domain: str, model_name: str) -> dict:
    for jf in glob.glob(os.path.join(INPUTS_DIR, domain, "*.json")):
        try:
            d = _load(jf)
            if d.get("modelName") == model_name:
                return d
        except Exception:
            continue
    return {}


def _find_report(domain: str, model_a: str, model_b: str):
    for jf in glob.glob(os.path.join(REPORTS_DIR, domain, "*.json")):
        try:
            d = _load(jf)
            meta = d.get("metadata", {})
            if {meta.get("model_a"), meta.get("model_b")} == {model_a, model_b}:
                return jf, d
        except Exception:
            continue
    return None, None


def _stage_file(directory: str, domain: str, stem: str, suffix: str) -> str | None:
    p = os.path.join(directory, domain, f"{stem}_{suffix}.json")
    return p if os.path.exists(p) else None

# --------------------------------------------------------------------------- #
# Ontology graph                                                               #
# --------------------------------------------------------------------------- #

def _entity_map(json_data: dict) -> dict[str, list]:
    result: dict[str, list] = {}
    for ent in json_data.get("entities", []):
        name  = ent.get("entityName") or ent.get("name", "")
        attrs = ent.get("entityAttributes") or ent.get("attributes", [])
        result[name] = attrs
    return result


# Common English verb stems that imply a containment / ownership relationship.
# These are generic linguistic words, not domain-specific terms.
_CONTAINMENT_VERBS = frozenset({
    "contains", "has", "includes", "aggregates", "owns", "holds", "comprises",
    "composed", "packaged", "assembles", "supports", "mounts", "houses",
    "carries", "manages", "controls", "provides", "groups",
    "in", "on", "part", "member", "inside", "within", "consists",
})


def _build_internal_graph(
    json_data: dict, entity_names: set
) -> tuple[nx.Graph, set[tuple]]:
    """
    Build the internal ontology graph using BOTH explicit associations AND
    attribute-type containment edges.

    Returns (G, inferred_edges) where inferred_edges is the set of (u,v) pairs
    that were added to connect orphaned nodes to the root — shown as dashed in
    the visualization.
    """
    G = nx.Graph()
    G.add_nodes_from(entity_names)

    # 1. Explicit associations (strictly Purple)
    for a in json_data.get("associations", []):
        aname = a.get("associationName") or a.get("name") or ""
        parts = a.get("associationParticipants") or a.get("participants", [])
        known = [p for p in parts if p in entity_names]
        for i in range(len(known)):
            for j in range(i + 1, len(known)):
                G.add_edge(known[i], known[j], kind="association", label=aname)

    # 2. Attribute-type containment (strictly Amber)
    emap = _entity_map(json_data)
    for parent, attrs in emap.items():
        if parent not in entity_names:
            continue
        for attr in attrs:
            child = attr.get("type", "")
            if child in entity_names:
                # Nested attributes take precedence as containment
                G.add_edge(parent, child, kind="containment")

    # 3. Connect orphaned components to root via inferred edges
    inferred_edges: set[tuple] = set()
    root = _find_root(G, entity_names, json_data)
    if root:
        main_comp = nx.node_connected_component(G, root)
        for node in list(entity_names):
            if node not in main_comp and node != root:
                G.add_edge(root, node, kind="inferred")
                inferred_edges.add((root, node))

    return G, inferred_edges


def _find_roots(
    G: nx.Graph, entity_names: set, json_data: dict
) -> list[str]:
    """
    Return the ordered list of root entities for one side of the ontology.
    Multiple roots are returned when the graph has genuinely independent
    subtrees (orthogonal components).

    Strategy (tried in order):

    1. Attribute-type containment — entities NOT referenced as an attribute
       type of any other entity have no ontological parent.  Used when a
       clear hierarchy exists (typical for V1/V2/V3 SystemCentric models).

    2. Association-name verb matching — for each named association whose
       camelCase tokens start with an entity name followed by a generic
       containment verb, that entity is scored as a candidate root.
       Works for models whose association names encode structural ownership
       (e.g. AutomobileComposedWithBodyFrame, VehicleContainsAll).

    3. First-participant in-degree with undirected-degree tiebreak — for
       models whose associations have no names or non-standard verbs, treat
       participants[0] as the "owner" and find nodes with in-degree 0.
       Sorted by undirected degree (most-connected root first).

    4. Fallback — the single highest-degree node.
    """
    if not entity_names:
        return []

    # ── Strategy 1: attribute-type containment ─────────────────────────────────
    children: set[str] = set()
    for ent in json_data.get("entities", []):
        attrs = ent.get("entityAttributes") or ent.get("attributes", [])
        for attr in attrs:
            t = attr.get("type", "")
            if t in entity_names:
                children.add(t)

    non_children = entity_names - children
    if non_children and len(non_children) < len(entity_names) * 0.5:
        return sorted(non_children, key=lambda n: -G.degree(n))

    # ── Strategy 2: containment-directed graph from named associations ─────────
    # Build a directed graph using only associations whose camelCase name
    # starts with an entity name followed by a containment verb.
    # In this graph, the true root has in-degree 0 AND out-degree > 0.
    sorted_ents = sorted(entity_names, key=len, reverse=True)
    CDG = nx.DiGraph()
    CDG.add_nodes_from(entity_names)
    for a in json_data.get("associations", []):
        raw = a.get("associationName") or a.get("name") or ""
        if not raw:
            continue
        tokens = [t.lower() for t in re.findall(
                      r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", raw)]
        for ent in sorted_ents:
            e_toks = [t.lower() for t in re.findall(
                          r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", ent)]
            if tokens[:len(e_toks)] == e_toks:
                rest = tokens[len(e_toks):]
                if rest and rest[0] in _CONTAINMENT_VERBS:
                    # Add directed edges: owner → all other participants
                    parts = (a.get("associationParticipants")
                             or a.get("participants", []))
                    for p in parts:
                        if p in entity_names and p != ent:
                            CDG.add_edge(ent, p)
                break

    # Roots = in-degree 0 AND out-degree > 0 in containment graph
    roots_s2 = [n for n in entity_names
                if CDG.in_degree(n) == 0 and CDG.out_degree(n) > 0]
    if roots_s2:
        return sorted(roots_s2, key=lambda n: (-CDG.out_degree(n), -G.degree(n)))

    # ── Strategy 3: first-participant in-degree ────────────────────────────────
    in_deg: dict[str, int] = {n: 0 for n in entity_names}
    for a in json_data.get("associations", []):
        parts = (a.get("associationParticipants")
                 or a.get("participants", []))
        known = [p for p in parts if p in entity_names]
        if len(known) >= 2:
            for child in known[1:]:
                in_deg[child] += 1

    roots_s3 = [n for n, d in in_deg.items() if d == 0]
    if roots_s3:
        return sorted(roots_s3, key=lambda n: -G.degree(n))

    # ── Strategy 4: fallback ───────────────────────────────────────────────────
    return [max(entity_names, key=lambda n: G.degree(n))]


def _find_root(G: nx.Graph, entity_names: set, json_data: dict) -> str | None:
    """Convenience wrapper — returns the single best root."""
    roots = _find_roots(G, entity_names, json_data)
    return roots[0] if roots else None


# --------------------------------------------------------------------------- #
# Hierarchical BFS layout                                                      #
# --------------------------------------------------------------------------- #

def _hierarchical_layout(
    G: nx.Graph,
    entity_names: set,
    x_lo: float, x_hi: float,
    y_lo: float = 0.03, y_hi: float = 0.97,
    json_data: dict | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Top-down hierarchical layout that supports multiple roots / subtrees.

    Each connected component of G gets its own vertical column within the
    [x_lo, x_hi] band, sized proportionally to the number of nodes in that
    component.  Within each column, the local root sits at the top and its
    height is strictly proportional to its distance from that root,
    normalized globally across all orthogonal sub-trees.
    """
    if not entity_names:
        return {}

    jdata = json_data or {}

    # ── Find connected components (using real edges, i.e. not inferred) ────────
    real_G = nx.Graph()
    real_G.add_nodes_from(entity_names)
    for u, v, d in G.edges(data=True):
        if d.get("kind") != "inferred":
            real_G.add_edge(u, v, **d)

    components = list(nx.connected_components(real_G))
    # Sort: largest component first
    components.sort(key=len, reverse=True)

    # ── 1. Calculate levels globally first ────────────────────────────────────
    all_levels: dict[str, int] = {}
    comp_info: list[tuple[set[str], dict[int, list[str]]]] = []

    for comp in components:
        comp_G  = real_G.subgraph(comp)
        roots   = _find_roots(comp_G, comp, jdata)
        # Roots in this component
        roots   = [r for r in roots if r in comp]
        if not roots:
            roots = [max(comp, key=lambda n: comp_G.degree(n))]

        # BFS from local roots to assign levels within this component
        levels: dict[str, int] = {r: 0 for r in roots}
        queue   = list(roots)
        visited = set(roots)
        while queue:
            node = queue.pop(0)
            for nb in comp_G.neighbors(node):
                if nb not in visited:
                    visited.add(nb)
                    levels[nb] = levels[node] + 1
                    queue.append(nb)

        # Handle any leftovers (not reached by BFS from roots)
        max_lv = max(levels.values()) if levels else 0
        for n in comp:
            if n not in levels:
                levels[n] = max_lv + 1

        all_levels.update(levels)
        
        # Group by level for X-positioning later
        by_level: dict[int, list[str]] = {}
        for n, lv in levels.items():
            by_level.setdefault(lv, []).append(n)
        comp_info.append((comp, by_level))

    # Find the global max level across all orthogonal sub-trees
    global_max_level = max(all_levels.values()) if all_levels else 0

    # ── 2. Assign (x, y) positions ────────────────────────────────────────────
    pos: dict[str, tuple[float, float]] = {}
    total_nodes = len(entity_names)
    x_cursor = x_lo

    for comp, by_level in comp_info:
        # Allocate x-band proportional to component size
        frac   = len(comp) / total_nodes
        x_band = (x_hi - x_lo) * frac
        cx_lo  = x_cursor
        cx_hi  = x_cursor + x_band
        x_cursor += x_band

        for lv, nodes in by_level.items():
            nodes_sorted = sorted(nodes)
            n_nodes = len(nodes_sorted)
            # Normalize Y using the GLOBAL max level
            y = y_hi - (lv / max(global_max_level, 1)) * (y_hi - y_lo)
            for i, node in enumerate(nodes_sorted):
                # Center nodes within their allotted X-band for this level
                x = cx_lo + (i + 0.5) / n_nodes * (cx_hi - cx_lo)
                pos[node] = (x, y)

    return pos

# --------------------------------------------------------------------------- #
# Collect alignment edges                                                      #
# --------------------------------------------------------------------------- #

def collect_alignment_edges(domain: str, smaller_name: str, larger_name: str) -> dict:
    report_path, report = _find_report(domain, smaller_name, larger_name)
    if report is None:
        raise FileNotFoundError(f"No AML report for {smaller_name} vs {larger_name} in {domain}")

    meta   = report["metadata"]
    na, nb = meta["model_a"], meta["model_b"]
    sa = report["summary"]["model_a"]
    sb = report["summary"]["model_b"]
    ta = sa["matched"] + sa["ambiguous"] + sa["missing"]
    tb = sb["matched"] + sb["ambiguous"] + sb["missing"]

    if ta <= tb:
        small_side, large_side, sname, lname = "model_a", "model_b", na, nb
    else:
        small_side, large_side, sname, lname = "model_b", "model_a", nb, na

    smaller_entities = [e["name"] for e in report[small_side]["entities"]]
    larger_entities  = [e["name"] for e in report[large_side]["entities"]]
    larger_set       = set(larger_entities)
    smaller_set      = set(smaller_entities)

    edges: list[dict] = []

    # S1: AML
    for e in report[small_side]["entities"]:
        if e["status"] != "matched":
            continue
        target = (e.get("matched_to") or e.get("match") or
                  e.get("matchedTo")  or e.get("alignedWith"))
        if target and target in larger_set:
            edges.append({"smaller": e["name"], "larger": target,
                          "stage": "S1_AML", "score": e.get("similarity", 1.0)})

    stem = os.path.splitext(os.path.basename(report_path))[0]

    # S2: MNLI
    p = _stage_file(MNLI_DIR, domain, stem, "mnli")
    if p:
        for m in _load(p).get("new_matches", []):
            edges.append({"smaller": m["smaller_entity"], "larger": m["larger_entity"],
                          "stage": "S2_MNLI", "score": m.get("mnli_score", 0.0)})

    # S3: Subsumption
    p = _stage_file(SUB_DIR, domain, stem, "subsumption")
    if p:
        for g in _load(p).get("subsumption_groups", []):
            abstract, direction = g["abstract_entity"], g["direction"]
            for i, member in enumerate(g["subsumed_entities"]):
                sc = g["fwd_scores"][i] if i < len(g["fwd_scores"]) else 0.0
                if direction == "large_abstracts_small":
                    if abstract in larger_set and member in smaller_set:
                        edges.append({"smaller": member, "larger": abstract,
                                      "stage": "S3_Subsumption", "score": sc})
                else:
                    if abstract in smaller_set and member in larger_set:
                        edges.append({"smaller": abstract, "larger": member,
                                      "stage": "S3_Subsumption", "score": sc})

    # S4: Child
    p = _stage_file(CHILD_DIR, domain, stem, "child")
    if p:
        for m in _load(p).get("new_matches", []):
            edges.append({"smaller": m["smaller_entity"], "larger": m["larger_entity"],
                          "stage": "S4_Child", "score": m.get("child_score", 0.0)})

    # S5: Association
    p = _stage_file(ASSOC_DIR, domain, stem, "assoc")
    if p:
        for m in _load(p).get("new_matches", []):
            edges.append({"smaller": m["smaller_entity"], "larger": m["larger_entity"],
                          "stage": "S5_Association", "score": m.get("assoc_score", 0.0)})

    # S6: Synonym (SBERT)
    p = _stage_file(SYNONYM_DIR, domain, stem, "synonym")
    if p:
        for m in _load(p).get("new_matches", []):
            edges.append({"smaller": m["smaller_entity"], "larger": m["larger_entity"],
                          "stage": "S6_Synonym", "score": m.get("sbert_score", 0.0)})

    # S7: Structural
    p = _stage_file(os.path.join(OUTPUTS, "structural"), domain, stem, "structural")
    if p:
        for m in _load(p).get("new_matches", []):
            edges.append({"smaller": m["smaller_entity"], "larger": m["larger_entity"],
                          "stage": "S7_Structural", "score": m.get("struct_score", 0.0)})

    return {
        "smaller_name": sname, "larger_name": lname,
        "smaller_entities": smaller_entities,
        "larger_entities":  larger_entities,
        "edges": edges, "stem": stem, "domain": domain,
    }

# --------------------------------------------------------------------------- #
# Build plotly figure                                                          #
# --------------------------------------------------------------------------- #

def _edge_xy(x0, y0, x1, y1):
    """Return x, y arrays for a plotly line segment with a gap (None separator)."""
    return [x0, x1, None], [y0, y1, None]


def _bezier_xy(x0, y0, x1, y1, t=0.15, steps=20):
    """
    Compute a quadratic bezier curve between two points.
    The control point is offset perpendicular to the midpoint.
    Returns x, y arrays with a None separator for plotly multi-edge traces.
    """
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    # perpendicular offset
    cx = mx - t * dy
    cy = my + t * dx
    xs, ys = [], []
    for i in range(steps + 1):
        s = i / steps
        bx = (1 - s) ** 2 * x0 + 2 * (1 - s) * s * cx + s ** 2 * x1
        by = (1 - s) ** 2 * y0 + 2 * (1 - s) * s * cy + s ** 2 * y1
        xs.append(bx)
        ys.append(by)
    xs.append(None)
    ys.append(None)
    return xs, ys


def build_figure(data: dict, matched_only: bool = False) -> go.Figure:
    sname  = data["smaller_name"]
    lname  = data["larger_name"]
    s_ents = list(data["smaller_entities"])
    l_ents = list(data["larger_entities"])
    edges  = data["edges"]
    domain = data["domain"]

    matched_smaller = {e["smaller"] for e in edges}
    matched_larger  = {e["larger"]  for e in edges}

    if matched_only:
        s_ents = [e for e in s_ents if e in matched_smaller]
        l_ents = [e for e in l_ents if e in matched_larger]

    s_set = set(s_ents)
    l_set = set(l_ents)

    # Build internal graphs and layout
    s_json  = _find_input_json(domain, sname)
    l_json  = _find_input_json(domain, lname)
    G_small, inferred_s = _build_internal_graph(s_json, s_set)
    G_large, inferred_l = _build_internal_graph(l_json, l_set)

    pos_s = _hierarchical_layout(G_small, s_set, 0.02, 0.46, json_data=s_json)
    pos_l = _hierarchical_layout(G_large, l_set, 0.54, 0.98, json_data=l_json)

    fig = go.Figure()

    # ── Internal edges: association and containment ───────────────────────────
    # Collect by kind so we can style them differently and add hover labels.
    def _internal_traces(G, pos_map):
        assoc_x, assoc_y = [], []
        assoc_hover_x, assoc_hover_y, assoc_hover_txt = [], [], []
        cont_x, cont_y   = [], []
        cont_hover_x, cont_hover_y, cont_hover_txt = [], [], []

        for u, v, d in G.edges(data=True):
            kind = d.get("kind", "")
            if kind == "inferred":
                continue
            if u not in pos_map or v not in pos_map:
                continue
            ex, ey = _edge_xy(*pos_map[u], *pos_map[v])
            mx = (pos_map[u][0] + pos_map[v][0]) / 2
            my = (pos_map[u][1] + pos_map[v][1]) / 2

            if kind == "association":
                assoc_x += ex; assoc_y += ey
                label = d.get("label") or "association"
                assoc_hover_x.append(mx); assoc_hover_y.append(my)
                assoc_hover_txt.append(f"<b>{label}</b><br>{u} ↔ {v}")
            else:  # containment
                cont_x += ex; cont_y += ey
                cont_hover_x.append(mx); cont_hover_y.append(my)
                cont_hover_txt.append(f"<b>composition</b><br>{u} → {v}")

        return (assoc_x, assoc_y, assoc_hover_x, assoc_hover_y, assoc_hover_txt,
                cont_x, cont_y, cont_hover_x, cont_hover_y, cont_hover_txt)

    (ax_s, ay_s, ahx_s, ahy_s, aht_s,
     cx_s, cy_s, chx_s, chy_s, cht_s) = _internal_traces(G_small, pos_s)
    (ax_l, ay_l, ahx_l, ahy_l, aht_l,
     cx_l, cy_l, chx_l, chy_l, cht_l) = _internal_traces(G_large, pos_l)

    # Association edges — named relationships from JSON associations array (purple)
    all_ax = ax_s + ax_l
    if all_ax:
        fig.add_trace(go.Scatter(
            x=all_ax, y=ay_s + ay_l, mode="lines",
            line=dict(color="#8844CC", width=1.8),
            opacity=0.70,
            name="Association",
            legendgroup="internal_assoc",
            hoverinfo="skip",
            showlegend=True,
        ))
        # Invisible hover markers at midpoints
        fig.add_trace(go.Scatter(
            x=ahx_s + ahx_l, y=ahy_s + ahy_l,
            mode="markers",
            marker=dict(size=6, color="#8844CC", opacity=0.0),
            text=aht_s + aht_l,
            hoverinfo="text",
            showlegend=False,
        ))

    # Composition edges — entity A's attribute has type=entity B (amber)
    all_cx = cx_s + cx_l
    if all_cx:
        fig.add_trace(go.Scatter(
            x=all_cx, y=cy_s + cy_l, mode="lines",
            line=dict(color="#E07820", width=1.8),
            opacity=0.70,
            name="Composition",
            legendgroup="internal_cont",
            hoverinfo="skip",
            showlegend=True,
        ))
        # Invisible hover markers at midpoints
        fig.add_trace(go.Scatter(
            x=chx_s + chx_l, y=chy_s + chy_l,
            mode="markers",
            marker=dict(size=6, color="#E07820", opacity=0.0),
            text=cht_s + cht_l,
            hoverinfo="text",
            showlegend=False,
        ))

    # ── Inferred containment edges (dashed) ───────────────────────────────────
    dx, dy = [], []
    for u, v in inferred_s:
        if u in pos_s and v in pos_s:
            ex, ey = _edge_xy(*pos_s[u], *pos_s[v])
            dx += ex; dy += ey
    for u, v in inferred_l:
        if u in pos_l and v in pos_l:
            ex, ey = _edge_xy(*pos_l[u], *pos_l[v])
            dx += ex; dy += ey

    if dx:
        fig.add_trace(go.Scatter(
            x=dx, y=dy, mode="lines",
            line=dict(color="#AAAAAA", width=1.0, dash="dash"),
            opacity=0.45,
            name="Inferred containment",
            legendgroup="inferred",
            hoverinfo="skip",
            showlegend=True,
        ))

    # ── Cross-ontology edges — one trace per stage ────────────────────────────
    for stg in STAGE_ORDER:
        stg_edges = [e for e in edges if e["stage"] == stg]
        if not stg_edges:
            continue

        ex_all, ey_all = [], []
        hover_texts = []
        for e in stg_edges:
            s, l = e["smaller"], e["larger"]
            if s not in pos_s or l not in pos_l:
                continue
            bx, by = _bezier_xy(*pos_s[s], *pos_l[l], t=0.12)
            ex_all += bx
            ey_all += by
            # invisible hover points at midpoint
            hover_texts.append(
                f"<b>{s}</b> → <b>{l}</b><br>Stage: {STAGE_LABELS[stg]}<br>"
                f"Score: {e.get('score', 0):.3f}"
            )

        fig.add_trace(go.Scatter(
            x=ex_all, y=ey_all, mode="lines",
            line=dict(color=STAGE_COLORS[stg], width=2.0),
            opacity=0.75,
            name=f"{STAGE_LABELS[stg]} ({len(stg_edges)})",
            legendgroup=stg,
            hoverinfo="skip",
        ))

        # Invisible midpoint markers for hover tooltips
        mid_x, mid_y, mid_hover = [], [], []
        for e in stg_edges:
            s, l = e["smaller"], e["larger"]
            if s not in pos_s or l not in pos_l:
                continue
            mx = (pos_s[s][0] + pos_l[l][0]) / 2
            my = (pos_s[s][1] + pos_l[l][1]) / 2
            mid_x.append(mx); mid_y.append(my)
            mid_hover.append(
                f"<b>{s}</b> → <b>{l}</b><br>"
                f"Stage: {STAGE_LABELS[stg]}<br>"
                f"Score: {e.get('score', 0):.3f}"
            )

        fig.add_trace(go.Scatter(
            x=mid_x, y=mid_y, mode="markers",
            marker=dict(size=8, color=STAGE_COLORS[stg], opacity=0.0),
            text=mid_hover, hoverinfo="text",
            name=STAGE_LABELS[stg],
            legendgroup=stg,
            showlegend=False,
        ))

    # ── Per-node dominant stage ───────────────────────────────────────────────
    small_stage: dict[str, str] = {}
    large_stage: dict[str, str] = {}
    for e in edges:
        s, l, stg = e["smaller"], e["larger"], e["stage"]
        if s not in small_stage or STAGE_ORDER.index(stg) < STAGE_ORDER.index(small_stage[s]):
            small_stage[s] = stg
        if l not in large_stage or STAGE_ORDER.index(stg) < STAGE_ORDER.index(large_stage[l]):
            large_stage[l] = stg

    # ── Nodes ─────────────────────────────────────────────────────────────────
    def _add_nodes(pos_map, stage_map, matched_set, side_label, G_side):
        for name, (x, y) in pos_map.items():
            matched  = name in matched_set
            stg      = stage_map.get(name)
            color    = STAGE_COLORS[stg] if matched and stg else "#AAAAAA"
            face     = NODE_MATCHED if matched else NODE_UNMATCHED
            degree   = G_side.degree(name) if name in G_side else 0
            node_size = 10 + min(degree * 2, 12)   # slightly larger for higher-degree nodes
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers+text",
                marker=dict(
                    size=node_size,
                    color=face,
                    line=dict(color=color, width=2.2 if matched else 0.8),
                ),
                text=[name],
                textposition="middle right" if x < 0.5 else "middle left",
                textfont=dict(size=9, color="#333333" if matched else "#AAAAAA"),
                hovertext=(
                    f"<b>{name}</b><br>Side: {side_label}<br>"
                    f"Status: {'matched (' + STAGE_LABELS.get(stg, '') + ')' if matched else 'unmatched'}<br>"
                    f"Connections: {degree}"
                ),
                hoverinfo="text",
                showlegend=False,
            ))

    _add_nodes(pos_s, small_stage, matched_smaller, "smaller", G_small)
    _add_nodes(pos_l, large_stage, matched_larger,  "larger",  G_large)

    # ── Divider line ──────────────────────────────────────────────────────────
    fig.add_shape(
        type="line", x0=0.50, x1=0.50, y0=0, y1=1,
        line=dict(color="#CCCCCC", width=1, dash="dot"),
        layer="below",
    )

    # ── Column header annotations ─────────────────────────────────────────────
    n_sm = len(matched_smaller & s_set)
    n_lg = len(matched_larger  & l_set)

    fig.add_annotation(x=0.24, y=1.03, xref="paper", yref="paper",
                       text=f"<b>{sname}</b>",
                       showarrow=False, font=dict(size=11), xanchor="center")
    fig.add_annotation(x=0.76, y=1.03, xref="paper", yref="paper",
                       text=f"<b>{lname}</b>",
                       showarrow=False, font=dict(size=11), xanchor="center")

    # ── Layout ────────────────────────────────────────────────────────────────
    n_total = len(s_ents) + len(l_ents)
    fig_h   = max(700, n_total * 5)

    fig.update_layout(
        title=dict(
            text=(f"Ontology Alignment — {domain}<br>"
                  f"<sup>{n_sm}/{len(s_ents)} smaller matched  |  "
                  f"{n_lg}/{len(l_ents)} larger matched  |  "
                  f"{len(edges)} total correspondences</sup>"),
            font=dict(size=14),
            x=0.5,
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.01, y=1.0,
            xanchor="left",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#DDDDDD",
            borderwidth=1,
            font=dict(size=10),
        ),
        xaxis=dict(visible=False, range=[-0.02, 1.10]),
        yaxis=dict(visible=False, range=[-0.02, 1.05], scaleanchor="x", scaleratio=1),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FAFAFA",
        hovermode="closest",
        height=fig_h,
        width=None,   # responsive width
        margin=dict(l=10, r=200, t=80, b=30),
    )

    return fig

# --------------------------------------------------------------------------- #
# CLI helpers                                                                  #
# --------------------------------------------------------------------------- #

def list_pairs() -> None:
    print("Available pairs:\n")
    for jf in sorted(glob.glob(os.path.join(REPORTS_DIR, "**", "*.json"), recursive=True)):
        try:
            d = _load(jf)
            meta   = d.get("metadata", {})
            domain = os.path.basename(os.path.dirname(jf))
            sa = d["summary"]["model_a"]
            sb = d["summary"]["model_b"]
            ta = sa["matched"] + sa["ambiguous"] + sa["missing"]
            tb = sb["matched"] + sb["ambiguous"] + sb["missing"]
            smaller = meta["model_a"] if ta <= tb else meta["model_b"]
            larger  = meta["model_b"] if ta <= tb else meta["model_a"]
            print(f"  {domain:<14}  {smaller}  vs  {larger}")
        except Exception:
            continue


def find_pair(domain: str, sh: str, lh: str) -> tuple[str, str]:
    for jf in glob.glob(os.path.join(REPORTS_DIR, domain, "*.json")):
        try:
            d = _load(jf)
            meta = d.get("metadata", {})
            a, b = meta["model_a"], meta["model_b"]
            if sh.lower() in a.lower() and lh.lower() in b.lower():
                return a, b
            if sh.lower() in b.lower() and lh.lower() in a.lower():
                return b, a
        except Exception:
            continue
    raise ValueError(f"No pair matching '{sh}' vs '{lh}' in '{domain}'")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive HTML ontology alignment visualization."
    )
    parser.add_argument("--list",         action="store_true")
    parser.add_argument("--domain",       type=str, default=None)
    parser.add_argument("--smaller",      type=str, default=None)
    parser.add_argument("--larger",       type=str, default=None)
    parser.add_argument("--matched-only", action="store_true")
    parser.add_argument("--out",          type=str, default=None)
    args = parser.parse_args()

    if args.list:
        list_pairs()
        return

    if not all([args.domain, args.smaller, args.larger]):
        parser.print_help()
        sys.exit(1)

    sname, lname = find_pair(args.domain, args.smaller, args.larger)
    print(f"Pair: {sname}  vs  {lname}")

    data = collect_alignment_edges(args.domain, sname, lname)
    fig  = build_figure(data, matched_only=args.matched_only)

    out_path = args.out or os.path.join(
        VIZ_DIR, args.domain,
        f"{data['stem']}{'_matched' if args.matched_only else ''}.html"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    print(f"Saved -> {out_path}")

    by_stage: dict[str, int] = {}
    for e in data["edges"]:
        by_stage[e["stage"]] = by_stage.get(e["stage"], 0) + 1
    print("\nMatches by stage:")
    for s in STAGE_ORDER:
        if s in by_stage:
            print(f"  {STAGE_LABELS[s]}: {by_stage[s]}")
    print(f"  Total: {len(data['edges'])}")


if __name__ == "__main__":
    main()
