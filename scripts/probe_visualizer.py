#!/usr/bin/env python3
"""
scripts/probe_visualizer.py  (v2)
==================================
Generate:
  docs/probe_visualizations.pdf  — 4 pages, one per series
  docs/probe_s1.png  through  docs/probe_s4.png  — individual PNGs

Nodes are drawn as text-sized rounded boxes (not fixed-size circles).
Below each node a yellow attribute box lists its observable types.
Inter-column arrows are anchored to actual axes positions after layout.

Usage:  python scripts/probe_visualizer.py
"""
from __future__ import annotations

import json, math, sys
from collections import Counter, deque, defaultdict
from pathlib import Path

try:
    import numpy as np
    import networkx as nx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox, DrawingArea
    import matplotlib.patches as mpatches
    import matplotlib.text as mtext
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\n  pip install numpy networkx matplotlib")

REPO  = Path(__file__).resolve().parent.parent
PAIRS = REPO / "enriched_ontology_matching" / "pairs"
OUT_PDF = REPO / "docs" / "probe_visualizations.pdf"

# ── Series definitions ─────────────────────────────────────────────────────────
SERIES = [
    dict(
        title            = "Series 1 — Naming Drift",
        subtitle         = "Controlled change: entity names  ·  Expected: lexical ↓, wl_struct / shape / attr stable",
        ring             = False,
        show_attrs       = False,
        show_edge_labels = False,
        layout_h_gap     = 2.2,
        layout_v_gap     = 2.2,
        steps        = [
            ("V0\nbaseline",  None),
            ("V1\n1 rename",  "probe_s1_v0_vs_v1.json"),
            ("V2\n2 renames", "probe_s1_v0_vs_v2.json"),
            ("V3\n3 renames", "probe_s1_v0_vs_v3.json"),
            ("V4\n4 renames", "probe_s1_v0_vs_v4.json"),
            ("V5\n5 renames", "probe_s1_v0_vs_v5.json"),
        ],
    ),
    dict(
        title    = "Series 2 — Attribute Drift",
        subtitle = "Controlled change: observable types  ·  Expected: attr ↓, lexical / wl_struct / shape stable",
        ring     = False,
        steps    = [
            ("V0\nbaseline",   None),
            ("V1\n1 entity",   "probe_s2_v0_vs_v1.json"),
            ("V2\n2 entities", "probe_s2_v0_vs_v2.json"),
            ("V3\n3 entities", "probe_s2_v0_vs_v3.json"),
            ("V4\n4 entities", "probe_s2_v0_vs_v4.json"),
            ("V5\n5 entities", "probe_s2_v0_vs_v5.json"),
        ],
    ),
    dict(
        title         = "Series 3 — Topology Drift  (chain → star)",
        subtitle      = "Controlled change: composition depth  ·  Expected: wl_struct + shape ↓, lexical / attr stable",
        ring          = False,
        edge_front    = True,
        show_attrs    = False,
        layout_h_gap  = 2.2,
        layout_v_gap  = 2.2,
        steps         = [
            ("T0\nchain",        None),
            ("T1\n+pendant",     "probe_s3_t0_vs_t1.json"),
            ("T2\n2 branches",   "probe_s3_t0_vs_t2.json"),
            ("T3\nbranch+2pend", "probe_s3_t0_vs_t3.json"),
            ("T4\ncaterpillar",  "probe_s3_t0_vs_t4.json"),
            ("T5\nstar",         "probe_s3_t0_vs_t5.json"),
        ],
    ),
    dict(
        title            = "Series 4 — Density Drift  (ring → K5)",
        subtitle         = "Controlled change: edge density  ·  Expected: wl_struct + shape ↓, lexical / attr stable",
        ring             = True,
        edge_front       = True,
        show_attrs       = False,
        show_edge_labels = False,
        ring_radius      = 2.2,
        steps       = [
            ("R0\n5-cycle",   None),
            ("R1\n+1 chord",  "probe_s4_r0_vs_r1.json"),
            ("R2\n+2 chords", "probe_s4_r0_vs_r2.json"),
            ("R3\n+3 chords", "probe_s4_r0_vs_r3.json"),
            ("R4\n+4 chords", "probe_s4_r0_vs_r4.json"),
            ("R5\nK5",        "probe_s4_r0_vs_r5.json"),
        ],
    ),
]

# ── Graph construction ─────────────────────────────────────────────────────────
def build_graph(model: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    enames = {e["entityName"] for e in model["entities"]}
    for e in model["entities"]:
        G.add_node(e["entityName"])
    for e in model["entities"]:
        for a in e.get("entityAttributes", []):
            t = a["type"]
            if t in enames and t != e["entityName"] and not G.has_edge(e["entityName"], t):
                G.add_edge(e["entityName"], t, kind="comp")
    for assoc in model.get("associations", []):
        pp = assoc["associationParticipants"]
        if len(pp) < 2:
            continue
        src = next((p["entityName"] for p in pp if p.get("participantRole") == "source"), pp[0]["entityName"])
        tgt = next((p["entityName"] for p in pp if p.get("participantRole") == "target"), pp[1]["entityName"])
        if src != tgt and not G.has_edge(src, tgt):
            G.add_edge(src, tgt, kind="assoc", name=assoc["associationName"])
    return G


# ── Layouts ────────────────────────────────────────────────────────────────────
def tree_layout(G: nx.DiGraph, h_gap: float = 3.0, v_gap: float = 3.2) -> dict[str, tuple[float, float]]:
    """Reingold-Tilford style: each leaf gets 1 slot; parents span their subtree."""
    roots = [n for n in G.nodes() if G.in_degree(n) == 0] or list(G.nodes())[:1]
    root = roots[0]
    children: dict[str, list] = {n: list(G.successors(n)) for n in G.nodes()}

    def subtree_w(n: str, seen: set) -> int:
        if n in seen:
            return 1
        seen.add(n)
        kids = children[n]
        return max(sum(subtree_w(k, seen) for k in kids), 1)

    def assign(n: str, left: float, depth: int, seen: set) -> None:
        if n in seen:
            return
        seen.add(n)
        w = subtree_w(n, set())
        pos[n] = ((left + w / 2) * h_gap, -depth * v_gap)
        cursor = left
        for k in children[n]:
            kw = subtree_w(k, set())
            assign(k, cursor, depth + 1, seen)
            cursor += kw

    pos: dict[str, tuple[float, float]] = {}
    total_w = subtree_w(root, set())
    assign(root, -total_w / 2, 0, set())
    # place any disconnected nodes below
    mx = max((v[1] for v in pos.values()), default=0) - v_gap
    for n in G.nodes():
        if n not in pos:
            pos[n] = (len(pos) * h_gap, mx)
    return pos


def ring_layout(node_list: list[str], radius: float = 2.0) -> dict[str, tuple[float, float]]:
    nodes = sorted(node_list)
    n = len(nodes)
    return {
        nd: (math.cos(2 * math.pi * i / n - math.pi / 2) * radius,
             math.sin(2 * math.pi * i / n - math.pi / 2) * radius)
        for i, nd in enumerate(nodes)
    }


# ── Metrics ────────────────────────────────────────────────────────────────────
def wl_sim(G1: nx.DiGraph, G2: nx.DiGraph, h: int = 3) -> float:
    def fv(G):
        lbl = {n: str(G.degree(n)) for n in G.nodes()}
        cs = []
        for _ in range(h + 1):
            cs.append(Counter(lbl.values()))
            nw = {}
            for n in G.nodes():
                nb = sorted(lbl[x] for x in list(G.predecessors(n)) + list(G.successors(n)))
                nw[n] = str(hash((lbl[n], tuple(nb))) % 10**9)
            lbl = nw
        return cs
    c1, c2 = fv(G1), fv(G2)
    ks = sorted({k for c in c1 + c2 for k in c})
    a = np.array([sum(c.get(k, 0) for c in c1) for k in ks], dtype=float)
    b = np.array([sum(c.get(k, 0) for c in c2) for k in ks], dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def shape_sim(G1: nx.DiGraph, G2: nx.DiGraph) -> float:
    def sg(G):
        U = G.to_undirected()
        if len(U) < 2: return 0.0
        ev = np.sort(np.linalg.eigvalsh(nx.laplacian_matrix(U).toarray().astype(float)))
        return float(ev[1]) if len(ev) > 1 else 0.0
    def dv(G):
        return sorted(d for _, d in G.degree())
    d1, d2 = dv(G1), dv(G2)
    ml = max(len(d1), len(d2))
    d1 += [0] * (ml - len(d1)); d2 += [0] * (ml - len(d2))
    denom = sum(max(a, b) for a, b in zip(d1, d2)) or 1.0
    dsim = 1.0 - sum(abs(a - b) for a, b in zip(d1, d2)) / (2.0 * denom)
    m = max(sg(G1), sg(G2), 1e-9)
    return (dsim + 1.0 - abs(sg(G1) - sg(G2)) / m) / 2.0


def lexical_sim(ma, mb) -> float:
    na = {e["entityName"] for e in ma["entities"]}
    nb = {e["entityName"] for e in mb["entities"]}
    return len(na & nb) / len(na | nb) if (na | nb) else 1.0


def attr_sim(ma, mb) -> float:
    def obs(m):
        en = {e["entityName"] for e in m["entities"]}
        return Counter(a["type"] for e in m["entities"]
                       for a in e.get("entityAttributes", []) if a["type"] not in en)
    ca, cb = obs(ma), obs(mb)
    keys = set(ca) | set(cb)
    return (sum(min(ca.get(k, 0), cb.get(k, 0)) for k in keys) /
            sum(max(ca.get(k, 0), cb.get(k, 0)) for k in keys)) if keys else 1.0


def compute_metrics(ma, mb) -> dict:
    Ga, Gb = build_graph(ma), build_graph(mb)
    lex, wl, sh, att = lexical_sim(ma, mb), wl_sim(Ga, Gb), shape_sim(Ga, Gb), attr_sim(ma, mb)
    dist = math.sqrt(((1-lex)**2 + (1-wl)**2 + (1-sh)**2 + (1-att)**2) / 4)
    return dict(lex=lex, wl=wl, sh=sh, att=att, dist=dist)


# ── Step-delta: what changed from the previous variant ────────────────────────
def compute_delta(prev_model: dict, curr_model: dict) -> dict:
    """Diff curr against prev. Returns changed nodes, attr types, and edge sets."""
    prev_names = {e["entityName"] for e in prev_model["entities"]}
    curr_names = {e["entityName"] for e in curr_model["entities"]}

    # Renamed entities (appear in curr but not prev)
    changed_nodes: set[str] = curr_names - prev_names

    # Entities whose observable attribute types changed.
    # Skip attributes whose type is an entity name — those are composition links
    # that cascade-change when a referenced entity is renamed, not real attr changes.
    all_entity_names = prev_names | curr_names
    changed_attr_types: dict[str, set[str]] = {}
    for ce in curr_model["entities"]:
        name = ce["entityName"]
        if name not in prev_names:
            continue
        pe = next(e for e in prev_model["entities"] if e["entityName"] == name)
        prev_types = {a["type"] for a in pe.get("entityAttributes", [])
                      if a["type"] not in all_entity_names}
        new_display = {
            _TYPE_SHORT.get(a["type"], a["type"])
            for a in ce.get("entityAttributes", [])
            if a["type"] not in all_entity_names
            and a["type"] not in prev_types
        }
        if new_display:
            changed_attr_types[name] = new_display
            changed_nodes.add(name)

    # Edge-level changes (works whenever entity names are preserved)
    prev_G = build_graph(prev_model)
    curr_G = build_graph(curr_model)
    raw_new     = set(curr_G.edges()) - set(prev_G.edges())
    raw_removed = set(prev_G.edges()) - set(curr_G.edges())

    # Drop edges whose endpoints were renamed — those are rename-consequent,
    # not genuine structural changes, and would double-highlight the renamed node.
    renamed_to   = curr_names - prev_names   # new node names (in curr, not prev)
    renamed_from = prev_names - curr_names   # old node names (in prev, not curr)
    new_edges     = {(u, v) for u, v in raw_new
                     if u not in renamed_to   and v not in renamed_to}
    removed_edges = {(u, v) for u, v in raw_removed
                     if u not in renamed_from and v not in renamed_from}

    return dict(
        changed_nodes=changed_nodes,
        changed_attr_types=changed_attr_types,
        new_edges=new_edges,
        removed_edges=removed_edges,
    )


# ── Visual constants ───────────────────────────────────────────────────────────
_ABBREV = {
    "Crankshaft": "Crankshft", "Driveshaft": "Driveshft",
    "Transmission": "Trans.", "BrakeSystem": "BrakeSys",
    "Powerplant": "Pwrplant", "PropShaft": "PropShft", "MainShaft": "MainShft",
}
_TYPE_SHORT = {
    "Identifier": "Identif.", "Temperature": "Temp.",
    "AngularVelocity": "AngVel.", "OperationalState": "OpState",
    "ElectricPotential": "ElecVolt.", "ElectricCurrent": "ElecCurr.",
}
_C_BASE    = "#2ca02c"
_C_VAR     = "#1f77b4"
_C_COMP    = "#303030"
_C_ASSOC   = "#d62728"
_C_CHANGED = "#ff7f0e"   # orange — highlights what changed in this step


def _draw_graph(
    ax: plt.Axes,
    model: dict,
    is_baseline: bool,
    use_ring: bool,
    fixed_pos: dict | None = None,
    delta: dict | None = None,
    edge_front: bool = False,
    show_attrs: bool = True,
    show_edge_labels: bool = True,
    layout_h_gap: float = 3.0,
    layout_v_gap: float = 3.2,
) -> None:
    G = build_graph(model)
    enames = {e["entityName"] for e in model["entities"]}
    pos = fixed_pos if (use_ring and fixed_pos) else (
        ring_layout(list(G.nodes())) if use_ring
        else tree_layout(G, h_gap=layout_h_gap, v_gap=layout_v_gap)
    )

    # ── Axis limits ────────────────────────────────────────────────────────
    x_pad   = 0.35 if not show_attrs else 0.6
    y_bot   = 0.18 if not show_attrs else 0.30   # space below deepest node
    y_top   = 0.08                                # small gap above root — keeps nodes near header
    if pos:
        xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
        if use_ring:
            m = max(abs(v) for p in pos.values() for v in p) + 1.4
            ax.set_xlim(-m, m); ax.set_ylim(-m, m)
            ax.set_aspect("equal")
        else:
            xr = max(max(xs) - min(xs), 1.0)
            yr = max(max(ys) - min(ys), 1.0)
            ax.set_xlim(min(xs) - xr * x_pad, max(xs) + xr * x_pad)
            ax.set_ylim(min(ys) - yr * y_bot, max(ys) + yr * y_top)

    # ── Edges (drawn first, behind nodes) ──────────────────────────────────
    new_edges     = (delta or {}).get("new_edges", set())
    removed_edges = (delta or {}).get("removed_edges", set())

    ez  = 6 if edge_front else 2   # edge zorder
    elz = 7 if edge_front else 3   # edge-label zorder

    # Removed edges first (dashed grey, only if both endpoints exist in layout)
    for u, v in removed_edges:
        if u not in pos or v not in pos:
            continue
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-|>", mutation_scale=10,
                color="#aaaaaa", lw=1.2,
                linestyle="dashed",
                connectionstyle="arc3,rad=0.06",
            ),
            zorder=ez,
        )

    for u, v, data in G.edges(data=True):
        is_comp = data.get("kind") == "comp"
        rad = 0.06 if is_comp else 0.22
        is_new = (u, v) in new_edges
        edge_color = _C_CHANGED if is_new else (_C_COMP if is_comp else _C_ASSOC)
        lw = 2.2 if is_new else 1.6
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-|>", mutation_scale=14,
                color=edge_color, lw=lw,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=ez,
        )
        if show_edge_labels:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            ln = math.sqrt(dx * dx + dy * dy) or 1.0
            lx = mx + (-dy / ln) * rad * ln * 0.5
            ly = my + ( dx / ln) * rad * ln * 0.5
            elabel = "partOf" if is_comp else data.get("name", "assoc")
            ax.text(
                lx, ly, elabel,
                ha="center", va="center",
                fontsize=4.5, color=edge_color,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", edgecolor="none", alpha=0.85),
                zorder=elz,
            )

    # ── Nodes: outer entity box with inner attribute box ──────────────────
    node_color        = _C_BASE if is_baseline else _C_VAR
    changed_nodes     = (delta or {}).get("changed_nodes", set())
    changed_attr_types = (delta or {}).get("changed_attr_types", {})

    for node, (x, y) in pos.items():
        entity = next((e for e in model["entities"] if e["entityName"] == node), None)
        obs = []
        if entity:
            obs = [_TYPE_SHORT.get(a["type"], a["type"])
                   for a in entity.get("entityAttributes", [])
                   if a["type"] not in enames]

        is_changed   = node in changed_nodes
        border_color = _C_CHANGED if is_changed else "white"
        border_lw    = 2.5 if is_changed else 1.5
        node_changed_types = changed_attr_types.get(node, set())

        # Entity name (white bold)
        name_ta = TextArea(
            _ABBREV.get(node, node),
            textprops=dict(color="white", fontsize=9, fontweight="bold", ha="center"),
        )

        if obs and show_attrs:
            n_a   = min(len(obs), 8)
            lh    = 12   # display units per row
            n_rows = (n_a + 1) // 2   # two attrs per row
            # Two-column layout: shorter box, wider, edges stay visible
            col_w = max(len(t) for t in obs[:n_a]) * 5 + 4
            pad   = 4
            iw = 2 * col_w + 3 * pad
            ih = n_rows * lh + 8

            inner_da = DrawingArea(iw, ih, 0, 0)
            inner_da.add_artist(mpatches.FancyBboxPatch(
                (1, 1), iw - 2, ih - 2,
                boxstyle="round,pad=2",
                facecolor="#fffde7", edgecolor="#c8a000", linewidth=0.9,
            ))
            for idx, t in enumerate(obs[:n_a]):
                row = idx // 2
                col = idx % 2
                txt_color = _C_CHANGED if t in node_changed_types else "#111111"
                x_pos = pad + col_w / 2 + col * (col_w + pad)
                y_pos = ih - 4 - row * lh - lh / 2
                inner_da.add_artist(mtext.Text(
                    x_pos, y_pos, t,
                    ha="center", va="center", fontsize=5.5, color=txt_color,
                ))
            packed = VPacker(children=[name_ta, inner_da], pad=4, sep=4, align="center")
        else:
            packed = name_ta

        ab = AnnotationBbox(
            packed, (x, y),
            frameon=True,
            bboxprops=dict(
                facecolor=node_color, edgecolor=border_color,
                linewidth=border_lw, boxstyle="round,pad=0.4",
            ),
            zorder=5,
            box_alignment=(0.5, 0.5),
        )
        ax.add_artist(ab)

    ax.set_axis_off()


def _metric_text(m: dict | None) -> str:
    if m is None:
        return (
            " lexical :  1.00\n"
            " wl_strct:  1.00\n"
            " shape   :  1.00\n"
            " attr    :  1.00\n"
            "-----------------\n"
            " dist    :  0.00"
        )
    return (
        f" lexical :  {m['lex']:.2f}\n"
        f" wl_strct:  {m['wl']:.2f}\n"
        f" shape   :  {m['sh']:.2f}\n"
        f" attr    :  {m['att']:.2f}\n"
        f"-----------------\n"
        f" dist    :  {m['dist']:.2f}"
    )


# ── Page renderer ──────────────────────────────────────────────────────────────
def _render_page(series: dict) -> plt.Figure:
    steps            = series["steps"]
    n                = len(steps)
    use_ring         = series["ring"]
    edge_front       = series.get("edge_front", False)
    show_attrs       = series.get("show_attrs", True)
    show_edge_labels = series.get("show_edge_labels", True)
    layout_h_gap     = series.get("layout_h_gap", 3.0)
    layout_v_gap     = series.get("layout_v_gap", 3.2)
    ring_radius      = series.get("ring_radius", 2.0)

    fig = plt.figure(figsize=(28, 10))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.99, series["title"],
             ha="center", va="top", fontsize=17, fontweight="bold")
    fig.text(0.5, 0.962, series["subtitle"],
             ha="center", va="top", fontsize=10, color="#555555", style="italic")

    # Load baseline
    baseline_model = None
    for _, fname in steps:
        if fname is not None:
            with open(PAIRS / fname) as f:
                baseline_model = json.load(f)["json_a"]
            break

    fixed_pos = None
    if use_ring and baseline_model:
        fixed_pos = ring_layout([e["entityName"] for e in baseline_model["entities"]], radius=ring_radius)

    gs = fig.add_gridspec(
        2, n,
        height_ratios=[4.0, 1.0],
        hspace=0.01, wspace=0.10,
        left=0.01, right=0.99, top=0.90, bottom=0.03,
    )

    axes_graph: list[plt.Axes] = []
    prev_model = baseline_model  # tracks the model from the previous column

    for col, (label, fname) in enumerate(steps):
        ax_g = fig.add_subplot(gs[0, col])
        ax_m = fig.add_subplot(gs[1, col])
        axes_graph.append(ax_g)

        is_base = fname is None
        if is_base:
            model, metrics, delta = baseline_model, None, None
        else:
            with open(PAIRS / fname) as f:
                pair = json.load(f)
            model   = pair["json_b"]
            metrics = compute_metrics(pair["json_a"], pair["json_b"])
            delta   = compute_delta(prev_model, model) if prev_model else None

        _draw_graph(ax_g, model, is_base, use_ring, fixed_pos,
                    delta=delta, edge_front=edge_front,
                    show_attrs=show_attrs, show_edge_labels=show_edge_labels,
                    layout_h_gap=layout_h_gap, layout_v_gap=layout_v_gap)
        prev_model = model

        ax_g.set_title(
            label,
            fontsize=10,
            fontweight="bold" if is_base else "normal",
            color=_C_BASE if is_base else "#222222",
            pad=6, linespacing=1.5,
        )

        ax_m.set_axis_off()
        ax_m.text(
            0.5, 0.98, _metric_text(metrics),
            transform=ax_m.transAxes,
            ha="center", va="top",
            fontsize=8.0, fontfamily="monospace", color="#111111",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#f0fff0" if is_base else "#f4f4ff",
                edgecolor=_C_BASE if is_base else "#8888cc",
                linewidth=1.0,
            ),
        )

    # ── Inter-column arrows — placed after axes are drawn ──────────────────
    # Force matplotlib to compute layout so get_position() is accurate
    fig.canvas.draw()
    for col in range(n - 1):
        p = axes_graph[col].get_position()
        q = axes_graph[col + 1].get_position()
        arrow_x = (p.x1 + q.x0) / 2.0
        arrow_y = (p.y0 + p.y1) / 2.0
        fig.text(
            arrow_x, arrow_y, "→",
            ha="center", va="center",
            fontsize=22, color="#999999", fontweight="bold",
            transform=fig.transFigure,
        )

    return fig


def _add_footnote(fig: plt.Figure) -> None:
    fig.text(
        0.5, 0.004,
        "Node: green = baseline, blue = variant  |  "
        "Edge: dark = composition, red = association  |  "
        "Yellow box = observable attribute types  |  "
        "Metrics: lexical=Jaccard(names), wl_struct=WL-kernel, "
        "shape=avg(deg-sim,spectral), attr=Jaccard(obs-types)",
        ha="center", va="bottom", fontsize=6.5, color="#666666", style="italic",
    )


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(OUT_PDF) as pdf:
        for i, series in enumerate(SERIES):
            safe = series["title"].encode("ascii", errors="replace").decode("ascii")
            print(f"  Rendering {safe} ...", end=" ", flush=True)

            fig = _render_page(series)
            _add_footnote(fig)

            # PNG per series (for visual inspection)
            png_path = OUT_PDF.parent / f"probe_s{i + 1}.png"
            fig.savefig(png_path, bbox_inches="tight", dpi=130)

            pdf.savefig(fig, bbox_inches="tight", dpi=130)
            plt.close(fig)
            print("done")

    print(f"\nSaved PDF: {OUT_PDF}")
    print(f"Saved PNGs: {OUT_PDF.parent}/probe_s1.png .. probe_s4.png")


if __name__ == "__main__":
    main()
