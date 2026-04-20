"""
compare_stage.py — Ontology distance map via composite metric + MDS → Plotly 2D HTML.

Composite metric: mean of non-zero values per entity pair, anchored by cosine_avg.
Pair-level score: mean composite across all entity pairs in a merged CSV.
Distance: 1 - similarity, then symmetrised.
Embedding: sklearn MDS (metric=True, 2D).
Output: enriched_ontology_matching/outputs/ontology_map.html
"""

import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import MDS

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
MERGED_DIR = ROOT / "outputs" / "merged"
PAIRS_DIR  = ROOT / "pairs"
OUT_HTML   = ROOT / "outputs" / "ontology_map.html"

# ── metric columns in merged CSVs ──────────────────────────────────────────────
METRICS = ["cosine_avg", "avg_wup", "lin_ic", "coherence_sym"]
CONTINUOUS = METRICS  # alias kept for backward compat


def _parse_float(v) -> float | None:
    """Return float or None for blank/empty cells."""
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def compute_global_fill_rates(csvs: list[Path]) -> dict[str, float]:
    """
    For each metric, compute the fraction of all entity-pair rows (across every
    merged CSV) where the value is non-blank and > 0.  This becomes the weight
    for that metric in the weighted Euclidean distance.
    """
    counts = {m: 0 for m in METRICS}
    total  = 0
    for csv_path in csvs:
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    total += 1
                    for m in METRICS:
                        v = _parse_float(row.get(m))
                        if v is not None and v > 0:
                            counts[m] += 1
        except Exception:
            pass
    if total == 0:
        return {m: 1.0 for m in METRICS}
    return {m: counts[m] / total for m in METRICS}


def load_pair_metrics(csv_path: Path) -> dict:
    """
    Return per-metric means (mean of all non-blank values, including 0) for one
    ontology pair.  'composite' is left as None here; main() fills it in after
    computing global fill rates.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    if not rows:
        return {"n_pairs": 0, "composite": None, "enriched": False,
                "avg_wup": 0.0, "lin_ic": 0.0,
                "coherence_sym": 0.0, "cosine_avg": 0.0,
                "match_rate": 0.0}

    matched_count = sum(1 for r in rows if _parse_float(r.get("matched")) == 1.0)

    col_vals = {m: [_parse_float(r.get(m)) for r in rows] for m in METRICS}

    def safe_mean(vals):
        nz = [v for v in vals if v is not None]
        return sum(nz) / len(nz) if nz else 0.0

    enriched = any(v is not None for v in col_vals["cosine_avg"])

    return {
        "n_pairs":       len(rows),
        "composite":     None,   # filled later by main()
        "enriched":      enriched,
        "match_rate":    matched_count / len(rows),
        "cosine_avg":    safe_mean(col_vals["cosine_avg"]),
        "avg_wup":       safe_mean(col_vals["avg_wup"]),
        "lin_ic":        safe_mean(col_vals["lin_ic"]),
        "coherence_sym": safe_mean(col_vals["coherence_sym"]),
    }


def _norm_name(name: str) -> str:
    """Normalise ontology name: collapse multiple consecutive underscores → single."""
    import re
    return re.sub(r"_+", "_", name).strip("_")


def stem_to_pair(stem: str) -> tuple[str, str]:
    """Split '<OntA>_vs_<OntB>' stem into (OntA, OntB), normalising names."""
    idx = stem.index("_vs_")
    a = _norm_name(stem[:idx])
    b = _norm_name(stem[idx + 4:])
    return a, b


def domain_of(name: str) -> str:
    low = name.lower()
    if "automobile" in low or "auto" in low:
        return "Automobile"
    if "hospital" in low:
        return "Hospital"
    if "university" in low:
        return "University"
    return "Other"


def short_name(name: str) -> str:
    """Compact label: strip 'Automobile_' prefix and '_Model' / '_Network' noise."""
    s = re.sub(r"^Automobile_Component_Network_Model_", "Net:", name)
    s = re.sub(r"^Automobile_Model_", "V:", s)
    s = re.sub(r"^Hospital_Model_", "Hosp:", s)
    s = re.sub(r"^Hospital_Facility_Resource_Network_Model_", "HospNet:", s)
    s = re.sub(r"^University_Model_", "Uni:", s)
    s = re.sub(r"^University_Academic_Lifecycle_Model$", "Uni:AcadLifecycle", s)
    s = re.sub(r"_Network$", "", s)
    return s


def json_key_for_pair(a: str, b: str, available_jsons: set[str]) -> str | None:
    """Try to locate a JSON filename that corresponds to this pair (heuristic)."""
    def sig(name: str) -> str:
        low = name.lower()
        for pat, rep in [
            (r"automobile_model_v(\d)_\w+", r"v\1"),
            (r"automobile_component_network_model_mechanical_and_structural_network", "net1"),
            (r"automobile_component_network_model_packaged_assemblies_network",       "net2"),
            (r"automobile_component_network_model_serviceable_parts_interaction_network", "net3"),
            (r"hospital_model_v(\d)_\w+", r"hosp_v\1"),
            (r"hospital_facility_resource_network_model_serviceable_facility_parts_network", "hosp_svc"),
            (r"hospital_facility_resource_network_model_spatial_infrastructure_network", "hosp_spatial"),
            (r"university_model_v(\d)_\w+", r"uni_v\1"),
            (r"university_academic_lifecycle_model", "uni_alc"),
        ]:
            low = re.sub(pat, rep, low)
        return low

    sa, sb = sig(a), sig(b)
    for fname in available_jsons:
        stem = Path(fname).stem.lower().replace("-", "_")
        # try both orderings
        if sa in stem and sb in stem:
            return fname
        if sb in stem and sa in stem:
            return fname
    return None


def build_distance_matrix(
    onts: list[str], pair_data: dict, fill_rates: dict[str, float]
) -> np.ndarray:
    """
    Weighted Euclidean distance in metric space.

    For pair (A, B):
        d(A,B) = sqrt( Σ_m  fill_rate[m] · (1 − v_m)² )
                 ─────────────────────────────────────────
                        sqrt( Σ_m  fill_rate[m] )

    fill_rate[m] is the global fraction of entity-pair rows where metric m
    is populated — sparse metrics contribute less to the distance.
    Normalising by sqrt(Σ fill_rates) keeps d in [0, 1].
    """
    n      = len(onts)
    idx    = {o: i for i, o in enumerate(onts)}
    norm   = math.sqrt(sum(fill_rates[m] for m in METRICS)) or 1.0
    dist   = np.full((n, n), 1.0)
    np.fill_diagonal(dist, 0.0)

    for (a, b), metrics in pair_data.items():
        i, j = idx[a], idx[b]
        d_sq = sum(fill_rates[m] * (1.0 - metrics[m]) ** 2 for m in METRICS)
        d    = min(math.sqrt(d_sq) / norm, 1.0)
        dist[i][j] = d
        dist[j][i] = d

    dist = (dist + dist.T) / 2
    return dist


# ── colour palette ─────────────────────────────────────────────────────────────
DOMAIN_COLOUR = {
    "Automobile": "#1565C0",
    "Hospital":   "#C62828",
    "University": "#2E7D32",
    "Other":      "#616161",
}


def _edge_colour(comp: float) -> str:
    """Continuous interpolation: light gray (low similarity) → deep blue (high)."""
    t = min(max(comp, 0.0), 1.0)
    r = int(200 + (21  - 200) * t)
    g = int(200 + (101 - 200) * t)
    bv = int(200 + (192 - 200) * t)
    alpha = 0.20 + 0.75 * t
    return f"rgba({r},{g},{bv},{alpha:.2f})"


def _edge_width(comp: float) -> float:
    """Width: 0.6 px (low) → 9 px (high), power-law to spread mid-range pairs."""
    return 0.6 + 8.4 * (min(max(comp, 0.0), 1.0) ** 1.8)


def _compute_mst(onts: list[str], pair_data: dict) -> set:
    """Maximum spanning tree (highest composite = lowest distance) via Kruskal's."""
    edges = sorted(pair_data.items(), key=lambda kv: kv[1]["composite"], reverse=True)
    parent = {o: o for o in onts}
    rnk    = {o: 0  for o in onts}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rnk[rx] < rnk[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rnk[rx] == rnk[ry]:
            rnk[rx] += 1
        return True

    mst = set()
    for (a, b), _ in edges:
        if union(a, b):
            mst.add(frozenset([a, b]))
        if len(mst) == len(onts) - 1:
            break
    return mst


def _compute_topk(onts: list[str], pair_data: dict, k: int) -> set:
    """Edges where at least one endpoint ranks the other in its top-k by composite."""
    nbrs: dict[str, list] = {o: [] for o in onts}
    for (a, b), m in pair_data.items():
        nbrs[a].append((m["composite"], b))
        nbrs[b].append((m["composite"], a))
    result = set()
    for o, lst in nbrs.items():
        lst.sort(reverse=True)
        for _, other in lst[:k]:
            result.add(frozenset([o, other]))
    return result


def build_html(onts, coords, pair_data, available_jsons, stress: float,
               fill_rates: dict | None = None):
    domains = [domain_of(o) for o in onts]
    labels  = [short_name(o) for o in onts]
    x_all   = coords[:, 0].tolist()
    y_all   = coords[:, 1].tolist()
    idx     = {o: i for i, o in enumerate(onts)}
    fr      = fill_rates or {}

    # ── compute edge subsets ──────────────────────────────────────────────────
    mst_set  = _compute_mst(onts, pair_data)
    top3_set = _compute_topk(onts, pair_data, 3)
    top5_set = _compute_topk(onts, pair_data, 5)

    # ── one trace per edge (sorted strongest first for legend ordering) ────────
    edge_order = sorted(pair_data.items(), key=lambda kv: -kv[1]["composite"])

    edge_traces   = []   # go.Scatter per pair
    is_mst_flags  = []   # parallel bool list
    is_top3_flags = []
    is_top5_flags = []

    for (a, b), metrics in edge_order:
        comp = metrics["composite"]
        i, j = idx[a], idx[b]
        key  = frozenset([a, b])

        tip = (
            f"<b>{short_name(a)}</b> — <b>{short_name(b)}</b><br>"
            f"Weighted composite: <b>{comp:.3f}</b><br>"
            f"cosine {metrics['cosine_avg']:.3f} (w={fr.get('cosine_avg',1):.2f})  "
            f"wup {metrics['avg_wup']:.3f} (w={fr.get('avg_wup',1):.2f})<br>"
            f"lin_ic {metrics['lin_ic']:.3f} (w={fr.get('lin_ic',1):.2f})  "
            f"coh {metrics['coherence_sym']:.3f} (w={fr.get('coherence_sym',1):.2f})<br>"
            f"n_pairs: {metrics['n_pairs']}"
        )

        in_mst = key in mst_set
        edge_traces.append(go.Scatter(
            x=[x_all[i], x_all[j]], y=[y_all[i], y_all[j]],
            mode="lines",
            line=dict(width=_edge_width(comp), color=_edge_colour(comp)),
            hoverinfo="text", text=[tip, tip],
            showlegend=False,
            visible=in_mst,   # default: MST view
        ))
        is_mst_flags.append(in_mst)
        is_top3_flags.append(key in top3_set)
        is_top5_flags.append(key in top5_set)

    n_edges = len(edge_traces)

    # ── node traces (one per domain) ──────────────────────────────────────────
    node_traces = []
    for domain in sorted(set(domains)):
        mask = [i for i, d in enumerate(domains) if d == domain]
        hover_texts = []
        for i in mask:
            o = onts[i]
            nbrs = sorted(
                [(m["composite"], short_name(b if a == o else a))
                 for (a, b), m in pair_data.items() if o in (a, b)],
                reverse=True,
            )
            tip = f"<b>{labels[i]}</b>  [{domain}]<br>"
            tip += "<br>".join(f"  {lbl}: {sc:.3f}" for sc, lbl in nbrs[:8])
            hover_texts.append(tip)

        colour = DOMAIN_COLOUR.get(domain, "#616161")
        node_traces.append(go.Scatter(
            x=[x_all[i] for i in mask],
            y=[y_all[i] for i in mask],
            mode="markers+text",
            marker=dict(size=22, color=colour,
                        line=dict(width=2.5, color="white"), opacity=0.92),
            text=[labels[i] for i in mask],
            textposition="top center",
            textfont=dict(size=10, color="#222"),
            hoverinfo="text", hovertext=hover_texts,
            name=domain,
            legendgroup=f"domain_{domain}",
            legendgrouptitle_text="Domain",
            showlegend=True,
        ))

    n_nodes = len(node_traces)
    fig = go.Figure(data=edge_traces + node_traces)

    # ── visibility helper ──────────────────────────────────────────────────────
    def vis(flags):
        return list(flags) + [True] * n_nodes

    n_mst  = sum(is_mst_flags)
    n_top3 = sum(is_top3_flags)
    n_top5 = sum(is_top5_flags)

    # ── layout + buttons ───────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"Ontology Distance Map  —  Sparsity-Weighted MDS  "
                f"(stress={stress:.2f})"
                f"<br><sup>Position = weighted Euclidean distance · "
                f"Edge width & opacity = composite similarity · "
                f"Default: MST backbone ({n_mst} edges)</sup>"
            ),
            font=dict(size=14), x=0.01, xanchor="left",
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x"),
        legend=dict(groupclick="toggleitem", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=11)),
        hovermode="closest",
        plot_bgcolor="#fafafa",
        paper_bgcolor="#f0f2f5",
        margin=dict(l=10, r=10, t=90, b=10),
        height=820,
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.0, y=1.055, xanchor="left",
            showactive=True, active=0,
            buttons=[
                dict(label=f"MST ({n_mst} edges)",
                     method="restyle",
                     args=[{"visible": vis(is_mst_flags)}]),
                dict(label=f"Top-3 neighbors ({n_top3} edges)",
                     method="restyle",
                     args=[{"visible": vis(is_top3_flags)}]),
                dict(label=f"Top-5 neighbors ({n_top5} edges)",
                     method="restyle",
                     args=[{"visible": vis(is_top5_flags)}]),
                dict(label=f"All ({n_edges} edges)",
                     method="restyle",
                     args=[{"visible": vis([True] * n_edges)}]),
                dict(label="Nodes only",
                     method="restyle",
                     args=[{"visible": vis([False] * n_edges)}]),
            ],
        )],
        annotations=[dict(
            x=1.0, y=-0.01, xref="paper", yref="paper",
            text=(
                "Edge width = composite^1.8  |  "
                "Weights: cosine=0.63 wup=0.16 lin_ic=0.15 coh=0.06  |  "
                "MDS stress = lower is better"
            ),
            showarrow=False, font=dict(size=9, color="#888"), xanchor="right",
        )],
    )

    return fig


def main():
    # ── load merged CSVs ───────────────────────────────────────────────────────
    csvs = sorted(MERGED_DIR.glob("*_metrics.csv"))
    if not csvs:
        print(f"No merged CSVs found in {MERGED_DIR}", file=sys.stderr)
        sys.exit(1)

    # ── global fill rates (sparsity weights) ──────────────────────────────────
    fill_rates = compute_global_fill_rates(csvs)
    w_total    = sum(fill_rates[m] for m in METRICS) or 1.0
    print("\nGlobal metric fill rates (weight per metric):")
    for m, fr in fill_rates.items():
        print(f"  {m:<18}  fill={fr:.3f}  weight={fr/w_total:.3f}")

    pair_data: dict[tuple[str, str], dict] = {}
    for csv_path in csvs:
        stem = csv_path.stem.replace("_metrics", "")
        try:
            a, b = stem_to_pair(stem)
        except ValueError:
            print(f"  skip (can't parse pair): {csv_path.name}")
            continue
        metrics = load_pair_metrics(csv_path)
        key = (a, b)
        rev = (b, a)
        if rev in pair_data:
            existing = pair_data[rev]
            if metrics["n_pairs"] > existing["n_pairs"]:
                del pair_data[rev]
                pair_data[key] = metrics
        else:
            pair_data[key] = metrics

    if not pair_data:
        print("No valid pairs loaded.", file=sys.stderr)
        sys.exit(1)

    # ── fill in weighted composite for each pair ───────────────────────────────
    # weighted_composite = fill-rate-weighted mean of metric values (display score)
    for metrics in pair_data.values():
        metrics["composite"] = (
            sum(fill_rates[m] * metrics[m] for m in METRICS) / w_total
        )

    for (a, b), metrics in sorted(pair_data.items(),
                                  key=lambda kv: -kv[1]["composite"]):
        print(f"  {short_name(a)} vs {short_name(b)}: "
              f"composite={metrics['composite']:.3f}  (n={metrics['n_pairs']})")

    # ── collect unique ontologies ──────────────────────────────────────────────
    onts = sorted({o for pair in pair_data for o in pair})
    print(f"\n{len(onts)} unique ontologies, {len(pair_data)} pairs")

    # ── find available JSONs ───────────────────────────────────────────────────
    json_files = [p.name for p in PAIRS_DIR.glob("*.json")]
    print(f"{len(json_files)} JSON files in pairs/: {json_files}")

    # ── distance matrix + MDS ─────────────────────────────────────────────────
    dist = build_distance_matrix(onts, pair_data, fill_rates)

    # Classical MDS init gives a far better starting point than random
    from sklearn.manifold import MDS as _MDS
    mds = _MDS(n_components=2, dissimilarity="precomputed",
               random_state=42, normalized_stress="auto",
               n_init=1, init="random", max_iter=1000)
    coords = mds.fit_transform(dist)
    stress = mds.stress_
    print(f"MDS stress: {stress:.4f}")

    # ── build and save plot ────────────────────────────────────────────────────
    fig = build_html(onts, coords, pair_data, json_files, stress, fill_rates)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUT_HTML), include_plotlyjs="cdn")
    print(f"Saved: {OUT_HTML}")


if __name__ == "__main__":
    main()
