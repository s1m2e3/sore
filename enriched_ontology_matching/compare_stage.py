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
MERGED_DIR    = ROOT / "outputs" / "merged"
WL_DIR        = ROOT / "outputs" / "wl"
ATTR_DIST_DIR = ROOT / "outputs" / "attr_dist"
PAIRS_DIR  = ROOT / "pairs"
OUT_HTML      = ROOT / "outputs" / "ontology_map.html"
SUMMARIES_DIR = ROOT / "summaries"

# ── combined metrics ──────────────────────────────────────────────────────────
# lexical_sim   = max(max(matched, primary_sim), wup)  — best of string/WUP/embedding.
#                 primary_sim is type_embed_sim (attribute-type) when confident,
#                 else cosine_avg (name) as fallback — see merge_stage.py.
# wl_structural = WL kernel, all nodes anonymous     — local edge-type motifs (pair level)
# shape_sim     = avg(degree_sim, spectral_sim)      — degree + spectral shape (pair level)
# attr_weighted = attr_dist_sim, unmodified          — purely attribute-type-based (pair level)
WUP_THRESHOLD = 0.75  # WUP below this is treated as 0 in lexical_sim

METRICS = [
    "lexical_sim",    # max(max(matched, primary_sim), wup≥0.75 else 0)
    "wl_structural",  # WL kernel: anonymous nodes, edge types preserved
    "shape_sim",      # global shape: avg(degree_sim, spectral_sim)
    "attr_weighted",  # attr_dist_sim, unmodified — purely attribute-type-based
]
CONTINUOUS = METRICS  # alias kept for backward compat


def _load_wl(merged_csv: Path) -> dict:
    """Load pair-level WL metrics from the corresponding WL CSV, or return empty dict."""
    wl_csv = WL_DIR / (merged_csv.stem + "_wl.csv")
    if not wl_csv.exists():
        return {}
    try:
        rows = list(csv.DictReader(open(wl_csv, newline="", encoding="utf-8")))
        return rows[0] if rows else {}
    except Exception:
        return {}


def _load_attr_dist(merged_csv: Path) -> dict:
    """Load pair-level attr_dist_sim from the attribute reach distribution CSV."""
    ad_csv = ATTR_DIST_DIR / (merged_csv.stem + "_attr_dist.csv")
    if not ad_csv.exists():
        return {}
    try:
        rows = list(csv.DictReader(open(ad_csv, newline="", encoding="utf-8")))
        return rows[0] if rows else {}
    except Exception:
        return {}


def _parse_float(v) -> float | None:
    """Return float or None for blank/empty cells."""
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _row_avg(*vals) -> float | None:
    """Mean of non-None values; returns None if all are None."""
    nz = [v for v in vals if v is not None]
    return sum(nz) / len(nz) if nz else None


def equal_weights() -> dict[str, float]:
    """Return uniform 1/N weight for each metric — no sparsity adjustment."""
    w = 1.0 / len(METRICS)
    return {m: w for m in METRICS}


def load_pair_metrics(csv_path: Path, n_entities: int | None = None) -> dict:
    """Return per-metric means for one ontology pair from its merged metrics CSV.

    Computes the mean of all non-blank values (including 0) for each metric.
    'composite' is left as None here; the caller fills it in after computing
    global fill rates so all pairs share the same sparsity weights.

    If n_entities is given, lexical_sim = min(sum(lexical_rows) / n_entities, 1.0),
    normalising for the fraction of entities actually matched rather than the mean
    over AML-output rows only.  Default (None) preserves the original mean behaviour.

    Raises FileNotFoundError if csv_path does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Merged metrics CSV not found: {csv_path}\n"
            "Run the full pipeline (run_all_pairs.py) to generate merged CSVs first."
        )
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    if not rows:
        return {"n_pairs": 0, "composite": None, "enriched": False,
                "lexical_sim": 0.0, "wl_structural": 0.0,
                "shape_sim": 0.0, "attr_weighted": 0.0, "match_rate": 0.0}

    matched_count = sum(1 for r in rows if _parse_float(r.get("matched")) == 1.0)

    def _g(r, col):
        return _parse_float(r.get(col))

    def safe_mean(vals):
        nz = [v for v in vals if v is not None]
        return sum(nz) / len(nz) if nz else 0.0

    def _primary(r):
        """primary_sim (type-embedding, name fallback); cosine_avg for old CSVs
        generated before primary_sim existed."""
        v = _g(r, "primary_sim")
        return v if v is not None else _g(r, "cosine_avg")

    lexical_rows = [
        max(
            max(float(r.get("matched") or 0), _primary(r) or 0.0),
            (_g(r, "wup") or 0.0) if (_g(r, "wup") or 0.0) >= WUP_THRESHOLD else 0.0,
        )
        for r in rows
    ]

    enriched = any(_primary(r) is not None for r in rows)

    # WL metrics are pair-level — load from WL CSV once
    wl        = _load_wl(csv_path)
    wl_struct = _parse_float(wl.get("wl_structural")) or 0.0
    shape     = _parse_float(wl.get("shape_sim"))     or 0.0
    ad        = _load_attr_dist(csv_path)
    attr_dist = _parse_float(ad.get("attr_dist_sim")) or 0.0

    # attr_weighted = attr_dist_sim, unmodified. attr_dist_sim is already purely
    # type-based (K-hop reach + type-name embeddings, see attribute_reach.py) —
    # it no longer gets scaled by avg entity-name WUP, since that made this
    # metric depend on entity naming instead of only on attribute types.
    attr_weighted = round(attr_dist, 4) if attr_dist > 0 else 0.0

    if n_entities is not None and n_entities > 0:
        lex = min(sum(v for v in lexical_rows if v is not None) / n_entities, 1.0)
    else:
        lex = safe_mean(lexical_rows)

    return {
        "n_pairs":       len(rows),
        "composite":     None,
        "enriched":      enriched,
        "match_rate":    matched_count / len(rows),
        "lexical_sim":   lex,
        "wl_structural": wl_struct,
        "shape_sim":     shape,
        "attr_weighted": attr_weighted,
    }


def _norm_name(name: str) -> str:
    """Normalise ontology name: collapse multiple consecutive underscores → single."""
    import re
    return re.sub(r"_+", "_", name).strip("_")


def stem_to_pair(stem: str) -> tuple[str, str]:
    """Split '<OntA>_vs_<OntB>' stem into (OntA, OntB), normalising names.

    Raises ValueError if the stem does not contain the '_vs_' separator.
    """
    if "_vs_" not in stem:
        raise ValueError(
            f"CSV stem '{stem}' does not contain the '_vs_' separator. "
            "Expected format: '<OntologyA>_vs_<OntologyB>_metrics.csv'."
        )
    idx = stem.index("_vs_")
    a = _norm_name(stem[:idx])
    b = _norm_name(stem[idx + 4:])
    return a, b


def domain_of(name: str) -> str:
    """Return the domain label for an ontology name based on keyword matching.

    Returns one of the KNOWN_DOMAINS values, or 'Other' if no keyword matches.
    """
    low = name.lower()
    if "automobile" in low or "auto" in low:
        return "Automobile"
    if "hospital" in low:
        return "Hospital"
    if "university" in low:
        return "University"
    if "coffee" in low:
        return "Coffee"
    if "homebrew" in low or "homebrewing" in low:
        return "Homebrewing"
    if "smarthome" in low or "smart_home" in low or "smart home" in low:
        return "SmartHome"
    return "Other"


def short_name(name: str) -> str:
    """Compact label: strip long prefixes and noise suffixes."""
    s = re.sub(r"^Automobile_Component_Network_Model_", "Auto:", name)
    s = re.sub(r"^Automobile_Model_", "Auto:", s)
    s = re.sub(r"^Hospital_Model_", "Hosp:", s)
    s = re.sub(r"^Hospital_Facility_Resource_Network_Model_", "HospNet:", s)
    s = re.sub(r"^University_Model_", "Uni:", s)
    s = re.sub(r"^University_Academic_Lifecycle_Model$", "Uni:AcadLifecycle", s)
    s = re.sub(r"^University_Academic_Resource_Network_Model_", "UniNet:", s)
    s = re.sub(r"^University_Institutional_Hierarchy_Model$", "Uni:InstHierarchy", s)
    s = re.sub(r"^University_Campus_Resource_and_Interaction_Network_Model$", "UniNet:Campus", s)
    s = re.sub(r"^Coffee_Model_", "Coffee:", s)
    s = re.sub(r"^Coffee_Environment_Workspace_and_Equipment_Hierarchy_Model$", "Coffee:WkspEquip", s)
    s = re.sub(r"^Coffee_Environment_Component_Connectivity_Network_Model$", "Coffee:CompConn", s)
    s = re.sub(r"^Coffee_Environment_Service_Setting_and_Interaction_Model$", "Coffee:SvcSetting", s)
    s = re.sub(r"^HomeBrewing_Model_", "Brew:", s)
    s = re.sub(r"^Homebrewing_Resource_Model_Equipment_and_Ingredient_Hierarchy$", "Brew:EquipIngred", s)
    s = re.sub(r"^Homebrewing_Resource_Model_Packaged_Resource_Clusters$", "Brew:PkgClusters", s)
    s = re.sub(r"^Homebrewing_Resource_Model_Serviceable_Equipment_and_Supplies_Network$", "Brew:SvcEquip", s)
    s = re.sub(r"^SmartHome_Model_", "SH:", s)
    s = re.sub(r"^Smart_Home_System_Hierarchy_Model$", "SH:SysHierarchy", s)
    s = re.sub(r"^Smart_Home_Event_and_Control_Flow_Model$", "SH:EventCtrl", s)
    s = re.sub(r"^Smart_Home_Space_and_Device_Interaction_Network_Model$", "SH:SpaceDevice", s)
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
    onts: list[str], pair_data: dict, fill_rates: dict[str, float] | None = None
) -> np.ndarray:
    """
    Equal-weight Euclidean distance in metric space.

    For pair (A, B), only metrics with a non-zero value are included:
        d(A,B) = sqrt( Σ_available  (1 − v_m)² / N_available )

    Metrics that are unavailable for a pair (attr_weighted = 0 when no
    observable types exist) are excluded rather than penalising the score.
    Result is in [0, 1].
    """
    n    = len(onts)
    idx  = {o: i for i, o in enumerate(onts)}
    dist = np.full((n, n), 1.0)
    np.fill_diagonal(dist, 0.0)

    for (a, b), metrics in pair_data.items():
        i, j = idx[a], idx[b]
        available = [m for m in METRICS if metrics.get(m, 0) > 0]
        if not available:
            d = 1.0
        else:
            d_sq = sum((1.0 - metrics[m]) ** 2 for m in available) / len(available)
            d    = min(math.sqrt(d_sq), 1.0)
        dist[i][j] = d
        dist[j][i] = d

    dist = (dist + dist.T) / 2
    return dist


def _kruskal_stress(dist_matrix: np.ndarray, coords: np.ndarray) -> float:
    """Normalised Kruskal stress-1 between target distances and embedded distances."""
    n = len(coords)
    num = den = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            delta = dist_matrix[i, j]
            d = float(np.linalg.norm(coords[i] - coords[j]))
            num += (d - delta) ** 2
            den += delta ** 2
    return math.sqrt(num / den) if den > 0 else 0.0


def build_hierarchical_coords(
    onts: list[str],
    pair_data: dict,
    fill_rates: dict[str, float],
) -> tuple[np.ndarray, float]:
    """
    Two-level domain-centroid layout.

    Level 1 — inter-domain MDS:
        For each pair of domains compute the mean composite similarity across all
        cross-domain ontology pairs, convert to distance, then run 3-D MDS.
        The resulting 3-D positions become the domain centroids.

    Level 2 — intra-domain MDS:
        Within each domain run 3-D MDS on intra-domain pairwise distances.
        Scale the resulting offsets to at most INTRA_FRAC of the minimum
        inter-centroid gap so domain clusters never overlap, then translate
        each cluster to sit on its centroid.

    Returns (coords array of shape (n_onts, 3), overall Kruskal stress).
    """
    from sklearn.manifold import MDS

    domains_list = sorted({domain_of(o) for o in onts})
    n_dom = len(domains_list)
    dom_idx = {d: i for i, d in enumerate(domains_list)}

    domain_members: dict[str, list[str]] = {d: [] for d in domains_list}
    for o in onts:
        domain_members[domain_of(o)].append(o)

    # ── Level 1: inter-domain distance matrix ─────────────────────────────
    dom_sim_sum = np.zeros((n_dom, n_dom))
    dom_sim_cnt = np.zeros((n_dom, n_dom))
    for (a, b), metrics in pair_data.items():
        da, db = domain_of(a), domain_of(b)
        if da == db:
            continue
        i, j = dom_idx[da], dom_idx[db]
        s = metrics["composite"]
        dom_sim_sum[i, j] += s
        dom_sim_sum[j, i] += s
        dom_sim_cnt[i, j] += 1
        dom_sim_cnt[j, i] += 1

    dom_dist = np.ones((n_dom, n_dom))
    np.fill_diagonal(dom_dist, 0.0)
    for i in range(n_dom):
        for j in range(n_dom):
            if i != j and dom_sim_cnt[i, j] > 0:
                dom_dist[i, j] = 1.0 - dom_sim_sum[i, j] / dom_sim_cnt[i, j]

    mds1 = MDS(n_components=3, dissimilarity="precomputed",
               random_state=42, normalized_stress="auto",
               n_init=8, max_iter=2000)
    centroid_coords = mds1.fit_transform(dom_dist)   # (n_dom, 3)
    print(f"  Domain-level MDS stress:    {mds1.stress_:.4f}")

    # normalise centroid spread to unit std
    std = centroid_coords.std(axis=0)
    std[std == 0] = 1.0
    centroid_coords /= std.max()

    # minimum inter-centroid gap → caps intra-domain cluster radius
    min_inter_gap = float("inf")
    for i in range(n_dom):
        for j in range(i + 1, n_dom):
            gap = float(np.linalg.norm(centroid_coords[i] - centroid_coords[j]))
            min_inter_gap = min(min_inter_gap, gap)
    if not math.isfinite(min_inter_gap) or min_inter_gap == 0:
        min_inter_gap = 1.0

    INTRA_FRAC = 0.38   # intra-cluster radius ≤ 38 % of closest centroid gap
    max_intra_radius = min_inter_gap * INTRA_FRAC

    # ── Level 2: per-domain intra MDS ────────────────────────────────────
    ont_idx = {o: i for i, o in enumerate(onts)}
    coords  = np.zeros((len(onts), 3))

    for d in domains_list:
        members = domain_members[d]
        centroid = centroid_coords[dom_idx[d]]

        if len(members) == 1:
            coords[ont_idx[members[0]]] = centroid
            continue

        n_m   = len(members)
        m_idx = {o: i for i, o in enumerate(members)}
        intra = np.ones((n_m, n_m))
        np.fill_diagonal(intra, 0.0)
        for (a, b), metrics in pair_data.items():
            if a in m_idx and b in m_idx:
                s = metrics["composite"]
                i, j = m_idx[a], m_idx[b]
                intra[i, j] = 1.0 - s
                intra[j, i] = 1.0 - s
        intra = (intra + intra.T) / 2

        if n_m == 2:
            offset = np.zeros(3)
            offset[0] = intra[0, 1] * max_intra_radius * 0.5
            coords[ont_idx[members[0]]] = centroid - offset
            coords[ont_idx[members[1]]] = centroid + offset
            continue

        mds2 = MDS(n_components=3, dissimilarity="precomputed",
                   random_state=42, normalized_stress="auto",
                   n_init=4, max_iter=1000)
        intra_coords = mds2.fit_transform(intra)   # (n_m, 3)
        print(f"  {d:12s} intra MDS stress: {mds2.stress_:.4f}")

        # scale so cluster radius ≤ max_intra_radius
        radius = np.linalg.norm(intra_coords - intra_coords.mean(axis=0),
                                axis=1).max()
        if radius > 0:
            intra_coords = intra_coords / radius * max_intra_radius

        # translate to centroid
        intra_coords += centroid - intra_coords.mean(axis=0)
        for o, pos in zip(members, intra_coords):
            coords[ont_idx[o]] = pos

    # ── overall Kruskal stress of the final embedding ─────────────────────
    full_dist = build_distance_matrix(onts, pair_data, fill_rates)
    stress = _kruskal_stress(full_dist, coords)
    return coords, stress


# ── colour palette ─────────────────────────────────────────────────────────────
DOMAIN_COLOUR = {
    "Automobile":  "#1565C0",   # deep blue
    "Hospital":    "#C62828",   # deep red
    "University":  "#2E7D32",   # deep green
    "Coffee":      "#6D4C41",   # brown
    "Homebrewing": "#F57F17",   # amber/orange
    "SmartHome":   "#6A1B9A",   # purple
    "Other":       "#616161",   # grey
}

KNOWN_DOMAINS = [d for d in DOMAIN_COLOUR if d != "Other"]


def build_regularised_mds_coords(
    onts: list[str],
    pair_data: dict,
    fill_rates: dict[str, float],
    lambda_centroid: float = 0.3,
) -> tuple[np.ndarray, float]:
    """
    Regularised MDS: full 36×36 joint MDS + domain centroid-cohesion penalty.

    Minimises:
        L(X) = Σ_{i<j} (‖xi−xj‖ − d_ij)²   [individual pairwise fidelity]
             + λ · Σ_i ‖xi − c_{d(i)}‖²      [same-domain cluster cohesion]

    where c_{d(i)} = live mean position of domain-d nodes.

    Unlike the two-level hierarchical approach, every individual cross-domain
    pairwise distance (Auto:V1 ↔ Hosp:V2, etc.) directly pulls the two nodes
    toward or away from each other.  The centroid term only adds a soft
    grouping force — it does not override the pairwise geometry.

    Analytic gradients let L-BFGS-B converge in ~100–300 iterations.
    """
    from sklearn.manifold import MDS
    from scipy.optimize import minimize

    n = len(onts)

    # domain membership indices
    domain_map: dict[str, list[int]] = {}
    for i, o in enumerate(onts):
        domain_map.setdefault(domain_of(o), []).append(i)

    # full pairwise distance matrix
    dist = build_distance_matrix(onts, pair_data, fill_rates)

    # initialise with standard 3-D MDS
    mds0 = MDS(n_components=3, dissimilarity="precomputed",
               random_state=42, normalized_stress="auto",
               n_init=6, max_iter=2000)
    x0 = mds0.fit_transform(dist)
    print(f"  MDS init stress (Kruskal): {_kruskal_stress(dist, x0):.4f}")

    # --- objective + analytic gradient -----------------------------------
    def _obj(x_flat):
        X = x_flat.reshape(n, 3)

        # vectorised pairwise differences  (n, n, 3)
        D3 = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        d_act = np.sqrt((D3 ** 2).sum(axis=2) + 1e-10)   # (n, n)

        # stress term: 0.5 * Σ_{all i≠j} — counts each pair twice
        res = d_act - dist                                 # (n, n)
        np.fill_diagonal(res, 0.0)
        f_s = 0.5 * (res ** 2).sum()
        safe_d = d_act.copy()
        np.fill_diagonal(safe_d, 1.0)
        ratio = res / safe_d                               # (n, n)
        np.fill_diagonal(ratio, 0.0)
        # gradient of 0.5*Σ res²: ∂/∂xi = Σ_j ratio_ij * (xi−xj)
        g_s = (ratio[:, :, np.newaxis] * D3).sum(axis=1)  # (n, 3)

        # centroid penalty: Σ_i ‖xi − c_{d(i)}‖²
        # gradient: ∂/∂xk = 2(xk − c_{d(k)})   [exact, centroid moves with xk]
        f_c = 0.0
        g_c = np.zeros_like(X)
        for members in domain_map.values():
            if len(members) < 2:
                continue
            mask = np.array(members)
            centroid = X[mask].mean(axis=0)
            dc = X[mask] - centroid                        # (n_d, 3)
            f_c += float((dc ** 2).sum())
            g_c[mask] += 2.0 * dc

        f = f_s + lambda_centroid * f_c
        g = g_s + lambda_centroid * g_c
        return float(f), g.ravel()

    result = minimize(
        _obj, x0.ravel(), jac=True, method="L-BFGS-B",
        options={"maxiter": 4000, "ftol": 1e-14, "gtol": 1e-9},
    )
    coords = result.x.reshape(n, 3)
    print(f"  L-BFGS-B: converged={result.success}  iters={result.nit}  f={result.fun:.4f}")

    stress = _kruskal_stress(dist, coords)
    return coords, stress


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
    """Build the interactive 3D Plotly figure and the JS post-script string.

    Returns (fig, post_script) where post_script is injected into the HTML after
    Plotly renders to wire up the edge-mode buttons and domain legend toggles.
    """
    import json as _json

    domains = [domain_of(o) for o in onts]
    labels  = [short_name(o) for o in onts]
    x_all   = coords[:, 0].tolist()
    y_all   = coords[:, 1].tolist()
    z_all   = coords[:, 2].tolist()
    idx     = {o: i for i, o in enumerate(onts)}
    fr      = fill_rates or {}

    # ── compute edge subsets ──────────────────────────────────────────────────
    mst_set  = _compute_mst(onts, pair_data)
    top3_set = _compute_topk(onts, pair_data, 3)
    top5_set = _compute_topk(onts, pair_data, 5)

    # ── one trace per edge (sorted strongest first) ───────────────────────────
    edge_order = sorted(pair_data.items(), key=lambda kv: -kv[1]["composite"])

    edge_traces    = []
    is_mst_flags   = []
    is_top3_flags  = []
    is_top5_flags  = []
    edge_domains_list = []   # [[domainA, domainB], ...] parallel to edge_traces

    for (a, b), metrics in edge_order:
        comp = metrics["composite"]
        i, j = idx[a], idx[b]
        key  = frozenset([a, b])

        tip = (
            f"<b>{short_name(a)}</b> — <b>{short_name(b)}</b><br>"
            f"Weighted composite: <b>{comp:.3f}</b><br>"
            f"lexical {metrics['lexical_sim']:.3f} (w={fr.get('lexical_sim',1):.2f})  "
            f"wl_structural {metrics['wl_structural']:.3f} (w={fr.get('wl_structural',1):.2f})<br>"
            f"shape {metrics['shape_sim']:.3f} (w={fr.get('shape_sim',1):.2f})  "
            f"attr_wtd {metrics['attr_weighted']:.3f} (w={fr.get('attr_weighted',1):.2f})<br>"
            f"n_pairs: {metrics['n_pairs']}"
        )

        in_mst = key in mst_set
        edge_traces.append(go.Scatter3d(
            x=[x_all[i], x_all[j]],
            y=[y_all[i], y_all[j]],
            z=[z_all[i], z_all[j]],
            mode="lines",
            line=dict(width=max(1, int(_edge_width(comp) * 0.55)),
                      color=_edge_colour(comp)),
            hoverinfo="text", text=[tip, tip],
            showlegend=False,
            visible=in_mst,
        ))
        is_mst_flags.append(in_mst)
        is_top3_flags.append(key in top3_set)
        is_top5_flags.append(key in top5_set)
        edge_domains_list.append([domain_of(a), domain_of(b)])

    n_edges = len(edge_traces)

    # ── node traces (one per domain) ──────────────────────────────────────────
    sorted_domains = sorted(set(domains))
    node_traces = []
    for domain in sorted_domains:
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
        node_traces.append(go.Scatter3d(
            x=[x_all[i] for i in mask],
            y=[y_all[i] for i in mask],
            z=[z_all[i] for i in mask],
            mode="markers+text",
            marker=dict(size=10, color=colour,
                        line=dict(width=2, color="white"), opacity=0.92),
            text=[labels[i] for i in mask],
            textposition="top center",
            textfont=dict(size=9, color="#222"),
            hoverinfo="text", hovertext=hover_texts,
            name=domain,
            legendgroup=f"domain_{domain}",
            legendgrouptitle_text="Domain",
            showlegend=True,
        ))

    n_nodes = len(node_traces)
    fig = go.Figure(data=edge_traces + node_traces)

    n_mst  = sum(is_mst_flags)
    n_top3 = sum(is_top3_flags)
    n_top5 = sum(is_top5_flags)

    # ── layout (no updatemenus — mode buttons injected by JS) ─────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"Ontology Distance Map  —  Sparsity-Weighted MDS  "
                f"(stress={stress:.2f})"
                f"<br><sup>Position = weighted Euclidean distance · "
                f"Edge width & opacity = composite similarity</sup>"
            ),
            font=dict(size=14), x=0.01, xanchor="left",
        ),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       showbackground=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       showbackground=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       showbackground=False, title=""),
            bgcolor="#fafafa",
            aspectmode="cube",
        ),
        legend=dict(groupclick="toggleitem", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=11)),
        hovermode="closest",
        plot_bgcolor="#fafafa",
        paper_bgcolor="#f0f2f5",
        margin=dict(l=10, r=10, t=90, b=10),
        height=820,
        annotations=[dict(
            x=1.0, y=-0.01, xref="paper", yref="paper",
            text=(
                "Edge width = composite^1.8  |  "
                f"Weights: {' '.join(f'{m}={fr.get(m,0):.2f}' for m in METRICS) if fr else 'sparsity-weighted'}  |  "
                "MDS stress = lower is better"
            ),
            showarrow=False, font=dict(size=9, color="#888"), xanchor="right",
        )],
    )

    # ── build post_script: JS state machine for domain↔edge linking ───────────
    post_script = """
(function() {
  var gd = document.querySelector('.plotly-graph-div');
  if (!gd) { setTimeout(arguments.callee, 100); return; }

  var N_EDGES   = """ + str(n_edges) + """;
  var N_NODES   = """ + str(n_nodes) + """;
  var ALL_DOMAINS = """ + _json.dumps(sorted_domains) + """;
  var EDGE_DOMAINS = """ + _json.dumps(edge_domains_list) + """;
  var FILTERS = {
    mst:  """ + _json.dumps([bool(x) for x in is_mst_flags]) + """,
    top3: """ + _json.dumps([bool(x) for x in is_top3_flags]) + """,
    top5: """ + _json.dumps([bool(x) for x in is_top5_flags]) + """,
    all:  """ + _json.dumps([True] * n_edges) + """,
    none: """ + _json.dumps([False] * n_edges) + """
  };

  var activeMode = 'mst';
  var visibleDomains = new Set(ALL_DOMAINS);

  function computeVis() {
    var base = FILTERS[activeMode];
    var vis = [];
    for (var i = 0; i < N_EDGES; i++) {
      var da = EDGE_DOMAINS[i][0], db = EDGE_DOMAINS[i][1];
      vis.push(base[i] && visibleDomains.has(da) && visibleDomains.has(db));
    }
    for (var j = 0; j < N_NODES; j++) {
      // Use 'legendonly' (not false) so the legend item stays clickable
      // even when the domain is hidden — allows toggling back on.
      vis.push(visibleDomains.has(gd.data[N_EDGES + j].name) ? true : 'legendonly');
    }
    return vis;
  }

  function applyVis() { Plotly.restyle(gd, {visible: computeVis()}); }

  // Intercept legend clicks on domain (node) traces
  gd.on('plotly_legendclick', function(d) {
    var ci = d.curveNumber;
    if (ci >= N_EDGES) {
      var dom = gd.data[ci].name;
      if (visibleDomains.has(dom)) visibleDomains.delete(dom);
      else visibleDomains.add(dom);
      applyVis();
    }
    return false;  // prevent Plotly default toggle
  });

  // Double-click on domain: isolate that domain (or restore all)
  gd.on('plotly_legenddoubleclick', function(d) {
    var ci = d.curveNumber;
    if (ci >= N_EDGES) {
      var dom = gd.data[ci].name;
      if (visibleDomains.size === 1 && visibleDomains.has(dom)) {
        visibleDomains = new Set(ALL_DOMAINS);  // restore all
      } else {
        visibleDomains = new Set([dom]);          // isolate
      }
      applyVis();
    }
    return false;
  });

  // Mode switcher (called by injected buttons)
  window._omapSetMode = function(mode) {
    activeMode = mode;
    applyVis();
    document.querySelectorAll('.omap-btn').forEach(function(b) {
      var on = b.dataset.mode === mode;
      b.style.background  = on ? '#1565C0' : '#e8eaf6';
      b.style.color       = on ? '#fff'    : '#333';
      b.style.fontWeight  = on ? '600'     : '400';
      b.style.borderColor = on ? '#0d47a1' : '#aaa';
    });
  };

  // Inject mode buttons above plot
  var bar = document.createElement('div');
  bar.style.cssText = [
    'position:absolute','top:52px','left:14px','z-index:999',
    'display:flex','gap:5px','flex-wrap:wrap'
  ].join(';');
  var modes = [
    {key:'mst',  label:'MST (""" + str(n_mst) + """ edges)'},
    {key:'top3', label:'Top-3 (""" + str(n_top3) + """ edges)'},
    {key:'top5', label:'Top-5 (""" + str(n_top5) + """ edges)'},
    {key:'all',  label:'All (""" + str(n_edges) + """ edges)'},
    {key:'none', label:'Nodes only'},
  ];
  modes.forEach(function(m) {
    var btn = document.createElement('button');
    btn.textContent = m.label;
    btn.dataset.mode = m.key;
    btn.className = 'omap-btn';
    var isActive = m.key === 'mst';
    btn.style.cssText = [
      'padding:4px 11px',
      'border:1px solid ' + (isActive ? '#0d47a1' : '#aaa'),
      'border-radius:4px',
      'cursor:pointer',
      'font-size:11px',
      'background:' + (isActive ? '#1565C0' : '#e8eaf6'),
      'color:'       + (isActive ? '#fff'    : '#333'),
      'font-weight:' + (isActive ? '600'     : '400'),
    ].join(';');
    btn.onclick = function() { window._omapSetMode(m.key); };
    bar.appendChild(btn);
  });
  var wrapper = gd.parentElement;
  wrapper.style.position = 'relative';
  wrapper.appendChild(bar);
})();
"""

    return fig, post_script


def domain_distance_summary(domain: str, out_path: Path | None = None) -> dict:
    """
    Compute within-domain pairwise distances and return a JSON-serialisable dict.

    Fill rates (metric weights) are computed only over the within-domain CSVs so
    that metrics populated exclusively within-domain (GNN, entailment) are not
    diluted by cross-domain pairs where those stages were not run.

    CLI:  python compare_stage.py --domain-summary Automobile [--out results.json]
    """
    if domain not in KNOWN_DOMAINS:
        raise ValueError(
            f"Unknown domain '{domain}'. "
            f"Valid options are: {KNOWN_DOMAINS}"
        )

    all_csvs = sorted(MERGED_DIR.glob("*_metrics.csv"))
    if not all_csvs:
        raise FileNotFoundError(
            f"No merged metrics CSVs found in {MERGED_DIR}.\n"
            "Run the full pipeline (run_all_pairs.py) to generate them first."
        )

    # Identify within-domain CSVs first so fill rates reflect only this domain's data.
    domain_csvs = []
    for csv_path in all_csvs:
        stem = csv_path.stem.replace("_metrics", "")
        try:
            a, b = stem_to_pair(stem)
        except ValueError:
            continue
        if domain_of(a) == domain and domain_of(b) == domain:
            domain_csvs.append(csv_path)

    if not domain_csvs:
        raise ValueError(
            f"No within-domain pairs found for domain '{domain}'. "
            f"Check that merged CSVs exist in {MERGED_DIR} for this domain."
        )

    weights = equal_weights()

    pair_data: dict[tuple[str, str], dict] = {}
    for csv_path in domain_csvs:
        stem = csv_path.stem.replace("_metrics", "")
        a, b = stem_to_pair(stem)
        metrics = load_pair_metrics(csv_path)
        key = (a, b)
        rev = (b, a)
        if rev in pair_data:
            if metrics["n_pairs"] > pair_data[rev]["n_pairs"]:
                del pair_data[rev]
                pair_data[key] = metrics
        else:
            pair_data[key] = metrics

    for m in pair_data.values():
        available = [m[met] for met in METRICS if m.get(met, 0) > 0]
        m["composite"] = sum(available) / len(available) if available else 0.0

    onts = sorted({o for pair in pair_data for o in pair})
    dist_matrix = build_distance_matrix(onts, pair_data)
    ont_idx = {o: i for i, o in enumerate(onts)}

    pairs_out = []
    distances = []
    for (a, b), m in pair_data.items():
        d = float(dist_matrix[ont_idx[a], ont_idx[b]])
        distances.append(d)
        pairs_out.append({
            "ont_a": a,
            "ont_b": b,
            "distance": round(d, 6),
            "composite": round(m["composite"], 6),
            "n_entity_pairs": m["n_pairs"],
            "metrics": {
                met: {
                    "mean": round(m[met], 6),
                    "weight": round(weights[met], 6),
                }
                for met in METRICS
            },
        })
    pairs_out.sort(key=lambda x: x["distance"])

    result = {
        "domain": domain,
        "n_ontologies": len(onts),
        "n_pairs": len(pair_data),
        "metric_weights": {m: round(weights[m], 6) for m in METRICS},
        "average_distance": round(sum(distances) / len(distances), 6),
        "average_composite": round(
            sum(m["composite"] for m in pair_data.values()) / len(pair_data), 6
        ),
        "pairs": pairs_out,
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"Saved: {out_path}")

    return result


def main():
    """Load all merged metrics CSVs, compute the regularised MDS layout, and
    write the interactive 3D ontology distance map to outputs/ontology_map.html.
    """
    # ── load merged CSVs ───────────────────────────────────────────────────────
    csvs = sorted(MERGED_DIR.glob("*_metrics.csv"))
    if not csvs:
        print(f"No merged CSVs found in {MERGED_DIR}", file=sys.stderr)
        sys.exit(1)

    weights = equal_weights()
    print("\nMetric weights (equal, 1/N each):")
    for m in METRICS:
        print(f"  {m:<18}  weight={weights[m]:.3f}")

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

    # ── composite = mean of available (non-zero) metrics ─────────────────────
    for metrics in pair_data.values():
        available = [metrics[m] for m in METRICS if metrics.get(m, 0) > 0]
        metrics["composite"] = sum(available) / len(available) if available else 0.0

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

    # ── regularised MDS: full pairwise + centroid cohesion ────────────────────
    print("\nBuilding regularised MDS layout (lambda=0.3) ...")
    coords, stress = build_regularised_mds_coords(onts, pair_data, weights,
                                                   lambda_centroid=0.3)
    print(f"Overall Kruskal stress (final embedding): {stress:.4f}")

    # ── build and save plot ────────────────────────────────────────────────────
    fig, post_script = build_html(onts, coords, pair_data, json_files, stress, weights)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUT_HTML), include_plotlyjs="cdn", post_script=post_script)
    print(f"Saved: {OUT_HTML}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ontology distance map / domain summary")
    ap.add_argument(
        "--domain-summary", metavar="DOMAIN",
        help="Print within-domain distance JSON for DOMAIN (e.g. Automobile)",
    )
    ap.add_argument(
        "--out", metavar="PATH",
        help="Write --domain-summary output to this JSON file instead of stdout",
    )
    args = ap.parse_args()

    if args.domain_summary:
        out_path = (
            Path(args.out) if args.out
            else SUMMARIES_DIR / f"{args.domain_summary.lower()}_summary.json"
        )
        result = domain_distance_summary(args.domain_summary, out_path=out_path)
        print(json.dumps(result, indent=2))
    else:
        main()
