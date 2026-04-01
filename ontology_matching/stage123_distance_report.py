"""
stage123_distance_report.py
----------------------------
Computes a normalized composite distance metric from the first three pipeline
stages (S1 AML, S2 Lin-IC Structural, S3 Matryoshka / Mahalanobis / BC) for
every within-domain pair and for a cross-domain Coffee anchor reference.

Normalization
-------------
S1 AML coverage    : aml_cov  = AML_matched / min(|A|, |B|)          [0..1]
S2 structural boost: s2_cov   = S2_new_matches / min(|A|, |B|)        [0..1]
S3 Mahalanobis norm: mah_norm = mahal / ref_mahal                      [0..∞, anchored to 1 at median cross-domain]
S3 BC              : bc       = bhattacharyya_coefficient               [0..1]

Composite similarity (higher = more similar):
  sim = 0.25 * aml_cov  +  0.25 * s2_cov  +  0.25 * (1 - min(mah_norm, 1))  +  0.25 * bc

Composite distance:
  dist = 1 - sim

Usage
-----
    cd ontology_matching
    .venv/Scripts/python.exe stage123_distance_report.py
"""

import csv
import glob
import json
import math
import os
import statistics

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR   = os.path.join(BASE_DIR, "outputs", "reports")
STRUCT_DIR    = os.path.join(BASE_DIR, "outputs", "structural")
CHAR_WITHIN   = os.path.join(BASE_DIR, "outputs", "characterization", "distances_within_domain.csv")
CHAR_CROSS    = os.path.join(BASE_DIR, "outputs", "characterization", "distances_cross_domain.csv")


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _s3_key(ma, mb):
    """Canonical sort-key for an unordered pair of model names."""
    return tuple(sorted([ma, mb]))


# ── build lookup tables ───────────────────────────────────────────────────────

def build_s3_lookup(rows):
    """Return {(ma_sorted, mb_sorted): row} for quick lookup."""
    lookup = {}
    for r in rows:
        key = _s3_key(r["model_a"], r["model_b"])
        lookup[key] = r
    return lookup


# ── cross-domain anchor: median mahalanobis across ALL coffee cross-domain pairs ──

def compute_ref_mahal(cross_rows):
    vals = [float(r["mahalanobis_distance"]) for r in cross_rows]
    return statistics.median(vals) if vals else 1.0


# ── per-pair analysis ─────────────────────────────────────────────────────────

def analyse_pair(report_path, struct_dir, s3_lookup, ref_mahal):
    report = _load_json(report_path)
    meta   = report["metadata"]
    domain = meta["domain"]
    ma     = meta["model_a"]
    mb     = meta["model_b"]

    # ── S1 AML ────────────────────────────────────────────────────────────────
    ents_a = report["model_a"]["entities"]
    ents_b = report["model_b"]["entities"]
    n_a, n_b = len(ents_a), len(ents_b)
    n_min = min(n_a, n_b)

    aml_matched = sum(1 for e in ents_a if e["status"] == "matched")
    aml_cov     = aml_matched / n_min if n_min > 0 else 0.0

    # ── S2 Structural ─────────────────────────────────────────────────────────
    stem        = os.path.splitext(os.path.basename(report_path))[0]
    struct_path = os.path.join(struct_dir, domain, f"{stem}_structural.json")
    s2_new      = 0
    s2_avg_lin  = float("nan")
    if os.path.exists(struct_path):
        s2_data    = _load_json(struct_path)
        matches    = s2_data.get("new_matches", [])
        s2_new     = len(matches)
        if matches:
            raw_lins   = [m["lin_sim"] for m in matches]
            # lin_sim can be >1 (ratio of IC values); cap at 1 for normalisation
            capped     = [min(ls, 1.0) for ls in raw_lins]
            s2_avg_lin = sum(capped) / len(capped)
    s2_cov = s2_new / n_min if n_min > 0 else 0.0

    # ── S3 Mahalanobis + BC ───────────────────────────────────────────────────
    key = _s3_key(ma, mb)
    mah = bc = float("nan")
    if key in s3_lookup:
        row = s3_lookup[key]
        mah = float(row["mahalanobis_distance"])
        bc  = float(row["bhattacharyya_coefficient"])

    mah_norm = (mah / ref_mahal) if not math.isnan(mah) else float("nan")
    mah_inv  = max(0.0, 1.0 - min(mah_norm, 1.0)) if not math.isnan(mah_norm) else float("nan")

    # ── Composite ─────────────────────────────────────────────────────────────
    components = [aml_cov, s2_cov, mah_inv, bc]
    if any(math.isnan(v) for v in components):
        composite_sim  = float("nan")
        composite_dist = float("nan")
    else:
        composite_sim  = sum(components) / 4.0
        composite_dist = 1.0 - composite_sim

    return {
        "domain":         domain,
        "model_a":        ma,
        "model_b":        mb,
        "n_a":            n_a,
        "n_b":            n_b,
        "n_min":          n_min,
        # S1
        "aml_matched":    aml_matched,
        "aml_cov":        round(aml_cov, 4),
        # S2
        "s2_new":         s2_new,
        "s2_cov":         round(s2_cov, 4),
        "s2_avg_lin":     round(s2_avg_lin, 4) if not math.isnan(s2_avg_lin) else "—",
        # S3
        "mahalanobis":    round(mah, 4)      if not math.isnan(mah)      else "—",
        "mah_norm":       round(mah_norm, 4) if not math.isnan(mah_norm) else "—",
        "bc":             round(bc, 6)       if not math.isnan(bc)       else "—",
        # Composite
        "composite_sim":  round(composite_sim,  4) if not math.isnan(composite_sim)  else "—",
        "composite_dist": round(composite_dist, 4) if not math.isnan(composite_dist) else "—",
    }


# ── reporting ─────────────────────────────────────────────────────────────────

def _fmt(v, w=8):
    return str(v).rjust(w)


def print_domain(domain, rows):
    header = (
        f"\n{'='*110}\n"
        f"  DOMAIN: {domain}\n"
        f"{'='*110}\n"
        f"  {'Pair':<62} {'n_min':>5}  "
        f"{'S1_cov':>6}  {'S2_cov':>6}  {'S2_lin':>6}  "
        f"{'Mah':>7}  {'MahN':>6}  {'BC':>8}  "
        f"{'sim':>6}  {'dist':>6}"
    )
    print(header)
    print("  " + "-"*106)

    for r in rows:
        pair = f"{r['model_a']}  vs  {r['model_b']}"
        if len(pair) > 60:
            # Shorten model names for display
            ma_short = r['model_a'].split()[-1] if ' ' in r['model_a'] else r['model_a'][-20:]
            mb_short = r['model_b'].split()[-1] if ' ' in r['model_b'] else r['model_b'][-20:]
            pair = f"{ma_short}  vs  {mb_short}"
        pair = pair[:60].ljust(62)

        line = (
            f"  {pair} {r['n_min']:>5}  "
            f"{str(r['aml_cov']):>6}  {str(r['s2_cov']):>6}  {str(r['s2_avg_lin']):>6}  "
            f"{str(r['mahalanobis']):>7}  {str(r['mah_norm']):>6}  {str(r['bc']):>8}  "
            f"{str(r['composite_sim']):>6}  {str(r['composite_dist']):>6}"
        )
        print(line)


def print_anchor(cross_rows, s3_lookup, ref_mahal, target_domains):
    """Show Coffee-vs-each-domain rows as anchor reference (S3 only, no S1/S2)."""
    print(f"\n{'='*110}")
    print(f"  ANCHOR REFERENCE — Coffee (cross-domain)  |  ref_mahal = {ref_mahal:.4f}")
    print(f"  (S1/S2 not computed for cross-domain; showing S3 Mah / MahN / BC only)")
    print(f"{'='*110}")
    print(f"  {'Pair':<62} {'n_min':>5}  {'Mah':>7}  {'MahN':>6}  {'BC':>8}")
    print("  " + "-"*92)

    for r in cross_rows:
        if r["domain_a"] not in target_domains and r["domain_b"] not in target_domains:
            continue
        # Only show one coffee pair per domain (first V1 model)
        ma, mb = r["model_a"], r["model_b"]
        mah    = float(r["mahalanobis_distance"])
        bc     = float(r["bhattacharyya_coefficient"])
        mah_n  = mah / ref_mahal
        n_min  = min(int(r["n_entities_a"]), int(r["n_entities_b"]))

        ma_s = ma.split()[-1] if ' ' in ma else ma[-22:]
        mb_s = mb.split()[-1] if ' ' in mb else mb[-22:]
        pair = f"{ma_s}  vs  {mb_s}"
        pair = pair[:60].ljust(62)

        print(f"  {pair} {n_min:>5}  {mah:>7.4f}  {mah_n:>6.4f}  {bc:>8.6f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    within_rows = _load_csv(CHAR_WITHIN)
    cross_rows  = _load_csv(CHAR_CROSS)

    s3_within  = build_s3_lookup(within_rows)
    ref_mahal  = compute_ref_mahal(cross_rows)

    print(f"\nCross-domain anchor  (median Mahal across all Coffee cross-domain pairs): {ref_mahal:.4f}")
    print("Normalization: mah_norm = raw_mahal / ref_mahal  ->  1.0 = as far as cross-domain median\n")

    domains = sorted(set(r["domain_a"] for r in within_rows))

    all_results = []
    for domain in domains:
        report_paths = sorted(glob.glob(os.path.join(REPORTS_DIR, domain, "*.json")))
        domain_rows  = []
        for rp in report_paths:
            row = analyse_pair(rp, STRUCT_DIR, s3_within, ref_mahal)
            domain_rows.append(row)
            all_results.append(row)
        print_domain(domain, domain_rows)

    # Cross-domain Coffee anchor (sample)
    print_anchor(cross_rows, s3_within, ref_mahal, set(domains))

    # ── Summary statistics ────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("  SUMMARY: composite_dist by domain  (lower = more similar)")
    print(f"{'='*110}")
    print(f"  {'Domain':<14}  {'Pairs':>5}  {'mean_dist':>9}  {'min_dist':>8}  {'max_dist':>8}  {'mean_aml_cov':>12}  {'mean_bc':>8}")
    print("  " + "-"*80)

    for domain in domains:
        dr = [r for r in all_results if r["domain"] == domain]
        dists   = [r["composite_dist"] for r in dr if isinstance(r["composite_dist"], float)]
        aml_covs = [r["aml_cov"]       for r in dr]
        bcs      = [r["bc"]            for r in dr if isinstance(r["bc"], float)]

        if dists:
            print(
                f"  {domain:<14}  {len(dr):>5}  "
                f"{statistics.mean(dists):>9.4f}  "
                f"{min(dists):>8.4f}  "
                f"{max(dists):>8.4f}  "
                f"{statistics.mean(aml_covs):>12.4f}  "
                f"{statistics.mean(bcs) if bcs else float('nan'):>8.6f}"
            )

    # Cross-domain Coffee reference stats
    cross_mahs = [float(r["mahalanobis_distance"]) for r in cross_rows]
    cross_bcs  = [float(r["bhattacharyya_coefficient"]) for r in cross_rows]
    print(f"\n  Coffee (cross-domain anchor):  "
          f"mean Mah = {statistics.mean(cross_mahs):.4f}  "
          f"mean MahN = {statistics.mean(m/ref_mahal for m in cross_mahs):.4f}  "
          f"mean BC = {statistics.mean(cross_bcs):.6f}")


if __name__ == "__main__":
    main()
