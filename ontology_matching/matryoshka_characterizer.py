"""
matryoshka_characterizer.py
---------------------------
Stage 3: Semantic Space Characterization via Matryoshka Embeddings.

This is a CHARACTERIZATION-ONLY stage. It produces no entity-to-entity matches
and does not modify the still_missing list consumed by Stage 4 onwards.

Purpose
-------
Before neural matching (S4–S8), this stage characterizes each ontology as a
probability distribution in a shared semantic embedding space, enabling a
model-agnostic, geometry-based answer to: "how far apart are two ontologies
as semantic distributions?"

Method
------
1.  Collect all JSON models from the inputs directory.
2.  Build a natural-language description for each entity (name + attribute types).
3.  Enrich each description with WordNet root-synset anchor words — the highest-
    level conceptual category (physical_entity, abstraction, process, etc.) that
    the entity's name tokens can be mapped to via hypernym traversal.
4.  Encode all enriched descriptions with nomic-ai/nomic-embed-text-v1.5,
    a Matryoshka sentence-transformer model.  The first `--dim` dimensions of
    each 768-d vector are used (default 128); these are guaranteed to be
    meaningful for similarity tasks at any Matryoshka truncation point.
5.  For each ontology, fit a multivariate Gaussian N(μ, Σ) to the entity
    embedding cloud using Ledoit-Wolf shrinkage (robust to small sample sizes
    relative to dimensionality).
6.  For every pair of ontologies within the same domain, compute:
    (a) Mahalanobis distance between distribution means (pooled covariance).
    (b) Bhattacharyya distance and coefficient — closed-form exact overlap
        integral between two multivariate Gaussians (range [0,1]: 1 = identical).

Outputs
-------
  outputs/characterization/semantic_distances.csv       pairwise summary table
  outputs/characterization/semantic_distances.json      full pairwise results
  outputs/characterization/<domain>/<model>_cloud.json  per-model Gaussian stats

Usage
-----
    cd ontology_matching
    python matryoshka_characterizer.py                    # all domains, dim=128
    python matryoshka_characterizer.py --domain Automobile
    python matryoshka_characterizer.py --dim 64           # smaller Matryoshka slice
    python matryoshka_characterizer.py --dim 256
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from itertools import combinations
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf

# ── Point HuggingFace at the local model cache ──────────────────────────────── #
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(_MODELS_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", _MODELS_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _MODELS_DIR)
# ─────────────────────────────────────────────────────────────────────────────── #

INPUTS_DIR = os.path.join(
    os.path.dirname(__file__), "inputs",
    "CONceptual_ExtractionCategory_Examples",
    "CONceptual_ExtractionCategory_Examples",
)
CHAR_DIR   = os.path.join(os.path.dirname(__file__), "outputs", "characterization")
OUT_JSON   = os.path.join(CHAR_DIR, "semantic_distances.json")
OUT_CSV    = os.path.join(CHAR_DIR, "semantic_distances.csv")

MATRYOSHKA_MODEL  = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_DIM       = 128       # Matryoshka slice: 64 | 128 | 256 | 512 | 768
REGULARIZE_ALPHA  = 1e-4      # fallback diagonal regularisation if LW fails


# ──────────────────────────────────────────────────────────────────────────────
# WordNet root-synset anchor resolution
# ──────────────────────────────────────────────────────────────────────────────

# Depth-1 and depth-2 children of entity.n.01 used as reference anchors.
# These are the highest-level conceptual categories in WordNet's noun hierarchy.
ROOT_SYNSET_NAMES = [
    "entity.n.01",
    "physical_entity.n.01",
    "abstraction.n.06",
    "object.n.01",
    "matter.n.03",
    "process.n.06",
    "attribute.n.02",
    "measure.n.02",
    "relation.n.01",
    "group.n.01",
    "event.n.01",
    "act.n.02",
    "communication.n.02",
]


def _load_wordnet_roots() -> dict:
    """
    Return {synset_name: synset_object} for the root synsets.
    Downloads NLTK WordNet corpus if not already present.
    """
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        try:
            wn.synsets("test")
        except LookupError:
            print("[WordNet] Downloading corpus …")
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        roots = {}
        for name in ROOT_SYNSET_NAMES:
            try:
                roots[name] = wn.synset(name)
            except Exception:
                pass  # synset not found — skip gracefully
        print(f"[WordNet] Loaded {len(roots)} root synsets.")
        return roots
    except ImportError:
        print("[WordNet] NLTK not installed; skipping anchor enrichment.")
        return {}


def _wordnet_ancestors(synset, root_synsets: dict, max_depth: int = 20) -> list[str]:
    """
    Walk hypernym paths from `synset` upwards; return the names of any root
    synsets encountered.  Stops early once all roots are found or max_depth hit.
    """
    found: list[str] = []
    visited = set()
    queue = [(synset, 0)]
    while queue:
        s, depth = queue.pop(0)
        if depth > max_depth or s in visited:
            continue
        visited.add(s)
        for name, root in root_synsets.items():
            if s == root and name not in found:
                found.append(name.split(".")[0])   # e.g. "physical_entity"
        for hyp in s.hypernyms():
            queue.append((hyp, depth + 1))
    return found


def _camel_tokens(name: str) -> list[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [t.lower() for t in re.split(r"[^A-Za-z0-9]+", s) if t]


def resolve_anchors(entity_name: str, root_synsets: dict) -> list[str]:
    """
    Return the root-synset label(s) for an entity name.

    Each camelCase token is looked up as a WordNet noun; its hypernym chain is
    walked until a root synset is reached.  The result is deduplicated.
    """
    if not root_synsets:
        return []
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return []

    anchors: list[str] = []
    seen: set[str] = set()
    for token in _camel_tokens(entity_name):
        synsets = wn.synsets(token, pos=wn.NOUN)
        if not synsets:
            continue
        # Use the first (most common) noun synset
        for anc in _wordnet_ancestors(synsets[0], root_synsets):
            if anc not in seen:
                seen.add(anc)
                anchors.append(anc)
    return anchors


# ──────────────────────────────────────────────────────────────────────────────
# Entity description builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_entity_descriptions(
    json_data: dict,
    root_synsets: dict,
) -> list[dict]:
    """
    Return a list of {name, description, anchors} dicts for one JSON model.

    Description format (optimised for nomic-embed-text-v1.5 with search_document prefix):
        "<EntityName>: <readable name>. Measures: <obs types>. Relations: <assoc context>. Category: <root synsets>."
    The Relations field encodes the association names and partner names this entity
    participates in — critical for network-type ontologies where association structure
    carries the primary semantic information.
    """
    declared_obs = set(json_data.get("observables", []))
    entity_names = {
        (e.get("entityName") or e.get("name", ""))
        for e in json_data.get("entities", [])
    }

    # Build per-entity association index (handles both JSON schemas)
    assoc_index: dict[str, list[str]] = {}
    for assoc in json_data.get("associations", []):
        assoc_name   = assoc.get("associationName", "") or assoc.get("name", "")
        participants = assoc.get("associationParticipants", []) or assoc.get("participants", [])
        # Make the association name human-readable by splitting camelCase
        readable_assoc = re.sub(r"([a-z])([A-Z])", r"\1 \2", assoc_name)
        readable_assoc = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", readable_assoc).lower().strip()
        for p in participants:
            partners = [other for other in participants if other != p]
            # Produce a short phrase: "<assoc verb> (<partner, ...>)"
            partner_str = ", ".join(partners[:2])
            phrase = f"{readable_assoc} ({partner_str})" if partner_str else readable_assoc
            assoc_index.setdefault(p, []).append(phrase)

    records: list[dict] = []
    for ent in json_data.get("entities", []):
        name  = ent.get("entityName") or ent.get("name", "")
        attrs = ent.get("entityAttributes") or ent.get("attributes", [])

        readable = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        readable = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", readable).lower().strip()

        obs_types = [
            a.get("type", "") for a in attrs
            if a.get("type", "") and (
                a.get("type", "") in declared_obs
                or (not declared_obs and a.get("type", "") not in entity_names)
            )
        ]
        obs_text = ", ".join(obs_types[:8]) if obs_types else "unspecified"

        # Association context: up to 4 associations this entity participates in
        assoc_phrases = assoc_index.get(name, [])[:4]
        assoc_text    = "; ".join(assoc_phrases) if assoc_phrases else ""

        anchors     = resolve_anchors(name, root_synsets)
        anchor_text = ", ".join(anchors) if anchors else "entity"

        description = (
            f"search_document: {name}: {readable}. "
            f"Measures: {obs_text}. "
            + (f"Relations: {assoc_text}. " if assoc_text else "")
            + f"Category: {anchor_text}."
        )

        records.append({
            "name":        name,
            "description": description,
            "anchors":     anchors,
        })

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Matryoshka encoder
# ──────────────────────────────────────────────────────────────────────────────

class MatryoshkaEncoder:
    """
    Wraps nomic-ai/nomic-embed-text-v1.5.

    The model produces 768-d L2-normalised vectors.  We truncate and
    re-normalise to the requested Matryoshka dimension so that cosine
    similarity is still well-defined in the reduced space.
    """

    def __init__(self, model_name: str = MATRYOSHKA_MODEL, dim: int = DEFAULT_DIM):
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Matryoshka] Loading '{model_name}' on {device} …")
        self._model  = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
        )
        self._dim    = dim
        print(f"[Matryoshka] Using {dim}-d Matryoshka slice.")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return (N, dim) array of L2-normalised Matryoshka embeddings."""
        emb = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )                                         # (N, 768)
        emb = emb[:, : self._dim]                 # Matryoshka truncation
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return emb / norms                        # re-normalise


# ──────────────────────────────────────────────────────────────────────────────
# Multivariate Gaussian fitting (Ledoit-Wolf)
# ──────────────────────────────────────────────────────────────────────────────

def fit_gaussian(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit N(μ, Σ) to an (n_entities × dim) embedding matrix.

    Uses Ledoit-Wolf shrinkage estimator, which is well-conditioned even when
    n_entities << dim (common for small ontologies).  Falls back to diagonal
    regularisation if LW fails.

    Returns (mean, covariance) as numpy arrays.
    """
    mu = embeddings.mean(axis=0)
    n, d = embeddings.shape

    if n < 2:
        # Cannot fit covariance from a single point; return identity.
        return mu, np.eye(d) * REGULARIZE_ALPHA

    try:
        lw = LedoitWolf(assume_centered=False)
        lw.fit(embeddings)
        sigma = lw.covariance_
    except Exception:
        # Fallback: empirical covariance + diagonal regularisation
        sigma = np.cov(embeddings, rowvar=False)
        if sigma.ndim == 0:
            sigma = np.array([[float(sigma)]])
        sigma += REGULARIZE_ALPHA * np.eye(d)

    return mu, sigma


# ──────────────────────────────────────────────────────────────────────────────
# Mahalanobis distance and Bhattacharyya coefficient
# ──────────────────────────────────────────────────────────────────────────────

def mahalanobis_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """
    Mahalanobis distance between two distribution means using the pooled
    covariance:  d_M = sqrt( (μ₁-μ₂)ᵀ Σ_pooled⁻¹ (μ₁-μ₂) )
    """
    sigma_pooled = (sigma1 + sigma2) / 2.0
    diff = mu1 - mu2
    try:
        inv_sigma = np.linalg.inv(sigma_pooled)
    except np.linalg.LinAlgError:
        inv_sigma = np.linalg.pinv(sigma_pooled)
    dist_sq = float(diff @ inv_sigma @ diff)
    return float(math.sqrt(max(dist_sq, 0.0)))


def bhattacharyya(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> tuple[float, float]:
    """
    Closed-form Bhattacharyya distance and coefficient for two multivariate
    Gaussians N(μ₁,Σ₁) and N(μ₂,Σ₂).

    D_B = (1/8)(μ₁-μ₂)ᵀ Σ⁻¹(μ₁-μ₂) + (1/2) ln(det(Σ) / sqrt(det(Σ₁)·det(Σ₂)))
    where Σ = (Σ₁+Σ₂)/2.

    BC (Bhattacharyya coefficient) = exp(−D_B), range [0,1]:
        1  → identical distributions (perfect overlap)
        0  → non-overlapping distributions

    Returns (bhattacharyya_distance, bhattacharyya_coefficient).
    """
    sigma = (sigma1 + sigma2) / 2.0
    diff  = mu1 - mu2

    try:
        inv_sigma = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        inv_sigma = np.linalg.pinv(sigma)

    # Mahalanobis term
    mah_term = (1.0 / 8.0) * float(diff @ inv_sigma @ diff)

    # Log-determinant term (numerically stable via slogdet)
    _, logdet_sigma  = np.linalg.slogdet(sigma)
    _, logdet_sigma1 = np.linalg.slogdet(sigma1)
    _, logdet_sigma2 = np.linalg.slogdet(sigma2)
    logdet_term = 0.5 * (logdet_sigma - 0.5 * (logdet_sigma1 + logdet_sigma2))

    d_b = mah_term + logdet_term
    d_b = max(float(d_b), 0.0)   # numerical guard + ensure Python float
    bc  = math.exp(-d_b)

    return round(d_b, 6), round(bc, 6)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _load_all_models(domain_filter: str | None = None) -> dict[tuple[str, str], dict]:
    """
    Returns {(domain, model_name): json_data} for all input JSONs.
    """
    index: dict[tuple[str, str], dict] = {}
    pattern = os.path.join(INPUTS_DIR, "**", "*.json")
    for jf in sorted(glob.glob(pattern, recursive=True)):
        domain = os.path.basename(os.path.dirname(jf))
        if domain_filter and domain.lower() != domain_filter.lower():
            continue
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        name = data.get("modelName", os.path.splitext(os.path.basename(jf))[0])
        index[(domain, name)] = data
    return index


def run(
    dim: int = DEFAULT_DIM,
    domain_filter: str | None = None,
) -> None:
    os.makedirs(CHAR_DIR, exist_ok=True)

    # ── Step 1: load all models ──────────────────────────────────────────────
    print("=== Stage 3: Semantic Space Characterization ===\n")
    print(f"[S3] Matryoshka model : {MATRYOSHKA_MODEL}  (dim={dim})")
    models = _load_all_models(domain_filter)
    print(f"[S3] Models loaded    : {len(models)}")

    # ── Step 2: load WordNet root synsets ────────────────────────────────────
    root_synsets = _load_wordnet_roots()

    # ── Step 3: build descriptions for every entity in every model ───────────
    print("\n[S3] Building entity descriptions …")
    model_records: dict[tuple[str, str], list[dict]] = {}
    for (domain, name), data in models.items():
        recs = _build_entity_descriptions(data, root_synsets)
        model_records[(domain, name)] = recs
        print(f"  {domain:<14} {name[:50]:<50}  {len(recs)} entities")

    # ── Step 4: encode all descriptions ─────────────────────────────────────
    encoder  = MatryoshkaEncoder(dim=dim)
    all_texts: list[str] = []
    key_order: list[tuple[str, str, int]] = []   # (domain, name, idx_in_texts)

    for (domain, name), recs in model_records.items():
        start = len(all_texts)
        all_texts.extend(r["description"] for r in recs)
        key_order.append((domain, name, start))

    print(f"\n[S3] Encoding {len(all_texts)} entity descriptions …")
    all_embeddings = encoder.encode(all_texts)   # (total_entities, dim)

    # ── Step 5: fit per-model Gaussians ─────────────────────────────────────
    print("\n[S3] Fitting multivariate Gaussians (Ledoit-Wolf) …")
    gaussians: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    cloud_outputs: list[dict] = []

    for domain, name, start in key_order:
        recs = model_records[(domain, name)]
        n    = len(recs)
        emb  = all_embeddings[start : start + n]
        mu, sigma = fit_gaussian(emb)
        gaussians[(domain, name)] = (mu, sigma)

        # Attach embedding vectors to records for output
        entity_data = []
        for i, r in enumerate(recs):
            entity_data.append({
                "name":    r["name"],
                "anchors": r["anchors"],
                "embedding": emb[i].tolist(),
            })

        cloud = {
            "domain":        domain,
            "model_name":    name,
            "n_entities":    n,
            "embedding_dim": dim,
            "mean":          mu.tolist(),
            "covariance_diag": np.diag(sigma).tolist(),   # full covariance is large; store diagonal only
            "entities":      entity_data,
        }
        cloud_outputs.append(cloud)

        # Save per-model cloud
        dom_dir = os.path.join(CHAR_DIR, domain)
        os.makedirs(dom_dir, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
        with open(os.path.join(dom_dir, f"{safe_name}_cloud.json"), "w",
                  encoding="utf-8") as f:
            json.dump(cloud, f, indent=2, ensure_ascii=False)

    # ── Step 6: ALL pairwise distances (within-domain + cross-domain) ────────
    print("\n[S3] Computing pairwise Mahalanobis + Bhattacharyya distances …")
    print("     (within-domain AND cross-domain)\n")

    # Build a flat sorted list of all (domain, name) keys
    all_keys = sorted(gaussians.keys())   # sorted by (domain, name)

    pairwise_results: list[dict] = []

    # ── 6a: within-domain ─────────────────────────────────────────────────── #
    by_domain: dict[str, list[str]] = {}
    for (domain, name) in all_keys:
        by_domain.setdefault(domain, []).append(name)

    print(f"{'Type':<14} {'Domain A':<14} {'Model A':<36} {'Domain B':<14} {'Model B':<36} "
          f"{'Mahal':>7} {'BC':>7} {'BD':>7}")
    print("-" * 135)

    for domain, names in sorted(by_domain.items()):
        for name_a, name_b in combinations(sorted(names), 2):
            mu_a, sig_a = gaussians[(domain, name_a)]
            mu_b, sig_b = gaussians[(domain, name_b)]
            d_mah       = mahalanobis_distance(mu_a, sig_a, mu_b, sig_b)
            d_b, bc     = bhattacharyya(mu_a, sig_a, mu_b, sig_b)
            pairwise_results.append({
                "comparison_type":           "within_domain",
                "domain_a":                  domain,
                "domain_b":                  domain,
                "model_a":                   name_a,
                "model_b":                   name_b,
                "n_entities_a":              len(model_records[(domain, name_a)]),
                "n_entities_b":              len(model_records[(domain, name_b)]),
                "embedding_dim":             dim,
                "mahalanobis_distance":      round(d_mah, 4),
                "bhattacharyya_distance":    d_b,
                "bhattacharyya_coefficient": bc,
            })
            print(
                f"{'within':<14} {domain:<14} {name_a[:35]:<36} {domain:<14} {name_b[:35]:<36} "
                f"{d_mah:>7.3f} {bc:>7.4f} {d_b:>7.4f}"
            )

    # ── 6b: cross-domain ──────────────────────────────────────────────────── #
    print()
    domain_keys = list(by_domain.keys())
    for i, dom_a in enumerate(sorted(domain_keys)):
        for dom_b in sorted(domain_keys)[i + 1:]:
            for name_a in sorted(by_domain[dom_a]):
                for name_b in sorted(by_domain[dom_b]):
                    mu_a, sig_a = gaussians[(dom_a, name_a)]
                    mu_b, sig_b = gaussians[(dom_b, name_b)]
                    d_mah       = mahalanobis_distance(mu_a, sig_a, mu_b, sig_b)
                    d_b, bc     = bhattacharyya(mu_a, sig_a, mu_b, sig_b)
                    pairwise_results.append({
                        "comparison_type":           "cross_domain",
                        "domain_a":                  dom_a,
                        "domain_b":                  dom_b,
                        "model_a":                   name_a,
                        "model_b":                   name_b,
                        "n_entities_a":              len(model_records[(dom_a, name_a)]),
                        "n_entities_b":              len(model_records[(dom_b, name_b)]),
                        "embedding_dim":             dim,
                        "mahalanobis_distance":      round(d_mah, 4),
                        "bhattacharyya_distance":    d_b,
                        "bhattacharyya_coefficient": bc,
                    })
                    print(
                        f"{'cross':<14} {dom_a:<14} {name_a[:35]:<36} {dom_b:<14} {name_b[:35]:<36} "
                        f"{d_mah:>7.3f} {bc:>7.4f} {d_b:>7.4f}"
                    )

    # ── Step 7: summary statistics ────────────────────────────────────────── #
    within = [r for r in pairwise_results if r["comparison_type"] == "within_domain"]
    cross  = [r for r in pairwise_results if r["comparison_type"] == "cross_domain"]

    def _stats(rows: list[dict], key: str) -> tuple[float, float, float, float]:
        vals = [r[key] for r in rows]
        return min(vals), float(np.mean(vals)), float(np.median(vals)), max(vals)

    print("\n" + "=" * 80)
    print("SUMMARY: Within-Domain vs Cross-Domain Distances")
    print("=" * 80)
    print(f"\n{'Metric':<32} {'Min':>8} {'Mean':>8} {'Median':>8} {'Max':>8}")
    print("-" * 64)
    for label, rows in [("Within-domain", within), ("Cross-domain", cross)]:
        mn, mu, med, mx = _stats(rows, "mahalanobis_distance")
        print(f"{label + ' Mahalanobis':<32} {mn:>8.3f} {mu:>8.3f} {med:>8.3f} {mx:>8.3f}")
        mn, mu, med, mx = _stats(rows, "bhattacharyya_coefficient")
        print(f"{label + ' BC (overlap)':<32} {mn:>8.4f} {mu:>8.4f} {med:>8.4f} {mx:>8.4f}")
        print()

    # Per-domain within-domain summary
    print("Per-domain within-domain mean Mahalanobis / mean BC:")
    print(f"  {'Domain':<14} {'N pairs':>8} {'Mean Mahal':>12} {'Mean BC':>10}")
    print("  " + "-" * 48)
    for domain in sorted(by_domain.keys()):
        rows_d = [r for r in within if r["domain_a"] == domain]
        if not rows_d:
            continue
        m_mah = float(np.mean([r["mahalanobis_distance"] for r in rows_d]))
        m_bc  = float(np.mean([r["bhattacharyya_coefficient"] for r in rows_d]))
        print(f"  {domain:<14} {len(rows_d):>8} {m_mah:>12.3f} {m_bc:>10.4f}")

    # ── Step 8: save outputs ──────────────────────────────────────────────── #
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(pairwise_results, f, indent=2, ensure_ascii=False)

    # Separate CSVs for clarity
    out_within = os.path.join(CHAR_DIR, "distances_within_domain.csv")
    out_cross  = os.path.join(CHAR_DIR, "distances_cross_domain.csv")
    out_all    = os.path.join(CHAR_DIR, "distances_all.csv")

    for subset, path in [(within, out_within), (cross, out_cross), (pairwise_results, out_all)]:
        if subset:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=subset[0].keys())
                w.writeheader()
                w.writerows(subset)

    print(f"\n[S3] JSON (all)          -> {OUT_JSON}")
    print(f"[S3] CSV within-domain   -> {out_within}")
    print(f"[S3] CSV cross-domain    -> {out_cross}")
    print(f"[S3] CSV all pairs       -> {out_all}")
    print(f"[S3] Per-model clouds    -> {CHAR_DIR}/<domain>/")
    print(f"\n=== Stage 3 complete: {len(within)} within-domain + {len(cross)} cross-domain "
          f"= {len(pairwise_results)} total pairs. ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: Matryoshka embedding characterization of ontology semantic space."
    )
    parser.add_argument(
        "--dim", type=int, default=DEFAULT_DIM,
        help=f"Matryoshka slice dimension (default {DEFAULT_DIM}). "
             "Valid values for nomic-embed-text-v1.5: 64, 128, 256, 512, 768.",
    )
    parser.add_argument(
        "--domain", type=str, default=None,
        help="Restrict to one domain (e.g. Automobile).",
    )
    args = parser.parse_args()
    run(dim=args.dim, domain_filter=args.domain)
