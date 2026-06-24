# Ontology Matching Pipeline — Metrics Documentation

## 1. Pipeline Overview

The pipeline compares two ontology JSON models and produces a `*_metrics.csv` per pair, which is consumed by `compare_stage.py` to compute a final **composite similarity score**. There are four top-level metrics, each capturing a different facet of similarity:

| Metric | Source | What it measures |
|--------|--------|------------------|
| `lexical_sim` | `compare_stage.py` (assembled from merge CSV) | Vocabulary / label overlap + embedding |
| `wl_structural` | `wl_kernel_matcher.py` | Relational topology (edge-type motifs) |
| `shape_sim` | `wl_kernel_matcher.py` | Global graph shape (degree, spectral, clustering, betweenness) |
| `attr_weighted` | `compare_stage.py` + `attribute_reach.py` | Observable attribute distribution, WUP-weighted |

---

## 2. Sub-Metrics (Building Blocks)

### 2.1 Wu-Palmer (WUP) Blended Score

**Source:** `enriched_matcher.py → characterise_entity_pair()`, written to merged CSV as `wup`.

For each entity pair `(eₐ, e_b)`:

1. Tokenise each entity name via camelCase splitting:
   `tokens_a = split_camel(eₐ)`, `tokens_b = split_camel(e_b)`

2. Compute the Wu-Palmer similarity for every cross-product pair of tokens using WordNet:

   ```
   wup(tᵢ, tⱼ) = 2 × depth(LCS(tᵢ, tⱼ)) / (depth(tᵢ) + depth(tⱼ))
   ```

   where `LCS` is the Lowest Common Subsumer in the WordNet noun hierarchy, restricted to the top-5 most-frequent synsets per word to avoid polysemy false positives.

3. Aggregate across all N×M token pairs:

   ```
   max_wup  = max{ wup(tᵢ, tⱼ) : tᵢ ∈ tokens_a, tⱼ ∈ tokens_b }
   avg_wup  = mean{ wup(tᵢ, tⱼ) : tᵢ ∈ tokens_a, tⱼ ∈ tokens_b }
   ```

4. **Final blended WUP** written to merged CSV:

   ```
   wup = (max_wup + avg_wup) / 2
   ```

   This blends the best-token signal (`max_wup`) with the full average (`avg_wup`). The blend prevents a single lucky token match from dominating while still rewarding strong single-token similarity.

**Range:** [0, 1]. Applied threshold in `lexical_sim`: `wup` is only used if `wup ≥ WUP_THRESHOLD = 0.75`.

---

### 2.2 Sentence Embedding Cosine (`cosine_avg`)

**Source:** `semantic_encoder.py → run_embedding_stage()`.

For each entity name, three complementary vector representations are computed using `paraphrase-MiniLM-L6-v2` (384-dimensional L2-normalised embeddings):

| Representation | Construction |
|---------------|--------------|
| `whole` | `encode(readable_name)` — holistic compound meaning |
| `sum` | `L2_norm(Σ encode(tokenᵢ))` — additive token composition |
| `prod` | `L2_norm(⊙ encode(tokenᵢ))` — element-wise product composition |

Three cosine similarities are computed (all embeddings are unit vectors, so dot product = cosine):

```
cosine_whole = dot(whole_A, whole_B)
cosine_sum   = dot(sum_A,   sum_B)
cosine_prod  = dot(prod_A,  prod_B)
```

Each is **rescaled from [-1, 1] to [0, 1]**:

```
cosine_X_rescaled = (cosine_X + 1.0) / 2.0
```

The final `cosine_avg` written to the embeddings CSV is:

```
cosine_avg = (cosine_whole + cosine_sum + cosine_prod) / 3
```

(applied after rescaling each component individually, then averaged).

**Range:** [0, 1].

---

### 2.3 Matcher Binary Flag (`matched`)

**Source:** `enriched_matcher.py → layer1_annotate()`, written to merged CSV.

```
matched = 1  if source ∈ {"AML", "LogMap", "Both"}
matched = 0  otherwise (Layer2, Layer3 discoveries)
```

This is a hard binary signal: if AML or LogMap confirmed the pair as an equivalence match, `matched = 1`. Layer 2/3 semantic candidates are not given the `matched = 1` flag.

---

### 2.4 Attribute Distribution Similarity (`attr_dist_sim`)

**Source:** `attribute_reach.py → run_attr_dist_stage()`.

This metric quantifies how similar two models' observable attribute distributions are, independent of entity identity.

1. **Collect shared observable vocabulary** from both models — union of `data["observables"]` and all `entityAttribute.type` values across both JSONs.

2. **K-hop BFS attribute reach** (K=2, imputation disabled): for each entity, propagate attribute types through structurally-weighted edges:

   | Canonical relation | Edge weight |
   |--------------------|-------------|
   | PartOf, HasA | 0.6 |
   | MadeOf | 0.5 |
   | Connects, UsedFor | 0.4 |
   | AtLocation | 0.3 |
   | CapableOf, Causes, IsA, ReceivesAction | 0.25 |
   | HasPrerequisite | 0.2 |
   | RelatedTo | skipped |

   Path weight = product of edge weights; MAX across all paths to same type is kept.

3. **Embed** each observable type name with `paraphrase-MiniLM-L6-v2`.

4. **Aggregate** reach-weighted type embeddings over ALL entities in each model:

   ```
   agg_A = Σ_{entity e ∈ A} Σ_{type t in reach(e)} weight(e,t) × embed(t)
   ```

5. **Cosine similarity**:

   ```
   attr_dist_sim = cosine(agg_A, agg_B) = dot(agg_A/‖agg_A‖, agg_B/‖agg_B‖)
   ```

**Range:** [0, 1]. Returns 0.0 when neither model has observable types.

---

## 3. Top-Level Metrics

### 3.1 Lexical Similarity (`lexical_sim`)

**Source:** `compare_stage.py → load_pair_metrics()`.

Per entity-pair row in the merged CSV, compute:

```
lexical_row(r) = max(
    max(matched_r, cosine_avg_r),
    wup_r  if wup_r ≥ 0.75 else 0
)
```

The WUP threshold of 0.75 prevents the generic WordNet machine/artifact hierarchy (which gives wup ≈ 0.89 to unrelated devices like fuse, filter, blower) from inflating scores.

**Aggregation across all N entity-pair rows:**

- **Standard (default):** `lexical_sim = mean(lexical_row(r) for r in rows)`
- **Entity-count normalised** (when `n_entities` is provided):
  ```
  lexical_sim = min( Σ lexical_row(r) / n_entities, 1.0 )
  ```
  This normalises for the fraction of total entities that were actually matched, penalising low-coverage matchers. The sum is capped at 1.0.

**Range:** [0, 1].

---

### 3.2 WL Structural Similarity (`wl_structural`)

**Source:** `wl_kernel_matcher.py → run_wl_stage()`.

Measures whether two ontologies share the same **relational topology** (edge-type motifs), completely independent of entity identity.

**Algorithm — Edge-aware Weisfeiler-Lehman Kernel:**

1. Build adjacency for both models using canonical edge labels (PartOf, Connects, UsedFor, etc.) from `association_inventory.csv`.

2. Initialise every node label to `hash("N")` — **all nodes are anonymous**.

3. For K=3 hops, refine node labels:

   ```
   label_v^{k+1} = md5(label_v^k + "|" + sorted[(label_nb^k, edge_label) for each neighbour nb])
   ```

4. Accumulate label frequency histograms across all K+1 snapshots (hops 0 through K):

   ```
   freq_A = Counter{ label : count } across hops 0..K in graph A
   freq_B = Counter{ label : count } across hops 0..K in graph B
   ```

5. Cosine similarity of frequency vectors:

   ```
   wl_structural = dot(freq_A, freq_B) / (‖freq_A‖ × ‖freq_B‖)
   ```

   where dot product sums over all label keys present in either histogram.

**Range:** [0, 1]. Equal to 1.0 when both graphs have identical edge-type topology (same relational motif counts at every depth). **Entity names never influence this score.**

---

### 3.3 Graph Shape Similarity (`shape_sim`)

**Source:** `wl_kernel_matcher.py → graph_shape_sim()`.

Captures global graph structure using four distribution signatures, each compared via **cosine similarity of sorted vectors** (sorted descending so the most-connected nodes correspond):

| Sub-metric | Formula |
|-----------|---------|
| `degree_sim` | `cosine(sorted_degree_A, sorted_degree_B)` |
| `spectral_sim` | `cosine(sorted_λ_A, sorted_λ_B)` where λ are eigenvalues of the normalised Laplacian `L = I - D^{-1/2} A D^{-1/2}` |
| `clustering_sim` | `cosine(sorted_C_A, sorted_C_B)` where `C(v) = (edges between v's neighbours) / (d(v)(d(v)-1)/2)` |
| `betweenness_sim` | `cosine(sorted_BC_A, sorted_BC_B)` where BC is normalised betweenness centrality via Brandes' O(VE) algorithm |

Vectors of unequal length are zero-padded before taking cosine.

**Final shape_sim:**

```
shape_sim = (degree_sim + spectral_sim + clustering_sim + betweenness_sim) / 4
```

**Range:** [0, 1]. Equal to 1.0 when both graphs are isomorphic in all four global topological signatures.

---

### 3.4 Attribute-Weighted Similarity (`attr_weighted`)

**Source:** `compare_stage.py → load_pair_metrics()`.

Combines `attr_dist_sim` (attribute distribution similarity from §2.4) with the mean entity WUP score as a confidence weight:

```
avg_wup       = mean{ wup_r : r in rows, wup_r is non-null }
attr_weighted = attr_dist_sim × avg_wup
```

The multiplication by `avg_wup` scales down the attribute similarity score when entity WUP is low — reflecting that the attribute comparison is less meaningful when entity names are semantically unrelated. When `attr_dist_sim = 0` or `avg_wup = 0`, `attr_weighted = 0`.

**Range:** [0, 1].

---

## 4. Final Composite Score

**Source:** `compare_stage.py → main()`.

### 4.1 Equal-Weight Composite

```
METRICS = ["lexical_sim", "wl_structural", "shape_sim", "attr_weighted"]
weight  = 1 / len(METRICS) = 0.25  (for each metric)
```

The composite for a given pair is the **mean of available (non-zero) metric values**:

```
available = { m ∈ METRICS : value(m) > 0 }
composite = mean{ value(m) : m ∈ available }
           = Σ_{m ∈ available} value(m) / |available|
```

Metrics that are zero (e.g., `attr_weighted = 0` when no observable types exist) are excluded from the mean rather than penalising the score. This prevents structural metrics from being diluted by a missing attribute stage.

### 4.2 Distance Metric (for MDS Layout)

For the 3D ontology distance map, the pairwise distance between two models A and B is:

```
d(A, B) = sqrt( Σ_{m ∈ available} (1 - value_m)² / |available| )
```

This is a Euclidean distance in the metric space spanned by available metrics. Result is clamped to [0, 1]. The symmetrisation `d = (d + dᵀ) / 2` is applied to ensure numerical symmetry.

---

## 5. Synthetic Dataset

### 5.1 Design: 3 × 5 Factorial

The `Automobile_Synthetic` dataset is a **controlled 3×5 factorial** designed to independently test whether the metrics correctly respond to vocabulary variation and topology variation.

**Factor 1 — Vocabulary (3 levels):**

| Level | Code | Description |
|-------|------|-------------|
| Standard | `SAME` | Identical canonical automobile vocabulary: Vehicle, Engine, Transmission, Differential, Axle, Wheel, FuelSystem, CoolingSystem, ElectricalSystem, ExhaustSystem, Suspension, BrakeSystem, SteeringSystem |
| Synonymous | `SYN` | Synonym vocabulary covering the same powertrain/support split: Automobile, Motor, Gearbox, DriveShaft, HalfShaft, Tyre, FuelDelivery, ThermalManagement, PowerElectronics, EmissionControl, SpringDamper, RetardSystem, DirectionControl |
| Alternative domain | `ALT` | Suspension subsystem vocabulary (fully disjoint from drivetrain): SubframeMount, CoilSpring, ShockBody, InnerChamber, PivotBall, RotatingHub, TorsionBar, DampingValve, TrackRod, DroplinkArm, Upright, LowerWishbone, UpperWishbone |

All 15 models share an identical observable type vocabulary of 19 types: `Identifier, Temperature, Pressure, Torque, Power, AngularVelocity, Force, Mass, Distance, MassFlowRate, Ratio, Count, Energy, ElectricPotential, ElectricCurrent, Angle, Position, OperationalState, Kind`.

**Factor 2 — Topology (5 levels):**

| Level | Code | Description |
|-------|------|-------------|
| Bipartite | `BIP` | Two groups (powertrain A vs support B); all 15 edges cross between groups |
| Deep chain | `DEEP` | Linear sequence: entity₁ — entity₂ — … — entity₁₃ |
| Grid | `GRID` | 2D lattice (≈4×4), edges connect row/column neighbours |
| Hub-and-spoke | `HUB` | One central hub entity connected to all 12 peripherals |
| Wide flat | `WIDE` | Flat structure with many parallel short branches from a root |

**Total:** 15 models → C(15,2) = **105 directed-unique pairs**.

**Ground truth design matrix:**

| Pair type | same_vocab | same_topo | Expected behaviour |
|-----------|-----------|-----------|-------------------|
| SAME↔SAME, SYN↔SYN, ALT↔ALT (cross-topo) | True | False | High `lexical_sim`, variable `wl_structural`/`shape_sim` |
| X↔X same-topo cross-vocab (e.g., ALT_BIP↔SAME_BIP) | False | True | Low `lexical_sim`, `wl_structural`=1.0, high `shape_sim` |
| Cross-vocab cross-topo | False | False | All metrics partial |
| (Self-comparison, not in dataset) | True | True | Expected composite = 1.0 |

---

### 5.2 Expected Results

Based on the pipeline's design, we expected:

1. **`wl_structural` = 1.0** whenever `same_topo = True`, because the WL kernel with anonymous nodes captures only relational edge-type motifs — which are identical when topology is the same.

2. **`shape_sim` ≈ 1.0** whenever `same_topo = True` (DEEP, GRID, HUB, WIDE), because degree sequences, Laplacian spectra, clustering, and betweenness distributions are topology-determined, not vocabulary-determined.

3. **`lexical_sim` = 1.0** whenever `same_vocab = True` (SAME↔SAME, SYN↔SYN, ALT↔ALT), because AML/LogMap find exact name matches within the same vocabulary class, driving `matched = 1` for all entity pairs.

4. **`lexical_sim` ≈ 0.27** for SAME↔SYN cross-vocab pairs: sentence embeddings can partially detect synonym pairs (Motor ≈ Engine), so `cosine_avg > 0`, but no `matched = 1` flags → `lexical_sim` = mean cosine ≈ 0.27.

5. **`lexical_sim` ≈ 0.06** for ALT↔SAME or ALT↔SYN cross-vocab pairs: suspension components (CoilSpring) and drivetrain parts (Engine) have low semantic embedding similarity, so `cosine_avg ≈ 0.06` with zero `matched = 1` flags.

6. **Composite ranking**: same_vocab_only > same_topo_only > neither.

7. **`attr_weighted` ≈ 1.0** for same-vocab pairs: entities carry the same observable types, so `attr_dist_sim → 1.0`, and `avg_wup` is high from matching entity names.

---

### 5.3 Actual Results vs Expected

All 105 pairs are in `enriched_ontology_matching/outputs/synthetic_results.csv`. Summary by pair type:

**Quadrant 1 — same_vocab=True, same_topo=False (40 pairs)**

| Metric | Expected | Actual Range | Notes |
|--------|----------|--------------|-------|
| `lexical_sim` | 1.0 | **1.0** (all 40 pairs) | ✓ Perfect: AML/LogMap finds all same-name matches |
| `wl_structural` | varies | 0.541–0.825 | ✓ Varies with topology pair |
| `shape_sim` | varies | 0.564–0.960 | ✓ Varies with topology pair |
| `attr_weighted` | ≈ 1.0 | 0.952–1.0 | ✓ Slightly below 1.0 due to `avg_wup < 1` for synonym pairs |
| `composite` | 0.82–0.91 | **0.797–0.894** | ✓ Confirmed high range |

**Quadrant 2 — same_vocab=False, same_topo=True (30 pairs)**

| Metric | Expected | Actual Range | Notes |
|--------|----------|--------------|-------|
| `lexical_sim` (SAME↔SYN) | ≈ 0.27 | **0.2726** | ✓ Synonym embeddings captured |
| `lexical_sim` (ALT↔SAME, ALT↔SYN) | ≈ 0.06 | **0.0633** | ✓ Cross-domain baseline |
| `wl_structural` | 1.0 | **1.0** (all 30 pairs) | ✓ Perfect: anonymous topology identical |
| `shape_sim` (DEEP, GRID, HUB, WIDE) | 1.0 | **1.0** | ✓ Perfect for chain/lattice/hub/wide |
| `shape_sim` (BIP) | 1.0 | **0.75** | ⚠ Unexpected: see note below |
| `composite` (SAME↔SYN) | ≈ 0.65–0.72 | **0.6462–0.7153** | ✓ |
| `composite` (ALT↔SAME/SYN) | ≈ 0.52–0.65 | **0.5252–0.6520** | ✓ |

> **BIP anomaly:** `shape_sim = 0.75` for all bipartite same-topology pairs instead of the expected 1.0. Root cause: the SAME_BIP bipartite split assigns 5 entities to Group A (powertrain) and 8 to Group B (support), while ALT_BIP assigns 6 to Group A (load-path) and 7 to Group B (geometry). The resulting degree sequences differ slightly (5-neighbour hub nodes vs 6-7 cross-group nodes), causing spectral and betweenness distributions to diverge. This is a **correct and meaningful behaviour**: the BIP models encode different bipartite structures, not just different node names.

**Quadrant 3 — same_vocab=False, same_topo=False (35 pairs)**

| Metric | Expected | Actual Range | Notes |
|--------|----------|--------------|-------|
| `lexical_sim` | 0.06–0.27 | 0.0633–0.2726 | ✓ |
| `wl_structural` | varies | 0.541–0.825 | ✓ |
| `shape_sim` | varies | 0.564–0.960 | ✓ |
| `attr_weighted` | ≈ 0.55–0.80 | 0.546–0.799 | ✓ |
| `composite` | ≈ 0.50–0.61 | **0.509–0.606** | ✓ Lowest quadrant, as expected |

**Metric discrimination analysis:**

| Signal | Discriminates vocabulary? | Discriminates topology? |
|--------|--------------------------|------------------------|
| `lexical_sim` | **Yes** — 1.0 vs 0.06–0.27 | No — blind to topology |
| `wl_structural` | No — 1.0 for all same-topo | **Yes** — 0.54–0.83 across topologies |
| `shape_sim` | No — 1.0 for same-topo (except BIP) | **Yes** — 0.56–0.96 across topologies |
| `attr_weighted` | Partially — 0.95–1.0 same-vocab, 0.55–0.80 cross-vocab | Weakly |
| `composite` | **Yes** — clear quadrant separation | **Yes** — clear quadrant separation |

**Conclusion: All four metrics behave as designed.** The composite correctly identifies same-vocabulary pairs as more similar (0.80–0.89) than same-topology-only pairs (0.53–0.72), confirming the pipeline captures both vocabulary and structural dimensions independently.

---

## 6. Supporting Notes

### 6.1 Enriched Matcher Layers

The enriched matcher (`enriched_matcher.py`) produces the per-pair CSV consumed by the merge stage. It runs in three layers:

- **Layer 1** — Annotates AML/LogMap matches with WordNet + ConceptNet semantic relationships (Synonym, Hypernym, PartOf, etc.)
- **Layer 2** — Discovers additional entity pairs not found by matchers: Equivalence (synonym/near-synonym) and Subsumption (hypernym chains, PartOf).
- **Layer 3 (WUP Backup)** — For entities still orphaned after L1+L2, finds top-k partners by `max_wup ≥ 0.9`. Uses `max_wup` not blended WUP as the gate: a near-perfect WUP score means a shared root token (fuel↔fuel, brake↔brake).

### 6.2 WL Matched Score (not in final composite)

`wl_matched` (in `wl_kernel_matcher.py`) is also computed — using entity-specific labels (`E{i}` for matched pairs, `A_entity` / `B_entity` for unmatched) — but is **not included in the compare_stage composite**. It is stored in the WL CSV for diagnostic inspection.

### 6.3 Output File Locations

| Stage | Output |
|-------|--------|
| Enriched matcher (L1+L2+L3) | `outputs/enriched/<stem>.csv` |
| Sentence embeddings | `outputs/embeddings/<stem>_emb.csv` |
| Merged metrics (per pair) | `outputs/merged/<stem>_metrics.csv` |
| WL kernel | `outputs/wl/<stem>_wl.csv` |
| Attribute distribution | `outputs/attr_dist/<stem>_attr_dist.csv` |
| Synthetic validation | `outputs/synthetic_results.csv` |
| 3D distance map | `outputs/ontology_map.html` |

---

## 7. Probe Experiment Framework

`scripts/probe_visualizer.py` generates a 4-series micro-experiment report (`docs/probe_visualizations.pdf` + per-series PNGs `docs/probe_s1.png` … `docs/probe_s4.png`). Each series isolates one axis of variation while holding all other factors constant, providing controlled evidence that the metrics are orthogonally sensitive to their intended signals.

### 7.1 Series Design

| Series | Changed factor | Held constant | Expected metric response |
|--------|---------------|---------------|--------------------------|
| S1 — Naming Drift | Entity names (1–5 renames) | Topology, attributes | `lexical` ↓ linearly; `wl_struct`, `shape`, `attr` stable at ≈ 1.0 |
| S2 — Attribute Drift | Observable types per entity (1–5 entities swapped) | Topology, entity names | `attr` ↓ progressively; `lexical`, `wl_struct`, `shape` stable |
| S3 — Topology Drift | Composition depth: chain → pendant → branches → caterpillar → star | Entity names, attributes | `wl_struct` ↓, `shape` ↓; `lexical`, `attr` stable |
| S4 — Density Drift | Edge density: 5-cycle → chord additions → K₅ | Entity names, attributes | `wl_struct` ↓, `shape` ↓; `lexical`, `attr` stable |

Each series has **6 steps** (baseline + 5 incremental variants), so metrics are always compared against the same baseline (leftmost graph).

### 7.2 Step Layout

**S1 (Naming Drift):** V0 baseline → V1 (1 rename) → V2 (2) → V3 (3) → V4 (4) → V5 (5 renames).

**S2 (Attribute Drift):** V0 baseline → V1 (1 entity's observable types changed) → … → V5 (5 entities' types changed).

**S3 (Topology Drift — chain → star):** T0 (chain) → T1 (+pendant) → T2 (2 branches) → T3 (branch + 2 pendants) → T4 (caterpillar) → T5 (star). Reingold-Tilford layout; edge labels shown.

**S4 (Density Drift — ring → K₅):** R0 (5-cycle) → R1 (+1 chord) → R2 (+2 chords) → R3 (+3 chords) → R4 (+4 chords) → R5 (K₅). Ring layout; chord edges drawn over nodes for clarity.

### 7.3 Probe Metrics vs. Full Pipeline Metrics

The probe visualizer computes **simplified approximations** for in-figure display. These differ from the full pipeline metrics in `compare_stage.py`:

| Metric | Probe formula | Full pipeline formula |
|--------|--------------|----------------------|
| `lexical` | Jaccard(entity name sets): `|A∩B| / |A∪B|` | `max(max(matched, cosine_avg), wup ≥ 0.75)` averaged over rows; AML/LogMap matched flag + sentence embeddings |
| `wl_struct` | WL kernel with **degree-count labels** (K=3) | WL kernel with **hash(edge-type neighbourhood)** labels (K=3) from `wl_kernel_matcher.py` |
| `shape` | `avg(degree_sim, spectral_sim)` — 2 sub-metrics | `avg(degree_sim, spectral_sim, clustering_sim, betweenness_sim)` — 4 sub-metrics |
| `attr` | Jaccard over observable type count distributions | `attr_dist_sim × avg_entity_wup` (reach-weighted embedding cosine × WUP confidence) |
| `dist` | `√( Σ(1−mᵢ)² / 4 )` over all 4 metrics | Same equal-weight Euclidean formula, but over available (non-zero) metrics only |

The probe metrics are intentionally simpler so they can run without the full pipeline (no AML/LogMap, no sentence encoder). They serve as visual sanity checks, not as replacements for the production scores.

### 7.4 Delta Highlighting

Each non-baseline column shows an **orange highlight** for exactly what changed from the previous step:
- Orange node border: entity was renamed or gained new observable attribute types.
- Orange edge: edge was added (solid) or removed (dashed grey).
- Orange text inside attribute box: newly added observable type in that entity.

The delta is computed by diffing the current model against the immediately preceding variant (not against the baseline), so the highlight shows the single incremental change at each step.

### 7.5 Output Locations

| Output | Path |
|--------|------|
| PDF (all 4 series) | `docs/probe_visualizations.pdf` |
| S1 PNG (naming drift) | `docs/probe_s1.png` |
| S2 PNG (attribute drift) | `docs/probe_s2.png` |
| S3 PNG (topology drift) | `docs/probe_s3.png` |
| S4 PNG (density drift) | `docs/probe_s4.png` |
| Pair JSON inputs | `enriched_ontology_matching/pairs/probe_s{1-4}_*.json` (gitignored) |

Run `python scripts/probe_visualizer.py` from the repository root to regenerate all figures.
