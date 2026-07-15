# Enriched Ontology Matching Pipeline

This pipeline compares pairs of conceptual (entity-relationship) models by combining two independent measurement approaches: **semantic/lexical matching** — WordNet, ConceptNet, and sentence-transformer embeddings, seeded by the AML and LogMap structural matchers — which establishes and characterises entity-level correspondences, and **graph-theoretic structural comparison** — a Weisfeiler-Lehman relational kernel, degree/spectral/clustering/betweenness graph-shape metrics, and a K-hop attribute-reachability comparison — which scores topology and attribute content without ever looking at what anything is named. The four resulting scores are combined into a per-pair composite similarity and distance, producing a domain distance summary JSON and an interactive distance map.

The primary use case is **within-domain analysis**: run all pairwise comparisons for a set of ontologies that share the same domain, then read the average pairwise distance between them as a JSON summary. Cross-domain comparisons and the interactive distance map are secondary outputs.

---

## Pipeline Overview

Every ontology pair goes through two tracks that never share signal, then a final assembly step.

**Track A — Entity-level semantic matching** (produces `lexical_sim`; entity names and their attribute-type vocabulary are the whole input)

| Step | Module | What happens |
|------|--------|--------------|
| A1 | `aml_runner.py` / `logmap_runner.py` | AML and LogMap independently propose candidate entity correspondences; results are merged and deduplicated, with `matched=1` wherever either (or both) confirms a pair |
| A2 | `enriched_matcher.py` — Layer 1 (Characterise) | Every AML/LogMap match is annotated with a WordNet + ConceptNet relationship label (Synonym, Hypernym, PartOf, …) and a blended Wu-Palmer score (`wup`); spurious matches (same-entity attribute pairs, parent-name-embedded PartOf pairs) are filtered out |
| A3 | `enriched_matcher.py` — Layer 2 (Discover) | For entities neither matcher found, the same WordNet/ConceptNet token- and compound-level analysis proposes new Equivalence / Subsumption / ConceptNet-relation candidates |
| A4 | `enriched_matcher.py` — Layer 3 (WUP backup) | Any entity still without a candidate partner is paired with its top-k WordNet WUP matches (`max_wup ≥ 0.9`) — a purely lexical rescue for entities AML/LogMap/WordNet-discovery all missed |
| A5 | `attribute_reach.py` — Layer 3b (attribute-type backup) | Any entity *still* orphaned is paired via attribute-type embedding similarity (`type_embed_sim ≥ 0.5`) instead of name — rescues entities with matching observable types but unrelated names |
| A6 | `semantic_encoder.py` | Sentence-embedding cosine similarity (`cosine_avg`) computed for every candidate pair, built from attribute-type tokens (entity-name tokens as fallback) |
| A7 | `attribute_reach.py` — `run_type_embed_stage` | Attribute-type embedding similarity (`type_embed_sim`) computed for every candidate pair via K-hop attribute reach |
| A8 | `merge_stage.py` | Joins A1–A7 into one row per candidate pair; `compare_stage.py` assembles `lexical_sim` from `matched`/`wup` |

**Track B — Whole-model graph-theoretic + attribute-reach comparison** (produces `wl_structural`, `shape_sim`, `attr_weighted`) — computed directly from each model's association graph; **entity names never enter this computation**

| Step | Module | What happens |
|------|--------|--------------|
| B1 | `wl_kernel_matcher.py` | Both models' association graphs are anonymised (every node initialised to `hash("N")`, canonical relation type kept on every edge) and compared with a K=3-hop Weisfeiler-Lehman kernel → `wl_structural` |
| B2 | `wl_kernel_matcher.py` — `graph_shape_sim` | Four global topology signatures — degree sequence, Laplacian spectrum, clustering coefficients, betweenness centrality — are each compared by cosine-of-sorted-vector and averaged → `shape_sim` |
| B3 | `attribute_reach.py` — `run_attr_dist_stage` | Each model's declared observable-attribute types are K-hop propagated, embedded, and aggregated into one vector per model, then compared by the stricter of an embedding cosine and a WordNet-WUP soft-cosine kernel → `attr_dist_sim` (reported as `attr_weighted`) |

**Assembly** — `compare_stage.py` combines the four scores (`lexical_sim`, `wl_structural`, `shape_sim`, `attr_weighted`) into an equal-weight composite similarity and a Euclidean distance per pair (excluding whichever metric is unavailable for that pair), then aggregates across all pairs in a domain into the summary JSON, or across every pair on disk into the interactive MDS map.

---

## The Four Metrics

Every entity/model pair is scored on four independent dimensions. Each is designed to be blind to the others' signal, so a high score on one says nothing about the others — that's what lets the composite separate "same vocabulary" from "same structure" from "same attributes."

### 1. `lexical_sim` — name-level similarity

**Purpose:** does the vocabulary match, independent of graph position or attribute content?

For each candidate entity pair `(eₐ, e_b)` in the merged CSV:

```
wup(eₐ, e_b)      = (max_wup + avg_wup) / 2

    where, for camelCase-split token sets Tₐ, T_b:
      max_wup = max{ WUP(tᵢ, tⱼ) : tᵢ∈Tₐ, tⱼ∈T_b }
      avg_wup = mean{ WUP(tᵢ, tⱼ) : tᵢ∈Tₐ, tⱼ∈T_b }
      WUP(tᵢ,tⱼ) = 2·depth(LCS(tᵢ,tⱼ)) / (depth(tᵢ) + depth(tⱼ))   [WordNet Wu-Palmer]

row_lex(eₐ, e_b)  = max( matched,  wup  if wup ≥ 0.75 else 0 )

    where matched ∈ {0, 1}: 1 iff AML or LogMap independently confirmed the pair

lexical_sim       = mean over all candidate rows of row_lex
```

`type_embed_sim` / `cosine_avg` are **deliberately excluded** from `lexical_sim` — that attribute-type signal is already the entire basis of `attr_weighted`, so folding it in here would double-count it. The 0.75 WUP gate exists because WordNet's generic machine/artifact hierarchy gives unrelated engineering nouns (fuse, filter, blower) a WUP of ≈0.89 — below the gate they contribute 0, not a false positive.

### 2. `wl_structural` — local relational topology

**Purpose:** do the two models reuse the same relation patterns in the same local neighbourhood — i.e. is this the same *modelling style* (process-centric vs entity-centric) — regardless of what anything is named?

Edge-aware Weisfeiler-Lehman kernel, K=3 hops, **all nodes anonymous**:

```
label_v⁽⁰⁾   = hash("N")                                     for every node v
label_v⁽ᵏ⁺¹⁾ = md5( label_v⁽ᵏ⁾ | sorted[(label_u⁽ᵏ⁾, edge_type) : u ∈ N(v)] )

freq_X = histogram of {label_v⁽ᵏ⁾ : v ∈ X, k = 0..K}          (pooled over all K+1 hops)

wl_structural = cosine(freq_A, freq_B) = (freq_A · freq_B) / (‖freq_A‖ ‖freq_B‖)
```

`edge_type` is the canonical relation name from `association_inventory.csv` (PartOf, Connects, UsedFor, …) — the only information that ever distinguishes two nodes. Entity names never enter this computation.

### 3. `shape_sim` — global graph topology

**Purpose:** separate models by scope/density (star vs chain vs clique) — a signal orthogonal to both vocabulary and local relational motifs.

Four cosine-of-sorted-vector sub-metrics, averaged:

```
degree_sim      = cosine( sorted_desc(deg_A), sorted_desc(deg_B) )
spectral_sim    = cosine( sorted(λ_A), sorted(λ_B) )     λ = eigenvalues of L = I − D^(−1/2) A D^(−1/2)
clustering_sim  = cosine( sorted_desc(C_A), sorted_desc(C_B) )
                    C(v) = triangles(v) / (d(v)(d(v)−1)/2)
betweenness_sim = cosine( sorted_desc(BC_A), sorted_desc(BC_B) )   [Brandes' O(VE), normalised]

shape_sim = ( degree_sim + spectral_sim + clustering_sim + betweenness_sim ) / 4
```

Vectors of unequal length are zero-padded before the cosine.

### 4. `attr_weighted` — attribute reachability

**Purpose:** are the two models "about" the same observable quantities (temperature, torque, pressure, …), independent of both entity names and graph shape?

```
V            = union of observable types declared across both models (data["observables"] ∪ entityAttribute.type)

reach(e)     = K-hop (K=2) weighted BFS propagation of e's own + neighbours' declared
               attribute types, through structurally-weighted edges:
                 PartOf/HasA=0.6, MadeOf=0.5, Connects/UsedFor=0.4, AtLocation=0.3,
                 CapableOf/Causes/IsA/ReceivesAction=0.25, HasPrerequisite=0.2, RelatedTo=skip
               path weight = product of edge weights; MAX kept across paths to the same type
               (name-based imputation is DISABLED for this whole-model stage — purely structural)

agg_X        = Σ_{e∈X} Σ_{t∈reach(e)} weight(e,t) · embed(t)      embed = paraphrase-MiniLM-L6-v2

embed_sim    = cosine(agg_A, agg_B)

wup_sim      = soft-cosine(w_A, w_B; S)  =  (w_Aᵀ S w_B) / √(w_Aᵀ S w_A · w_Bᵀ S w_B)
                 S = |V|×|V| pairwise WUP-similarity kernel over the closed type vocabulary
                 w_X = raw (non-embedded) summed attribute weight vector for model X

attr_dist_sim = min(embed_sim, wup_sim)     — the stricter of the two; WUP alone runs too
                                               forgiving for engineering/physics nouns

attr_weighted = attr_dist_sim               — unmodified; no longer scaled by entity-name WUP
```

`attr_weighted` shares **no data at all** with `lexical_sim` — `wup_sim` above is a WUP kernel `S` computed over the closed *observable-type* vocabulary `V` (Temperature, Torque, Pressure, …), entirely independent from `lexical_sim`'s entity-*name* WUP (§1). Both call the same underlying WordNet Wu-Palmer function, but over disjoint inputs (attribute-type tokens vs. entity-name tokens) with no value passed between them. This was previously coupled — `attr_weighted` used to be `attr_dist_sim × avg_entity_wup` — but that scaling was removed (see `attribute_reach.py::run_attr_dist_stage` docstring) because it made a supposedly attribute-only metric depend on entity naming.

### Composite score and distance

```
METRICS   = [lexical_sim, wl_structural, shape_sim, attr_weighted]     (nominal weight 1/4 each)
available = { m ∈ METRICS : value(m) > 0 }

composite(A,B) = mean{ value(m) : m ∈ available }
                   — metrics unavailable for a pair (e.g. attr_weighted=0 when neither
                     model has observable types) are excluded, not counted as a penalty

distance(A,B)  = √( Σ_{m∈available} (1 − value(m))² / |available| )     ∈ [0,1]
                   — symmetrised: d = (d + dᵀ) / 2   (used for the domain summary and the MDS map)
```

### Supporting (non-composite) signals

These feed the metrics above or internal entity-match declaration, but are not themselves one of the four composite dimensions:

| Signal | Written by | Used by |
|--------|-----------|---------|
| `matched` | `enriched_matcher.py` (AML/LogMap confirmation) | `lexical_sim` |
| `wup` | `enriched_matcher.py` (entity-*name* WUP) | `lexical_sim` only — **not** used by `attr_weighted`. `attribute_reach.py` computes its own separate WUP kernel over the attribute-*type* vocabulary (§4) with no data passed between the two |
| `cosine_avg` | `semantic_encoder.py` (name-based sentence embedding) | fallback for entity-match declaration when no attribute-type evidence exists |
| `type_embed_sim` | `attribute_reach.py` (attribute-type embedding, per entity pair) | primary signal for entity-match declaration (`wl_kernel_matcher.declare_entity_matches`), which feeds the diagnostic `wl_matched`/`wl_composite`/`wl_consistency` columns in the WL CSV — **not** part of the top-level composite |

---

## Prerequisites

### 1. Python 3.10+

```bash
python --version   # must be 3.10 or higher
```

### 2. Java 11+ (required by AML and LogMap)

```bash
java -version   # must be 11 or higher
```

Download from [Adoptium](https://adoptium.net/) if needed.

### 3. LogMap JAR

Already included in the repo:

```
enriched_ontology_matching/tools/logmap/logmap-matcher-4.0.jar
```

### 4. AML JAR

Download `AgreementMakerLight.jar` from the [AML releases page](https://github.com/AgreementMakerLight/AML-Project/releases) and place it at:

```
enriched_ontology_matching/tools/AML/AML_v3.2/AgreementMakerLight.jar
```

Extract the full AML release zip into that folder — the `store/` directory (config + stop list) must be present alongside the JAR.

### 5. ConceptNet (optional but recommended for bulk runs)

Without a local file the pipeline falls back to the ConceptNet REST API (rate-limited). For bulk runs:

```bash
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
gunzip conceptnet-assertions-5.7.0.csv.gz
```

Place the extracted file at `ontology_matching/inputs/conceptnet-assertions-5.7.0.csv/assertions.csv`.

---

## Setup

All commands run from the **repository root**.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# .venv/bin/activate               # macOS / Linux

pip install -e .
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

This installs two CLI commands into the active environment: `eom-run` and `eom-compare`.

---

## Input Models

```
enriched_ontology_matching/inputs/
  Automobile/            6 models   (V1, V2, V3, Net1, Net2, Net3)
  Automobile_Synthetic/  15 models  (3×5 factorial: SAME/SYN/ALT × DEEP/WIDE/HUB/BIP/GRID)
  Coffee/                6 models
  Homebrewing/           6 models
  Hospital/              6 models
  SmartHome/             6 models
  University/            6 models
```

Each JSON is one conceptual model:

```json
{
  "modelName": "Automobile_Model_V1_SystemCentric",
  "observables": ["Power", "Torque", "..."],
  "entities": [
    { "entityName": "Engine", "entityAttributes": [{"name": "power", "type": "Power"}] },
    { "entityName": "Wheel" }
  ],
  "associations": [
    { "associationName": "EngineDrivesWheel", "associationParticipants": ["Engine", "Wheel"] }
  ]
}
```

Network-variant models (`Net1`/`Net2`/`Net3`) use `"name"` for entities and `"participants"` for associations.

---

## CLI: end-to-end run over one domain

```bash
# Step 1 — run every pairwise comparison within a domain (6 stages per pair)
eom-run --inputs-dir enriched_ontology_matching/inputs --domains Automobile

# Step 2 — get the domain distance summary as JSON
eom-compare --domain-summary Automobile
```

**Other `eom-run` flags:**

| Flag | Effect |
|------|--------|
| `--inputs-dir PATH` | Root containing `<Domain>/*.json` subdirectories (default: built-in `inputs/`) |
| `--domains D1 D2 …` | Limit to specific domains (default: all 7 registered domains) |
| `--matcher {aml,logmap,both}` | Which structural matcher(s) to run (default: `both`) |
| `--skip-existing` | Reuse any per-pair CSV that already exists instead of recomputing |
| `--cross-domain` | Also run cross-domain pairs (needed before the map, see below) |

`--domain-summary` accepts one of the 6 domain labels `compare_stage.py` recognizes: `Automobile`, `Hospital`, `University`, `Coffee`, `Homebrewing`, `SmartHome` (`Automobile_Synthetic` model pairs are still produced by `eom-run` but are classified under `Automobile` for summary/map purposes, since domain classification is by name-keyword match).

### What `eom-run` needs and generates, per pair

`run_all_pairs.py` runs 6 stages per model pair and writes one file per stage. `<stem>` = `<ModelA_modelName>_vs_<ModelB_modelName>`; `<domain_key>` = short id like `auto_V1_V2`.

| Stage | Module | Needs | Writes | Contents |
|-------|--------|-------|--------|----------|
| 0 | `run_all_pairs.py` | the two model JSONs | `enriched_ontology_matching/pairs/<domain_key>.json` | combined `{json_a, json_b}` input for the rest of the pipeline (gitignored — regenerated every run) |
| 1 | `enriched_matcher.py` | stage-0 pair JSON, `association_inventory.csv`, WordNet, ConceptNet (local CSV or REST) | `outputs/enriched/<stem>.csv` | AML + LogMap candidate matches, annotated with `wup`, `max_wup`, `avg_wup`, WordNet/ConceptNet relation labels; plus `outputs/enriched/all_domains_combined.csv` (all pairs concatenated across the whole `eom-run` invocation) |
| 2 | `semantic_encoder.py` | stage-1 CSV, both model JSONs | `outputs/embeddings/<domain_key>_emb.csv` | `cosine_avg` — name/attribute-type sentence-embedding cosine (rescaled to [0,1]) per candidate row |
| 3 | `attribute_reach.py` (`run_type_embed_stage`) | stage-1 CSV, both model JSONs | `outputs/type_embed/<domain_key>_type_emb.csv` | `type_embed_sim` — attribute-type embedding cosine per candidate row |
| 4 | `merge_stage.py` | stages 1–3 CSVs | `outputs/merged/<stem>_metrics.csv` | one row per candidate entity pair: `entity_a, entity_b, matched, wup, cosine_avg, type_embed_sim` |
| 5 | `wl_kernel_matcher.py` (`run_wl_stage`) | both model JSONs, stage-4 merged CSV | `outputs/wl/<stem>_metrics_wl.csv` | one row: `wl_structural, shape_sim` (+ sub-metrics `degree_sim/spectral_sim/clustering_sim/betweenness_sim`), plus diagnostic-only `wl_matched, wl_composite, wl_consistency, match_coverage, induced_frac_a/b, n_entity_matches, n_shared_labels, n_nodes_a/b, n_edges_a/b` |
| 6 | `attribute_reach.py` (`run_attr_dist_stage`) | both model JSONs | `outputs/attr_dist/<stem>_metrics_attr_dist.csv` | one row: `attr_dist_sim` |

Then `eom-compare --domain-summary <Domain>` reads every `outputs/merged/*_metrics.csv` (pulling `shape_sim`/`wl_structural` from the matching `outputs/wl/` CSV and `attr_dist_sim` from the matching `outputs/attr_dist/` CSV) for the requested domain and writes:

```
enriched_ontology_matching/summaries/<domain_lower>_summary.json
```

containing `n_ontologies`, `n_pairs`, `metric_weights` (0.25 each), `average_distance`, `average_composite`, and a `pairs[]` array (sorted by ascending distance) with each pair's `distance`, `composite`, `n_entity_pairs`, and per-metric `{mean, weight}`.

```json
{
  "domain": "Automobile",
  "n_ontologies": 6,
  "n_pairs": 15,
  "metric_weights": {"lexical_sim": 0.25, "wl_structural": 0.25, "shape_sim": 0.25, "attr_weighted": 0.25},
  "average_distance": 0.376,
  "average_composite": 0.675,
  "pairs": [
    {"ont_a": "...", "ont_b": "...", "distance": 0.21, "composite": 0.84, "n_entity_pairs": 291,
     "metrics": {"lexical_sim": {"mean": 0.874, "weight": 0.25}, "...": "..."}}
  ]
}
```

---

## Cross-domain analysis and the interactive map (secondary)

```bash
# Run every within-domain AND cross-domain pair (skip anything already cached)
eom-run --cross-domain --skip-existing

# Build the interactive distance map from every merged CSV under outputs/merged/
eom-compare
```

`eom-compare` with no arguments globs **all** `outputs/merged/*_metrics.csv` files — run it after enough domains (ideally all, with `--cross-domain`) have been processed, or the map will only show whatever partial set of ontologies has merged CSVs on disk.

Output: `enriched_ontology_matching/outputs/ontology_map.html` — self-contained, open directly in a browser. Node position = regularised-MDS embedding of the weighted-Euclidean distance matrix (full pairwise fidelity + a soft domain-cohesion penalty, λ=0.3); edge width/opacity = composite similarity; toggle buttons switch between MST / top-3 / top-5 / all-edges / nodes-only; click a domain in the legend to hide it, double-click to isolate it.

---

## Reproducing the `docs/` visualizations

`scripts/probe_visualizer.py` renders 4 controlled micro-experiments — each isolates one axis of variation while holding the others fixed — by running each probe pair through the **real** production pipeline (the same modules `eom-run` uses) and reading back `lexical_sim`, `wl_structural`, `shape_sim`, `attr_weighted`, `composite` via `compare_stage.load_pair_metrics()`. There is no separate simplified metric reimplementation.

```bash
# Regenerate all 4 series
python scripts/probe_visualizer.py

# Regenerate only specific series (1=Naming, 2=Attribute, 3=Topology, 4=Density)
python scripts/probe_visualizer.py --series 1 3

# Point at different fixtures / output location
python scripts/probe_visualizer.py --pairs-dir path/to/fixtures --out-dir path/to/out
```

| Series | Changed factor | Held constant |
|--------|----------------|----------------|
| 1 — Naming Drift | Entity names | Topology, attributes |
| 2 — Attribute Drift | Observable types per entity | Topology, entity names |
| 3 — Topology Drift | Composition depth (chain → star) | Entity names, attributes |
| 4 — Density Drift | Edge density (5-cycle → K₅) | Entity names, attributes |

**What the script needs:** small hand-authored pair-fixture JSONs (`{json_a, json_b}`, same schema as the domain inputs) under `--pairs-dir` (default `enriched_ontology_matching/pairs/`), named `probe_s{1-4}_*.json` and `Probe_*.json`. These ~60 files are committed to the repo (a narrow `.gitignore` exception carves them out of the otherwise-ignored `pairs/` directory) since there is no generator script for them — they're small (≈340 KB total), stable, and this is the only way the visualizations reproduce from a clean clone.

**What it writes**, under `--out-dir` (default `docs/`):

| Output | Contents |
|--------|----------|
| `probe_visualizations.pdf` | one page per rendered series |
| `probe_s1.png` … `probe_s4.png` | one PNG per rendered series (only the ones selected via `--series`) |

Each run also populates the normal pipeline intermediate directories (`outputs/enriched/`, `outputs/embeddings/`, `outputs/type_embed/`, `outputs/merged/`, `outputs/wl/`, `outputs/attr_dist/`) for every probe pair, exactly like `eom-run` does, since it calls the same stage functions.

---

## Codebase Map

| File | Purpose |
|------|---------|
| `run_all_pairs.py` | **`eom-run` entry point** — batch runner for within-domain (and optionally cross-domain) pairs |
| `compare_stage.py` | **`eom-compare` entry point** — `--domain-summary DOMAIN` writes the JSON distance summary; no args builds the interactive HTML map |
| `enriched_matcher.py` | Structural matching (AML + LogMap), WordNet/ConceptNet semantic discovery, WUP backup for orphan entities |
| `semantic_encoder.py` | Name/attribute-type sentence-embedding cosine similarity (`cosine_avg`) |
| `attribute_reach.py` | K-hop attribute-type reach; `run_type_embed_stage` (per-entity-pair `type_embed_sim`) and `run_attr_dist_stage` (whole-model `attr_dist_sim`) |
| `wl_kernel_matcher.py` | WL graph kernel (`wl_structural`), graph-shape sub-metrics (`shape_sim`), and entity-match declaration used by the diagnostic WL fields |
| `merge_stage.py` | Joins enriched-matcher + embedding + type-embedding CSVs into the per-pair metrics CSV |
| `aml_runner.py` / `logmap_runner.py` | Wrappers around the AML / LogMap JARs |
| `root_comparator.py` | CamelCase splitting, WUP, ConceptNet CSV lookup |
| `model_normalizer.py` | Normalises model JSON schema variants (entity/association field names) |
| `scripts/probe_visualizer.py` | Regenerates `docs/probe_visualizations.pdf` and `docs/probe_s*.png` from the probe fixtures |
