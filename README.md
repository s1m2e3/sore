# Enriched Ontology Matching Pipeline

This pipeline compares pairs of conceptual, entity-relationship models by combining two independent measurement approaches. Semantic and lexical matching, using WordNet, ConceptNet, and sentence-transformer embeddings and seeded by the AML and LogMap structural matchers, establishes and characterizes entity-level correspondences. Graph-theoretic structural comparison, using a Weisfeiler-Lehman relational kernel, degree, spectral, clustering, and betweenness graph-shape metrics, and a K-hop attribute-reachability comparison, scores topology and attribute content independently of entity nomenclature. The four resulting scores are combined into a per-pair composite similarity and distance, producing a domain distance summary in JSON format and an interactive distance map.

The primary use case is **within-domain analysis**: all pairwise comparisons are computed for a set of ontologies that share a domain, and the average pairwise distance is reported as a JSON summary. Cross-domain comparisons and the interactive distance map are secondary outputs.

---

## Pipeline Overview

The pipeline has two entry points. `run_all_pairs.py` (the `eom-run` command) orchestrates entity matching and metric estimation for every model pair, calling the following modules:

| Script | Produces |
|--------|----------|
| `enriched_matcher.py` | Candidate entity matches (AML, LogMap) and their semantic characterisation (WordNet, ConceptNet), feeding `lexical_sim` |
| `semantic_encoder.py` | Sentence-embedding similarity between candidate entity pairs (`cosine_avg`) |
| `attribute_reach.py` | Attribute-type similarity, both per entity pair (`type_embed_sim`) and per model pair (`attr_dist_sim`, reported as `attr_weighted`) |
| `merge_stage.py` | Merges the above into one metrics CSV per model pair |
| `wl_kernel_matcher.py` | Graph-theoretic structural comparison (`wl_structural`, `shape_sim`) |

`compare_stage.py` (the `eom-compare` command) is the second entry point: it reads these outputs and produces either a domain distance summary JSON or the interactive distance map.

---

## The Four Metrics

Every entity or model pair is scored on four independent dimensions. Each dimension is designed to be blind to the signal captured by the others, so that a high score on one carries no implication for the others. This independence allows the composite score to separate vocabulary similarity, structural similarity, and attribute similarity as distinct sources of evidence.

### 1. `lexical_sim`: name-level similarity

**Purpose.** This metric measures whether the vocabulary matches, independent of graph position or attribute content.

For each candidate entity pair `(eₐ, e_b)` in the merged CSV:

```
wup(eₐ, e_b)      = (max_wup + avg_wup) / 2

    where, for camelCase-split token sets Tₐ, T_b:
      max_wup = max{ WUP(tᵢ, tⱼ) : tᵢ∈Tₐ, tⱼ∈T_b }
      avg_wup = mean{ WUP(tᵢ, tⱼ) : tᵢ∈Tₐ, tⱼ∈T_b }
      WUP(tᵢ,tⱼ) = 2·depth(LCS(tᵢ,tⱼ)) / (depth(tᵢ) + depth(tⱼ))   [WordNet Wu-Palmer]

row_lex(eₐ, e_b)  = max( matched,  wup  if wup ≥ 0.75 else 0 )

    where matched ∈ {0, 1}: 1 if AML or LogMap independently confirmed the pair

lexical_sim       = mean over all candidate rows of row_lex
```

`type_embed_sim` and `cosine_avg` are deliberately excluded from `lexical_sim`, since that attribute-type signal already forms the entire basis of `attr_weighted`; including it here would double-count the signal. The 0.75 WUP threshold exists because WordNet's generic machine or artifact hierarchy assigns unrelated engineering nouns, such as fuse, filter, and blower, a WUP similarity of approximately 0.89. Below this threshold, such pairs contribute a value of zero rather than being registered as a false match.

### 2. `wl_structural`: local relational topology

**Purpose.** This metric measures whether the two models reuse the same relation patterns within the same local neighborhood, that is, whether they share the same modeling style (process-centric versus entity-centric), regardless of entity nomenclature.

Edge-aware Weisfeiler-Lehman kernel, K=3 hops, with all nodes anonymous:

```
label_v⁽⁰⁾   = hash("N")                                     for every node v
label_v⁽ᵏ⁺¹⁾ = md5( label_v⁽ᵏ⁾ | sorted[(label_u⁽ᵏ⁾, edge_type) : u ∈ N(v)] )

freq_X = histogram of {label_v⁽ᵏ⁾ : v ∈ X, k = 0..K}          (pooled over all K+1 hops)

wl_structural = cosine(freq_A, freq_B) = (freq_A · freq_B) / (‖freq_A‖ ‖freq_B‖)
```

`edge_type` is the canonical relation name defined in `association_inventory.csv` (for example, PartOf, Connects, UsedFor). It is the only information that distinguishes two nodes; entity names are not used in this computation.

### 3. `shape_sim`: global graph topology

**Purpose.** This metric distinguishes models by scope and density, for example star, chain, or clique topologies, providing a signal orthogonal to both vocabulary and local relational motifs.

Four sub-metrics, each a cosine similarity of sorted vectors, are averaged:

```
degree_sim      = cosine( sorted_desc(deg_A), sorted_desc(deg_B) )
spectral_sim    = cosine( sorted(λ_A), sorted(λ_B) )     λ = eigenvalues of L = I − D^(−1/2) A D^(−1/2)
clustering_sim  = cosine( sorted_desc(C_A), sorted_desc(C_B) )
                    C(v) = triangles(v) / (d(v)(d(v)−1)/2)
betweenness_sim = cosine( sorted_desc(BC_A), sorted_desc(BC_B) )   [Brandes' O(VE), normalised]

shape_sim = ( degree_sim + spectral_sim + clustering_sim + betweenness_sim ) / 4
```

Vectors of unequal length are zero-padded before computing the cosine similarity.

### 4. `attr_weighted`: attribute reachability

**Purpose.** This metric measures whether the two models describe the same observable quantities, for example temperature, torque, and pressure, independent of both entity names and graph shape.

```
V            = union of observable types declared across both models (data["observables"] ∪ entityAttribute.type)

reach(e)     = K-hop (K=2) weighted BFS propagation of e's own and neighbours' declared
               attribute types, through structurally-weighted edges:
                 PartOf/HasA=0.6, MadeOf=0.5, Connects/UsedFor=0.4, AtLocation=0.3,
                 CapableOf/Causes/IsA/ReceivesAction=0.25, HasPrerequisite=0.2, RelatedTo=skip
               path weight = product of edge weights; MAX kept across paths to the same type
               (name-based imputation is disabled for this whole-model stage, keeping the
               computation purely structural)

agg_X        = Σ_{e∈X} Σ_{t∈reach(e)} weight(e,t) · embed(t)      embed = paraphrase-MiniLM-L6-v2

embed_sim    = cosine(agg_A, agg_B)

wup_sim      = soft-cosine(w_A, w_B; S)  =  (w_Aᵀ S w_B) / √(w_Aᵀ S w_A · w_Bᵀ S w_B)
                 S = |V|×|V| pairwise WUP-similarity kernel over the closed type vocabulary
                 w_X = raw (non-embedded) summed attribute weight vector for model X

attr_dist_sim = min(embed_sim, wup_sim), the stricter of the two; WUP alone tends to be too
                forgiving for engineering and physics nouns

attr_weighted = attr_dist_sim, unmodified; no longer scaled by entity-name WUP
```

`attr_weighted` shares no data with `lexical_sim`. The kernel `wup_sim` above is computed over the closed observable-type vocabulary `V` (for example, Temperature, Torque, and Pressure), and is entirely independent of the entity-name WUP score used in `lexical_sim` (Section 1). Both computations call the same underlying WordNet Wu-Palmer function, but over disjoint inputs, attribute-type tokens in one case and entity-name tokens in the other, with no value passed between them. In an earlier version of the pipeline these metrics were coupled: `attr_weighted` was computed as `attr_dist_sim` multiplied by `avg_entity_wup`. This scaling was removed, see the docstring of `run_attr_dist_stage` in `attribute_reach.py`, because it made a metric intended to capture attribute similarity alone depend on entity naming.

### Composite score and distance

```
METRICS   = [lexical_sim, wl_structural, shape_sim, attr_weighted]     (nominal weight 1/4 each)
available = { m ∈ METRICS : value(m) > 0 }

composite(A,B) = mean{ value(m) : m ∈ available }
                   metrics unavailable for a pair (for example, attr_weighted=0 when neither
                   model has observable types) are excluded rather than counted as a penalty

distance(A,B)  = √( Σ_{m∈available} (1 − value(m))² / |available| )     ∈ [0,1]
                   symmetrised as d = (d + dᵀ) / 2, and used for both the domain summary and
                   the MDS map
```

### Supporting (non-composite) signals

These feed the metrics above or the internal entity-match declaration, but are not themselves one of the four composite dimensions.

| Signal | Written by | Used by |
|--------|-----------|---------|
| `matched` | `enriched_matcher.py` (AML/LogMap confirmation) | `lexical_sim` |
| `wup` | `enriched_matcher.py` (entity-*name* WUP) | `lexical_sim` only. It is not used by `attr_weighted`; `attribute_reach.py` computes a separate WUP kernel over the attribute-*type* vocabulary (Section 4), and no data is passed between the two. |
| `cosine_avg` | `semantic_encoder.py` (name-based sentence embedding) | fallback for entity-match declaration when no attribute-type evidence exists |
| `type_embed_sim` | `attribute_reach.py` (attribute-type embedding, per entity pair) | primary signal for entity-match declaration (`wl_kernel_matcher.declare_entity_matches`), which feeds the diagnostic `wl_matched`, `wl_composite`, and `wl_consistency` columns in the WL CSV. It is not part of the top-level composite. |

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

Already included in the repository:

```
enriched_ontology_matching/tools/logmap/logmap-matcher-4.0.jar
```

### 4. AML JAR

Download `AgreementMakerLight.jar` from the [AML releases page](https://github.com/AgreementMakerLight/AML-Project/releases) and place it at:

```
enriched_ontology_matching/tools/AML/AML_v3.2/AgreementMakerLight.jar
```

Extract the full AML release archive into that folder. The `store/` directory, containing configuration files and the stop list, must be present alongside the JAR.

### 5. ConceptNet (optional, recommended for bulk runs)

Without a local file, the pipeline falls back to the ConceptNet REST API, which is rate-limited. For bulk runs:

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
  Automobile_Synthetic/  15 models  (3x5 factorial: SAME/SYN/ALT x DEEP/WIDE/HUB/BIP/GRID)
  Coffee/                6 models
  Homebrewing/           6 models
  Hospital/              6 models
  SmartHome/             6 models
  University/            6 models
```

Each JSON file is one conceptual model:

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

Network-variant models (`Net1`, `Net2`, `Net3`) use `"name"` for entities and `"participants"` for associations.

---

## CLI: end-to-end run over one domain

```bash
# Step 1: run every pairwise comparison within a domain (6 stages per pair)
eom-run --inputs-dir enriched_ontology_matching/inputs --domains Automobile

# Step 2: get the domain distance summary as JSON
eom-compare --domain-summary Automobile
```

**Other `eom-run` flags:**

| Flag | Effect |
|------|--------|
| `--inputs-dir PATH` | Root containing `<Domain>/*.json` subdirectories (default: built-in `inputs/`) |
| `--domains D1 D2 ...` | Limit to specific domains (default: all 7 registered domains) |
| `--matcher {aml,logmap,both}` | Which structural matcher(s) to run (default: `both`) |
| `--skip-existing` | Reuse any per-pair CSV that already exists instead of recomputing |
| `--cross-domain` | Also run cross-domain pairs (needed before the map, see below) |

`--domain-summary` accepts one of the six domain labels that `compare_stage.py` recognizes: `Automobile`, `Hospital`, `University`, `Coffee`, `Homebrewing`, `SmartHome`. `Automobile_Synthetic` model pairs are still produced by `eom-run`, but are classified under `Automobile` for summary and map purposes, since domain classification is performed by name-keyword match.

### What `eom-run` needs and generates, per pair

`run_all_pairs.py` runs 6 stages per model pair and writes one file per stage. `<stem>` denotes `<ModelA_modelName>_vs_<ModelB_modelName>`; `<domain_key>` denotes a short identifier such as `auto_V1_V2`.

| Stage | Module | Needs | Writes | Contents |
|-------|--------|-------|--------|----------|
| 0 | `run_all_pairs.py` | the two model JSONs | `enriched_ontology_matching/pairs/<domain_key>.json` | combined `{json_a, json_b}` input for the rest of the pipeline (gitignored; regenerated on every run) |
| 1 | `enriched_matcher.py` | stage-0 pair JSON, `association_inventory.csv`, WordNet, ConceptNet (local CSV or REST) | `outputs/enriched/<stem>.csv` | AML and LogMap candidate matches, annotated with `wup`, `max_wup`, `avg_wup`, and WordNet/ConceptNet relation labels; plus `outputs/enriched/all_domains_combined.csv` (all pairs concatenated across the whole `eom-run` invocation) |
| 2 | `semantic_encoder.py` | stage-1 CSV, both model JSONs | `outputs/embeddings/<domain_key>_emb.csv` | `cosine_avg`: name and attribute-type sentence-embedding cosine similarity (rescaled to [0, 1]) for each candidate row |
| 3 | `attribute_reach.py` (`run_type_embed_stage`) | stage-1 CSV, both model JSONs | `outputs/type_embed/<domain_key>_type_emb.csv` | `type_embed_sim`: attribute-type embedding cosine similarity for each candidate row |
| 4 | `merge_stage.py` | stages 1 to 3 CSVs | `outputs/merged/<stem>_metrics.csv` | one row per candidate entity pair: `entity_a, entity_b, matched, wup, cosine_avg, type_embed_sim` |
| 5 | `wl_kernel_matcher.py` (`run_wl_stage`) | both model JSONs, stage-4 merged CSV | `outputs/wl/<stem>_metrics_wl.csv` | one row: `wl_structural, shape_sim` (plus sub-metrics `degree_sim/spectral_sim/clustering_sim/betweenness_sim`), and diagnostic-only `wl_matched, wl_composite, wl_consistency, match_coverage, induced_frac_a/b, n_entity_matches, n_shared_labels, n_nodes_a/b, n_edges_a/b` |
| 6 | `attribute_reach.py` (`run_attr_dist_stage`) | both model JSONs | `outputs/attr_dist/<stem>_metrics_attr_dist.csv` | one row: `attr_dist_sim` |

`eom-compare --domain-summary <Domain>` then reads every `outputs/merged/*_metrics.csv` file for the requested domain, pulling `shape_sim` and `wl_structural` from the matching `outputs/wl/` CSV and `attr_dist_sim` from the matching `outputs/attr_dist/` CSV, and writes:

```
enriched_ontology_matching/summaries/<domain_lower>_summary.json
```

This file contains `n_ontologies`, `n_pairs`, `metric_weights` (0.25 each), `average_distance`, `average_composite`, and a `pairs[]` array, sorted by ascending distance, with each pair's `distance`, `composite`, `n_entity_pairs`, and per-metric `{mean, weight}`.

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
# Run every within-domain and cross-domain pair, skipping anything already cached
eom-run --cross-domain --skip-existing

# Build the interactive distance map from every merged CSV under outputs/merged/
eom-compare
```

`eom-compare` invoked with no arguments collects every `outputs/merged/*_metrics.csv` file on disk. It should be run only after enough domains, ideally all of them, have been processed with `--cross-domain`; otherwise the map will show only the partial set of ontologies for which merged CSVs exist.

Output: `enriched_ontology_matching/outputs/ontology_map.html`, a self-contained file that can be opened directly in a browser. Node position is determined by a regularized MDS embedding of the weighted Euclidean distance matrix, balancing full pairwise fidelity against a soft domain-cohesion penalty (λ = 0.3). Edge width and opacity encode composite similarity. Toggle buttons switch between the minimum spanning tree, top-3 neighbors, top-5 neighbors, all edges, and nodes-only views. Clicking a domain in the legend hides it, and double-clicking isolates it.

---

## Reproducing the `docs/` visualizations

`scripts/probe_visualizer.py` renders four controlled micro-experiments, each isolating a single axis of variation while holding the others fixed. Each probe pair is processed through the same production pipeline that `eom-run` uses, and the resulting `lexical_sim`, `wl_structural`, `shape_sim`, `attr_weighted`, and `composite` values are read back through `compare_stage.load_pair_metrics()`. No separate, simplified reimplementation of the metrics exists in this script.

```bash
# Regenerate all 4 series
python scripts/probe_visualizer.py

# Regenerate only specific series (1=Naming, 2=Attribute, 3=Topology, 4=Density)
python scripts/probe_visualizer.py --series 1 3

# Point to alternate fixture or output locations
python scripts/probe_visualizer.py --pairs-dir path/to/fixtures --out-dir path/to/out
```

| Series | Changed factor | Held constant |
|--------|----------------|----------------|
| 1: Naming Drift | Entity names | Topology, attributes |
| 2: Attribute Drift | Observable types per entity | Topology, entity names |
| 3: Topology Drift | Composition depth (chain to star) | Entity names, attributes |
| 4: Density Drift | Edge density (5-cycle to K5) | Entity names, attributes |

**Input requirements.** The script requires small, hand-authored pair-fixture JSON files (`{json_a, json_b}`, following the same schema as the domain inputs) under `--pairs-dir` (default `enriched_ontology_matching/pairs/`), named `probe_s{1-4}_*.json` and `Probe_*.json`. These approximately 60 files are committed to the repository, carved out of the otherwise ignored `pairs/` directory by a narrow `.gitignore` exception, because no generator script exists for them. They are small (approximately 340 KB in total) and stable, and committing them is the only way the visualizations can be reproduced from a clean clone.

**Output.** Under `--out-dir` (default `docs/`), the script writes:

| Output | Contents |
|--------|----------|
| `probe_visualizations.pdf` | one page per rendered series |
| `probe_s1.png` through `probe_s4.png` | one PNG per rendered series (only the ones selected via `--series`) |

Each run also populates the standard pipeline intermediate directories (`outputs/enriched/`, `outputs/embeddings/`, `outputs/type_embed/`, `outputs/merged/`, `outputs/wl/`, `outputs/attr_dist/`) for every probe pair, in the same manner as `eom-run`, since it invokes the same stage functions.

---

## Codebase Map

| File | Purpose |
|------|---------|
| `run_all_pairs.py` | **`eom-run` entry point.** Batch runner for within-domain and, optionally, cross-domain pairs. |
| `compare_stage.py` | **`eom-compare` entry point.** `--domain-summary DOMAIN` writes the JSON distance summary; without arguments, it builds the interactive HTML map. |
| `enriched_matcher.py` | Structural matching (AML and LogMap), WordNet/ConceptNet semantic discovery, WUP backup for orphan entities |
| `semantic_encoder.py` | Name and attribute-type sentence-embedding cosine similarity (`cosine_avg`) |
| `attribute_reach.py` | K-hop attribute-type reach; `run_type_embed_stage` (per-entity-pair `type_embed_sim`) and `run_attr_dist_stage` (whole-model `attr_dist_sim`) |
| `wl_kernel_matcher.py` | WL graph kernel (`wl_structural`), graph-shape sub-metrics (`shape_sim`), and entity-match declaration used by the diagnostic WL fields |
| `merge_stage.py` | Joins the enriched-matcher, embedding, and type-embedding CSVs into the per-pair metrics CSV |
| `aml_runner.py`, `logmap_runner.py` | Wrappers around the AML and LogMap JARs |
| `root_comparator.py` | CamelCase splitting, WUP, ConceptNet CSV lookup |
| `model_normalizer.py` | Normalises model JSON schema variants (entity/association field names) |
| `scripts/probe_visualizer.py` | Regenerates `docs/probe_visualizations.pdf` and `docs/probe_s*.png` from the probe fixtures |
