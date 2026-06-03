# Enriched Ontology Matching Pipeline

A semantic enrichment pipeline that runs structural matchers (AML + LogMap) over pairs of conceptual models, scores every entity pair across four complementary metrics, and produces a domain distance summary JSON and an interactive distance map.

The primary use case is **within-domain analysis**: run all pairwise comparisons for a set of ontologies that share the same domain, then read the average pairwise distance between them as a JSON summary. Cross-domain comparisons and the interactive distance map are also supported as secondary outputs.

---

## Pipeline Overview

| Step | Name | What it does |
|------|------|--------------|
| **1** | Structural Matching | AML + LogMap find entity pairs; results merged and de-duplicated |
| **2** | Semantic Discovery | WordNet + ConceptNet discover equivalence/subsumption candidates among unmatched entities |
| **3** | WUP Backup (L3) | Top-k pairs with a shared root token get Wu-Palmer scores when no matcher confirms them |
| **4** | Sentence Embedding | `paraphrase-MiniLM-L6-v2` cosine similarity per entity pair (rescaled to [0, 1]) |
| **5** | WL + Shape | Weisfeiler-Leman kernel similarity and graph-shape sub-metrics (degree, spectral, clustering, betweenness) |
| **6** | Attribute Reach | K-hop observable-type attribute reach distribution; weighted by entity WUP confidence |
| **7** | Merge | Joins enriched-matcher + embedding outputs into one metrics-only CSV per pair |
| **8** | Distance Visualisation | Sparsity-weighted MDS map of all ontologies as interactive HTML |

---

## Four Scoring Dimensions

The pipeline produces four orthogonal metrics, one per composite dimension:

| Metric | Source | What it discriminates |
|--------|--------|-----------------------|
| **`lexical`** — `matched`, `wup`, `cosine_avg` | `enriched_matcher.py` + `semantic_encoder.py` | **Name-level similarity.** `matched` (0/1) signals structural-matcher confirmation; `wup` captures Wu-Palmer path distance in WordNet across CamelCase tokens; `cosine_avg` is the sentence-embedding cosine of entity name strings. Together these separate entities that share vocabulary from those that do not, regardless of graph position. |
| **`wl_structural`** | `wl_kernel_matcher.py` | **Local edge-type motifs.** The Weisfeiler-Leman graph kernel hashes anonymous node labels while preserving edge types (canonical relation names). A high score means the two models reuse the same relational patterns in the same local neighbourhood — it distinguishes models by *modelling style* (process-centric vs entity-centric) rather than by terminology. |
| **`shape_sim`** | `wl_kernel_matcher.py` | **Global graph topology.** Composed of four sub-metrics averaged together: `degree_sim` (degree-sequence cosine), `spectral_sim` (leading eigenvalue ratio), `clustering_sim` (mean clustering coefficient similarity), and `betweenness_sim` (normalised betweenness centrality cosine). Shape separates models by *scope and density* — a star topology vs a chain vs a clique will score very differently even if they name entities identically. |
| **`attr_weighted`** | `attribute_reach.py` | **Attribute reachability × lexical confidence.** `attr_dist_sim` is the cosine distance between the K-hop observable-type attribute-reach distributions of each model; it is then scaled by `avg_entity_wup` (the mean Wu-Palmer score across matched entity pairs). The product down-weights the structural attribute signal when lexical confidence is low, making this metric reliable only when the entity correspondence is already meaningful. Together `attr_weighted` and `wl_structural` discriminate domain (what the model is about) from modelling approach (how it is structured). |

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

No download needed.

### 4. AML JAR

Download `AgreementMakerLight.jar` from the [AML releases page](https://github.com/AgreementMakerLight/AML-Project/releases) and place it at:

```
enriched_ontology_matching/tools/AML/AML_v3.2/AgreementMakerLight.jar
```

Extract the full AML release zip into that folder — the `store/` directory (config + stop list) must be present alongside the JAR.

### 5. ConceptNet (optional but recommended for bulk runs)

Without a local file the pipeline falls back to the ConceptNet REST API (rate-limited). For bulk runs, download the assertions CSV (≈ 1.5 GB compressed):

```bash
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
gunzip conceptnet-assertions-5.7.0.csv.gz
```

Place the extracted file at:

```
ontology_matching/inputs/conceptnet-assertions-5.7.0.csv/assertions.csv
```

---

## Setup

All commands are run from the **repository root** (`sore/`).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

# 2. Install the package and all dependencies (editable install)
pip install -e .

# 3. Download required NLTK data (one-time)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

After installation two CLI commands become available in the active environment:

| Command | Description |
|---------|-------------|
| `eom-run` | Run the full pipeline (L1→L6) for one or more domains |
| `eom-compare` | Generate a domain distance summary JSON or the interactive HTML map |

These commands work from any directory once the venv is active.

---

## Input Models

Domain model JSON files are included in the repo under `enriched_ontology_matching/inputs/`:

```
enriched_ontology_matching/inputs/
  Automobile/      — 6 model JSONs  (V1, V2, V3, Net1, Net2, Net3)
  Coffee/          — 6 model JSONs
  Homebrewing/     — 6 model JSONs
  Hospital/        — 6 model JSONs
  SmartHome/       — 6 model JSONs
  University/      — 6 model JSONs
```

These files are **already included** in the repository — no download required.

Each JSON file represents a single conceptual model with this schema:

```json
{
  "modelName": "Automobile_Model_V1_SystemCentric",
  "entities": [
    { "entityName": "Engine" },
    { "entityName": "Wheel" }
  ],
  "associations": [
    {
      "associationName": "EngineDrivesWheel",
      "associationParticipants": ["Engine", "Wheel"]
    }
  ]
}
```

Network-variant models (`Net1`/`Net2`/`Net3`) use `"name"` for entities and `"participants"` for associations.

---

## Generating Results (End-to-End)

### Quick start — within-domain analysis (primary use case)

```bash
# Step 1 — run all pairwise comparisons for one domain
eom-run --inputs-dir enriched_ontology_matching/inputs --domains Automobile

# Step 2 — get the average pairwise distance summary as JSON
eom-compare --domain-summary Automobile
```

Output: `enriched_ontology_matching/summaries/Automobile_summary.json`

`eom-run` will automatically run all pipeline stages for every pair:

1. **Discover** all model JSON files in `<inputs-dir>/Automobile/`
2. **Generate pair JSONs** in `enriched_ontology_matching/pairs/`
3. **Run enriched matching** (AML + LogMap + WordNet + ConceptNet + WUP backup) → per-pair CSV in `outputs/enriched/`
4. **Run sentence embeddings** (cosine_avg, rescaled to [0, 1]) → `outputs/embeddings/`
5. **Merge** enriched + embeddings → `outputs/merged/<ModelA>_vs_<ModelB>_metrics.csv`
6. **Combine** all per-pair CSVs into `outputs/enriched/all_domains_combined.csv`

After step 5 (merge), run the WL and attribute reach stages for each pair to populate `outputs/wl/` and `outputs/attr_dist/`. `compare_stage.py` reads those pre-computed files when building the distance map or domain summary — it does **not** recompute them on-the-fly. See **Steps 4 and 5** under "Running Individual Stages Manually" for the per-pair commands, or run the batch helper below:

```bash
# Batch: re-run WL + attr_dist for all merged pairs in one shot
python - <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, 'enriched_ontology_matching')
from logmap_runner import _safe_local
from wl_kernel_matcher import run_wl_stage
from attribute_reach import run_attr_dist_stage

inputs  = Path('enriched_ontology_matching/inputs')
merged  = Path('enriched_ontology_matching/outputs/merged')
wl_dir  = Path('enriched_ontology_matching/outputs/wl');        wl_dir.mkdir(exist_ok=True)
ad_dir  = Path('enriched_ontology_matching/outputs/attr_dist'); ad_dir.mkdir(exist_ok=True)

model_data = {}
for d in inputs.iterdir():
    if not d.is_dir(): continue
    for jf in d.glob('*.json'):
        data = json.loads(jf.read_text(encoding='utf-8'))
        name = data.get('modelName', '')
        if name: model_data[_safe_local(name)] = data

for f in sorted(merged.glob('*_metrics.csv')):
    stem = f.stem.replace('_metrics', '')
    if '_vs_' not in stem: continue
    a_key, b_key = stem.split('_vs_')[0], stem.split('_vs_')[1]
    da, db = model_data.get(a_key), model_data.get(b_key)
    if not da or not db: continue
    run_wl_stage(da, db, f, wl_dir / (f.stem + '_wl.csv'))
    run_attr_dist_stage(da, db, ad_dir / (f.stem + '_attr_dist.csv'))
EOF
```

### Bring your own input models

```bash
eom-run --inputs-dir path/to/my_models --domains MyDomain
eom-compare --domain-summary MyDomain --out path/to/my_models/MyDomain_summary.json
```

Expected layout: `<inputs-dir>/<Domain>/*.json`

### Run multiple domains

```bash
eom-run --inputs-dir enriched_ontology_matching/inputs --domains Automobile Hospital University

for domain in Automobile Hospital University; do
  eom-compare --domain-summary $domain
done
```

### Re-run with cached results

```bash
eom-run --inputs-dir enriched_ontology_matching/inputs --domains Automobile --skip-existing
```

### Use only one structural matcher

```bash
eom-run --matcher aml
# or: --matcher logmap
```

---

## Cross-Domain Analysis and Interactive Map (Secondary)

Run all within-domain pairs first, then add cross-domain pairs.

```bash
# Step 1 — run all 630 pairs (within-domain + cross-domain), skip completed ones
eom-run --cross-domain --skip-existing

# Step 2 — generate the interactive distance map
eom-compare
```

Output: `enriched_ontology_matching/outputs/ontology_map.html` — open in any browser, no server required.

To verify all pairs are present before generating the map:

```bash
ls enriched_ontology_matching/outputs/merged/ | grep "_metrics.csv" | wc -l
# Should print 630
```

---

## Running Individual Stages Manually

Useful for debugging a single pair. Assumes the venv is active and commands run from the repository root.

### Step 1: Structural Matching + Semantic Discovery

```bash
.venv/Scripts/python.exe enriched_ontology_matching/enriched_matcher.py \
    enriched_ontology_matching/pairs/auto_V1_V2.json
```

Output: `enriched_ontology_matching/outputs/enriched/<ModelA>_vs_<ModelB>.csv`

Add `--matcher logmap` or `--matcher aml` to use only one structural matcher.

### Step 2: Sentence Embeddings

```bash
python enriched_ontology_matching/semantic_encoder.py \
    --a           enriched_ontology_matching/inputs/Automobile/automobile_model_v1.json \
    --b           enriched_ontology_matching/inputs/Automobile/automobile_model_v2.json \
    --matches-csv enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --out-csv     enriched_ontology_matching/outputs/embeddings/auto_V1_V2_emb.csv
```

### Step 3: Merge into Metrics CSV

```bash
python enriched_ontology_matching/merge_stage.py \
    --enriched enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --emb      enriched_ontology_matching/outputs/embeddings/auto_V1_V2_emb.csv \
    --out      enriched_ontology_matching/outputs/merged/<stem>_metrics.csv
```

### Step 4: WL Kernel + Shape Metrics

```bash
python enriched_ontology_matching/wl_kernel_matcher.py \
    --a       enriched_ontology_matching/inputs/Automobile/automobile_model_v1.json \
    --b       enriched_ontology_matching/inputs/Automobile/automobile_model_v2.json \
    --metrics enriched_ontology_matching/outputs/merged/<stem>_metrics.csv \
    --out     enriched_ontology_matching/outputs/wl/<stem>_metrics_wl.csv
```

Output: one-row CSV with `wl_structural`, `shape_sim` (and sub-components: `degree_sim`, `spectral_sim`, `clustering_sim`, `betweenness_sim`).

### Step 5: Attribute Reach Distribution

```bash
python - <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, 'enriched_ontology_matching')
from attribute_reach import run_attr_dist_stage

da = json.loads(Path('enriched_ontology_matching/inputs/Automobile/automobile_model_v1.json').read_text())
db = json.loads(Path('enriched_ontology_matching/inputs/Automobile/automobile_model_v2.json').read_text())
run_attr_dist_stage(da, db,
    out_csv=Path('enriched_ontology_matching/outputs/attr_dist/<stem>_metrics_attr_dist.csv'))
EOF
```

Output: one-row CSV with `attr_dist_sim`. `compare_stage.py` multiplies this by `avg_entity_wup` from the merged CSV to produce `attr_weighted`.

---

## Output Structure

```
enriched_ontology_matching/
├── inputs/
│   ├── Automobile/      # 6 input model JSONs
│   ├── Coffee/
│   ├── Homebrewing/
│   ├── Hospital/
│   ├── SmartHome/
│   └── University/
├── pairs/
│   └── auto_V1_V2.json                         # generated pair JSON (json_a + json_b)
├── summaries/
│   └── Automobile_summary.json                 # domain distance summary JSON
└── outputs/
    ├── enriched/
    │   ├── <ModelA>_vs_<ModelB>.csv            # Step 1 per-pair results
    │   └── all_domains_combined.csv            # all pairs concatenated
    ├── embeddings/
    │   └── <key>_emb.csv                       # Step 2 embedding cosine scores
    ├── merged/
    │   └── <ModelA>_vs_<ModelB>_metrics.csv    # Step 3 final metrics CSV (5 columns)
    ├── wl/
    │   └── <ModelA>_vs_<ModelB>_wl.csv         # WL kernel + shape scores (computed by compare_stage)
    ├── attr_dist/
    │   └── <ModelA>_vs_<ModelB>_attr_dist.csv  # attribute reach scores (computed by compare_stage)
    └── ontology_map.html                       # interactive distance map
```

`<key>` is a short domain-pair identifier (e.g. `auto_V1_V2`, `hosp_V1_Net1`).

---

## Merged CSV Format

Each `outputs/merged/*_metrics.csv` has one row per entity pair and 5 columns:

| Column | Type | Description |
|--------|------|-------------|
| `entity_a` | string | Entity name from model A |
| `entity_b` | string | Entity name from model B |
| `matched` | 0 / 1 | 1 if AML or LogMap confirmed this pair; contributes to `lexical_sim` via `max(matched, cosine_avg)` |
| `wup` | float 0–1 | Blended Wu-Palmer: `(max_wup + avg_wup) / 2` across CamelCase token combinations |
| `cosine_avg` | float 0–1 | Token-average sentence-embedding cosine similarity (rescaled from [−1, 1] to [0, 1]) |

> **Note on `wup`:** for multi-word entity names (e.g. `CoffeeMakerAssembly`), names are CamelCase-tokenised and WUP is evaluated over all possible token-pair combinations. The merged value is `(max_wup + avg_wup) / 2`.

---

## Composite Scoring and Metric Weights

`compare_stage.py` combines four orthogonal dimensions into a single weighted composite score. Each dimension captures a different aspect of model similarity:

| Dimension | Inputs | What it captures |
|-----------|--------|-----------------|
| `lexical_sim` | `matched`, `cosine_avg`, `wup` from merged CSV | Name-level similarity: `max(max(matched, cosine_avg), wup≥0.75 else 0)` |
| `wl_structural` | `wl_structural` from `outputs/wl/` CSV | Local edge-type motifs: WL kernel over edge-type-labelled graphs |
| `shape_sim` | `shape_sim` from `outputs/wl/` CSV | Global graph topology: `avg(degree_sim, spectral_sim, clustering_sim, betweenness_sim)` |
| `attr_weighted` | `attr_dist_sim` × `avg_entity_wup` from `outputs/attr_dist/` | Attribute reachability scaled by lexical confidence: `attr_dist_sim × avg_wup` |

`lexical_sim` gates WUP at 0.75: scores below this threshold are treated as 0 so weakly-similar token pairs do not inflate the lexical signal.

`shape_sim` sub-components:
- **`degree_sim`** — cosine similarity of degree sequences (hub/leaf structure)
- **`spectral_sim`** — ratio of leading eigenvalues of the adjacency matrices
- **`clustering_sim`** — similarity of mean clustering coefficients (triangle density)
- **`betweenness_sim`** — cosine of normalised betweenness-centrality vectors (bridge nodes)

`attr_weighted` multiplies attribute-reach distribution similarity by the mean WUP across matched entity pairs, so it is reliable only when the entity correspondence is already semantically meaningful.

Each dimension weight blends uniform weight (1/4) with its sparsity fill rate, then renormalises:

```
w_raw[m] = (1/4 + fill_rate[m]) / 2
```

This ensures sparse dimensions are down-weighted without any fully-populated dimension dominating unconditionally. These 4 dimension scores and their blended weights appear in the domain summary JSON.

---

## Domain Distance Summary JSON

`eom-compare --domain-summary` reads the merged metrics CSVs for all within-domain pairs and produces a JSON summary.

```bash
eom-compare --domain-summary Automobile
# writes: enriched_ontology_matching/summaries/Automobile_summary.json

# Custom output path:
eom-compare --domain-summary Automobile --out my_results/auto_summary.json
```

### Summary JSON schema

```json
{
  "domain": "Automobile",
  "n_ontologies": 6,
  "n_pairs": 15,
  "metric_weights": {
    "lexical_sim":   0.312,
    "wl_structural": 0.241,
    "shape_sim":     0.228,
    "attr_weighted": 0.219
  },
  "average_distance": 0.376,
  "average_composite": 0.675,
  "pairs": [
    {
      "ont_a": "Automobile_Model_V1_SystemCentric",
      "ont_b": "Automobile_Model_V2_ComponentCentric",
      "distance": 0.21,
      "composite": 0.84,
      "n_entity_pairs": 291,
      "metrics": {
        "lexical_sim":   {"mean": 0.874, "weight": 0.312},
        "wl_structural": {"mean": 0.731, "weight": 0.241},
        "shape_sim":     {"mean": 0.612, "weight": 0.228},
        "attr_weighted": {"mean": 0.558, "weight": 0.219}
      }
    }
  ]
}
```

- **`metric_weights`** — blended weight for each of the 4 orthogonal dimensions; blend of uniform (1/4) and sparsity fill rate, renormalised to sum to 1
- **`average_distance`** — weighted Euclidean distance averaged over all within-domain pairs (lower = more similar)
- **`average_composite`** — dimension-weighted mean similarity score averaged over all pairs (higher = more similar)
- **`pairs`** — all within-domain pairs sorted by `distance` ascending; each pair reports per-dimension `mean` and `weight`

---

## Interactive Distance Map

`eom-compare` (no arguments) reads all merged CSVs and writes `outputs/ontology_map.html`.

- **Node position** — encodes weighted Euclidean distance between ontologies
- **Edge width / opacity** — encode pairwise composite similarity (thicker = more similar)
- **Node colour** — encodes domain (Automobile: blue, Hospital: red, University: green, Coffee: brown, Homebrewing: amber, SmartHome: purple)

### Edge Views (toggle buttons in the HTML)

| Button | Description |
|--------|-------------|
| **MST** (default) | Minimum spanning tree — connects all nodes via highest-similarity paths |
| **Top-3 neighbors** | Each ontology's 3 closest peers |
| **Top-5 neighbors** | Each ontology's 5 closest peers |
| **All edges** | All pairwise edges |
| **Nodes only** | Positions only, no edges |

Click a domain in the legend to hide/show its nodes and edges. Double-click to isolate one domain.

---

## Codebase Map

| File | Purpose |
|------|---------|
| `run_all_pairs.py` | **Main entry point** — batch runner for within-domain (and optionally cross-domain) pairs; supports `--inputs-dir`, `--domains`, `--skip-existing`, `--cross-domain` |
| `compare_stage.py` | **Summary + visualisation** — `--domain-summary DOMAIN` outputs a JSON distance summary; no args generates the interactive HTML map; reads pre-computed WL and attr_dist CSVs from `outputs/wl/` and `outputs/attr_dist/` |
| `enriched_matcher.py` | Structural matching (AML + LogMap), semantic discovery (WN + CN), WUP backup for orphan entities |
| `semantic_encoder.py` | Sentence-embedding cosine similarity (`cosine_avg`, rescaled to [0, 1]) |
| `wl_kernel_matcher.py` | WL graph kernel (`wl_structural`) and graph-shape sub-metrics (`degree_sim`, `spectral_sim`, `clustering_sim`, `betweenness_sim` → `shape_sim`) |
| `attribute_reach.py` | K-hop observable-type attribute reach distribution; `run_attr_dist_stage` produces `attr_dist_sim` used by `compare_stage.py` to compute `attr_weighted` |
| `merge_stage.py` | Joins enriched-matcher + embedding CSVs into a 5-column metrics-only CSV |
| `aml_runner.py` | Wrapper around the AML JAR |
| `logmap_runner.py` | Wrapper around the LogMap JAR |
| `root_comparator.py` | CamelCase splitting, WUP, ConceptNet CSV lookup |
| `model_normalizer.py` | Normalises model JSON schemas (entity/association field name variants) |
