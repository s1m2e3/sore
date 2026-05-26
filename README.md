# Enriched Ontology Matching Pipeline

A multi-layer semantic enrichment pipeline that runs structural matchers (AML + LogMap) over pairs of conceptual models, scores every entity pair across seven complementary metrics, and produces a domain distance summary JSON and an interactive distance map.

The primary use case is **within-domain analysis**: run all pairwise comparisons for a set of ontologies that share the same domain, then read the average pairwise distance between them as a JSON summary. Cross-domain comparisons and the interactive distance map are also supported as secondary outputs.

---

## Pipeline Overview

| Layer | Name | What it does |
|-------|------|--------------|
| **L0** | Neighbourhood Coherence | Validates each pair via `sqrt(WUP × cosine)` geometric mean over local graph neighbours |
| **L1** | Structural Matching | AML + LogMap find entity pairs; results merged and de-duplicated |
| **L2** | Semantic Discovery | WordNet + ConceptNet discover equivalence/subsumption candidates among unmatched entities |
| **L3** | Sentence Embedding | `paraphrase-MiniLM-L6-v2` cosine similarity per entity pair (rescaled to [0, 1]) |
| **L4** | GNN Similarity | Symmetric GNN aggregates sentence-embedded entity nodes and canonical edge labels over K hops; produces `gnn_sim` for every matched pair |
| **L5** | Containment Closure | Cross-encoder NLI (`nli-MiniLM2-L6-H768`) scores directional entailment between observable-type signatures for every A×B entity pair |
| **L6** | Merge | Joins all layer outputs into one metrics-only CSV per pair |
| **L7** | Distance Visualisation | Sparsity-weighted MDS map of all ontologies as interactive HTML |

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

**GPU note (recommended for L6 NLI):** install the CUDA-enabled PyTorch build before `pip install -e .` for faster NLI inference:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

The NLI model (`cross-encoder/nli-MiniLM2-L6-H768`) is downloaded automatically on first run and cached under `enriched_ontology_matching/models/`. Subsequent runs load it from the local cache.

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

`eom-run` will automatically run all 6 pipeline stages for every pair:

1. **Discover** all model JSON files in `<inputs-dir>/Automobile/`
2. **Generate pair JSONs** in `enriched_ontology_matching/pairs/`
3. **Run L1+L2** (AML + LogMap + WordNet + ConceptNet) → per-pair CSV in `outputs/enriched/`
4. **Run L0** (neighbourhood coherence with `sqrt(WUP × cosine)`) → `outputs/neighbourhood/`
5. **Run L3** (sentence embedding cosine, rescaled to [0, 1]) → `outputs/embeddings/`
6. **Run L4** (GNN similarity — symmetric K-hop embedding aggregation) → `outputs/gnn/`
7. **Run L5** (NLI containment closure — entailment between observable-type signatures) → `outputs/closure/`
8. **Run L6** (merge all metrics) → `outputs/merged/<ModelA>_vs_<ModelB>_metrics.csv`
9. **Combine** all per-pair CSVs into `outputs/enriched/all_domains_combined.csv`
10. **Generate** a Markdown report at `outputs/all_domains_results.md`

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

### L1+L2: Structural Matching + Semantic Discovery

```bash
.venv/Scripts/python.exe enriched_ontology_matching/enriched_matcher.py \
    enriched_ontology_matching/pairs/auto_V1_V2.json
```

Output: `enriched_ontology_matching/outputs/enriched/<ModelA>_vs_<ModelB>.csv`

Add `--matcher logmap` or `--matcher aml` to use only one structural matcher.

### L0: Neighbourhood Coherence

```bash
.venv/Scripts/python.exe enriched_ontology_matching/neighbourhood_coherence.py \
    --pair        enriched_ontology_matching/pairs/auto_V1_V2.json \
    --matches-csv enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --out-csv     enriched_ontology_matching/outputs/neighbourhood/auto_V1_V2_coherence.csv
```

### L3: Sentence Embeddings

```bash
.venv/Scripts/python.exe enriched_ontology_matching/semantic_encoder.py \
    --csv enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --out enriched_ontology_matching/outputs/embeddings/auto_V1_V2_emb.csv
```

### L4: GNN Similarity

```bash
.venv/Scripts/python.exe enriched_ontology_matching/gnn_matcher.py \
    --a  enriched_ontology_matching/pairs/auto_V1_V2.json --key-a json_a \
    --b  enriched_ontology_matching/pairs/auto_V1_V2.json --key-b json_b \
    --hops 2 --top-pairs 20
```

Output: `enriched_ontology_matching/outputs/gnn/<key>_gnn.csv`

The GNN uses undirected adjacency — inverse-expressed associations (A→B vs B←A) produce the same neighbourhood signal. Entity nodes are embedded with `paraphrase-MiniLM-L6-v2`; edge labels use the canonical relation type.

### L5: Containment Closure (NLI Entailment)

Requires the two raw model JSON files (not the pair JSON):

```bash
.venv/Scripts/python.exe enriched_ontology_matching/containment_closure.py \
    --json-a enriched_ontology_matching/inputs/Automobile/Automobile_Model_V1_SystemCentric.json \
    --json-b enriched_ontology_matching/inputs/Automobile/Automobile_Model_V2_ComponentCentric.json \
    --out    enriched_ontology_matching/outputs/closure/auto_V1_V2_closure.csv
```

The NLI model is downloaded and cached automatically on first run.

### L6: Merge into Metrics CSV

```bash
.venv/Scripts/python.exe enriched_ontology_matching/merge_stage.py \
    --enriched enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --nbr      enriched_ontology_matching/outputs/neighbourhood/auto_V1_V2_coherence.csv \
    --emb      enriched_ontology_matching/outputs/embeddings/auto_V1_V2_emb.csv \
    --gnn      enriched_ontology_matching/outputs/gnn/auto_V1_V2_gnn.csv \
    --closure  enriched_ontology_matching/outputs/closure/auto_V1_V2_closure.csv \
    --out      enriched_ontology_matching/outputs/merged/<stem>_metrics.csv
```

---

## Output Structure

```
enriched_ontology_matching/
├── models/
│   └── cross-encoder--nli-MiniLM2-L6-H768/    # NLI model cache (auto-downloaded)
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
    │   ├── <ModelA>_vs_<ModelB>.csv            # L1+L2 per-pair results
    │   └── all_domains_combined.csv            # all pairs concatenated
    ├── neighbourhood/
    │   └── <key>_coherence.csv                 # L0 coherence scores
    ├── embeddings/
    │   └── <key>_emb.csv                       # L3 embedding cosine scores
    ├── gnn/
    │   └── <key>_gnn.csv                       # L4 GNN similarity scores
    ├── closure/
    │   └── <ModelA>_vs_<ModelB>_closure.csv    # L5 NLI entailment scores
    ├── merged/
    │   └── <ModelA>_vs_<ModelB>_metrics.csv    # final metrics CSV (L6, 12 columns)
    ├── all_domains_results.md                  # human-readable summary report
    └── ontology_map.html                       # interactive distance map
```

`<key>` is a short domain-pair identifier (e.g. `auto_V1_V2`, `hosp_V1_Net1`).

---

## Merged CSV Format

Each `outputs/merged/*_metrics.csv` has one row per entity pair and 12 columns:

| Column | Type | Description |
|--------|------|-------------|
| `entity_a` | string | Entity name from model A |
| `entity_b` | string | Entity name from model B |
| `matched` | 0 / 1 | 1 if AML or LogMap confirmed this pair; used in `lexical_sim` via `max(matched, cosine_avg)` |
| `wup` | float 0–1 | Blended Wu-Palmer: `(max_wup + avg_wup) / 2` across token combinations |
| `coherence_sym` | float 0–1 | Symmetric neighbourhood coherence (`sqrt(WUP × cosine)` geometric mean) |
| `verb_coherence` | float 0–1 | Jaccard overlap of canonical relation types used by each entity (inverse-expressed associations resolve to the same canonical type, so inverse pairs score 1.0) |
| `attr_reach_sim` | float 0–1 | Weighted Jaccard similarity between the K-hop observable-type attribute-reach signatures of each entity (`Σ min / Σ max` over all observable types) |
| `cosine_avg` | float 0–1 | Token-average sentence embedding cosine similarity (rescaled from [−1, 1] to [0, 1]) |
| `gnn_sim` | float 0–1 | Symmetric GNN similarity: K-hop sentence-embedding aggregation over entity nodes and canonical edge labels |
| `entailment_a_covers_b` | float 0–1 | P(A's observable-type signature entails B's) from NLI cross-encoder |
| `entailment_b_covers_a` | float 0–1 | P(B's observable-type signature entails A's) from NLI cross-encoder |
| `entailment_f1` | float 0–1 | `max(entailment_a_covers_b, entailment_b_covers_a)` — captures the strongest directional relationship (equivalence or subsumption) |

> **Note on `entailment_f1` naming:** the column is named `entailment_f1` for historical reasons but its value is the directional max, not a harmonic mean.

> **Note on `wup`:** for multi-word entity names (e.g. `CoffeeMakerAssembly`), names are tokenised and WUP is evaluated over all possible token-pair combinations. The merged value is `(max_wup + avg_wup) / 2`.

---

## Composite Scoring and Metric Weights

Before scoring, `compare_stage.py` consolidates the 12 raw columns into **4 orthogonal dimensions** by averaging pairs of metrics that are highly correlated (r ≥ 0.77). This removes redundant signal so no single underlying phenomenon dominates the composite score.

| Dimension | Formula | Pairwise correlation |
|-----------|---------|---------------------|
| `lexical_sim` | `avg(max(matched, cosine_avg), wup)` | r = 0.768 between cosine and wup |
| `coherence_sym` | (standalone) | — |
| `graph_sim` | `avg(verb_coherence, gnn_sim)` | r = 0.778 |
| `transfer_sim` | `avg(attr_reach_sim, entailment_f1)` | r = 0.776 |

`lexical_sim` uses `max(matched, cosine_avg)` as its first term: the structural-matcher binary confirmation (0/1) and the sentence-embedding cosine (rescaled to [0, 1]) carry the same lexical signal, so the stronger of the two is preferred before averaging with WUP.

Each dimension weight blends uniform weight (1/4) with its sparsity fill rate, then renormalises:

```
w_raw[m] = (1/4 + fill_rate[m]) / 2
```

This ensures sparse dimensions are down-weighted without any fully-populated dimension dominating unconditionally. These 4 dimension scores and their blended weights are what appear in the domain summary JSON.

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
    "coherence_sym": 0.241,
    "graph_sim":     0.228,
    "transfer_sim":  0.219
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
        "coherence_sym": {"mean": 0.731, "weight": 0.241},
        "graph_sim":     {"mean": 0.612, "weight": 0.228},
        "transfer_sim":  {"mean": 0.558, "weight": 0.219}
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

## Association Relation Mapping

`relation_normalizer.py` scans all conceptual-model JSONs, extracts every unique association name, and maps it to a canonical ConceptNet / RO / SSN relation using a combined WUP + BERT cosine similarity score.

```bash
python enriched_ontology_matching/relation_normalizer.py
```

Outputs:

| Output | Description |
|--------|-------------|
| `config/relation_map.json` | Association name → `{canonical, score, wup, bert, method, …}` |
| `enriched_ontology_matching/association_inventory.csv` | Full table with all scores and participant info (not committed) |

Options:

```bash
# Flag mappings below a combined score threshold (default: 0.5)
python enriched_ontology_matching/relation_normalizer.py --threshold 0.5

# Adjust WUP vs BERT weighting (default: 0.5/0.5)
python enriched_ontology_matching/relation_normalizer.py --wup-weight 0.7

# Scan a custom input directory
python enriched_ontology_matching/relation_normalizer.py --input-dir path/to/my_models
```

Before the first run, seed ConceptNet exemplar phrases so BERT has short-phrase targets:

```bash
python enriched_ontology_matching/seed_exemplars.py
```

---

## Codebase Map

| File | Purpose |
|------|---------|
| `run_all_pairs.py` | **Main entry point** — batch runner for within-domain (and optionally cross-domain) pairs (L0→L7); supports `--inputs-dir`, `--domains`, `--skip-existing`, `--cross-domain` |
| `compare_stage.py` | **Summary + visualisation** — `--domain-summary DOMAIN` outputs a JSON distance summary; no args generates the interactive HTML map |
| `enriched_matcher.py` | L1 (AML + LogMap structural merge) + L2 (WN+CN discovery) |
| `neighbourhood_coherence.py` | L0 — graph neighbourhood semantic coherence (`coherence_sym`, `verb_coherence`, `attr_reach_sim`) |
| `semantic_encoder.py` | L3 — sentence embedding cosine similarity (rescaled to [0, 1]) |
| `gnn_matcher.py` | L4 — symmetric GNN similarity: K-hop aggregation over sentence-embedded entity nodes and canonical edge labels |
| `containment_closure.py` | L5 — NLI cross-encoder entailment between observable-type signatures; model cached under `models/` |
| `merge_stage.py` | L6 — join all stages into metrics-only CSV (12 columns) |
| `aml_runner.py` | Wrapper around the AML JAR |
| `logmap_runner.py` | Wrapper around the LogMap JAR |
| `root_comparator.py` | CamelCase splitting, WUP, ConceptNet CSV lookup |
| `generate_report.py` | Renders a human-readable Markdown report from the combined CSV |
| `relation_normalizer.py` | Maps association names to canonical relations via WUP + BERT; writes `config/relation_map.json` |
| `seed_exemplars.py` | Seeds ConceptNet exemplar phrases into `config/canonical_relations.json` |
| `regenerate_domains.py` | Force-regenerates the full pipeline for selected domains and produces summary JSONs |
| `summary_to_csv.py` | Converts domain summary JSONs (`summaries/*_summary.json`) to flat CSV files |
| `attribute_reach.py` | K-hop observable-type attribute reach computation (used by neighbourhood_coherence.py) |
| `model_normalizer.py` | Normalises model JSON schemas (entity/association field name variants) |
