# Enriched Ontology Matching Pipeline

A multi-layer semantic enrichment pipeline that runs structural matchers (AML + LogMap) over pairs of conceptual models, scores every entity pair across four complementary similarity metrics, and produces an interactive 2D distance map of all ontologies.

---

## Pipeline Overview

| Layer | Name | What it does |
|-------|------|--------------|
| **L1** | Structural Matching | AML + LogMap find entity pairs; results merged and de-duplicated |
| **L2** | Semantic Discovery | WordNet + ConceptNet discover equivalence/subsumption candidates among unmatched entities |
| **L0** | Neighbourhood Coherence | Validates each pair via `sqrt(WUP × cosine)` geometric mean over local graph neighbours |
| **L3** | Lin-IC Scoring | Corpus-based Lin Information Content (Brown corpus) scores every pair |
| **L4** | Sentence Embedding | `paraphrase-MiniLM-L6-v2` cosine similarity per entity pair |
| **L5** | Merge | Joins all layer outputs into one metrics-only CSV |
| **L6** | Cross-Domain + Full Pipeline | Semantic-only pipeline for cross-domain pairs; enriches all remaining pairs with WUP + coherence |
| **L7** | Distance Visualisation | Sparsity-weighted MDS map of all ontologies as interactive HTML |

---

## Quick Start — Full Cross-Domain Pipeline

To run the complete pipeline across all 18 models (6 Automobile, 6 Hospital, 6 University) and generate the interactive distance map:

```bash
# Step 1 — complete within-domain enrichment + run all cross-domain pairs
python enriched_ontology_matching/complete_and_crossdomain.py

# Step 2 — regenerate the distance map (also called automatically by step 1)
python enriched_ontology_matching/compare_stage.py
```

This produces `enriched_ontology_matching/outputs/ontology_map.html` — open in any browser.

---

## Interactive Distance Map

`compare_stage.py` reads all merged CSVs and renders a **2D MDS (Multidimensional Scaling)** scatter plot where:

- **Node position** encodes the sparsity-weighted Euclidean distance between ontologies
- **Edge width and opacity** encode pairwise composite similarity (thicker = more similar)
- **Edge colour** interpolates continuously from light gray (low similarity) to deep blue (high)
- **Node colour** encodes domain (blue = Automobile, red = Hospital, green = University)

### Composite Metric

For each ontology pair, a composite similarity score is derived from up to four metrics:

```
composite = Σ_m  fill_rate[m] · v_m  /  Σ_m  fill_rate[m]
```

Where `fill_rate[m]` is the global fraction of entity-pair rows where metric `m` is populated (non-blank, > 0). This **sparsity-weighted mean** automatically down-weights sparse metrics:

| Metric | Fill rate | Weight |
|--------|-----------|--------|
| `cosine_avg` (sentence embedding) | ~1.00 | 0.63 |
| `avg_wup` (WordNet Wu-Palmer) | ~0.26 | 0.16 |
| `lin_ic` (Lin Information Content) | ~0.24 | 0.15 |
| `coherence_sym` (neighbourhood coherence) | ~0.10 | 0.06 |

### Distance Matrix

```
d(A, B) = sqrt( Σ_m  fill_rate[m] · (1 − v_m)² )  /  sqrt( Σ_m  fill_rate[m] )
```

This weighted Euclidean distance is the input to MDS, ensuring node positions faithfully reflect all four metrics proportionally to their reliability.

### Edge Views (toggle buttons in the HTML)

| Button | Description |
|--------|-------------|
| **MST** (default) | Minimum spanning tree — 17 edges connecting all 18 ontologies via highest-similarity paths |
| **Top-3 neighbors** | Each ontology's 3 closest peers |
| **Top-5 neighbors** | Each ontology's 5 closest peers |
| **All edges** | All 153 pairwise edges |
| **Nodes only** | Positions only, no edges |

---

## Prerequisites

### 1. Python 3.10+

```bash
python --version   # 3.10 or higher
```

### 2. Java 11+ (for AML and LogMap)

```bash
java -version   # must be 11 or higher
```

Download from [Adoptium](https://adoptium.net/).

### 3. LogMap JAR

Already included:

```
enriched_ontology_matching/tools/logmap/logmap-matcher-4.0.jar
```

### 4. AML JAR

Download `AgreementMakerLight.jar` from the [AML releases page](https://github.com/AgreementMakerLight/AML-Project/releases) and place it at:

```
enriched_ontology_matching/tools/AML/AML_v3.2/AgreementMakerLight.jar
```

### 5. ConceptNet (optional)

Without a local file the pipeline falls back to the ConceptNet API (rate-limited). For bulk runs:

```bash
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
gunzip conceptnet-assertions-5.7.0.csv.gz
```

Place at `ontology_matching/inputs/conceptnet-assertions-5.7.0.csv/assertions.csv`.

### 6. Input Models

The pipeline expects conceptual-model JSON files organised under `enriched_ontology_matching/models/`:

```
enriched_ontology_matching/models/
  Automobile/
    Automobile_Model_V1_SystemCentric.json
    Automobile_Model_V2_ComponentCentric.json
    Automobile_Model_V3_FunctionalDomain.json
    Automobile_Component_Network_Model_Mechanical_and_Structural_Network.json
    Automobile_Component_Network_Model_Packaged_Assemblies_Network.json
    Automobile_Component_Network_Model_Serviceable_Parts_Interaction_Network.json
  Hospital/
    Hospital_Model_V1_DepartmentalStructure.json
    Hospital_Model_V2_PatientCentric.json
    Hospital_Model_V3_ClinicalWorkflow.json
    Hospital_Facility_Resource_Network_Model_Equipment_and_Technology_Network.json
    Hospital_Facility_Resource_Network_Model_Serviceable_Facility_Parts_Network.json
    Hospital_Facility_Resource_Network_Model_Spatial_Infrastructure_Network.json
  University/
    University_Model_V1_InstitutionalStructure.json
    University_Model_V2_AcademicProgramCentric.json
    University_Model_V3_ResearchFocused.json
    University_Academic_Lifecycle_Model.json
    University_Academic_Resource_Network_Model_Campus_Services_Network.json
    University_Academic_Resource_Network_Model_Research_and_Innovation_Network.json
```

Each JSON follows this schema (V-model format):

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

Network-schema models use `"name"` / `"attributes"` for entities and `"participants"` for associations.

---

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate          # macOS / Linux

# Install dependencies
pip install -r enriched_ontology_matching/requirements.txt

# Download NLTK data (one-time)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('wordnet_ic'); nltk.download('brown')"
```

---

## Running Individual Stages Manually

### Step 1 — L1+L2: Structural Matching + Semantic Discovery

```bash
python enriched_ontology_matching/enriched_matcher.py \
    enriched_ontology_matching/pairs/auto_V1_V2.json
```

Output: `outputs/enriched/<ModelA>_vs_<ModelB>.csv`

### Step 2 — L0: Neighbourhood Coherence

```bash
python enriched_ontology_matching/neighbourhood_coherence.py \
    --pair        enriched_ontology_matching/pairs/auto_V1_V2.json \
    --matches-csv enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --out-csv     enriched_ontology_matching/outputs/neighbourhood/auto_V1_V2_coherence.csv
```

### Step 3 — L3: Lin-IC Scoring

```bash
python enriched_ontology_matching/lin_ic_stage.py \
    --csv enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --out enriched_ontology_matching/outputs/lin_ic/auto_V1_V2_lin_ic.csv
```

### Step 4 — L4: Sentence Embeddings

```bash
python enriched_ontology_matching/semantic_encoder.py \
    --csv enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --out enriched_ontology_matching/outputs/embeddings/auto_V1_V2_emb.csv
```

### Step 5 — L5: Merge

```bash
python enriched_ontology_matching/merge_stage.py \
    --enriched enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --nbr      enriched_ontology_matching/outputs/neighbourhood/auto_V1_V2_coherence.csv \
    --lin-ic   enriched_ontology_matching/outputs/lin_ic/auto_V1_V2_lin_ic.csv \
    --emb      enriched_ontology_matching/outputs/embeddings/auto_V1_V2_emb.csv \
    --out      enriched_ontology_matching/outputs/merged/<stem>_metrics.csv
```

### Step 6 — L6: Cross-Domain + Full Pipeline

```bash
python enriched_ontology_matching/complete_and_crossdomain.py
```

Runs all C(18,2) = 153 ontology pairs. For pairs not covered by AML/LogMap (cross-domain), uses sentence-embedding cosine similarity to find candidate matches, then enriches with WUP and neighbourhood coherence.

### Step 7 — L7: Regenerate Distance Map

```bash
python enriched_ontology_matching/compare_stage.py
```

---

## Output Structure

```
enriched_ontology_matching/
├── models/
│   ├── Automobile/         # 6 model JSONs
│   ├── Hospital/           # 6 model JSONs
│   └── University/         # 6 model JSONs
├── pairs/
│   └── auto_V1_V2.json     # generated pair JSON (json_a + json_b)
└── outputs/
    ├── enriched/
    │   └── <A>_vs_<B>.csv          # L1+L2 per-pair CSV (with avg_wup)
    ├── neighbourhood/
    │   └── <key>_coherence.csv     # L0 coherence scores
    ├── lin_ic/
    │   └── <key>_lin_ic.csv        # L3 Lin-IC scores
    ├── embeddings/
    │   └── <key>_emb.csv           # L4 embedding cosine scores
    ├── merged/
    │   └── <A>_vs_<B>_metrics.csv  # final 7-column metrics CSV
    └── ontology_map.html            # interactive 2D distance map
```

---

## Merged CSV Format

Each `outputs/merged/*_metrics.csv` has one row per entity pair and 7 columns:

| Column | Type | Description |
|--------|------|-------------|
| `entity_a` | string | Entity name from model A |
| `entity_b` | string | Entity name from model B |
| `matched` | 0 / 1 | 1 if AML or LogMap confirmed this pair |
| `avg_wup` | float 0–1 | Mean WordNet Wu-Palmer similarity across token combinations |
| `lin_ic` | float 0–1 | Lin Information Content similarity (best token pair) |
| `coherence_sym` | float 0–1 | Symmetric neighbourhood coherence (`sqrt(WUP × cosine)`) |
| `cosine_avg` | float 0–1 | Sentence embedding cosine similarity |

---

## Codebase Map

| File | Purpose |
|------|---------|
| `run_all_pairs.py` | Batch runner for within-domain pairs (L1→L5) |
| `enriched_matcher.py` | L1 (AML + LogMap) + L2 (WN+CN discovery) |
| `neighbourhood_coherence.py` | L0 — graph neighbourhood semantic coherence |
| `lin_ic_stage.py` | L3 — Lin Information-Content scoring |
| `semantic_encoder.py` | L4 — sentence embedding cosine similarity |
| `merge_stage.py` | L5 — join all stages into metrics-only CSV |
| `complete_and_crossdomain.py` | L6 — cross-domain semantic pipeline + full enrichment for all 153 pairs |
| `compare_stage.py` | L7 — sparsity-weighted MDS + interactive Plotly HTML |
| `aml_runner.py` | Wrapper around the AML JAR |
| `logmap_runner.py` | Wrapper around the LogMap JAR |
| `root_comparator.py` | CamelCase splitting, WUP, ConceptNet CSV lookup |
| `generate_report.py` | Renders a human-readable MD report from the combined CSV |
