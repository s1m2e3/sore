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
| **L5** | Merge | Joins all layer outputs into one metrics-only CSV per pair |
| **L6** | Distance Visualisation | Sparsity-weighted MDS map of all ontologies as interactive HTML |

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

Without a local file the pipeline falls back to the ConceptNet REST API (rate-limited).
For bulk runs, download the assertions CSV (≈ 1.5 GB compressed):

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

All commands are run from the **repository root**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download required NLTK data (one-time)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('wordnet_ic'); nltk.download('brown')"
```

After activation, use `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (macOS/Linux) for all pipeline commands.

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

Each JSON follows this schema:

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

## End-to-End Run

### Step 1 — Run all pairs (within-domain + cross-domain)

```bash
.venv/Scripts/python.exe enriched_ontology_matching/run_all_pairs.py --cross-domain
```

This runs the full 5-stage pipeline (L1→L5) for every pair of models — 630 pairs total (C(36, 2) across all 6 domains × 6 models). Each pair produces one `outputs/merged/*_metrics.csv`.

Options:
- `--skip-existing` — skip pairs whose merged CSV already exists (safe to re-run)
- `--domains Automobile Hospital` — limit to specific domains
- `--matcher aml` / `--matcher logmap` — use only one structural matcher (default: both)

Expected runtime: **2–3 hours** on first run; near-instant with `--skip-existing`.

### Step 2 — Generate the interactive distance map

```bash
.venv/Scripts/python.exe enriched_ontology_matching/compare_stage.py
```

Reads all `outputs/merged/*_metrics.csv` and writes `enriched_ontology_matching/outputs/ontology_map.html`.
Open the file in any browser — no server required.

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
│   └── auto_V1_V2.json                      # generated pair JSON (json_a + json_b)
└── outputs/
    ├── enriched/
    │   ├── <ModelA>_vs_<ModelB>.csv          # L1+L2 per-pair results
    │   └── all_domains_combined.csv          # all pairs concatenated
    ├── neighbourhood/
    │   └── <key>_coherence.csv              # L0 coherence scores
    ├── lin_ic/
    │   └── <key>_lin_ic.csv                 # L3 Lin-IC scores
    ├── embeddings/
    │   └── <key>_emb.csv                    # L4 embedding cosine scores
    ├── merged/
    │   └── <ModelA>_vs_<ModelB>_metrics.csv # final metrics CSV (7 columns)
    ├── all_domains_results.md               # human-readable summary
    └── ontology_map.html                    # interactive 2D distance map
```

---

## Merged CSV Format

Each `outputs/merged/*_metrics.csv` has one row per entity pair and 7 columns:

| Column | Type | Description |
|--------|------|-------------|
| `entity_a` | string | Entity name from model A |
| `entity_b` | string | Entity name from model B |
| `matched` | 0 / 1 | 1 if AML or LogMap confirmed this pair |
| `wup` | float 0–1 | Blended Wu-Palmer: `(max_wup + avg_wup) / 2` across token combinations |
| `lin_ic` | float 0–1 | Lin Information Content similarity (best token pair, Brown corpus) |
| `coherence_sym` | float 0–1 | Symmetric neighbourhood coherence (`sqrt(WUP × cosine)`) |
| `cosine_avg` | float 0–1 | Sentence embedding cosine similarity |

---

## Interactive Distance Map

`compare_stage.py` renders a **2D MDS (Multidimensional Scaling)** scatter plot where:

- **Node position** encodes sparsity-weighted Euclidean distance between ontologies
- **Edge width / opacity** encode pairwise composite similarity (thicker = more similar)
- **Node colour** encodes domain

### Composite Metric

```
composite = Σ_m  fill_rate[m] · v_m  /  Σ_m  fill_rate[m]
```

`fill_rate[m]` is the global fraction of entity-pair rows where metric `m` is populated.
This automatically down-weights sparse metrics.

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
| `run_all_pairs.py` | **Main entry point** — batch runner for all within- and cross-domain pairs (L1→L5) |
| `compare_stage.py` | **HTML map generator** — sparsity-weighted MDS + interactive Plotly visualisation |
| `enriched_matcher.py` | L1 (AML + LogMap structural merge) + L2 (WN+CN discovery) |
| `neighbourhood_coherence.py` | L0 — graph neighbourhood semantic coherence |
| `lin_ic_stage.py` | L3 — Lin Information-Content scoring |
| `semantic_encoder.py` | L4 — sentence embedding cosine similarity |
| `merge_stage.py` | L5 — join all stages into metrics-only CSV |
| `aml_runner.py` | Wrapper around the AML JAR |
| `logmap_runner.py` | Wrapper around the LogMap JAR |
| `root_comparator.py` | CamelCase splitting, WUP, ConceptNet CSV lookup |
| `generate_report.py` | Renders a human-readable Markdown report from the combined CSV |
