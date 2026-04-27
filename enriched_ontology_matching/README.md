# Enriched Ontology Matching Pipeline

A five-layer semantic enrichment pipeline that runs structural matchers (AML + LogMap) over pairs of conceptual models and scores every entity pair using WordNet, ConceptNet, neighbourhood graph coherence, Lin Information-Content similarity, and sentence embedding cosine similarity.

The primary use case is **within-domain analysis**: run all pairwise comparisons for a set of ontologies that share the same domain, then read the average pairwise distance between them as a JSON summary. Cross-domain comparisons and the interactive 3-D distance map are also supported as secondary outputs.

## Pipeline Overview

| Layer | Name | What it does |
|-------|------|--------------|
| **L0** | Neighbourhood Coherence | Validates each pair via `sqrt(WUP × cosine)` geometric mean over local graph neighbours |
| **L1** | Structural Matching | AML + LogMap find entity pairs; results are merged and de-duplicated |
| **L2** | Semantic Discovery | WordNet + ConceptNet discover additional equivalence/subsumption candidates among *unmatched* entities |
| **L3** | Lin-IC Scoring | Corpus-based Lin Information Content (Brown corpus) scores every L1+L2 pair |
| **L4** | Sentence Embedding | `paraphrase-MiniLM-L6-v2` cosine similarity per entity pair |
| **L5** | Merge | Joins all layer outputs into one metrics-only CSV |

---

## Prerequisites

### 1. Python 3.10+

```bash
python --version   # 3.10 or higher
```

### 2. Java 11+

Both AML and LogMap require Java.

```bash
java -version   # must be 11 or higher
```

Download from [Adoptium](https://adoptium.net/).

### 3. LogMap JAR

Already included in this repo at:

```
enriched_ontology_matching/tools/logmap/logmap-matcher-4.0.jar
```

No download needed.

### 4. AML JAR

Download `AgreementMakerLight.jar` from the [AML releases page](https://github.com/AgreementMakerLight/AML-Project/releases) and place it at:

```
enriched_ontology_matching/tools/AML/AML_v3.2/AgreementMakerLight.jar
```

AML also requires its `store/` directory (config + stop list) to be present alongside the JAR. Extract the full AML release zip into `enriched_ontology_matching/tools/AML/AML_v3.2/`.

### 5. ConceptNet (optional but recommended)

Without a local file the pipeline falls back to the ConceptNet API (rate-limited).

For bulk runs, download the assertions CSV (≈ 1.5 GB compressed):

```bash
# https://github.com/commonsense/conceptnet5/wiki/Downloads
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
gunzip conceptnet-assertions-5.7.0.csv.gz
```

Place the extracted file at:

```
ontology_matching/inputs/conceptnet-assertions-5.7.0.csv/assertions.csv
```

### 6. Input Models (Constitutive JSON Files)

The pipeline expects conceptual-model JSON files organised by domain. `run_all_pairs.py` looks for them at:

```
enriched_ontology_matching/inputs/
  Automobile/       — 6 model JSONs  (V1, V2, V3, Net1, Net2, Net3)
  Coffee/           — 6 model JSONs
  Homebrewing/      — 6 model JSONs
  Hospital/         — 6 model JSONs
  SmartHome/        — 6 model JSONs
  University/       — 6 model JSONs
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

Network-schema models use `"name"` and `"participants"` instead of `"associationName"` / `"associationParticipants"`.

---

## Setup

All commands are run from the **repository root** (`sore/`).

### Step 0 — Create a virtual environment and install the package

```bash
# Create the venv
python -m venv .venv

# Activate it
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

# Install the package and all dependencies (editable install — changes to
# source files are reflected immediately without reinstalling)
pip install -e .

# Download required NLTK data (one-time)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('wordnet_ic'); nltk.download('brown')"
```

After installation two CLI commands become available in the active environment:

| Command | Description |
|---------|-------------|
| `eom-run` | Run the full pipeline (L1→L5) for one or more domains |
| `eom-compare` | Generate a domain distance summary JSON or the interactive HTML map |

These commands work from any directory once the venv is active.

---

## Generating Results (End-to-End)

The pipeline has two primary steps: run all pairwise comparisons for a domain, then compute the average distance summary.

### Quick start — within-domain analysis (primary use case)

```bash
# Step 1 — run all pairwise comparisons for one domain
#   --inputs-dir  path to folder containing domain subdirectories with model JSONs
#   --domains     which domain(s) to process
eom-run --inputs-dir enriched_ontology_matching/inputs --domains Automobile

# Step 2 — get the average pairwise distance summary as JSON
eom-compare --domain-summary Automobile
```

Output: `enriched_ontology_matching/summaries/Automobile_summary.json`

The JSON contains the domain name, number of ontologies, number of pairs, metric weights (fill-rate-based), `average_distance`, `average_composite`, and a ranked list of all pairs.

`run_all_pairs.py` will:

1. **Discover** all model JSON files in `<inputs-dir>/Automobile/`
2. **Generate pair JSONs** in `enriched_ontology_matching/pairs/`
3. **Run L1+L2** (AML + LogMap + WordNet + ConceptNet) → per-pair CSV in `outputs/enriched/`
4. **Run L0** (neighbourhood coherence with `sqrt(WUP × cosine)`) → `outputs/neighbourhood/`
5. **Run L3** (Lin-IC scoring) → `outputs/lin_ic/`
6. **Run L4** (sentence embedding cosine) → `outputs/embeddings/`
7. **Run L5** (merge all metrics) → `outputs/merged/<ModelA>_vs_<ModelB>_metrics.csv`
8. **Combine** all per-pair CSVs into `outputs/enriched/all_domains_combined.csv`
9. **Generate** a Markdown report at `outputs/all_domains_results.md`

### Bring your own input models

Point `--inputs-dir` at any directory that contains domain subdirectories with model JSON files:

```
my_models/
  MyDomain/
    Model_A.json
    Model_B.json
    Model_C.json
```

```bash
eom-run --inputs-dir path/to/my_models --domains MyDomain

eom-compare --domain-summary MyDomain --out path/to/my_models/MyDomain_summary.json
```

### Run multiple domains

```bash
eom-run --inputs-dir enriched_ontology_matching/inputs --domains Automobile Hospital University
```

Then generate a summary for each:

```bash
for domain in Automobile Hospital University; do
  eom-compare --domain-summary $domain
done
```

### Re-run with cached results

To skip pairs whose outputs already exist (e.g. after adding new models):

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

The pipeline also supports cross-domain comparisons and an interactive 3-D distance map. These are secondary outputs — run all within-domain pairs first, then add cross-domain pairs.

### Run all 630 pairs (within-domain + cross-domain) and generate HTML

```bash
# Step 1 — run all 630 pairs (within-domain and cross-domain)
eom-run --cross-domain

# Step 2 — generate the interactive distance map
eom-compare
```

`compare_stage.py` (called without arguments) reads all merged CSVs and writes `outputs/ontology_map.html`.

---

## Running Individual Stages Manually

These commands invoke the underlying scripts directly with Python — useful for debugging a single pair. They assume the venv is active and you are running from the repository root (`sore/`).

### Step 1 — L1+L2: Structural Matching + Semantic Discovery

```bash
.venv/Scripts/python.exe enriched_ontology_matching/enriched_matcher.py \
    enriched_ontology_matching/pairs/auto_V1_V2.json
```

Output: `enriched_ontology_matching/outputs/enriched/<ModelA>_vs_<ModelB>.csv`

To use only one matcher add `--matcher logmap` or `--matcher aml`.

### Step 2 — L0: Neighbourhood Coherence

Requires a pair JSON (for model graphs) and the enriched CSV from Step 1:

```bash
.venv/Scripts/python.exe enriched_ontology_matching/neighbourhood_coherence.py \
    --pair        enriched_ontology_matching/pairs/auto_V1_V2.json \
    --matches-csv enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --out-csv     enriched_ontology_matching/outputs/neighbourhood/auto_V1_V2_coherence.csv
```

Coherence is computed as `sqrt(WUP × cosine)` per neighbour pair (geometric mean), then averaged over all neighbours in each direction.

### Step 3 — L3: Lin-IC Scoring

```bash
.venv/Scripts/python.exe enriched_ontology_matching/lin_ic_stage.py \
    --csv enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --out enriched_ontology_matching/outputs/lin_ic/auto_V1_V2_lin_ic.csv
```

### Step 4 — L4: Sentence Embeddings

```bash
.venv/Scripts/python.exe enriched_ontology_matching/semantic_encoder.py \
    --csv enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --out enriched_ontology_matching/outputs/embeddings/auto_V1_V2_emb.csv
```

### Step 5 — L5: Merge into Metrics CSV

```bash
.venv/Scripts/python.exe enriched_ontology_matching/merge_stage.py \
    --enriched enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --nbr      enriched_ontology_matching/outputs/neighbourhood/auto_V1_V2_coherence.csv \
    --lin-ic   enriched_ontology_matching/outputs/lin_ic/auto_V1_V2_lin_ic.csv \
    --emb      enriched_ontology_matching/outputs/embeddings/auto_V1_V2_emb.csv \
    --out      enriched_ontology_matching/outputs/merged/<stem>_metrics.csv
```

---

## Output Structure

```
enriched_ontology_matching/
├── pairs/
│   └── auto_V1_V2.json                        # generated pair JSON (json_a + json_b)
├── outputs/
│   ├── enriched/
│   │   ├── <ModelA>_vs_<ModelB>.csv            # L1+L2 per-pair CSV
│   │   └── all_domains_combined.csv            # all pairs concatenated
│   ├── neighbourhood/
│   │   └── <key>_coherence.csv                 # L0 coherence scores
│   ├── lin_ic/
│   │   └── <key>_lin_ic.csv                    # L3 Lin-IC scores
│   ├── embeddings/
│   │   └── <key>_emb.csv                       # L4 embedding cosine scores
│   ├── merged/
│   │   └── <ModelA>_vs_<ModelB>_metrics.csv    # final metrics CSV
│   └── all_domains_results.md                  # human-readable summary report
└── tools/
    ├── logmap/
    │   └── logmap-matcher-4.0.jar
    └── AML/
        └── AML_v3.2/
            └── AgreementMakerLight.jar
```

`<key>` is a short domain-pair identifier (e.g. `auto_V1_V2`, `hosp_V1_Net1`).

---

## Merged CSV Format

Each `outputs/merged/*_metrics.csv` has one row per entity pair and 7 columns:

| Column | Type | Description |
|--------|------|-------------|
| `entity_a` | string | Entity name from model A |
| `entity_b` | string | Entity name from model B |
| `matched` | 0 / 1 | 1 if AML or LogMap (or both) found this pair |
| `wup` | float 0–1 | Blended WUP score: `(max_wup + avg_wup) / 2` across all token-pair combinations |
| `lin_ic` | float 0–1 | Lin Information Content similarity (best token pair, Brown corpus) |
| `coherence_sym` | float 0–1 | Symmetric neighbourhood coherence (`sqrt(WUP × cosine)` geometric mean) |
| `cosine_avg` | float 0–1 | Token-average sentence embedding cosine similarity |

Rows with `matched=0` are pairs discovered only by the semantic layers (L2).

> **Note on `wup`**: For multi-word entity names (e.g. `CoffeeMakerAssembly`), names are tokenised and WUP is evaluated over all possible token-pair combinations. The merged `wup` value is `(max_wup + avg_wup) / 2` — a blend that captures both the best single-token match and the overall average, penalising entities with unmatched tokens.

---

## Domain Distance Summary JSON

`compare_stage.py --domain-summary` reads the merged metrics CSVs for all within-domain pairs of a given domain and produces a JSON summary.

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
    "cosine_avg": 0.98,
    "wup": 1.0,
    "lin_ic": 0.95,
    "coherence_sym": 0.72
  },
  "average_distance": 0.376,
  "average_composite": 0.675,
  "pairs": [
    {
      "model_a": "Automobile_Model_V1_SystemCentric",
      "model_b": "Automobile_Model_V2_UserCentric",
      "distance": 0.21,
      "composite": 0.84
    }
  ]
}
```

- **`metric_weights`** — fill-rate of each metric across all available merged CSVs; used to weight the distance and composite calculations
- **`average_distance`** — sparsity-weighted Euclidean distance averaged over all within-domain pairs (lower = more similar)
- **`average_composite`** — fill-rate-weighted mean similarity score averaged over all pairs (higher = more similar)
- **`pairs`** — all within-domain pairs, sorted by `distance` ascending (most similar first)

---

## Regenerating the Ontology Distance Map (HTML, Advanced)

The interactive 2-D map (`outputs/ontology_map.html`) is generated by `compare_stage.py` (called without arguments) using sparsity-weighted MDS on the merged metrics. **All pair-to-pair comparisons must be complete before running this step.**

### Step 1 — Run all pairs first (if not already done)

Run the full pipeline across all six domains (within-domain) **and** all cross-domain combinations. Use `--skip-existing` to avoid re-running pairs that already have outputs:

```bash
# Within-domain + cross-domain for all 6 domains, skipping completed pairs:
eom-run --cross-domain --skip-existing
```

This produces one `outputs/merged/*_metrics.csv` file per pair. For 6 domains × 6 models each, the full run covers **630 pairs** (C(36, 2)). Expect ~2–3 hours on first run; subsequent runs with `--skip-existing` are near-instant.

To verify all pairs are present before proceeding:

```bash
# Should print 630 (or however many pairs you expect)
ls enriched_ontology_matching/outputs/merged/ | grep "_metrics.csv" | wc -l
```

### Step 2 — Regenerate the HTML map

Once all merged CSVs are in place, regenerate the map with:

```bash
eom-compare
```

Output: `enriched_ontology_matching/outputs/ontology_map.html`

Open it in any browser — no server required.

### What the map shows

- **Nodes** — one per ontology model, coloured by domain:
  - Automobile (blue), Hospital (red), University (green), Coffee (brown), Homebrewing (amber), SmartHome (purple)
- **Edges** — pairwise composite similarity (width and opacity = strength)
- **Legend** — click a domain to hide/show its nodes **and** all edges connected to it; double-click to isolate one domain
- **Mode buttons** — filter edges to MST backbone, Top-3/Top-5 neighbours, all edges, or nodes only

---

## Codebase Map

| File | Purpose |
|------|---------|
| `run_all_pairs.py` | **Main entry point** — batch runner for within-domain (and optionally cross-domain) pairs (L1→L5); supports `--inputs-dir` and `--domains` |
| `compare_stage.py` | **Summary + visualisation** — `--domain-summary DOMAIN` outputs a JSON distance summary; no args generates the interactive HTML map |
| `enriched_matcher.py` | L1 (AML + LogMap structural merge) + L2 (WN+CN discovery) |
| `neighbourhood_coherence.py` | L0 — graph neighbourhood semantic coherence |
| `lin_ic_stage.py` | L3 — Lin Information-Content scoring |
| `semantic_encoder.py` | L4 — sentence embedding cosine similarity |
| `merge_stage.py` | L5 — join all stages into metrics-only CSV |
| `aml_runner.py` | Wrapper around the AML JAR |
| `logmap_runner.py` | Wrapper around the LogMap JAR |
| `root_comparator.py` | CamelCase splitting, WUP, ConceptNet CSV lookup |
| `generate_report.py` | Renders a human-readable MD report from the combined CSV |
