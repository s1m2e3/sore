# Enriched Ontology Matching Pipeline

A five-layer semantic enrichment pipeline that runs structural matchers (AML + LogMap) over pairs of conceptual models and scores every entity pair using WordNet, ConceptNet, neighbourhood graph coherence, Lin Information-Content similarity, and sentence embedding cosine similarity.

The final output is a clean **metrics-only CSV** per pair with one row per entity pair and one column per method.

## Pipeline Overview

| Layer | Name | What it does |
|-------|------|--------------|
| **L1** | Structural Matching | AML + LogMap find entity pairs; results are merged and de-duplicated |
| **L2** | Semantic Discovery | WordNet + ConceptNet discover additional equivalence/subsumption candidates among *unmatched* entities |
| **L0** | Neighbourhood Coherence | Validates each pair by comparing local graph neighbourhoods across ontologies |
| **L3** | Lin-IC Scoring | Corpus-based Lin Information Content (Brown corpus) scores every L1+L2 pair |
| **L4** | Sentence Embedding | `paraphrase-MiniLM-L6-v2` cosine similarity per entity pair |
| **L5** | Merge | Joins all layer outputs into one metrics-only CSV |

---

## Prerequisites

### 1. Python 3.10+

```bash
python --version   # 3.10 or higher
```

Install Python dependencies from the repo root:

```bash
pip install nltk sentence-transformers
```

Then download NLTK data once:

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('wordnet_ic')
nltk.download('brown')
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
enriched_ontology_matching/inputs/conceptnet-assertions-5.7.0.csv
```

---

## Running the Pipeline

All commands are run from the **repository root**.

### Step 1 — Run a single pair (L1 + L2)

Input files are pre-built pair JSONs in `enriched_ontology_matching/pairs/`.

```bash
python enriched_ontology_matching/enriched_matcher.py \
    enriched_ontology_matching/pairs/auto_V1_V2.json
```

Output: `enriched_ontology_matching/outputs/enriched/<ModelA>_vs_<ModelB>.csv`

To use only one matcher:

```bash
python enriched_ontology_matching/enriched_matcher.py \
    enriched_ontology_matching/pairs/auto_V1_V2.json \
    --matcher logmap   # or: aml
```

### Step 2 — Neighbourhood Coherence (L0)

```bash
python enriched_ontology_matching/neighbourhood_coherence.py \
    --matches-csv enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --out-csv     enriched_ontology_matching/outputs/neighbourhood/auto_V1_V2_coherence.csv
```

### Step 3 — Lin-IC Scoring (L3)

```bash
python enriched_ontology_matching/lin_ic_stage.py \
    --csv enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --out enriched_ontology_matching/outputs/lin_ic/auto_V1_V2_lin_ic.csv
```

### Step 4 — Sentence Embeddings (L4)

```bash
python enriched_ontology_matching/semantic_encoder.py \
    --csv enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --out enriched_ontology_matching/outputs/embeddings/auto_V1_V2_emb.csv
```

### Step 5 — Merge into metrics CSV (L5)

```bash
python enriched_ontology_matching/merge_stage.py \
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
│   └── auto_V1_V2.json              # input pair JSON (json_a + json_b)
├── outputs/
│   ├── enriched/
│   │   └── <ModelA>_vs_<ModelB>.csv # L1+L2 per-pair CSV
│   ├── neighbourhood/
│   │   └── <key>_coherence.csv      # L0 coherence scores
│   ├── lin_ic/
│   │   └── <key>_lin_ic.csv         # L3 Lin-IC scores
│   ├── embeddings/
│   │   └── <key>_emb.csv            # L4 embedding cosine scores
│   └── merged/
│       └── <ModelA>_vs_<ModelB>_metrics.csv   # final metrics CSV
└── tools/
    ├── logmap/
    │   └── logmap-matcher-4.0.jar
    └── AML/
        └── AML_v3.2/
            └── AgreementMakerLight.jar
```

---

## Merged CSV Format

Each `outputs/merged/*_metrics.csv` has one row per entity pair and 7 columns:

| Column | Type | Description |
|--------|------|-------------|
| `entity_a` | string | Entity name from model A |
| `entity_b` | string | Entity name from model B |
| `matched` | 0 / 1 | 1 if AML or LogMap (or both) found this pair |
| `avg_wup` | float 0–1 | Average WordNet Wu-Palmer similarity across all token combinations |
| `lin_ic` | float 0–1 | Lin Information Content similarity (best token pair) |
| `coherence_sym` | float 0–1 | Symmetric neighbourhood coherence |
| `cosine_avg` | float 0–1 | Token-average sentence embedding cosine similarity |

Rows with `matched=0` are pairs discovered only by the semantic layers (L2).

---

## Input Pair JSON Format

Each file in `pairs/` has this structure:

```json
{
  "json_a": {
    "modelName": "Automobile Model V1",
    "entities": [{ "entityName": "Engine" }],
    "associations": [{ "associationName": "EngineDrivesWheel" }]
  },
  "json_b": {
    "modelName": "Automobile Model V2",
    "entities": [{ "entityName": "Motor" }],
    "associations": []
  }
}
```

Both `entityName`/`associationName` (Type-A) and `name` (Type-B network schema) are supported.

---

## Codebase Map

| File | Purpose |
|------|---------|
| `enriched_matcher.py` | L1 (AML + LogMap structural merge) + L2 (WN+CN discovery) |
| `neighbourhood_coherence.py` | L0 — graph neighbourhood semantic coherence |
| `lin_ic_stage.py` | L3 — Lin Information-Content scoring |
| `semantic_encoder.py` | L4 — sentence embedding cosine similarity |
| `merge_stage.py` | L5 — join all stages into metrics-only CSV |
| `aml_runner.py` | Wrapper around the AML JAR |
| `logmap_runner.py` | Wrapper around the LogMap JAR |
| `root_comparator.py` | CamelCase splitting, WUP, ConceptNet CSV lookup |
| `run_all_pairs.py` | Batch runner for all pairs in a domain |
| `generate_report.py` | Renders a human-readable MD report from the combined CSV |
