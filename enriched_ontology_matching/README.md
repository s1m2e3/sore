# Enriched Ontology Matching Pipeline

A five-layer semantic enrichment pipeline that runs structural matchers (AML + LogMap) over pairs of ontology models and characterises every match using WordNet, ConceptNet, neighbourhood graph coherence, Lin Information-Content similarity, and sentence embedding cosine similarity.

## Pipeline Overview

| Layer | Name | What it does |
|-------|------|--------------|
| **L0** | Neighbourhood Coherence | Validates each match by comparing the local graph neighbourhood of each entity across ontologies using WUP similarity |
| **L1** | Structural Matching | AML + LogMap find entity pairs via string/structural similarity; results are merged and de-duplicated |
| **L2** | Semantic Discovery | WordNet (WUP, hyponymy) and ConceptNet (local CSV) discover additional equivalence and subsumption candidates among *unmatched* entities |
| **L3** | Lin-IC Scoring | Corpus-based Lin Information Content (Brown corpus) with explicit Least Common Subsumer (LCS) validates and ranks every L1 + L2 pair |
| **L4** | Sentence Embedding | `paraphrase-MiniLM-L6-v2` cosine similarity with three representations per CamelCase entity: whole name, sum of token embeddings, element-wise product of token embeddings |

---

## Prerequisites

### 1. Java Runtime (11+)

Both AML and LogMap require Java. Install from [Adoptium](https://adoptium.net/) or your system package manager:

```bash
java -version   # must be 11 or higher
```

### 2. AML — AgreementMakerLight

Download the JAR from the [AML releases page](https://github.com/AgreementMakerLight/AML-Project/releases) and place it at:

```
enriched_ontology_matching/AML/AML.jar
```

The runner (`aml_runner.py`) expects this exact path by default. You can override it with the `AML_JAR` environment variable.

### 3. LogMap

Download the standalone JAR from the [LogMap releases page](https://github.com/ernestojimenezruiz/logmap-matcher/releases) and place it at:

```
enriched_ontology_matching/logmap/logmap-matcher-4.0-standalone.jar
```

The runner (`logmap_runner.py`) expects this path. Override with the `LOGMAP_JAR` environment variable.

### 4. ConceptNet

The pipeline supports two ConceptNet modes — use whichever suits your setup:

#### Option A — ConceptNet API (zero setup, rate-limited)
No configuration needed. The pipeline queries `https://api.conceptnet.io` automatically. Works out of the box but will be throttled on large runs.

#### Option B — Local CSV (recommended for bulk runs)
Download the full ConceptNet 5 assertions file (≈ 1.5 GB compressed):

```bash
# From the ConceptNet downloads page:
# https://github.com/commonsense/conceptnet5/wiki/Downloads
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz

# Extract:
gunzip conceptnet-assertions-5.7.0.csv.gz

# Place at (create the directory if needed):
ontology_matching/inputs/conceptnet-assertions-5.7.0.csv/assertions.csv
```

The pipeline detects the local file automatically and builds an in-memory index on first run (~30 s, then cached for the session). If the file is absent it silently falls back to the API.

### 5. Python Environment

Create and activate a virtual environment, then install dependencies:

```bash
# Windows
python -m venv ontology_matching/.venv
ontology_matching\.venv\Scripts\activate

# Linux / macOS
python -m venv ontology_matching/.venv
source ontology_matching/.venv/bin/activate

# Install
pip install -r ontology_matching/requirements.txt
```

After installing, download the required NLTK data once:

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('wordnet_ic')   # needed for Lin-IC (Layer 3)
```

---

## Running the Full Pipeline

All commands are run from the **repository root**. Always use the venv Python.

### Run a single domain

```bash
ontology_matching/.venv/Scripts/python.exe \
    enriched_ontology_matching/run_all_pairs.py \
    --domains Automobile
```

This generates all C(6,2) = 15 within-domain pairs for the Automobile domain, runs every layer, and produces the combined CSV and MD report.

### Run all six domains (90 pairs total)

```bash
ontology_matching/.venv/Scripts/python.exe \
    enriched_ontology_matching/run_all_pairs.py
```

Domains: `Automobile`, `Coffee`, `Homebrewing`, `Hospital`, `SmartHome`, `University`.

### Skip already-computed pairs

```bash
ontology_matching/.venv/Scripts/python.exe \
    enriched_ontology_matching/run_all_pairs.py \
    --skip-existing
```

Each layer (enriched CSV, neighbourhood CSV, Lin-IC CSV) is skipped independently if its output file already exists.

### Limit to specific matchers

```bash
# AML only
... run_all_pairs.py --domains Automobile --matcher aml

# LogMap only
... run_all_pairs.py --domains Automobile --matcher logmap
```

---

## Running Individual Layers

### Layer 1+2 only (enriched matcher)

```bash
ontology_matching/.venv/Scripts/python.exe \
    enriched_ontology_matching/enriched_matcher.py \
    enriched_ontology_matching/pairs/auto_V1_V2.json
```

Input: a JSON file with keys `json_a` and `json_b`, each containing one model's entity/association data.

Output: `enriched_ontology_matching/outputs/enriched/<ModelA>_vs_<ModelB>.csv`

### Layer 0 — Neighbourhood Coherence

```bash
ontology_matching/.venv/Scripts/python.exe \
    enriched_ontology_matching/neighbourhood_coherence.py \
    --a  ontology_matching/inputs/.../automobile_model_v1.json \
    --b  ontology_matching/inputs/.../automobile_model_v2.json \
    --matches-csv enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --out-csv     enriched_ontology_matching/outputs/neighbourhood/auto_V1_V2_coherence.csv
```

### Layer 3 — Lin-IC Scoring

```bash
ontology_matching/.venv/Scripts/python.exe \
    enriched_ontology_matching/lin_ic_stage.py \
    --csv enriched_ontology_matching/outputs/enriched/<stem>.csv \
    --out enriched_ontology_matching/outputs/lin_ic/auto_V1_V2_lin_ic.csv
```

### Regenerate the MD report only

```bash
ontology_matching/.venv/Scripts/python.exe \
    enriched_ontology_matching/generate_report.py \
    --csv enriched_ontology_matching/outputs/enriched/all_domains_combined.csv \
    --out enriched_ontology_matching/outputs/all_domains_results.md
```

---

## Output Structure

```
enriched_ontology_matching/
├── outputs/
│   ├── enriched/
│   │   ├── <ModelA>_vs_<ModelB>.csv   # per-pair L1+L2 CSV
│   │   └── all_domains_combined.csv   # all pairs merged
│   ├── neighbourhood/
│   │   └── <domain_key>_coherence.csv # L0 scores per pair
│   ├── lin_ic/
│   │   └── <domain_key>_lin_ic.csv    # L3 Lin-IC scores per pair
│   ├── embeddings/
│   │   └── <domain_key>_emb.csv       # L4 sentence embedding cosine scores
│   └── all_domains_results.md         # final human-readable report
└── pairs/
    └── <domain_key>.json              # intermediate combined test JSONs
```

### Per-pair enriched CSV columns

| Column | Description |
|--------|-------------|
| `entity_a`, `entity_b` | Matched entity names |
| `source` | `AML`, `LogMap`, or `Both` |
| `matcher_conf` | Structural matcher confidence [0–1] |
| `layer` | `1` = structural match, `2` = discovered |
| `wup_score` | Wu-Palmer similarity of best token pair |
| `max_wup` | Max WUP across all N×M token combinations |
| `avg_wup` | Average WUP across all N×M token combinations |
| `wn_relation` | WordNet relation (`Hyponym`, `Hypernym`, `PartOf`, …) |
| `cn_relations` | ConceptNet relation type |
| `semantic_label` | `Identical`, `Synonym`, `Near-Synonym`, … |
| `layer2_type` | `Equivalence`, `Subsumption`, ConceptNet relation |

### Lin-IC CSV columns

| Column | Description |
|--------|-------------|
| `lin_ic` | Lin similarity of the best token pair |
| `max_lin_ic` | Max Lin-IC across all N×M token combinations |
| `avg_lin_ic` | Average Lin-IC |
| `lcs` | WordNet Least Common Subsumer synset name |
| `ic_lcs` | IC(LCS) — specificity of the common ancestor |
| `token_lin_details` | Semicolon-separated per-token breakdown: `ta/tb:score(lcs)` |

**Interpreting IC(LCS):**
- `>= 5` — specific common ancestor, strong semantic evidence
- `3 – 5` — moderate specificity
- `< 3` — only a generic root (e.g. `artifact.n.01`, `physical_entity.n.01`) — treat with caution

### Embedding CSV columns (`embeddings/<domain_key>_emb.csv`)

| Column | Description |
|--------|-------------|
| `entity_a`, `entity_b` | Matched entity names |
| `layer` | `1` = structural match, `2` = discovered |
| `semantic_label` | `Identical`, `Synonym`, `Near-Synonym`, … |
| `cosine_whole` | Cosine of full readable name encodings |
| `cosine_sum` | Cosine of normalised sum-of-token embeddings |
| `cosine_prod` | Cosine of normalised element-wise product of token embeddings |
| `cosine_avg` | Mean of the three cosine scores |
| `tokens_a`, `tokens_b` | CamelCase tokens used (slash-separated) |

**Interpreting the three representations:**
- `cosine_whole` captures the holistic compound meaning of the full entity name
- `cosine_sum` emphasises shared token vocabulary (additive composition)
- `cosine_prod` emphasises token interaction effects (multiplicative composition)
- Low `cosine_whole` but high `cosine_sum` often signals partial lexical overlap — inspect `tokens_a`/`tokens_b`

---

## Input Format

Each model JSON must have this structure:

```json
{
  "modelName": "My Ontology V1",
  "entities": [
    { "entityName": "Engine", "entityAttributes": [...] }
  ],
  "associations": [
    {
      "associationName": "EngineToTransmissionLink",
      "associationParticipants": ["Engine", "Transmission"]
    }
  ]
}
```

The pipeline also supports the Network-model format (`name` / `participants` keys instead of `associationName` / `associationParticipants`).

---

## Codebase Map

| File | Purpose |
|------|---------|
| `run_all_pairs.py` | Entry point — generates all pairs, runs all layers, writes combined CSV + MD |
| `enriched_matcher.py` | L1 (structural merge) + L2 (WN+CN discovery) |
| `neighbourhood_coherence.py` | L0 — graph neighbourhood semantic coherence |
| `lin_ic_stage.py` | L3 — Lin Information-Content scoring with LCS |
| `semantic_encoder.py` | L4 — Multi-representation sentence embedding cosine similarity |
| `generate_report.py` | Reads all output CSVs, renders the MD report |
| `aml_runner.py` | Wrapper around the AML JAR |
| `logmap_runner.py` | Wrapper around the LogMap JAR |
| `root_comparator.py` | CamelCase splitting, WUP, ConceptNet CSV lookup |
