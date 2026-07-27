# Enriched Ontology Matching Pipeline

This pipeline compares pairs of conceptual, entity-relationship models by combining two independent measurement approaches. Semantic and lexical matching, using WordNet, ConceptNet, and sentence-transformer embeddings and seeded by the AML and LogMap structural matchers, establishes and characterizes entity-level correspondences. Graph-theoretic structural comparison, using a Weisfeiler-Lehman relational kernel, degree, spectral, clustering, and betweenness graph-shape metrics, and a K-hop attribute-reachability comparison, scores topology and attribute content independently of entity nomenclature. The four resulting scores are combined into a per-pair composite similarity and distance, producing a domain distance summary in JSON format and an interactive distance map.

The primary use case is **within-domain analysis**: all pairwise comparisons are computed for a set of ontologies that share a domain, and the average pairwise distance is reported as a JSON summary. Cross-domain comparisons and the interactive distance map are secondary outputs.

---

## Quick Reference

```bash
# Run the pipeline for one domain, then read its distance summary
eom-run --domains Automobile
eom-compare --domain-summary Automobile

# Run within-domain and cross-domain pairs, then build the interactive map
eom-run --cross-domain --skip-existing
eom-compare

# Regenerate the visualizations under docs/
python scripts/probe_visualizer.py
```

See "CLI: end-to-end run over one domain" and "Reproducing the `docs/` visualizations" below for the full set of flags and output paths.

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

Two further modules run only when explicitly requested via `--with-entailment` / `--with-encapsulation`, and feed neither the composite score nor the summary/map; see "Optional extra metrics" below.

| Script | Produces |
|--------|----------|
| `entailment_matcher.py` | Cross-encoder NLI entailment scores, entity-name level and attribute-type level |
| `subgraph_candidates.py` + `group_encapsulation.py` | Conceptual-encapsulation (1:n group-match) candidate discovery and scoring |

---

## The Six Metrics

Every entity or model pair is scored across six independent dimensions. Sections 1-4 form the composite similarity/distance score below; each is designed to be blind to the signal captured by the others, so that a high score on one carries no implication for the others. Sections 5 and 6 are optional, disabled by default, enabled per `eom-run` invocation with `--with-entailment` / `--with-encapsulation`, and excluded from the composite score, distance, domain summary, and interactive map (see "Optional extra metrics" for CLI usage and output-path reference).

### 1. `lexical_sim`: name-level similarity

**Purpose.** This metric measures whether the vocabulary matches, independent of graph position or attribute content.

For each candidate entity pair $(e_a, e_b)$ in the merged CSV, let $T_a, T_b$ be their camelCase-split token sets. The Wu-Palmer similarity of two tokens, based on their lowest common subsumer (LCS) in the WordNet noun hierarchy, is:

$$
\mathrm{WUP}(t_i, t_j) = \frac{2 \cdot \mathrm{depth}(\mathrm{LCS}(t_i, t_j))}{\mathrm{depth}(t_i) + \mathrm{depth}(t_j)}
$$

This is aggregated across all token pairs two ways, then blended:

$$
\mathrm{max\_wup} = \max_{t_i \in T_a,\, t_j \in T_b} \mathrm{WUP}(t_i, t_j), \qquad
\mathrm{avg\_wup} = \mathrm{mean}_{t_i \in T_a,\, t_j \in T_b} \mathrm{WUP}(t_i, t_j)
$$

$$
\mathrm{wup}(e_a, e_b) = \frac{\mathrm{max\_wup} + \mathrm{avg\_wup}}{2}
$$

Each candidate row then contributes:

$$
\mathrm{row\_lex}(e_a, e_b) = \max\Big(\mathrm{matched},\ \ \mathrm{wup} \cdot \mathbb{1}[\mathrm{wup} \ge 0.75]\Big)
$$

where $\mathrm{matched} \in \{0, 1\}$, equal to 1 if AML or LogMap independently confirmed the pair. The metric is the mean over all $N$ candidate rows:

$$
\mathrm{lexical\_sim} = \frac{1}{N}\sum_{r=1}^{N} \mathrm{row\_lex}(r)
$$

`type_embed_sim` and `cosine_avg` are deliberately excluded from `lexical_sim`, since that attribute-type signal already forms the entire basis of `attr_weighted`; including it here would double-count the signal. The 0.75 WUP threshold exists because WordNet's generic machine or artifact hierarchy assigns unrelated engineering nouns, such as fuse, filter, and blower, a WUP similarity of approximately 0.89. Below this threshold, such pairs contribute a value of zero rather than being registered as a false match.

### 2. `wl_structural`: local relational topology

**Purpose.** This metric measures whether the two models reuse the same relation patterns within the same local neighborhood, that is, whether they share the same modeling style (process-centric versus entity-centric), regardless of entity nomenclature.

Edge-aware Weisfeiler-Lehman kernel, $K=3$ hops, with all nodes anonymous. Every node $v$ starts with the same label, then is iteratively refined using its neighbourhood:

$$
\ell_v^{(0)} = \mathrm{hash}(\text{"N"}) \quad \text{for every node } v
$$

$$
\ell_v^{(k+1)} = \mathrm{md5}\Big(\ell_v^{(k)} \,\|\, \mathrm{sorted}\{(\ell_u^{(k)},\, \mathrm{edge\_type}(u,v)) : u \in N(v)\}\Big)
$$

where $\mathrm{edge\_type}(u,v)$ is the canonical relation name defined in `association_inventory.csv` (for example, PartOf, Connects, UsedFor); it is the only information that distinguishes two nodes, since entity names are not used in this computation. Label frequencies are pooled over all $K+1$ hops:

$$
\mathrm{freq}_X = \text{histogram of } \{\ell_v^{(k)} : v \in X,\ k = 0, \dots, K\}
$$

$$
\mathrm{wl\_structural} = \cos(\mathrm{freq}_A, \mathrm{freq}_B) = \frac{\mathrm{freq}_A \cdot \mathrm{freq}_B}{\lVert \mathrm{freq}_A \rVert\, \lVert \mathrm{freq}_B \rVert}
$$

### 3. `shape_sim`: global graph topology

**Purpose.** This metric distinguishes models by scope and density, for example star, chain, or clique topologies, providing a signal orthogonal to both vocabulary and local relational motifs.

Five sub-metrics, each a cosine similarity of sorted vectors, are averaged. Vectors of unequal length are zero-padded before computing the cosine similarity. There is no plain (undirected) degree sub-metric: out-degree and in-degree are tracked separately instead, since a node that is always the source of a directed association (e.g. the whole of a `PartOf`) and one that is always the target (the part) should not be treated as structurally interchangeable just because their total degree matches.

$$
\mathrm{out\_degree\_sim} = \cos\big(\mathrm{sort}_{\text{desc}}(\deg^{\text{out}}_A),\ \mathrm{sort}_{\text{desc}}(\deg^{\text{out}}_B)\big), \qquad
\mathrm{in\_degree\_sim} = \cos\big(\mathrm{sort}_{\text{desc}}(\deg^{\text{in}}_A),\ \mathrm{sort}_{\text{desc}}(\deg^{\text{in}}_B)\big)
$$

$$
\mathrm{spectral\_sim} = \cos\big(\mathrm{sort}(\lambda_A),\ \mathrm{sort}(\lambda_B)\big), \qquad \lambda = \mathrm{eig}\big(L\big),\ \ L = I - D^{-1/2} A D^{-1/2}
$$

$$
\mathrm{clustering\_sim} = \cos\big(\mathrm{sort}_{\text{desc}}(C_A),\ \mathrm{sort}_{\text{desc}}(C_B)\big), \qquad C(v) = \frac{\mathrm{triangles}(v)}{d(v)\big(d(v)-1\big)/2}
$$

$$
\mathrm{betweenness\_sim} = \cos\big(\mathrm{sort}_{\text{desc}}(\mathrm{BC}_A),\ \mathrm{sort}_{\text{desc}}(\mathrm{BC}_B)\big)
$$

$$
\mathrm{shape\_sim} = \frac{\mathrm{out\_degree\_sim} + \mathrm{in\_degree\_sim} + \mathrm{spectral\_sim} + \mathrm{clustering\_sim} + \mathrm{betweenness\_sim}}{5}
$$

Betweenness centrality $\mathrm{BC}$ is computed via Brandes' $O(VE)$ algorithm and normalised to $[0,1]$.

### 4. `attr_weighted`: attribute reachability

**Purpose.** This metric measures whether the two models describe the same observable quantities, for example temperature, torque, and pressure, independent of both entity names and graph shape.

Let $V$ be the union of observable types declared across both models (`data["observables"]` union `entityAttribute.type`). For each entity $e$, $\mathrm{reach}(e)$ is the $K$-hop ($K=2$) weighted propagation of its own and its neighbours' declared attribute types through structurally weighted edges:

$$
\text{PartOf/HasA} = 0.6,\quad \text{MadeOf} = 0.5,\quad \text{Connects/UsedFor} = 0.4,\quad \text{AtLocation} = 0.3,
$$

$$
\text{CapableOf/Causes/IsA/ReceivesAction} = 0.25,\quad \text{HasPrerequisite} = 0.2,\quad \text{RelatedTo skipped}
$$

The weight of a path is the product of its edge weights, and the maximum is kept across multiple paths reaching the same type. Name-based imputation is disabled at this stage, keeping the computation purely structural. Each model is then aggregated into a single vector, using sentence-transformer embeddings ($\mathrm{embed}$, `paraphrase-MiniLM-L6-v2`):

$$
\mathrm{agg}_X = \sum_{e \in X} \sum_{t \in \mathrm{reach}(e)} w(e,t) \cdot \mathrm{embed}(t)
$$

$$
\mathrm{embed\_sim} = \cos(\mathrm{agg}_A, \mathrm{agg}_B)
$$

A second, independent signal is computed as a soft cosine over the same attribute-type vocabulary, using a WUP similarity kernel $S \in \mathbb{R}^{|V|\times|V|}$ in place of the identity matrix, where $w_X$ is the raw (non-embedded) summed attribute-weight vector for model $X$:

$$
\mathrm{wup\_sim} = \frac{w_A^\top S\, w_B}{\sqrt{w_A^\top S\, w_A \cdot w_B^\top S\, w_B}}
$$

The metric takes the stricter of the two, since WUP alone tends to be too forgiving for engineering and physics nouns:

$$
s_{\text{attr}} = \min(\mathrm{embed\_sim},\, \mathrm{wup\_sim})
$$

$$
\mathrm{attr\_weighted} = s_{\text{attr}}
$$

where $s_{\text{attr}}$ denotes `attr_dist_sim` in the pipeline's output CSVs. `attr_weighted` shares no data with `lexical_sim`. The kernel `wup_sim` above is computed over the closed observable-type vocabulary `V` (for example, Temperature, Torque, and Pressure), and is entirely independent of the entity-name WUP score used in `lexical_sim` (Section 1). Both computations call the same underlying WordNet Wu-Palmer function, but over disjoint inputs, attribute-type tokens in one case and entity-name tokens in the other, with no value passed between them. In an earlier version of the pipeline these metrics were coupled: `attr_weighted` was computed as `attr_dist_sim` multiplied by `avg_entity_wup`. This scaling was removed, see the docstring of `run_attr_dist_stage` in `attribute_reach.py`, because it made a metric intended to capture attribute similarity alone depend on entity naming.

### 5. Entailment (optional, `--with-entailment`)

**Purpose.** This metric tests directional textual entailment, whether the description of one entity (or its declared attributes) logically implies the other's. It requires a cross-encoder, which jointly encodes premise and hypothesis in one transformer pass; the bi-encoders used elsewhere in this pipeline (`cosine_avg`, `type_embed_sim`) only ever produce independent embeddings and cannot represent this directional, joint-context relationship.

A cross-encoder NLI model $M_{\mathrm{nli}}$ (default `cross-encoder/nli-MiniLM2-L6-H768`, overridable with `--nli-model`) maps a (premise, hypothesis) pair jointly to class logits; the entailment-class probability is:

$$
p_{\mathrm{ent}}(x \to y) = \big[\mathrm{softmax}(M_{\mathrm{nli}}(x, y))\big]_{\mathrm{entail}}
$$

where $x$ is the premise text, $y$ the hypothesis text, and $\mathrm{entail}$ is the model's own entailment-class index, read from its `id2label` config since NLI models do not agree on class ordering. (A minority of binary-relevance-style cross-encoders emit a single logit instead of three; for those, $p_{\mathrm{ent}}(x\to y) = \sigma(z)$, the sigmoid of that logit.)

Two independent probes reuse this same $p_{\mathrm{ent}}$ function over different text-construction functions:

**Entity-level** (`run_entity_entailment_stage`). Text is the camelCase-split, lower-cased entity name: $\pi(e) = \text{"This is a "} + \mathrm{readable}(e) + \text{"."}$. For every candidate pair $(e_a, e_b)$:

$$
\mathrm{ent}_{a \to b} = p_{\mathrm{ent}}(\pi(e_a) \to \pi(e_b)), \qquad \mathrm{ent}_{b \to a} = p_{\mathrm{ent}}(\pi(e_b) \to \pi(e_a))
$$

Validated to correctly recover directional hyponym/hypernym relationships, for example `MasterCylinder → Cylinder` scores 0.99 one-way and 0.006 in reverse, and to avoid the naive lexical-overlap trap where `SteeringWheel` would otherwise be confused with `Wheel`.

**Attribute-level** (`run_attribute_entailment_stage`). Text is built from the same $K$-hop ($K=2$) attribute-reach signature $\mathrm{reach}(e)$ used by `attr_weighted` (Section 4), with name-based imputation disabled: $\sigma(e) = \text{"The component has these properties: "} + \mathrm{sorted}\{\mathrm{readable}(t) : t \in \mathrm{reach}(e)\} + \text{"."}$. The same two directional scores are computed with $\sigma$ substituted for $\pi$.

Both probes report the same triple, since, unlike Sections 1-4, entailment is not symmetric:

$$
\mathrm{entailment\_f1}(e_a, e_b) = \max\big(\mathrm{ent}_{a \to b},\ \mathrm{ent}_{b \to a}\big)
$$

written as `entailment_a_covers_b`, `entailment_b_covers_a`, `entailment_f1` to `outputs/entailment_entity/<stem>_entity_entailment.csv` (entity-level) and `outputs/entailment_attr/<stem>_attr_entailment.csv` (attribute-level).

### 6. Conceptual encapsulation (optional, `--with-encapsulation`)

**Purpose.** Tests a relationship none of Sections 1-5 can, since every one of them compares exactly one entity in A against one entity in B: whether one entity in A corresponds instead to a *group* of entities in B, the same concept described at a different resolution (1:n / complex matching in the ontology-alignment literature).

**Step 1: candidate group discovery** (`subgraph_candidates.py`). Build an undirected graph $G_X = (V_X, E_X)$ per model $X$, one node per entity, one edge for every pair of co-participants in any association, regardless of relation type, so that a ring of `Connects` edges is found exactly as readily as a `PartOf` composition tree. Partition via Louvain modularity-maximizing community detection into candidate groups:

$$
\mathcal{C}_X = \{\, c \in \mathrm{Louvain}(G_X) : 2 \le |c| \le 0.6\,|V_X| \,\}
$$

the lower bound drops singleton groups, the upper bound drops the degenerate whole-graph-as-one-group case.

**Step 2: scoring** (`group_encapsulation.py`). For a coarse entity $e_a \in A$ and a candidate group $g \in \mathcal{C}_B$, two independent scores are computed, reusing Sections 4 and 5's machinery rather than new machinery.

*Attribute-level*, reusing `attr_weighted`'s embed+WUP blend (Section 4) with the group's reach vectors summed before comparison:

$$
\mathrm{agg}(g) = \sum_{e \in g}\sum_{t \in \mathrm{reach}(e)} w(e,t)\cdot \mathrm{embed}(t), \qquad w_g = \sum_{e \in g} w_e
$$

$$
\mathrm{embed\_sim}(e_a, g) = \cos\big(\mathrm{agg}(\{e_a\}),\, \mathrm{agg}(g)\big), \qquad
\mathrm{wup\_sim}(e_a, g) = \frac{w_{e_a}^\top S\, w_g}{\sqrt{w_{e_a}^\top S\, w_{e_a} \cdot w_g^\top S\, w_g}}
$$

$$
\mathrm{attr\_score}(e_a, g) = \min\big(\mathrm{embed\_sim}(e_a, g),\ \mathrm{wup\_sim}(e_a, g)\big)
$$

where $S$ is the same attribute-type WUP kernel used in Section 4.

*Name-level*, reusing Section 5's $p_{\mathrm{ent}}$ over a group premise $\gamma(g) = \text{"This is a system made of "} + \mathrm{sorted}\{\mathrm{readable}(m) : m \in g\} + \text{"."}$:

$$
\mathrm{name\_f1}(e_a, g) = \max\Big(p_{\mathrm{ent}}(\gamma(g) \to \pi(e_a)),\ p_{\mathrm{ent}}(\pi(e_a) \to \gamma(g))\Big)
$$

written as `name_group_covers_entity`, `name_entity_covers_group`, `name_f1`.

**Abstention gate** (`classify_top_match`). For each $e_a$, every candidate group $g \in \mathcal{C}_B$ is ranked by $\mathrm{name\_f1}(e_a, g)$; let $s_1 \ge s_2 \ge \dots$ be the sorted scores and $\mathrm{margin} = s_1 - s_2$. Only the top-ranked group is ever reported, and only when:

$$
\mathrm{decision}(e_a) = \begin{cases}
\text{match} & s_1 \ge \tau_{\mathrm{hc}} \\
\text{match} & \tau_{\mathrm{min}} \le s_1 < \tau_{\mathrm{hc}} \ \text{and}\ \mathrm{margin} \ge \tau_{\mathrm{margin}} \\
\text{abstain} & \text{otherwise}
\end{cases}
$$

`abstain` is written to the output CSV as the `no_match` decision label. Defaults: $\tau_{\mathrm{min}} = 0.5$, $\tau_{\mathrm{margin}} = 0.15$, $\tau_{\mathrm{hc}} = 0.9$ (tunable via `min_score`/`min_margin`/`high_confidence`). A top score and margin alone cannot separate correct from incorrect top-1 picks (see the docstring in `group_encapsulation.py` for the specific near-tie cases that motivated the two-tier rule); a 33-case Automobile-domain evaluation found this gate raises precision on reported matches from ~77% to ~89.5%, at the cost of recall dropping from ~88% to ~68%, an expected and, for this purpose, worthwhile trade.

Tested in both directions (`a_in_b`: entities of A against B's candidate groups, and `b_in_a`) and written to `outputs/encapsulation/<stem>_encapsulation.csv`: `direction`, `decision`, `best_group` (winning candidate's members), `attr_score`, `name_group_covers_entity`/`name_entity_covers_group`/`name_f1`, and `margin`.

### Composite score and distance

Let $\mathcal{M} = \{\mathrm{lexical\_sim},\ \mathrm{wl\_structural},\ \mathrm{shape\_sim},\ \mathrm{attr\_weighted}\}$ (Sections 1-4 only; Sections 5 and 6 are deliberately excluded, see above), each with nominal weight $1/4$, and let $\mathcal{A} = \{m \in \mathcal{M} : \mathrm{value}(m) > 0\}$ be the subset available for a given pair. Metrics unavailable for a pair, for example $\mathrm{attr\_weighted}=0$ when neither model has observable types, are excluded from $\mathcal{A}$ rather than counted as a penalty.

$$
\mathrm{composite}(A,B) = \frac{1}{|\mathcal{A}|}\sum_{m \in \mathcal{A}} \mathrm{value}(m)
$$

$$
\mathrm{distance}(A,B) = \sqrt{\frac{1}{|\mathcal{A}|}\sum_{m \in \mathcal{A}} \big(1 - \mathrm{value}(m)\big)^2} \ \in [0,1]
$$

The distance matrix is symmetrised as $d = (d + d^\top)/2$, and used for both the domain summary and the MDS map.

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

### 6. Neural network model cache (no setup needed, informational)

Every sentence-transformer or cross-encoder model this pipeline uses (`paraphrase-MiniLM-L6-v2` for `cosine_avg`/`attr_weighted`'s embed_sim; `cross-encoder/nli-MiniLM2-L6-H768` for the optional Section 5/6 metrics) goes through the same local-cache-or-download check (`model_cache.py`): if it's already present under `enriched_ontology_matching/models/`, it's loaded from there; otherwise it's downloaded once and saved to that directory for every later run. That directory is gitignored, since these are large binary weights, not something to check into the repo, so the first run of any command that touches one of these models will download it.

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

Running the two commands above produces the final result, stored at:

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

### What `eom-run` needs and generates, per pair

`run_all_pairs.py` runs 6 stages per model pair and writes one file per stage. `<stem>` denotes `<ModelA_modelName>_vs_<ModelB_modelName>`; `<domain_key>` denotes a short identifier such as `auto_V1_V2`.

| Stage | Module | Needs | Writes | Contents |
|-------|--------|-------|--------|----------|
| 0 | `run_all_pairs.py` | the two model JSONs | `enriched_ontology_matching/pairs/<domain_key>.json` | combined `{json_a, json_b}` input for the rest of the pipeline (gitignored; regenerated on every run) |
| 1 | `enriched_matcher.py` | stage-0 pair JSON, `association_inventory.csv`, WordNet, ConceptNet (local CSV or REST) | `outputs/enriched/<stem>.csv` | AML and LogMap candidate matches, annotated with `wup`, `max_wup`, `avg_wup`, and WordNet/ConceptNet relation labels; plus `outputs/enriched/all_domains_combined.csv` (all pairs concatenated across the whole `eom-run` invocation) |
| 2 | `semantic_encoder.py` | stage-1 CSV, both model JSONs | `outputs/embeddings/<domain_key>_emb.csv` | `cosine_avg`: name and attribute-type sentence-embedding cosine similarity (rescaled to [0, 1]) for each candidate row |
| 3 | `attribute_reach.py` (`run_type_embed_stage`) | stage-1 CSV, both model JSONs | `outputs/type_embed/<domain_key>_type_emb.csv` | `type_embed_sim`: attribute-type embedding cosine similarity for each candidate row |
| 4 | `merge_stage.py` | stages 1 to 3 CSVs | `outputs/merged/<stem>_metrics.csv` | one row per candidate entity pair: `entity_a, entity_b, matched, wup, cosine_avg, type_embed_sim` |
| 5 | `wl_kernel_matcher.py` (`run_wl_stage`) | both model JSONs, stage-4 merged CSV | `outputs/wl/<stem>_metrics_wl.csv` | one row: `wl_structural, shape_sim` (plus sub-metrics `out_degree_sim/in_degree_sim/spectral_sim/clustering_sim/betweenness_sim`), and diagnostic-only `wl_matched, wl_composite, wl_consistency, match_coverage, induced_frac_a/b, n_entity_matches, n_shared_labels, n_nodes_a/b, n_edges_a/b` |
| 6 | `attribute_reach.py` (`run_attr_dist_stage`) | both model JSONs | `outputs/attr_dist/<stem>_metrics_attr_dist.csv` | one row: `attr_dist_sim` |

These are the intermediate stages that `eom-compare --domain-summary <Domain>` reads to assemble the final result above: it collects every `outputs/merged/*_metrics.csv` file for the requested domain, pulling `shape_sim` and `wl_structural` from the matching `outputs/wl/` CSV and `attr_dist_sim` from the matching `outputs/attr_dist/` CSV.

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

## Optional extra metrics (`--with-entailment`, `--with-encapsulation`)

Sections 5 (Entailment) and 6 (Conceptual encapsulation) of "The Six Metrics" above are experimental extras: disabled by default, and excluded from `composite`, `distance`, the domain summary JSON, and the interactive map. Enable them per `eom-run` invocation:

```bash
eom-run --domains Automobile --with-entailment --with-encapsulation
```

Both use a cross-encoder NLI model (default `cross-encoder/nli-MiniLM2-L6-H768`, overridable with `--nli-model`, downloaded and cached under `enriched_ontology_matching/models/` on first use).

| Flag | Modules | Output |
|------|---------|--------|
| `--with-entailment` | `entailment_matcher.py` | `outputs/entailment_entity/<stem>_entity_entailment.csv` (entity-name-level), `outputs/entailment_attr/<stem>_attr_entailment.csv` (attribute-type-level); both carry `entailment_a_covers_b`, `entailment_b_covers_a`, `entailment_f1` |
| `--with-encapsulation` | `subgraph_candidates.py`, `group_encapsulation.py` | `outputs/encapsulation/<stem>_encapsulation.csv`: one row per (`direction`, entity) tested; `decision` is `match` or `no_match` (`no_match` rows have blank score/group fields); `match` rows carry `best_group`, `attr_score`, `name_group_covers_entity`/`name_entity_covers_group`/`name_f1`, and `margin` |

See Sections 5 and 6 above for the underlying computation, validation notes, and abstention-gate thresholds.

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
| `entailment_matcher.py` | Optional (`--with-entailment`): cross-encoder NLI entailment, entity-name level and attribute-type level |
| `subgraph_candidates.py` | Optional (`--with-encapsulation`, step 1): topology-agnostic candidate subgroup discovery via graph community detection |
| `group_encapsulation.py` | Optional (`--with-encapsulation`, step 2): scores candidate subgroups against coarse entities; `classify_top_match` abstention gate |
| `scripts/probe_visualizer.py` | Regenerates `docs/probe_visualizations.pdf` and `docs/probe_s*.png` from the probe fixtures |
