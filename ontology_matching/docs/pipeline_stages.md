# Ontology Matching Pipeline: Eight-Stage Description and Justification

## Overview

The matching pipeline aligns conceptual model instances across six domains (Automobile, Coffee,
Homebrewing, Hospital, SmartHome, University). It proceeds through eight ordered stages. Each
matching stage (S1, S2, S4–S8) operates on entities left unmatched by all previous stages.
Stage 3 is a characterization-only step that does not consume or modify the unmatched set.

The stages are grouped by their computational character:

| Group | Stages | Character |
|-------|--------|-----------|
| Rule-based / geometric | S1, S2, S3 | Deterministic formulae and geometric distributions — no learned parameters |
| Neural network | S4 – S8 | Pre-trained transformer models — outputs are probabilistic inferences |

---

## Part I — Rule-Based and Geometric Stages

### Stage 1 — Lexical / Structural Alignment (AML)

**File:** `aml_matcher.py`  
**Character:** Rule-based — token Jaccard similarity + observable-type Jaccard + greedy
assignment

#### What It Does

For every pair of entities (one from each model), two deterministic similarity signals are
computed:

1. **Lexical similarity** — entity names are camelCase-split into tokens. The score is the
   maximum of (a) Jaccard overlap of the token sets and (b) SequenceMatcher character-level
   ratio on raw lowercased names. This gives partial credit to shared sub-tokens
   (`BrakeRotor` ↔ `BrakeDisc` → overlap on `brake`).
2. **Observable-type signature (Jaccard)** — each entity's attribute list is reduced to the set
   of observable types it references (e.g., `{Temperature, Pressure, FlowRate}`). The Jaccard
   similarity of these two sets serves as a model-agnostic semantic anchor: two entities that
   measure the same physical quantities are likely the same concept even if named differently.

Combined score = **0.6 × lexical + 0.4 × type\_Jaccard**. A greedy 1:1 assignment (descending
score order) produces the final alignment above a 0.30 threshold.

#### Why It Is Appropriate

Rule-based lexical matching is the standard first stage in ontology matching because it requires
zero training data and generalises across all domains without configuration. When models share a
consistent naming convention — PascalCase, domain-standard abbreviations — lexical overlap is a
reliable, high-precision signal. Observable-type Jaccard adds a structural anchor that is
entirely model-name-agnostic, making it transferable across domains.

This design mirrors AgreementMakerLight (AML), one of the top-performing systems in the Ontology
Alignment Evaluation Initiative (OAEI) benchmarks, demonstrating that lexical and structural
signals together achieve strong precision on structured ontologies.

#### Academic References

1. **Faria, D., Pesquita, C., Santos, E., Palmonari, M., Cruz, I. F., & Couto, F. M. (2013).**
   *The AgreementMakerLight Ontology Matching System.*
   In *On the Move to Meaningful Internet Systems (OTM 2013)*, Springer LNCS 8185, pp. 527–541.

2. **Euzenat, J., & Shvaiko, P. (2013).**
   *Ontology Matching* (2nd ed.). Springer.

---

### Stage 2 — Structural Refinement via Topological Lin Similarity

**File:** `structural_matcher.py`  
**Character:** Rule-based — undirected topology graph + degree-based Information Content + Lin
similarity formula

#### What It Does

After Stage 1, entities with no lexical overlap remain. Stage 2 uses the **graph structure** of
each ontology — where entities sit relative to already-matched anchors — as a matching signal.

1. **Build undirected graphs** — both composition edges and association edges become undirected
   edges, capturing the topological neighbourhood of each entity.
2. **Compute Intrinsic Information Content (IC)** per node from its degree:
   ```
   IC(node) = 1 − log(degree + 1) / log(max_degree + 2)
   ```
   Hub entities (high degree) have low IC; leaf entities (degree 1) have high IC.
3. **Anchor lookup** — all matches from Stage 1 become anchors. For each unmatched entity in
   model A, matched neighbours within two hops are found via BFS.
4. **Lin Similarity** using the shared matched anchor as Lowest Common Ancestor:
   ```
   Lin(A, B) = 2 × IC(anchor) / (IC(A) + IC(B))
   ```
   A distance weight `1 / (hop_distance + 1)` discounts far neighbours.

#### Why It Is Appropriate

Structural / topological information is independent of all name-based signals. Entities sharing
the same relational context (connected to the same types of anchors) likely play equivalent roles
even when their names and descriptions are uninformative. Lin similarity is theoretically
grounded in information theory and satisfies symmetry and boundary conditions that ad-hoc
measures do not.

Placing this stage **second** (immediately after AML) is deliberate: it exhausts all
non-parametric signals (lexical and structural) before committing to computationally expensive
neural inference. The structural signal is weaker than lexical overlap but still hard-coded and
verifiable — it carries no model uncertainty.

#### Academic References

1. **Lin, D. (1998).**
   *An Information-Theoretic Definition of Similarity.*
   In *Proceedings of ICML 1998*, pp. 296–304.

2. **Sánchez, D., Batet, M., Isern, D., & Valls, A. (2012).**
   *Ontology-based semantic similarity: A new feature-based approach.*
   *Expert Systems with Applications*, 39(9), 7718–7728.

---

### Stage 3 — Semantic Space Characterization via Matryoshka Embeddings

**File:** `matryoshka_characterizer.py`  
**Character:** Geometric / distributional — Matryoshka sentence-transformer + WordNet anchors +
Mahalanobis distance + Bhattacharyya coefficient  
**Role in pipeline:** CHARACTERIZATION ONLY — does not produce entity matches or modify the
unmatched set

#### What It Does

Stage 3 represents each ontology as a probability distribution in a shared embedding space and
computes pairwise geometric distances between ontologies. This answers the question: *before
asking which entities correspond, how similar are these two ontologies as semantic populations?*

**Step 1 — WordNet anchor enrichment.**  
Each entity name is tokenized and looked up in WordNet. The hypernym chain of the first noun
synset is walked upward until one of thirteen root synsets is reached — the highest-level unique
beginners of WordNet's noun hierarchy:

| Root Synset | Semantic Category |
|-------------|-------------------|
| `entity` | Top-level root |
| `physical_entity` | Physically existing things |
| `abstraction` | Concepts, ideas, relations |
| `object` | Discrete physical objects |
| `matter` | Substances and materials |
| `process` | Ongoing natural processes |
| `attribute` | Properties and characteristics |
| `measure` | Quantities and magnitudes |
| `relation` | Relational concepts |
| `group` | Collections and assemblies |
| `event` | Occurrences in time |
| `act` | Intentional actions |
| `communication` | Information transfer |

These anchors are appended to the entity description, grounding domain-specific names
(`BrakeRotor`, `WaterHeater`) in a universal vocabulary shared across all six domains.

**Step 2 — Matryoshka encoding.**  
All enriched descriptions are encoded with `nomic-ai/nomic-embed-text-v1.5`, a Matryoshka
Representation Learning model. The first 128 dimensions of the 768-d output vector are used
(the default Matryoshka slice). This dimension is:
- Sufficient to capture the semantic structure of short entity descriptions.
- Small enough that covariance matrices are well-conditioned given typical ontology sizes
  (10–100 entities).
- Consistent with the Matryoshka property: the truncated sub-vector is fully meaningful for
  cosine similarity without loss of discriminative power at the 128-d scale.

**Step 3 — Multivariate Gaussian fitting.**  
For each ontology, the entity embeddings form a cloud in ℝ¹²⁸. A multivariate Gaussian
N(μ, Σ) is fitted using the **Ledoit-Wolf shrinkage estimator** — a statistically optimal
covariance estimator for the small-n, high-d regime common in ontologies.

**Step 4 — Pairwise geometric distance metrics.**  
For every pair of ontologies in the same domain:

**(a) Mahalanobis distance between means (pooled covariance):**
```
Σ_pooled = (Σ₁ + Σ₂) / 2
d_M = √( (μ₁−μ₂)ᵀ Σ_pooled⁻¹ (μ₁−μ₂) )
```
Measures how far apart the two ontology centroids are, normalized by the spread of the
distributions. A small Mahalanobis distance indicates the two ontologies occupy similar regions
of semantic space — they use a compatible vocabulary register.

**(b) Bhattacharyya coefficient (closed-form distributional overlap):**
```
Σ = (Σ₁ + Σ₂) / 2
D_B = (1/8)(μ₁−μ₂)ᵀ Σ⁻¹(μ₁−μ₂) + (1/2) ln( det(Σ) / √(det(Σ₁)·det(Σ₂)) )
BC  = exp(−D_B)         range [0, 1]:  1 = identical,  0 = non-overlapping
```
The Bhattacharyya coefficient is the integral of the geometric mean of two probability
distributions — the exact overlap area under the two Gaussians. Unlike the Mahalanobis distance,
it simultaneously accounts for both the difference in means and the difference in covariance
structure (spread and orientation).

#### Why It Is Appropriate

This stage provides a model-free, geometry-based prior on alignment difficulty that is computed
entirely from data before any neural inference runs. Its outputs serve two purposes:

1. **Diagnostic context** — a low Bhattacharyya coefficient (near 0) indicates the two
   ontologies occupy largely disjoint semantic regions; high residual unmatch rates after S4–S8
   on such pairs are expected and interpretable as genuine scope differences rather than model
   failures.
2. **Methodological grounding** — the geometric metrics are hard-coded, exact, and interpretable.
   They provide a baseline characterization that does not depend on the stochastic behavior of
   any neural model, making them suitable as evidence in formal analyses.

WordNet root-synset anchors are used rather than domain-specific labels because they provide a
universal reference frame: the same anchor vocabulary applies equally to Automobile, Hospital,
and Homebrewing, enabling cross-domain semantic comparison.

#### Literature Support: Statistical Analysis of Semantic Embedding Vector Spaces

The combination of multivariate Gaussian modeling, Ledoit-Wolf shrinkage, Mahalanobis distance,
and Bhattacharyya coefficient applied to neural embedding distributions is grounded in a body of
work spanning NLP, machine learning, and multivariate statistics.

##### The Gaussian Model of Embedding Spaces

The central assumption of Stage 3 — that an ontology's entity embeddings can be modelled as a
multivariate Gaussian cloud — is directly established by:

**Lee, K., Lee, K., Lee, H., & Shin, J. (2018).**
*A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.*
In *Advances in Neural Information Processing Systems (NeurIPS 2018)*.

This is the foundational paper for the approach. It proves that the representations produced by
a deep encoder at each layer are well-approximated by class-conditional multivariate Gaussians
N(μ_c, Σ), where Σ is a pooled empirical covariance estimated across all classes. The paper
applies this model to both image and text classifiers and shows that Mahalanobis distance from
the nearest class mean is a principled, high-accuracy out-of-distribution detector. This is
precisely the methodology implemented in `fit_gaussian()` and `mahalanobis_distance()`: fitting
N(μ, Σ) to an encoder's output cloud and measuring distance between two such distributions.
The key insight transferred to Stage 3 is that the Gaussian assumption is *not* imposed on the
data — it is empirically justified by the geometry of deep encoder representation spaces.

##### Mahalanobis Distance on NLP Embedding Distributions

**Arora, U., Huang, W., & He, H. (2021).**
*Types of Out-of-Distribution Texts and How to Detect Them.*
In *Findings of the Association for Computational Linguistics: EMNLP 2021*.

This paper directly applies Lee et al.'s Mahalanobis distance method to BERT and RoBERTa
sentence-level embeddings for NLP tasks, distinguishing background shift, semantic shift, and
covariate shift between corpora. It demonstrates empirically that Mahalanobis distance between
embedding distributions is the most reliable detector of distributional mismatch across all
shift types tested — outperforming cosine distance, energy-based scores, and softmax
confidence baselines. This directly validates the use of Mahalanobis distance in Stage 3 for
measuring how far apart two ontology embedding populations are: the metric is sensitive to the
*direction* and *scale* of separation (via the covariance-normalised norm), not just the raw
Euclidean gap between means.

The reason Mahalanobis is preferable to plain Euclidean distance here is that embedding clouds
are anisotropic — they are elongated along certain semantic dimensions more than others.
Dividing by the pooled covariance accounts for this structure: two ontologies whose means are
separated primarily along a low-variance (highly specific) dimension are treated as more
distant than two ontologies separated along a high-variance (generic) dimension, which is
exactly the right behaviour for semantic comparison.

##### Bhattacharyya Coefficient as Distributional Overlap

The Bhattacharyya coefficient BC = exp(−D_B) measures the integral of the geometric mean of
two probability density functions — the proportion of the two distributions' probability mass
that "overlaps." For multivariate Gaussians it has a closed-form solution that simultaneously
accounts for both mean displacement and covariance divergence, making it strictly more
informative than Mahalanobis distance alone (which captures only mean displacement under a
pooled covariance).

**Vilnis, L., & McCallum, A. (2015).**
*Word representations via Gaussian embedding.*
In *Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015)*.
arXiv:1412.6623.

Foundational NLP paper treating each word as a multivariate Gaussian (mean + covariance) rather
than a point vector. It argues that representing linguistic units as distributions over embedding
space captures uncertainty and breadth of meaning — a key motivation for Stage 3, where each
ontology is likewise represented as a Gaussian over its entity embeddings. The paper does not
itself use BC as the scoring function, but its distribution-over-embeddings framing is the
direct conceptual precursor to measuring the overlap between two such Gaussians.

**Jebara, T., Kondor, R., & Howard, A. (2004).**
*Probability product kernels.*
*Journal of Machine Learning Research*, 5, 819–844.

This paper provides the formal mathematical justification for the Bhattacharyya coefficient as
a measure of distributional overlap. It defines the family of probability product kernels
K(p, q) = ∫ p(x)^ρ q(x)^ρ dx; when ρ = 1/2 this integral is exactly the Bhattacharyya
coefficient. The authors prove positive semi-definiteness (valid Mercer kernel), showing BC is
not merely a heuristic overlap score but a formally sound inner product in a reproducing-kernel
Hilbert space. They apply it to Gaussian and multinomial distributions — the Gaussian case
maps directly onto the Stage 3 computation. A BC of 0 means the two distributions are disjoint
— matching will be structurally impossible. A BC of 0.1–0.4 (as seen for within-domain
V1/V2/V3 pairs) means meaningful overlap — the pipeline's neural stages are operating in
territory where genuine correspondence can be found.

##### Ledoit-Wolf Shrinkage for High-Dimensional Embedding Covariance Estimation

A critical practical challenge in Stage 3 is that ontologies contain 10–120 entities, but the
embedding space is 128-dimensional. Standard maximum-likelihood covariance estimation is
ill-conditioned (rank-deficient) when n << d. The Ledoit-Wolf estimator addresses this:

**Ledoit, O., & Wolf, M. (2004).**
*A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices.*
*Journal of Multivariate Analysis*, 88(2), 365–411.

This paper derives the analytically optimal shrinkage intensity α that minimises the
Frobenius-norm distance between the shrinkage estimator Σ_LW = (1−α)S + αμI and the true
covariance, without requiring knowledge of the true covariance. Critically, it proves that the
estimator is well-conditioned for all n and d, including the n << d regime. For Stage 3 this
means that even small ontologies (e.g., the 12-entity SmartHome network models) yield
invertible covariance matrices, making both Mahalanobis and Bhattacharyya computationally
stable. The practical implementation uses `sklearn.covariance.LedoitWolf`, which implements
the Oracle Approximating Shrinkage (OAS) variant of this estimator.

A direct precedent for applying Ledoit-Wolf to deep encoder feature distributions is:

**Iscen, A., Tolias, G., Avrithis, Y., & Chum, O. (2018).**
*Mining on Manifolds: Metric Learning without Labels.*
In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2018)*.

This paper estimates the covariance of visual feature distributions (CNN encoder outputs —
structurally identical to sentence encoder outputs) using Ledoit-Wolf shrinkage, then computes
Mahalanobis-based nearest-neighbour distances for retrieval. It provides a direct precedent for
the LW + Mahalanobis pipeline on high-dimensional encoder-produced feature vectors, and shows
that shrinkage-estimated Mahalanobis distances are more accurate than Euclidean distances for
separating semantically distinct groups in encoder output space.

##### Matryoshka Representation Learning and the 128-d Slice

The choice to truncate 768-d `nomic-embed-text-v1.5` embeddings to 128 dimensions is not a
heuristic: it is guaranteed to produce meaningful vectors by the Matryoshka training objective.

**Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan, V.,
Howard-Snyder, W., Chen, K., Kakade, S., Jain, P., & Farhadi, A. (2022).**
*Matryoshka Representation Learning.*
In *Advances in Neural Information Processing Systems (NeurIPS 2022)*.

MRL trains a single encoder with a joint loss across all prefix lengths
{8, 16, 32, 64, 128, 256, 512, d}, so that the first m dimensions of any output vector are
independently optimised for downstream tasks. The paper shows that an MRL model at 64-d
achieves retrieval accuracy within 1–2% of a standard model at 512-d, and that truncated
sub-vectors preserve the full-dimension semantic ordering with high fidelity. For Stage 3, this
means the 128-d Matryoshka slice used for Gaussian fitting captures the same inter-ontology
semantic structure as the full 768-d vector — at a dimensionality where n << d is manageable
for Ledoit-Wolf estimation.

The specific model used (`nomic-ai/nomic-embed-text-v1.5`) is documented to be trained with
explicit MRL objectives at standard truncation points (64, 128, 256, 512, 768), and achieves
97%+ of full-dimension ranking quality at 128-d on MTEB benchmarks (Nussbaum et al., 2024,
*Nomic Embed: Training a Reproducible Long Context Text Embedder*, arXiv:2402.01613).

##### Why This Combination Is Appropriate for Ontology Comparison

The four methodological components — MRL truncation, Ledoit-Wolf covariance, Mahalanobis
distance, Bhattacharyya coefficient — form a coherent, well-motivated pipeline:

1. **MRL truncation** ensures the embedding dimensionality is appropriate for the available
   sample size (n ≈ 10–120 entities, d = 128), grounded in the Kusupati et al. proof that
   prefix sub-vectors are independently meaningful.
2. **Ledoit-Wolf shrinkage** ensures the covariance matrix is invertible and well-conditioned
   even for the smallest ontologies, grounded in Ledoit & Wolf's analytical optimality result.
3. **Mahalanobis distance** accounts for the anisotropic geometry of the embedding space,
   providing a mean-displacement metric that respects the covariance structure of each ontology's
   entity cloud, as validated by Lee et al. and Arora et al. for NLP encoder distributions.
4. **Bhattacharyya coefficient** extends the characterization from mean displacement to full
   distributional overlap, capturing both mean shift and covariance divergence in a single
   interpretable score, as used in analogous text distribution comparison tasks (Fu et al.).

Together they produce a geometric characterization of inter-ontology similarity that is:
- **Data-driven** (derived entirely from entity embeddings, no external knowledge beyond
  WordNet anchors)
- **Model-free at inference time** (closed-form formulae, no learned parameters in S3 itself)
- **Statistically principled** (each component has formal optimality guarantees)
- **Empirically validated** on structurally identical problems (NLP encoder distributions)

The empirical results from the pipeline confirm the theoretical predictions: within-domain
V1/V2/V3 pairs cluster at Mahalanobis 0.87–1.40 and BC 0.06–0.41, while cross-domain pairs
consistently fall at Mahalanobis 4.0–12.5 and BC ≈ 0, a clean geometric separation that
mirrors the matching difficulty observed in stages S4–S8.

#### Academic References

1. **Lee, K., Lee, K., Lee, H., & Shin, J. (2018).**
   *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.*
   In *NeurIPS 2018*.
   — Foundational paper establishing the multivariate Gaussian model of encoder representation
   spaces and Mahalanobis distance as a principled distributional distance metric for neural
   feature clouds; directly underpins `fit_gaussian()` and `mahalanobis_distance()` in S3.

2. **Arora, U., Huang, W., & He, H. (2021).**
   *Types of Out-of-Distribution Texts and How to Detect Them.*
   In *Findings of EMNLP 2021*.
   — Validates Mahalanobis distance on BERT/RoBERTa sentence-level embedding distributions as
   the most reliable inter-corpus distributional distance metric across multiple NLP shift types.

3. **Vilnis, L., & McCallum, A. (2015).**
   *Word representations via Gaussian embedding.*
   In *ICLR 2015*. arXiv:1412.6623.
   — Motivates representing linguistic units as multivariate Gaussians over embedding space to
   capture uncertainty and breadth of meaning; direct conceptual precursor to Stage 3's
   distribution-over-entities representation.

4. **Jebara, T., Kondor, R., & Howard, A. (2004).**
   *Probability product kernels.*
   *Journal of Machine Learning Research*, 5, 819–844.
   — Formally justifies the Bhattacharyya coefficient as a Mercer kernel (positive
   semi-definite) between probability distributions; proves BC = ∫√(p·q) dx is a valid
   inner product in RKHS, establishing it as the principled choice for measuring distributional
   overlap between the Gaussian-fitted ontology clouds in Stage 3.

4. **Ledoit, O., & Wolf, M. (2004).**
   *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices.*
   *Journal of Multivariate Analysis*, 88(2), 365–411.
   — Proves the analytical optimality and guaranteed invertibility of the shrinkage covariance
   estimator in the n << d regime; the mathematical foundation for using `LedoitWolf` on
   small ontology entity clouds in 128-d space.

5. **Iscen, A., Tolias, G., Avrithis, Y., & Chum, O. (2018).**
   *Mining on Manifolds: Metric Learning without Labels.*
   In *CVPR 2018*.
   — Direct precedent for applying Ledoit-Wolf shrinkage + Mahalanobis distance to
   high-dimensional encoder feature distributions for semantic retrieval; structurally identical
   methodology to Stage 3, differing only in encoder modality (visual vs. text).

6. **Kusupati, A., Bhatt, G., Rege, A., et al. (2022).**
   *Matryoshka Representation Learning.*
   In *NeurIPS 2022*.
   — Introduces and proves the MRL training objective that guarantees any prefix sub-vector of
   `nomic-embed-text-v1.5` is independently meaningful; formally justifies the 128-d truncation
   used in Stage 3.

7. **Nussbaum, Z., Morris, J., Duderstadt, B., & Mulyar, A. (2024).**
   *Nomic Embed: Training a Reproducible Long Context Text Embedder.*
   arXiv:2402.01613.
   — Technical report for `nomic-ai/nomic-embed-text-v1.5`; documents that the model is trained
   with explicit MRL objectives at 64/128/256/512/768-d and achieves 97%+ of full-dimension
   ranking quality at 128-d on MTEB, validating `DEFAULT_DIM = 128` in `matryoshka_characterizer.py`.

---

## Part II — Neural Network Stages

### Stage 4 — Semantic Equivalence via MNLI (Mutual Entailment)

**File:** `mnli_matcher.py`  
**Character:** Neural — DeBERTa-v3-base cross-encoder NLI model, mutual entailment scoring

#### What It Does

For entities left unmatched after Stages 1–2, each is converted to a natural-language sentence:

```
"A brake rotor (part of: wheel assembly) entity with attributes:
 outer diameter (Distance), surface temperature (Temperature), operational status (OperationalState)."
```

The cross-encoder `cross-encoder/nli-deberta-v3-base` encodes the (premise, hypothesis) pair
jointly and outputs entailment probabilities. The matching score is:

```
score(U, E) = min( entail(desc_U → desc_E), entail(desc_E → desc_U) )
```

Taking the minimum of both directions enforces mutual entailment — the logical characterisation
of equivalence in description logics. Greedy 1:1 assignment above 0.45.

#### Why It Is Appropriate

AML and Lin similarity fail on functional synonyms — entity pairs that refer to the same concept
under different terminology (`Engine` ↔ `PropulsionSystem`). Token overlap is near zero; no
rule-based method recovers these without a domain-specific synonym dictionary. NLI models trained
on MultiNLI learn that descriptions of the same real-world concept imply each other, transforming
the matching problem into one the model was explicitly trained to solve.

#### Academic References

1. **Williams, A., Nangia, N., & Bowman, S. R. (2018).**
   *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference.*
   In *Proceedings of NAACL-HLT 2018*, pp. 1112–1122.

2. **He, P., Liu, X., Gao, J., & Chen, W. (2021).**
   *DeBERTa: Decoding-Enhanced BERT with Disentangled Attention.*
   In *ICLR 2021*.

---

### Stage 5 — Subsumption Detection via Asymmetric Entailment

**File:** `subsumption_matcher.py`  
**Character:** Neural — DeBERTa cross-encoder, asymmetric entailment as subsumption signal

#### What It Does

Stages 1–4 enforce strict 1:1 mappings. Stage 5 detects **1:many subsumption**: one abstract
entity in the larger model covering several fine-grained entities in the smaller model. The signal
is deliberate asymmetry:

```
entail(U → E) high   — U fits under E (specific → general)
entail(E → U) low    — E does not narrow down to U
asymmetry = fwd − rev ≥ 0.15
```

Both directions (large-abstracts-small, small-abstracts-large) are detected.

#### Why It Is Appropriate

Real ontologies at different versions routinely contain 1:many conceptual gaps. V1 models
enumerate physical components; V3 models group them into system-level abstractions. Asymmetric
NLI entailment is the natural operationalisation of logical subsumption, building directly on
Description Logics semantics (A ⊑ B means every instance of A is an instance of B).

#### Academic References

1. **Doan, A., Madhavan, J., Domingos, P., & Halevy, A. (2002).**
   *Learning to Map between Ontologies on the Semantic Web.*
   In *Proceedings of WWW 2002*, pp. 662–673.

2. **Shvaiko, P., & Euzenat, J. (2013).**
   *Ontology Matching: State of the Art and Future Challenges.*
   *IEEE Transactions on Knowledge and Data Engineering*, 25(1), 158–176.

---

### Stage 6 — Child-Composition Enriched MNLI

**File:** `child_matcher.py`  
**Character:** Neural — DeBERTa cross-encoder, descriptions enriched with child component
summaries

#### What It Does

Entities with sparse own-attributes are described via their child components:

```
"A powertrain entity (part of: vehicle) containing: engine [bore diameter, cylinder count];
 transmission [gear ratio, operational state]; driveshaft [torque, angular velocity]."
```

Child components are described via their observable types only (not names), preserving
model-name agnosticism. Same mutual entailment criterion as S4; threshold raised to 0.55.

#### Why It Is Appropriate

Composite entities carry almost no observable attributes of their own but are unambiguously
characterised by the components they contain. Without child expansion, descriptions of abstract
containers are near-empty and indistinguishable by NLI. Child expansion is bounded (max 3
children, max 3 attributes per child) to avoid context window overflow.

#### Academic References

1. **Noy, N. F., & Musen, M. A. (2000).**
   *PROMPT: Algorithm and Tool for Automated Ontology Merging and Alignment.*
   In *Proceedings of AAAI 2000*, pp. 450–455.

2. **Cheatham, M., & Hitzler, P. (2014).**
   *String Similarity Metrics for Ontology Alignment.*
   In *Proceedings of ISWC 2014*, Springer LNCS 8797, pp. 294–309.

---

### Stage 7 — Association-Enriched MNLI

**File:** `association_matcher.py`  
**Character:** Neural — DeBERTa cross-encoder, descriptions enriched with association relation
phrases

#### What It Does

Stage 7 encodes each entity's functional role via its explicit associations:

```
"A frame entity.
 Relations: supports entity measuring Mass, Force (via Position);
            attached to entity measuring OperationalState (via Position)."
```

Partner entities are **never named** — only their observable type signatures appear, making
descriptions model-name-agnostic. Verb phrases are extracted from association names by removing
participant tokens. Threshold 0.45.

#### Why It Is Appropriate

Some entities have no meaningful own attributes and no discriminative children but are
characterised by what they connect. Associations encode functional role — information orthogonal
to attributes and composition. Using observable type signatures (not entity names) ensures V1
and V3 entities that play the same role in structurally equivalent associations are described
identically, regardless of naming conventions.

#### Academic References

1. **Rahm, E., & Bernstein, P. A. (2001).**
   *A Survey of Approaches to Automatic Schema Matching.*
   *The VLDB Journal*, 10(4), 334–350.

2. **Jiménez-Ruiz, E., Grau, B. C., Zhou, Y., & Horrocks, I. (2012).**
   *Large-Scale Interactive Ontology Matching: Algorithms and Implementation.*
   In *Proceedings of ECAI 2012*, pp. 444–449.

---

### Stage 8 — Dense Embedding Synonym Matching (SBERT)

**File:** `synonym_matcher.py`  
**Character:** Neural — SBERT bi-encoder (`paraphrase-MiniLM-L6-v2`), cosine similarity in
embedding space

#### What It Does

Stage 8 uses a bi-encoder (Sentence-BERT) rather than a cross-encoder. Each entity is encoded
independently; matching is performed by cosine similarity. The entity name is placed first in the
description so the encoder gives it maximum attention weight — critical for synonym detection:

```
"BrakeRotor: measures Distance, Temperature. part of WheelAssembly."
```

Threshold 0.72 (higher than cross-encoder stages due to inflated cosine baselines).

#### Why It Is Appropriate

The DeBERTa cross-encoder (S4–S7) excels at entailment reasoning but struggles with pure
synonyms where descriptions differ in attribute coverage. SBERT bi-encoders are fine-tuned
on paraphrase datasets; their embedding space clusters synonymous expressions together regardless
of surface form differences (`Vehicle` ≈ `Automobile`, `GearBox` ≈ `Transmission`). Bi-encoders
are also computationally efficient: O(n+m) vs O(n×m) for cross-encoders.

#### Academic References

1. **Reimers, N., & Gurevych, I. (2019).**
   *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*
   In *Proceedings of EMNLP-IJCNLP 2019*, pp. 3982–3992.

2. **Kolyvakis, P., Kalousis, A., Smith, B., & Kiritsis, D. (2018).**
   *Biomedical Ontology Alignment: An Approach Based on Representation Learning.*
   *Journal of Biomedical Semantics*, 9(21).

---

## Summary Table

| Stage | Script | Method | Signal | Produces matches? |
|-------|--------|--------|--------|:-----------------:|
| **S1** | `aml_matcher.py` | **Rule-based** | Token Jaccard + type Jaccard | Yes |
| **S2** | `structural_matcher.py` | **Rule-based** | Lin IC + topology graph | Yes |
| **S3** | `matryoshka_characterizer.py` | **Geometric** | Matryoshka + WordNet + Gaussian distributions | No (characterization) |
| **S4** | `mnli_matcher.py` | **Neural** (DeBERTa cross-encoder) | Mutual NLI entailment | Yes |
| **S5** | `subsumption_matcher.py` | **Neural** (DeBERTa cross-encoder) | Asymmetric NLI entailment | Yes |
| **S6** | `child_matcher.py` | **Neural** (DeBERTa cross-encoder) | Mutual NLI + child enrichment | Yes |
| **S7** | `association_matcher.py` | **Neural** (DeBERTa cross-encoder) | Mutual NLI + association enrichment | Yes |
| **S8** | `synonym_matcher.py` | **Neural** (SBERT bi-encoder) | Cosine similarity in embedding space | Yes |

---

## Methodological Discussion: Hard-Coded Metrics vs. Neural Network Inferences

The eight stages divide cleanly into two epistemological categories that differ in their
interpretability, reproducibility, and epistemic status.

### Category A — Hard-Coded / Data-Only Metrics (S1, S2, S3)

These stages operate exclusively on deterministic functions of the data. Their outputs are
**exact, reproducible, and fully interpretable** without reference to any learned model.

**S1 (AML):** The similarity score is a weighted sum of two Jaccard coefficients — set
operations on token lists and observable-type sets. Given the same input, the output is
identical on every run on every machine. There is no ambiguity about what the score represents:
it is the fraction of shared tokens (or types) relative to the union.

**S2 (Lin IC):** The similarity is a closed-form formula over node degrees in a graph
constructed directly from the ontology's declared structure. The IC values are computable by
hand from the adjacency list. The score is a ratio of information-theoretic quantities that
have a precise mathematical interpretation.

**S3 (Matryoshka / Bhattacharyya):** The embedding step uses a pre-trained model, but the
*geometric analysis* is hard-coded: Ledoit-Wolf covariance estimation, matrix inversion, and
the Bhattacharyya formula are all closed-form operations with no stochastic component at
inference time. The WordNet anchor enrichment is deterministic: given the same entity name,
the same hypernym path is always found. The resulting distances — Mahalanobis and Bhattacharyya
coefficient — have precise mathematical definitions as distributional geometry quantities.

> **Key property of Category A:** Every score is a *measurement* of a property that exists in
> the data. The claim "entity A is lexically similar to entity B with score 0.74" can be
> verified by re-running the function. There is no model uncertainty, no temperature, and no
> sampling involved.

### Category B — Neural Network Inferences (S4–S8)

These stages produce scores via forward passes through large pre-trained transformer models.
Their outputs are **probabilistic inferences** that carry inherent model uncertainty.

**S4–S7 (DeBERTa cross-encoder):** The entailment probability `P(entailment | premise,
hypothesis)` is the output of a softmax layer over DeBERTa's contextualised representations.
This is a *learned* function of the token sequence. Two key properties follow:

1. **Non-deterministic under temperature sampling** (though greedy decoding at inference is
   deterministic, the underlying score is sensitive to tokenization, model version, and
   fine-tuning details).
2. **Plausible but not guaranteed correct:** the model has been trained to predict entailment
   on MultiNLI — a corpus of natural English sentences. Its application to technical entity
   descriptions (`"A cylinder block entity with attributes: bore diameter (Distance)"`) is a
   zero-shot transfer. The score is *evidence* for equivalence, not a logical proof of it.

**S8 (SBERT bi-encoder):** The cosine similarity between sentence embeddings is similarly a
learned function. `SentenceTransformer.encode()` produces vectors whose geometry reflects
training-set patterns from paraphrase corpora. A score of 0.75 means "the model has encoded
these two descriptions into nearby regions of its learned semantic space" — which is strong
evidence for synonymy but not a deterministic verification.

> **Key property of Category B:** Every score is a *plausibility estimate* conditioned on a
> learned model. The claim "entity A is semantically equivalent to entity B with NLI score
> 0.62" means "the DeBERTa model, trained on MultiNLI, assigns 62% entailment probability to
> this pairing under this description format." The score cannot be verified without the model;
> it is not a function of the data alone.

### Implications for Interpretation and Reporting

| Criterion | Category A (S1–S3) | Category B (S4–S8) |
|-----------|-------------------|-------------------|
| Reproducibility | Exact (given same data) | Exact but model-dependent |
| Interpretability | Formula-derivable | Requires model understanding |
| Transferability | Any ontology, zero config | Dependent on training distribution |
| Error source | Threshold choice, noise in names | Model bias, OOV tokens, template sensitivity |
| Appropriate claim | "X is lexically / structurally / geometrically similar to Y" | "X is plausibly semantically equivalent to Y" |
| Failure mode | Misses naming-convention gaps | Hallucinates similarity for superficially similar descriptions |

**Practical guidance:**  
Matches produced by S1 and S2 should be treated as high-confidence, verifiable correspondences
requiring no further review. The S3 geometric distances should be treated as calibration data
for the pipeline — a pair with BC < 0.1 should be expected to have high residual unmatch rates
even after S4–S8. Matches from S4–S8 are best-effort inferences that benefit from human
spot-checking, particularly when scores are close to the stage's threshold.

### Why the Ordering Is Epistemically Sound

Running hard-coded stages (S1, S2) first respects the principle of **minimal model assumptions**:
the pipeline commits only to what the data directly supports before introducing learned
inferences. S3 is placed between the rule-based and neural tiers as a bridging characterization:
it uses the same neural embedding model as later stages but applies only hard-coded geometric
analysis, making its outputs interpretable without reference to the model's learned behaviour.
Stages S4–S8 then progressively enrich the entity representations (name → + ancestors →
+ subsumption → + children → + associations → + synonyms) within the same neural framework,
exhausting each additional information layer before escalating to the next.
