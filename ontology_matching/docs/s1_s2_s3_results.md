# S1–S3 Pipeline Results: Automobile & Hospital Domains

## Column Definitions

Each row represents a **pairwise comparison** between two ontology models within the same domain.
The pipeline runs three stages sequentially; later stages build on earlier matches.

| Column | Stage | Description |
|---|---|---|
| **Type** | — | Pair category: `V-V` (two versioned models), `V-Net` (versioned vs network model), `Net-Net` (two network models) |
| **Model A** | — | Short name of the first model (see model key below each domain) |
| **Model B** | — | Short name of the second model |
| **n_min** | — | Size of the *smaller* ontology in the pair — used as the denominator for coverage fractions |
| **S1 matched** | S1 | Raw count of entity matches found by AML (Stage 1 lexical + type Jaccard + association vocabulary) expressed as `matched / n_min` |
| **S1_cov** | S1 | S1 coverage fraction = S1 matched / n_min. Range [0, 1]. Measures how much of the smaller ontology AML could align lexically. |
| **S2 new** | S2 | Count of *additional* matches found by Lin-IC structural propagation (Stage 2), not already found by S1. Expressed as `new / n_min`. |
| **S2_cov** | S2 | S2 coverage fraction = S2 new / n_min. Range [0, 1]. Measures incremental structural alignment beyond S1. |
| **S2_lin** | S2 | Average Lin similarity of the S2 new matches. Range [0, 1]. Lin sim = 2·IC(LCA) / (IC(A) + IC(B)) where LCA is the lowest common ancestor in the triadic graph. A value of 1.0 means every new match shares a high-IC common ancestor with its propagating anchor. `—` when S2 found no new matches. Matches via the injected domain-root anchor are penalised by ×0.20 before this average. |
| **Mah** | S3 | Raw Mahalanobis distance between the two multivariate Gaussian embeddings (Matryoshka 128-d, Ledoit-Wolf covariance). Smaller = more similar in embedding space. |
| **MahN** | S3 | Normalised Mahalanobis = Mah / 9.93, where 9.93 is the median cross-domain Mahalanobis (Automobile vs Coffee). **MahN = 1.0 means the pair is as far apart in embedding space as a typical cross-domain comparison.** MahN < 0.3 indicates strong within-type proximity. |
| **BC** | S3 | Bhattacharyya coefficient between the two Gaussian distributions. Range [0, 1]. Measures the overlap of the probability distributions. BC ≈ 0 means the distributions do not overlap; BC closer to 1 means near-identical distributions. |
| **sim** | Composite | Composite similarity = 0.25 × S1_cov + 0.25 × S2_cov + 0.25 × (1 − min(MahN, 1)) + 0.25 × BC. Each of the four signals contributes equally. `—` when any component is missing. |
| **dist** | Composite | Composite distance = 1 − sim. Range [0, 1]. Lower = more similar overall. |

### Composite formula

```
sim  = 0.25 × S1_cov
     + 0.25 × S2_cov
     + 0.25 × (1 − min(MahN, 1.0))   ← S3 Mahalanobis component; capped so it cannot go negative
     + 0.25 × BC                       ← S3 Bhattacharyya component
dist = 1 − sim
```

Cross-domain anchor: Coffee domain (median Mah = **9.93**). A pair with MahN = 1.0 is at the same
embedding-space distance as an Automobile–Coffee cross-domain comparison.

---

## Domain: Automobile

### Model key

| Short name | Full name | Size |
|---|---|---|
| V1-System | Automobile_Model_V1_SystemCentric | ~69 entities |
| V2-Component | Automobile_Model_V2_ComponentCentric | ~72 entities |
| V3-Functional | Automobile_Model_V3_FunctionalDomain | ~108 entities |
| NetMech | Component_Network_Mechanical_and_Structural | 17 entities |
| NetPack | Component_Network_Packaged_Assemblies | 16 entities |
| NetSvc | Component_Network_Serviceable_Parts_Interaction | 17 entities |

### Results

| Type | Model A | Model B | n_min | S1 matched | S1_cov | S2 new | S2_cov | S2_lin | Mah | MahN | BC | sim | dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V-V | V1-System | V2-Component | 72 | 47/72 | 0.653 | 2/72 | 0.028 | 1.000 | 2.8688 | 0.2890 | 0.009122 | 0.3502 | 0.6498 |
| V-V | V1-System | V3-Functional | 77 | 49/77 | 0.636 | 11/77 | 0.143 | 1.000 | 1.4930 | 0.1504 | 0.033784 | 0.4156 | 0.5844 |
| V-V | V2-Component | V3-Functional | 72 | 62/72 | 0.861 | 3/72 | 0.042 | 1.000 | 2.3135 | 0.2331 | 0.021176 | 0.4227 | 0.5773 |
| V-Net | V1-System | NetPack | 16 | 1/16 | 0.062 | 4/16 | 0.250 | 1.000 | 9.9503 | 1.0025 | 0.000000 | 0.0781 | 0.9219 |
| V-Net | V1-System | NetMech | 17 | 9/17 | 0.529 | 4/17 | 0.235 | 1.000 | 5.9346 | 0.5979 | 0.000022 | 0.2917 | 0.7083 |
| V-Net | V1-System | NetSvc | 17 | 11/17 | 0.647 | 4/17 | 0.235 | 1.000 | 5.1469 | 0.5185 | 0.000111 | 0.3410 | 0.6590 |
| V-Net | V2-Component | NetPack | 16 | 0/16 | 0.000 | 0/16 | 0.000 | — | 9.3374 | 0.9407 | 0.000000 | 0.0148 | 0.9852 |
| V-Net | V2-Component | NetMech | 17 | 9/17 | 0.529 | 5/17 | 0.294 | 1.000 | 5.3281 | 0.5368 | 0.000048 | 0.3217 | 0.6783 |
| V-Net | V2-Component | NetSvc | 17 | 10/17 | 0.588 | 4/17 | 0.235 | 0.750 | 4.4421 | 0.4475 | 0.000264 | 0.3441 | 0.6559 |
| V-Net | V3-Functional | NetPack | 16 | 1/16 | 0.062 | 0/16 | 0.000 | — | 7.3612 | 0.7416 | 0.000000 | 0.0802 | 0.9198 |
| V-Net | V3-Functional | NetMech | 17 | 7/17 | 0.412 | 5/17 | 0.294 | 0.600 | 5.0136 | 0.5051 | 0.000038 | 0.3002 | 0.6998 |
| V-Net | V3-Functional | NetSvc | 17 | 11/17 | 0.647 | 4/17 | 0.235 | 1.000 | 4.5331 | 0.4567 | 0.000123 | 0.3564 | 0.6436 |
| Net-Net | NetPack | NetSvc | 16 | 0/16 | 0.000 | 4/16 | 0.250 | 0.750 | 9.9456 | 1.0020 | 0.000000 | 0.0625 | 0.9375 |
| Net-Net | NetMech | NetPack | 16 | 1/16 | 0.062 | 4/16 | 0.250 | 0.500 | 8.7234 | 0.8789 | 0.000000 | 0.1084 | 0.8916 |
| Net-Net | NetMech | NetSvc | 17 | 0/17 | 0.000 | 3/17 | 0.176 | 0.667 | 6.4520 | 0.6500 | 0.000022 | 0.1316 | 0.8684 |

### Observations

**V–V pairs** show strong S1 coverage (0.64–0.86), low MahN (0.15–0.29), and non-trivial BC (0.009–0.034).
V1 vs V3 is the closest pair (MahN = 0.150, dist = 0.584); V1 vs V2 is the most distant V–V pair (MahN = 0.289).

**V–Net pairs** split into two groups:
- *NetMech and NetSvc*: moderate S1 coverage (0.41–0.65), MahN 0.45–0.60 — these network models share mechanical/serviceable vocabulary with the V-models.
- *NetPack (Packaged Assemblies)*: near-zero S1 (0–0.06), MahN 0.74–1.00 — packaging vocabulary does not overlap with functional/system naming. This model is the most isolated in the entire domain.

**Net–Net pairs**: S1 ≈ 0, S2 provides some structural propagation (0.18–0.25 coverage) via association anchors, but BC = 0 on all pairs and MahN 0.65–1.00. The three network models occupy distinct embedding clusters with no lexical or distributional overlap.

---

## Domain: Hospital

### Model key

| Short name | Full name | Size |
|---|---|---|
| V1-Dept | Hospital_Model_V1_DepartmentalStructure | ~103 entities |
| V2-Equip | Hospital_Model_V2_EquipmentAndSpaceCentric | ~57 entities |
| V3-Func | Hospital_Model_V3_FunctionalLayer | ~114 entities |
| NetORC | Facility_Resource_Network_Operational_Resource_Clusters | 14 entities |
| NetSFP | Facility_Resource_Network_Serviceable_Facility_Parts | 14 entities |
| NetSIN | Facility_Resource_Network_Spatial_Infrastructure | 14 entities |

### Results

| Type | Model A | Model B | n_min | S1 matched | S1_cov | S2 new | S2_cov | S2_lin | Mah | MahN | BC | sim | dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V-V | V1-Dept | V2-Equip | 64 | 57/64 | 0.891 | 3/64 | 0.047 | 1.000 | 2.2952 | 0.2312 | 0.058685 | 0.4412 | 0.5588 |
| V-V | V1-Dept | V3-Func | 112 | 93/112 | 0.830 | 10/112 | 0.089 | 1.000 | 1.1805 | 0.1189 | 0.273654 | 0.5186 | 0.4814 |
| V-V | V2-Equip | V3-Func | 64 | 59/64 | 0.922 | 2/64 | 0.031 | 1.000 | 1.9931 | 0.2008 | 0.072902 | 0.4563 | 0.5437 |
| V-Net | V1-Dept | NetORC | 14 | 0/14 | 0.000 | 0/14 | 0.000 | — | 7.7303 | 0.7788 | 0.000001 | 0.0553 | 0.9447 |
| V-Net | V1-Dept | NetSIN | 14 | 5/14 | 0.357 | 5/14 | 0.357 | 1.000 | 6.0965 | 0.6142 | 0.000028 | 0.2750 | 0.7250 |
| V-Net | V1-Dept | NetSFP | 14 | 1/14 | 0.071 | 0/14 | 0.000 | — | 7.1338 | 0.7187 | 0.000024 | 0.0882 | 0.9118 |
| V-Net | V2-Equip | NetORC | 14 | 1/14 | 0.071 | 0/14 | 0.000 | — | 8.1894 | 0.8251 | 0.000001 | 0.0616 | 0.9384 |
| V-Net | V2-Equip | NetSIN | 14 | 6/14 | 0.429 | 4/14 | 0.286 | 1.000 | 5.8895 | 0.5934 | 0.000057 | 0.2802 | 0.7198 |
| V-Net | V2-Equip | NetSFP | 14 | 1/14 | 0.071 | 1/14 | 0.071 | 1.000 | 7.1948 | 0.7249 | 0.000021 | 0.1045 | 0.8955 |
| V-Net | V3-Func | NetORC | 14 | 1/14 | 0.071 | 3/14 | 0.214 | 1.000 | 7.9097 | 0.7969 | 0.000001 | 0.1222 | 0.8778 |
| V-Net | V3-Func | NetSIN | 14 | 6/14 | 0.429 | 6/14 | 0.429 | 1.000 | 6.2293 | 0.6276 | 0.000021 | 0.3074 | 0.6926 |
| V-Net | V3-Func | NetSFP | 14 | 1/14 | 0.071 | 0/14 | 0.000 | — | 7.2623 | 0.7317 | 0.000017 | 0.0849 | 0.9151 |
| Net-Net | NetORC | NetSIN | 14 | 0/14 | 0.000 | 0/14 | 0.000 | — | 9.9527 | 1.0027 | 0.000000 | 0.0000 | 1.0000 |
| Net-Net | NetORC | NetSFP | 14 | 0/14 | 0.000 | 0/14 | 0.000 | — | 10.0195 | 1.0094 | 0.000000 | 0.0000 | 1.0000 |
| Net-Net | NetSFP | NetSIN | 14 | 0/14 | 0.000 | 0/14 | 0.000 | — | 9.4301 | 0.9501 | 0.000000 | 0.0125 | 0.9875 |

### Observations

**V–V pairs** are the closest pairs in the domain. V1-Dept vs V3-Func is the strongest match overall
(MahN = 0.119, BC = 0.274, dist = 0.481 — the only pair in either domain with dist < 0.50).
S1 coverage is very high (0.83–0.92), confirming that the three V-models share a common entity vocabulary.

**V–Net pairs** again split by network model:
- *NetSIN (Spatial Infrastructure)*: consistently the best-connected network model. S1 picks up 5–6 matches against all three V-models; S2 adds a further 4–6; MahN ranges 0.61–0.63. The spatial/room/ward terminology overlaps with both departmental and equipment-centric vocabularies.
- *NetSFP (Serviceable Facility Parts)*: mostly isolated (S1 = 0–1, S2 = 0–1, MahN 0.72–0.73). Only NetSIN is semantically closer.
- *NetORC (Operational Resource Clusters)*: most isolated network model. S1 = 0 against V1; MahN 0.78–0.82; BC ≈ 0 everywhere. No lexical or distributional overlap with V-models.

**Net–Net pairs**: complete failure on S1 and S2. All three pairs have MahN ≈ 1.0 (cross-domain level)
and BC = 0, meaning the three Hospital network models occupy fully disjoint semantic regions —
they are as unrelated to each other as they would be to a model from a different domain entirely.

---

## Cross-domain reference

| Reference | Mah | MahN |
|---|---|---|
| Automobile V-model vs Coffee V-model (median) | ~4.9 | ~0.49 |
| Automobile Network vs Coffee V-model (median) | ~10.5 | ~1.06 |
| Calibration anchor (median all Coffee cross-domain) | 9.93 | **1.00** |

The calibration anchor confirms that Hospital Net–Net pairs (MahN ≈ 1.0) sit at the same
embedding distance as typical cross-domain comparisons — they are not just different
model types, they are different semantic worlds.
