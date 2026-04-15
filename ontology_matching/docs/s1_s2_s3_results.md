# S1–S3 Pipeline Results: Automobile & Hospital Domains

## Column Definitions

Each row represents a **pairwise comparison** between two ontology models within the same domain.
The pipeline runs three stages sequentially; later stages build on earlier matches.

| Column | Stage | Description |
|---|---|---|
| **Model A** | — | Short name of the first model (see model key below each domain) |
| **Model B** | — | Short name of the second model |
| **n_min** | — | Size of the *smaller* ontology in the pair — used as the denominator for coverage fractions |
| **S1 matched** | S1 | Raw count of entity matches found by AML (Stage 1 lexical + type Jaccard + association vocabulary) expressed as `matched / n_min` |
| **S1_cov** | S1 | S1 coverage fraction = S1 matched / n_min. Range [0, 1]. Measures how much of the smaller ontology AML could align lexically. |
| **S2 new** | S2 | Count of *additional* matches found by Lin-IC structural propagation (Stage 2), not already found by S1. Expressed as `new / n_min`. |
| **S2_cov** | S2 | S2 coverage fraction = S2 new / n_min. Range [0, 1]. Measures incremental structural alignment beyond S1. |
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
### Results (semantic-first SBERT + BFS-2 topology gate + WordNet filter — run 2026-04-02)
| Model A | Model B | n_min | S1 matched | S1_cov | Mah | MahN | BC |
|---|---|---|---|---|---|---|---|
| V1-System | V2-Component | 72 | 47/72 | 0.6528 | 2.8688 | 0.2890 | 0.009122 |
| V1-System | V3-Functional | 77 | 49/77 | 0.6364 | 1.4930 | 0.1504 | 0.033784 |
| V2-Component | V3-Functional | 72 | 62/72 | 0.8611 | 2.3135 | 0.2331 | 0.021176 |
| V1-System | NetMech | 17 | 9/17 | 0.5294 | 5.9346 | 0.5979 | 0.000022 |
| V1-System | NetPack | 16 | 1/16 | 0.0625 | 9.9503 | 1.0025 | 0.000000 |
| V1-System | NetSvc | 17 | 11/17 | 0.6471 | 5.1469 | 0.5185 | 0.000111 |
| V2-Component | NetMech | 17 | 9/17 | 0.5294 | 5.3281 | 0.5368 | 0.000048 |
| V2-Component | NetPack | 16 | 0/16 | 0.0000 | 9.3374 | 0.9407 | 0.000000 |
| V2-Component | NetSvc | 17 | 10/17 | 0.5882 | 4.4421 | 0.4475 | 0.000264 |
| V3-Functional | NetMech | 17 | 7/17 | 0.4118 | 5.0136 | 0.5051 | 0.000038 |
| V3-Functional | NetPack | 16 | 1/16 | 0.0625 | 7.3612 | 0.7416 | 0.000000 |
| V3-Functional | NetSvc | 17 | 11/17 | 0.6471 | 4.5331 | 0.4567 | 0.000123 |
| NetMech | NetPack | 16 | 1/16 | 0.0625 | 8.7234 | 0.8789 | 0.000000 |
| NetMech | NetSvc | 17 | 0/17 | 0.0000 | 6.4520 | 0.6500 | 0.000022 |
| NetPack | NetSvc | 16 | 0/16 | 0.0000 | 9.9456 | 1.0020 | 0.000000 |

### Observations

**S2 architecture change in this run**: inverted to semantic-first.
Previously: topology-first (anchor BFS neighbours as candidate pool, then cosine ranks).
Now: semantic-first (full SBERT cosine matrix across all unmatched pairs, then topology validates).
- All unmatched entities in A and B are batch-encoded in one GPU call each.
- Pairs above the cosine threshold (0.60) are tested for topological adjacency via matched anchors.
- Borderline pairs (0.60 ≤ cos < 0.75) must additionally pass a WordNet compound_sim ≥ 0.30 gate.
- Final assignment is global-greedy (sort by score, no sequential first-pick bias).

**S2 matches confirmed** (6 total across all pairs):
- Automobile ↔ Vehicle (V1-V2, cos=0.850, cosine_struct)
- Valve ↔ ValveTrain (V2-NetMech, cos=0.803, cosine_struct)
- CombustionAssembly ↔ EngineAssembly (V3-NetPack, cos=0.624, cosine_struct)
- ExhaustSystem ↔ ExhaustAssembly (V1-NetPack, assoc_vocab)
- FuelRail ↔ InjectorRail (V2-NetSvc, assoc_vocab)
- ExhaustAssembly ↔ Radiator (NetPack-NetSvc, assoc_vocab)

**V–V pairs**: V1-V2 retains 1 cosine_struct match (Automobile↔Vehicle, cos=0.850). V1-V3 and
V2-V3 drop to 0 S2 matches — the remaining unmatched entities in these well-aligned pairs lack
topological paths through the current anchors. S3 (Mah/BC) remains strong for these pairs
(V2-V3 dist=0.588, V1-V3 dist=0.620), confirming they are well-aligned overall.

**V–Net pairs**: Most V-Net pairs now show S2_cov=0 because the unmatched V-model entities
(Axle, Piston, BrakeDisc, Suspension) are either absent from the graph or lack topological
connections to mapped anchors. V2-NetMech and V2-NetSvc each recover 1 match via the semantic
gate (Valve↔ValveTrain) or assoc_vocab (FuelRail↔InjectorRail). V-Net composite distances
range 0.70–0.97, reflecting the genuine cross-type structural divergence.

**Net–Net pairs**: Only NetPack-NetSvc finds 1 assoc_vocab match. NetMech-NetPack and NetMech-NetSvc
find nothing — consistent with MahN ≈ 0.88 and 0.65 showing moderate-to-high embedding divergence.

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

### Results (Sanchez 2011 IC + Lastra-Diaz 2015 WB-sim)

| Type | Model A | Model B | n_min | S1 matched | S1_cov | S2 new | S2_cov | S2_raw | S2_lin | Mah | MahN | BC | sim | dist |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V-V | V1-Dept | V2-Equip | 64 | 57/64 | 0.891 | 1/64 | 0.016 | 0.0 | 0.0 | 2.2952 | 0.2312 | 0.058685 | 0.4334 | 0.5666 |
| V-V | V1-Dept | V3-Func | 112 | 93/112 | 0.830 | 1/112 | 0.009 | 0.0 | 0.0 | 1.1805 | 0.1189 | 0.273654 | 0.4985 | 0.5015 |
| V-V | V2-Equip | V3-Func | 64 | 59/64 | 0.922 | 1/64 | 0.016 | 0.0 | 0.0 | 1.9931 | 0.2008 | 0.072902 | 0.4524 | 0.5476 |
| V-Net | V1-Dept | NetORC | 14 | 0/14 | 0.000 | 0/14 | 0.000 | — | — | 7.7303 | 0.7788 | 0.000001 | 0.0553 | 0.9447 |
| V-Net | V1-Dept | NetSIN | 14 | 5/14 | 0.357 | 5/14 | 0.357 | 1.582 | 1.000 | 6.0965 | 0.6142 | 0.000028 | 0.2750 | 0.7250 |
| V-Net | V1-Dept | NetSFP | 14 | 1/14 | 0.071 | 0/14 | 0.000 | — | — | 7.1338 | 0.7187 | 0.000024 | 0.0882 | 0.9118 |
| V-Net | V2-Equip | NetORC | 14 | 1/14 | 0.071 | 0/14 | 0.000 | — | — | 8.1894 | 0.8251 | 0.000001 | 0.0616 | 0.9384 |
| V-Net | V2-Equip | NetSIN | 14 | 6/14 | 0.429 | 4/14 | 0.286 | 1.311 | 1.000 | 5.8895 | 0.5934 | 0.000057 | 0.2802 | 0.7198 |
| V-Net | V2-Equip | NetSFP | 14 | 1/14 | 0.071 | 1/14 | 0.071 | 2.026 | 1.000 | 7.1948 | 0.7249 | 0.000021 | 0.1045 | 0.8955 |
| V-Net | V3-Func | NetORC | 14 | 1/14 | 0.071 | 3/14 | 0.214 | 1.479 | 1.000 | 7.9097 | 0.7969 | 0.000001 | 0.1222 | 0.8778 |
| V-Net | V3-Func | NetSIN | 14 | 6/14 | 0.429 | 6/14 | 0.429 | 1.489 | 1.000 | 6.2293 | 0.6276 | 0.000021 | 0.3074 | 0.6926 |
| V-Net | V3-Func | NetSFP | 14 | 1/14 | 0.071 | 0/14 | 0.000 | — | — | 7.2623 | 0.7317 | 0.000017 | 0.0849 | 0.9151 |
| Net-Net | NetORC | NetSIN | 14 | 0/14 | 0.000 | 0/14 | 0.000 | — | — | 9.9527 | 1.0027 | 0.000000 | 0.0000 | 1.0000 |
| Net-Net | NetORC | NetSFP | 14 | 0/14 | 0.000 | 0/14 | 0.000 | — | — | 10.0195 | 1.0094 | 0.000000 | 0.0000 | 1.0000 |
| Net-Net | NetSFP | NetSIN | 14 | 0/14 | 0.000 | 0/14 | 0.000 | — | — | 9.4301 | 0.9501 | 0.000000 | 0.0125 | 0.9875 |

### Observations

**V–V pairs**: the Sanchez IC + WB-sim update caused a significant reduction in S2 coverage.
Former Lin-IC matches (old: 2–10 per pair, S2_lin = 1.000) dropped below the 0.5 threshold;
only 1 assoc_vocab match per pair survives (S2_raw = 0.0, S2_lin = 0.0). This reflects the stricter
Sanchez leaf-count IC redistributing scores for Hospital's densely connected V-model graphs.
Composite distances increased slightly: V1-Dept vs V3-Func moved from dist = 0.481 to 0.501
(no longer sub-0.50 — it was the only such pair in either domain under the old formula).
S3 (Mah/BC) is unchanged: V1-Dept vs V3-Func remains the strongest pair overall (MahN = 0.119,
BC = 0.274).

**V–Net pairs**: V-Net Lin-IC matches are preserved for all pairs where they existed before
(NetSIN pairings and V3-Func vs NetORC). S2_raw values (1.31–2.03) confirm the WB-sim
denominator reduced but did not eliminate the >1 anomaly on undirected graphs.
V1-Dept vs NetSIN and V3-Func vs NetSIN retain their high S2_cov (0.357 and 0.429).

**Net–Net pairs**: unchanged from before — complete failure on S1 and S2. MahN ≈ 1.0 and BC = 0
on all three pairs.

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
