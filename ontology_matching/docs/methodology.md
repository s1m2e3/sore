# Ontology Matching Methodology: AML + MNLI Two-Stage Pipeline

## Overview

This document describes the methodology used to align conceptual model instances drawn from six domains: Automobile, Coffee, Homebrewing, Hospital, SmartHome, and University. Each domain contains multiple model variants (V1, V2, V3) that describe the same physical or organisational system at different levels of detail, abstraction, or modelling conventions. The goal is to determine, for any two models within the same domain, which entities and attributes correspond to one another — and to quantify the coverage achieved.

The pipeline proceeds in two ordered stages:

1. **Stage 1 — Lexical/Structural Alignment via AgreementMakerLight (AML)**
2. **Stage 2 — Semantic Recovery via Multi-Genre Natural Language Inference (MNLI)**

---

## Stage 1: AgreementMakerLight (AML)

### What AML Is

AgreementMakerLight (v3.2) is a **mechanistic, rule-based ontology matching system**. It does not learn from data; it applies deterministic procedures to find correspondences between OWL individuals based on their names and structural context. Its core matching strategy relies on:

- **Lexical label similarity**: string edit distance, sub-token overlap, camelCase decomposition, and acronym expansion are applied to entity and attribute names.
- **Structural property profiles**: AML compares the set of OWL object/data properties attached to two individuals. If two entities share similar property profiles (e.g., both have a `hasAttribute` linking to similar observable types), their correspondence score is boosted.
- **Propagation**: confirmed matches propagate confidence to neighbouring individuals in the graph.

### Why AML Is Suitable Here

The source data — conceptual models encoded as OWL instance graphs — are produced programmatically from structured JSON files. Entity names follow consistent naming conventions (PascalCase, domain-standard abbreviations), and attribute observable types are drawn from a shared vocabulary. These properties make the dataset well-suited for rule-based lexical matching:

- Names like `BrakeRotor`, `FuelTank`, `WaterHeater` map cleanly via sub-token overlap.
- Observable type profiles (e.g., both entities have attributes of type `Temperature`, `Pressure`, `FlowRate`) provide a strong structural signal for disambiguation.
- No training data is required; AML generalises across all six domains with zero configuration change.

### Alignment Setup

For each model pair (A, B) within a domain, AML is run **bidirectionally**: A→B and B→A. An entity in model A is counted as **matched** only when the correspondence is confirmed in both directions (mutual agreement). This prevents asymmetric false positives where AML maps many entities in A to a single prominent entity in B.

All models are first serialised to OWL/RDF-XML using a shared conceptual metamodel (entities, attributes, observable types, associations). The OWL representation faithfully mirrors the JSON source without imposing structural hierarchies that could bias AML's property-profile comparisons.

### Coverage Metric

Coverage is computed using the **smaller model as reference**. If model A has 56 entities and model B has 83, coverage = matched_A / 56. This ensures the metric reflects how completely the smaller, less redundant vocabulary is represented in the larger model — a meaningful question when models differ in scope and granularity.

### AML Results Summary

| Domain       | Pair (smaller → larger)     | AML Coverage |
|--------------|-----------------------------|--------------|
| Automobile   | V2 → V1                     | 53.6 %       |
| Automobile   | V1 → V3                     | 59.5 %       |
| Automobile   | V2 → V3                     | 69.1 %       |
| Coffee       | V2 → V1                     | 79.9 %       |
| Coffee       | V3 → V1                     | 71.9 %       |
| Coffee       | V2 → V3                     | 81.9 %       |
| Homebrewing  | V2 → V1                     | 73.2 %       |
| Homebrewing  | V1 → V3                     | 67.5 %       |
| Homebrewing  | V2 → V3                     | 79.4 %       |
| Hospital     | V2 → V1                     | 83.2 %       |
| Hospital     | V3 → V1                     | 79.7 %       |
| Hospital     | V2 → V3                     | 68.1 %       |
| SmartHome    | V2 → V1                     | 62.4 %       |
| SmartHome    | V3 → V1                     | 67.7 %       |
| SmartHome    | V2 → V3                     | 79.9 %       |
| University   | V2 → V1                     | 84.8 %       |
| University   | V3 → V1                     | 68.1 %       |
| University   | V2 → V3                     | 86.6 %       |

### Limitations of Lexical Matching

AML's lexical stage performs well when models share surface-form names. However, conceptual models at different abstraction levels frequently express the same concept under different terminology:

- `Engine` (V1) vs `PropulsionSystem` (V3)
- `HVACSystem` (V2) vs `ClimateControlDomain` (V3)
- `ColdBrewContainer` vs `ColdBrewMaker`
- `ImagingSuite` vs `DiagnosticsLayer`

These are **functional synonyms**: entities that refer to the same real-world concept but whose names share no lexical overlap. Edit-distance and token-overlap metrics assign near-zero similarity scores to these pairs; they leave entities unmapped even when a human expert would immediately recognise the correspondence.

This motivates a second, semantically richer stage.

---

## Stage 2: Semantic Recovery via MNLI

### Motivation

For entities that AML leaves unmatched, we ask a different question: do the two entities **entail** one another in natural language? If a concise description of entity A logically implies a description of entity B, and vice versa, they are strong candidates for correspondence.

This is operationalised using **Natural Language Inference (NLI)**, a well-established task in computational linguistics where a model is given a premise and a hypothesis and must predict whether the premise entails, contradicts, or is neutral with respect to the hypothesis.

### Entity Description Construction

Each entity is converted to a short natural-language sentence using its name and attribute inventory:

```
"A brake rotor entity with attributes: outer diameter (Distance),
 surface temperature (Temperature), operational status (OperationalState)."
```

The name is camelCase-split to recover human-readable tokens (`BrakeRotor` → `brake rotor`). Attributes are listed with their observable type in parentheses. The description is capped at 12 attributes to stay within the model's token limit.

This representation encodes both the **nominal identity** of the entity (its name) and its **functional signature** (what it measures or contains), giving the NLI model a richer signal than name alone.

### NLI Model

The cross-encoder `cross-encoder/nli-MiniLM2-L6-H768` (~120 MB) is used. It is a bidirectional transformer fine-tuned on MNLI and related datasets. Given a (premise, hypothesis) pair encoded jointly, it outputs three logits corresponding to **entailment**, **neutral**, and **contradiction**.

### Mutual Entailment Score

For two entities U (unmatched, smaller model) and E (candidate, larger model), the matching score is defined as:

```
score(U, E) = min( entail(desc_U → desc_E), entail(desc_E → desc_U) )
```

The **minimum** of both directions is taken deliberately:

- **Asymmetric entailment** (e.g., `ECU` → `ElectricalSystem` but not vice versa) signals subsumption, not equivalence, and produces a low minimum score.
- **Mutual entailment** (both directions above threshold) signals semantic equivalence and is the target criterion.

This formulation is principled: mutual entailment is a standard logical characterisation of synonymy/equivalence in description logics. A high mutual entailment score means neither description introduces information the other contradicts, and neither is strictly more general — they describe the same concept.

### Matching Procedure

1. Identify all entities with `status == "missing"` in the smaller model after AML.
2. Build the cross-product of missing entities × all entities in the larger model.
3. Score all pairs via mutual entailment.
4. Apply **greedy 1:1 assignment** (Hungarian-style, score-descending) above a threshold of 0.45.
5. Write supplementary JSON per pair and append to the combined coverage CSV.

Threshold 0.45 was selected empirically: it admits confident functional synonyms while rejecting near-misses and generalisation chains.

---

## Stage 3: Subsumption Detection

### Motivation

Stages 1 and 2 enforce a strict **1:1 mapping** constraint. This is correct for equivalence — two entities that describe the same concept at the same level of abstraction. However, the residual analysis revealed a consistent pattern across all domains: fine-grained entities in one model (e.g., `CylinderBlock`, `CylinderHead`, `Cylinder`) have no equivalents in the other model because the other model expresses the same territory as a single abstract entity (e.g., `Engine`). This is a **subsumption** relationship, not equivalence, and requires a different matching structure.

### Asymmetric Entailment as a Subsumption Signal

Stage 2 uses *mutual* entailment (`min(A→B, B→A)`) to detect equivalence. The same NLI model produces an **asymmetric** signal when one entity is more specific than another:

```
entail(U → E) high   :  U's description is consistent with / implied by E
entail(E → U) low    :  E does not narrow down to U — E is more general
asymmetry = entail(U→E) − entail(E→U) > threshold
```

When multiple residual entities U₁, U₂, … from one model each satisfy this criterion against the same entity E in the other model, E is a semantic abstraction that **covers** the group. This is the 1:many relationship the 1:1 constraint cannot capture.

### Both Directions

Subsumption can run in either direction regardless of model size:

- **`large_abstracts_small`**: One entity in the larger model is an abstraction of several fine-grained entities in the smaller model. Most common pattern: V3 system entities covering V1 component entities.
- **`small_abstracts_large`**: One abstract entity in the smaller model covers several entities in the larger model. Occurs when the smaller model is architecturally high-level (e.g., `DiagnosticsLayer` in a small V3 model subsuming `ImagingSuite`, `CardiologyLab`, `PulmonaryLab` in a larger V1 model).

### Algorithm

```
For each residual entity U (not matched by Stage 1 or Stage 2):
  For each entity E in the other model:
    fwd       = entail(desc_U → desc_E)
    rev       = entail(desc_E → desc_U)
    asymmetry = fwd − rev

  If fwd ≥ 0.50 AND asymmetry ≥ 0.15:
    candidate: E subsumes U  (direction: large_abstracts_small)

Group by abstract entity E:
  If multiple U_i map to the same E → report subsumption group

Run the inverse direction symmetrically.
```

### Output: Three-Tier Alignment Status

Stage 3 introduces a new alignment status, giving three distinct outcomes per entity:

| Status | Meaning | Detected by |
|---|---|---|
| **matched** | 1:1 semantic equivalence | AML or Stage 2 MNLI |
| **covered** | Subsumed by / subsumes an abstract entity | Stage 3 asymmetric MNLI |
| **absent** | No correspondence found at any level | None of the three stages |

This distinction is important: "covered" entities are not absent from the other ontology — they are present at a different level of granularity. "Absent" entities represent genuine modelling scope differences.

---

## Combined Results

| Domain      | Pair                 | AML Coverage | MNLI Recovered | Combined Coverage |
|-------------|----------------------|:------------:|:--------------:|:-----------------:|
| Automobile  | V2 → V1              | 53.6 %       | 0              | 53.6 %            |
| Automobile  | V1 → V3              | 59.5 %       | 7              | 71.4 %            |
| Automobile  | V2 → V3              | 69.1 %       | 0              | 69.1 %            |
| Coffee      | V2 → V1              | 79.9 %       | 1              | 81.0 %            |
| Coffee      | V3 → V1              | 71.9 %       | 6              | 84.4 %            |
| Coffee      | V2 → V3              | 81.9 %       | 0              | 81.9 %            |
| Homebrewing | V2 → V1              | 73.2 %       | 1              | 75.0 %            |
| Homebrewing | V1 → V3              | 67.5 %       | 4              | 77.5 %            |
| Homebrewing | V2 → V3              | 79.4 %       | 0              | 79.4 %            |
| Hospital    | V2 → V1              | 83.2 %       | 1              | 84.4 %            |
| Hospital    | V3 → V1              | 79.7 %       | 2              | 82.1 %            |
| Hospital    | V2 → V3              | 68.1 %       | 0              | 68.1 %            |
| SmartHome   | V2 → V1              | 62.4 %       | 0              | 62.4 %            |
| SmartHome   | V3 → V1              | 67.7 %       | 3              | 72.6 %            |
| SmartHome   | V2 → V3              | 79.9 %       | 1              | 81.0 %            |
| University  | V2 → V1              | 84.8 %       | 0              | 84.8 %            |
| University  | V3 → V1              | 68.1 %       | 2              | 71.0 %            |
| University  | V2 → V3              | 86.6 %       | 0              | 86.6 %            |

MNLI recovery is most effective on **V1↔V3 pairs** — where V3 models adopt a functional/domain-oriented vocabulary (e.g., `PropulsionSystem`, `ClimateControlDomain`) while V1 models use component-level engineering names. V2↔V3 pairs show zero MNLI recovery, indicating the residual gaps are genuine conceptual absences rather than naming differences.

---

## Per-Domain Analysis of Residual Unmatched Entities

The following entities remain unmatched after **both** AML and MNLI. These represent either (a) concepts that exist in the smaller model but have no counterpart in the larger model, or (b) concepts expressed at a level of granularity that neither lexical nor entailment-based matching can bridge.

---

### Automobile

**V2 → V1** (53.6 % combined, 25 unmatched)

AML coverage on this pair is the lowest in the dataset. V2 presents a component-level decomposition (individual engine sub-parts, individual chassis members) that V1 does not model at the same granularity. MNLI recovered 0 additional matches, confirming these are genuine absences in V1.

> Vehicle, TimingChain, IntakeManifold, ThrottleBody, AirFilter, Turbocharger, Intercooler, OxygenSensor, FuelRail, OilPump, OilFilter, OilPan, Flywheel, UniversalJoint, HalfShaft, CVJoint, IgnitionCoil, ECU, Sensor, SwayBar, BrakeRotor, ABSModule, Frame, BodyPanel, GlassPanel

**V1 → V3** (71.4 % combined, 21 unmatched)

MNLI recovered 7 matches here, the highest count in the dataset, reflecting V3's functional renaming of V1 subsystems. Residual unmapped entities are sub-component references in V1 (individual cylinders, cam lobes, cooling sub-parts) that V3 aggregates into system-level abstractions.

> Engine, CylinderBlock, Cylinder, CylinderHead, CamLobe, Transmission, TorqueConverter, Driveshaft, Axle, ShockAbsorber, SteeringSystem, PowerSteeringUnit, BrakePedal, BrakeDisc, FuelLine, CoolingSystem, CoolantReservoir, Windshield, SideWindow, Headlight, Taillight

**V2 → V3** (69.1 % combined, 10 unmatched)

V3 groups powertrain components into high-level system entities; V2's fine-grained decomposition has no V3 counterpart. Zero MNLI recoveries confirm these are structural gaps.

> EngineBlock, Cylinder, Transmission, GearSet, TorqueConverter, ClutchAssembly, Driveshaft, Headlight, Frame, GlassPanel

---

### Coffee

**V2 → V1** (81.0 % combined, 7 unmatched)

Coffee models achieve the best coverage overall. Residual unmapped items are peripheral equipment (TDS meters, cold brew accessories) that V1 excludes from its scope.

> StandardKettle, Thermoblock, Solenoid, FlowMeter, MetalPourOverFilter, ColdBrewContainer, TDSMeter

**V3 → V1** (84.4 % combined, 12 unmatched)

V3 models brew methods as first-class domain objects (`EspressoBrewingDomain`, `ManualBrewingDomain`) which V1 does not represent as entities. MNLI recovered 6 matches on this pair.

> StandardKettle, MineralAdditionKit, EspressoBrewingDomain, Solenoid, FlowMeter, ManualBrewingDomain, MetalPourOverFilter, ColdBrewMaker, CappuccinoCup, CupWarmer, DescalingSolution, TDSMeter

**V2 → V3** (81.9 % combined, 3 unmatched)

Minimal residual gap. Only specialised brewing components without V3 counterparts remain.

> Thermoblock, PreInfusionChamber, ColdBrewContainer

---

### Homebrewing

**V2 → V1** (75.0 % combined, 10 unmatched)

V2 includes specific heating equipment (propane burner, induction cooktop) and vessel fittings that V1 abstracts away. Zero MNLI recovery on this pair.

> ManifoldScreen, PropaneBurner, InductionCooktop, Stopper, ThreeWayValve, HeatingWrap, KegCoupler, GasManifold, BottleWasher, BrewStand

**V1 → V3** (77.5 % combined, 13 unmatched)

V3 organises homebrewing as systems (`MillingSystem`, `MashingSystem`, `ConditioningSystem`, `PackagingSystem`) while V1 names individual vessels and tubes. MNLI recovered 4 matches.

> MillingSystem, GrainStorageContainer, MashingSystem, BlowoffTube, ConditioningSystem, BrighteningTank, BeerFilter, CarbonationEquipment, PackagingSystem, CanningEquipment, MeasurementSystem, WaterSource, WaterFilter

**V2 → V3** (79.4 % combined, 12 unmatched)

Both V2 component-level items and V3 process-system items remain unmatched, confirming a genuine vocabulary divergence between the two model styles.

> GrainContainer, ManifoldScreen, PropaneBurner, InductionCooktop, Stopper, ThreeWayValve, HeatingWrap, BlowoffTube, KegCoupler, CarbonationCap, BeerFilter, WaterFilter

---

### Hospital

**V2 → V1** (84.4 % combined, 6 unmatched)

The highest AML baseline in the Hospital domain. Remaining unmatched items are physical spaces (bays, suites) and clinical roles that V1 omits.

> EmergencyBay, RecoveryBay, ImagingSuite, PharmacySpace, Surgeon, Anesthesiologist

**V3 → V1** (82.1 % combined, 27 unmatched)

V3 introduces a layered architectural model (diagnostic layer, clinical technology layer, support services layer, building infrastructure layer) with no counterparts in V1's entity vocabulary. The 27 unmatched entities reflect V3's substantially broader scope. MNLI recovered 2 matches.

> CentralMonitorDisplay, DiagnosticsLayer, ImagingSuite, FluoroscopyUnit, SpecimenStorage, CardiologyLab, PulmonaryLab, BodyPlethysmograph, PathologyLab, Microtome, TissueProcessor, UltrasonicCleaner, Surgeon, Anesthesiologist, EmergencyServicesLayer, CrashCart, PharmacyLayer, BuildingInfrastructureLayer, ClinicalTechnologyLayer, SupportServicesLayer, TransportService, Wheelchair, Stretcher, DietaryService, Kitchen, Cafeteria, HousekeepingService

**V2 → V3** (68.1 % combined, 5 unmatched)

V2 models physical clinical spaces (bays, suites) that V3 either omits or subsumes into layer-level abstractions. Zero MNLI recovery.

> EmergencyBay, PreOpBay, RecoveryBay, PharmacySpace, IncubatorUnit

---

### SmartHome

**V2 → V1** (62.4 % combined, 5 unmatched)

Low AML baseline reflects vocabulary divergence between V2's networked infrastructure perspective (NAS devices, inverters, environmental sensors) and V1's appliance-focused vocabulary. Zero MNLI recovery.

> Floor, AirQualitySensor, OccupancySensor, SolarInverter, NASDevice

**V3 → V1** (72.6 % combined, 21 unmatched)

V3 adopts a layered smart-home architecture (PhysicalInfrastructureLayer, SensingLayer, ActuationLayer, EnergyLayer) not present in V1. MNLI recovered 3 matches. The large residual count reflects a fundamental structural difference: V3 is a reference architecture, V1 is a device catalogue.

> PhysicalInfrastructureLayer, Building, Floor, OutdoorArea, SensingLayer, OccupancySensor, AirQualitySensor, PowerMonitor, NoiseSensor, WeatherStation, ActuationLayer, SmartRelay, MotorizedBlinds, SmartFan, NASDevice, EnergyLayer, SolarArray, SolarInverter, SmartLightFixture, SecurityPanel, SmartProjector

**V2 → V3** (81.0 % combined, 8 unmatched)

High coverage with one MNLI recovery. Residual items are consumer appliances (smart oven, smart dryer, robot vacuum) that V3's infrastructure-oriented model does not enumerate as entities.

> Home, SmartBulb, LEDLightStrip, SmartOven, SmartRefrigerator, SmartDryer, RobotVacuum, SmartWindowCovering

---

### University

**V2 → V1** (84.8 % combined, 7 unmatched)

Highest AML coverage in the dataset. Residual gaps are administrative roles (teaching assistant, advisor) and physical/AV infrastructure details absent from V1.

> TeachingAssistant, Advisor, LectureHall, VideoConferenceUnit, LectureCapture, ParkingPermit, UtilityMeter

**V3 → V1** (71.0 % combined, 29 unmatched)

V3 adopts a domain-decomposed architecture (AcademicDomain, StudentLifeDomain, ResearchDomain, OperationsDomain) with nested subdomains. These meta-level organisational concepts have no equivalents in V1's flat entity list. MNLI recovered 2 matches.

> AcademicDomain, Concentration, TeachingAssistant, CampusPhysicalDomain, InstructionalSpace, AVEquipment, VideoConferenceUnit, LectureCaptureSystem, OutdoorSpace, VPNGateway, SoftwarePlatform, StudentLifeDomain, HousingSystem, CommonArea, LaundryRoom, DiningSystem, Cafe, HealthServices, AthleticsSubdomain, ResearchDomain, OperationsDomain, TransportSystem, ParkingSystem, ParkingPermit, UtilitySystem, UtilityMeter, Generator, SolarInstallation, HRSystem

**V2 → V3** (86.6 % combined, 4 unmatched)

Best combined coverage in the entire study. Residual items are specific physical spaces that V3 abstracts into generic `InstructionalSpace`.

> Advisor, Classroom, LectureHall, Laboratory

---

## Interpretation and Conclusions

### Pattern 1 — Abstraction Gap

The most consistent source of residual unmatched entities across all domains is the **abstraction gap** between model versions. V1 models tend to enumerate physical components; V3 models group these into system-level or domain-level abstractions. Neither lexical matching nor entailment recovers these correspondences because they are genuinely many-to-one: one V3 system entity corresponds to multiple V1 component entities. This is a structural alignment problem beyond the scope of entity-to-entity matching.

### Pattern 2 — Scope Differences

Several unmatched entities reflect genuine differences in modelling **scope** rather than naming. V3 Hospital introduces `HousekeepingService`, `DietaryService`, and `TransportService`; these are outside V1's clinical focus. V2 Automobile includes `ECU`, `ABSModule`, and `OxygenSensor` as first-class entities; V1's higher-level model considers these implementation details. No matching strategy should be expected to recover these: the models are making different ontological commitments.

### Pattern 3 — MNLI Effectiveness

MNLI is most effective on V1↔V3 pairs where both models cover the same scope but use different vocabulary registers. The Automobile V1→V3 pair (7 recoveries) and Coffee V3→V1 pair (6 recoveries) demonstrate the strongest benefit. V2↔V3 pairs yield zero MNLI recoveries in every domain, confirming that V2 and V3 models diverge in scope rather than terminology.

### Recommendation for Future Work

Three directions merit investigation:

1. **Group matching**: For abstraction gaps, allow one entity in the smaller model to match a set of entities in the larger model. This requires relaxing the 1:1 assignment constraint.
2. **Hierarchy-aware matching**: Exploit the implicit composition hierarchy derivable from attribute-type references in the JSON to give AML structural context without introducing OWL subclass constraints that distort property profiles.
3. **Threshold calibration per domain**: Domains with dense attribute vocabulary (Hospital, University) may benefit from a lower MNLI threshold; sparse domains (Coffee, SmartHome) from a higher one to avoid false positives.

---

*Pipeline implementation:* `ontology_matching/generate_alignment_reports.py` (Stage 1 — AML), `ontology_matching/mnli_matcher.py` (Stage 2 — MNLI equivalence), `ontology_matching/subsumption_matcher.py` (Stage 3 — asymmetric subsumption), `ontology_matching/list_unmapped.py` (residual analysis).
*Output artefacts:* `outputs/alignment_summary.csv`, `outputs/alignment_summary_mnli.csv`, `outputs/alignment_summary_subsumption.csv`, `outputs/unmapped_summary.csv`, `outputs/mnli/`, `outputs/subsumption/`.
