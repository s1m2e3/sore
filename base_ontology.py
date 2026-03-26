from owlready2 import *

def build_uddl_metamodel(
    base_iri: str = "http://example.org/uddl_metamodel.owl#",
    output_path: str = "./outputs/uddl_metamodel.owl",
):
    onto = get_ontology(base_iri)

    with onto:
        # ==============t===========================================
        # Top-level model structure
        # =========================================================
        class DataModel(Thing):
            pass

        class ConceptualDataModel(DataModel):
            pass

        class LogicalDataModel(DataModel):
            pass

        class PlatformDataModel(DataModel):
            pass

        # =========================================================
        # Common abstract types
        # =========================================================
        class UDDL_Element(Thing):
            pass

        class NamedElement(UDDL_Element):
            pass

        class Characteristic(NamedElement):
            pass

        class ComposableElement(NamedElement):
            pass

        class AssociationLike(ComposableElement):
            pass

        # =========================================================
        # Conceptual layer
        # =========================================================
        class ConceptualElement(NamedElement):
            pass

        class BasisEntity(ConceptualElement):
            pass

        class ConceptualEntity(ConceptualElement, ComposableElement):
            pass

        class ConceptualObservable(ConceptualElement):
            pass

        class ConceptualComposition(Characteristic):
            pass

        class ConceptualParticipant(Characteristic):
            pass

        class ConceptualAssociation(AssociationLike, ConceptualEntity):
            pass

        class ConceptualView(ConceptualElement):
            pass

        class ConceptualQuery(ConceptualView):
            pass

        class ConceptualCompositeQuery(ConceptualQuery):
            pass

        # =========================================================
        # Logical layer
        # =========================================================
        class LogicalElement(NamedElement):
            pass

        class LogicalEntity(LogicalElement, ComposableElement):
            pass

        class LogicalAssociation(AssociationLike, LogicalEntity):
            pass

        class LogicalComposition(Characteristic):
            pass

        class LogicalParticipant(Characteristic):
            pass

        class LogicalMeasurement(LogicalElement):
            pass

        class LogicalValueType(LogicalElement):
            pass

        class LogicalUnit(LogicalElement):
            pass

        class LogicalValueTypeUnit(LogicalElement):
            pass

        class LogicalCoordinateSystem(LogicalElement):
            pass

        class LogicalCoordinateAxis(LogicalElement):
            pass

        class LogicalMeasurementSystem(LogicalElement):
            pass

        class LogicalMeasurementAxis(LogicalElement):
            pass

        class LogicalReferencePoint(LogicalElement):
            pass

        class LogicalLandmark(LogicalElement):
            pass

        class LogicalConstraint(LogicalElement):
            pass

        class LogicalConversion(LogicalElement):
            pass

        class LogicalView(LogicalElement):
            pass

        class LogicalQuery(LogicalView):
            pass

        class LogicalCompositeQuery(LogicalQuery):
            pass

        # =========================================================
        # Platform layer
        # =========================================================
        class PlatformElement(NamedElement):
            pass

        class PlatformEntity(PlatformElement, ComposableElement):
            pass

        class PlatformAssociation(AssociationLike, PlatformEntity):
            pass

        class PlatformComposition(Characteristic):
            pass

        class PlatformParticipant(Characteristic):
            pass

        class PlatformDataType(PlatformElement):
            pass

        class PlatformPrimitive(PlatformDataType):
            pass

        class PlatformStruct(PlatformDataType):
            pass

        class PlatformView(PlatformElement):
            pass

        class PlatformQuery(PlatformView):
            pass

        class PlatformCompositeQuery(PlatformQuery):
            pass

        # =========================================================
        # Supporting classes
        # =========================================================
        class Cardinality(Thing):
            pass

        class PathNode(Thing):
            pass

        class Constraint(Thing):
            pass

        class RangeConstraint(Constraint):
            pass

        class EnumerationConstraint(Constraint):
            pass

        class RegexConstraint(Constraint):
            pass

        class LengthConstraint(Constraint):
            pass

        # =========================================================
        # Object properties: model containment
        # =========================================================
        class hasConceptualModel(ObjectProperty):
            domain = [DataModel]
            range = [ConceptualDataModel]

        class hasLogicalModel(ObjectProperty):
            domain = [DataModel]
            range = [LogicalDataModel]

        class hasPlatformModel(ObjectProperty):
            domain = [DataModel]
            range = [PlatformDataModel]

        class hasElement(ObjectProperty):
            domain = [DataModel]
            range = [NamedElement]

        # =========================================================
        # Object properties: specialization / realization
        # =========================================================
        class specializes(ObjectProperty):
            domain = [NamedElement]
            range = [NamedElement]

        # Generic realization anchor — all typed sub-properties below inherit from this.
        class realizes(ObjectProperty):
            domain = [NamedElement]
            range = [NamedElement]

        # Typed realization sub-properties.
        # These form the navigable semantic chain:
        #   PlatformComposition --platformCompositionRealizes-->
        #   LogicalMeasurement  --measurementRealizes-->
        #   ConceptualObservable
        # which is the backbone of the human-vs-LLM comparison.

        class logicalEntityRealizes(realizes, FunctionalProperty):
            """LogicalEntity realizes exactly one ConceptualEntity."""
            domain = [LogicalEntity]
            range = [ConceptualEntity]

        class measurementRealizes(realizes, FunctionalProperty):
            """LogicalMeasurement realizes exactly one ConceptualObservable."""
            domain = [LogicalMeasurement]
            range = [ConceptualObservable]

        class platformEntityRealizes(realizes, FunctionalProperty):
            """PlatformEntity realizes exactly one LogicalEntity."""
            domain = [PlatformEntity]
            range = [LogicalEntity]

        class platformCompositionRealizes(realizes, FunctionalProperty):
            """PlatformComposition (field) realizes exactly one LogicalMeasurement.
            This is the critical traceability link that connects a platform-level
            field name/type back to its semantic observable and measurement system."""
            domain = [PlatformComposition]
            range = [LogicalMeasurement]

        # =========================================================
        # Conceptual relationships
        # =========================================================
        class hasBasisEntity(ObjectProperty):
            domain = [ConceptualDataModel]
            range = [BasisEntity]

        class hasConceptualEntity(ObjectProperty):
            domain = [ConceptualDataModel]
            range = [ConceptualEntity]

        class hasConceptualObservable(ObjectProperty):
            domain = [ConceptualDataModel]
            range = [ConceptualObservable]

        class hasConceptualAssociation(ObjectProperty):
            domain = [ConceptualDataModel]
            range = [ConceptualAssociation]

        class hasConceptualView(ObjectProperty):
            domain = [ConceptualDataModel]
            range = [ConceptualView]

        # Generic composition anchor — layer-specific sub-properties below.
        class hasComposition(ObjectProperty):
            domain = [ComposableElement]
            range = [Characteristic]

        class hasConceptualComposition(hasComposition):
            """Typed sub-property: ConceptualEntity → ConceptualComposition."""
            domain = [ConceptualEntity]
            range = [ConceptualComposition]

        class hasLogicalComposition(hasComposition):
            """Typed sub-property: LogicalEntity → LogicalComposition."""
            domain = [LogicalEntity]
            range = [LogicalComposition]

        class hasPlatformComposition(hasComposition):
            """Typed sub-property: PlatformEntity → PlatformComposition."""
            domain = [PlatformEntity]
            range = [PlatformComposition]

        class hasParticipant(ObjectProperty):
            domain = [AssociationLike]
            range = [ConceptualParticipant]

        class compositionTarget(ObjectProperty):
            # Range includes ComposableElement (ConceptualEntity, ConceptualAssociation)
            # AND ConceptualObservable, because in UDDL a composition like
            # `Speed airspeed [1:1]` targets an observable type directly.
            domain = [ConceptualComposition]
            range = [ComposableElement, ConceptualObservable]

        class participantTarget(ObjectProperty):
            domain = [ConceptualParticipant]
            range = [ConceptualEntity]

        class appliesTo(ObjectProperty):
            domain = [ConceptualObservable]
            range = [ConceptualEntity]

        # =========================================================
        # Logical relationships
        # =========================================================
        class hasLogicalEntity(ObjectProperty):
            domain = [LogicalDataModel]
            range = [LogicalEntity]

        class hasLogicalAssociation(ObjectProperty):
            domain = [LogicalDataModel]
            range = [LogicalAssociation]

        class hasLogicalMeasurement(ObjectProperty):
            domain = [LogicalDataModel]
            range = [LogicalMeasurement]

        class hasLogicalValueType(ObjectProperty):
            domain = [LogicalDataModel]
            range = [LogicalValueType]

        class hasLogicalUnit(ObjectProperty):
            domain = [LogicalDataModel]
            range = [LogicalUnit]

        class hasLogicalValueTypeUnit(ObjectProperty):
            domain = [LogicalDataModel]
            range = [LogicalValueTypeUnit]

        class hasCoordinateSystem(ObjectProperty):
            domain = [LogicalDataModel]
            range = [LogicalCoordinateSystem]

        class hasMeasurementSystem(ObjectProperty):
            domain = [LogicalDataModel]
            range = [LogicalMeasurementSystem]

        class hasConstraint(ObjectProperty):
            domain = [LogicalElement]
            range = [Constraint]

        class measurementValueType(ObjectProperty):
            domain = [LogicalMeasurement]
            range = [LogicalValueType]

        class measurementUnit(ObjectProperty):
            domain = [LogicalMeasurement]
            range = [LogicalUnit]

        class measurementSystem(ObjectProperty):
            domain = [LogicalMeasurement]
            range = [LogicalMeasurementSystem]

        class hasAxis(ObjectProperty):
            domain = [LogicalCoordinateSystem, LogicalMeasurementSystem]
            range = [LogicalCoordinateAxis, LogicalMeasurementAxis]

        class defaultUnit(ObjectProperty):
            domain = [LogicalMeasurementAxis]
            range = [LogicalValueTypeUnit]

        class coordinateReferencePoint(ObjectProperty):
            domain = [LogicalCoordinateSystem]
            range = [LogicalReferencePoint]

        class hasConversion(ObjectProperty):
            domain = [LogicalMeasurementSystem]
            range = [LogicalConversion]

        class logicalCompositionTarget(ObjectProperty):
            domain = [LogicalComposition]
            range = [LogicalEntity, LogicalAssociation]

        class logicalParticipantTarget(ObjectProperty):
            domain = [LogicalParticipant]
            range = [LogicalEntity]

        # =========================================================
        # Platform relationships
        # =========================================================
        class hasPlatformEntity(ObjectProperty):
            domain = [PlatformDataModel]
            range = [PlatformEntity]

        class hasPlatformAssociation(ObjectProperty):
            domain = [PlatformDataModel]
            range = [PlatformAssociation]

        class hasPlatformDataType(ObjectProperty):
            domain = [PlatformDataModel]
            range = [PlatformDataType]

        class hasPlatformView(ObjectProperty):
            domain = [PlatformDataModel]
            range = [PlatformView]

        class platformCompositionTarget(ObjectProperty):
            domain = [PlatformComposition]
            range = [PlatformDataType]

        class platformParticipantTarget(ObjectProperty):
            domain = [PlatformParticipant]
            range = [PlatformEntity]

        class hasFieldType(ObjectProperty):
            """Alias for platformCompositionTarget — kept for backward compatibility."""
            domain = [PlatformComposition]
            range = [PlatformDataType]

        # =========================================================
        # Cardinality / path support
        # =========================================================
        class hasCardinality(ObjectProperty):
            domain = [Characteristic]
            range = [Cardinality]

        class hasPathNode(ObjectProperty):
            domain = [ConceptualParticipant, ConceptualView, LogicalView, PlatformView]
            range = [PathNode]

        # =========================================================
        # Data properties
        # =========================================================
        class hasName(DataProperty, FunctionalProperty):
            domain = [NamedElement]
            range = [str]

        class hasDescription(DataProperty):
            domain = [NamedElement]
            range = [str]

        class hasIdentifier(DataProperty, FunctionalProperty):
            domain = [NamedElement]
            range = [str]

        class hasRoleName(DataProperty, FunctionalProperty):
            domain = [Characteristic]
            range = [str]

        class lowerBound(DataProperty, FunctionalProperty):
            domain = [Cardinality]
            range = [int]

        class upperBound(DataProperty, FunctionalProperty):
            domain = [Cardinality]
            range = [int]

        class isOrdered(DataProperty, FunctionalProperty):
            domain = [Characteristic]
            range = [bool]

        class isUnique(DataProperty, FunctionalProperty):
            domain = [Characteristic]
            range = [bool]

        class literalValue(DataProperty, FunctionalProperty):
            domain = [LogicalMeasurement]
            range = [str]

        class numericValue(DataProperty, FunctionalProperty):
            domain = [LogicalMeasurement]
            range = [float]

        class primitiveDataType(DataProperty, FunctionalProperty):
            """The UDDL primitive value type used by this measurement.
            Valid values mirror the UDDL DSL keywords:
              'real', 'real+' (NonNegativeReal), 'int', 'nat', 'bool', 'char', 'str'
            This is distinct from valueTypeName on LogicalValueType (which holds
            the dimensional category like 'Length', 'Speed', 'Temperature')."""
            domain = [LogicalMeasurement]
            range = [str]

        class symbol(DataProperty, FunctionalProperty):
            domain = [LogicalUnit]
            range = [str]

        class valueTypeName(DataProperty, FunctionalProperty):
            """Human-readable name of a LogicalValueType (e.g. 'Length', 'Speed', 'Temperature').
            Used as the primary semantic label during instance matching."""
            domain = [LogicalValueType]
            range = [str]

        class measurementSystemName(DataProperty, FunctionalProperty):
            """Human-readable name of a LogicalMeasurementSystem
            (e.g. 'AviationAltitudeSystem', 'SIKinematicsSystem').
            Used to detect measurement system mismatches in LLM-generated UDDL."""
            domain = [LogicalMeasurementSystem]
            range = [str]

        # ------------------------------------------------------------------
        # Provenance — tag each DataModel as human-authored or LLM-generated,
        # and record which ICD it was derived from.  These are the core labels
        # the comparison framework uses to distinguish the two graphs.
        # ------------------------------------------------------------------
        class hasSource(DataProperty, FunctionalProperty):
            """Origin of this DataModel: 'human' or 'llm'.
            Set at parse time so every SPARQL/instance-matching query can
            filter to the correct graph without graph-name indirection."""
            domain = [DataModel]
            range = [str]

        class derivedFromICD(DataProperty, FunctionalProperty):
            """Identifier of the ICD this DataModel was parsed from.
            Must match between the human and LLM graphs for a valid comparison pair."""
            domain = [DataModel]
            range = [str]

        class generationModel(DataProperty, FunctionalProperty):
            """LLM model name used to generate this DataModel, e.g. 'gpt-4o', 'claude-3-5-sonnet'.
            Empty for human-authored models."""
            domain = [DataModel]
            range = [str]

        class primitiveName(DataProperty, FunctionalProperty):
            domain = [PlatformPrimitive]
            range = [str]

        class structName(DataProperty, FunctionalProperty):
            domain = [PlatformStruct]
            range = [str]

        class pathExpression(DataProperty, FunctionalProperty):
            domain = [PathNode]
            range = [str]

        class regexPattern(DataProperty, FunctionalProperty):
            domain = [RegexConstraint]
            range = [str]

        class minValue(DataProperty, FunctionalProperty):
            domain = [RangeConstraint]
            range = [float]

        class maxValue(DataProperty, FunctionalProperty):
            domain = [RangeConstraint]
            range = [float]

        class allowedLiteral(DataProperty):
            domain = [EnumerationConstraint]
            range = [str]

        class minLength(DataProperty, FunctionalProperty):
            domain = [LengthConstraint]
            range = [int]

        class maxLength(DataProperty, FunctionalProperty):
            domain = [LengthConstraint]
            range = [int]

        # =========================================================
        # Labels
        # =========================================================
        DataModel.label = ["DataModel"]
        ConceptualDataModel.label = ["ConceptualDataModel"]
        LogicalDataModel.label = ["LogicalDataModel"]
        PlatformDataModel.label = ["PlatformDataModel"]

        ConceptualEntity.label = ["ConceptualEntity"]
        BasisEntity.label = ["BasisEntity"]
        ConceptualObservable.label = ["ConceptualObservable"]
        ConceptualComposition.label = ["ConceptualComposition"]
        ConceptualParticipant.label = ["ConceptualParticipant"]
        ConceptualAssociation.label = ["ConceptualAssociation"]

        LogicalEntity.label = ["LogicalEntity"]
        LogicalAssociation.label = ["LogicalAssociation"]
        LogicalMeasurement.label = ["LogicalMeasurement"]
        LogicalValueType.label = ["LogicalValueType"]
        LogicalUnit.label = ["LogicalUnit"]
        LogicalValueTypeUnit.label = ["LogicalValueTypeUnit"]
        LogicalCoordinateSystem.label = ["LogicalCoordinateSystem"]
        LogicalMeasurementSystem.label = ["LogicalMeasurementSystem"]
        LogicalConstraint.label = ["LogicalConstraint"]

        PlatformEntity.label = ["PlatformEntity"]
        PlatformAssociation.label = ["PlatformAssociation"]
        PlatformDataType.label = ["PlatformDataType"]
        PlatformPrimitive.label = ["PlatformPrimitive"]
        PlatformStruct.label = ["PlatformStruct"]

        # Typed realization sub-property labels
        logicalEntityRealizes.label = ["logicalEntityRealizes"]
        measurementRealizes.label = ["measurementRealizes"]
        platformEntityRealizes.label = ["platformEntityRealizes"]
        platformCompositionRealizes.label = ["platformCompositionRealizes"]

        # Layer-specific composition sub-property labels
        hasConceptualComposition.label = ["hasConceptualComposition"]
        hasLogicalComposition.label = ["hasLogicalComposition"]
        hasPlatformComposition.label = ["hasPlatformComposition"]

        # Value type / measurement system name labels
        valueTypeName.label = ["valueTypeName"]
        measurementSystemName.label = ["measurementSystemName"]

        # Provenance property labels
        hasSource.label = ["hasSource"]
        derivedFromICD.label = ["derivedFromICD"]
        generationModel.label = ["generationModel"]

    onto.save(file=output_path, format="rdfxml")
    print("saved onto")
    return onto