from owlready2 import *


def build_icd_metamodel(
    base_iri: str = "http://example.org/icd_metamodel.owl#",
    output_path: str = "icd_metamodel.owl",
):
    onto = get_ontology(base_iri)

    with onto:
        # -------------------------
        # Core classes
        # -------------------------
        class Message(Thing):
            pass

        class ConceptualEntity(Thing):
            pass

        class Field(Thing):
            pass

        class Observable(Thing):
            pass

        class Measurement(Thing):
            pass

        class Unit(Thing):
            pass

        class PlatformType(Thing):
            pass

        # -------------------------
        # Range classes
        # -------------------------
        class RangeSpec(Thing):
            pass

        class DocumentReferenceRange(RangeSpec):
            pass

        class IntervalRange(RangeSpec):
            pass

        class EnumerationRange(RangeSpec):
            pass

        # -------------------------
        # Object properties
        # -------------------------
        class hasEntity(ObjectProperty):
            domain = [Message]
            range = [ConceptualEntity]

        class hasSubEntity(ObjectProperty):
            domain = [ConceptualEntity]
            range = [ConceptualEntity]

        class hasField(ObjectProperty):
            domain = [ConceptualEntity]
            range = [Field]

        class hasObservable(ObjectProperty):
            domain = [Field]
            range = [Observable]

        class hasMeasurement(ObjectProperty):
            domain = [Field]
            range = [Measurement]

        class hasUnit(ObjectProperty):
            domain = [Field]
            range = [Unit]

        class hasPlatformType(ObjectProperty):
            domain = [Field]
            range = [PlatformType]

        class hasRangeSpec(ObjectProperty):
            domain = [Field]
            range = [RangeSpec]

        # -------------------------
        # Data properties
        # -------------------------
        class hasMessageId(DataProperty, FunctionalProperty):
            domain = [Message]
            range = [str]

        class hasMessageName(DataProperty, FunctionalProperty):
            domain = [Message]
            range = [str]

        class hasEntityId(DataProperty, FunctionalProperty):
            domain = [ConceptualEntity]
            range = [str]

        class hasEntityName(DataProperty, FunctionalProperty):
            domain = [ConceptualEntity]
            range = [str]

        class hasUsageContext(DataProperty):
            domain = [ConceptualEntity]
            range = [str]

        class hasFieldId(DataProperty, FunctionalProperty):
            domain = [Field]
            range = [str]

        class hasFieldName(DataProperty, FunctionalProperty):
            domain = [Field]
            range = [str]

        class referencesEntityId(DataProperty, FunctionalProperty):
            domain = [Field]
            range = [str]

        class hasObservableId(DataProperty, FunctionalProperty):
            domain = [Observable]
            range = [str]

        class hasMeasurementId(DataProperty, FunctionalProperty):
            domain = [Measurement]
            range = [str]

        class hasUnitName(DataProperty, FunctionalProperty):
            domain = [Unit]
            range = [str]

        class hasPlatformTypeName(DataProperty, FunctionalProperty):
            domain = [PlatformType]
            range = [str]

        class isRequired(DataProperty, FunctionalProperty):
            domain = [Field]
            range = [bool]

        # -------------------------
        # Generic range text
        # -------------------------
        class hasRawRangeText(DataProperty, FunctionalProperty):
            domain = [RangeSpec]
            range = [str]

        # -------------------------
        # Document reference range
        # -------------------------
        class hasReferenceText(DataProperty, FunctionalProperty):
            domain = [DocumentReferenceRange]
            range = [str]

        # -------------------------
        # Interval range
        # -------------------------
        class hasIntervalVariable(DataProperty, FunctionalProperty):
            domain = [IntervalRange]
            range = [str]

        class hasLowerBoundText(DataProperty, FunctionalProperty):
            domain = [IntervalRange]
            range = [str]

        class hasUpperBoundText(DataProperty, FunctionalProperty):
            domain = [IntervalRange]
            range = [str]

        class isLowerInclusive(DataProperty, FunctionalProperty):
            domain = [IntervalRange]
            range = [bool]

        class isUpperInclusive(DataProperty, FunctionalProperty):
            domain = [IntervalRange]
            range = [bool]

        class hasIntervalQualifierText(DataProperty, FunctionalProperty):
            domain = [IntervalRange]
            range = [str]

        # -------------------------
        # Enumeration range
        # -------------------------
        class hasAllowedLiteral(DataProperty):
            domain = [EnumerationRange]
            range = [str]

        # Optional labels
        Message.label = ["Message"]
        ConceptualEntity.label = ["ConceptualEntity"]
        Field.label = ["Field"]
        Observable.label = ["Observable"]
        Measurement.label = ["Measurement"]
        Unit.label = ["Unit"]
        PlatformType.label = ["PlatformType"]
        RangeSpec.label = ["RangeSpec"]
        DocumentReferenceRange.label = ["DocumentReferenceRange"]
        IntervalRange.label = ["IntervalRange"]
        EnumerationRange.label = ["EnumerationRange"]

    onto.save(file=output_path, format="rdfxml")
    return onto