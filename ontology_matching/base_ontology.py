"""
base_ontology.py
----------------
Defines the conceptual-model metamodel OWL ontology.

The metamodel captures the shared structure found across all JSON files in
inputs/CONceptual_ExtractionCategory_Examples/:

    ConceptualModel   – a named model with a set of entities and associations
        hasEntity         → Entity
        hasAssociation    → Association
        hasObservableType → ObservableType   (Type-A models only)

    Entity            – a named concept in the model
        hasAttribute      → Attribute

    Association       – a named relationship between entities
        hasParticipant    → Entity
        hasAttribute      → Attribute

    Attribute         – a named property belonging to an Entity or Association
        observableTypeRef → ObservableType   (when type is a physical quantity)
        entityTypeRef     → Entity           (when type is a composition ref)

    ObservableType    – a named physical/measurable quantity (Temperature, Mass …)

Usage
-----
    from base_ontology import build_concept_metamodel
    onto = build_concept_metamodel()       # saves outputs/concept_metamodel.owl
"""

from __future__ import annotations

import os
from owlready2 import (
    get_ontology,
    Thing,
    ObjectProperty,
    DataProperty,
    FunctionalProperty,
    AllDisjoint,
)

BASE_IRI = "http://example.org/concept_metamodel.owl#"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "concept_metamodel.owl")


def build_concept_metamodel(
    base_iri: str = BASE_IRI,
    output_path: str = OUTPUT_PATH,
) -> object:
    """Build and save the conceptual-model metamodel OWL ontology."""

    onto = get_ontology(base_iri)

    with onto:
        # ------------------------------------------------------------------ #
        # Core metaclasses                                                    #
        # ------------------------------------------------------------------ #

        class ConceptualModel(Thing):
            """A named conceptual model containing entities and associations."""

        class Entity(Thing):
            """A named concept (class) within a conceptual model."""

        class Association(Thing):
            """A named relationship between two or more entities."""

        class Attribute(Thing):
            """A named property of an entity or association,
            typed by an ObservableType or referencing another Entity."""

        class ObservableType(Thing):
            """A named physical or categorical quantity used as an attribute type
            (e.g. Temperature, Mass, Identifier, Kind)."""

        # ------------------------------------------------------------------ #
        # Object properties                                                   #
        # ------------------------------------------------------------------ #

        class hasEntity(ObjectProperty):
            """A ConceptualModel contains one or more Entities."""
            domain = [ConceptualModel]
            range  = [Entity]

        class hasAssociation(ObjectProperty):
            """A ConceptualModel contains zero or more Associations."""
            domain = [ConceptualModel]
            range  = [Association]

        class hasObservableType(ObjectProperty):
            """A ConceptualModel declares a set of ObservableTypes it uses."""
            domain = [ConceptualModel]
            range  = [ObservableType]

        class hasAttribute(ObjectProperty):
            """An Entity or Association owns zero or more Attributes."""
            domain = [Entity | Association]
            range  = [Attribute]

        class hasParticipant(ObjectProperty):
            """An Association involves one or more Entities as participants."""
            domain = [Association]
            range  = [Entity]

        class observableTypeRef(ObjectProperty, FunctionalProperty):
            """An Attribute whose type is an ObservableType points here."""
            domain = [Attribute]
            range  = [ObservableType]

        class entityTypeRef(ObjectProperty, FunctionalProperty):
            """An Attribute whose type is another Entity (composition) points here."""
            domain = [Attribute]
            range  = [Entity]

        # ------------------------------------------------------------------ #
        # Data properties                                                     #
        # ------------------------------------------------------------------ #

        class modelName(DataProperty, FunctionalProperty):
            domain = [ConceptualModel]
            range  = [str]

        class modelDescription(DataProperty, FunctionalProperty):
            """Free-text description of the model (Type-A models)."""
            domain = [ConceptualModel]
            range  = [str]

        class topicDomain(DataProperty, FunctionalProperty):
            """High-level subject area (Type-B variation models)."""
            domain = [ConceptualModel]
            range  = [str]

        class methodBasis(DataProperty, FunctionalProperty):
            """Modelling approach description (Type-B variation models)."""
            domain = [ConceptualModel]
            range  = [str]

        class elementName(DataProperty, FunctionalProperty):
            """Local name of an Entity, Association, Attribute, or ObservableType."""
            domain = [Entity | Association | Attribute | ObservableType]
            range  = [str]

        class typeName(DataProperty, FunctionalProperty):
            """Raw type string from the source JSON (observable or entity name)."""
            domain = [Attribute]
            range  = [str]

        # Disjointness between the five core metaclasses
        AllDisjoint([ConceptualModel, Entity, Association, Attribute, ObservableType])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onto.save(file=output_path, format="rdfxml")
    print(f"[metamodel] saved -> {output_path}")
    return onto
