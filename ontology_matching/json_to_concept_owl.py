"""
json_to_concept_owl.py
----------------------
Converts a conceptual-model JSON (either Type-A or Type-B schema) into an
OWL instance graph that imports the conceptual-model metamodel.

Supported JSON schemas
----------------------
Type A  (v1 / v2 / v3 files)
    keys: modelName, modelDescription, observables, entities, associations
    entity keys: entityName, entityAttributes  (list of {name, type})
    assoc  keys: associationName, associationParticipants, associationAttributes

Type B  (variation files)
    keys: modelName, topicDomain, methodBasis, entities, associations
    entity keys: name, attributes  (list of {name, type})
    assoc  keys: name, participants, attributes

Output
------
One OWL/RDF-XML file per JSON, saved to outputs/<safe_model_name>.owl.
Each file imports the shared metamodel and contains OWL individuals that
represent the model's entities, attributes, associations, and observable types.

Usage
-----
    from json_to_concept_owl import ConceptJsonToOWL
    converter = ConceptJsonToOWL()
    onto = converter.convert(json_data, output_path="outputs/Automobile_v1.owl")
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from owlready2 import get_ontology

METAMODEL_PATH = os.path.join(os.path.dirname(__file__), "outputs", "concept_metamodel.owl")
INSTANCE_BASE  = "http://example.org/concept_instances/"
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "outputs")


def _safe(text: str) -> str:
    """Sanitise a string for use as an OWL local name."""
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _set(individual, prop_class, value) -> None:
    """Set a FunctionalDataProperty value on an individual."""
    setattr(individual, prop_class.python_name, value)


def _append(individual, prop_class, obj) -> None:
    """Append a value to a non-functional ObjectProperty on an individual."""
    getattr(individual, prop_class.python_name).append(obj)


class ConceptJsonToOWL:
    """
    Parse a conceptual-model JSON (Type-A or Type-B) and produce an OWL
    instance graph conforming to the concept_metamodel.owl.

    Parameters
    ----------
    metamodel_path : path to the saved metamodel OWL (concept_metamodel.owl)
    output_dir     : directory where per-JSON OWL files are saved
    """

    def __init__(
        self,
        meta_onto=None,
        metamodel_path: str = METAMODEL_PATH,
        output_dir: str = OUTPUT_DIR,
    ):
        """
        Parameters
        ----------
        meta_onto      : already-built owlready2 ontology object (preferred).
                         If None, the metamodel is loaded from metamodel_path.
        metamodel_path : fallback path used when meta_onto is None.
        output_dir     : directory where per-JSON OWL files are saved.
        """
        self._meta_onto = meta_onto
        self.metamodel_path = os.path.abspath(metamodel_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_metamodel(self):
        """Return the metamodel ontology, loading from disk only if needed."""
        if self._meta_onto is None:
            # Windows-safe: use the owlready2 world that already has the
            # metamodel if build_concept_metamodel() was called first.
            from owlready2 import default_world
            meta_iri = "http://example.org/concept_metamodel.owl#"
            cached = default_world.get_ontology(meta_iri)
            if cached.loaded:
                self._meta_onto = cached
            else:
                # Last resort: load from local path directly via owlready2's
                # Windows-compatible loader
                self._meta_onto = get_ontology(
                    "file:///" + self.metamodel_path.replace("\\", "/")
                ).load()
        return self._meta_onto

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def convert(self, json_data: dict[str, Any], output_path: str | None = None) -> object:
        """
        Convert one JSON model dict to OWL.

        Parameters
        ----------
        json_data    : parsed JSON dict
        output_path  : explicit output path; if None, derived from modelName

        Returns
        -------
        owlready2 ontology object (already saved to disk)
        """
        model_name = json_data.get("modelName", "UnknownModel")
        safe_name  = _safe(model_name)

        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{safe_name}.owl")

        instance_iri = f"{INSTANCE_BASE}{safe_name}/"

        # Ensure metamodel is available in the shared default_world
        meta_onto = self._load_metamodel()

        # Fresh instance ontology in the same default_world as the metamodel.
        # We do NOT add an owl:imports triple here to avoid owlready2 trying
        # to reload the metamodel from disk (which breaks on Windows paths).
        # All metaclasses are already accessible via the shared world.
        inst_onto = get_ontology(instance_iri)

        with inst_onto:
            self._populate(meta_onto, json_data, safe_name)

        inst_onto.save(file=output_path, format="rdfxml")
        print(f"[converter] {model_name} -> {output_path}")
        return inst_onto

    def convert_file(
        self,
        json_path: str,
        output_path: str | None = None,
        output_dir: str | None = None,
    ) -> object:
        """Convenience wrapper: read JSON from disk, then call convert().

        output_dir overrides self.output_dir for this one call (used by
        main.py to save into per-domain subdirectories).
        """
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if output_dir is not None and output_path is None:
            safe_name = _safe(data.get("modelName", "UnknownModel"))
            output_path = os.path.join(output_dir, f"{safe_name}.owl")
        return self.convert(data, output_path=output_path)

    # ------------------------------------------------------------------ #
    # Internal population logic                                            #
    # ------------------------------------------------------------------ #

    def _populate(self, M, data: dict, prefix: str) -> None:
        """Create OWL individuals for all model elements.

        M         – the loaded metamodel ontology (provides classes/properties)
        data      – parsed JSON dict
        prefix    – safe IRI prefix derived from modelName
        """
        model_name = data.get("modelName", "UnknownModel")

        # ---- ConceptualModel individual -------------------------------- #
        model_ind = M.ConceptualModel(f"{prefix}_model")
        model_ind.modelName     = model_name

        desc = data.get("modelDescription")
        if desc:
            model_ind.modelDescription = desc

        topic = data.get("topicDomain")
        if topic:
            model_ind.topicDomain = topic

        method = data.get("methodBasis")
        if method:
            model_ind.methodBasis = method

        # ---- Observable types (Type-A explicit list; Type-B built lazily) #
        obs_individuals: dict[str, object] = {}

        def _get_or_create_obs(obs_name: str) -> object:
            if obs_name in obs_individuals:
                return obs_individuals[obs_name]
            obs_id  = f"{prefix}_obs_{_safe(obs_name)}"
            obs_ind = M.ObservableType(obs_id)
            obs_ind.elementName = obs_name
            model_ind.hasObservableType.append(obs_ind)
            obs_individuals[obs_name] = obs_ind
            return obs_ind

        for obs_name in data.get("observables", []):
            _get_or_create_obs(obs_name)

        # ---- Entities (first pass – create all individuals) ------------ #
        entity_individuals: dict[str, object] = {}

        raw_entities = data.get("entities", [])
        for ent in raw_entities:
            ent_name = ent.get("entityName") or ent.get("name", "UnknownEntity")
            ent_id   = f"{prefix}_ent_{_safe(ent_name)}"
            ent_ind  = M.Entity(ent_id)
            ent_ind.elementName = ent_name
            model_ind.hasEntity.append(ent_ind)
            entity_individuals[ent_name] = ent_ind

        # ---- Entity attributes (second pass – all entities now exist) -- #
        for ent in raw_entities:
            ent_name  = ent.get("entityName") or ent.get("name", "UnknownEntity")
            ent_ind   = entity_individuals[ent_name]
            raw_attrs = ent.get("entityAttributes") or ent.get("attributes", [])

            for attr in raw_attrs:
                attr_name = attr.get("name", "unknownAttr")
                attr_type = attr.get("type", "")
                attr_id   = f"{prefix}_attr_{_safe(ent_name)}_{_safe(attr_name)}"
                attr_ind  = M.Attribute(attr_id)
                attr_ind.elementName = attr_name
                attr_ind.typeName    = attr_type
                ent_ind.hasAttribute.append(attr_ind)

                # Resolve type: observable reference or entity composition
                if attr_type in entity_individuals:
                    attr_ind.entityTypeRef = entity_individuals[attr_type]
                else:
                    attr_ind.observableTypeRef = _get_or_create_obs(attr_type)

        # ---- Associations ---------------------------------------------- #
        raw_assocs = data.get("associations", [])
        for assoc in raw_assocs:
            assoc_name = (
                assoc.get("associationName") or assoc.get("name", "UnknownAssociation")
            )
            assoc_id  = f"{prefix}_assoc_{_safe(assoc_name)}"
            assoc_ind = M.Association(assoc_id)
            assoc_ind.elementName = assoc_name
            model_ind.hasAssociation.append(assoc_ind)

            # Participants
            participants = (
                assoc.get("associationParticipants") or assoc.get("participants", [])
            )
            for p_name in participants:
                if p_name in entity_individuals:
                    assoc_ind.hasParticipant.append(entity_individuals[p_name])

            # Association attributes
            raw_attrs = (
                assoc.get("associationAttributes") or assoc.get("attributes", [])
            )
            for attr in raw_attrs:
                attr_name = attr.get("name", "unknownAttr")
                attr_type = attr.get("type", "")
                attr_id   = f"{prefix}_attr_{_safe(assoc_name)}_{_safe(attr_name)}"
                attr_ind  = M.Attribute(attr_id)
                attr_ind.elementName = attr_name
                attr_ind.typeName    = attr_type
                assoc_ind.hasAttribute.append(attr_ind)

                if attr_type in entity_individuals:
                    attr_ind.entityTypeRef = entity_individuals[attr_type]
                else:
                    attr_ind.observableTypeRef = _get_or_create_obs(attr_type)

