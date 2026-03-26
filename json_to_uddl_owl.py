"""
json_to_uddl_owl.py
-------------------
Converts a UDDL layered JSON message into:
  1. An OWL/RDF instance graph that conforms to the UDDL metamodel ontology
     (base_ontology.py → outputs/uddl_metamodel.owl).
  2. A flat list of OWL-style assertion tuples compatible with visualization.py.

The resulting OWL graph populates the three-layer semantic chain:

    PlatformComposition
        --platformCompositionRealizes-->
    LogicalMeasurement
        --measurementRealizes-->
    ConceptualObservable

which is the backbone of the human-vs-LLM comparison framework.

Axiom tuple shapes (mirrors ICDJSONToTuples convention):
    ("ClassAssertion",          class_name,   individual_name)
    ("ObjectPropertyAssertion", prop_name,    subject,         object)
    ("DataPropertyAssertion",   prop_name,    subject,         literal_value)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from owlready2 import (
    get_ontology,
    destroy_entity,
    default_world,
    FunctionalProperty,
)

Axiom = Tuple[str, ...]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(value: Any) -> str:
    """Sanitise a string for use as an OWL local name."""
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

class UDDLJsonToOWL:
    """
    Parse a UDDL layered JSON message and populate an OWL instance ontology
    that imports the UDDL metamodel.

    Usage
    -----
    converter = UDDLJsonToOWL()
    onto, axioms = converter.parse(json_data, output_path="outputs/msg_001.owl")
    """

    def __init__(
        self,
        metamodel_path: str = "./outputs/uddl_metamodel.owl",
        instance_base_iri: str = "http://example.org/uddl_instances/",
    ):
        self.metamodel_path = os.path.abspath(metamodel_path)
        self.instance_base_iri = instance_base_iri.rstrip("/") + "/"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        payload: Union[str, Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> Tuple[Any, List[Axiom]]:
        """
        Parameters
        ----------
        payload     : UDDL JSON as a string or already-parsed dict.
        output_path : If given, save the OWL file here.

        Returns
        -------
        (ontology, axiom_list)
            ontology   – owlready2 Ontology object (importable, query-able)
            axiom_list – flat list of OWL assertion tuples
        """
        data = json.loads(payload) if isinstance(payload, str) else payload
        msg = data.get("uddl_layered_message", data)

        # Per-call state
        self._axioms: List[Axiom] = []
        self._id_map: Dict[str, Any] = {}   # json element id → OWL individual

        # ── Load metamodel ──────────────────────────────────────────────
        # owlready2 accepts the raw absolute OS path directly on all platforms.
        meta = get_ontology(self.metamodel_path).load()
        self._meta = meta

        # ── Create instance ontology ────────────────────────────────────
        msg_id   = msg.get("message_id", "unknown")
        inst_iri = self.instance_base_iri + _safe(msg_id) + "#"
        onto     = get_ontology(inst_iri)
        onto.imported_ontologies.append(meta)
        self._onto = onto

        with onto:
            dm_ind = self._make_data_model(msg, msg_id)
            self._parse_conceptual(msg.get("conceptual", {}), dm_ind)
            self._parse_logical(msg.get("logical", {}), dm_ind)
            self._parse_platform(msg.get("platform", {}), dm_ind)

        if output_path:
            onto.save(file=output_path, format="rdfxml")

        return onto, self._axioms

    # ------------------------------------------------------------------
    # DataModel (top-level container)
    # ------------------------------------------------------------------

    def _make_data_model(self, msg: Dict, msg_id: str) -> Any:
        dm = self._meta.DataModel(self._local(msg_id, "dm"))
        self._set_data(dm, "hasIdentifier", msg_id)
        self._set_data(dm, "hasName",       msg.get("message_name", msg_id))
        self._set_data(dm, "hasDescription",msg.get("description"))
        self._set_data(dm, "hasSource",     msg.get("source"))
        self._set_data(dm, "derivedFromICD",msg.get("derived_from_icd"))
        self._set_data(dm, "generationModel",msg.get("generation_model"))
        self._emit_ca("DataModel", dm.name)
        return dm

    # ------------------------------------------------------------------
    # Conceptual layer
    # ------------------------------------------------------------------

    def _parse_conceptual(self, c: Dict, dm_ind: Any) -> Any:
        cdm = self._meta.ConceptualDataModel(self._local(dm_ind.name, "cdm"))
        self._set_data(cdm, "hasName", "ConceptualLayer")
        self._set_obj(dm_ind, "hasConceptualModel", cdm)
        self._emit_ca("ConceptualDataModel", cdm.name)
        self._emit_opa("hasConceptualModel", dm_ind.name, cdm.name)

        # ── Basis entities ───────────────────────────────────────────────
        for b in c.get("basis_entities", []):
            self._parse_basis_entity(b, cdm)

        # ── Observables (defined first; compositions reference them) ─────
        for obs in c.get("observables", []):
            self._parse_observable(obs, cdm)

        # ── Conceptual entities ──────────────────────────────────────────
        for ent in c.get("entities", []):
            self._parse_conceptual_entity(ent, cdm)

        return cdm

    def _parse_basis_entity(self, b: Dict, cdm: Any) -> Any:
        bid  = b["id"]
        ind  = self._meta.BasisEntity(self._local(bid, "basis"))
        self._set_data(ind, "hasIdentifier",  bid)
        self._set_data(ind, "hasName",        b.get("name", bid))
        self._set_data(ind, "hasDescription", b.get("description"))
        self._set_obj(cdm, "hasBasisEntity", ind)
        self._id_map[bid] = ind
        self._emit_ca("BasisEntity", ind.name)
        self._emit_dpa("hasIdentifier", ind.name, bid)
        self._emit_dpa("hasName",       ind.name, b.get("name", bid))
        self._emit_opa("hasBasisEntity", cdm.name, ind.name)
        return ind

    def _parse_observable(self, obs: Dict, cdm: Any) -> Any:
        oid  = obs["id"]
        ind  = self._meta.ConceptualObservable(self._local(oid, "obs"))
        self._set_data(ind, "hasIdentifier",  oid)
        self._set_data(ind, "hasName",        obs.get("name", oid))
        self._set_data(ind, "hasDescription", obs.get("description"))
        self._set_obj(cdm, "hasConceptualObservable", ind)
        self._id_map[oid] = ind
        self._emit_ca("ConceptualObservable", ind.name)
        self._emit_dpa("hasIdentifier", ind.name, oid)
        self._emit_dpa("hasName",       ind.name, obs.get("name", oid))
        self._emit_opa("hasConceptualObservable", cdm.name, ind.name)
        return ind

    def _parse_conceptual_entity(self, ent: Dict, cdm: Any) -> Any:
        eid  = ent["id"]
        ind  = self._meta.ConceptualEntity(self._local(eid, "centity"))
        self._set_data(ind, "hasIdentifier",  eid)
        self._set_data(ind, "hasName",        ent.get("name", eid))
        self._set_data(ind, "hasDescription", ent.get("description"))
        self._set_obj(cdm, "hasConceptualEntity", ind)
        self._id_map[eid] = ind
        self._emit_ca("ConceptualEntity", ind.name)
        self._emit_dpa("hasIdentifier", ind.name, eid)
        self._emit_dpa("hasName",       ind.name, ent.get("name", eid))
        self._emit_opa("hasConceptualEntity", cdm.name, ind.name)

        # specializes link
        spec_ref = ent.get("specializes")
        if spec_ref and spec_ref in self._id_map:
            self._set_obj(ind, "specializes", self._id_map[spec_ref])
            self._emit_opa("specializes", ind.name, self._id_map[spec_ref].name)

        # applies_to: make each composition's observable apply to this entity
        for comp in ent.get("compositions", []):
            self._parse_conceptual_composition(comp, ind)

        return ind

    def _parse_conceptual_composition(self, comp: Dict, entity_ind: Any) -> Any:
        cid  = comp["id"]
        ind  = self._meta.ConceptualComposition(self._local(cid, "ccomp"))
        self._set_data(ind, "hasIdentifier", cid)
        self._set_data(ind, "hasRoleName",   comp.get("role_name", cid))
        self._set_data(ind, "hasDescription",comp.get("description"))
        self._set_obj(entity_ind, "hasConceptualComposition", ind)
        self._id_map[cid] = ind

        self._emit_ca("ConceptualComposition", ind.name)
        self._emit_dpa("hasRoleName", ind.name, comp.get("role_name", cid))
        self._emit_opa("hasConceptualComposition", entity_ind.name, ind.name)

        # ── compositionTarget: observable_ref OR entity_ref ─────────────
        obs_ref = comp.get("observable_ref")
        ent_ref = comp.get("entity_ref")

        if obs_ref and obs_ref in self._id_map:
            target = self._id_map[obs_ref]
            self._set_obj(ind, "compositionTarget", target)
            self._emit_opa("compositionTarget", ind.name, target.name)
            # appliesTo: the observable applies to the containing entity
            self._set_obj(target, "appliesTo", entity_ind)
            self._emit_opa("appliesTo", target.name, entity_ind.name)

        elif ent_ref and ent_ref in self._id_map:
            target = self._id_map[ent_ref]
            self._set_obj(ind, "compositionTarget", target)
            self._emit_opa("compositionTarget", ind.name, target.name)

        # ── Cardinality ──────────────────────────────────────────────────
        card_str = comp.get("cardinality")
        if card_str:
            card_ind = self._make_cardinality(cid, card_str)
            self._set_obj(ind, "hasCardinality", card_ind)
            self._emit_opa("hasCardinality", ind.name, card_ind.name)

        return ind

    # ------------------------------------------------------------------
    # Logical layer
    # ------------------------------------------------------------------

    def _parse_logical(self, l: Dict, dm_ind: Any) -> Any:
        ldm = self._meta.LogicalDataModel(self._local(dm_ind.name, "ldm"))
        self._set_data(ldm, "hasName", "LogicalLayer")
        self._set_obj(dm_ind, "hasLogicalModel", ldm)
        self._emit_ca("LogicalDataModel", ldm.name)
        self._emit_opa("hasLogicalModel", dm_ind.name, ldm.name)

        # ── Logical entity realizations ──────────────────────────────────
        for er in l.get("entity_realizations", []):
            self._parse_logical_entity(er, ldm)

        # ── Measurements ─────────────────────────────────────────────────
        for meas in l.get("measurements", []):
            self._parse_logical_measurement(meas, ldm)

        return ldm

    def _parse_logical_entity(self, er: Dict, ldm: Any) -> Any:
        eid  = er["id"]
        ind  = self._meta.LogicalEntity(self._local(eid, "lentity"))
        self._set_data(ind, "hasIdentifier",  eid)
        self._set_data(ind, "hasName",        er.get("name", eid))
        self._set_data(ind, "hasDescription", er.get("description"))
        self._set_obj(ldm, "hasLogicalEntity", ind)
        self._id_map[eid] = ind
        self._emit_ca("LogicalEntity", ind.name)
        self._emit_dpa("hasIdentifier", ind.name, eid)
        self._emit_dpa("hasName",       ind.name, er.get("name", eid))
        self._emit_opa("hasLogicalEntity", ldm.name, ind.name)

        # logicalEntityRealizes → ConceptualEntity
        real_ref = er.get("realizes")
        if real_ref and real_ref in self._id_map:
            target = self._id_map[real_ref]
            self._set_obj(ind, "logicalEntityRealizes", target)
            self._emit_opa("logicalEntityRealizes", ind.name, target.name)

        return ind

    def _parse_logical_measurement(self, meas: Dict, ldm: Any) -> Any:
        mid  = meas["id"]
        ind  = self._meta.LogicalMeasurement(self._local(mid, "lmeas"))
        self._set_data(ind, "hasIdentifier",    mid)
        self._set_data(ind, "hasName",          meas.get("name", mid))
        self._set_data(ind, "hasDescription",   meas.get("description"))
        self._set_data(ind, "primitiveDataType",meas.get("primitive_data_type"))
        self._set_obj(ldm, "hasLogicalMeasurement", ind)
        self._id_map[mid] = ind
        self._emit_ca("LogicalMeasurement", ind.name)
        self._emit_dpa("hasIdentifier",     ind.name, mid)
        self._emit_dpa("hasName",           ind.name, meas.get("name", mid))
        self._emit_opa("hasLogicalMeasurement", ldm.name, ind.name)

        if meas.get("primitive_data_type"):
            self._emit_dpa("primitiveDataType", ind.name, meas["primitive_data_type"])

        # numeric value
        nv = meas.get("numeric_value")
        if nv is not None:
            self._set_data(ind, "numericValue", float(nv))
            self._emit_dpa("numericValue", ind.name, str(nv))

        # measurementRealizes → ConceptualObservable  (KEY semantic link)
        real_ref = meas.get("realizes")
        if real_ref and real_ref in self._id_map:
            target = self._id_map[real_ref]
            self._set_obj(ind, "measurementRealizes", target)
            self._emit_opa("measurementRealizes", ind.name, target.name)

        # ── LogicalValueType ─────────────────────────────────────────────
        vtype_name = meas.get("value_type")
        if vtype_name:
            vtype_id  = f"vtype_{_safe(vtype_name)}"
            vtype_ind = self._get_or_create(
                vtype_id, self._meta.LogicalValueType,
                {"hasName": vtype_name, "valueTypeName": vtype_name}
            )
            self._set_obj(ldm,  "hasLogicalValueType",    vtype_ind)
            self._set_obj(ind,  "measurementValueType",   vtype_ind)
            self._emit_opa("measurementValueType", ind.name, vtype_ind.name)

        # ── LogicalUnit ──────────────────────────────────────────────────
        unit_name = meas.get("unit")
        if unit_name:
            unit_id  = f"unit_{_safe(unit_name)}"
            unit_ind = self._get_or_create(
                unit_id, self._meta.LogicalUnit,
                {"hasName": unit_name, "symbol": unit_name}
            )
            self._set_obj(ldm, "hasLogicalUnit",    unit_ind)
            self._set_obj(ind, "measurementUnit",   unit_ind)
            self._emit_opa("measurementUnit", ind.name, unit_ind.name)

        # ── LogicalMeasurementSystem ─────────────────────────────────────
        msys_name = meas.get("measurement_system")
        if msys_name:
            msys_id  = f"msys_{_safe(msys_name)}"
            msys_ind = self._get_or_create(
                msys_id, self._meta.LogicalMeasurementSystem,
                {"hasName": msys_name, "measurementSystemName": msys_name}
            )
            self._set_obj(ldm, "hasMeasurementSystem", msys_ind)
            self._set_obj(ind, "measurementSystem",    msys_ind)
            self._emit_opa("measurementSystem", ind.name, msys_ind.name)

        # ── LogicalCoordinateSystem ──────────────────────────────────────
        csys_name = meas.get("coordinate_system")
        if csys_name:
            csys_id  = f"csys_{_safe(csys_name)}"
            csys_ind = self._get_or_create(
                csys_id, self._meta.LogicalCoordinateSystem,
                {"hasName": csys_name}
            )
            self._set_obj(ldm, "hasCoordinateSystem", csys_ind)

        return ind

    # ------------------------------------------------------------------
    # Platform layer
    # ------------------------------------------------------------------

    def _parse_platform(self, p: Dict, dm_ind: Any) -> Any:
        pdm = self._meta.PlatformDataModel(self._local(dm_ind.name, "pdm"))
        self._set_data(pdm, "hasName", p.get("message_type", "PlatformLayer"))
        self._set_obj(dm_ind, "hasPlatformModel", pdm)
        self._emit_ca("PlatformDataModel", pdm.name)
        self._emit_opa("hasPlatformModel", dm_ind.name, pdm.name)

        # ── Platform entity realizations ─────────────────────────────────
        for er in p.get("entity_realizations", []):
            self._parse_platform_entity(er, pdm)

        # ── Platform compositions (fields) ───────────────────────────────
        # Find the first platform entity to attach compositions to.
        # For multi-entity messages this would need to be resolved by the
        # pentity id, but for the current single-entity JSON format the
        # first entity is the owner.
        pentity_ind = self._id_map.get(
            next(iter(er["id"] for er in p.get("entity_realizations", [])), None),
            None,
        )

        for pcomp in p.get("platform_compositions", []):
            self._parse_platform_composition(pcomp, pdm, pentity_ind)

        return pdm

    def _parse_platform_entity(self, er: Dict, pdm: Any) -> Any:
        eid  = er["id"]
        ind  = self._meta.PlatformEntity(self._local(eid, "pentity"))
        self._set_data(ind, "hasIdentifier",  eid)
        self._set_data(ind, "hasName",        er.get("name", eid))
        self._set_data(ind, "hasDescription", er.get("description"))
        self._set_obj(pdm, "hasPlatformEntity", ind)
        self._id_map[eid] = ind
        self._emit_ca("PlatformEntity", ind.name)
        self._emit_dpa("hasIdentifier", ind.name, eid)
        self._emit_dpa("hasName",       ind.name, er.get("name", eid))
        self._emit_opa("hasPlatformEntity", pdm.name, ind.name)

        # platformEntityRealizes → LogicalEntity  (KEY realization link)
        real_ref = er.get("realizes")
        if real_ref and real_ref in self._id_map:
            target = self._id_map[real_ref]
            self._set_obj(ind, "platformEntityRealizes", target)
            self._emit_opa("platformEntityRealizes", ind.name, target.name)

        return ind

    def _parse_platform_composition(
        self, pcomp: Dict, pdm: Any, pentity_ind: Optional[Any]
    ) -> Any:
        pcid = pcomp["id"]
        ind  = self._meta.PlatformComposition(self._local(pcid, "pcomp"))
        self._set_data(ind, "hasIdentifier", pcid)
        self._set_data(ind, "hasRoleName",   pcomp.get("role_name", pcid))
        self._set_data(ind, "hasDescription",pcomp.get("description"))
        self._id_map[pcid] = ind

        self._emit_ca("PlatformComposition", ind.name)
        self._emit_dpa("hasIdentifier", ind.name, pcid)
        self._emit_dpa("hasRoleName",   ind.name, pcomp.get("role_name", pcid))

        # Attach to the platform entity that owns it
        if pentity_ind is not None:
            self._set_obj(pentity_ind, "hasPlatformComposition", ind)
            self._emit_opa("hasPlatformComposition", pentity_ind.name, ind.name)

        # ── PlatformDataType ─────────────────────────────────────────────
        dtype_name = pcomp.get("data_type")
        if dtype_name:
            dtype_id  = f"dtype_{_safe(dtype_name)}"
            dtype_cls = (
                self._meta.PlatformPrimitive
                if dtype_name in ("Double", "Float", "Int32", "Int64",
                                  "UInt32", "Char", "Boolean", "String",
                                  "Integer", "Natural")
                else self._meta.PlatformDataType
            )
            dtype_ind = self._get_or_create(
                dtype_id, dtype_cls,
                {"hasName": dtype_name, "primitiveName": dtype_name}
            )
            self._set_obj(pdm, "hasPlatformDataType",       dtype_ind)
            self._set_obj(ind, "platformCompositionTarget", dtype_ind)
            self._set_obj(ind, "hasFieldType",              dtype_ind)
            self._emit_opa("platformCompositionTarget", ind.name, dtype_ind.name)

        # ── Cardinality ──────────────────────────────────────────────────
        card_str = pcomp.get("cardinality")
        if card_str:
            card_ind = self._make_cardinality(pcid, card_str)
            self._set_obj(ind, "hasCardinality", card_ind)
            self._emit_opa("hasCardinality", ind.name, card_ind.name)

        # ── platformCompositionRealizes → LogicalMeasurement  (KEY link) ─
        real_ref = pcomp.get("realizes")
        if real_ref and real_ref in self._id_map:
            target = self._id_map[real_ref]
            self._set_obj(ind, "platformCompositionRealizes", target)
            self._emit_opa("platformCompositionRealizes", ind.name, target.name)

        # Data value (instance-level — stored as a data property annotation)
        val = pcomp.get("value")
        if val is not None:
            self._emit_dpa("hasFieldValue", ind.name, str(val))

        return ind

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_or_create(
        self,
        local_id: str,
        cls: Any,
        data_props: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Return an existing individual or create a new one."""
        if local_id in self._id_map:
            return self._id_map[local_id]
        ind = cls(local_id)
        for prop_name, value in (data_props or {}).items():
            if value:
                self._set_data(ind, prop_name, value)
        self._id_map[local_id] = ind
        self._emit_ca(cls.name, local_id)
        for prop_name, value in (data_props or {}).items():
            if value:
                self._emit_dpa(prop_name, local_id, value)
        return ind

    def _make_cardinality(self, owner_id: str, card_str: str) -> Any:
        """Parse a '1..1' or '0..*' string and create a Cardinality individual."""
        card_id = f"card_{_safe(owner_id)}"
        if card_id in self._id_map:
            return self._id_map[card_id]
        ind = self._meta.Cardinality(card_id)
        self._id_map[card_id] = ind
        self._emit_ca("Cardinality", card_id)
        try:
            parts = re.split(r"\.\.", card_str.strip())
            lower = int(parts[0]) if parts[0].isdigit() else 0
            upper_str = parts[1] if len(parts) > 1 else parts[0]
            upper = -1 if upper_str == "*" else int(upper_str)
            ind.lowerBound = [lower]
            ind.upperBound = [upper]
            self._emit_dpa("lowerBound", card_id, str(lower))
            self._emit_dpa("upperBound", card_id, str(upper))
        except Exception:
            pass
        return ind

    def _local(self, *parts: str) -> str:
        """Build a sanitised local name from parts."""
        return "_".join(_safe(p) for p in parts if p)

    def _set_data(self, ind: Any, prop_name: str, value: Any) -> None:
        if value is None:
            return
        prop = getattr(self._meta, prop_name, None)
        if prop is None:
            return
        try:
            # FunctionalProperty requires a scalar; non-functional takes a list
            if issubclass(prop, FunctionalProperty):
                setattr(ind, prop_name, value)
            else:
                setattr(ind, prop_name, [value])
        except Exception:
            pass

    def _set_obj(self, subject: Any, prop_name: str, obj: Any) -> None:
        prop = getattr(self._meta, prop_name, None)
        if prop is None:
            return
        try:
            if issubclass(prop, FunctionalProperty):
                setattr(subject, prop_name, obj)
            else:
                current = getattr(subject, prop_name) or []
                if obj not in current:
                    setattr(subject, prop_name, current + [obj])
        except Exception:
            pass

    # ── Axiom tuple emitters ────────────────────────────────────────────

    def _emit_ca(self, class_name: str, individual: str) -> None:
        t = ("ClassAssertion", class_name, individual)
        if t not in self._axioms:
            self._axioms.append(t)

    def _emit_opa(self, prop: str, subj: str, obj: str) -> None:
        t = ("ObjectPropertyAssertion", prop, subj, obj)
        if t not in self._axioms:
            self._axioms.append(t)

    def _emit_dpa(self, prop: str, subj: str, value: str) -> None:
        if value is None:
            return
        t = ("DataPropertyAssertion", prop, subj, str(value))
        if t not in self._axioms:
            self._axioms.append(t)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def parse_uddl_json(
    payload: Union[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    metamodel_path: str = "./outputs/uddl_metamodel.owl",
) -> Tuple[Any, List[Axiom]]:
    """One-shot helper: parse a UDDL JSON message and return (onto, axioms)."""
    converter = UDDLJsonToOWL(metamodel_path=metamodel_path)
    return converter.parse(payload, output_path=output_path)
