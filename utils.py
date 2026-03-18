import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

Axiom = Tuple[str, ...]


class ICDJSONToTuples:
    """
    Convert an ICD-style JSON payload into OWL-style assertion tuples.

    Output tuple shapes:
      ("ClassAssertion", class_name, individual_name)
      ("ObjectPropertyAssertion", property_name, subject_individual, object_individual)
      ("DataPropertyAssertion", property_name, subject_individual, literal_value)
    """

    ENTITY_CHILD_KEYS = ("entities", "subentities", "subEntities", "children")
    FIELD_KEYS = ("fieldMappings", "fields")

    # Examples supported:
    #   x \in [a,b]
    #   x ∈ [a,b]
    #   x in [a,b]
    #   [a,b]
    #   (200-ε,200+ε), ε > 0
    INTERVAL_RE = re.compile(
        r"""
        ^\s*
        (?:(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\\in|∈|\bin\b)\s*)?
        (?P<left>[\[\(])
        \s*(?P<lower>[^,\]\)]+?)\s*
        ,\s*
        (?P<upper>[^\]\)]+?)\s*
        (?P<right>[\]\)])
        \s*(?:,\s*(?P<qualifier>.+))?
        \s*$
        """,
        re.VERBOSE,
    )

    # Examples supported:
    #   {A,B,C}
    #   x \in {A,B,C}
    #   x ∈ {A,B,C}
    #   x belongs to {A,B,C}
    ENUM_RE = re.compile(
        r"""
        ^\s*
        (?:
            (?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\\in|∈|\bin\b)\s*
            |
            (?P<var2>[A-Za-z_][A-Za-z0-9_]*)\s+belongs\s+to\s+
        )?
        \{\s*(?P<values>[^{}]+?)\s*\}
        \s*$
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.axioms: List[Axiom] = []
        self.entity_lookup: Dict[str, List[str]] = {}
        self.created_individuals: set[str] = set()
        self.field_counter = 0

    def parse(self, payload: Union[str, Dict[str, Any]]) -> List[Axiom]:
        self.axioms = []
        self.entity_lookup = {}
        self.created_individuals = set()
        self.field_counter = 0

        data = json.loads(payload) if isinstance(payload, str) else payload
        if not isinstance(data, dict):
            raise TypeError("Top-level JSON must be an object")

        message_ind = self._parse_message(data)
        self._parse_message_level_fields(data, message_ind)
        return self.axioms

    def _parse_message(self, obj: Dict[str, Any]) -> str:
        message_id = str(obj.get("message_id", "unknown"))
        message_ind = f"message_{self._safe(message_id)}"

        self._ensure_class_assertion("Message", message_ind)
        self._add_data("hasMessageId", message_ind, message_id)

        if "message_name" in obj:
            self._add_data("hasMessageName", message_ind, obj["message_name"])

        for entity_obj in obj.get("entities", []):
            self._parse_entity(
                entity_obj=entity_obj,
                message_ind=message_ind,
                parent_entity_ind=None,
            )

        return message_ind

    def _parse_entity(
        self,
        entity_obj: Dict[str, Any],
        message_ind: str,
        parent_entity_ind: Optional[str],
    ) -> str:
        entity_id = str(entity_obj.get("entity_id", "unknown"))
        scope = parent_entity_ind or message_ind
        entity_ind = f"entity_{self._safe(entity_id)}__in__{self._safe(scope)}"

        if entity_ind not in self.created_individuals:
            self._ensure_class_assertion("ConceptualEntity", entity_ind)
            self._add_data("hasEntityId", entity_ind, entity_id)

            if "entity_name" in entity_obj:
                self._add_data("hasEntityName", entity_ind, entity_obj["entity_name"])

            if "usage_context" in entity_obj:
                self._add_data("hasUsageContext", entity_ind, entity_obj["usage_context"])

            self.entity_lookup.setdefault(entity_id, []).append(entity_ind)

        if parent_entity_ind is None:
            self._add_object("hasEntity", message_ind, entity_ind)
        else:
            self._add_object("hasSubEntity", parent_entity_ind, entity_ind)

        for field_key in self.FIELD_KEYS:
            for field_obj in entity_obj.get(field_key, []):
                self._parse_field(field_obj, owner_entity_ind=entity_ind)

        for child_key in self.ENTITY_CHILD_KEYS:
            for child_entity_obj in entity_obj.get(child_key, []):
                self._parse_entity(
                    entity_obj=child_entity_obj,
                    message_ind=message_ind,
                    parent_entity_ind=entity_ind,
                )

        return entity_ind

    def _parse_message_level_fields(self, obj: Dict[str, Any], message_ind: str) -> None:
        for field_key in self.FIELD_KEYS:
            for field_obj in obj.get(field_key, []):
                owner = self._resolve_owner(field_obj)
                if owner is None:
                    owner = f"entity_unresolved__in__{self._safe(message_ind)}"
                    self._ensure_class_assertion("ConceptualEntity", owner)
                    self._add_data("hasEntityId", owner, "unresolved")
                    self._add_object("hasEntity", message_ind, owner)

                self._parse_field(field_obj, owner_entity_ind=owner)

    def _parse_field(self, field_obj: Dict[str, Any], owner_entity_ind: str) -> str:
        self.field_counter += 1
        field_id = str(field_obj.get("field_id", f"auto_{self.field_counter}"))
        field_name = str(field_obj.get("field_name", "unnamed_field"))

        field_ind = (
            f"field_{self._safe(field_id)}__{self._safe(field_name)}"
            f"__in__{self._safe(owner_entity_ind)}"
        )

        self._ensure_class_assertion("Field", field_ind)
        self._add_object("hasField", owner_entity_ind, field_ind)

        self._add_data("hasFieldId", field_ind, field_id)
        self._add_data("hasFieldName", field_ind, field_name)

        if "entity_id" in field_obj:
            self._add_data("referencesEntityId", field_ind, field_obj["entity_id"])

        if "required" in field_obj:
            self._add_data("isRequired", field_ind, bool(field_obj["required"]))

        if "observable_id" in field_obj:
            observable_ind = f"observable_{self._safe(field_obj['observable_id'])}"
            self._ensure_class_assertion("Observable", observable_ind)
            self._add_data("hasObservableId", observable_ind, field_obj["observable_id"])
            self._add_object("hasObservable", field_ind, observable_ind)

        if "measurement_id" in field_obj:
            measurement_ind = f"measurement_{self._safe(field_obj['measurement_id'])}"
            self._ensure_class_assertion("Measurement", measurement_ind)
            self._add_data("hasMeasurementId", measurement_ind, field_obj["measurement_id"])
            self._add_object("hasMeasurement", field_ind, measurement_ind)

        if "platform_units" in field_obj:
            unit_ind = f"unit_{self._safe(field_obj['platform_units'])}"
            self._ensure_class_assertion("Unit", unit_ind)
            self._add_data("hasUnitName", unit_ind, field_obj["platform_units"])
            self._add_object("hasUnit", field_ind, unit_ind)

        if "platform_type" in field_obj:
            ptype_ind = f"platform_type_{self._safe(field_obj['platform_type'])}"
            self._ensure_class_assertion("PlatformType", ptype_ind)
            self._add_data("hasPlatformTypeName", ptype_ind, field_obj["platform_type"])
            self._add_object("hasPlatformType", field_ind, ptype_ind)

        if "range" in field_obj:
            self._parse_range(field_obj["range"], field_id=field_id, field_ind=field_ind)

        return field_ind

    def _parse_range(self, raw_range: Any, field_id: str, field_ind: str) -> None:
        raw_text = str(raw_range).strip()

        interval_match = self.INTERVAL_RE.match(raw_text)
        if interval_match:
            self._emit_interval_range(interval_match, raw_text, field_id, field_ind)
            return

        enum_match = self.ENUM_RE.match(raw_text)
        if enum_match:
            self._emit_enum_range(enum_match, raw_text, field_id, field_ind)
            return

        self._emit_document_reference_range(raw_text, field_id, field_ind)

    def _emit_document_reference_range(self, raw_text: str, field_id: str, field_ind: str) -> None:
        range_ind = f"range_doc_{self._safe(field_id)}"

        self._ensure_class_assertion("RangeSpec", range_ind)
        self._ensure_class_assertion("DocumentReferenceRange", range_ind)
        self._add_data("hasRawRangeText", range_ind, raw_text)
        self._add_data("hasReferenceText", range_ind, raw_text)
        self._add_object("hasRangeSpec", field_ind, range_ind)

    def _emit_interval_range(self, match: re.Match, raw_text: str, field_id: str, field_ind: str) -> None:
        range_ind = f"range_interval_{self._safe(field_id)}"

        var = match.group("var")
        lower = match.group("lower").strip()
        upper = match.group("upper").strip()
        left = match.group("left")
        right = match.group("right")
        qualifier = match.group("qualifier")

        self._ensure_class_assertion("RangeSpec", range_ind)
        self._ensure_class_assertion("IntervalRange", range_ind)

        self._add_data("hasRawRangeText", range_ind, raw_text)
        if var:
            self._add_data("hasIntervalVariable", range_ind, var)

        self._add_data("hasLowerBoundText", range_ind, lower)
        self._add_data("hasUpperBoundText", range_ind, upper)
        self._add_data("isLowerInclusive", range_ind, left == "[")
        self._add_data("isUpperInclusive", range_ind, right == "]")

        if qualifier:
            self._add_data("hasIntervalQualifierText", range_ind, qualifier.strip())

        self._add_object("hasRangeSpec", field_ind, range_ind)

    def _emit_enum_range(self, match: re.Match, raw_text: str, field_id: str, field_ind: str) -> None:
        range_ind = f"range_enum_{self._safe(field_id)}"

        values_text = match.group("values").strip()
        values = [v.strip() for v in values_text.split(",") if v.strip()]

        self._ensure_class_assertion("RangeSpec", range_ind)
        self._ensure_class_assertion("EnumerationRange", range_ind)
        self._add_data("hasRawRangeText", range_ind, raw_text)

        for value in values:
            self._add_data("hasAllowedLiteral", range_ind, value)

        self._add_object("hasRangeSpec", field_ind, range_ind)

    def _resolve_owner(self, field_obj: Dict[str, Any]) -> Optional[str]:
        entity_id = field_obj.get("entity_id")
        if entity_id is None:
            return None
        matches = self.entity_lookup.get(str(entity_id), [])
        return matches[0] if matches else None

    def _ensure_class_assertion(self, class_name: str, individual_name: str) -> None:
        axiom = ("ClassAssertion", class_name, individual_name)
        if axiom not in self.axioms:
            self.axioms.append(axiom)
        self.created_individuals.add(individual_name)

    def _add_object(self, property_name: str, subject: str, obj: str) -> None:
        axiom = ("ObjectPropertyAssertion", property_name, subject, obj)
        if axiom not in self.axioms:
            self.axioms.append(axiom)

    def _add_data(self, property_name: str, subject: str, value: Any) -> None:
        if value is None:
            return

        if isinstance(value, bool):
            literal = "true" if value else "false"
        else:
            literal = str(value)

        axiom = ("DataPropertyAssertion", property_name, subject, literal)
        if axiom not in self.axioms:
            self.axioms.append(axiom)

    @staticmethod
    def _safe(value: Any) -> str:
        text = str(value).strip()
        text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text or "unknown"

