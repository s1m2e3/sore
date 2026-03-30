"""
aml_matcher.py
--------------
Python-based ontology alignment engine modelled after AgreementMakerLight (AML).

Given two conceptual-model JSON files it computes correspondences between
their entities using a two-signal similarity measure:

    Signal 1 – Lexical similarity (name-based)
        Normalised token-set similarity over the entity's camelCase/snake_case
        name tokens.  Exact matches score 1.0; partial overlaps score < 1.0.

    Signal 2 – Observable-type signature similarity (semantic anchor)
        Jaccard similarity over the sets of observable types (Temperature,
        Torque, …) that appear in an entity's attribute list.
        Entities with the same physical-quantity profile are likely the same
        concept even if named differently.

    Combined score = 0.6 * lexical + 0.4 * type_jaccard

Greedy 1-to-1 assignment: pairs are accepted in descending score order;
each entity from either ontology may be matched at most once.

Output format – OWL Alignment API (standard RDF/XML used by AML itself):
    http://alignapi.gforge.inria.fr/format.html

Usage
-----
    from aml_matcher import AMLMatcher
    matcher = AMLMatcher()
    alignment = matcher.match(json_a, json_b, iri_a="...", iri_b="...")
    matcher.save(alignment, "AML/Automobile/V1_vs_V2.rdf")
"""

from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any

# Similarity thresholds
MATCH_THRESHOLD   = 0.30   # minimum combined score to include a pair
LEXICAL_WEIGHT    = 0.60
TYPE_SIM_WEIGHT   = 0.40


# --------------------------------------------------------------------------- #
# Token helpers                                                                #
# --------------------------------------------------------------------------- #

def _tokenize(name: str) -> list[str]:
    """Split a camelCase / PascalCase / snake_case / space-separated name."""
    # Insert space before capital letters that follow lower-case letters
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    # Split on non-alphanumeric
    tokens = re.split(r"[^A-Za-z0-9]+", s)
    return [t.lower() for t in tokens if t]


def _lexical_sim(a: str, b: str) -> float:
    """Token-set ratio between two entity names (0–1)."""
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    # Jaccard on tokens
    jac = len(ta & tb) / len(ta | tb)
    # Sequence ratio on raw lowercased names (catches partial-word overlaps)
    seq = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return max(jac, seq)


def _type_jaccard(types_a: set[str], types_b: set[str]) -> float:
    """Jaccard similarity of observable-type sets."""
    if not types_a or not types_b:
        return 0.0
    return len(types_a & types_b) / len(types_a | types_b)


# --------------------------------------------------------------------------- #
# Model parsing                                                                #
# --------------------------------------------------------------------------- #

def _parse_model(data: dict[str, Any]) -> dict[str, dict]:
    """
    Extract entity records from a JSON model (Type-A or Type-B schema).

    Returns
    -------
    dict  entity_name -> {
              "obs_types": set of observable type strings,
              "attr_names": set of attribute names,
          }
    """
    entities: dict[str, dict] = {}
    # Collect the declared observable list (Type-A only) for type resolution
    declared_obs = set(data.get("observables", []))

    for ent in data.get("entities", []):
        ent_name  = ent.get("entityName") or ent.get("name", "")
        raw_attrs = ent.get("entityAttributes") or ent.get("attributes", [])

        obs_types  = set()
        attr_names = set()
        for attr in raw_attrs:
            attr_type = attr.get("type", "")
            attr_names.add(attr.get("name", ""))
            # Include type if it looks like an observable (not another entity)
            if attr_type in declared_obs or (
                declared_obs == set() and attr_type  # Type-B: include all
            ):
                obs_types.add(attr_type)

        entities[ent_name] = {"obs_types": obs_types, "attr_names": attr_names}

    return entities


# --------------------------------------------------------------------------- #
# Alignment cell and result                                                    #
# --------------------------------------------------------------------------- #

class AlignmentCell:
    __slots__ = ("entity_a", "entity_b", "score", "relation")

    def __init__(self, entity_a: str, entity_b: str, score: float, relation: str = "="):
        self.entity_a  = entity_a
        self.entity_b  = entity_b
        self.score     = score
        self.relation  = relation


class Alignment:
    def __init__(
        self,
        model_name_a: str,
        model_name_b: str,
        iri_a: str,
        iri_b: str,
        owl_path_a: str,
        owl_path_b: str,
    ):
        self.model_name_a = model_name_a
        self.model_name_b = model_name_b
        self.iri_a        = iri_a
        self.iri_b        = iri_b
        self.owl_path_a   = owl_path_a
        self.owl_path_b   = owl_path_b
        self.cells: list[AlignmentCell] = []


# --------------------------------------------------------------------------- #
# Matcher                                                                      #
# --------------------------------------------------------------------------- #

class AMLMatcher:
    """
    Approximates AgreementMakerLight's matching pipeline using:
      1. Lexical similarity on entity names
      2. Observable-type Jaccard similarity
      3. Greedy 1:1 assignment above a threshold
    """

    def match(
        self,
        json_a: dict[str, Any],
        json_b: dict[str, Any],
        iri_a: str  = "http://example.org/onto_a#",
        iri_b: str  = "http://example.org/onto_b#",
        owl_path_a: str = "",
        owl_path_b: str = "",
    ) -> Alignment:
        model_name_a = json_a.get("modelName", "OntologyA")
        model_name_b = json_b.get("modelName", "OntologyB")

        alignment = Alignment(
            model_name_a=model_name_a,
            model_name_b=model_name_b,
            iri_a=iri_a,
            iri_b=iri_b,
            owl_path_a=owl_path_a,
            owl_path_b=owl_path_b,
        )

        entities_a = _parse_model(json_a)
        entities_b = _parse_model(json_b)

        # Build all candidate pairs with their combined score
        candidates: list[tuple[float, str, str]] = []
        for name_a, info_a in entities_a.items():
            for name_b, info_b in entities_b.items():
                lex  = _lexical_sim(name_a, name_b)
                tsim = _type_jaccard(info_a["obs_types"], info_b["obs_types"])
                score = LEXICAL_WEIGHT * lex + TYPE_SIM_WEIGHT * tsim
                if score >= MATCH_THRESHOLD:
                    candidates.append((score, name_a, name_b))

        # Greedy 1:1 assignment (highest score first)
        candidates.sort(reverse=True)
        used_a: set[str] = set()
        used_b: set[str] = set()
        for score, name_a, name_b in candidates:
            if name_a in used_a or name_b in used_b:
                continue
            alignment.cells.append(AlignmentCell(name_a, name_b, score))
            used_a.add(name_a)
            used_b.add(name_b)

        return alignment

    def match_files(
        self,
        json_path_a: str,
        json_path_b: str,
        owl_path_a: str = "",
        owl_path_b: str = "",
    ) -> Alignment:
        """Load two JSON files and match them."""
        with open(json_path_a, encoding="utf-8") as f:
            data_a = json.load(f)
        with open(json_path_b, encoding="utf-8") as f:
            data_b = json.load(f)

        safe_a = re.sub(r"[^A-Za-z0-9_]+", "_", data_a.get("modelName", "onto_a"))
        safe_b = re.sub(r"[^A-Za-z0-9_]+", "_", data_b.get("modelName", "onto_b"))
        iri_a  = f"http://example.org/concept_instances/{safe_a}/"
        iri_b  = f"http://example.org/concept_instances/{safe_b}/"

        return self.match(
            data_a, data_b,
            iri_a=iri_a, iri_b=iri_b,
            owl_path_a=owl_path_a, owl_path_b=owl_path_b,
        )

    # ---------------------------------------------------------------------- #
    # Output                                                                  #
    # ---------------------------------------------------------------------- #

    def save(self, alignment: Alignment, output_path: str) -> None:
        """Serialise the alignment in OWL Alignment API RDF/XML format."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<rdf:RDF',
            '  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
            '  xmlns:xsd="http://www.w3.org/2001/XMLSchema#"',
            '  xmlns:align="http://knowledgeweb.semanticweb.org/heterogeneity/alignment#">',
            '',
            '  <align:Alignment>',
            '    <align:xml>yes</align:xml>',
            '    <align:level>0</align:level>',
            '    <align:type>**</align:type>',
            '',
            '    <!-- Source ontology -->',
            '    <align:onto1>',
            f'      <align:Ontology rdf:about="{alignment.iri_a}">',
            f'        <align:location>{alignment.owl_path_a}</align:location>',
            f'        <align:formalism><align:Formalism align:name="{alignment.model_name_a}"/></align:formalism>',
            '      </align:Ontology>',
            '    </align:onto1>',
            '',
            '    <!-- Target ontology -->',
            '    <align:onto2>',
            f'      <align:Ontology rdf:about="{alignment.iri_b}">',
            f'        <align:location>{alignment.owl_path_b}</align:location>',
            f'        <align:formalism><align:Formalism align:name="{alignment.model_name_b}"/></align:formalism>',
            '      </align:Ontology>',
            '    </align:onto2>',
            '',
            f'    <!-- {len(alignment.cells)} correspondences -->',
        ]

        for cell in alignment.cells:
            entity_iri_a = f"{alignment.iri_a}{cell.entity_a}"
            entity_iri_b = f"{alignment.iri_b}{cell.entity_b}"
            lines += [
                '    <align:map>',
                '      <align:Cell>',
                f'        <align:entity1 rdf:resource="{entity_iri_a}"/>',
                f'        <align:entity2 rdf:resource="{entity_iri_b}"/>',
                f'        <align:measure rdf:datatype="xsd:float">{cell.score:.4f}</align:measure>',
                f'        <align:relation>{cell.relation}</align:relation>',
                '      </align:Cell>',
                '    </align:map>',
            ]

        lines += [
            '  </align:Alignment>',
            '</rdf:RDF>',
        ]

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
