from pathlib import Path
from typing import Any, Dict, List, Tuple

from owlready2 import (
    get_ontology,
    sync_reasoner,
    sync_reasoner_pellet,
    OwlReadyInconsistentOntologyError,
    default_world,
)

# Optional on Windows if Java is not auto-detected:
# import owlready2
# owlready2.JAVA_EXE = r"C:\Path\To\java.exe"

# These come from your earlier scripts
from base_ontology import build_icd_metamodel
from utils import ICDJSONToTuples

Axiom = Tuple[str, ...]


def coerce_literal(value: str) -> Any:
    """
    Convert string literals from the tuple layer into Python values
    suitable for Owlready2 data properties.
    """
    v = value.strip()

    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False

    return v


class OWLAxiomApplier:
    """
    Takes OWL-style tuples and populates an Owlready2 ontology with
    individuals and property assertions.
    """

    def __init__(self, onto):
        self.onto = onto
        self.individuals: Dict[str, Any] = {}

    def apply_axioms(self, axioms: List[Axiom]) -> None:
        # First pass: create all individuals from ClassAssertion
        for ax in axioms:
            if ax[0] == "ClassAssertion":
                _, class_name, individual_name = ax
                self._ensure_individual(class_name, individual_name)

        # Second pass: add object/data property assertions
        for ax in axioms:
            kind = ax[0]

            if kind == "ObjectPropertyAssertion":
                _, prop_name, subj_name, obj_name = ax
                self._apply_object_property(prop_name, subj_name, obj_name)

            elif kind == "DataPropertyAssertion":
                _, prop_name, subj_name, literal = ax
                self._apply_data_property(prop_name, subj_name, literal)

    def _ensure_individual(self, class_name: str, individual_name: str):
        if individual_name in self.individuals:
            return self.individuals[individual_name]

        cls = self.onto[class_name]
        if cls is None:
            raise ValueError(f"Class '{class_name}' not found in ontology")

        ind = cls(individual_name)
        self.individuals[individual_name] = ind
        return ind

    def _get_property(self, prop_name: str):
        prop = self.onto[prop_name]
        if prop is None:
            raise ValueError(f"Property '{prop_name}' not found in ontology")
        return prop

    def _apply_object_property(self, prop_name: str, subj_name: str, obj_name: str) -> None:
        prop = self._get_property(prop_name)
        subj = self.individuals[subj_name]
        obj = self.individuals[obj_name]

        attr_name = getattr(prop, "python_name", None) or prop.name
        current_value = getattr(subj, attr_name)

        # Functional properties get scalar assignment, others get list append
        if prop.is_functional_for(subj):
            setattr(subj, attr_name, obj)
        else:
            if obj not in current_value:
                current_value.append(obj)

    def _apply_data_property(self, prop_name: str, subj_name: str, literal: str) -> None:
        prop = self._get_property(prop_name)
        subj = self.individuals[subj_name]
        value = coerce_literal(literal)

        attr_name = getattr(prop, "python_name", None) or prop.name
        current_value = getattr(subj, attr_name)

        if prop.is_functional_for(subj):
            setattr(subj, attr_name, value)
        else:
            if value not in current_value:
                current_value.append(value)


def build_and_load_metamodel(owl_path: str = "icd_metamodel.owl"):
    """
    Build the ontology file if missing, then load it.
    """
    path = Path(owl_path)
    if not path.exists():
        build_icd_metamodel(output_path=owl_path)

    onto = get_ontology(str(path.absolute())).load()
    return onto


def run_reasoner(onto, use_pellet: bool = False, infer_property_values: bool = True) -> None:
    """
    Run HermiT (default) or Pellet over the populated ontology.
    Running inside 'with onto:' stores inferred facts in the same ontology.
    """
    with onto:
        if use_pellet:
            sync_reasoner_pellet(
                infer_property_values=infer_property_values,
                infer_data_property_values=True,
                debug=1,
            )
        else:
            sync_reasoner(
                infer_property_values=infer_property_values,
                ignore_unsupported_datatypes=True,
            )


def print_summary(onto) -> None:
    print("\n=== Individuals by class ===")
    for cls_name in [
        "Message",
        "ConceptualEntity",
        "Field",
        "Observable",
        "Measurement",
        "Unit",
        "PlatformType",
        "RangeSpec",
        "DocumentReferenceRange",
        "IntervalRange",
        "EnumerationRange",
    ]:
        cls = onto[cls_name]
        if cls is None:
            continue

        instances = list(cls.instances())
        print(f"{cls_name}: {len(instances)}")
        for inst in instances:
            print(f"  - {inst.name}")

    print("\n=== Inconsistent classes (if any) ===")
    bad = list(default_world.inconsistent_classes())
    if not bad:
        print("  none")
    else:
        for cls in bad:
            print(f"  - {cls}")
