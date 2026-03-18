from utils import ICDJSONToTuples
from reasoner import build_and_load_metamodel, OWLAxiomApplier, run_reasoner, print_summary

from owlready2 import OwlReadyInconsistentOntologyError

if __name__ == "__main__":
    # Example input JSON
    sample_json = """{
      "message_id": "101",
      "message_name": "Inertial States",
      "entities": [
        {
          "entity_id": "E001",
          "usage_context": "Reports vehicle inertial state inclu[truncated]"
        }
      ],
      "fieldMappings": [
        {
          "field_id": "0101.01",
          "field_name": "Time Stamp",
          "entity_id": "E001",
          "observable_id": "0008",
          "measurement_id": "M045",
          "platform_type": "Double",
          "platform_units": "Seconds",
          "range": "See Section 1.7.2",
          "required": true
        },
        {
          "field_id": "0101.02",
          "field_name": "Speed",
          "entity_id": "E001",
          "platform_type": "Double",
          "platform_units": "m/s",
          "range": "x ∈ [0, 100]"
        },
        {
          "field_id": "0101.03",
          "field_name": "Mode",
          "entity_id": "E001",
          "range": "{OFF, IDLE, ACTIVE}"
        }
      ]
    }"""

    # 1. Load/build ontology schema
    onto = build_and_load_metamodel("icd_metamodel.owl")

    # 2. Convert JSON to tuples
    parser = ICDJSONToTuples()
    axioms = parser.parse(sample_json)
    
    # 3. Populate ontology with individuals and assertions
    applier = OWLAxiomApplier(onto)
    applier.apply_axioms(axioms)

    # 4. Run the reasoner
    try:
        run_reasoner(
            onto,
            use_pellet=False,              # True => Pellet, False => HermiT
            infer_property_values=True,
        )
    except OwlReadyInconsistentOntologyError as e:
        print("\nOntology is inconsistent!")
        print(e)
        raise

    # 5. Print a quick summary
    print_summary(onto)

    # 6. Save the populated + inferred ontology
    onto.save(file="icd_populated_inferred.owl", format="rdfxml")
    print("\nSaved populated ontology to icd_populated_inferred.owl")