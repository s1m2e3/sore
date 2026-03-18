from utils import ICDJSONToTuples
from visualization import render_owl_axioms_graph

if __name__ == "__main__":
    sample = """{
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

    parser = ICDJSONToTuples()
    axioms = parser.parse(sample)
    
    path = render_owl_axioms_graph(
        axioms,
        output_basename="ontology_graph_colored",
        output_format="svg",
        show_data_properties=True,   # set False to draw literals as separate nodes
    )
    print(f"Rendered: {path}")

    for axiom in axioms:
        print(axiom)