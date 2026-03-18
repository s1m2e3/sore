from collections import defaultdict
from graphviz import Digraph


def render_owl_axioms_graph(
    axioms,
    output_basename="ontology_graph",
    output_format="svg",
    show_data_properties=True,
):
    """
    Render OWL-style assertion tuples as a colored directed graph.

    Expected tuple forms:
      ("ClassAssertion", class_name, individual_name)
      ("ObjectPropertyAssertion", property_name, subject_individual, object_individual)
      ("DataPropertyAssertion", property_name, subject_individual, literal_value)
    """

    node_colors = {
        "Message": "#4C78A8",
        "ConceptualEntity": "#72B7B2",
        "Field": "#F58518",
        "Observable": "#54A24B",
        "Measurement": "#E45756",
        "Unit": "#B279A2",
        "PlatformType": "#9D755D",
        "RangeSpec": "#BAB0AC",
        "DocumentReferenceRange": "#8CD17D",
        "IntervalRange": "#FF9DA6",
        "EnumerationRange": "#EDC948",
        "Literal": "#EEEEEE",
        "Unknown": "#CCCCCC",
    }

    edge_colors = {
        "hasEntity": "#1F77B4",
        "hasSubEntity": "#17BECF",
        "hasField": "#FF7F0E",
        "hasObservable": "#2CA02C",
        "hasMeasurement": "#D62728",
        "hasUnit": "#9467BD",
        "hasPlatformType": "#8C564B",
        "hasRangeSpec": "#7F7F7F",
        "data_property": "#555555",
        "default": "#444444",
    }

    # Keep all asserted classes per individual, because ranges may be both
    # RangeSpec and a subclass like IntervalRange.
    class_assertions = defaultdict(set)
    data_props = defaultdict(list)
    object_props = []

    for ax in axioms:
        kind = ax[0]

        if kind == "ClassAssertion":
            _, class_name, individual = ax
            class_assertions[individual].add(class_name)

        elif kind == "DataPropertyAssertion":
            _, prop, subject, value = ax
            data_props[subject].append((prop, value))

        elif kind == "ObjectPropertyAssertion":
            _, prop, subject, obj = ax
            object_props.append((prop, subject, obj))

    # Choose a display class for node coloring.
    # Prefer more specific range subclasses over generic RangeSpec.
    def choose_display_class(classes):
        priority = [
            "Message",
            "ConceptualEntity",
            "Field",
            "Observable",
            "Measurement",
            "Unit",
            "PlatformType",
            "DocumentReferenceRange",
            "IntervalRange",
            "EnumerationRange",
            "RangeSpec",
        ]
        for c in priority:
            if c in classes:
                return c
        return next(iter(classes), "Unknown")

    dot = Digraph("ontology", format=output_format)
    dot.attr(rankdir="LR", bgcolor="white", splines="true", overlap="false")
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")
    dot.attr("edge", fontname="Helvetica", arrowsize="0.8")

    # Create main ontology individual nodes
    all_individuals = set(class_assertions.keys())
    for _, s, o in object_props:
        all_individuals.add(s)
        all_individuals.add(o)
    all_individuals.update(data_props.keys())

    for individual in sorted(all_individuals):
        classes = class_assertions.get(individual, {"Unknown"})
        display_class = choose_display_class(classes)
        fill = node_colors.get(display_class, node_colors["Unknown"])

        # Keep data properties inside the node label by default
        rows = []
        rows.append(
            f'<TR><TD BGCOLOR="{fill}"><B>{individual}</B></TD></TR>'
        )
        rows.append(
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10">type: {display_class}</FONT></TD></TR>'
        )

        other_classes = sorted(c for c in classes if c != display_class)
        if other_classes:
            rows.append(
                '<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10">also: '
                + ", ".join(other_classes)
                + "</FONT></TD></TR>"
            )

        if not show_data_properties:
            # Do not list data properties inside node
            pass
        else:
            for prop, value in data_props.get(individual, []):
                safe_value = (
                    str(value)
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                rows.append(
                    f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10">{prop}: {safe_value}</FONT></TD></TR>'
                )

        label = "<<TABLE BORDER='1' CELLBORDER='0' CELLSPACING='0'>" + "".join(rows) + "</TABLE>>"
        dot.node(
            individual,
            label=label,
            fillcolor=fill,
            color="#333333",
            fontcolor="#111111",
        )

    # Optional: render data properties as dashed edges to literal nodes instead
    if not show_data_properties:
        literal_counter = 0
        for subject, props in data_props.items():
            for prop, value in props:
                literal_counter += 1
                lit_id = f"literal_{literal_counter}"
                dot.node(
                    lit_id,
                    label=str(value),
                    shape="note",
                    style="filled",
                    fillcolor=node_colors["Literal"],
                    color="#777777",
                    fontcolor="#222222",
                )
                dot.edge(
                    subject,
                    lit_id,
                    label=prop,
                    color=edge_colors["data_property"],
                    fontcolor=edge_colors["data_property"],
                    style="dashed",
                )

    # Object property edges
    for prop, subject, obj in object_props:
        color = edge_colors.get(prop, edge_colors["default"])
        dot.edge(
            subject,
            obj,
            label=prop,
            color=color,
            fontcolor=color,
            penwidth="1.8",
        )

    # Legend
    with dot.subgraph(name="cluster_legend") as c:
        c.attr(label="Legend", color="#BBBBBB", fontname="Helvetica")
        for cls, color in node_colors.items():
            if cls in {"Literal", "Unknown"}:
                continue
            legend_node = f"legend_{cls}"
            c.node(
                legend_node,
                label=cls,
                shape="box",
                style="rounded,filled",
                fillcolor=color,
                color="#555555",
                fontcolor="#111111",
            )

    outpath = dot.render(output_basename, cleanup=True)
    return outpath

