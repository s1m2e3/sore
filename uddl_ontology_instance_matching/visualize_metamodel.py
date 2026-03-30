"""
visualize_metamodel.py
----------------------
Renders the UDDL metamodel ontology (base_ontology.py) as a layered graph.

Outputs (in ./outputs/):
  uddl_metamodel_graph.svg   – vector, best for viewing/zooming
  uddl_metamodel_graph.pdf   – print-ready
  uddl_metamodel_graph.png   – raster fallback

Layout
------
Three colour-coded clusters (Conceptual / Logical / Platform) plus a
Common cluster for shared types.  Within each cluster:
  - Solid black arrows  = subclass (rdfs:subClassOf)
  - Coloured arrows     = key object properties (realizes chain, compositions)
  - Dashed grey arrows  = other object properties
  - Data properties     = listed inside each class box
"""

from __future__ import annotations
import os
from owlready2 import get_ontology, ObjectProperty, DataProperty, Thing
from graphviz import Digraph

# ── Config ────────────────────────────────────────────────────────────────────
METAMODEL_PATH = os.path.abspath("./outputs/uddl_metamodel.owl")
OUTPUT_BASE    = "./outputs/uddl_metamodel_graph"

# Layer colours (fill, border)
COLOURS = {
    "conceptual": ("#D6EAF8", "#2980B9"),   # blue
    "logical":    ("#D5F5E3", "#1E8449"),   # green
    "platform":   ("#FDEBD0", "#CA6F1E"),   # orange
    "common":     ("#F5F5F5", "#7F8C8D"),   # grey
    "meta":       ("#EBE9F9", "#6C3483"),   # purple  (DataModel layer)
}

# Map each class name to its cluster
LAYER_MAP = {
    # ── Meta / top-level ─────────────────────────────────────────────────────
    "DataModel":             "meta",
    "ConceptualDataModel":   "meta",
    "LogicalDataModel":      "meta",
    "PlatformDataModel":     "meta",

    # ── Conceptual ───────────────────────────────────────────────────────────
    "UDDL_Element":          "conceptual",
    "NamedElement":          "conceptual",
    "ComposableElement":     "conceptual",
    "AssociationLike":       "conceptual",
    "Characteristic":        "conceptual",
    "ConceptualElement":     "conceptual",
    "BasisEntity":           "conceptual",
    "ConceptualEntity":      "conceptual",
    "ConceptualObservable":  "conceptual",
    "ConceptualComposition": "conceptual",
    "ConceptualParticipant": "conceptual",
    "ConceptualAssociation": "conceptual",
    "ConceptualView":        "conceptual",
    "ConceptualQuery":       "conceptual",
    "ConceptualCompositeQuery": "conceptual",

    # ── Logical ──────────────────────────────────────────────────────────────
    "LogicalElement":             "logical",
    "LogicalEntity":              "logical",
    "LogicalAssociation":         "logical",
    "LogicalComposition":         "logical",
    "LogicalParticipant":         "logical",
    "LogicalMeasurement":         "logical",
    "LogicalValueType":           "logical",
    "LogicalUnit":                "logical",
    "LogicalValueTypeUnit":       "logical",
    "LogicalCoordinateSystem":    "logical",
    "LogicalCoordinateAxis":      "logical",
    "LogicalMeasurementSystem":   "logical",
    "LogicalMeasurementAxis":     "logical",
    "LogicalReferencePoint":      "logical",
    "LogicalLandmark":            "logical",
    "LogicalConstraint":          "logical",
    "LogicalConversion":          "logical",
    "LogicalView":                "logical",
    "LogicalQuery":               "logical",
    "LogicalCompositeQuery":      "logical",

    # ── Platform ─────────────────────────────────────────────────────────────
    "PlatformElement":           "platform",
    "PlatformEntity":            "platform",
    "PlatformAssociation":       "platform",
    "PlatformComposition":       "platform",
    "PlatformParticipant":       "platform",
    "PlatformDataType":          "platform",
    "PlatformPrimitive":         "platform",
    "PlatformStruct":            "platform",
    "PlatformView":              "platform",
    "PlatformQuery":             "platform",
    "PlatformCompositeQuery":    "platform",

    # ── Common supporting ────────────────────────────────────────────────────
    "Cardinality":           "common",
    "PathNode":              "common",
    "Constraint":            "common",
    "RangeConstraint":       "common",
    "EnumerationConstraint": "common",
    "RegexConstraint":       "common",
    "LengthConstraint":      "common",
}

# Object properties to highlight with distinct colours (name → hex colour)
HIGHLIGHT_PROPS = {
    "measurementRealizes":         "#C0392B",  # red   – KEY semantic link
    "platformCompositionRealizes": "#E74C3C",  # red   – KEY semantic link
    "logicalEntityRealizes":       "#922B21",  # dark red
    "platformEntityRealizes":      "#CB4335",  # red
    "realizes":                    "#E59866",  # orange (generic fallback)
    "specializes":                 "#884EA0",  # purple
    "hasConceptualComposition":    "#2471A3",  # blue
    "hasLogicalComposition":       "#1A5276",  # dark blue
    "hasPlatformComposition":      "#1F618D",  # mid blue
    "compositionTarget":           "#17A589",  # teal
    "appliesTo":                   "#148F77",  # dark teal
    "measurementValueType":        "#229954",  # green
    "measurementUnit":             "#1E8449",  # dark green
    "measurementSystem":           "#196F3D",  # very dark green
}

# Properties to skip entirely (too noisy or redundant)
SKIP_PROPS = {
    "hasElement", "hasConceptualModel", "hasLogicalModel", "hasPlatformModel",
    "hasConceptualEntity", "hasConceptualObservable", "hasConceptualAssociation",
    "hasConceptualView", "hasBasisEntity",
    "hasLogicalEntity", "hasLogicalAssociation", "hasLogicalMeasurement",
    "hasLogicalValueType", "hasLogicalUnit", "hasLogicalValueTypeUnit",
    "hasCoordinateSystem", "hasMeasurementSystem",
    "hasPlatformEntity", "hasPlatformAssociation", "hasPlatformDataType",
    "hasPlatformView",
    "hasAxis", "defaultUnit", "coordinateReferencePoint", "hasConversion",
    "hasPathNode", "hasCardinality",
    "hasFieldType",   # redundant with platformCompositionTarget
}

# Data properties to show per class (class → [prop, ...])
SHOW_DATA_PROPS = {
    "NamedElement":          ["hasName", "hasIdentifier", "hasDescription"],
    "Characteristic":        ["hasRoleName", "isOrdered", "isUnique"],
    "Cardinality":           ["lowerBound", "upperBound"],
    "LogicalMeasurement":    ["numericValue", "literalValue", "primitiveDataType"],
    "LogicalValueType":      ["valueTypeName"],
    "LogicalUnit":           ["symbol"],
    "LogicalMeasurementSystem": ["measurementSystemName"],
    "PlatformPrimitive":     ["primitiveName"],
    "PlatformStruct":        ["structName"],
    "DataModel":             ["hasSource", "derivedFromICD", "generationModel"],
    "RangeConstraint":       ["minValue", "maxValue"],
    "EnumerationConstraint": ["allowedLiteral"],
    "RegexConstraint":       ["regexPattern"],
    "LengthConstraint":      ["minLength", "maxLength"],
    "PathNode":              ["pathExpression"],
}


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph(onto) -> Digraph:
    dot = Digraph(
        "UDDL_Metamodel",
        comment="UDDL Metamodel Ontology",
    )
    dot.attr(
        rankdir="TB",
        splines="polyline",
        nodesep="0.5",
        ranksep="1.0",
        fontname="Helvetica",
        bgcolor="white",
        label="UDDL Metamodel Ontology",
        labelloc="t",
        fontsize="22",
        compound="true",
    )
    dot.attr("node", fontname="Helvetica", fontsize="10")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    all_classes = [c for c in onto.classes()]
    class_names = {c.name for c in all_classes}

    # ── Helper: data-props label rows for a class ─────────────────────────
    def data_rows(class_name: str) -> str:
        props = SHOW_DATA_PROPS.get(class_name, [])
        if not props:
            return ""
        rows = "".join(
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#555555">'
            f'  {p}</FONT></TD></TR>'
            for p in props
        )
        return rows

    # ── Clusters ──────────────────────────────────────────────────────────
    cluster_labels = {
        "meta":       "Data Model (top-level)",
        "conceptual": "Conceptual Layer (CDM)",
        "logical":    "Logical Layer (LDM)",
        "platform":   "Platform Layer (PDM)",
        "common":     "Common / Supporting",
    }
    clusters = {k: Digraph(name=f"cluster_{k}") for k in COLOURS}

    for sub in clusters.values():
        sub.attr(style="rounded,filled", fontname="Helvetica", fontsize="13")

    for layer, sub in clusters.items():
        fill, border = COLOURS[layer]
        sub.attr(
            label=cluster_labels[layer],
            fillcolor=fill,
            color=border,
            penwidth="2",
        )

    # ── Nodes ─────────────────────────────────────────────────────────────
    for cls in sorted(all_classes, key=lambda c: c.name):
        name  = cls.name
        layer = LAYER_MAP.get(name, "common")
        fill, border = COLOURS[layer]

        rows = (
            f'<TR><TD BGCOLOR="{border}"><FONT COLOR="white"><B>{name}</B></FONT></TD></TR>'
            + data_rows(name)
        )
        label = f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">{rows}</TABLE>>'

        clusters[layer].node(
            name,
            label=label,
            shape="none",
            margin="0",
        )

    for sub in clusters.values():
        dot.subgraph(sub)

    # ── Subclass edges ────────────────────────────────────────────────────
    for cls in all_classes:
        for parent in cls.is_a:
            if hasattr(parent, "name") and parent.name in class_names and parent.name != cls.name:
                dot.edge(
                    parent.name, cls.name,
                    arrowhead="onormal",
                    color="#444444",
                    penwidth="1.2",
                    style="solid",
                )

    # ── Object property edges ─────────────────────────────────────────────
    for prop in onto.object_properties():
        pname = prop.name
        if pname in SKIP_PROPS:
            continue

        domains = list(prop.domain) if prop.domain else []
        ranges  = list(prop.range)  if prop.range  else []

        for d in domains:
            if not hasattr(d, "name") or d.name not in class_names:
                continue
            for r in ranges:
                if not hasattr(r, "name") or r.name not in class_names:
                    continue

                colour  = HIGHLIGHT_PROPS.get(pname, "#AAAAAA")
                is_key  = pname in HIGHLIGHT_PROPS
                penwidth = "2.2" if is_key else "1.0"
                style    = "solid" if is_key else "dashed"
                fontcolor = colour if is_key else "#888888"

                dot.edge(
                    d.name, r.name,
                    label=pname,
                    color=colour,
                    fontcolor=fontcolor,
                    penwidth=penwidth,
                    style=style,
                    arrowhead="vee",
                    constraint="false" if not is_key else "true",
                )

    # ── Legend ────────────────────────────────────────────────────────────
    with dot.subgraph(name="cluster_legend") as leg:
        leg.attr(label="Legend", style="rounded", color="#BBBBBB", fontsize="11")

        leg.node("leg_subclass", label="subclassOf",
                 shape="plaintext", fontsize="9")
        leg.node("leg_subclass_a", label="", shape="point", width="0")
        leg.node("leg_subclass_b", label="", shape="point", width="0")
        leg.edge("leg_subclass_a", "leg_subclass_b",
                 arrowhead="onormal", color="#444444", style="solid",
                 label="  subclassOf  ", fontsize="9")

        leg.node("leg_realize", label="  realizes chain  ",
                 shape="plaintext", fontsize="9", fontcolor="#C0392B")
        leg.node("leg_prop",    label="  object property  ",
                 shape="plaintext", fontsize="9", fontcolor="#888888")

    return dot


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading metamodel …")
    onto = get_ontology(METAMODEL_PATH).load()

    print("Building graph …")
    dot = build_graph(onto)

    os.makedirs("./outputs", exist_ok=True)

    # SVG
    svg_path = dot.render(OUTPUT_BASE, format="svg", cleanup=True)
    print(f"Saved SVG : {svg_path}")

    # PDF
    pdf_path = dot.render(OUTPUT_BASE, format="pdf", cleanup=True)
    print(f"Saved PDF : {pdf_path}")

    # PNG  (higher DPI via graphviz attribute)
    dot.attr(dpi="180")
    png_path = dot.render(OUTPUT_BASE, format="png", cleanup=True)
    print(f"Saved PNG : {png_path}")


if __name__ == "__main__":
    main()
