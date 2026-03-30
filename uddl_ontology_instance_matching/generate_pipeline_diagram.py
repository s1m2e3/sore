"""
generate_pipeline_diagram.py
-----------------------------
Renders the UDDL message comparison pipeline as a vector diagram (SVG + PDF).
Output: outputs/pipeline_diagram.svg and outputs/pipeline_diagram.pdf
"""

import os
from graphviz import Digraph

def build_pipeline_diagram() -> Digraph:
    dot = Digraph(
        "UDDL_Pipeline",
        comment="UDDL Message Comparison Pipeline",
    )
    dot.attr(
        rankdir="TB",
        splines="polyline",
        nodesep="0.6",
        ranksep="0.8",
        fontname="Helvetica",
        bgcolor="white",
        label="UDDL Message Comparison Pipeline\n(ICD Fidelity Evaluation Framework)",
        labelloc="t",
        fontsize="18",
        pad="0.5",
    )
    dot.attr("node", fontname="Helvetica", fontsize="11")
    dot.attr("edge", fontname="Helvetica", fontsize="10")

    # ── Colour palette ──────────────────────────────────────────────────────
    C_ICD      = ("#FDEBD0", "#CA6F1E")   # orange  – source documents
    C_UDDL     = ("#D6EAF8", "#2980B9")   # blue    – UDDL JSON inputs
    C_OWL      = ("#D5F5E3", "#1E8449")   # green   – OWL instance graphs
    C_LIMES    = ("#EBE9F9", "#6C3483")   # purple  – LIMES lexical
    C_MNLI     = ("#FDEDEC", "#C0392B")   # red     – MNLI semantic
    C_ALIGN    = ("#FDFEFE", "#626567")   # grey    – alignment
    C_METRICS  = ("#F9F9F9", "#1A252F")   # dark    – output metrics
    C_STEP     = ("#F8F9FA", "#444444")   # light   – step boxes

    def node(name, label, fill, border, shape="box", style="filled,rounded"):
        dot.node(name, label=label, shape=shape, style=style,
                 fillcolor=fill, color=border, penwidth="1.8")

    def step_label(n, text):
        return f"Step {n}\n{text}"

    # ── ICD source documents (top) ──────────────────────────────────────────
    with dot.subgraph(name="cluster_inputs") as c:
        c.attr(label="Source ICDs (Input Documents)", style="rounded,dashed",
               color="#AAAAAA", fontsize="12", fontcolor="#555555")
        node("icd1", "ICD 1\n(Reference / Human-authored)",
             C_ICD[0], C_ICD[1], shape="folder")
        node("icd2", "ICD 2\n(LLM-parsed / Auto-generated)",
             C_ICD[0], C_ICD[1], shape="folder")

    # ── LLM Parser ─────────────────────────────────────────────────────────
    node("llm_parser", "LLM ICD Parser\n(GPT-4 / Claude / etc.)",
         "#FFFDE7", "#F9A825", shape="ellipse")

    dot.edge("icd1", "llm_parser", label="parse")
    dot.edge("icd2", "llm_parser", label="parse", style="dashed")

    # ── UDDL JSON inputs ────────────────────────────────────────────────────
    with dot.subgraph(name="cluster_json") as c:
        c.attr(label="UDDL Layered JSON (pipeline input)", style="rounded,dashed",
               color="#AAAAAA", fontsize="12", fontcolor="#555555")
        node("json_a", "msg_A.json\n(Conceptual / Logical / Platform)",
             C_UDDL[0], C_UDDL[1])
        node("json_b", "msg_B.json\n(Conceptual / Logical / Platform)",
             C_UDDL[0], C_UDDL[1])

    dot.edge("llm_parser", "json_a", label=" ICD1 UDDL")
    dot.edge("llm_parser", "json_b", label=" ICD2 UDDL", style="dashed")

    # ── Step 2: JSON -> OWL ─────────────────────────────────────────────────
    node("step2", step_label(2, "json_to_uddl_owl.py\nJSON -> OWL Instance Graph"),
         C_STEP[0], C_STEP[1])

    dot.edge("json_a", "step2", label=" input A")
    dot.edge("json_b", "step2", label=" input B", style="dashed")

    # ── OWL instance graphs ─────────────────────────────────────────────────
    with dot.subgraph(name="cluster_owl") as c:
        c.attr(label="OWL Instance Graphs (UDDL Metamodel)", style="rounded,filled",
               fillcolor="#F0FFF4", color="#1E8449", fontsize="12")
        node("owl_a", "OWL Graph A\nmsg_A.owl\n(ConceptualObservable /\nLogicalMeasurement /\nPlatformComposition)",
             C_OWL[0], C_OWL[1])
        node("owl_b", "OWL Graph B\nmsg_B.owl\n(ConceptualObservable /\nLogicalMeasurement /\nPlatformComposition)",
             C_OWL[0], C_OWL[1])

    dot.edge("step2", "owl_a")
    dot.edge("step2", "owl_b", style="dashed")

    # ── Step 3: NT conversion ───────────────────────────────────────────────
    node("step3", step_label(3, "rdflib\nOWL -> N-Triples (.nt)"),
         C_STEP[0], C_STEP[1])
    dot.edge("owl_a", "step3")
    dot.edge("owl_b", "step3", style="dashed")

    # ── Step 4+5: LIMES lexical ─────────────────────────────────────────────
    with dot.subgraph(name="cluster_limes") as c:
        c.attr(label="Lexical Matching (Step 4-5)", style="rounded,filled",
               fillcolor="#F5F0FF", color="#6C3483", fontsize="12")
        node("step4", step_label(4, "generate_limes_config()\nDynamic XML config"),
             C_LIMES[0], C_LIMES[1])
        node("step5", step_label(5, "LIMES 1.7.9 (Java 11)\nTrigram / Jaccard / Cosine"),
             C_LIMES[0], C_LIMES[1])

    dot.edge("step3", "step4", label=" .nt files")
    dot.edge("step4", "step5", label=" config.xml")

    # ── Step 6: MNLI semantic ───────────────────────────────────────────────
    with dot.subgraph(name="cluster_mnli") as c:
        c.attr(label="Semantic Matching (Step 6)", style="rounded,filled",
               fillcolor="#FFF5F5", color="#C0392B", fontsize="12")
        node("step6", step_label(6, "MNLI Entailment\n(facebook/bart-large-mnli)\nGreedy 1-to-1 alignment"),
             C_MNLI[0], C_MNLI[1])

    dot.edge("step3", "step6", label=" .ttl files", constraint="false",
             style="dashed", color="#999999")

    # ── Alignment candidates ────────────────────────────────────────────────
    node("alignment", "Candidate Alignments\n(sameAs pairs + scores)\nper matcher",
         C_ALIGN[0], C_ALIGN[1], shape="diamond")

    dot.edge("step5", "alignment", label=" .nt alignment")
    dot.edge("step6", "alignment", label=" scored pairs")

    # ── Optional ground truth ───────────────────────────────────────────────
    node("gt", "Ground Truth\n(optional JSON)\n{src_name, tgt_name}",
         "#FEFEFE", "#AAAAAA", shape="note", style="filled")

    dot.edge("gt", "alignment", style="dotted", color="#AAAAAA",
             label=" if available", arrowhead="none")

    # ── Step 7: Metrics output ──────────────────────────────────────────────
    with dot.subgraph(name="cluster_output") as c:
        c.attr(label="Comparison Metrics (Output)", style="rounded,filled",
               fillcolor="#F8F9FA", color="#1A252F", fontsize="12")
        node("metrics", "evaluate_matching.py\nPrecision / Recall / F1\nper matcher + per layer",
             C_METRICS[0], C_METRICS[1])
        node("csv", "results_A_vs_B.csv\n(full per-field analysis)",
             "#EAFAF1", "#1E8449", shape="note")
        node("report", "Research Metrics\n(observable / measurement /\nplatform divergence)",
             "#F4ECF7", "#6C3483", shape="note")

    dot.edge("alignment", "metrics")
    dot.edge("metrics", "csv")
    dot.edge("metrics", "report")

    # ── Metamodel anchor ────────────────────────────────────────────────────
    node("metamodel", "UDDL Metamodel\n(base_ontology.py)\nuddl_metamodel.owl",
         "#FEF9E7", "#F39C12", shape="cylinder")

    dot.edge("metamodel", "step2", label=" schema", style="dotted",
             color="#F39C12", constraint="false")

    return dot


def main():
    os.makedirs("./outputs", exist_ok=True)
    dot = build_pipeline_diagram()

    svg_path = dot.render("./outputs/pipeline_diagram", format="svg", cleanup=True)
    print(f"Saved SVG: {svg_path}")

    pdf_path = dot.render("./outputs/pipeline_diagram", format="pdf", cleanup=True)
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
