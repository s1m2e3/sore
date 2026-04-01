
"""
visualize_semantic_space.py
--------------------------
Visualizes the "Semantic Footprint" of ontologies in a 2D embedding space.

Logic
-----
1. Two Encoding Strategies (Subplots):
   - Row 1: Root Word (First token only).
   - Row 2: Summed (Sum of all tokens).
2. Shared X-Axis: Allows direct vertical comparison of semantic shifting.
3. Clean Density Lines: 2D Contour lines color-matched to ontologies (no fill).
4. Consistent Reference: Grid labels from Strategy A are mirrored in Strategy B.

Usage
-----
    cd ontology_matching
    python visualize_semantic_space.py --domain Automobile
"""

import argparse
import glob
import json
import os
import re
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "inputs", "CONceptual_ExtractionCategory_Examples", "CONceptual_ExtractionCategory_Examples")
VIZ_DIR = os.path.join(BASE_DIR, "outputs", "visualizations", "semantic_space")

def load_domain_data(domain):
    pattern = os.path.join(INPUTS_DIR, domain, "*.json")
    files = glob.glob(pattern)
    data = {}
    for f in files:
        with open(f, "r", encoding="utf-8") as j:
            d = json.load(j)
            name = d.get("modelName") or os.path.basename(f)
            data[name] = d
    return data

def _tokenize(name: str) -> list[str]:
    if not name: return []
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return s.split()

def extract_terms(model_data):
    terms = []
    for ent in model_data.get("entities", []):
        ename = ent.get("entityName") or ent.get("name")
        tokens = _tokenize(ename)
        terms.append({"name": ename, "tokens": tokens, "kind": "Entity"})
        attrs = ent.get("entityAttributes") or ent.get("attributes", [])
        for a in attrs:
            aname = a.get("name")
            atokens = _tokenize(aname)
            terms.append({"name": aname, "tokens": atokens, "kind": "Attribute", "parent": ename})
    return terms

def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{opacity})"

def main():
    parser = argparse.ArgumentParser(description="Visualize Semantic Space")
    parser.add_argument("--domain", type=str, default="Automobile")
    parser.add_argument("--model", type=str, default="paraphrase-MiniLM-L6-v2")
    args = parser.parse_args()

    print(f"--- Loading Domain: {args.domain} ---")
    domain_data = load_domain_data(args.domain)
    if not domain_data: return

    print(f"--- Encoding with {args.model} ---")
    embedder = SentenceTransformer(args.model)
    
    WORDNET_PRIMARIES = ["act", "animal", "artifact", "attribute", "body", "cognition", "communication", "event", "feeling", "food", "group", "location", "motive", "object", "person", "phenomenon", "plant", "possession", "process", "quantity", "relation", "shape", "state", "substance", "time"]
    primary_embeddings = embedder.encode(WORDNET_PRIMARIES)
    ref_centroid = np.mean(primary_embeddings, axis=0)
    norm_ref = np.linalg.norm(ref_centroid)

    all_records = []
    for model_name, data in domain_data.items():
        for t in extract_terms(data):
            t["model"] = model_name
            all_records.append(t)

    print("--- Generating Dual Encodings ---")
    unique_tokens = sorted(list(set(tok for r in all_records for tok in r["tokens"])))
    token_map = dict(zip(unique_tokens, embedder.encode(unique_tokens, show_progress_bar=True)))

    def process_mode(mode):
        mode_embs = []
        for r in all_records:
            toks = r["tokens"]
            if mode == "root":
                vec = token_map.get(toks[0], np.zeros(384)) if toks else np.zeros(384)
            else: # summed
                vec = np.sum([token_map.get(t, np.zeros(384)) for t in toks], axis=0) if toks else np.zeros(384)
            mode_embs.append(vec)
        mode_embs = np.array(mode_embs)
        cos = np.dot(mode_embs, ref_centroid) / (np.linalg.norm(mode_embs, axis=1) * norm_ref)
        gen = (cos - np.min(cos)) / (np.max(cos) - np.min(cos)) if np.max(cos) > np.min(cos) else np.ones_like(cos)
        coords = PCA(n_components=2).fit_transform(mode_embs)
        return coords, gen

    coords_root, gen_root = process_mode("root")
    coords_sum, gen_sum = process_mode("sum")

    def get_grid_idx(coords):
        indices = []
        for x in range(-6, 7, 2):
            for y in range(-6, 7, 2):
                in_cell = [ (i, (c[0]-x)**2 + (c[1]-y)**2) for i, c in enumerate(coords) if x-1 <= c[0] < x+1 and y-1 <= c[1] < y+1 ]
                if in_cell: indices.append(min(in_cell, key=lambda x: x[1])[0])
        return indices

    g_idx = get_grid_idx(coords_root)

    # ── SUBPLOT VISUALIZATION ───────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Strategy A: Root Word (First Token Only)", "Strategy B: Summed Meanings (All Tokens)")
    )

    models = sorted(list(domain_data.keys()))
    kinds = ["Entity", "Attribute"]
    symbols = {"Entity": "circle", "Attribute": "triangle-up"}
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

    def add_mode_to_subplot(row, coords, gen, show_legend_base):
        sizes = 6 + (gen**2 * 15)
        for i, model in enumerate(models):
            m_idx = [j for j, r in enumerate(all_records) if r["model"] == model]
            if not m_idx: continue
            
            # 1. Density (COLORED LINES ONLY, 10 LEVELS, THICKER)
            fig.add_trace(go.Histogram2dContour(
                x=coords[m_idx, 0], y=coords[m_idx, 1],
                colorscale=[[0, "rgba(0,0,0,0)"], [1, colors[i % len(colors)]]],
                opacity=0.7, showscale=False, ncontours=10, 
                contours=dict(coloring='lines'), # Line mode
                line=dict(width=2.5), # Thicker lines for visibility
                legendgroup=model, showlegend=False, hoverinfo="skip"
            ), row=row, col=1)
            
            # 2. Scatter
            for kind in kinds:
                k_idx = [j for j in m_idx if all_records[j]["kind"] == kind]
                if not k_idx: continue
                fig.add_trace(go.Scatter(
                    x=coords[k_idx, 0], y=coords[k_idx, 1], mode="markers",
                    name=f"{model}", legendgroup=model, 
                    showlegend=(show_legend_base and kind == "Entity"),
                    marker=dict(size=sizes[k_idx], color=colors[i % len(colors)], symbol=symbols[kind], opacity=0.8, line=dict(width=0.8, color="white")),
                    text=[f"{all_records[j]['name']}" for j in k_idx],
                    customdata=[[gen[j], all_records[j].get("parent", "N/A")] for j in k_idx],
                    hovertemplate="<b>%{text}</b><br>Generality: %{customdata[0]:.4f}<extra></extra>"
                ), row=row, col=1)
        
        # 3. Add Reference Grid Labels
        fig.add_trace(go.Scatter(
            x=coords[g_idx, 0], y=coords[g_idx, 1], mode="text",
            text=[all_records[j]["tokens"][0] for j in g_idx],
            textfont=dict(size=11, color="black", family="Arial Black"),
            showlegend=False, hoverinfo="skip"
        ), row=row, col=1)

    add_mode_to_subplot(1, coords_root, gen_root, show_legend_base=True)
    add_mode_to_subplot(2, coords_sum, gen_sum, show_legend_base=False)

    fig.update_layout(
        title=f"Semantic Abstraction Landscape: {args.domain}<br><sup>Contours show ontology concentrations (Colored Lines); Text tracks core concepts across strategies</sup>",
        template="plotly_white",
        height=1100,
        legend_title="Ontology (Click to Toggle)",
        margin=dict(t=120, b=50)
    )

    os.makedirs(VIZ_DIR, exist_ok=True)
    out_path = os.path.join(VIZ_DIR, f"{args.domain}_semantic_space.html")
    fig.write_html(out_path)
    print(f"\nSaved Visualization -> {out_path}")

if __name__ == "__main__":
    main()
