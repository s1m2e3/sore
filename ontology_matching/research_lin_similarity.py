
import json
import math
import os
import networkx as nx

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(HERE, "inputs", "CONceptual_ExtractionCategory_Examples", "CONceptual_ExtractionCategory_Examples", "Automobile")
REPORTS_DIR = os.path.join(HERE, "outputs", "reports", "Automobile")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_undirected_graph(data):
    G = nx.Graph()
    entities = {e["entityName"] for e in data["entities"]}
    G.add_nodes_from(entities)
    for assoc in data.get("associations", []):
        name = assoc.get("associationName") or assoc.get("name")
        if name:
            G.add_node(name)
            parts = assoc.get("associationParticipants") or assoc.get("participants", [])
            for p in parts:
                if p in entities:
                    G.add_edge(name, p)
    for ent in data["entities"]:
        parent = ent["entityName"]
        for attr in ent.get("entityAttributes", []):
            child = attr.get("type")
            if child in entities:
                G.add_edge(parent, child)
    return G

def calculate_undirected_ic(G):
    ic_map = {}
    total_nodes = len(G.nodes)
    if total_nodes <= 1: return {n: 1.0 for n in G.nodes}
    max_deg = max(dict(G.degree()).values()) if total_nodes > 0 else 1
    for node, deg in G.degree():
        ic_map[node] = 1 - (math.log(deg + 1) / math.log(max_deg + 2))
    return ic_map

def get_unmatched_near_anchor(anchor, G, anchors_list, max_dist=2):
    """Find all unmatched entities near a matched anchor."""
    found = []
    visited = {anchor}
    queue = [(anchor, 0)]
    while queue:
        curr, dist = queue.pop(0)
        if dist >= max_dist: continue
        for nb in G.neighbors(curr):
            if nb not in visited:
                visited.add(nb)
                # If it's not an anchor, it's a candidate
                if nb not in anchors_list:
                    found.append((nb, dist + 1))
                queue.append((nb, dist + 1))
    return found

# 1. Load Data
v1_data = load_json(os.path.join(INPUTS_DIR, "automobile_model_v1.json"))
v2_data = load_json(os.path.join(INPUTS_DIR, "automobile_model_v2.json"))
report = load_json(os.path.join(REPORTS_DIR, "Automobile_Model_V1_SystemCentric_vs_Automobile_Model_V2_ComponentCentric.json"))

# 2. Build Graphs
G1 = build_undirected_graph(v1_data)
G2 = build_undirected_graph(v2_data)

# 3. IC
ic1 = calculate_undirected_ic(G1)
ic2 = calculate_undirected_ic(G2)

# 4. Anchors
anchors_v1_to_v2 = {} 
for ent in report["model_a"]["entities"]:
    if ent["status"] == "matched":
        target = ent.get("matched_to") or ent.get("match") or ent.get("matchedTo")
        if target:
            anchors_v1_to_v2[ent["name"]] = target

anchors_v1 = set(anchors_v1_to_v2.keys())
anchors_v2 = set(anchors_v1_to_v2.values())

# 5. Semantic Neighborhood Discovery
print("=== Lin-Structural Discovery (Expanded Neighborhood) ===")
print(f"{'V1 Unmatched':<20} | {'V2 Unmatched':<20} | {'Lin-IC Score'} | {'Anchor Basis'}")
print("-" * 90)

results = []
# Iterate through each anchor pair
for v1_anc, v2_anc in anchors_v1_to_v2.items():
    # Find unmatched things near v1_anc
    near_v1 = get_unmatched_near_anchor(v1_anc, G1, anchors_v1)
    # Find unmatched things near v2_anc
    near_v2 = get_unmatched_near_anchor(v2_anc, G2, anchors_v2)
    
    for u1, d1 in near_v1:
        for u2, d2 in near_v2:
            # Lin Similarity logic: (2 * IC(LCA)) / (IC(u1) + IC(u2))
            # Here LCA is the anchor pair context
            ic_lca = (ic1[v1_anc] + ic2[v2_anc]) / 2
            denom = ic1[u1] + ic2[u2]
            lin_sim = (2 * ic_lca / denom) if denom > 0 else 0
            # Weight by distance? (closer to anchor = more certain)
            score = lin_sim / (d1 + d2) 
            results.append((u1, u2, score, v1_anc, lin_sim))

# Show top unique pairs
seen = set()
for r in sorted(results, key=lambda x: x[2], reverse=True):
    pair = (r[0], r[1])
    if pair not in seen:
        print(f"{r[0]:<20} | {r[1]:<20} | {r[4]:.4f}       | {r[3]}")
        seen.add(pair)
    if len(seen) >= 20: break
