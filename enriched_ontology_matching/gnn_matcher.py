"""
gnn_matcher.py
--------------
Symmetric GNN-based ontology similarity using canonical associations from
association_inventory.csv.

Relationship to owl_builder.py
-------------------------------
owl_builder.py handles the *structural* bidirectionality case:
  Ontology A: A --UsedFor--> B
  Ontology B: B --IsUsedBy--> A      ← same, expressed in reverse
  OWL owl:inverseOf lets a reasoner equate these automatically.

gnn_matcher.py handles the *semantic* bidirectionality case:
  Ontology A: car --ReceivesAction--> human   (car is driven by human)
  Ontology B: driver --UsedFor--> car         (driver uses car)
  These share the same entity pair (car≈car, human≈driver) and a related
  canonical type — but no owl:inverseOf links ReceivesAction ↔ UsedFor.
  The GNN captures this because the undirected graph produces the same
  neighbourhood signal regardless of which direction the edge was written.

Graph representation
--------------------
  Nodes : entity names  →  sentence_embed(entity_name)
  Edges : undirected, labeled by "{verb_tokens} {canonical}"
          →  sentence_embed(edge_phrase)

  Both A→B and B→A adjacency entries are created for every association,
  so K-hop aggregation is fully symmetric.

Usage
-----
  python enriched_ontology_matching/gnn_matcher.py \\
      --a  enriched_ontology_matching/test_auto_v1_v2.json --key-a json_a \\
      --b  enriched_ontology_matching/test_auto_v1_v2.json --key-b json_b \\
      --hops 2 --top-pairs 20
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from model_normalizer import load_inventory as _norm_load_inventory, normalize_model

# ── Paths ────────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_INVENTORY = _REPO_ROOT / "enriched_ontology_matching" / "association_inventory.csv"

DEFAULT_ENCODER = "paraphrase-MiniLM-L6-v2"

# ── Sentence encoder singleton ───────────────────────────────────────────────────
_ENCODER       = None
_ENCODER_MODEL = None


def _get_encoder(model_name: str = DEFAULT_ENCODER):
    global _ENCODER, _ENCODER_MODEL
    if _ENCODER is None or _ENCODER_MODEL != model_name:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[GNN] Loading {model_name} on {device} ...", file=sys.stderr)
            _ENCODER       = SentenceTransformer(model_name, device=device)
            _ENCODER_MODEL = model_name
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install: pip install sentence-transformers"
            )
    return _ENCODER


def _encode(texts: list[str], model_name: str = DEFAULT_ENCODER) -> np.ndarray:
    """Batch-encode, returning L2-normalised rows of shape (N, d)."""
    enc = _get_encoder(model_name)
    return enc.encode(
        texts, batch_size=64, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    )


def _l2_norm_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.where(norms > 1e-10, arr / norms, arr)


# ── Inventory loading ────────────────────────────────────────────────────────────

def load_inventory(path: Path = _INVENTORY) -> dict[str, dict]:
    """
    Load association_inventory.csv → {assoc_name: {canonical, verb_tokens, ...}}.

    The inventory is the normalised lower set of canonical associations.
    We use it here instead of raw JSON associations so both models in a
    comparison are judged against the same reduced relation vocabulary.
    """
    inventory: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name     = row["association_name"]
            verb_raw = row.get("verb_tokens", "")
            inventory[name] = {
                "canonical":      row["canonical"],
                "score":          float(row["score"]) if row["score"] else 0.0,
                "verb_tokens":    [t.strip() for t in verb_raw.split("|") if t.strip()],
                "low_confidence": row.get("low_confidence", "") == "YES",
            }
    return inventory


# ── Edge phrase ───────────────────────────────────────────────────────────────────

def _edge_phrase(verb_tokens: list[str], canonical: str) -> str:
    """Combine verb tokens and canonical into a single embeddable phrase."""
    return (" ".join(verb_tokens) + " " + canonical).strip()


# ── OntologyGraph ─────────────────────────────────────────────────────────────────

class OntologyGraph:
    """
    Undirected multigraph for one ontology model.

    Construction
    ------------
    Participants come from the model JSON (tied to the actual model entities).
    Canonical relation + verb tokens come from the inventory CSV (normalised).

    Both A→B and B→A adjacency entries are inserted for each association,
    making aggregation direction-agnostic.  This is the core mechanism:

      "car  ReceivesAction  human"  →  edge(car, human, "driven ReceivesAction")
      "driver  UsedFor  car"        →  edge(driver, car, "uses UsedFor")

    After K-hop symmetric message passing:
      • car's embedding encodes "I connect to a human-like entity via a
        ReceivesAction-type relation"
      • car's embedding in the second ontology encodes "I connect to a
        driver-like entity via a UsedFor-type relation"

    These two are similar when human≈driver (sentence embeddings) and when
    the edge phrases have overlapping canonical type.
    """

    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self.entities:    list[str]                         = []
        self._ent_idx:    dict[str, int]                    = {}
        # Each entry: (node_i, node_j, edge_phrase)  — one entry per raw association
        self.edge_list:   list[tuple[int, int, str]]        = []
        # Both directions: adj[i] = [(j, edge_idx), ...]
        self.adj:         dict[int, list[tuple[int, int]]]  = defaultdict(list)
        # Observable/measurable attribute types per entity (populated from entityAttributes)
        self.entity_obs:  dict[str, list[str]]              = {}

        self._node_embs:  Optional[np.ndarray] = None   # (N, d)
        self._edge_embs:  Optional[np.ndarray] = None   # (M, d)
        self._graph_emb:  Optional[np.ndarray] = None   # (d,)  cached

    # ── construction ──────────────────────────────────────────────────────────

    def _add_entity(self, name: str) -> int:
        if name not in self._ent_idx:
            idx = len(self.entities)
            self.entities.append(name)
            self._ent_idx[name] = idx
            self.adj[idx] = []
        return self._ent_idx[name]

    def _add_edge(self, i: int, j: int, phrase: str) -> None:
        """Register one undirected edge (creates both A→B and B→A adjacency)."""
        eidx = len(self.edge_list)
        self.edge_list.append((i, j, phrase))
        self.adj[i].append((j, eidx))
        self.adj[j].append((i, eidx))   # ← bidirectional adjacency

    @classmethod
    def from_model_json(
        cls,
        model_json:          dict,
        inventory:           dict[str, dict],
        skip_low_confidence: bool = False,
    ) -> "OntologyGraph":
        """
        Build an OntologyGraph from a parsed model JSON.

        The model is normalized first (canonical annotation + composition→PartOf
        synthesis) so the graph includes composition edges from V-models and uses
        canonical relation types for all edge phrases.
        """
        # Derive simple name→canonical map from full inventory for normalization
        simple_inv = {name: info["canonical"] for name, info in inventory.items()}
        model_json = normalize_model(model_json, simple_inv)

        g = cls(model_json.get("modelName", ""))

        # Collect entity names so we can distinguish entity-type references from
        # observable types in entityAttributes (same logic as model_normalizer).
        entity_name_set: set[str] = {
            e.get("entityName") or e.get("name", "")
            for e in model_json.get("entities", [])
        }
        entity_name_set.discard("")

        for ent in model_json.get("entities", []):
            name = ent.get("entityName") or ent.get("name") or ""
            if not name:
                continue
            g._add_entity(name)
            # Collect observable/measurable attribute types (exclude entity references)
            obs = [
                attr["type"]
                for attr in (ent.get("entityAttributes") or ent.get("attributes") or [])
                if attr.get("type") and attr["type"] not in entity_name_set
            ]
            if obs:
                g.entity_obs[name] = obs

        for assoc in model_json.get("associations", []):
            assoc_name   = assoc.get("associationName") or assoc.get("name") or ""
            participants = (
                assoc.get("associationParticipants")
                or assoc.get("participants")
                or []
            )
            if not assoc_name or len(participants) < 2:
                continue

            # Use the canonical field set by normalize_model; fall back to
            # full inventory lookup for verb_tokens (not stored by normalizer).
            canonical = assoc.get("canonical", "RelatedTo")
            if assoc_name in inventory:
                inv = inventory[assoc_name]
                if skip_low_confidence and inv.get("low_confidence"):
                    continue
                verb_tokens = inv["verb_tokens"]
            else:
                verb_tokens = []

            phrase = _edge_phrase(verb_tokens, canonical)

            for a_idx in range(len(participants) - 1):
                for b_idx in range(a_idx + 1, len(participants)):
                    ea, eb = participants[a_idx], participants[b_idx]
                    i = g._add_entity(ea)
                    j = g._add_entity(eb)
                    if i != j:
                        g._add_edge(i, j, phrase)

        return g

    # ── encoding ──────────────────────────────────────────────────────────────

    def encode(self, model_name: str = DEFAULT_ENCODER) -> None:
        """
        Sentence-encode all entity names and unique edge phrases in one batch.

        For entities that carry observable/measurable attribute types (V-models),
        the encoding text is enriched: "{entity_name} {obs1} {obs2} ...".
        This injects the attribute signature directly into the node embedding so
        that message passing can propagate attribute-informed representations.
        Net-model entities with no attributes use only their name (unchanged).
        """
        if not self.entities:
            self._node_embs = np.zeros((0, 384))
            self._edge_embs = np.zeros((0, 384))
            return

        entity_texts = [
            _readable(e) + (
                " " + " ".join(_readable(o) for o in self.entity_obs[e])
                if e in self.entity_obs else ""
            )
            for e in self.entities
        ]
        unique_phrases  = list({ep for _, _, ep in self.edge_list} or {"none"})
        all_embs        = _encode(entity_texts + unique_phrases, model_name)

        self._node_embs = all_embs[:len(entity_texts)]
        phrase_map      = {p: all_embs[len(entity_texts) + i]
                           for i, p in enumerate(unique_phrases)}
        self._edge_embs = (
            np.stack([phrase_map[ep] for _, _, ep in self.edge_list])
            if self.edge_list
            else np.zeros((0, all_embs.shape[1]))
        )

    # ── symmetric message passing ──────────────────────────────────────────────

    def message_pass(self, K: int = 2) -> np.ndarray:
        """
        K-hop symmetric message passing.

        Each hop: h_i ← L2_norm( mean( [h_i] + [(h_j + edge_ij)/2  ∀j∈N(i)] ) )

        Because adj[i] contains *both* outgoing and incoming neighbours
        (we added both directions at construction time), this aggregation
        is fully symmetric: the graph topology looks the same from either
        end of an association.

        Returns updated node embeddings (N, d), L2-normalised per row.
        """
        if self._node_embs is None:
            raise RuntimeError("Call encode() before message_pass().")
        h = self._node_embs.copy()
        E = self._edge_embs
        d = h.shape[1] if len(h) > 0 else 384

        for _ in range(K):
            h_new = np.zeros_like(h)
            for i in range(len(self.entities)):
                msgs = [h[i]]          # self-loop keeps identity information
                for j, eidx in self.adj.get(i, []):
                    ev = E[eidx] if eidx < len(E) else np.zeros(d)
                    # Edge-aware message: average of neighbour embedding + edge type
                    msgs.append((h[j] + ev) * 0.5)
                h_new[i] = np.mean(msgs, axis=0)
            h = _l2_norm_rows(h_new)
        return h

    def embedding(self, K: int = 2) -> np.ndarray:
        """
        Graph-level L2-normalised embedding: mean-pool over post-GNN node embeddings.
        Result is cached after the first call.
        """
        if self._graph_emb is not None:
            return self._graph_emb
        h   = self.message_pass(K)
        emb = np.mean(h, axis=0) if len(h) > 0 else np.zeros(384)
        n   = np.linalg.norm(emb)
        self._graph_emb = emb / n if n > 1e-10 else emb
        return self._graph_emb


# ── Similarity functions ──────────────────────────────────────────────────────────

def graph_similarity(
    ga: OntologyGraph,
    gb: OntologyGraph,
    K:  int = 2,
    model_name: str = DEFAULT_ENCODER,
) -> float:
    """
    Cosine similarity between the two graph-level embeddings.

    Undirected graph construction + symmetric message passing ensures that
    inverse-expressed associations contribute the same neighbourhood signal,
    so this score is high when two graphs encode the same relational structure
    regardless of edge direction conventions.
    """
    for g in (ga, gb):
        if g._node_embs is None:
            g.encode(model_name)
    return float(np.dot(ga.embedding(K), gb.embedding(K)))


def entity_pair_similarities(
    ga: OntologyGraph,
    gb: OntologyGraph,
    K:  int = 2,
    model_name: str = DEFAULT_ENCODER,
    top_k: int = 20,
) -> list[dict]:
    """
    Top-K entity pairs ranked by post-GNN cosine similarity.

    After K hops, each entity embedding encodes both its name and its
    relational neighbourhood (canonical types + connected entity types).
    Cross-ontology pairs score high when both dimensions agree — even when
    associations were expressed in opposite directions in the two ontologies.

    Returns list of {entity_a, entity_b, sim} sorted descending.
    """
    for g in (ga, gb):
        if g._node_embs is None:
            g.encode(model_name)

    ha = ga.message_pass(K)   # (Na, d)
    hb = gb.message_pass(K)   # (Nb, d)
    # Both are L2-normalised per row → dot product = cosine similarity
    sim_matrix = ha @ hb.T    # (Na, Nb)

    pairs = [
        {
            "entity_a": ga.entities[i],
            "entity_b": gb.entities[j],
            "sim":      round(float(sim_matrix[i, j]), 4),
        }
        for i in range(len(ga.entities))
        for j in range(len(gb.entities))
    ]
    pairs.sort(key=lambda x: -x["sim"])
    return pairs[:top_k]


# ── Pipeline-stage helper ────────────────────────────────────────────────────────

def run_gnn_stage(
    data_a: dict,
    data_b: dict,
    matches_csv: Path,
    out_csv: Path,
    K: int = 2,
    model_name: str = DEFAULT_ENCODER,
    skip_low_confidence: bool = False,
) -> list[dict]:
    """
    Compute per-pair GNN entity similarity for every (entity_a, entity_b) row in
    matches_csv and write the result to out_csv.

    Builds attribute-enriched OntologyGraph for both models, runs K-hop symmetric
    message passing, then scores each pair by cosine similarity of post-GNN node
    embeddings.  Pairs whose entities are absent from either graph receive an
    empty gnn_sim value.
    """
    inventory = load_inventory()

    ga = OntologyGraph.from_model_json(data_a, inventory, skip_low_confidence)
    gb = OntologyGraph.from_model_json(data_b, inventory, skip_low_confidence)

    print(f"[GNN] {ga.model_name}: {len(ga.entities)} ents, {len(ga.edge_list)} edges",
          file=sys.stderr)
    print(f"[GNN] {gb.model_name}: {len(gb.entities)} ents, {len(gb.edge_list)} edges",
          file=sys.stderr)

    ga.encode(model_name)
    gb.encode(model_name)

    ha = ga.message_pass(K)   # (Na, d) — L2-normalised
    hb = gb.message_pass(K)   # (Nb, d) — L2-normalised
    sim_matrix = ha @ hb.T    # (Na, Nb), dot product = cosine

    idx_a = {e: i for i, e in enumerate(ga.entities)}
    idx_b = {e: i for i, e in enumerate(gb.entities)}

    with open(matches_csv, newline="", encoding="utf-8") as fh:
        match_rows = list(csv.DictReader(fh))

    results = []
    for row in match_rows:
        ea, eb = row["entity_a"], row["entity_b"]
        if ea in idx_a and eb in idx_b:
            sim = round(float(sim_matrix[idx_a[ea], idx_b[eb]]), 4)
        else:
            sim = ""
        results.append({"entity_a": ea, "entity_b": eb, "gnn_sim": sim})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["entity_a", "entity_b", "gnn_sim"])
        w.writeheader()
        w.writerows(results)

    print(f"[GNN] {len(results)} pairs -> {out_csv}", file=sys.stderr)
    return results


# ── Utilities ─────────────────────────────────────────────────────────────────────

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _readable(name: str) -> str:
    return _CAMEL_RE.sub(" ", name).lower()


def _load_model(path: Path, key: Optional[str]) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if key and key in data:
        return data[key]
    # Auto-detect: if wrapper format, default to json_a
    if "json_a" in data and key != "json_b":
        return data["json_a"]
    return data


# ── CLI ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "GNN ontology similarity — undirected graph + symmetric message passing.\n"
            "Handles semantic bidirectionality that OWL owl:inverseOf cannot capture."
        )
    )
    ap.add_argument("--a",       required=True, help="Ontology A JSON path")
    ap.add_argument("--b",       required=True, help="Ontology B JSON path")
    ap.add_argument("--key-a",   default=None,  help="Sub-key in A JSON (e.g. json_a)")
    ap.add_argument("--key-b",   default=None,  help="Sub-key in B JSON (e.g. json_b)")
    ap.add_argument("--inventory", default=str(_INVENTORY))
    ap.add_argument("--hops",      type=int, default=2,
                    help="Message-passing hops (default: 2)")
    ap.add_argument("--top-pairs", type=int, default=20)
    ap.add_argument("--skip-low-confidence", action="store_true")
    args = ap.parse_args()

    inventory = load_inventory(Path(args.inventory))
    print(f"Loaded {len(inventory)} canonical associations.")

    data_a = _load_model(Path(args.a), args.key_a)
    data_b = _load_model(Path(args.b), args.key_b)

    ga = OntologyGraph.from_model_json(data_a, inventory, args.skip_low_confidence)
    gb = OntologyGraph.from_model_json(data_b, inventory, args.skip_low_confidence)

    print(f"\n{ga.model_name}: {len(ga.entities)} entities, {len(ga.edge_list)} edges")
    print(f"{gb.model_name}: {len(gb.entities)} entities, {len(gb.edge_list)} edges")

    print("\nEncoding ...")
    ga.encode()
    gb.encode()

    sim = graph_similarity(ga, gb, K=args.hops)
    print(f"\nGraph similarity ({args.hops}-hop GNN, cosine): {sim:.4f}")

    print(f"\nTop-{args.top_pairs} entity pairs by post-GNN similarity:")
    for row in entity_pair_similarities(ga, gb, K=args.hops, top_k=args.top_pairs):
        print(
            f"  {row['entity_a']:<35}  ↔  {row['entity_b']:<35}  {row['sim']:.4f}"
        )
