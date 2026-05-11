"""
containment_closure.py
-----------------------
Transitive containment graph and observable-reach similarity.

Every entity in a conceptual-model JSON may have attributes whose TYPE is
another entity class (e.g. Engine.crankshaft: Crankshaft).  These form a
directed containment graph A → B meaning "A structurally contains B".

BFS from each entity gives its full reachable set — every entity and every
observable type (Temperature, Torque, …) accessible through any composition
chain, no matter how deep.

Two entities from *different* ontologies score high containment similarity
when their transitive observable signatures overlap.  For example:

  V1:  Engine     reaches {Torque, Temperature, Pressure, AngularVelocity, …}
  Net: EngineBlock reaches {Temperature, Mass, Torque, AngularVelocity, …}

They share most observables even though they have different names and
different internal structure.

Usage
-----
    from containment_closure import ContainmentClosure, run_closure_stage
    from pathlib import Path

    c_a = ContainmentClosure(json_a)
    c_b = ContainmentClosure(json_b)

    sim = c_a.transitive_jaccard("Engine", c_b, "EngineBlock")

    # Produce full pairwise CSV (all A×B entity pairs):
    run_closure_stage(json_a, json_b, out_csv=Path("closure.csv"))
"""

from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _entity_name(ent: dict) -> str:
    return ent.get("entityName") or ent.get("name", "")


def _entity_attrs(ent: dict) -> list[dict]:
    return ent.get("entityAttributes") or ent.get("attributes") or []


def _safe(name: str) -> str:
    """Lower-case, strip whitespace — used only for containment-graph keys."""
    return name.strip()


# ---------------------------------------------------------------------------
# ContainmentClosure
# ---------------------------------------------------------------------------

class ContainmentClosure:
    """
    Containment graph and transitive observable-reach for one JSON model.

    Attributes
    ----------
    entity_names : set[str]
        All entity class names in the model (original casing).
    direct_contained : dict[str, set[str]]
        entity → set of entity names directly referenced by its attributes.
    direct_observables : dict[str, set[str]]
        entity → set of observable type names on its own (non-entity) attributes.
    reachable_entities : dict[str, frozenset[str]]
        entity → frozenset of ALL entities reachable via BFS (includes self).
    reachable_observables : dict[str, frozenset[str]]
        entity → frozenset of ALL observable types reachable via BFS.
    depth1_observables : dict[str, frozenset[str]]
        entity → frozenset of observables on self + direct children only.
    """

    def __init__(self, json_data: dict) -> None:
        # ── collect entity names ──────────────────────────────────────────
        self.entity_names: set[str] = {
            _safe(_entity_name(e))
            for e in json_data.get("entities", [])
            if _entity_name(e)
        }

        # ── build direct containment and direct observable maps ───────────
        self.direct_contained: dict[str, set[str]] = {}
        self.direct_observables: dict[str, set[str]] = {}

        for ent in json_data.get("entities", []):
            name = _safe(_entity_name(ent))
            if not name:
                continue
            contained: set[str] = set()
            observables: set[str] = set()
            for attr in _entity_attrs(ent):
                attr_type = _safe(attr.get("type", ""))
                if not attr_type:
                    continue
                if attr_type in self.entity_names:
                    contained.add(attr_type)
                else:
                    observables.add(attr_type)
            self.direct_contained[name] = contained
            self.direct_observables[name] = observables

        # Fill in any entity that had no attributes
        for name in self.entity_names:
            self.direct_contained.setdefault(name, set())
            self.direct_observables.setdefault(name, set())

        # ── BFS transitive closure ────────────────────────────────────────
        self.reachable_entities: dict[str, frozenset[str]] = {}
        self.reachable_observables: dict[str, frozenset[str]] = {}
        self.depth1_observables: dict[str, frozenset[str]] = {}

        for root in self.entity_names:
            visited_ents: set[str] = set()
            all_obs: set[str] = set()
            queue: deque[str] = deque([root])
            while queue:
                node = queue.popleft()
                if node in visited_ents:
                    continue
                visited_ents.add(node)
                all_obs.update(self.direct_observables.get(node, set()))
                for child in self.direct_contained.get(node, set()):
                    if child not in visited_ents:
                        queue.append(child)

            self.reachable_entities[root] = frozenset(visited_ents)
            self.reachable_observables[root] = frozenset(all_obs)

            # depth-1: self + immediate children only
            d1_obs = set(self.direct_observables.get(root, set()))
            for child in self.direct_contained.get(root, set()):
                d1_obs.update(self.direct_observables.get(child, set()))
            self.depth1_observables[root] = frozenset(d1_obs)

    # ── similarity methods ────────────────────────────────────────────────

    @staticmethod
    def _jaccard(a: frozenset, b: frozenset) -> float:
        if not a and not b:
            return 1.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def transitive_jaccard(
        self,
        entity_a: str,
        other: "ContainmentClosure",
        entity_b: str,
    ) -> float:
        """Jaccard similarity on full transitive observable signatures."""
        obs_a = self.reachable_observables.get(_safe(entity_a), frozenset())
        obs_b = other.reachable_observables.get(_safe(entity_b), frozenset())
        return round(self._jaccard(obs_a, obs_b), 4)

    def depth1_jaccard(
        self,
        entity_a: str,
        other: "ContainmentClosure",
        entity_b: str,
    ) -> float:
        """Jaccard similarity on depth-1 (self + direct children) observable signatures."""
        obs_a = self.depth1_observables.get(_safe(entity_a), frozenset())
        obs_b = other.depth1_observables.get(_safe(entity_b), frozenset())
        return round(self._jaccard(obs_a, obs_b), 4)

    def entailment_scores(
        self,
        entity_a: str,
        other: "ContainmentClosure",
        entity_b: str,
    ) -> dict:
        """Directional entailment-type similarity on transitive observable signatures.

        Treat entity_a's observable set as the *premise* and entity_b's as the
        *hypothesis*:

            a_covers_b  = |obs_A ∩ obs_B| / |obs_B|  (how much of B does A explain)
            b_covers_a  = |obs_B ∩ obs_A| / |obs_A|  (how much of A does B explain)
            entailment_f1 = harmonic mean (symmetric F1-style combined score)

        Both sets empty → (1.0, 1.0, 1.0) (trivially equivalent null descriptions).
        One set empty, other non-empty → 0 coverage in the non-empty direction.
        """
        obs_a = self.reachable_observables.get(_safe(entity_a), frozenset())
        obs_b = other.reachable_observables.get(_safe(entity_b), frozenset())
        overlap = len(obs_a & obs_b)

        a_covers_b = overlap / len(obs_b) if obs_b else (1.0 if not obs_a else 0.0)
        b_covers_a = overlap / len(obs_a) if obs_a else (1.0 if not obs_b else 0.0)
        return {
            "entailment_a_covers_b": round(a_covers_b, 4),
            "entailment_b_covers_a": round(b_covers_a, 4),
            "entailment_f1":         round(max(a_covers_b, b_covers_a), 4),
        }

    def structure_summary(self, entity: str) -> dict:
        """Return a human-readable summary dict for an entity."""
        name = _safe(entity)
        return {
            "entity":               name,
            "direct_children":      sorted(self.direct_contained.get(name, set())),
            "direct_observables":   sorted(self.direct_observables.get(name, set())),
            "reachable_entities":   sorted(self.reachable_entities.get(name, frozenset())),
            "reachable_observables": sorted(self.reachable_observables.get(name, frozenset())),
        }


# ---------------------------------------------------------------------------
# Neural NLI helpers
# ---------------------------------------------------------------------------

_MODULE_DIR       = Path(__file__).parent
_MODELS_DIR       = _MODULE_DIR / "models"
DEFAULT_NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
_NLI_MODEL        = None
_NLI_MODEL_NAME   = None


def _model_local_path(model_name: str) -> Path:
    return _MODELS_DIR / model_name.replace("/", "--")


def _get_nli_model(model_name: str = DEFAULT_NLI_MODEL):
    global _NLI_MODEL, _NLI_MODEL_NAME
    if _NLI_MODEL is None or _NLI_MODEL_NAME != model_name:
        try:
            from sentence_transformers import CrossEncoder
            import torch
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for neural entailment. "
                "Install with: pip install sentence-transformers"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local  = _model_local_path(model_name)
        if local.is_dir():
            print(f"[NLI] Loading {model_name} from {local} ...")
            _NLI_MODEL = CrossEncoder(str(local), device=device)
        else:
            print(f"[NLI] Downloading {model_name} ...")
            _MODELS_DIR.mkdir(parents=True, exist_ok=True)
            _NLI_MODEL = CrossEncoder(model_name, device=device)
            _NLI_MODEL.save(str(local))
            print(f"[NLI] Saved to {local}")
        _NLI_MODEL_NAME = model_name
        print("[NLI] Ready.")
    return _NLI_MODEL


def _get_entailment_idx(model) -> int:
    id2label = getattr(model.model.config, "id2label", {})
    for idx, lbl in id2label.items():
        if lbl.lower() == "entailment":
            return int(idx)
    return 1  # default for nli-MiniLM2-L6-H768


def _obs_to_text(obs: frozenset) -> str:
    if not obs:
        return "the component has no observable attributes"
    return "the component observes: " + ", ".join(sorted(obs))


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Stage: produce a pairwise CSV for all A×B entity combinations
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "entity_a", "entity_b",
    "obs_a_size",
    "obs_b_size",
    "obs_overlap",
    "entailment_a_covers_b",
    "entailment_b_covers_a",
    "entailment_f1",
]


def run_closure_stage(
    json_a: dict,
    json_b: dict,
    out_csv: Path,
    nli_model_name: str = DEFAULT_NLI_MODEL,
    matched_pairs: list[tuple[str, str]] | None = None,
) -> Path:
    """Compute pairwise neural entailment and write to out_csv.

    If matched_pairs is provided, only those (entity_a, entity_b) pairs are
    scored instead of the full A×B cartesian product.  This is a 30–90× speedup
    for large model pairs where the enriched CSV has ~100–300 matched entries
    but the full product is 9,000+.  merge_stage left-joins on (entity_a, entity_b)
    so restricting the closure CSV to matched pairs has no effect on final output.
    """
    c_a = ContainmentClosure(json_a)
    c_b = ContainmentClosure(json_b)

    if matched_pairs is not None:
        # Filter to only pairs whose entities actually exist in both closures
        entity_pairs = [
            (ea, eb) for ea, eb in matched_pairs
            if ea in c_a.entity_names and eb in c_b.entity_names
        ]
    else:
        entity_pairs = [
            (ea, eb)
            for ea in sorted(c_a.entity_names)
            for eb in sorted(c_b.entity_names)
        ]

    if not entity_pairs:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=_CSV_FIELDS).writeheader()
        return out_csv

    model   = _get_nli_model(nli_model_name)
    ent_idx = _get_entailment_idx(model)

    texts_a = {
        ea: _obs_to_text(c_a.reachable_observables.get(ea, frozenset()))
        for ea in c_a.entity_names
    }
    texts_b = {
        eb: _obs_to_text(c_b.reachable_observables.get(eb, frozenset()))
        for eb in c_b.entity_names
    }

    # Two predictions per pair: (A→B) then (B→A)
    batch = []
    for ea, eb in entity_pairs:
        batch.append((texts_a[ea], texts_b[eb]))
        batch.append((texts_b[eb], texts_a[ea]))

    print(f"  [NLI] Scoring {len(entity_pairs)} pairs ({len(batch)} inferences) ...")
    raw = np.array(model.predict(batch, batch_size=64, show_progress_bar=False))

    if raw.ndim == 1:
        # Binary model — apply sigmoid, treat as entailment score directly
        ent_probs = 1.0 / (1.0 + np.exp(-raw))
        a_covers_b_arr = ent_probs[0::2]
        b_covers_a_arr = ent_probs[1::2]
    else:
        probs = _softmax_np(raw)
        a_covers_b_arr = probs[0::2, ent_idx]
        b_covers_a_arr = probs[1::2, ent_idx]

    rows: list[dict] = []
    for i, (ea, eb) in enumerate(entity_pairs):
        obs_a = c_a.reachable_observables.get(ea, frozenset())
        obs_b = c_b.reachable_observables.get(eb, frozenset())
        acb   = float(a_covers_b_arr[i])
        bca   = float(b_covers_a_arr[i])
        rows.append({
            "entity_a":              ea,
            "entity_b":              eb,
            "obs_a_size":            len(obs_a),
            "obs_b_size":            len(obs_b),
            "obs_overlap":           len(obs_a & obs_b),
            "entailment_a_covers_b": round(acb, 4),
            "entailment_b_covers_a": round(bca, 4),
            "entailment_f1":         round(max(acb, bca), 4),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_a, n_b = len(c_a.entity_names), len(c_b.entity_names)
    print(f"  [Closure] {n_a} × {n_b} = {len(rows)} pairs -> {out_csv}")
    return out_csv


# ---------------------------------------------------------------------------
# CLI — inspect a single JSON or compare two
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Containment closure tool.\n"
            "  --json-a FILE          Inspect one model's transitive reach.\n"
            "  --json-a FILE --json-b FILE  Compare two models pairwise."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json-a", required=True, help="First model JSON")
    parser.add_argument("--json-b", default=None,  help="Second model JSON (optional)")
    parser.add_argument("--out",    default=None,
                        help="Write pairwise CSV here (only with --json-b)")
    parser.add_argument("--top",    type=int, default=10,
                        help="Top-N pairs by containment_sim to print (default 10)")
    args = parser.parse_args()

    data_a = json.loads(Path(args.json_a).read_text(encoding="utf-8"))
    c_a = ContainmentClosure(data_a)

    if args.json_b is None:
        # ── Single-model inspection mode ──────────────────────────────────
        print(f"\nModel: {data_a.get('modelName', args.json_a)}")
        print(f"Entities: {len(c_a.entity_names)}")
        print(f"\n{'Entity':<30} {'Reach':>5}  {'Obs':>4}  Top observables")
        print("-" * 80)
        for ent in sorted(c_a.entity_names):
            reach = len(c_a.reachable_entities[ent])
            obs   = sorted(c_a.reachable_observables[ent])
            preview = ", ".join(obs[:5]) + ("…" if len(obs) > 5 else "")
            print(f"{ent:<30} {reach:>5}  {len(obs):>4}  {preview}")
    else:
        # ── Two-model comparison mode ─────────────────────────────────────
        data_b = json.loads(Path(args.json_b).read_text(encoding="utf-8"))
        c_b = ContainmentClosure(data_b)

        out = Path(args.out) if args.out else None
        if out:
            run_closure_stage(data_a, data_b, out)
            print(f"Written: {out}")

        # Print top-N pairs
        pairs = []
        for ea in sorted(c_a.entity_names):
            for eb in sorted(c_b.entity_names):
                sim = c_a.transitive_jaccard(ea, c_b, eb)
                pairs.append((sim, ea, eb))
        pairs.sort(reverse=True)

        print(f"\nTop-{args.top} pairs by transitive observable Jaccard:")
        print(f"{'Entity A':<32} {'Entity B':<32} {'Sim':>5}  Details")
        print("-" * 80)
        for sim, ea, eb in pairs[:args.top]:
            obs_a = c_a.reachable_observables.get(ea, frozenset())
            obs_b = c_b.reachable_observables.get(eb, frozenset())
            overlap = obs_a & obs_b
            print(f"{ea:<32} {eb:<32} {sim:>5.3f}  "
                  f"(|A|={len(obs_a)} |B|={len(obs_b)} overlap={len(overlap)})")
