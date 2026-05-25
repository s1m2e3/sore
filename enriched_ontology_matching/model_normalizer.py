"""
model_normalizer.py
-------------------
Shared preprocessing for all metrics: normalize a model JSON before processing.

Two transformations applied to a deep copy (original never mutated):

1.  Association → canonical
    Every association in the JSON gets a ``canonical`` field looked up in
    association_inventory.csv.  Unknown associations get ``"RelatedTo"``.
    The original ``associationName`` is preserved so display / debugging still
    shows the raw name.

2.  Composition → PartOf
    entityAttributes whose ``type`` matches another entity in the same model
    represent UML/SysML composition (the parent owns the child as a part).
    For each such attribute a synthetic association is appended:

        associationName        : "{child}PartOf{parent}"   (unique per pair)
        canonical              : "PartOf"
        _synthetic             : True
        associationParticipants: [child, parent]

    This bridges composition-centric models (V-style, parts expressed via
    attributes) with association-centric ones (Net-style, explicit edges) so
    every downstream metric sees a uniform, fully-explicit association graph.

Usage
-----
    from model_normalizer import normalize_model, load_inventory

    inventory = load_inventory()          # cached singleton
    norm_data = normalize_model(raw_data, inventory)
"""
from __future__ import annotations

import copy
import csv
from pathlib import Path

_DIR       = Path(__file__).resolve().parent
_INVENTORY = _DIR / "association_inventory.csv"

_cache: dict[str, str] | None = None


def load_inventory(path: Path = _INVENTORY) -> dict[str, str]:
    """
    Load association_inventory.csv → {association_name: canonical}.
    Result is a module-level singleton — safe to call repeatedly.
    """
    global _cache
    if _cache is None:
        _cache = {}
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                _cache[row["association_name"]] = row["canonical"]
    return _cache


def normalize_model(data: dict, inventory: dict[str, str] | None = None) -> dict:
    """
    Return a normalized deep copy of a single model JSON dict.

    Parameters
    ----------
    data      : raw model JSON dict (a single model, not a {json_a, json_b} wrapper).
    inventory : {assoc_name: canonical}; loaded from the default CSV if None.

    Returns
    -------
    dict
        Deep copy with:
        - Every association annotated with a ``canonical`` field.
        - Synthetic ``{child}PartOf{parent}`` associations appended for each
          composition attribute (entityAttribute.type ∈ entity_names).
    """
    if inventory is None:
        inventory = load_inventory()

    data = copy.deepcopy(data)

    # Collect entity names for composition detection
    entity_names: set[str] = {
        e.get("entityName") or e.get("name", "")
        for e in data.get("entities", [])
    }
    entity_names.discard("")

    # 1. Annotate every existing association with its canonical relation type
    for assoc in data.get("associations", []):
        raw = assoc.get("associationName") or assoc.get("name") or ""
        assoc["canonical"] = inventory.get(raw, "RelatedTo")

    # 2. Synthesize PartOf associations from compositional entityAttributes
    # Track (child, parent) pairs already represented by existing PartOf
    # associations so we never emit a duplicate edge.
    covered: set[tuple[str, str]] = set()
    for assoc in data.get("associations", []):
        if assoc.get("canonical") == "PartOf":
            parts = assoc.get("associationParticipants") or assoc.get("participants") or []
            if len(parts) >= 2:
                covered.add((parts[0], parts[1]))
                covered.add((parts[1], parts[0]))

    seen_pairs: set[tuple[str, str]] = set(covered)
    synthetic: list[dict] = []

    for entity in data.get("entities", []):
        parent = entity.get("entityName") or entity.get("name") or ""
        if not parent:
            continue
        for attr in (entity.get("entityAttributes") or entity.get("attributes") or []):
            child = attr.get("type", "")
            if child not in entity_names or child == parent:
                continue
            pair = (child, parent)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            synthetic.append({
                "associationName":        f"{child}PartOf{parent}",
                "canonical":              "PartOf",
                "_synthetic":             True,
                "associationParticipants": [child, parent],
            })

    data.setdefault("associations", []).extend(synthetic)
    return data
