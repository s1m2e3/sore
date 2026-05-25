"""
owl_builder.py
--------------
Build OWL/Turtle from a model JSON using canonical associations from
association_inventory.csv.

Key design: for every canonical relation, its counter-directional inverse is
also declared and emitted.  If ontology A has:

    ex:Car  ex:UsedFor  ex:Driver .

then this module also emits:

    ex:Driver  ex:IsUsedBy  ex:Car .

and declares:

    ex:UsedFor   owl:inverseOf  ex:IsUsedBy .
    ex:IsUsedBy  owl:inverseOf  ex:UsedFor .

This way an OWL reasoner (HermiT, ELK) or an AML/LogMap matcher that
understands owl:inverseOf can find the relationship regardless of which
direction the other ontology expressed it.

Symmetric relations (Connects, RelatedTo) are declared as
owl:SymmetricProperty — no separate inverse needed.

Usage
-----
    python enriched_ontology_matching/owl_builder.py \\
        --model  enriched_ontology_matching/test_auto_v1_v2.json \\
        --key    json_a \\
        --out    enriched_ontology_matching/outputs/owl/auto_v1.ttl

    # Both models from a comparison JSON:
    python enriched_ontology_matching/owl_builder.py \\
        --model  enriched_ontology_matching/test_auto_v1_v2.json \\
        --out-dir enriched_ontology_matching/outputs/owl/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_normalizer import load_inventory as _norm_load_inventory, normalize_model

# ── Paths ────────────────────────────────────────────────────────────────────────
_REPO_ROOT  = Path(__file__).resolve().parent.parent
_INVENTORY  = _REPO_ROOT / "enriched_ontology_matching" / "association_inventory.csv"
_CANONICALS = _REPO_ROOT / "config" / "canonical_relations.json"
_OUT_DIR    = _REPO_ROOT / "enriched_ontology_matching" / "outputs" / "owl"

DEFAULT_BASE = "http://example.org/onto#"


# ── Load canonical relation metadata ─────────────────────────────────────────────

def load_canonicals(path: Path = _CANONICALS) -> dict[str, dict]:
    """
    Returns {canonical_id: {inverse_id, symmetric, definition, ...}}.
    Also includes entries for every inverse_id so lookups work both ways.
    """
    raw: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    index: dict[str, dict] = {}
    for entry in raw:
        index[entry["id"]] = entry
    # Add inverse entries for non-symmetric relations (so we can look up by inverse_id too)
    for entry in raw:
        inv_id = entry.get("inverse_id", "")
        if inv_id and inv_id != entry["id"] and inv_id not in index:
            index[inv_id] = {
                "id":                 inv_id,
                "inverse_id":         entry["id"],
                "symmetric":          False,
                "definition":         entry.get("inverse_definition", ""),
                "inverse_definition": entry.get("definition", ""),
                "conceptnet_iri":     None,
                "ro_iri":             None,
                "ssn_iri":            None,
                "exemplar_phrases":   [],
            }
    return index


# ── Load inventory ────────────────────────────────────────────────────────────────

def load_inventory(path: Path = _INVENTORY) -> dict[str, dict]:
    """
    Returns {association_name: {canonical, verb_tokens, low_confidence}}.
    """
    inventory: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["association_name"]
            verb_raw = row.get("verb_tokens", "")
            inventory[name] = {
                "canonical":      row["canonical"],
                "score":          float(row["score"]) if row["score"] else 0.0,
                "verb_tokens":    [t.strip() for t in verb_raw.split("|") if t.strip()],
                "low_confidence": row.get("low_confidence", "") == "YES",
            }
    return inventory


# ── IRI helpers ───────────────────────────────────────────────────────────────────

def _safe(name: str) -> str:
    """Escape name to a valid local IRI fragment."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


# ── Turtle header ─────────────────────────────────────────────────────────────────

def _turtle_header(base: str) -> list[str]:
    return [
        "@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix owl:  <http://www.w3.org/2002/07/owl#> .",
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
        f"@prefix ex:   <{base}> .",
        "",
        "<> rdf:type owl:Ontology .",
        "",
    ]


# ── Property declarations ─────────────────────────────────────────────────────────

def _property_declarations(canonicals: dict[str, dict]) -> list[str]:
    """
    Emit owl:ObjectProperty declarations for every canonical and its inverse.

    Symmetric relations get owl:SymmetricProperty.
    Non-symmetric relations get paired owl:inverseOf declarations.
    """
    lines = [
        "# ═══════════════════════════════════════════════════════════════════════",
        "# Canonical relation properties  (forward + counter-directional inverse)",
        "# ═══════════════════════════════════════════════════════════════════════",
        "",
    ]
    seen: set[str] = set()

    # Iterate in a stable order: canonical entries first
    ordered = [c for c in canonicals.values() if c["id"] in {
        "PartOf", "HasA", "UsedFor", "Causes", "CapableOf", "IsA",
        "AtLocation", "MadeOf", "ReceivesAction", "Connects",
        "HasPrerequisite", "RelatedTo",
    }]

    for entry in ordered:
        cid    = entry["id"]
        inv_id = entry.get("inverse_id", cid)
        sym    = entry.get("symmetric", False)

        if cid in seen:
            continue
        seen.add(cid)

        defn = entry.get("definition", "")

        if sym:
            lines += [
                f"ex:{_safe(cid)}",
                f"    rdf:type owl:ObjectProperty, owl:SymmetricProperty ;",
                f'    rdfs:comment "{defn}"^^xsd:string .',
                "",
            ]
        else:
            seen.add(inv_id)
            inv_entry  = canonicals.get(inv_id, {})
            inv_defn   = inv_entry.get("definition", entry.get("inverse_definition", ""))

            lines += [
                f"# {cid} ↔ {inv_id}",
                f"ex:{_safe(cid)}",
                f"    rdf:type owl:ObjectProperty ;",
                f"    owl:inverseOf ex:{_safe(inv_id)} ;",
                f'    rdfs:comment "{defn}"^^xsd:string .',
                "",
                f"ex:{_safe(inv_id)}",
                f"    rdf:type owl:ObjectProperty ;",
                f"    owl:inverseOf ex:{_safe(cid)} ;",
                f'    rdfs:comment "{inv_defn}"^^xsd:string .',
                "",
            ]

    return lines


# ── Entity declarations ───────────────────────────────────────────────────────────

def _entity_declarations(entities: list[dict]) -> list[str]:
    lines = [
        "# ═══════════════════════════════════════════════════════════════════════",
        "# Entity classes",
        "# ═══════════════════════════════════════════════════════════════════════",
        "",
    ]
    for ent in entities:
        name = ent.get("entityName") or ent.get("name") or ""
        if not name:
            continue
        safe = _safe(name)
        lines.append(f"ex:{safe} rdf:type owl:Class .")
    lines.append("")
    return lines


# ── Association triples (forward + inverse) ───────────────────────────────────────

def _association_triples(
    model_json: dict,
    inventory:  dict[str, dict],
    canonicals: dict[str, dict],
    skip_low_confidence: bool = False,
) -> list[str]:
    """
    For every association in the model, emit:
      1. Forward triple:   entity_a  :canonical         entity_b
      2. Inverse triple:   entity_b  :inverse_canonical entity_a
      3. Annotation triple with verb_tokens and score (rdfs:comment on a blank node)

    Symmetric relations only emit one triple (no separate inverse needed).
    """
    lines = [
        "# ═══════════════════════════════════════════════════════════════════════",
        "# Associations  (forward + counter-directional inverse)",
        "# ═══════════════════════════════════════════════════════════════════════",
        "",
    ]

    for assoc in model_json.get("associations", []):
        assoc_name   = assoc.get("associationName") or assoc.get("name") or ""
        participants = (
            assoc.get("associationParticipants")
            or assoc.get("participants")
            or []
        )
        if not assoc_name or len(participants) < 2:
            continue

        # Prefer the canonical field set by normalize_model; fall back to
        # full inventory lookup for verb_tokens and score metadata.
        if assoc_name in inventory:
            inv_row = inventory[assoc_name]
            if skip_low_confidence and inv_row["low_confidence"]:
                continue
            canonical   = assoc.get("canonical") or inv_row["canonical"]
            verb_tokens = inv_row["verb_tokens"]
            score       = inv_row["score"]
        else:
            canonical   = assoc.get("canonical", "RelatedTo")
            verb_tokens = []
            score       = 0.0

        canon_entry = canonicals.get(canonical, {})
        inv_id      = canon_entry.get("inverse_id", canonical)
        symmetric   = canon_entry.get("symmetric", False)
        verb_str    = " ".join(verb_tokens) if verb_tokens else canonical.lower()

        # Emit triples for all consecutive participant pairs
        for k in range(len(participants) - 1):
            for m in range(k + 1, len(participants)):
                ea = _safe(participants[k])
                eb = _safe(participants[m])

                lines.append(f"# {assoc_name}  [{canonical}]  score={score:.3f}")
                # Forward
                lines.append(f"ex:{ea}  ex:{_safe(canonical)}  ex:{eb} .")
                # Inverse (skip for symmetric — A→B already implies B→A)
                if not symmetric and inv_id != canonical:
                    lines.append(f"ex:{eb}  ex:{_safe(inv_id)}  ex:{ea} .")
                lines.append("")

    return lines


# ── Main builder ──────────────────────────────────────────────────────────────────

def build_owl(
    model_json:          dict,
    inventory:           dict[str, dict],
    canonicals:          dict[str, dict],
    base_uri:            str  = DEFAULT_BASE,
    skip_low_confidence: bool = False,
) -> str:
    """
    Build a complete Turtle OWL document for one model JSON.

    The model is normalized first (canonical annotation + composition→PartOf
    synthesis) so the OWL includes structural relationships from V-style
    composition attributes alongside all explicit associations.

    The document includes:
    - owl:ObjectProperty declarations for all canonicals + their inverses
    - owl:inverseOf links between each canonical and its counter-directional
    - owl:SymmetricProperty for Connects and RelatedTo
    - owl:Class declarations for all entities
    - Forward + inverse assertion triples for every association
    """
    simple_inv = {name: info["canonical"] for name, info in inventory.items()}
    model_json = normalize_model(model_json, simple_inv)

    lines: list[str] = []
    lines += _turtle_header(base_uri)
    lines += _property_declarations(canonicals)
    lines += _entity_declarations(model_json.get("entities", []))
    lines += _association_triples(model_json, inventory, canonicals, skip_low_confidence)
    return "\n".join(lines)


# ── Convenience: write both models from a comparison JSON ────────────────────────

def build_pair(
    pair_json_path:      Path,
    out_dir:             Path,
    inventory:           dict[str, dict],
    canonicals:          dict[str, dict],
    base_uri:            str  = DEFAULT_BASE,
    skip_low_confidence: bool = False,
) -> tuple[Path, Path]:
    """
    Read a JSON with {json_a, json_b} structure, write two .ttl files.
    Returns (path_a, path_b).
    """
    data = json.loads(pair_json_path.read_text(encoding="utf-8"))
    stem = pair_json_path.stem

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key in ("json_a", "json_b"):
        model = data.get(key, data)
        suffix = key.replace("json_", "")
        out_path = out_dir / f"{stem}_{suffix}.ttl"
        owl_text = build_owl(model, inventory, canonicals, base_uri, skip_low_confidence)
        out_path.write_text(owl_text, encoding="utf-8")
        print(f"Wrote: {out_path}")
        paths.append(out_path)

    return tuple(paths)  # type: ignore[return-value]


# ── CLI ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Build bidirectional OWL/Turtle from a model JSON using canonical "
            "associations.  Emits both forward and counter-directional inverse "
            "triples so OWL reasoners can find matches regardless of direction."
        )
    )
    ap.add_argument("--model", required=True,
                    help="Path to model JSON (single model or {json_a, json_b} pair)")
    ap.add_argument("--key", default=None,
                    help="Sub-key to extract from the JSON (e.g. json_a, json_b). "
                         "If omitted and json_a/json_b keys exist, both are written.")
    ap.add_argument("--out",     default=None, help="Output .ttl file path (single model)")
    ap.add_argument("--out-dir", default=str(_OUT_DIR),
                    help="Output directory when writing a pair (default: outputs/owl/)")
    ap.add_argument("--base",    default=DEFAULT_BASE, help="OWL base IRI")
    ap.add_argument("--inventory", default=str(_INVENTORY))
    ap.add_argument("--canonicals", default=str(_CANONICALS))
    ap.add_argument("--skip-low-confidence", action="store_true")
    args = ap.parse_args()

    inventory  = load_inventory(Path(args.inventory))
    canonicals = load_canonicals(Path(args.canonicals))
    print(f"Loaded {len(inventory)} canonical associations.")
    print(f"Loaded {len(canonicals)} relation entries (canonical + inverse).")

    data = json.loads(Path(args.model).read_text(encoding="utf-8"))

    if args.key:
        model = data.get(args.key, data)
        out_path = Path(args.out) if args.out else (
            _OUT_DIR / f"{Path(args.model).stem}_{args.key}.ttl"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            build_owl(model, inventory, canonicals, args.base, args.skip_low_confidence),
            encoding="utf-8",
        )
        print(f"Wrote: {out_path}")

    elif "json_a" in data and "json_b" in data:
        build_pair(
            Path(args.model),
            Path(args.out_dir),
            inventory, canonicals, args.base, args.skip_low_confidence,
        )

    else:
        out_path = Path(args.out) if args.out else (
            _OUT_DIR / f"{Path(args.model).stem}.ttl"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            build_owl(data, inventory, canonicals, args.base, args.skip_low_confidence),
            encoding="utf-8",
        )
        print(f"Wrote: {out_path}")
