"""
model_cache.py
---------------
Shared local-cache helper for every neural network model this pipeline
downloads (sentence-transformer bi-encoders, cross-encoder NLI models):
look under MODELS_DIR first, and only hit the network if the model isn't
there yet, saving it to MODELS_DIR afterward so later runs never re-download.

MODELS_DIR is `enriched_ontology_matching/models/`, gitignored since these
are large binary weights, not something to check into the repo.

Usage
-----
    from model_cache import load_or_download
    from sentence_transformers import SentenceTransformer

    encoder = load_or_download("paraphrase-MiniLM-L6-v2", SentenceTransformer, device)
"""
from __future__ import annotations

import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"


def local_model_path(model_name: str) -> Path:
    return MODELS_DIR / model_name.replace("/", "--")


def load_or_download(model_name: str, loader_cls, device: str):
    """
    Load `model_name` from MODELS_DIR if already cached there; otherwise
    download it via `loader_cls(model_name, device=device)` and save it to
    MODELS_DIR for every subsequent run. `loader_cls` is any class with this
    constructor signature and a `.save(path)` method, e.g.
    sentence_transformers.SentenceTransformer or .CrossEncoder.
    """
    local = local_model_path(model_name)
    if local.is_dir():
        print(f"[ModelCache] Loading {model_name} from {local} ...", file=sys.stderr)
        return loader_cls(str(local), device=device)

    print(f"[ModelCache] Downloading {model_name} ...", file=sys.stderr)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = loader_cls(model_name, device=device)
    model.save(str(local))
    print(f"[ModelCache] Saved to {local}", file=sys.stderr)
    return model
