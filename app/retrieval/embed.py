"""Embedders + a disk cache, so each (model, texts) pair is embedded once and every engine
gets identical vectors. Register new embedders in EMBEDDERS.

Cache: data/embeddings/<model>/<sha of texts>.npy — safe to delete, rebuilt on demand.
"""

import hashlib
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app.retrieval.types import Embedder

CACHE = Path(__file__).resolve().parents[2] / "data" / "embeddings"


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class FastEmbed:
    """Local ONNX models via fastembed (CPU). name = "fastembed:<model id>"."""

    def __init__(self, model: str):
        from fastembed import TextEmbedding
        self.name = f"fastembed:{model}"
        self._m = TextEmbedding(model_name=model)
        self.dim = len(next(iter(self._m.embed(["probe"]))))

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return _normalize(list(self._m.passage_embed(texts)))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return _normalize(list(self._m.query_embed(texts)))


EMBEDDERS: dict[str, Callable[[], Embedder]] = {
    "bge-small": lambda: FastEmbed("BAAI/bge-small-en-v1.5"),
}

_loaded: dict[str, Embedder] = {}


def get_embedder(key: str) -> Embedder:
    if key not in _loaded:
        _loaded[key] = EMBEDDERS[key]()
    return _loaded[key]


def cached_embed(emb: Embedder, texts: list[str], kind: str) -> np.ndarray:
    """kind: "docs" or "queries" (models may embed them differently)."""
    h = hashlib.sha256("\x1e".join(texts).encode()).hexdigest()[:16]
    path = CACHE / re.sub(r"[^A-Za-z0-9._-]+", "_", emb.name) / f"{kind}-{h}.npy"
    if path.exists():
        return np.load(path)
    vecs = emb.embed_docs(texts) if kind == "docs" else emb.embed_queries(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs
