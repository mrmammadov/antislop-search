"""Core types shared by the chunkers, engine and pipeline.

Vocabulary
- Doc      one resource from docs.jsonl (what we evaluate on: relevance is judged per doc)
- Passage  the unit an engine indexes: a whole doc or a chunk of one (made by a chunker)
- Hit      (passage id, score) returned by an engine; the pipeline turns passage hits into doc rankings
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Passage:
    id: str          # unique, e.g. "<doc_id>#0"
    doc_id: str
    title: str
    text: str        # body text of this passage only
    fields: dict = field(default_factory=dict, hash=False, compare=False)
    # fields = doc-level metadata copied onto every passage: author, type, date, summary, key_highlight


@dataclass(frozen=True)
class Hit:
    id: str          # Passage.id
    score: float     # higher is better; scales differ between modes


class Embedder(Protocol):
    name: str  # stable id used as cache key, e.g. "fastembed:BAAI/bge-small-en-v1.5"
    dim: int

    def embed_docs(self, texts: list[str]) -> np.ndarray: ...     # (n, dim) float32, L2-normalized
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...  # (n, dim) float32, L2-normalized
