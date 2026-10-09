"""A retrieval pipeline = chunker -> engine (text / vector / both) -> fusion -> [rerank] -> doc ranking.

One Config fully describes a pipeline; its .name labels it (and namespaces its SQLite tables).
Fusion/rerankers live here as small registries.
"""

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from app.retrieval.chunking import chunk, embed_text
from app.retrieval.embed import cached_embed, get_embedder
from app.retrieval.sqlite import SqliteEngine
from app.retrieval.types import Hit, Passage


@dataclass(frozen=True)
class Config:
    retrieval: str = "bm25"          # bm25 | dense | hybrid
    chunker: str = "doc"
    embedder: str | None = None      # key in embed.EMBEDDERS; required for dense/hybrid
    text_fields: tuple[str, ...] = ()  # extra Passage.fields to make keyword-searchable
    fusion: str = "rrf"              # hybrid only
    reranker: str | None = None
    depth: int = 100                 # passages fetched per retriever before fusion/rerank

    @property
    def name(self) -> str:
        parts = [self.retrieval, self.chunker]
        if self.retrieval != "bm25":
            parts.append(self.embedder)
        if self.text_fields:
            parts.append("+".join(self.text_fields))
        if self.retrieval == "hybrid":
            parts.append(self.fusion)
        if self.reranker:
            parts.append(f"rr-{self.reranker}")
        return "__".join(p.replace("/", "-").replace(":", "-") for p in parts)

    def to_dict(self) -> dict:
        return asdict(self)


# --- fusion: list of ranked hit lists -> one ranked hit list ---------------------------------

def rrf(runs: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: score = sum 1/(k + rank). Scale-free, so BM25 and cosine mix safely."""
    scores: dict[str, float] = {}
    for run in runs:
        for rank, h in enumerate(run, 1):
            scores[h.id] = scores.get(h.id, 0.0) + 1.0 / (k + rank)
    return [Hit(i, s) for i, s in sorted(scores.items(), key=lambda x: -x[1])]


FUSERS: dict[str, Callable[[list[list[Hit]]], list[Hit]]] = {"rrf": rrf}

# --- rerankers: (query, candidate passages) -> scores, higher better --------------------------

RERANKERS: dict[str, Callable[[], Callable[[str, list[Passage]], list[float]]]] = {}


class Pipeline:
    def __init__(self, cfg: Config):
        if cfg.retrieval in ("dense", "hybrid") and not cfg.embedder:
            raise ValueError(f"retrieval={cfg.retrieval} needs an embedder")
        self.cfg = cfg
        ns = "b_" + hashlib.sha1(cfg.name.encode()).hexdigest()[:10]
        self.engine = SqliteEngine(namespace=ns)
        self.embedder = get_embedder(cfg.embedder) if cfg.embedder else None
        self.reranker = RERANKERS[cfg.reranker]() if cfg.reranker else None
        self.passages: dict[str, Passage] = {}
        self.stats: dict = {}

    def build(self, docs: list[dict]) -> None:
        import time
        passages = chunk(docs, self.cfg.chunker)
        self.passages = {p.id: p for p in passages}
        self.stats = {"n_docs": len(docs), "n_passages": len(passages)}
        if self.cfg.retrieval in ("bm25", "hybrid"):
            t = time.perf_counter()
            self.engine.build_text(passages, self.cfg.text_fields)
            self.stats["build_text_s"] = time.perf_counter() - t
        if self.cfg.retrieval in ("dense", "hybrid"):
            t = time.perf_counter()
            vecs = cached_embed(self.embedder, [embed_text(p) for p in passages], "docs")
            self.stats["embed_docs_s"] = time.perf_counter() - t
            t = time.perf_counter()
            self.engine.build_vectors(passages, vecs)
            self.stats["build_vectors_s"] = time.perf_counter() - t

    def embed_queries(self, queries: list[str]) -> np.ndarray | None:
        """Batch + cache query vectors up front so query latency measures the engine, not the model."""
        if not self.embedder:
            return None
        return cached_embed(self.embedder, queries, "queries")

    def search(self, query: str, k: int, qvec: np.ndarray | None = None) -> list[tuple[str, float]]:
        """Top-k (doc_id, score). A doc's score is its best passage's score."""
        return [(d, s) for d, s, _ in self.search_passages(query, k, qvec)]

    def search_passages(self, query: str, k: int,
                        qvec: np.ndarray | None = None) -> list[tuple[str, float, str]]:
        """Top-k (doc_id, score, best passage id) — the passage is what a UI shows as the snippet."""
        runs = []
        if self.cfg.retrieval in ("bm25", "hybrid"):
            runs.append(self.engine.search_text(query, self.cfg.depth))
        if self.cfg.retrieval in ("dense", "hybrid"):
            if qvec is None:
                qvec = self.embedder.embed_queries([query])[0]  # live query: no disk cache
            runs.append(self.engine.search_vectors(qvec, self.cfg.depth))
        hits = runs[0] if len(runs) == 1 else FUSERS[self.cfg.fusion](runs)

        if self.reranker and hits:
            cands = [self.passages[h.id] for h in hits]
            scores = self.reranker(query, cands)
            hits = sorted((Hit(p.id, s) for p, s in zip(cands, scores)), key=lambda h: -h.score)

        best: dict[str, tuple[float, str]] = {}
        for h in hits:  # hits are best-first, so the first hit per doc is its best
            d = self.passages[h.id].doc_id
            if d not in best:
                best[d] = (h.score, h.id)
                if len(best) == k:
                    break
        return [(d, s, pid) for d, (s, pid) in best.items()]

    def close(self) -> None:
        self.engine.close()
