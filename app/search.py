"""Search service: retrieval mode -> retrieval Config -> a built Pipeline (cached), plus result shaping.

The UI only switches the retrieval mode; chunker and embedder are fixed to DEFAULTS (para256 won
the benchmark's chunking grid). `app.quality` can still try other chunkers. Each Config is built
once on first use and kept in memory.
"""

import json
import re
import threading
import time
from pathlib import Path

from app.retrieval.chunking import CHUNKERS
from app.retrieval.embed import EMBEDDERS
from app.retrieval.pipeline import Config, Pipeline

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "clean" / "docs.jsonl"
DEFAULTS = {"retrieval": "hybrid", "chunker": "para256", "embedder": "bge-small"}


def load_docs() -> list[dict]:
    """All docs except non-English transcripts (flag lang_xx). Metadata-only docs stay searchable."""
    with DOCS.open(encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    return [d for d in docs if not any(fl.startswith("lang_") for fl in d["metadata"]["flags"])]


def options() -> dict:
    return {
        "retrieval": ["hybrid", "bm25", "dense"],
        "defaults": DEFAULTS,
    }


def _snippet(text: str, query: str, width: int = 320) -> tuple[str, list[str]]:
    """Window of the passage around the densest cluster of query terms; terms returned for highlighting."""
    terms = sorted({t for t in re.findall(r"\w+", query.lower()) if len(t) > 2}, key=len, reverse=True)
    if not text:
        return "", terms
    low = text.lower()
    hits = sorted(m.start() for t in terms for m in re.finditer(r"\b" + re.escape(t[:6]), low))
    start = 0
    if hits:
        best = max(hits, key=lambda h: sum(h <= x < h + width for x in hits))
        start = max(0, best - 60)
        start = text.rfind(" ", 0, start) + 1 if start else 0
    out = text[start:start + width].strip()
    return ("…" if start else "") + out + ("…" if start + width < len(text) else ""), terms


class SearchService:
    def __init__(self):
        self.docs = {d["id"]: d for d in load_docs()}
        self._pipes: dict[Config, Pipeline] = {}
        self._lock = threading.Lock()

    def config(self, retrieval: str, chunker: str = DEFAULTS["chunker"],
               embedder: str | None = DEFAULTS["embedder"]) -> Config:
        if chunker not in CHUNKERS or retrieval not in ("bm25", "dense", "hybrid"):
            raise ValueError("unknown knob value")
        if retrieval == "bm25":
            embedder = None
        elif embedder not in EMBEDDERS:
            raise ValueError("unknown embedder")
        return Config(retrieval=retrieval, chunker=chunker, embedder=embedder)

    def pipeline(self, cfg: Config) -> tuple[Pipeline, float | None]:
        """Built pipeline for cfg, and how long building took if it happened now."""
        with self._lock:  # one build at a time: builds are CPU/RAM heavy
            if cfg in self._pipes:
                return self._pipes[cfg], None
            t = time.perf_counter()
            pipe = Pipeline(cfg)
            pipe.build(list(self.docs.values()))
            self._pipes[cfg] = pipe
            return pipe, time.perf_counter() - t

    def search(self, query: str, cfg: Config, k: int = 20) -> dict:
        pipe, built_s = self.pipeline(cfg)
        t = time.perf_counter()
        ranked = pipe.search_passages(query, k)
        took_ms = (time.perf_counter() - t) * 1000
        results = []
        for doc_id, score, pid in ranked:
            d, m = self.docs[doc_id], self.docs[doc_id]["metadata"]
            snippet, terms = _snippet(pipe.passages[pid].text, query)
            results.append({
                "id": doc_id, "title": d["title"], "author": d["author"], "url": d["url"],
                "antislop_url": m["antislop_url"], "type": m["type"], "date": m["date"],
                "read_time": m["read_time"], "summary": m["summary"], "image": m["image"],
                "snippet": snippet or m["summary"], "terms": terms, "score": score,
                "no_text": "no_text" in m["flags"],
            })
        return {"config": cfg.name, "took_ms": took_ms, "built_s": built_s,
                "n_passages": len(pipe.passages), "results": results}

    def close(self) -> None:
        for p in self._pipes.values():
            p.close()
