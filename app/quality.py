"""Quality check: score knob combinations against the app's eval set (data/eval/v1).

    uv run -m app.quality                                   # the default knobs
    uv run -m app.quality --engine sqlite --chunker para256 --retrieval hybrid

Run it before/after changing a default, to see whether search got better or worse.
v1 = 33 queries (known-item + topical) with graded judgments; partial, see data/eval/v1/README.md.
"""

import argparse
import json
import statistics
from pathlib import Path

from app.search import DEFAULTS, ROOT, SearchService
from bench.metrics import all_metrics

EVAL = ROOT / "data" / "eval" / "v1"


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    for knob, default in DEFAULTS.items():
        ap.add_argument(f"--{knob}", default=default)
    a = ap.parse_args()

    queries = _jsonl(EVAL / "queries.jsonl")
    qrels: dict[str, dict[str, int]] = {}
    for r in _jsonl(EVAL / "qrels.jsonl"):
        qrels.setdefault(r["qid"], {})[r["doc_id"]] = r["grade"]

    svc = SearchService()
    cfg = svc.config(a.engine, a.retrieval, a.chunker, a.embedder)
    pipe, _ = svc.pipeline(cfg)
    by_kind: dict[str, list[dict]] = {}
    for q in queries:
        ranked = [d for d, _ in pipe.search(q["query"], 100)]
        by_kind.setdefault(q["kind"], []).append(all_metrics(ranked, qrels[q["qid"]]))
    svc.close()

    print(f"{cfg.name}")
    rows = [("all", [m for ms in by_kind.values() for m in ms]), *sorted(by_kind.items())]
    for kind, ms in rows:
        avg = {k: statistics.fmean(m[k] for m in ms) for k in ms[0]}
        print(f"  {kind:11} n={len(ms):3}  nDCG@10 {avg['ndcg@10']:.3f}  R@10 {avg['recall@10']:.3f}  "
              f"MRR {avg['mrr@10']:.3f}  judged@10 {avg['judged@10']:.2f}")


if __name__ == "__main__":
    main()
