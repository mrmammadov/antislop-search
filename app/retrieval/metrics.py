"""IR metrics over graded relevance. qrels: {doc_id: grade}, grade 0 = judged not relevant,
1 = partially, 2 = relevant, 3 = highly relevant. Docs missing from qrels are unjudged (treated as 0).
"""

import math


def ndcg(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    dcg = sum((2 ** qrels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    rel = {d for d, g in qrels.items() if g > 0}
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def mrr(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if qrels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def judged(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """Share of the top-k that has any judgment. Low = the metrics above are unreliable for this
    system (it surfaces docs nobody assessed) — fix by judging those docs, not by ignoring it."""
    top = ranked[:k]
    return sum(d in qrels for d in top) / len(top) if top else 0.0


def all_metrics(ranked: list[str], qrels: dict[str, int]) -> dict[str, float]:
    return {
        "ndcg@10": ndcg(ranked, qrels, 10),
        "recall@10": recall(ranked, qrels, 10),
        "recall@50": recall(ranked, qrels, 50),
        "mrr@10": mrr(ranked, qrels, 10),
        "judged@10": judged(ranked, qrels, 10),
    }
