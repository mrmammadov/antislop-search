# v1 — app quality test set (partial)

33 queries (12 `known_item`, 21 `topical`) with graded judgments (0–3, 1,263 qrels), written by
an LLM agent on 2026-10-07 in the search_experiments repo; the agent was stopped before adding
`question`/`entity` queries, a review pass, and docs. Used by `uv run -m app.quality`.

Known issues — fix as you notice them:
- **Pooling bias**: candidates were pooled from whole-doc BM25 / dense / hybrid runs only. Chunked
  configs surface docs nobody judged (judged@10 ≈ 0.87 vs 1.00), and unjudged counts as
  irrelevant, so chunked configs are under-scored. Fix: judge the unjudged top-10 docs of the
  configs you care about (ticket 012).
- 10 positive judgments are on docs with no text (metadata only), graded from title/summary.
- Some grade-1s look generous. Owner review pending (ticket 011).
