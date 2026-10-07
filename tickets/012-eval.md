---
id: 012
title: Fix v1 pooling bias
track: eval
kind: task
owner: agent
status: open
depends: []
---

v1 candidates were pooled from whole-doc runs only, so chunked configs surface unjudged docs (judged@10 0.87 vs 1.00) and get under-scored. Judge the unjudged top-10 docs of the configs that matter (default knobs + bm25/dense variants), in small batches written to disk. See data/eval/v1/README.md.
