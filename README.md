# antislop search

Personal search over the [antislop](https://antislop.xyz) collection: ~170 essays, talks and videos.
Web UI with knobs to switch search engine, retrieval mode, chunking and embedding model.

```
uv run uvicorn app.server:app --reload      # → http://127.0.0.1:8000   ( / focuses search )
uv run -m app.quality [--engine sqlite …]   # score a knob combination on data/eval/v1
```

- Engines / chunkers / models come from the sibling benchmark repo `../search_experiments`
  (`bench` package, editable path dependency). New ones there appear as knobs here.
- Docker engines (qdrant, pgvector, meilisearch, opensearch) start on demand on first search
  (OpenSearch takes ~20 s and 1.6 GB RAM). `memory`, `sqlite`, `lancedb` run in-process.
- First search with a new knob combination builds its index (seconds; embedding a new chunker or
  model can take minutes on CPU — cached afterwards in `data/embeddings/`).

## Data

`extract/` turns `data/raw/antislop.csv` into `data/clean/docs.jsonl` — see `extract/README.md`.
