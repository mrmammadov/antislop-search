# antislop search

Personal search over the [antislop](https://antislop.xyz) collection: ~170 essays, talks and videos.
Web UI with one knob: retrieval mode (hybrid / bm25 / dense). Engine: SQLite (FTS5 for keywords +
sqlite-vec for vectors), in-process. Chunker `para256` and model `bge-small` are fixed — the
benchmark's best (see `GLOSSARY.md` for the terms).

```
uv run uvicorn app.server:app --reload      # → http://127.0.0.1:8000   ( / focuses search )
uv run -m app.quality [--chunker para400 …] # score a setup on data/eval/v1
```

- Chunkers / models live in `app/retrieval/`; `app.quality` can score any of them.
- First search per mode builds its index (seconds; embeddings are cached in `data/embeddings/`).

## Data

`extract/` turns `data/raw/antislop.csv` into `data/clean/docs.jsonl` — see `extract/README.md`.
Its libraries are the `extract` dependency group (installed by default; a server can skip them
with `uv sync --no-default-groups`).
