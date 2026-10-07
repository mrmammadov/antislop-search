# antislop-search

Personal search web app over the antislop corpus — for the owner's daily use, not benchmarking
(benchmarking lives in `../search_experiments`). Python 3.13, run everything with `uv run`.

- `app/search.py` — knobs → `bench.pipeline.Config` → cached, built `Pipeline`; result shaping.
- `app/server.py` — FastAPI; `app/static/index.html` — the whole UI (vanilla JS, no build step).
- `app/quality.py` — score knob combinations on `data/eval/v1` (check before changing defaults).
- `extract/` — corpus pipeline (resolve → fetch → clean), see `extract/README.md`.
- `tickets/` — backlog; read `tickets/README.md` before picking up or finishing work.

Engine adapters, chunkers, embedders and `Pipeline` belong to `../search_experiments/bench`:
change them there (it's an editable dependency) so the benchmark and this app stay in sync.
The app sets `BENCH_EMBED_CACHE` to its own `data/embeddings/`.
