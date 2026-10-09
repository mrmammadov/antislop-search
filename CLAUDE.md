# antislop-search

Read `MEMORY.md` first: owner preferences, project history, numbers and decisions so far.

Personal search web app over the antislop corpus — for the owner's daily use, not benchmarking
(benchmarking lives in `../search_experiments`). Python 3.13, run everything with `uv run`.

- `app/search.py` — knobs → `retrieval.pipeline.Config` → cached, built `Pipeline`; result shaping.
- `app/retrieval/` — chunkers, embedder, SQLite engine (FTS5 + sqlite-vec), `Pipeline`, metrics.
- `app/server.py` — FastAPI; `app/static/index.html` — the whole UI (vanilla JS, no build step).
- `app/quality.py` — score knob combinations on `data/eval/v1` (check before changing defaults).
- `extract/` — corpus pipeline (resolve → fetch → clean), see `extract/README.md`.
- `GLOSSARY.md` — the search terms used in this project, in plain words.
- `tickets/` — backlog; read `tickets/README.md` before picking up or finishing work.

`app/retrieval/` started as a copy of `../search_experiments/bench` (2026-10-09) so the app deploys
on its own, then cut down to SQLite as the only engine; it is maintained here now and the two may drift. To adopt something proven in the
benchmark, port it into `app/retrieval/` and check it with `app.quality`.
`extract/` libraries are an optional `extract` dependency group (default locally, skipped on a server).
