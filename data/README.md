# data/

| Folder | Contents | Who writes it |
|---|---|---|
| `raw/` | Source CSV(s) with links. Never edited. | You |
| `fetched/` | Raw HTML/PDF snapshots as downloaded, one file per URL (hashed name) + `manifest.jsonl` (url, fetched_at, status, content hash). | Fetch step |
| `clean/` | `docs.jsonl` — cleaned docs, schema: `id, url, title, text, fetched_at, metadata`. | Clean step |

**`raw/`, `fetched/` and `clean/` are not in git** (third-party content; the repo is public).
They live only on machines that ran `extract/`. A fresh clone needs them copied in, e.g.
`rsync -a data/raw data/fetched data/clean server:antislop-search/data/` (or re-run `extract/`).
`eval/` (our own queries and judgments) is versioned.

`LAST_EXTRACTED.txt` logs each extraction run. Fetching is slow and the web changes,
so it runs rarely; cleaning reads only from `fetched/` and can be re-run any time
without touching the network — that's what makes it reproducible.
