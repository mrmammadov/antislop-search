# data/

| Folder | Contents | Who writes it |
|---|---|---|
| `raw/` | Source CSV(s) with links. Never edited. | You |
| `fetched/` | Raw HTML/PDF snapshots as downloaded, one file per URL (hashed name) + `manifest.jsonl` (url, fetched_at, status, content hash). | Fetch step |
| `clean/` | `docs.jsonl` — cleaned docs, schema: `id, url, title, text, fetched_at, metadata`. | Clean step |

`LAST_EXTRACTED.txt` logs each extraction run. Fetching is slow and the web changes,
so it runs rarely; cleaning reads only from `fetched/` and can be re-run any time
without touching the network — that's what makes it reproducible.
