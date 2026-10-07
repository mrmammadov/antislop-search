# extract/

Turns `data/raw/antislop.csv` (links to antislop.xyz resource pages) into `data/clean/docs.jsonl`.

```
uv run extract/resolve.py   # 1. render antislop pages (Playwright) -> source URL + key highlight
uv run extract/fetch.py     # 2. download sources: HTML / PDF / YouTube transcript -> data/fetched/source/
uv run extract/clean.py     # 3. offline: extract + normalize text -> data/clean/docs.jsonl
```

**Re-running.** Stages 1–2 hit the network, so they skip anything fetched successfully in the
last 30 days (`MAX_AGE` in `common.py`) — re-running only retries failures and new CSV rows.
`--force` re-fetches everything, `--limit N` is for testing. Every run appends a line to
`data/LAST_EXTRACTED.txt`. Stage 3 is offline and deterministic; run it whenever cleaning changes.

**Fallbacks in fetch.py**
- JS-rendered page (< 500 chars via plain HTTP) -> re-render in Chromium, incl. iframe contents
- HTTP error, or still thin (expired domain, CAPTCHA) -> latest Wayback Machine snapshot
- YouTube links (incl. ones wrapped in web.archive.org) -> captions, one request every 3s

**Quality flags in docs.jsonl** (`metadata.flags`): `no_text`, `too_short` (< 30% of the words
implied by antislop's read time), `too_long` (> 3x). Every doc also carries antislop's
`summary` and `key_highlight`, so no-text docs are still searchable by metadata.
Flagged items are also written to `data/clean/flagged.csv` on every clean run.

**Known gaps** (as of 2026-10-06)
- YouTube IP-blocked us after a burst; ~11 videos need a re-run once the block clears.
- 7 transcripts are non-English (flag `lang_xx`) from an earlier fallback bug; fixed in fetch.py,
  they are marked for re-fetch and keep their old text until that succeeds.
- Some videos have transcripts disabled; SoundCloud talk is audio-only.
- Steve Jobs Palo Alto speech is scanned handwriting (would need OCR).
