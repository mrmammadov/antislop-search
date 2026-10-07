"""Stage 3 (offline): turn everything in data/fetched/ into data/clean/docs.jsonl.

    uv run extract/clean.py

Never touches the network, so it can be re-run freely and is deterministic for a
given fetched/ snapshot. Items that failed to fetch are still emitted (empty text,
text_source="none") so downstream code sees all 177 resources. Flagged items are
also listed in data/clean/flagged.csv.

Each doc gets quality flags comparing its word count to antislop's read time:
  too_short  — got < 30% of the expected words (paywall, wrong page, partial render)
  too_long   — got > 3x (whole book PDF, page chrome, comments)
  lang_<xx>  — YouTube transcript isn't English
"""

import csv
import html
import json
import re
import unicodedata

import pymupdf
import trafilatura

from common import CLEAN, SOURCE_DIR, load_manifest

READ_WPM = 230   # antislop read times for text
SPEAK_WPM = 150  # for video/lecture/speech transcripts
SPOKEN_TYPES = {"Video", "Lecture"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")                         # soft hyphens
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def from_transcript(path) -> tuple[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    parts = (html.unescape(s["text"]).replace("\n", " ") for s in data["snippets"])
    text = " ".join(p for p in parts if not re.fullmatch(r"\[[^\]]*\]", p.strip()))  # [Music], [Applause]
    return re.sub(r"\s+", " ", text), data["language"]


def from_pdf(path) -> str:
    with pymupdf.open(path) as doc:
        pages = [page.get_text("text", sort=True) for page in doc]
    text = "\n\n".join(pages)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)              # re-join hyphenated line breaks
    text = re.sub(r"(?<![.!?:\"”])\n(?!\n)", " ", text)       # unwrap hard-wrapped lines
    return text


def from_html(path, url: str) -> tuple[str, str | None]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = trafilatura.extract(raw, url=url, include_comments=False, include_tables=True,
                               favor_precision=True, deduplicate=True) or ""
    if len(text) < 200:  # precision mode can be too strict on unusual layouts
        text = trafilatura.extract(raw, url=url, include_comments=False, favor_recall=True) or text
    meta = trafilatura.extract_metadata(raw)
    return text, (meta.title if meta else None)


def expected_words(rec: dict) -> int | None:
    m = re.match(r"(\d+)\s*min", rec.get("read_time", ""))
    if not m:
        return None
    wpm = SPEAK_WPM if rec["type"] in SPOKEN_TYPES or rec.get("fetch_kind") == "youtube_transcript" else READ_WPM
    return int(m.group(1)) * wpm


def clean_one(rec: dict) -> dict:
    kind, page_title, text, lang = rec.get("fetch_kind"), None, "", None
    # Use the last good copy even if the latest re-fetch failed (fetch_error says so).
    if rec.get("fetch_path") and (path := SOURCE_DIR / rec["fetch_path"]).exists():
        if kind == "youtube_transcript":
            text, lang = from_transcript(path)
        elif kind == "pdf":
            text = from_pdf(path)
        else:
            text, page_title = from_html(path, rec.get("final_url") or rec["source_url"])
    text = normalize(text)

    n_words = len(text.split())
    flags = []
    if not text:
        flags.append("no_text")
    elif exp := expected_words(rec):
        ratio = n_words / exp
        if ratio < 0.3:
            flags.append("too_short")
        elif ratio > 3:
            flags.append("too_long")
    if kind == "youtube_transcript" and text and not lang.startswith("en"):
        flags.append(f"lang_{lang}")

    return {
        "id": rec["id"],
        "url": rec.get("source_url"),
        "title": rec["title"],
        "author": rec["author"],
        "text": text,
        "fetched_at": rec.get("fetched_at"),
        "metadata": {
            "antislop_url": rec["antislop_url"],
            "date": rec["date"],
            "type": rec["type"],
            "read_time": rec["read_time"],
            "summary": rec["summary"],
            "key_highlight": rec.get("key_highlight"),
            "image": rec["image"],
            "page_title": page_title,
            "text_source": kind or "none",
            "via": rec.get("via"),
            "fetch_error": None if rec.get("fetch_status") == "ok" else rec.get("fetch_status"),
            "n_words": n_words,
            "expected_words": expected_words(rec),
            "flags": flags,
        },
    }


def main() -> None:
    docs = [clean_one(r) for r in sorted(load_manifest().values(), key=lambda r: r["id"])]
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    with CLEAN.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    flagged = [d for d in docs if d["metadata"]["flags"]]
    # Human-readable list of what needs attention; regenerated every run.
    with (CLEAN.parent / "flagged.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["flags", "n_words", "expected_words", "text_source", "type", "title", "url",
                    "fetch_error"])
        for d in flagged:
            m = d["metadata"]
            w.writerow([",".join(m["flags"]), m["n_words"], m["expected_words"], m["text_source"],
                        m["type"], d["title"], d["url"], (m["fetch_error"] or "").split("\n")[0]])
    print(f"wrote {len(docs)} docs -> {CLEAN.relative_to(CLEAN.parents[2])}")
    print(f"  ok: {len(docs) - len(flagged)}   flagged: {len(flagged)}")
    for d in flagged:
        m = d["metadata"]
        print(f"  {','.join(m['flags']):10} {m['n_words']:>6}/{m['expected_words'] or '?':<6} "
              f"{m['text_source']:18} {d['title'][:45]:45} {d['url'][:50]}")


if __name__ == "__main__":
    main()
