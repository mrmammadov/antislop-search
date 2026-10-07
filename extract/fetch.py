"""Stage 2: download each resolved source into data/fetched/source/.

    uv run extract/fetch.py [--force] [--limit N]

- YouTube        -> <id>.transcript.json (captions via youtube-transcript-api)
- PDF            -> <id>.pdf
- HTML           -> <id>.html; if plain HTTP yields little text (JS-rendered site),
                    the page is re-rendered with Playwright and that HTML is kept.
- HTTP 4xx/5xx,  -> retried from the latest Wayback Machine snapshot (manifest: via=wayback).
  or still thin
  after rendering
Items fetched within MAX_AGE are skipped.
"""

import argparse
import asyncio
import contextlib
import json
import re
from urllib.parse import parse_qs, urlparse

import httpx
import trafilatura
from playwright.async_api import async_playwright
from youtube_transcript_api import YouTubeTranscriptApi

from common import SOURCE_DIR, UA, is_fresh, load_manifest, log_run, now, save_manifest

HTTP_CONCURRENCY = 6
BROWSER_CONCURRENCY = 3
YOUTUBE_DELAY_S = 3  # transcripts are fetched one at a time; YouTube IP-blocks bursts
MIN_TEXT_CHARS = 500  # below this, a plain-HTTP page is treated as JS-rendered


def youtube_id(url: str) -> str | None:
    # Unwrap Wayback links to videos, e.g. web.archive.org/web/2020.../https://www.youtube.com/watch?v=...
    if m := re.match(r"^https?://web\.archive\.org/web/\d+[a-z_]*/(.+)$", url):
        url = m.group(1)
    u = urlparse(url)
    host = u.netloc.removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        return u.path.strip("/").split("/")[0] or None
    if host == "youtube.com":
        if u.path == "/watch":
            return parse_qs(u.query).get("v", [None])[0]
        m = re.match(r"^/(embed|shorts|live|v)/([^/?]+)", u.path)
        return m.group(2) if m else None
    return None


def fetch_youtube(rec: dict, vid: str) -> dict:
    api = YouTubeTranscriptApi()
    transcripts = api.list(vid)
    en = ["en", "en-US", "en-GB", "en-CA", "en-AU"]
    # Prefer: manual English > auto-generated English > any track machine-translated to English.
    # Plain "first track" can be a dubbed/translated language (ru, ar, bn...) on English talks.
    t, translated = None, False
    for find in (transcripts.find_manually_created_transcript, transcripts.find_generated_transcript):
        try:
            t = find(en)
            break
        except Exception:  # noqa: BLE001 — not available, try the next option
            pass
    if t is None:
        tracks = list(transcripts)
        t = next((x for x in tracks if not x.is_generated), tracks[0])
        if t.is_translatable and any(lang.language_code == "en" for lang in t.translation_languages):
            t, translated = t.translate("en"), True
    data = t.fetch()
    out = {"video_id": vid, "language": t.language_code, "is_generated": t.is_generated,
           "translated": translated,
           "snippets": [{"text": s.text, "start": s.start, "duration": s.duration} for s in data]}
    path = SOURCE_DIR / f"{rec['id']}.transcript.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return {"fetch_kind": "youtube_transcript", "fetch_path": path.name}


async def render(browser, url: str) -> str:
    """Rendered HTML, with child-frame documents (e.g. newsletter bodies in <iframe srcdoc>)
    appended before </body> so text extraction sees them."""
    page = await browser.new_page(user_agent=UA)
    try:
        try:
            await page.goto(url, wait_until="networkidle", timeout=45_000)
        except Exception:  # noqa: BLE001 — networkidle can time out on chatty sites; take what loaded
            await page.wait_for_timeout(2000)
        html = await page.content()
        frames = []
        for f in page.frames[1:]:
            try:
                body = await f.locator("body").inner_html(timeout=5000)
            except Exception:  # noqa: BLE001 — cross-origin/detached frames are skipped
                continue
            frames.append(f'<div data-frame-url="{f.url}">{body}</div>')
        if frames:
            html = html.replace("</body>", "\n".join(frames) + "</body>", 1)
        return html
    finally:
        await page.close()


async def wayback(client, url: str) -> httpx.Response:
    """Latest Wayback Machine snapshot of url, as the original bytes (id_ = no toolbar)."""
    meta = (await client.get("https://archive.org/wayback/available", params={"url": url})).json()
    snap = meta.get("archived_snapshots", {}).get("closest")
    if snap and snap.get("available"):
        raw = re.sub(r"/web/(\d+)/", r"/web/\1id_/", snap["url"], count=1)
    else:
        # The availability API often misses pages that do exist; the CDX index is authoritative.
        cdx = await client.get("https://web.archive.org/cdx/search/cdx", params={
            "url": url, "filter": "statuscode:200", "output": "json",
            "fl": "timestamp,original", "collapse": "digest", "limit": "-1"})
        cdx.raise_for_status()
        rows = cdx.json()[1:]  # first row is the header
        if not rows:
            raise RuntimeError("no Wayback snapshot")
        ts, original = rows[-1]
        raw = f"https://web.archive.org/web/{ts}id_/{original}"
    r = await client.get(raw)
    r.raise_for_status()
    return r


async def fetch_one(client, browser, browser_sem, yt_lock, rec: dict) -> dict:
    url = rec["source_url"]
    if vid := youtube_id(url):
        async with yt_lock:
            try:
                return await asyncio.to_thread(fetch_youtube, rec, vid)
            finally:
                await asyncio.sleep(YOUTUBE_DELAY_S)

    via = {}
    try:
        r = await client.get(url)
        r.raise_for_status()
    except httpx.HTTPError as e:
        r = await wayback(client, url)
        via = {"via": "wayback", "origin_error": f"{type(e).__name__}: {e}"[:200]}
    ctype = r.headers.get("content-type", "").lower()
    if "pdf" in ctype or r.content[:5] == b"%PDF-":
        path = SOURCE_DIR / f"{rec['id']}.pdf"
        path.write_bytes(r.content)
        return {"fetch_kind": "pdf", "fetch_path": path.name, "final_url": str(r.url), **via}

    html, kind = r.text, "http"
    text = trafilatura.extract(html) or ""
    if len(text) < MIN_TEXT_CHARS and not via:
        async with browser_sem:
            rendered = await render(browser, url)
        if len(rendered_text := trafilatura.extract(rendered) or "") > len(text):
            html, kind, text = rendered, "browser", rendered_text
    if len(text) < MIN_TEXT_CHARS and not via:
        # Still thin: expired/parked domain, CAPTCHA wall, etc. — try the archived copy.
        try:
            wb = await wayback(client, url)
        except Exception:  # noqa: BLE001 — no snapshot; keep what we have
            wb = None
        if wb is not None:
            if "pdf" in wb.headers.get("content-type", "").lower() or wb.content[:5] == b"%PDF-":
                path = SOURCE_DIR / f"{rec['id']}.pdf"
                path.write_bytes(wb.content)
                return {"fetch_kind": "pdf", "fetch_path": path.name, "final_url": str(wb.url),
                        "via": "wayback", "origin_error": "thin page"}
            if len(trafilatura.extract(wb.text) or "") > len(text):
                html, kind, r = wb.text, "http", wb
                via = {"via": "wayback", "origin_error": "thin page"}
    path = SOURCE_DIR / f"{rec['id']}.html"
    path.write_text(html, encoding="utf-8")
    return {"fetch_kind": kind, "fetch_path": path.name, "final_url": str(r.url), **via}


async def main(force: bool, limit: int | None) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    recs = [r for r in manifest.values() if r.get("resolve_status") == "ok"][:limit]
    todo = [r for r in recs
            if not (r.get("fetch_status") == "ok" and is_fresh(r.get("fetched_at"), force))]
    print(f"{len(recs)} resolved, {len(todo)} to fetch")

    http_sem = asyncio.Semaphore(HTTP_CONCURRENCY)
    browser_sem = asyncio.Semaphore(BROWSER_CONCURRENCY)
    yt_lock = asyncio.Lock()
    async with (
        httpx.AsyncClient(headers={"User-Agent": UA}, follow_redirects=True, timeout=45) as client,
        async_playwright() as p,
    ):
        browser = await p.chromium.launch()

        async def run(rec):
            # YouTube items queue on yt_lock instead of holding an HTTP slot while waiting.
            async with (contextlib.nullcontext() if youtube_id(rec["source_url"]) else http_sem):
                try:
                    result = {**await fetch_one(client, browser, browser_sem, yt_lock, rec), "fetch_status": "ok"}
                except Exception as e:  # noqa: BLE001 — record any failure and move on
                    result = {"fetch_status": f"error: {type(e).__name__}: {e}"[:300]}
            result["fetched_at"] = now()
            if result["fetch_status"] == "ok":
                for k in ("fetch_kind", "fetch_path", "final_url", "via", "origin_error"):
                    rec.pop(k, None)  # drop leftovers from a previous fetch of this item
            # On failure the previous fetch_kind/fetch_path stay, so clean.py keeps the
            # last good copy; fetch_status records the error and the item is retried next run.
            rec.update(result)
            label = result.get("fetch_kind") or result["fetch_status"]
            print(f"  {label[:50]:50} {rec['source_url'][:70]}")

        await asyncio.gather(*(run(r) for r in todo))
        await browser.close()

    save_manifest(manifest)
    failed = sum(1 for r in todo if r["fetch_status"] != "ok")
    log_run("fetch", len(recs), len(todo) - failed, len(recs) - len(todo), failed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-fetch even if fresh")
    ap.add_argument("--limit", type=int, help="only the first N resolved items (for testing)")
    a = ap.parse_args()
    asyncio.run(main(a.force, a.limit))
