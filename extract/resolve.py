"""Stage 1: render each antislop page (client-side JS) and record its source link.

    uv run extract/resolve.py [--force] [--limit N]

Saves the rendered HTML to data/fetched/antislop/<id>.html and writes
source_url + key_highlight into the manifest. Items resolved within MAX_AGE are skipped.
"""

import argparse
import asyncio

from playwright.async_api import async_playwright

from common import ANTISLOP_DIR, UA, is_fresh, load_csv, load_manifest, log_run, now, save_manifest

CONCURRENCY = 4


async def resolve_one(browser, row: dict) -> dict:
    page = await browser.new_page(user_agent=UA)
    try:
        await page.goto(row["antislop_url"], wait_until="domcontentloaded", timeout=30_000)
        link = page.locator("h1 a[href]").first
        await link.wait_for(timeout=20_000)
        source_url = await link.get_attribute("href")
        # The one highlight that's visible without logging in; the rest are locked placeholders.
        highlight = None
        key = page.get_by_text("Key Highlight:", exact=False).first
        if await key.count():
            text = await key.locator("xpath=..").inner_text()
            highlight = text.replace("Key Highlight:", "", 1).strip() or None
        (ANTISLOP_DIR / f"{row['id']}.html").write_text(await page.content(), encoding="utf-8")
        return {"source_url": source_url, "key_highlight": highlight,
                "resolve_status": "ok", "resolved_at": now()}
    except Exception as e:  # noqa: BLE001 — record any failure and move on
        return {"resolve_status": f"error: {type(e).__name__}: {e}"[:300], "resolved_at": now()}
    finally:
        await page.close()


async def main(force: bool, limit: int | None) -> None:
    ANTISLOP_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_csv()[:limit]
    manifest = load_manifest()
    todo = [r for r in rows
            if not (manifest.get(r["id"], {}).get("resolve_status") == "ok"
                    and is_fresh(manifest[r["id"]].get("resolved_at"), force))]
    print(f"{len(rows)} rows, {len(todo)} to resolve")

    sem = asyncio.Semaphore(CONCURRENCY)
    async with async_playwright() as p:
        browser = await p.chromium.launch()

        async def run(row):
            async with sem:
                result = await resolve_one(browser, row)
            manifest[row["id"]] = {**manifest.get(row["id"], {}), **row, **result}
            print(f"  {result['resolve_status'][:40]:40} {row['title'][:60]}")

        await asyncio.gather(*(run(r) for r in todo))
        await browser.close()

    save_manifest(manifest)
    failed = sum(1 for r in todo if manifest[r["id"]]["resolve_status"] != "ok")
    log_run("resolve", len(rows), len(todo) - failed, len(rows) - len(todo), failed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-resolve even if fresh")
    ap.add_argument("--limit", type=int, help="only the first N rows (for testing)")
    a = ap.parse_args()
    asyncio.run(main(a.force, a.limit))
