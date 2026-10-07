"""Shared paths, manifest and run-tracker helpers for the extraction pipeline."""

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_CSV = DATA / "raw" / "antislop.csv"
FETCHED = DATA / "fetched"
ANTISLOP_DIR = FETCHED / "antislop"
SOURCE_DIR = FETCHED / "source"
MANIFEST = FETCHED / "manifest.jsonl"
CLEAN = DATA / "clean" / "docs.jsonl"
TRACKER = DATA / "LAST_EXTRACTED.txt"

# An item fetched more recently than this is reused instead of re-fetched.
MAX_AGE = timedelta(days=30)

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_fresh(ts: str | None, force: bool) -> bool:
    if force or not ts:
        return False
    return datetime.now(timezone.utc) - datetime.fromisoformat(ts) < MAX_AGE


def load_csv() -> list[dict]:
    """Rows from the antislop export, with the scraper's column names mapped to real ones."""
    rows = []
    with RAW_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            url = r["absolute href"].strip()
            author, _, date = r["mt-0"].partition(" · ")
            kind, _, read_time = r["mt-0 2"].partition(" · ")
            rows.append({
                "id": url.rstrip("/").rsplit("/", 1)[-1],
                "antislop_url": url,
                "title": r["underline"].strip(),
                "author": author.strip(),
                "date": date.strip(),
                "type": kind.strip(),
                "read_time": read_time.strip(),
                "summary": r["mt-2"].strip(),
                "image": r["object-cover src"].strip(),
            })
    return rows


def load_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(encoding="utf-8") as f:
        return {(rec := json.loads(line))["id"]: rec for line in f if line.strip()}


def save_manifest(manifest: dict[str, dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in sorted(manifest.values(), key=lambda r: r["id"]):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(MANIFEST)


def log_run(stage: str, total: int, done: int, cached: int, failed: int) -> None:
    line = (f"{now()} | {stage} | {RAW_CSV.name} | total={total} "
            f"done={done} cached={cached} failed={failed}\n")
    with TRACKER.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="")
