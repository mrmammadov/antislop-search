"""Local web app over the antislop corpus.

    uv run uvicorn app.server:app --reload        # then open http://127.0.0.1:8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.search import SearchService, options

STATIC = Path(__file__).resolve().parent / "static"
service: SearchService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    service = SearchService()
    yield
    service.close()


app = FastAPI(title="antislop search", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/options")
def get_options() -> dict:
    return {**options(), "n_docs": len(service.docs)}


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, max_length=500),
    engine: str = "memory", retrieval: str = "hybrid", chunker: str = "para256",
    embedder: str = "bge-small", k: int = Query(20, ge=1, le=100),
) -> dict:
    # Plain `def`: FastAPI runs it in a thread pool, so a slow first build doesn't block the server.
    try:
        cfg = service.config(engine, retrieval, chunker, embedder)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        return service.search(q, cfg, k)
    except Exception as e:  # noqa: BLE001 — e.g. Docker not running for a docker engine
        raise HTTPException(503, f"{type(e).__name__}: {e}")
