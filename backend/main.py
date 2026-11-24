"""FastAPI backend (fully async).

Run:  uvicorn backend.main:app --port 8000 --loop asyncio:SelectorEventLoop     (docs at /docs)
      (the selector loop is required by psycopg's async driver on Windows)

Thin HTTP layer over the `audiosearch` library: the lifespan opens one AppContainer (pool +
services) and one JobManager per app; routes live in backend/routers/, request/response
models in backend/schemas.py. Slow work (transcription, embedding, eval) runs as jobs.
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from audiosearch.container import AppContainer
from audiosearch.core import log as logs

from .jobs import JobManager
from .middleware import log_requests
from .routers import ROUTERS

logs.setup("api")
log = logs.get("backend.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    container = AppContainer()
    s = container.settings
    log.info("API starting: embedder=%s reranker=%s (default %s)", s.embedder, container.reranker.name,
             "on" if s.rerank_default else "off")
    await container.start()
    app.state.container = container
    app.state.jobs = JobManager(s.job_concurrency)
    try:
        yield
    finally:
        log.info("API shutting down")
        await app.state.jobs.shutdown()
        await container.close()


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(title="Audio Transcript Search API", version="1.0", lifespan=lifespan,
                  description="Hybrid (keyword + semantic) search over two-speaker audio transcripts.")
    app.middleware("http")(log_requests)
    for router in ROUTERS:
        app.include_router(router)
    return app


app = create_app()
