"""Gemini embeddings: async HTTP with bounded parallelism and free-tier backoff."""
import asyncio
from pathlib import Path

import numpy as np

from ..core import runtime
from ..core.errors import MissingApiKeyError
from ..core.log import get, timed
from .base import Embedder, normalize_rows

log = get(__name__)


class GeminiEmbedder(Embedder):
    name = "gemini-embedding-001"
    dim = 1536  # Matryoshka-truncated from 3072 (HNSW supports <= 2000); must be re-normalized.
    URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
    BATCH = 100          # API maximum requests per batch call
    MAX_RETRIES = 6

    def __init__(self, api_key: str, concurrency: int = 4, cache_dir: Path | None = None,
                 min_similarity: float | None = None):
        super().__init__(cache_dir, min_similarity)
        self._api_key = api_key   # checked on first use, so the app can start without it
        self._concurrency = concurrency

    async def _batch(self, texts: list[str], task: str) -> list[list[float]]:
        if not self._api_key:
            raise MissingApiKeyError("GEMINI_API_KEY is not set in .env")
        body = {"requests": [{"model": "models/gemini-embedding-001",
                              "content": {"parts": [{"text": t}]},
                              "taskType": task,
                              "outputDimensionality": self.dim} for t in texts]}
        for attempt in range(self.MAX_RETRIES):
            r = await runtime.http().post(self.URL, json=body, headers={"x-goog-api-key": self._api_key})
            if r.status_code not in (429, 500, 503):
                break
            wait = 2 ** attempt * 5  # free-tier rate limits: exponential backoff
            log.warning("gemini %d on batch of %d (%s); retry %d/%d in %ds", r.status_code, len(texts), task,
                        attempt + 1, self.MAX_RETRIES, wait)
            await asyncio.sleep(wait)
        if r.status_code >= 400:
            log.error("gemini error status=%d body=%s", r.status_code, r.text[:300])
        r.raise_for_status()
        return [e["values"] for e in r.json()["embeddings"]]

    async def _call(self, texts: list[str], task: str) -> np.ndarray:
        sem = asyncio.Semaphore(self._concurrency)

        async def one(chunk: list[str]):
            async with sem:
                return await self._batch(chunk, task)

        parts = await asyncio.gather(*(one(texts[i:i + self.BATCH]) for i in range(0, len(texts), self.BATCH)))
        return normalize_rows(np.array([v for p in parts for v in p], dtype=np.float32))

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        with timed(log, "gemini document embedding", n=len(texts)):
            return await self._call(texts, "RETRIEVAL_DOCUMENT")

    async def _embed_queries(self, texts: list[str]) -> np.ndarray:
        return await self._call(texts, "RETRIEVAL_QUERY")
