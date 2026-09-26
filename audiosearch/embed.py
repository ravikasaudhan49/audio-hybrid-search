"""Pluggable embedders. Select with EMBEDDER=gemini | bge-base | bge-small.

Documents (transcript chunks) and queries are embedded differently: Gemini takes a
task type, BGE takes an instruction prefix on the query side only.
Query vectors are cached in memory and on disk so repeated queries cost nothing.

Gemini calls are async HTTP with bounded parallelism; BGE inference is CPU-bound,
so it runs in a worker thread to keep the event loop free.
"""
import asyncio
import hashlib
import json
import threading

import numpy as np

from . import aio, config
from .log import get, timed

log = get(__name__)


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)


class Embedder:
    name: str
    dim: int
    min_similarity: float = config.VECTOR_MIN_SIM   # vector-ranker floor; cosine scales differ per model

    def __init__(self):
        self._query_cache: dict[str, np.ndarray] = {}

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    async def _embed_queries(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    async def embed_query(self, text: str) -> np.ndarray:
        if text in self._query_cache:
            log.debug("query embedding memory hit model=%s", self.name)
            return self._query_cache[text]
        path = (config.CACHE_DIR / "query_embeddings" / f"{self.name}-{self.dim}"
                / f"{hashlib.sha1(text.encode()).hexdigest()}.json")
        if path.exists():
            log.debug("query embedding disk hit model=%s", self.name)
            vec = np.array(json.loads(path.read_text()), dtype=np.float32)
        else:
            with timed(log, "query embedding", model=self.name):
                vec = (await self._embed_queries([text]))[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(vec.tolist()))
        self._query_cache[text] = vec
        return vec


class GeminiEmbedder(Embedder):
    name = "gemini-embedding-001"
    dim = 1536  # Matryoshka-truncated from 3072 (HNSW supports <= 2000); must be re-normalized.
    URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
    BATCH = 100          # API maximum requests per batch call
    MAX_RETRIES = 6

    def __init__(self):
        super().__init__()
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")

    async def _batch(self, texts: list[str], task: str) -> list[list[float]]:
        body = {"requests": [{"model": "models/gemini-embedding-001",
                              "content": {"parts": [{"text": t}]},
                              "taskType": task,
                              "outputDimensionality": self.dim} for t in texts]}
        for attempt in range(self.MAX_RETRIES):
            r = await aio.http().post(self.URL, json=body, headers={"x-goog-api-key": config.GEMINI_API_KEY})
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
        sem = asyncio.Semaphore(config.EMBED_CONCURRENCY)

        async def one(chunk: list[str]):
            async with sem:
                return await self._batch(chunk, task)

        parts = await asyncio.gather(*(one(texts[i:i + self.BATCH]) for i in range(0, len(texts), self.BATCH)))
        return _normalize(np.array([v for p in parts for v in p], dtype=np.float32))

    async def embed_documents(self, texts):
        with timed(log, "gemini document embedding", n=len(texts)):
            return await self._call(texts, "RETRIEVAL_DOCUMENT")

    async def _embed_queries(self, texts):
        return await self._call(texts, "RETRIEVAL_QUERY")


class BGEEmbedder(Embedder):
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, size: str = "base"):
        super().__init__()
        from sentence_transformers import SentenceTransformer  # heavy import, only when used
        self.name = f"bge-{size}-en-v1.5"
        with timed(log, "load local embedding model", model=self.name):
            self.model = SentenceTransformer(f"BAAI/{self.name}", device="cpu")
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str], **kw) -> np.ndarray:
        return self.model.encode(texts, batch_size=16, normalize_embeddings=True, **kw).astype(np.float32)

    async def embed_documents(self, texts):
        with timed(log, "bge document embedding", model=self.name, n=len(texts)):
            return await asyncio.to_thread(self._encode, texts, show_progress_bar=True)

    async def _embed_queries(self, texts):
        return await asyncio.to_thread(self._encode, [self.QUERY_PREFIX + t for t in texts])


_embedders: dict[str, Embedder] = {}
_load_lock = threading.Lock()


def _load(name: str) -> Embedder:
    with _load_lock:  # one instance per model per process; BGE weights load only once
        if name not in _embedders:
            if name == "gemini":
                _embedders[name] = GeminiEmbedder()
            elif name in ("bge-base", "bge-small"):
                _embedders[name] = BGEEmbedder(name.split("-")[1])
            else:
                raise ValueError(f"unknown embedder {name!r}; use gemini | bge-base | bge-small")
        return _embedders[name]


async def get_embedder(name: str | None = None) -> Embedder:
    name = name or config.EMBEDDER
    return _embedders.get(name) or await asyncio.to_thread(_load, name)
