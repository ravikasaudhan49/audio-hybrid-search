"""Embedder interface.

Documents (transcript chunks) and queries are embedded differently (Gemini takes a task type).
`embed_query` caches results (memory, then disk) in one place; a subclass only implements
`_embed_queries` and `embed_documents`.
"""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..core.cache import JsonDiskCache
from ..core.log import get, timed

log = get(__name__)


def normalize_rows(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)


class Embedder(ABC):
    name: str
    dim: int
    min_similarity: float = 0.50   # vector-ranker floor; cosine scales differ per model

    def __init__(self, cache_dir: Path | None = None, min_similarity: float | None = None):
        """cache_dir: where query vectors are cached on disk (None = memory only)."""
        self._query_cache: dict[str, np.ndarray] = {}
        self._cache_dir = cache_dir
        if min_similarity is not None:
            self.min_similarity = min_similarity

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        """L2-normalized row per text."""

    @abstractmethod
    async def _embed_queries(self, texts: list[str]) -> np.ndarray:
        """L2-normalized row per query (uncached)."""

    def _disk_cache(self) -> JsonDiskCache | None:
        # Per model AND dimension: a different truncation is a different vector space.
        if self._cache_dir is None:
            return None
        return JsonDiskCache(self._cache_dir / "query_embeddings" / f"{self.name}-{self.dim}")

    async def embed_query(self, text: str) -> np.ndarray:
        if text in self._query_cache:
            log.debug("query embedding memory hit model=%s", self.name)
            return self._query_cache[text]
        disk = self._disk_cache()
        key = JsonDiskCache.key_of_text(text)
        cached = disk.get(key) if disk else None
        if cached is not None:
            log.debug("query embedding disk hit model=%s", self.name)
            vec = np.array(cached, dtype=np.float32)
        else:
            with timed(log, "query embedding", model=self.name):
                vec = (await self._embed_queries([text]))[0]
            if disk:
                disk.put(key, vec.tolist())
        self._query_cache[text] = vec
        return vec
