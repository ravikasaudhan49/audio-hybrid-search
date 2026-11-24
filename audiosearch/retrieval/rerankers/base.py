"""Second-stage reranker interface.

Rerankers score (query, chunk) pairs jointly, which is more precise than comparing
independently computed vectors, but too slow/costly for the whole corpus, so they only
re-score the top fused candidates.

`scores` is a Template Method: the on-disk cache per (model, query, candidates) is handled
here, so repeated queries and eval runs don't spend API calls; subclasses implement `_scores`.
"""
from abc import ABC, abstractmethod
from pathlib import Path

from ...core.cache import JsonDiskCache
from ...core.errors import RerankUnavailable
from ...core.log import get, timed

log = get(__name__)


class Reranker(ABC):
    name: str

    def __init__(self, cache_dir: Path | None = None):
        self._cache = JsonDiskCache(cache_dir / "rerank") if cache_dir else None

    async def scores(self, query: str, docs: list[str], patient: bool = False) -> list[float]:
        """Relevance score per doc, in input order (higher = more relevant).
        patient=True waits out rate limits (evaluation) instead of raising RerankUnavailable."""
        key = JsonDiskCache.key_of(self.name, query, docs)
        if self._cache and (hit := self._cache.get(key)) is not None:
            log.info("rerank cache hit model=%s docs=%d", self.name, len(docs))
            return hit
        with timed(log, "rerank", expected=(RerankUnavailable,), model=self.name, docs=len(docs)):
            out = await self._scores(query, docs, patient)
        if self._cache:
            self._cache.put(key, out)
        return out

    @abstractmethod
    async def _scores(self, query: str, docs: list[str], patient: bool) -> list[float]:
        """Uncached scores."""
