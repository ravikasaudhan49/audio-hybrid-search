"""bge-reranker-base cross-encoder on CPU (runs in a worker thread; weights load on first use)."""
import asyncio
import threading
from pathlib import Path

from .base import Reranker


class LocalReranker(Reranker):
    name = "bge-reranker-base"

    def __init__(self, cache_dir: Path | None = None):
        super().__init__(cache_dir)
        self._model = None
        self._lock = threading.Lock()

    def _predict(self, query: str, docs: list[str]) -> list[float]:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
        return [float(s) for s in self._model.predict([(query, d) for d in docs])]

    async def _scores(self, query: str, docs: list[str], patient: bool) -> list[float]:
        return await asyncio.to_thread(self._predict, query, docs)
