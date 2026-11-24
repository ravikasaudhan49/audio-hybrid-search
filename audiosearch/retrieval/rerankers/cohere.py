"""Cohere Rerank (hosted, default).

The trial tier allows 10 rerank calls/minute: in interactive use a rate limit raises
RerankUnavailable (the caller falls back to the fused order immediately); with
`patient=True` (evaluation) it waits for the window to reset.
"""
import asyncio
from pathlib import Path

from ...core import runtime
from ...core.errors import RerankUnavailable
from ...core.log import get
from .base import Reranker

log = get(__name__)

COHERE_URL = "https://api.cohere.com/v2/rerank"


class CohereReranker(Reranker):
    RATE_WINDOW_S = 61   # trial keys: 10 calls per rolling minute

    def __init__(self, api_key: str, model: str, cache_dir: Path | None = None):
        super().__init__(cache_dir)
        self._api_key = api_key
        self.name = model

    async def _scores(self, query: str, docs: list[str], patient: bool) -> list[float]:
        if not self._api_key:   # checked on use: search still works, in fused order
            log.warning("COHERE_API_KEY is not set; rerank disabled")
            raise RerankUnavailable("COHERE_API_KEY is not set in .env")
        body = {"model": self.name, "query": query, "documents": docs, "top_n": len(docs)}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        for _ in range(3 if patient else 1):
            try:
                r = await runtime.http().post(COHERE_URL, json=body, headers=headers, timeout=30)
            except Exception as e:
                log.warning("cohere unreachable: %s: %s", type(e).__name__, e)
                raise RerankUnavailable(f"Cohere unreachable: {e}") from e
            if r.status_code != 429:
                break
            if not patient:
                log.warning("cohere rate limit (429); falling back to fused order")
                raise RerankUnavailable("Cohere rate limit reached (trial key: 10 calls/min)")
            log.warning("cohere rate limit (429); waiting %ds before retrying", self.RATE_WINDOW_S)
            await asyncio.sleep(self.RATE_WINDOW_S)
        if r.status_code >= 400:
            log.error("cohere error status=%d body=%s", r.status_code, r.text[:300])
            raise RerankUnavailable(f"Cohere error {r.status_code}: {r.text[:200]}")
        out = [0.0] * len(docs)
        for item in r.json()["results"]:
            out[item["index"]] = float(item["relevance_score"])
        return out
