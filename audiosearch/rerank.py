"""Second-stage rerankers: Cohere Rerank (default, hosted) or a local cross-encoder.

Rerankers score (query, chunk) pairs jointly, which is more precise than comparing
independently computed vectors, but too slow/costly for the whole corpus, so they only
re-score the top fused candidates.

Scores are cached on disk per (model, query, candidates), so repeated queries and eval
runs don't spend API calls. The Cohere trial tier allows 10 rerank calls/minute: in
interactive use a rate-limit raises RerankUnavailable (the caller falls back to the
fused order immediately); `patient=True` (evaluation) waits for the window to reset.
"""
import asyncio
import hashlib
import json
import threading

from . import aio, config
from .log import get, timed

log = get(__name__)

COHERE_URL = "https://api.cohere.com/v2/rerank"


class RerankUnavailable(Exception):
    """Reranking could not run right now (rate limit, network, missing key)."""


class Reranker:
    name: str

    async def scores(self, query: str, docs: list[str]) -> list[float]:
        """Relevance score per doc, in input order (higher = more relevant)."""
        key = hashlib.sha1(json.dumps([self.name, query, docs]).encode()).hexdigest()
        path = config.CACHE_DIR / "rerank" / f"{key}.json"
        if path.exists():
            log.info("rerank cache hit model=%s docs=%d", self.name, len(docs))
            return json.loads(path.read_text())
        with timed(log, "rerank", expected=(RerankUnavailable,), model=self.name, docs=len(docs)):
            out = await self._scores(query, docs)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        return out

    async def _scores(self, query: str, docs: list[str]) -> list[float]:
        raise NotImplementedError


class CohereReranker(Reranker):
    RATE_WINDOW_S = 61   # trial keys: 10 calls per rolling minute

    def __init__(self, patient: bool = False):
        if not config.COHERE_API_KEY:
            log.warning("COHERE_API_KEY is not set; rerank disabled")
            raise RerankUnavailable("COHERE_API_KEY is not set in .env")
        self.name = config.COHERE_RERANK_MODEL
        self.patient = patient

    async def _scores(self, query, docs):
        body = {"model": self.name, "query": query, "documents": docs, "top_n": len(docs)}
        headers = {"Authorization": f"Bearer {config.COHERE_API_KEY}"}
        for _ in range(3 if self.patient else 1):
            try:
                r = await aio.http().post(COHERE_URL, json=body, headers=headers, timeout=30)
            except Exception as e:
                log.warning("cohere unreachable: %s: %s", type(e).__name__, e)
                raise RerankUnavailable(f"Cohere unreachable: {e}") from e
            if r.status_code != 429:
                break
            if not self.patient:
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


class LocalReranker(Reranker):
    """bge-reranker-base cross-encoder on CPU (runs in a worker thread)."""
    name = "bge-reranker-base"
    _model = None
    _lock = threading.Lock()

    def _predict(self, query: str, docs: list[str]) -> list[float]:
        with self._lock:
            if LocalReranker._model is None:
                from sentence_transformers import CrossEncoder
                LocalReranker._model = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
        return [float(s) for s in LocalReranker._model.predict([(query, d) for d in docs])]

    async def _scores(self, query, docs):
        return await asyncio.to_thread(self._predict, query, docs)


def get_reranker(provider: str | None = None, patient: bool = False) -> Reranker:
    provider = provider or config.RERANK_PROVIDER
    if provider == "cohere":
        return CohereReranker(patient)
    if provider == "local":
        return LocalReranker()
    raise ValueError(f"unknown rerank provider {provider!r}; use cohere | local")
