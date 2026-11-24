"""Hybrid retrieval: full-text + trigram (typos) + vector, fused with weighted RRF,
then optionally reranked (Cohere by default) and diversified with MMR.

Retrieval runs on CHILD chunks (~128 tokens, one speaker); each child carries its PARENT
(~512 tokens of dialogue) for context. Pipeline per query (one method per stage below):
  1. understand  detect "what did <name>/the guest say about X" -> speaker/role filter
  2. embed       embed the query (cached; never while holding a DB connection)
  3. retrieve    one pooled connection: rankers back to back (~1-3 ms each), weighted RRF,
                 top candidates hydrated on the same connection
  4. group       keep the best child per parent_id                          [on by default]
  5. rerank      a cross-encoder scores (query, parent_text): the full exchange, so an
                 answer that never repeats the question's words still ranks [on by default]
  6. threshold   drop results below the reranker relevance cut-off (ADR-0004)
  7. mmr         pick results that are relevant AND unlike those already picked [opt-in]

Every result is one child: a single speaker, a file and a tight time span (keyword hits
get the exact word timestamp), plus its parent dialogue as context.
"""
import time

import numpy as np
from psycopg_pool import AsyncConnectionPool

from ..core.errors import RerankUnavailable
from ..core.log import get
from ..core.settings import Settings, get_settings
from ..db import SegmentRepository, SpeakerRepository, collection_conn, schema_name
from ..embeddings import Embedder
from ..models import METHODS, Filters, Result, SearchOutput, SearchRequest
from .fusion import apply_thresholds, group_by_parent, mmr_select, rrf, unit_range
from .highlight import locate, query_terms
from .query_parser import parse as parse_query
from .rankers import FuzzyRanker, KeywordRanker, Ranker, VectorRanker
from .rerankers import Reranker

log = get(__name__)


class SpeakerNameCache:
    """Distinct speaker names per collection, cached for `ttl` seconds: query understanding
    needs them on every search, but they only change when a file is ingested."""

    def __init__(self, pool: AsyncConnectionPool, ttl: float = 30.0):
        self._pool = pool
        self._ttl = ttl
        self._entries: dict[str, tuple[float, list[str]]] = {}

    async def names(self, collection: str) -> list[str]:
        hit = self._entries.get(collection)
        if hit and time.monotonic() - hit[0] < self._ttl:
            return hit[1]
        async with collection_conn(self._pool, collection) as conn:
            names = await SpeakerRepository(conn).distinct_names()
        self._entries[collection] = (time.monotonic(), names)
        return names


class _StageTimer:
    """Per-stage wall time in ms (understand, embed, retrieve, rerank, post, total)."""

    def __init__(self):
        self.t0 = self._mark = time.perf_counter()
        self.timings: dict[str, float] = {}

    def lap(self, stage: str) -> None:
        now = time.perf_counter()
        self.timings[stage] = round((now - self._mark) * 1000, 1)
        self._mark = now

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000


class SearchService:
    """Hybrid search over one collection. Cheap to construct (per request): the embedder and
    reranker are built once by AppContainer and passed in."""

    def __init__(self, pool: AsyncConnectionPool, *, embedder: Embedder, reranker: Reranker,
                 collection: str | None = None, origin: str | None = None, settings: Settings | None = None,
                 speakers: SpeakerNameCache | None = None):
        """collection selects the dataset (default: DEFAULT_COLLECTION);
        origin='golden' restricts every query to the evaluation dataset."""
        self.pool = pool
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings or get_settings()
        self.collection = collection or self.settings.default_collection
        schema_name(self.collection)  # validate early
        self.origin = origin
        self.speakers = speakers or SpeakerNameCache(pool)

    async def search(self, req: SearchRequest) -> SearchOutput:
        if req.method not in METHODS:
            raise ValueError(f"unknown method {req.method!r}; use one of {METHODS}")
        s = self.settings
        rerank = s.rerank_default if req.rerank is None else req.rerank
        lam = s.mmr_lambda if req.mmr_lambda is None else req.mmr_lambda
        timer = _StageTimer()

        # 1. understand
        text, speaker, role, detected = req.q, req.speaker, req.role, None
        if req.auto_speaker and not speaker and not role:
            detected = parse_query(req.q, await self.speakers.names(self.collection))
            if detected.speaker or detected.role:
                text, speaker, role = detected.text, detected.speaker, detected.role
                log.info("query understanding: %r -> speaker=%s role=%s text=%r", req.q, speaker, role, text)
        filters = Filters(speaker, req.file_id, self.origin, role)
        log.info("search [%s] q=%r method=%s k=%d group=%s rerank=%s mmr=%s%s embedder=%s", self.collection,
                 req.q[:120], req.method, req.k, req.group,
                 rerank, f"{req.mmr}(λ={lam})" if req.mmr else req.mmr,
                 "".join(f" {n}={v}" for n, v in vars(filters).items() if v), self.embedder.name)
        timer.lap("understand")

        # 2. embed (before taking a connection: a slow API call must not hold a pool slot)
        emb = self.embedder if req.method in ("vector", "hybrid") or req.mmr else None
        rankers = await self._rankers(req.method, text, emb)
        timer.lap("embed")

        # 3. retrieve
        depth = max(req.k * 3, s.rerank_depth) if (rerank or req.mmr or req.group) else req.k
        results, vectors = await self._retrieve(rankers, text, filters, depth,
                                                emb.name if req.mmr else None, timer)
        timer.lap("retrieve")

        out = SearchOutput(results, search_text=text,
                           speaker_detected=detected.speaker if detected else None,
                           role_detected=detected.role if detected else None)
        # 4. group
        if req.group and results:
            out.results = group_by_parent(out.results)
            out.grouped = True
            log.info("grouped %d children into %d parents", len(results), len(out.results))
        out.results = out.results[:s.rerank_depth] if (rerank or req.mmr) else out.results[:req.k]
        # 5. rerank: the reranker gets the full question ("Gene Kranz: ..." context makes the name useful)
        if rerank and out.results:
            await self._rerank(req.q, out, req.patient_rerank)
        timer.lap("rerank")
        # 6. threshold
        if req.threshold and out.reranked:
            self._apply_threshold(out)
        # 7. mmr
        if req.mmr and out.results:
            self._diversify(out, vectors, emb, req.k, lam)
        out.results = out.results[:req.k]
        timer.lap("post")

        timer.timings["total"] = round(timer.elapsed_ms(), 1)
        out.timings_ms = timer.timings
        top = out.results[0] if out.results else None
        log.info("search done in %.0f ms: %d results reranked=%s mmr=%s top=%s", timer.elapsed_ms(),
                 len(out.results), out.reranked, out.mmr,
                 f"{top.file_id}@{top.timestamp}/{top.speaker}" if top else None)
        for note in out.notes:
            log.warning("search note: %s", note)
        return out

    # ---------- stages ----------

    async def _rankers(self, method: str, text: str, emb: Embedder | None) -> list[Ranker]:
        rankers: list[Ranker] = []
        if method in ("keyword", "hybrid"):
            rankers.append(KeywordRanker())
        # In hybrid mode the typo ranker only runs for short lookups (see Settings.fuzzy_max_terms).
        if method == "fuzzy" or (method == "hybrid" and len(query_terms(text)) <= self.settings.fuzzy_max_terms):
            rankers.append(FuzzyRanker())
        if method in ("vector", "hybrid"):
            rankers.append(VectorRanker(emb, await emb.embed_query(text)))
        return rankers

    async def _retrieve(self, rankers: list[Ranker], text: str, filters: Filters, depth: int,
                        vector_model: str | None, timer: _StageTimer) -> tuple[list[Result], dict[int, np.ndarray]]:
        """One pooled connection for all rankers, fusion and hydration (ADR-0006)."""
        async with collection_conn(self.pool, self.collection) as conn:
            segments = SegmentRepository(conn)
            ranked = {r.name: await segments.ranked_ids(r.query(text, filters)) for r in rankers}
            fused = rrf(ranked, {r.name: r.weight for r in rankers})[:depth]
            rows = await segments.fetch_for_results([sid for sid, _, _ in fused], vector_model)
        log.info("retrieval %s -> fused %d candidates in %.0f ms",
                 " ".join(f"{m}={len(ids)}" for m, ids in ranked.items()), len(fused), timer.elapsed_ms())
        return self._hydrate(text, fused, rows)

    @staticmethod
    def _hydrate(text: str, fused: list[tuple[int, float, dict]],
                 rows: dict[int, dict]) -> tuple[list[Result], dict[int, np.ndarray]]:
        """Results in fused order, plus stored vectors (for MMR) where requested."""
        terms = query_terms(text)
        results, vectors = [], {}
        for sid, score, src in fused:
            row = rows[sid]
            hit, highlighted = locate(row["words"], terms, row["start_s"])
            results.append(Result(
                segment_id=sid, file_id=row["file_id"], file_title=row["title"], audio_path=row["audio_path"],
                speaker=row["name"], start_s=row["start_s"], end_s=row["end_s"], hit_s=hit,
                child_text=row["child_text"], highlighted=highlighted, score=score, sources=dict(src),
                parent_id=row["parent_id"], parent_start_s=row["parent_start_s"], parent_end_s=row["parent_end_s"],
                parent_text=row["parent_text"], audio_url=row["audio_url"], source_offset_s=row["source_offset_s"]))
            vec = row["embedding"]
            if vec is not None:
                # untyped `vector` column: pgvector returns a Vector object, not an ndarray
                vectors[sid] = np.asarray(vec.to_numpy() if hasattr(vec, "to_numpy") else vec, dtype=np.float32)
        return results, vectors

    async def _rerank(self, q: str, out: SearchOutput, patient: bool) -> None:
        # Grouped: one result per parent, so rerank the parent dialogue (question + answer).
        # Ungrouped: siblings share a parent, so rerank each child ("Speaker: text") instead.
        docs = ([r.parent_text for r in out.results] if out.grouped
                else [f"{r.speaker}: {r.child_text}" for r in out.results])
        try:
            scores = await self.reranker.scores(q, docs, patient=patient)
        except RerankUnavailable as e:
            out.notes.append(f"rerank skipped, showing hybrid order: {e}")
            return
        for r, sc in zip(out.results, scores, strict=True):
            r.score = sc
        out.results.sort(key=lambda r: -r.score)
        for rank, r in enumerate(out.results, 1):
            r.sources["rerank"] = rank
        out.reranked, out.reranker = True, self.reranker.name

    def _apply_threshold(self, out: SearchOutput) -> None:
        before = len(out.results)
        out.results, cutoff = apply_thresholds(out.results, self.settings.rerank_min_score,
                                               self.settings.rerank_min_relative)
        out.below_threshold = before - len(out.results)
        if out.below_threshold:
            log.info("threshold: dropped %d of %d results below relevance %.2f", out.below_threshold, before, cutoff)
        if not out.results:
            out.notes.append(f"no confident match: every candidate scored below the relevance "
                             f"threshold ({cutoff:.2f})")

    @staticmethod
    def _diversify(out: SearchOutput, vectors: dict[int, np.ndarray], emb: Embedder, k: int, lam: float) -> None:
        cands = out.results
        relevance = unit_range(np.array([r.score for r in cands], dtype=np.float32))
        mat = np.stack([vectors.get(r.segment_id, np.zeros(emb.dim, np.float32)) for r in cands])
        order = mmr_select(relevance, mat, k, lam)
        out.results = [cands[i] for i in order]
        for rank, r in enumerate(out.results, 1):
            r.sources["mmr"] = rank
        out.mmr = True
        log.info("mmr λ=%.2f picked %d of %d candidates (missing vectors: %d)", lam, len(order),
                 len(cands), sum(1 for r in cands if r.segment_id not in vectors))
