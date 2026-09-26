"""Hybrid retrieval: full-text + trigram (typos) + vector, fused with weighted RRF,
then optionally reranked (Cohere by default) and diversified with MMR.

Retrieval runs on CHILD chunks (~128 tokens, one speaker); each child carries its PARENT
(~512 tokens of dialogue) for context. Pipeline per query:
  1. embed the query (cached; never while holding a DB connection)
  2. one pooled connection: the ranker queries run back to back (~1-3 ms each)
  3. weighted Reciprocal Rank Fusion -> top child candidates, hydrated on the same connection
  4. group by parent_id: keep the best child per parent            [on by default]
  5. rerank: a cross-encoder scores (query, parent_text) pairs -- the full exchange, so an
     answer that never repeats the question's words still ranks   [on by default]
  6. MMR: pick results that are relevant AND unlike those already picked [opt-in]

Every result is one child: a single speaker, a file and a tight time span (keyword hits
get the exact word timestamp), plus its parent dialogue as context.
"""
import re
import time
from dataclasses import dataclass, field

import numpy as np
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from . import config, db
from .embed import Embedder, get_embedder
from .log import get
from .query import parse as parse_query
from .rerank import RerankUnavailable, get_reranker

log = get(__name__)

RRF_K = 60
CANDIDATES = 50       # per-ranker candidate depth fed into fusion
WEIGHTS = {"keyword": 1.0, "vector": 1.0, "fuzzy": 0.5}
METHODS = ("hybrid", "keyword", "vector", "fuzzy")

STOPWORDS = set("""a an the and or but if of to in on at by for with about from into over
is are was were be been being do does did have has had it its this that these those
i you he she we they me him her us them my your his our their what which who whom how
why when where there here not no so as than then too very can could would should will
just also say said tell talk talked talking""".split())


@dataclass
class Result:
    segment_id: int
    file_id: str
    file_title: str
    audio_path: str
    speaker: str
    start_s: float
    end_s: float
    hit_s: float          # exact word time for keyword hits, else segment start
    child_text: str
    highlighted: str      # child_text with matched words wrapped in [[ ]]
    score: float          # fused RRF score, or the reranker's relevance score when reranked
    sources: dict         # stage -> rank (1-based): keyword/fuzzy/vector, then rerank, mmr
    parent_id: str
    parent_start_s: float
    parent_end_s: float
    parent_text: str
    grouped: int = 1      # children of this parent that matched (collapsed into this result)
    audio_url: str | None = None    # the original episode's audio
    source_offset_s: float = 0.0    # where the stored clip starts in that episode

    # start_s / end_s / hit_s are CLIP time (seek position in the stored clip);
    # episode_* are what users see: time in the original episode.
    @property
    def episode_start_s(self) -> float:
        return self.start_s + self.source_offset_s

    @property
    def episode_end_s(self) -> float:
        return self.end_s + self.source_offset_s

    @property
    def episode_hit_s(self) -> float:
        return self.hit_s + self.source_offset_s

    @property
    def parent_episode_start_s(self) -> float:
        return self.parent_start_s + self.source_offset_s

    @property
    def parent_episode_end_s(self) -> float:
        return self.parent_end_s + self.source_offset_s

    @property
    def timestamp(self) -> str:
        """Hit time in the original episode, e.g. '28:51'."""
        return fmt_ts(self.episode_hit_s)


@dataclass
class SearchOutput:
    results: list[Result]
    reranked: bool = False            # True only if the reranker actually ran
    reranker: str | None = None       # model name when reranked
    mmr: bool = False
    grouped: bool = False             # results were deduplicated by parent_id
    search_text: str = ""             # what the rankers searched (query minus a detected speaker)
    speaker_detected: str | None = None
    role_detected: str | None = None
    below_threshold: int = 0          # results dropped for low relevance
    timings_ms: dict = field(default_factory=dict)   # per stage: understand, embed, retrieve, rerank, post, total
    notes: list[str] = field(default_factory=list)   # e.g. "rerank skipped: rate limit"


@dataclass(frozen=True)
class Filters:
    speaker: str | None = None     # case-insensitive substring of the speaker name
    file_id: str | None = None
    origin: str | None = None      # 'golden' | 'upload'
    role: str | None = None        # 'host' | 'guest'

    def sql(self) -> tuple[str, dict]:
        """WHERE-clause fragment over `segments s`, with named parameters."""
        sql, params = "", {}
        if self.origin:
            sql += " AND s.file_id IN (SELECT id FROM files WHERE origin = %(f_origin)s)"
            params["f_origin"] = self.origin
        if self.file_id:
            sql += " AND s.file_id = %(f_file)s"
            params["f_file"] = self.file_id
        if self.speaker:
            sql += " AND s.speaker_id IN (SELECT id FROM speakers WHERE name ILIKE %(f_speaker)s)"
            params["f_speaker"] = f"%{self.speaker}%"
        if self.role:
            sql += " AND s.speaker_id IN (SELECT id FROM speakers WHERE role = %(f_role)s)"
            params["f_role"] = self.role
        return sql, params


def fmt_ts(s: float) -> str:
    """mm:ss, or h:mm:ss past an hour (episodes can run over 60 minutes)."""
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


# ---------- rankers ----------
# Each ranker is a self-contained SQL statement returning segment ids in rank order.
# All rankers of one search run sequentially on ONE pooled connection (psycopg pipeline mode was
# measured 7x slower here, ~48 ms vs 6.6 ms, due to TCP delayed-ACK on Windows->Docker),
# so a search costs one connection regardless of how many rankers run.

Query = tuple[str, dict]


def keyword_query(q: str, f: Filters, k: int = CANDIDATES) -> Query:
    """Full-text ranker. A quoted phrase or all-terms match ranks first, then chunks by how
    many query terms they cover, then by cover density (terms appearing close together).

    A chunk sharing one common word with a long question is a weak keyword signal, yet in RRF
    only rank counts. So 3+ term queries must match at least half their terms; 1-2 term
    queries (names, jargon) accept any match. Lexemes are the query after stemming and
    stopword removal, e.g. "how to evaluate search" -> {evalu, search}."""
    fsql, fparams = f.sql()
    sql = f"""
        WITH q AS (
            SELECT websearch_to_tsquery('english', %(q)s) AS strict,
                   CASE WHEN %(phrase)s THEN websearch_to_tsquery('english', %(q)s)
                        ELSE nullif(replace(plainto_tsquery('english', %(q)s)::text, '&', '|'), '')::tsquery
                   END AS loose,
                   tsvector_to_array(to_tsvector('english', %(q)s)) AS lex),
        scored AS (
            SELECT s.id, s.tsv,
                   (SELECT count(*) FROM unnest(q.lex) l WHERE s.tsv @@ quote_literal(l)::tsquery) AS cover
            FROM segments s, q
            WHERE s.tsv @@ q.loose {fsql})
        SELECT scored.id FROM scored, q
        WHERE %(phrase)s
           OR cover >= CASE WHEN cardinality(q.lex) <= 2 THEN 1 ELSE ceil(cardinality(q.lex) / 2.0) END
        ORDER BY (scored.tsv @@ q.strict) DESC, cover DESC, ts_rank_cd(scored.tsv, q.loose, 32) DESC, scored.id
        LIMIT %(k)s"""
    return sql, {"q": q, "phrase": '"' in q, "k": k, **fparams}


def fuzzy_query(q: str, f: Filters, k: int = CANDIDATES) -> Query:
    """Trigram word-similarity ranker: catches misspellings and ASR spelling variants."""
    fsql, fparams = f.sql()
    sql = f"""
        SELECT s.id FROM segments s
        WHERE %(q)s <%% s.child_text {fsql}
        ORDER BY word_similarity(%(q)s, s.child_text) DESC, s.id
        LIMIT %(k)s"""
    return sql, {"q": q, "k": k, **fparams}


def vector_query(qv, emb: Embedder, f: Filters, k: int = CANDIDATES) -> Query:
    """Cosine k-NN over this model's partial HNSW index (same cast + WHERE as the index).
    Candidates below the model's similarity floor are dropped (cosine distance = 1 - sim)."""
    fsql, fparams = f.sql()
    cast = f"vector({emb.dim})"
    sql = f"""
        SELECT s.id FROM embeddings e JOIN segments s ON s.id = e.segment_id
        WHERE e.model = %(model)s {fsql}
          AND (e.embedding::{cast} <=> %(qv)s::{cast}) <= %(max_dist)s
        ORDER BY e.embedding::{cast} <=> %(qv)s::{cast}
        LIMIT %(k)s"""
    return sql, {"model": emb.name, "qv": qv, "k": k, "max_dist": 1 - emb.min_similarity, **fparams}


# ---------- fusion + rerank ----------

def rrf(ranked: dict[str, list[int]], weights=WEIGHTS) -> list[tuple[int, float, dict]]:
    """Weighted Reciprocal Rank Fusion: score = sum(w / (RRF_K + rank)) over rankers."""
    scores: dict[int, float] = {}
    sources: dict[int, dict] = {}
    for method, ids in ranked.items():
        for rank, sid in enumerate(ids, 1):
            scores[sid] = scores.get(sid, 0.0) + weights.get(method, 1.0) / (RRF_K + rank)
            sources.setdefault(sid, {})[method] = rank
    return sorted(((sid, sc, sources[sid]) for sid, sc in scores.items()),
                  key=lambda x: (-x[1], x[0]))


def group_by_parent(results: list[Result]) -> list[Result]:
    """Keep the best-ranked child of each parent (results arrive best-first); the kept
    result records how many sibling children also matched."""
    best: dict[str, Result] = {}
    for r in results:
        if r.parent_id in best:
            best[r.parent_id].grouped += 1
        else:
            best[r.parent_id] = r
    return list(best.values())


def mmr_select(relevance: np.ndarray, vectors: np.ndarray, k: int, lam: float = config.MMR_LAMBDA) -> list[int]:
    """Maximal Marginal Relevance. Greedily picks the candidate maximizing
        lam * relevance(d) - (1 - lam) * max cosine(d, already picked)
    so near-duplicates (e.g. overlapping neighbour chunks) give way to new information.
    `relevance` is in [0, 1]; `vectors` are L2-normalized rows (zero rows = unknown)."""
    picked: list[int] = []
    remaining = list(range(len(relevance)))
    while remaining and len(picked) < k:
        if picked:
            redundancy = (vectors[remaining] @ vectors[picked].T).max(axis=1)
            scores = lam * relevance[remaining] - (1 - lam) * redundancy
        else:
            scores = relevance[remaining]
        best = remaining[int(np.argmax(scores))]
        picked.append(best)
        remaining.remove(best)
    return picked


def apply_thresholds(results: list[Result], min_score: float = config.RERANK_MIN_SCORE,
                     min_relative: float = config.RERANK_MIN_RELATIVE) -> tuple[list[Result], float]:
    """Drop reranked results below an absolute score or a fraction of the query's best score,
    even if they made the top k. Returns (kept, cut-off used)."""
    if not results:
        return results, min_score
    cutoff = max(min_score, min_relative * max(r.score for r in results))
    return [r for r in results if r.score >= cutoff], cutoff


_speaker_cache: dict[tuple[int, str], tuple[float, list[str]]] = {}   # (pool, collection) -> (at, names)


async def known_speakers(pool: AsyncConnectionPool, collection: str) -> list[str]:
    """Distinct speaker names in a collection (cached 30 s) for detecting speakers in queries."""
    key = (id(pool), collection)
    hit = _speaker_cache.get(key)
    if hit and time.monotonic() - hit[0] < 30:
        return hit[1]
    async with db.collection_conn(pool, collection) as conn:
        cur = await conn.execute("SELECT DISTINCT name FROM speakers")
        names = [r[0] for r in await cur.fetchall()]
    _speaker_cache[key] = (time.monotonic(), names)
    return names


def _unit_range(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min() if len(x) else 0
    return (x - x.min()) / span if span > 0 else np.ones_like(x)


# ---------- searcher ----------

class Searcher:
    """Hybrid search over a connection pool. Cheap to construct; models are cached per process."""

    def __init__(self, pool: AsyncConnectionPool, embedder: str | None = None, origin: str | None = None,
                 collection: str | None = None):
        """collection selects the dataset (default config.DEFAULT_COLLECTION);
        origin='golden' restricts every query to the evaluation dataset."""
        self.pool = pool
        self.embedder_name = embedder
        self.origin = origin
        self.collection = collection or config.DEFAULT_COLLECTION
        db.schema_name(self.collection)  # validate early

    async def search(self, q: str, method: str = "hybrid", k: int = 10, speaker: str | None = None,
                     file_id: str | None = None, rerank: bool | None = None, mmr: bool = False,
                     mmr_lambda: float = config.MMR_LAMBDA, group: bool = True,
                     patient_rerank: bool = False, role: str | None = None, auto_speaker: bool = True,
                     threshold: bool = True) -> SearchOutput:
        """rerank=None uses config.RERANK_DEFAULT. group=True keeps one (best) child per parent.
        auto_speaker detects "what did <name>/the guest say about X" when no speaker/role is given.
        threshold drops low-relevance results (reranker score cut-off).
        patient_rerank waits out API rate limits (evaluation) instead of falling back to the
        fused order (interactive use)."""
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
        rerank = config.RERANK_DEFAULT if rerank is None else rerank
        t0 = time.perf_counter()
        text, detected = q, None
        if auto_speaker and not speaker and not role:
            detected = parse_query(q, await known_speakers(self.pool, self.collection))
            if detected.speaker or detected.role:
                text, speaker, role = detected.text, detected.speaker, detected.role
                log.info("query understanding: %r -> speaker=%s role=%s text=%r", q, speaker, role, text)
        f = Filters(speaker, file_id, self.origin, role)
        log.info("search [%s] q=%r method=%s k=%d group=%s rerank=%s mmr=%s%s embedder=%s", self.collection,
                 q[:120], method, k, group,
                 rerank, f"{mmr}(λ={mmr_lambda})" if mmr else mmr,
                 "".join(f" {n}={v}" for n, v in vars(f).items() if v), self.embedder_name or config.EMBEDDER)
        timings: dict[str, float] = {}
        mark = time.perf_counter()

        def lap(stage: str) -> None:
            nonlocal mark
            now = time.perf_counter()
            timings[stage] = round((now - mark) * 1000, 1)
            mark = now

        lap("understand")                      # query understanding (speaker/role detection)
        emb = await get_embedder(self.embedder_name) if method in ("vector", "hybrid") or mmr else None
        queries: dict[str, Query] = {}
        if method in ("keyword", "hybrid"):
            queries["keyword"] = keyword_query(text, f)
        # In hybrid mode the typo ranker only runs for short lookups (see config.FUZZY_MAX_TERMS).
        if method == "fuzzy" or (method == "hybrid" and len(query_terms(text)) <= config.FUZZY_MAX_TERMS):
            queries["fuzzy"] = fuzzy_query(text, f)
        if method in ("vector", "hybrid"):
            # Embed before taking a connection: a slow API call must not hold a pool slot.
            queries["vector"] = vector_query(await emb.embed_query(text), emb, f)
        lap("embed")                           # query embedding (API call unless cached)

        depth = max(k * 3, config.RERANK_DEPTH) if (rerank or mmr or group) else k
        async with db.collection_conn(self.pool, self.collection) as conn:
            ranked = {m: [r[0] for r in await (await conn.execute(sql, params)).fetchall()]
                      for m, (sql, params) in queries.items()}
            fused = rrf(ranked)[:depth]
            results, vectors = await self._hydrate(conn, text, fused, emb.name if mmr else None)
        lap("retrieve")                        # pooled connection + rankers + fusion + hydrate
        log.info("retrieval %s -> fused %d candidates in %.0f ms",
                 " ".join(f"{m}={len(ids)}" for m, ids in ranked.items()), len(fused),
                 (time.perf_counter() - t0) * 1000)

        out = SearchOutput(results, search_text=text,
                           speaker_detected=detected.speaker if detected else None,
                           role_detected=detected.role if detected else None)
        if group and results:
            out.results = group_by_parent(out.results)
            out.grouped = True
            log.info("grouped %d children into %d parents", len(results), len(out.results))
        out.results = out.results[:config.RERANK_DEPTH] if (rerank or mmr) else out.results[:k]
        if rerank and out.results:
            # The reranker gets the full question: "Gene Kranz: ..." context makes the name useful there.
            await self._rerank(q, out, patient_rerank)
        lap("rerank")                          # rerank API call (0 when off or cached)
        if threshold and out.reranked:
            before = len(out.results)
            out.results, cutoff = apply_thresholds(out.results)
            out.below_threshold = before - len(out.results)
            if out.below_threshold:
                log.info("threshold: dropped %d of %d results below relevance %.2f",
                         out.below_threshold, before, cutoff)
            if not out.results:
                out.notes.append(f"no confident match: every candidate scored below the relevance "
                                 f"threshold ({cutoff:.2f})")
        if mmr and out.results:
            cands = out.results
            relevance = _unit_range(np.array([r.score for r in cands], dtype=np.float32))
            mat = np.stack([vectors.get(r.segment_id, np.zeros(emb.dim, np.float32)) for r in cands])
            order = mmr_select(relevance, mat, k, mmr_lambda)
            out.results = [cands[i] for i in order]
            for rank, r in enumerate(out.results, 1):
                r.sources["mmr"] = rank
            out.mmr = True
            log.info("mmr λ=%.2f picked %d of %d candidates (missing vectors: %d)", mmr_lambda, len(order),
                     len(cands), sum(1 for r in cands if r.segment_id not in vectors))
        out.results = out.results[:k]
        lap("post")                            # grouping, thresholds, MMR
        timings["total"] = round((time.perf_counter() - t0) * 1000, 1)
        out.timings_ms = timings
        top = out.results[0] if out.results else None
        log.info("search done in %.0f ms: %d results reranked=%s mmr=%s top=%s", (time.perf_counter() - t0) * 1000,
                 len(out.results), out.reranked, out.mmr,
                 f"{top.file_id}@{top.timestamp}/{top.speaker}" if top else None)
        for note in out.notes:
            log.warning("search note: %s", note)
        return out

    @staticmethod
    async def _rerank(q: str, out: SearchOutput, patient: bool) -> None:
        # Grouped: one result per parent, so rerank the parent dialogue (question + answer).
        # Ungrouped: siblings share a parent, so rerank each child ("Speaker: text") instead.
        docs = ([r.parent_text for r in out.results] if out.grouped
                else [f"{r.speaker}: {r.child_text}" for r in out.results])
        try:
            reranker = get_reranker(patient=patient)
            scores = await reranker.scores(q, docs)
        except RerankUnavailable as e:
            out.notes.append(f"rerank skipped, showing hybrid order: {e}")
            return
        for r, sc in zip(out.results, scores, strict=True):
            r.score = sc
        out.results.sort(key=lambda r: -r.score)
        for rank, r in enumerate(out.results, 1):
            r.sources["rerank"] = rank
        out.reranked, out.reranker = True, reranker.name

    @staticmethod
    async def _hydrate(conn: AsyncConnection, q: str, fused,
                       vector_model: str | None = None) -> tuple[list[Result], dict[int, np.ndarray]]:
        """Load segment details in fused order; with `vector_model`, also each segment's stored
        embedding (used by MMR, so diversification needs no extra API calls)."""
        if not fused:
            return [], {}
        cur = await conn.execute(
            """SELECT s.id, s.file_id, f.title, f.audio_path, sp.name, s.start_s, s.end_s, s.child_text, s.words,
                      s.parent_id, s.parent_start_s, s.parent_end_s, s.parent_text,
                      s.audio_url, s.source_offset_s, e.embedding
               FROM segments s JOIN files f ON f.id = s.file_id JOIN speakers sp ON sp.id = s.speaker_id
               LEFT JOIN embeddings e ON e.segment_id = s.id AND e.model = %s
               WHERE s.id = ANY(%s)""", (vector_model, [sid for sid, _, _ in fused]))
        rows = {r[0]: r for r in await cur.fetchall()}
        terms = query_terms(q)
        results, vectors = [], {}
        for sid, score, src in fused:
            (_, fid, title, audio, spk, start, end, text, words, pid, pstart, pend, ptext,
             url, offset, vec) = rows[sid]
            hit, highlighted = locate(words, terms, start)
            results.append(Result(sid, fid, title, audio, spk, start, end, hit, text, highlighted, score, dict(src),
                                  pid, pstart, pend, ptext, audio_url=url, source_offset_s=offset))
            if vec is not None:
                # untyped `vector` column: pgvector returns a Vector object, not an ndarray
                vectors[sid] = np.asarray(vec.to_numpy() if hasattr(vec, "to_numpy") else vec, dtype=np.float32)
        return results, vectors


# ---------- highlighting ----------

def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def query_terms(q: str) -> list[str]:
    return [t for t in (_norm(w) for w in q.split()) if t and t not in STOPWORDS]


def _matches(word: str, term: str) -> bool:
    # Cheap stem-tolerant match: "embeddings" ~ "embedding", "running" ~ "run".
    stem = term[: max(4, len(term) - 2)] if len(term) > 4 else term
    return word == term or (len(stem) >= 4 and word.startswith(stem))


def locate(words: list, terms: list[str], default_s: float) -> tuple[float, str]:
    """Return (time of first matched query word, text with matched words wrapped in [[ ]])."""
    hit = None
    parts = []
    for w, ws, _ in words:
        nw = _norm(w)
        if nw and any(_matches(nw, t) for t in terms):
            hit = ws if hit is None else hit
            parts.append(f"[[{w}]]")
        else:
            parts.append(w)
    return (hit if hit is not None else default_s), " ".join(parts)
