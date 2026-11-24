"""Application settings, read once from `.env` into one immutable object.

`get_settings()` is the process-wide Singleton. Code reads settings at call time (never as
import-time default arguments), so tests can build a modified copy with
`dataclasses.replace(get_settings(), ...)` and hand it to the object under test.
"""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str = "") -> str:
    # Tolerate inline comments like `EMBEDDER=gemini   # the only option`.
    return os.getenv(name, default).split("#", 1)[0].strip()


def _env_or(name: str, default: str) -> str:
    """Like _env, but an empty value also falls back to the default."""
    return _env(name, default) or default


@dataclass(frozen=True)
class Settings:
    # ---------- paths ----------
    root: Path
    data_dir: Path
    audio_dir: Path
    raw_dir: Path
    transcript_dir: Path
    manifest_path: Path
    queries_path: Path
    golden_spec_path: Path
    cache_dir: Path
    eval_dir: Path
    log_dir: Path

    # ---------- services ----------
    database_url: str
    # Collections isolate datasets/tenants (Postgres schema col_<name>). The golden set lives in this one.
    default_collection: str
    deepgram_api_key: str
    assemblyai_api_key: str
    transcriber: str                 # assemblyai (names) | deepgram
    # Placeholder host for each episode's audio_url until real hosting exists.
    audio_url_base: str
    gemini_api_key: str
    embedder: str                    # gemini (the only embedder)
    cohere_api_key: str
    cohere_rerank_model: str
    rerank_provider: str             # cohere | local

    # ---------- retrieval ----------
    rerank_default: bool             # rerank is on by default in the API, UI and CLI
    rerank_depth: int                # fused candidates re-scored by the reranker / considered by MMR
    mmr_lambda: float                # MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity
    # Relevance thresholds (see ADR-0004). Raw cosine does NOT separate relevant from irrelevant
    # chunks with gemini-embedding-001 (relevant 0.59-0.78, irrelevant up to 0.76), so the vector
    # ranker only gets a low safety floor. The reranker's score does separate them (answers
    # 0.64-0.97, junk 0.26-0.30), so results are cut on that: below rerank_min_score, or below
    # rerank_min_relative x the best score of the query, a result is dropped even inside the top k.
    vector_min_sim: float
    # The fuzzy (trigram) ranker exists for short misspelled lookups ("Jery Bostik"). On sentences it
    # finds nothing and costs a sequential scan (~30 ms), so it only runs for short queries.
    fuzzy_max_terms: int
    rerank_min_score: float
    rerank_min_relative: float
    # Grounded summary answer above the results (Gemini text model; top-N results as context).
    # gemini-2.5-flash-lite is "no longer available to new users" (API 404); 3.5 Flash-Lite is its successor.
    answer_model: str
    answer_top_n: int

    # ---------- logging ----------
    log_level: str

    # ---------- concurrency ----------
    # A search holds exactly one connection (its ranker queries run back to back), for a few ms,
    # and never while waiting on an embedding API. Postgres allows 100 connections by default.
    db_pool_min: int
    db_pool_max: int
    db_pool_timeout_s: float
    http_max_connections: int
    embed_concurrency: int           # parallel Gemini batch requests
    ingest_concurrency: int          # files transcribed in parallel
    job_concurrency: int             # background jobs running at once

    @classmethod
    def from_env(cls, root: Path = ROOT) -> "Settings":
        load_dotenv(root / ".env")
        data = root / "data"
        return cls(
            root=root,
            data_dir=data,
            audio_dir=data / "audio",
            raw_dir=data / "raw",
            transcript_dir=data / "transcripts",
            manifest_path=data / "manifest.json",
            queries_path=data / "queries.json",
            golden_spec_path=data / "golden_spec.json",
            cache_dir=root / ".cache",
            eval_dir=root / "eval",
            log_dir=root / "logs",
            database_url=_env("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/audiosearch"),
            default_collection=_env_or("DEFAULT_COLLECTION", "nasa"),
            deepgram_api_key=_env("DEEPGRAM_API_KEY"),
            assemblyai_api_key=_env("ASSEMBLYAI_API_KEY"),
            transcriber=_env_or("TRANSCRIBER", "assemblyai"),
            audio_url_base=_env("AUDIO_URL_BASE", "https://media.example.com/podcasts/"),
            gemini_api_key=_env("GEMINI_API_KEY"),
            embedder=_env_or("EMBEDDER", "gemini"),
            # Cohere key: COHERE_API_KEY, or the older RERANKER name in .env.
            cohere_api_key=_env("COHERE_API_KEY") or _env("RERANKER"),
            cohere_rerank_model=_env("COHERE_RERANK_MODEL", "rerank-v4.0-pro"),
            rerank_provider=_env("RERANK_PROVIDER", "cohere"),
            rerank_default=True,
            rerank_depth=30,
            mmr_lambda=0.6,
            vector_min_sim=float(_env("VECTOR_MIN_SIM", "0.50")),
            fuzzy_max_terms=int(_env("FUZZY_MAX_TERMS", "4")),
            rerank_min_score=float(_env("RERANK_MIN_SCORE", "0.40")),
            rerank_min_relative=float(_env("RERANK_MIN_RELATIVE", "0.50")),
            answer_model=_env_or("ANSWER_MODEL", "gemini-3.5-flash-lite"),
            answer_top_n=int(_env("ANSWER_TOP_N", "5")),
            log_level=_env_or("LOG_LEVEL", "INFO").upper(),
            db_pool_min=int(_env("DB_POOL_MIN", "2")),
            db_pool_max=int(_env("DB_POOL_MAX", "10")),
            db_pool_timeout_s=float(_env("DB_POOL_TIMEOUT_S", "10")),
            http_max_connections=int(_env("HTTP_MAX_CONNECTIONS", "20")),
            embed_concurrency=int(_env("EMBED_CONCURRENCY", "4")),
            ingest_concurrency=int(_env("INGEST_CONCURRENCY", "3")),
            job_concurrency=int(_env("JOB_CONCURRENCY", "2")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings (Singleton): `.env` is read once."""
    return Settings.from_env()
