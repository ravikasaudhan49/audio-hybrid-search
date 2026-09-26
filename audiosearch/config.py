"""Central configuration, loaded from .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    # Tolerate inline comments like `EMBEDDER=gemini   # gemini | bge-base`.
    return os.getenv(name, default).split("#", 1)[0].strip()


# ---------- paths ----------
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
RAW_DIR = DATA_DIR / "raw"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
MANIFEST_PATH = DATA_DIR / "manifest.json"
QUERIES_PATH = DATA_DIR / "queries.json"
CACHE_DIR = ROOT / ".cache"
EVAL_DIR = ROOT / "eval"
LOG_DIR = ROOT / "logs"

# ---------- services ----------
DATABASE_URL = _env("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/audiosearch")
# Collections isolate datasets/tenants (Postgres schema col_<name>). The golden set lives in this one.
DEFAULT_COLLECTION = _env("DEFAULT_COLLECTION", "nasa") or "nasa"
DEEPGRAM_API_KEY = _env("DEEPGRAM_API_KEY")
ASSEMBLYAI_API_KEY = _env("ASSEMBLYAI_API_KEY")
TRANSCRIBER = _env("TRANSCRIBER", "assemblyai") or "assemblyai"   # assemblyai (names) | deepgram
# Placeholder host for each episode's audio_url until real hosting exists.
AUDIO_URL_BASE = _env("AUDIO_URL_BASE", "https://media.example.com/podcasts/")
GEMINI_API_KEY = _env("GEMINI_API_KEY")
EMBEDDER = _env("EMBEDDER", "gemini") or "gemini"
# Cohere key: COHERE_API_KEY, or the older RERANKER name in .env.
COHERE_API_KEY = _env("COHERE_API_KEY") or _env("RERANKER")
COHERE_RERANK_MODEL = _env("COHERE_RERANK_MODEL", "rerank-v4.0-pro")
RERANK_PROVIDER = _env("RERANK_PROVIDER", "cohere")   # cohere | local

# ---------- retrieval ----------
RERANK_DEFAULT = True     # rerank is on by default in the API, UI and CLI
RERANK_DEPTH = 30         # fused candidates re-scored by the reranker / considered by MMR
MMR_LAMBDA = 0.6          # MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity

# Relevance thresholds (calibrated on cached sample queries; re-tune with the golden set).
# Raw cosine does NOT separate relevant from irrelevant chunks with gemini-embedding-001
# (relevant 0.59-0.78, irrelevant up to 0.76, an off-topic query's best 0.58), so the vector
# ranker only gets a low safety floor. The reranker's score does separate them (answers
# 0.64-0.97, junk 0.26-0.30), so results are cut on that: below RERANK_MIN_SCORE, or below
# RERANK_MIN_RELATIVE x the best score of the query, a result is dropped even inside the top k.
VECTOR_MIN_SIM = float(_env("VECTOR_MIN_SIM", "0.50"))
# The fuzzy (trigram) ranker exists for short misspelled lookups ("Jery Bostik"). On sentences it
# finds nothing (0% semantic recall in eval) and costs a sequential scan (~30 ms), so it only runs
# for queries of at most this many meaningful terms.
FUZZY_MAX_TERMS = int(_env("FUZZY_MAX_TERMS", "4"))
# Grounded summary answer above the results (Gemini text model; top-N results as context).
# gemini-2.5-flash-lite is "no longer available to new users" (API 404); 3.5 Flash-Lite is Google's
# named successor. Override with ANSWER_MODEL in .env.
ANSWER_MODEL = _env("ANSWER_MODEL", "gemini-3.5-flash-lite") or "gemini-3.5-flash-lite"
ANSWER_TOP_N = int(_env("ANSWER_TOP_N", "5"))
RERANK_MIN_SCORE = float(_env("RERANK_MIN_SCORE", "0.40"))
RERANK_MIN_RELATIVE = float(_env("RERANK_MIN_RELATIVE", "0.50"))

# ---------- logging ----------
LOG_LEVEL = (_env("LOG_LEVEL", "INFO") or "INFO").upper()

# ---------- concurrency ----------
# A search holds exactly one connection (its ranker queries run back to back), for a few ms,
# and never while waiting on an embedding API. Postgres allows 100 connections by default.
DB_POOL_MIN = int(_env("DB_POOL_MIN", "2"))
DB_POOL_MAX = int(_env("DB_POOL_MAX", "10"))
DB_POOL_TIMEOUT_S = float(_env("DB_POOL_TIMEOUT_S", "10"))
HTTP_MAX_CONNECTIONS = int(_env("HTTP_MAX_CONNECTIONS", "20"))
EMBED_CONCURRENCY = int(_env("EMBED_CONCURRENCY", "4"))     # parallel Gemini batch requests
INGEST_CONCURRENCY = int(_env("INGEST_CONCURRENCY", "3"))   # files transcribed in parallel
JOB_CONCURRENCY = int(_env("JOB_CONCURRENCY", "2"))         # background jobs running at once

# ---------- chunking (parent-child, sizes in tiktoken cl100k_base tokens) ----------
# Child: the retrieval unit (embedded, keyword-indexed, returned with speaker + timestamp).
#   ~128 tokens ≈ 40-60 s of speech, always inside ONE speaker turn, 25-token overlap.
# Parent: ~512 tokens ≈ 2-3 min of whole turns (question + answer stay together); context
#   for reranking/display and the dedup key. 8-10 min files -> ~4-6 parents, ~20-25 children.
TOKENIZER = "cl100k_base"
CHILD_TOKENS = 128
CHILD_OVERLAP_TOKENS = 25
CHILD_MIN_TOKENS = 32          # a smaller tail is merged into the previous child
PARENT_TOKENS = 512
MIN_CHILD_WORDS = 3            # shorter turns ("Yeah.", "Right.") stay in parent context only
