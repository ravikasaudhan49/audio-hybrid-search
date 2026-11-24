# Simplify provider wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the three `factory.py` registries; the embedder, reranker and transcriber are chosen once from `.env` when the app starts and passed into the services that use them.

**Architecture:** `AppContainer.__init__` builds one `Embedder`, one `Reranker` and one `Transcriber` (a plain `if name == ...` in `container.py`). `SearchService` and `IngestionService` receive those objects in their constructors. Provider constructors never fail for a missing key; the key is checked on first use, so the API still starts without keys (keyword search works).

**Tech Stack:** Python 3.12+, FastAPI, psycopg 3 async, pytest, ruff.

**Spec:** the owner's request in chat: "lets not [create factory.py]… when the service is triggered, we decide the embedding model, llm model etc one time and use that throughout… easy to understand and not very complex."

## Global Constraints

- Search results and stored rows must be identical to before (verify with the old-vs-new comparison used for ADR-0009).
- Tests never call paid APIs; the fake embedder is injected via `AppContainer(settings, embedder=FakeEmbedder())`.
- CLI commands, REST paths and response fields unchanged.
- CLAUDE.md rule 3 and ADR-0009 updated to the new wiring.

## Review Focus

- No API keys in `.env`: API must still start and keyword search must work (covered by `tests/test_api.py`).
- Missing Cohere key: search falls back to fused order with a note, never a 500 (`test_rerank_falls_back_to_fused_order_when_unavailable`).
- `embedder=bge-base` per request / `eval --embedders gemini bge-base`: a second embedder is built once and reused (`test_embedder_for_reuses_instances`).
- Cached raw transcripts must normalize without any key (`test_transcript_store_normalizes_cached_raw`).
- Unknown embedder name in the API: 400, not 500 (`test_search_rejects_unknown_method_and_embedder`).

---

### Task 1: Providers without factories

**Files:**
- Delete: `audiosearch/embeddings/factory.py`, `audiosearch/rerankers/factory.py`, `audiosearch/transcription/factory.py`, `audiosearch/transcription/service.py`
- Modify: `audiosearch/embeddings/{__init__,gemini}.py`, `audiosearch/rerankers/{__init__,cohere}.py`, `audiosearch/transcription/{__init__,store}.py`

**Interfaces — Produces:**
- `embeddings.EMBEDDER_MODELS: dict[str, str]` = `{"gemini": "gemini-embedding-001", "bge-base": "bge-base-en-v1.5", "bge-small": "bge-small-en-v1.5"}`
- `GeminiEmbedder(api_key, ...)` raises `MissingApiKeyError` on first call, not in `__init__`.
- `CohereReranker(api_key, model, cache_dir)` raises `RerankUnavailable` in `_scores` when the key is empty.
- `transcription.store.NORMALIZERS: dict[str, Callable[[dict], dict]]` (assemblyai, deepgram).

- [ ] Make the changes; `ruff check .` passes.

### Task 2: Container builds providers once; services receive them

**Files:** `audiosearch/container.py`, `audiosearch/search/service.py`, `audiosearch/services/ingestion.py`, `audiosearch/evaluation/evaluator.py`

**Interfaces — Produces:**
- `AppContainer(settings=None, *, embedder=None, reranker=None, transcriber=None)`; attributes `embedder`, `reranker`, `transcriber`; `embedder_for(name: str | None) -> Embedder`.
- `SearchService(pool, *, embedder: Embedder, reranker: Reranker, collection=None, origin=None, settings=None, speakers=None)`
- `IngestionService(pool, *, embedder: Embedder, transcriber: Transcriber, collection=None, settings=None, transcripts=None, audio=None)`; `embed_missing(embedder: Embedder | None = None, progress=None)`; `transcribe(file_id, audio_path, keyterms=None, force=False) -> dict`.
- `Evaluator(app: AppContainer, collection=None)`.

- [ ] Write tests `test_embedder_for_reuses_instances`, `test_transcript_store_normalizes_cached_raw` in `tests/test_components.py`; replace the factory tests.
- [ ] Implement; `pytest -q` passes.

### Task 3: Callers and docs

**Files:** `audiosearch/cli/{main,commands}.py`, `backend/{main,dependencies}.py`, `backend/routers/{health,search,uploads,evaluation}.py`, `tests/test_pipeline_db.py`, `tests/test_search_units.py`, `CLAUDE.md`, `docs/adr/0009-layered-package-structure.md`, `docs/ARCHITECTURE.md`, `audiosearch/__init__.py`

- [ ] Replace factory calls with `app.embedder_for(name)` / `app.reranker` / `EMBEDDER_MODELS`.
- [ ] Update docs (rule 3, ADR-0009, code-structure section).
- [ ] Verify: `ruff check .`, `pytest -q`, `python -m audiosearch doctor`, old-vs-new comparison on the golden data → identical.
