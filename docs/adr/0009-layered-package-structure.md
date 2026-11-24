# ADR-0009: Layered package structure; providers chosen once at startup

**Status:** accepted · **Decided by:** project owner

**Context.** The core library had grown to a flat set of modules (`config.py`, `search.py`,
`ingest.py`, ...). Several problems followed from that layout:
- Configuration was module globals, some of them bound as default arguments at import time.
- There were eight hidden singletons, each built a different way.
- Providers were picked by `if/elif` chains or parallel dicts of functions.
- SQL lived in the API, the CLI and ingestion.
- The upload pipeline was written twice (CLI `add` and the API jobs).
- The backend imported the `eval/` tooling package.
- `Searcher.search` was one ~100-line method taking 13 parameters.

Tests had to monkeypatch module state to swap in fakes.

**Decision.**
* **Packages by responsibility.** These sit under `audiosearch/`:
  - `core/`: settings, logging, runtime, disk cache, errors, text helpers, progress
  - `models.py`: domain dataclasses
  - `db/`: database, repositories, schema
  - `ingestion/`: the ingestion pipeline (`pipeline.py`, `audio.py`, `transcription/`, `chunking/`)
  - `retrieval/`: the retrieval pipeline (`pipeline.py`, query parser, rankers, fusion, `rerankers/`, `answer.py`)
  - `embeddings/`: shared by both pipelines
  - `evaluation/`: golden set, labels, evaluator
  - `cli/`
* **Settings**: one frozen `Settings` read from `.env`; `get_settings()` is the process-wide singleton.
  Code reads settings at call time; tests pass a modified copy (`dataclasses.replace`).
* **Providers chosen once.** When the app starts, `AppContainer` reads `.env` and builds the embedder,
  reranker and transcriber, using a plain `if name == ...` in `container.py`. Services receive these
  objects in their constructor and never pick a provider themselves. API keys are checked on first
  use, so the app starts without them. Gemini is the only embedder (ADR-0005), so nothing chooses
  a model per request.
  *Revised:* a first version used a registry `factory.py` per package (`EmbedderFactory`, ...).
  The owner judged it harder to follow than the problem needed, and the plain wiring replaced it.
* **Strategy** for the first-stage rankers (`KeywordRanker`, `FuzzyRanker`, `VectorRanker`). `SearchService`
  runs named stages: understand → embed → retrieve → group → rerank → threshold → MMR.
* **Template Method** for caching: `Embedder.embed_query` and `Reranker.scores` cache once, in the base
  class, through one `JsonDiskCache`. The key layouts are unchanged, so existing `.cache/` contents stay valid.
* **Repositories** hold all SQL outside the rankers. They take a connection already scoped by
  `collection_conn`, so callers choose the transaction boundary (rule 3 unchanged).
* **Composition root**: `AppContainer` owns the settings, the providers, the pool and the services. The API keeps one per app
  (lifespan); each CLI command and tool opens one with `async with AppContainer.open()`.
* **Backend**:
  - `backend/main.py` builds the app (`create_app()`); routes live in `backend/routers/`, one module per resource.
  - `dependencies.py` provides the container, the job manager and validated parameters.
  - `JobManager` is a class with one instance per app; it evicts old finished jobs.
* **Frontend**: `app.py` (navigation), `sidebar.py` (returns a `UiContext`), `components.py`, and
  `views/` (one module per page). It stays a pure HTTP client (rule 1).
* **Tooling**: `eval/*.py` are thin entry points over `audiosearch.evaluation`.

**Consequences.**
* Commands and REST paths unchanged (`python -m audiosearch …`, `python -m eval.*`). With BGE removed
  (ADR-0005), the per-request embedder choice is gone: the `embedder` parameter of `/search`,
  `/uploads/{id}/index` and `/eval/run`, and the CLI's `--embedder` / `--embedders`. `/health` reports
  one `embedding_model` instead of a list of choices.
* uvicorn target is now `backend.main:app`; shared logging is `audiosearch.core.log`.
* Small behaviour fixes made along the way:
  - The MMR λ default is 0.6 everywhere; the CLI and UI client had used 0.7.
  - The report detects negatives by their empty label instead of an id prefix.
  - The HNSW index DDL uses quoted SQL literals and identifiers instead of string interpolation.
  - DB tests skip, instead of erroring, when Postgres is down.

**Evidence.**
* The golden manifest was ingested offline (cached transcripts) with the old code (git `c2af8de`) and
  the new code into two scratch collections. `files`, `speakers` and all 152 `segments` rows were
  byte-identical, and so were all 304 keyword and fuzzy searches (scores, highlights, timestamps,
  speaker detection).
* Re-run after removing the factories: still identical (rows; 304 keyword/fuzzy runs; 304
  vector/hybrid/MMR runs on the same index).
* With a deterministic fake embedder, all 304 vector, hybrid and MMR searches were identical when old
  and new code queried the same collection, including segment ids. (Two independently built HNSW
  indexes differ slightly at vector ranks 25–30, which changed 2 MMR runs across collections. That
  difference comes from the approximate index, not the code.)
* ruff clean; pytest: 68 passed, 12 skipped. The skipped tests are the recall gate, which needs the golden
  set embedded with Gemini; it was not run (no API keys on the verifying machine).
