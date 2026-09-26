# CLAUDE.md: rules for coding agents working on this repository

These are the standing instructions the project owner gave the coding agent (Claude Code) during
development, codified in one place. Any agent or contributor must follow them. Decisions behind
the rules are recorded in `docs/adr/`; the prompt-by-prompt trace is in `AGENT_LOG.md`.

## What this project is
Hybrid (keyword + semantic) search over two-speaker podcast audio. Results must show the **file,
the timestamp of the matched word, and the speaker's name**. See `README.md` and
`docs/ARCHITECTURE.md`.

## Architecture rules (do not change without an ADR)
1. **Layering.** `audiosearch/` is the async core library. `backend/` is a thin FastAPI layer (it
   owns the pool and the models; slow work runs as jobs). `frontend/` is a pure HTTP client: no
   database or model imports except `audiosearch.log`.
2. **One code path.** Everything is async; scripts use `aio.run()` (SelectorEventLoop, required by
   psycopg on Windows); uvicorn runs with `--loop asyncio:SelectorEventLoop`.
3. **Database access goes through `db.collection_conn(pool, collection)`.** Never query tables on
   a bare pooled connection; `search_path` selects the collection and is reset on release.
   Collection names are validated by `db.schema_name()`.
4. **One connection per search, held for milliseconds.** Call embedding and rerank APIs *before*
   or *after* holding a connection, never while holding it. No psycopg pipeline mode (measured 7×
   slower on this stack, see ADR-0006).
5. **Chunking contract.** Children ≤ 128 tokens (tiktoken cl100k), one speaker, 25-token overlap;
   parents ≤ 512 tokens of whole turns; `parent_id` is the dedup key. Times are stored clip-relative
   and displayed in episode time (`source_offset_s`).
6. **Transcripts are provider-neutral.** Downstream code reads only `<id>.transcript.json`
   (normalized); raw provider responses are cached next to it.
7. **Speaker identity is data**: names and `host`/`guest` roles live in `speakers`; the embedded
   text is `"Speaker: child_text"`.
8. **Thresholds use reranker scores, not cosine** (see ADR-0004). Change them only with evidence
   from `eval`.

## API budget rules (free-tier keys)
* Tests must never call paid APIs. Use cached results or the fake embedder (`tests/test_pipeline_db.py`).
* Every hosted call is cached on disk (`.cache/`, `data/transcripts/`). Never add an uncached call.
* Cohere trial = 10 calls/min: interactive paths fall back to the fused order; evaluation waits.
* Before any bulk API run (evaluation, re-embedding, transcription), state the expected number of
  calls and get the owner's go-ahead.

## Evaluation rules
* Golden labels come **only** from `data/golden_spec.json` anchors (verbatim transcript quotes)
  via `python -m eval.build_golden`; never hand-type timestamps.
* Never tune thresholds or labels to make a specific query pass. If a label is wrong, fix the
  labelling *method*, re-run everything, and log the before/after numbers in `AGENT_LOG.md`.
* Evaluation searches `origin=golden` only.

## Definition of done (run before claiming a change works)
```bash
.venv\Scripts\python -m ruff check .          # project lint config in pyproject.toml
.venv\Scripts\python -m pytest -q             # unit, API, DB and recall@k gate
.venv\Scripts\python -m audiosearch doctor    # environment and data sanity
```
For retrieval changes, also run `python -m audiosearch eval --rerank` and report the table.
For performance changes, run `python -m audiosearch bench` and report p50/p90/p99.
  
## Conventions
* Python 3.12+, ruff (line length 120), type hints, small functions, docstrings that explain *why*.
* Logging via `audiosearch.log.get(__name__)`; every line carries the request / job id; never log keys.
* Secrets only in `.env` (git-ignored); `.env.example` lists every key.
* Every user-visible change gets an `AGENT_LOG.md` entry: the prompt, what was done, how it was verified.
