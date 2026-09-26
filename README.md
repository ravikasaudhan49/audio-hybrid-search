# Hybrid Search over Two-Speaker Audio

**G2 AI Hiring Hackathon, Problem 1: "Effective retrieval from audio transcripts".**

Search six NASA podcast interviews by **exact words** ("Jerry Bostick", `"failure is not an option"`)
and by **meaning** ("how did the famous Apollo 13 movie line come about"). Every hit gives the
**episode, the timestamp of the matched word, and the speaker's name**, with a grounded summary
answer on top.

| | |
|---|---|
| **Golden dataset** | 6 clips × 9 min, 6 unique speaker pairs, 76 labeled queries (71 answerable + 5 off-topic) |
| **Recall@5 / @10 (full pipeline)** | **1.00 / 1.00** · Recall@1 0.96 · MRR 0.98 |
| **Off-topic → "no confident match"** | 4 of 5 (0.80) |
| **Speaker attribution of hits** | 0.97 |
| **Latency, 5 concurrent (warm)** | p50 186 ms · p90 274 ms · p99 371 ms · 25 req/s |
| **Tests** | 71 automated tests incl. a recall@k quality gate (`python -m audiosearch test`) |

Architecture diagrams and the data model: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.
Coding-agent disclosure (how the agent was directed, decision by decision): **[AGENT_LOG.md](AGENT_LOG.md)**.

## 🏢 Built for production: multi-tenant knowledge bases, designed to scale

### Every user or team gets their own knowledge base
* **Collections = isolated knowledge bases.** Anyone can create one from the **UI** (sidebar →
  *➕ New collection*), the **API** (`POST /collections`) or the **CLI**
  (`python -m audiosearch collections create acme_calls`), then **upload audio into it and search
  only within it**. The NASA golden set is just one collection (`nasa`).
* **Hard isolation, not a filter.** Each collection is its own Postgres schema (`col_<name>`) with its
  own tables and indexes. Queries are scoped per connection, and a test proves that one tenant's data
  never appears in another's results. There is no `WHERE tenant_id = …` that could be forgotten.
* **Same shape everywhere.** Every knowledge base is created from the same schema file, so columns,
  indexes, chunking and ranking behave identically. Onboarding a tenant is one API call.
* **Per-tenant growth.** Indexes (HNSW, full-text, trigram) are per collection, so a large tenant
  doesn't slow a small one; a collection can later move to its own database or partition with no
  query changes.

### Throughput and latency are configurable, not hard-coded
| Lever (`.env`) | Default | Effect |
|---|---|---|
| `DB_POOL_MIN` / `DB_POOL_MAX` | 2 / 10 | **Async Postgres connection pool.** Each search holds exactly **one** connection for a few milliseconds (never while waiting on an external API), so raising `DB_POOL_MAX` raises the number of concurrent searches almost linearly until the database CPU is the limit. Pool statistics (size, available, requests waiting) are exposed at `/health`. |
| `DB_POOL_TIMEOUT_S` | 10 | fail fast instead of queuing forever under overload |
| `HTTP_MAX_CONNECTIONS` | 20 | shared keep-alive HTTP pool for ASR / embedding / rerank APIs |
| `EMBED_CONCURRENCY` | 4 | parallel embedding batches during ingestion |
| `INGEST_CONCURRENCY` | 3 | files transcribed and ingested in parallel |
| `JOB_CONCURRENCY` | 2 | background jobs (uploads, indexing, evaluation) running at once |
| `RERANK_DEPTH`, `FUZZY_MAX_TERMS` | 30, 4 | latency / quality trade-offs per query |
| API workers | 1 | `uvicorn --workers N` or more replicas behind a load balancer; each worker has its own pool |

### Why it scales
* **Async end to end** (FastAPI, psycopg async connection pool, async HTTP clients): one process serves many
  concurrent requests while waiting on I/O.
* **Measured and optimised on the hot path:** one connection per search, no pipeline round-trip
  penalty, the trigram scan only for short queries. **p50 186 ms · p90 274 ms · p99 371 ms at 5
  concurrent requests, 25 req/s on a laptop, 0 errors.** Every response carries `timings_ms` per stage.
* **Expensive work is cached and asynchronous:** transcripts, query embeddings, rerank scores and
  answers are cached; uploads and indexing run as background jobs with progress, not inside requests.
* **Stateless API** (apart from in-memory job status, the first thing to move to a durable queue),
  so it scales horizontally; Postgres scales with read replicas, PgBouncer, HNSW tuning
  (`ef_search`, `m`) and per-collection partitioning.
* **Observable:** request-id tracing across UI → API → jobs in the logs, per-stage timings,
  `/health` pool stats, `bench` for repeatable p50/p90/p99.

More in [§8 Taking it to production](#8-taking-it-to-production-metrics-and-evaluation).

---

## Contents
0. [Built for production: multi-tenant, scalable](#-built-for-production-multi-tenant-knowledge-bases-designed-to-scale)
1. [Quickstart for judges](#1-quickstart-for-judges)
2. [What the task asked and how it is met](#2-what-the-task-asked-and-how-it-is-met)
3. [Design and rationale](#3-design-and-rationale)
4. [Decision log: the major architectural changes and why](#4-decision-log-the-major-architectural-changes-and-why)
5. [Evaluation: golden dataset, method, results](#5-evaluation-golden-dataset-method-results)
6. [Latency](#6-latency)
7. [Success criteria and level of achievement](#7-success-criteria-and-level-of-achievement)
8. [Taking it to production: metrics and evaluation](#8-taking-it-to-production-metrics-and-evaluation)
9. [Limitations](#9-limitations)
10. [Coding-agent disclosure](#10-coding-agent-disclosure)

---

## 1. Quickstart for judges

**Prerequisites:** Docker Desktop, Python 3.12+. Windows, macOS and Linux all work (developed on Windows 11).

```bash
git clone <repo> && cd <repo>
python -m venv .venv
.venv\Scripts\activate                      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d                        # Postgres 17 + pgvector
copy .env.example .env                      # then fill in the keys below
```

| Key in `.env` | Needed for | Free tier |
|---|---|---|
| `GEMINI_API_KEY` | chunk + query embeddings, summary answers | aistudio.google.com |
| `COHERE_API_KEY` | reranking (or set `RERANK_PROVIDER=local` for an offline cross-encoder) | dashboard.cohere.com |
| `ASSEMBLYAI_API_KEY` | transcribing **new** audio with speaker names (golden transcripts are committed) | assemblyai.com |

Fully local alternative for embeddings: `EMBEDDER=bge-base` (see [§9](#9-limitations)).

### One CLI for everything: `python -m audiosearch <command>`

| Command | What it does |
|---|---|
| `doctor` | checks database, keys, data, collections and indexes; tells you what to fix |
| `ingest` | ingests the golden set from `data/manifest.json` (clips and transcripts are committed, so no transcription call is made; it embeds the chunks) |
| `search "query"` | hybrid search from the terminal: `--method keyword\|vector\|fuzzy\|hybrid`, `--speaker`, `--role host\|guest`, `--mmr`, `--no-rerank`, `--context`, `--collection` |
| `add FILE` | ingests **any** audio file end to end: normalize → transcribe with diarization and speaker names → chunk → embed. `--collection`, `--title`, `--host 0\|1` |
| `collections [create NAME]` | lists or creates collections (isolated datasets, like MongoDB collections) |
| `eval [--rerank]` | builds the golden labels and prints the recall@k / MRR / no-answer / speaker-accuracy report (`eval/results.md`) |
| `bench` | latency p50 / p90 / p99 at concurrency 5 against the running API (`eval/latency.md`) |
| `test` | the automated test suite, including the recall@k gate |
| `report` | regenerates **all evidence** in [`test-results/`](test-results/README.md): evaluation tables, per-query hits, pytest output, latency |
| `serve` | API on :8000 (OpenAPI docs at `/docs`) + UI on :8501 |

Typical session:

```bash
python -m audiosearch doctor
python -m audiosearch ingest
python -m audiosearch search "what did Kranz say about accountability"
python -m audiosearch eval --rerank
python -m audiosearch test
python -m audiosearch serve             # then open http://localhost:8501
python -m audiosearch bench             # in a second terminal, while serve is running
```

**The UI:** search with a summary answer and the retrieved chunks below it, each with its speaker,
episode time, a "play from here" player and the surrounding dialogue. Also: a *compare methods*
view (keyword | vector | hybrid side by side), upload and ingest with every stage visible, a
library of transcripts, the evaluation dashboard, and a collection picker.

---

## 2. What the task asked and how it is met

| Task requirement | Where / how |
|---|---|
| **1. Golden dataset: 5–6 files of 8–10 min, each a unique pair of speakers** | 6 NASA *Houston We Have a Podcast* clips of **9:00** each, cut from full episodes ([`data/manifest.json`](data/manifest.json)). Pairs: Jordan/Zubair, Cheshire/Kranz, Jordan/O'Neil, Turner/Williams, Ramji/Murphy, Ramji/Garcia-Galan, all unique. Public domain (US government work). |
| **2. A hybrid strategy for exact words + semantically similar terms** | Three rankers fused with Reciprocal Rank Fusion: Postgres full-text (exact words and phrases), trigram (misspellings), pgvector cosine (meaning). Then parent grouping, reranking, relevance thresholds and optional MMR. See [§3](#3-design-and-rationale). |
| **3. Local solution; transcripts generated; results show file, timestamp, speaker** | Postgres/pgvector, the API, UI and search all run locally (Docker + Python). Transcripts are generated per file (AssemblyAI / Deepgram, allowed to be hosted). Each hit shows the **file, the exact second the matched word was spoken, and the speaker's name**, with the matched words highlighted. ⚠ Embeddings: see [§9](#9-limitations). |
| **4. Automated tests measuring recall@k against a labeled query set** | 76-query golden set with transcript-anchored labels; `eval/evaluate.py` reports recall@1/3/5/10, MRR, no-answer accuracy and speaker accuracy per configuration and query type; `tests/test_recall.py` **fails the build** if recall@5 < 0.90, recall@10 < 0.95, off-topic handling < 0.75 or speaker accuracy < 0.95. |
| Design, rationale, success criteria, achievement, limitations | This README |
| Coding-agent disclosure | [§10](#10-coding-agent-disclosure) and [AGENT_LOG.md](AGENT_LOG.md) |
| Code, golden dataset and tests in a repository | This repository: `data/` (clips, transcripts, manifest, golden spec and labels), `tests/`, `eval/` |

---

## 3. Design and rationale

```
audio ─► ffmpeg 16 kHz mono ─► ASR + diarization + speaker names (AssemblyAI; Deepgram supported)
      ─► normalized transcript (words, seconds, speaker 0/1, names)
      ─► speaker clean-up (exactly 2 speakers, diarization-flip smoothing)
      ─► parent chunks (~512 tokens of whole turns) ─► child chunks (~128 tokens, one speaker)
      ─► Postgres: child row = child_text + word timings + parent_text + speaker + episode time
             ├── tsvector + GIN         keyword / phrase ranker
             ├── pg_trgm GIN            typo ranker (short queries only)
             └── pgvector HNSW (1536-d) semantic ranker
query ─► speaker/role detection ("what did Jordan / the guest say about X")
      ─► keyword ∥ fuzzy ∥ vector ─► weighted RRF ─► group by parent (best child per parent)
      ─► Cohere rerank on the parent dialogue ─► relevance threshold ─► (MMR)
      ─► summary answer (Gemini Flash-Lite, cites results) + results: file · speaker · mm:ss
```

**Why these choices**

* **Hybrid, fused by rank (RRF).** Keyword search finds names, jargon and quotes exactly but misses
  paraphrases; vectors find paraphrases but blur rare names and can't say "no match". RRF combines
  them by rank only, so no score calibration between rankers is needed. The evaluation shows the
  complementarity: keyword recall@5 is 0.76 (semantic queries 0.46, typos 0.40), vector recall@5 is
  0.97, and the hybrid reaches 1.00.
* **Parent-child chunking.** The unit of retrieval is a **child** (~128 tokens, always one speaker)
  so the speaker and timestamp are exact. Each child carries its **parent** (~512 tokens of whole
  turns), so a question and its answer stay together for reranking, display and deduplication.
  Sizes are tuned to 9-minute conversations (about 1,750 tokens per file); see decision D6.
* **Reranking on the parent dialogue.** An answer often doesn't repeat the question's words. The
  cross-encoder reads (question, whole exchange) and fixes the ordering: it moved paraphrase
  queries from rank 2 to rank 1, and it is what makes off-topic queries answerable with "no match".
* **Thresholds on the reranker, not on cosine.** Measured: cosine similarity cannot separate
  relevant from irrelevant chunks (relevant 0.59–0.78, irrelevant up to 0.76); reranker scores can
  (answers 0.64–0.97, junk 0.26–0.30). So results below 0.40, or below 50% of the query's best
  score, are dropped even inside the top k.
* **Speakers are first-class.** Speaker names and host/guest roles are stored per file.
  "What did Kranz say about trust" / "what did the guest say about trust" become a speaker or role
  filter plus a topic search. The embedded text is `"Speaker: text"`, so meaning search also
  knows who spoke.
* **Exact timestamps.** Word-level timings are stored per chunk; a keyword hit points to the
  second the word was said. Times are stored per clip and shown in **episode time** (clip offset
  + time), with an `audio_url` per row.
* **One store: Postgres.** Text search, trigram, vectors, metadata and transactions in one engine:
  one query plan, one backup, no sync between a vector DB and a text index.
* **Collections.** Each tenant / dataset is a Postgres schema with identical tables (`nasa` →
  `col_nasa`), selected per connection via `search_path`: hard isolation and no query rewriting.

---

## 4. Decision log: the major architectural changes and why

Chronological. Each change was driven by a measurement, a test or a failed assumption. Full traces:
[AGENT_LOG.md](AGENT_LOG.md).

| # | Decision | Why |
|---|---|---|
| D1 | **Problem 1**, hosted ASR, everything else local | Objective metric (recall@k) and the most design space. The dev laptop (i5-8265U, 8 GB RAM, 2 GB MX250) is too weak for Whisper-large + pyannote, and the task allows hosted transcription. |
| D2 | **Rejected Gemini Live / separate VAD** | Live is for real-time conversation, and language-model transcription gives unreliable word timestamps, paraphrases words (breaks exact search) and is non-deterministic (breaks reproducible recall). Batch ASR's utterance segmentation already does what VAD would. |
| D3 | **Deepgram Nova-3 first** | Word timestamps and diarization in one call. It transcribed the first full episodes (163 min) in 46 s, with 0.997–0.999 confidence and exactly 2 speakers each. It also showed a real ASR issue: "Kranz" came back as both *Kranz* and *Krantz* (the fuzzy ranker and keyterm prompting address this). |
| D4 | **Switched transcription to AssemblyAI** | Deepgram's diarization gives only numbered speakers (confirmed in Deepgram's docs and forum), and the task needs "what did *Jordan* say". Researched options: context inference with a language model, speaking order, voice enrollment, episode data. **Tested AssemblyAI Speaker Identification on a 3-minute clip with no names supplied: it returned `{A: Gary Jordan, B: Patrick O'Neill}` from the conversation alone.** Transcription was made provider-neutral (normalized transcript format), so Deepgram remains a supported alternative (`TRANSCRIBER=deepgram`). |
| D5 | **Gemini `gemini-embedding-001` (1536-d) with task types** | The team chose a hosted embedding API over local BGE (see the limitation in §9). Mitigation: an embedder interface where `bge-base`/`bge-small` run locally with the same code, and vectors of several models side by side (one partial HNSW index per model). |
| D6 | **Chunking v1 → parent-child 128/512 tokens (tiktoken)** | v1 (~30 s speaker-turn windows) split questions from answers: "And how do you combine the two?" became its own 2-second chunk. A 512/2000-token child/parent split was proposed but rejected on arithmetic: a 9-minute file is ~1,750 tokens, so the parent would be the whole file (one result per file, meaningless recall) and 512-token children would mix both speakers. Result: 128-token children (25 overlap, one speaker) and 512-token parents of whole turns. |
| D7 | **Keyword ranker: term-coverage gate** | A smoke test showed hybrid ranking chunks that shared only the word "search" above the correct semantic answer (RRF trusts ranks). 3+ term queries now need ≥ 50% of their terms and are ranked by coverage. |
| D8 | **Cohere Rerank (default on) + MMR (opt-in)** | Reranking fixed paraphrase ordering. MMR (λ·relevance − (1−λ)·redundancy, using stored vectors, no API call) removes near-duplicate overlapping chunks when results feel repetitive. Trial limit of 10 calls/min: results are cached; interactive use falls back to the fused order and says so, while evaluation waits the limit out. |
| D9 | **Thresholds calibrated on data** | Cosine can't separate relevant from irrelevant (measured), reranker scores can, so the cut-off applies to reranker scores. Without it, "Mars helicopter flights" returned 3 confident-looking results. |
| D10 | **Speaker / role detection in queries** | Names were known but the query wasn't understood. The golden set then caught a bug: "affect **gene** expression" was read as the speaker **Gene** Kranz. Fix: a first or last name alone counts only in speaker context ("what did Gene…", "Kranz on…", "Jordan's"). |
| D11 | **FastAPI backend + Streamlit frontend, fully async** | The UI is a pure HTTP client; the backend owns the pool and models; slow work runs as background jobs with progress. One async code path serves the API, CLI, evaluation and tests. |
| D12 | **One DB connection per search; pipeline mode removed** | A load test (25 concurrent) showed requests waiting for pool connections when each ranker took its own connection, so each search now uses one connection. Later profiling: **psycopg pipeline mode was 7× slower than sequential here (48 ms vs 6.6 ms)** because of TCP delayed-ACK on Windows → Docker, so it was removed. The trigram ranker (~30 ms scan, 0% semantic recall) now runs only for queries of ≤ 4 terms. |
| D13 | **Collections as Postgres schemas** | Multi-tenant isolation with identical columns per collection; existing data migrated with `ALTER TABLE … SET SCHEMA` (embeddings kept, no re-embedding). A test proves isolation. |
| D14 | **Episode timeline + `audio_url`** | Clips are cut from longer episodes: times are stored clip-relative for playback, and `episode_*` columns (generated by Postgres) show episode time. |
| D15 | **Grounded summary answer** | The model sees only the top results, cites them as [n], and says so when they don't answer; no model call when there are no confident results. `gemini-2.5-flash-lite` returned *404: no longer available to new users*, so the default is `gemini-3.5-flash-lite` (configurable). |
| D16 | **Evaluation labels anchored to the transcript** | Hand-typed timestamps are unverifiable. Each query stores a verbatim anchor, and a script derives the answer passage. Labels were corrected twice, each time because a run showed a *correct* result scored as a miss (see §5). |

---

## 5. Evaluation: golden dataset, method, results

### Golden dataset
* **Audio:** 6 clips × 9:00 (`data/audio/ep{399,424,432,433,435,436}.mp3`), windows chosen from
  the full-episode transcripts for the most two-speaker back-and-forth, skipping intros and outros.
  Clip transcripts are **cut from the full-episode transcripts by timestamp** (no second transcription).
* **Queries:** [`data/golden_spec.json`](data/golden_spec.json), 76 queries:

  | keyword | phrase | entity | semantic | speaker | role | typo | negative |
  |---|---|---|---|---|---|---|---|
  | 9 | 7 | 11 | 26 | 6 | 7 | 5 | 5 |

  Semantic queries were deliberately worded away from the transcript ("what would people look like
  if we evolved on a much heavier planet" → *"we might look more like a lizard"*). Negatives are
  off-topic ("best coffee brewing methods") and must return **nothing**.

### How the labels are made (reproducible)
Each answerable query names its clip and an **anchor**: a phrase copied verbatim from the
transcript. `python -m eval.build_golden` finds the anchor's words and writes
[`data/queries.json`](data/queries.json) with the **answer passage**: the same speaker's turn
within ±30 s of the anchor, spanning short backchannels ("Right."), in **episode time**, plus the
speaker's name. Phrase queries accept any place the exact phrase is said.

### How the tests run
`python -m eval.evaluate` (or `python -m audiosearch eval --rerank`) sends every query through the
real search and checks each result: **same file, and does its time span overlap the labeled
passage (±2 s)?**
* **recall@k:** is the answer in the top k? · **MRR:** how high is it?
* **no-answer accuracy:** do off-topic queries return nothing?
* **speaker accuracy:** does the matching result carry the labeled speaker?

Evaluation searches only the golden clips (`origin=golden`), so demo uploads can't skew it.
`tests/test_recall.py` turns the thresholds into a pass/fail gate.

### Results (76 queries, 6 clips)

> **Evidence for every number below is in [`test-results/`](test-results/README.md)**: the evaluation
> tables, a **per-query file** showing the expected passage and the top-5 retrieved chunks marked
> ✅/▫️ for all 76 queries ([per_query_evidence.md](test-results/per_query_evidence.md)), the full
> `pytest -v` output (71 passed) and the latency run. Regenerate with `python -m audiosearch report`.

| configuration | R@1 | R@3 | R@5 | R@10 | MRR | off-topic → none | speaker acc. |
|---|---|---|---|---|---|---|---|
| keyword (Postgres FTS) | 0.72 | 0.75 | 0.76 | 0.76 | 0.73 | 1.00 | 0.96 |
| fuzzy (trigram) | 0.63 | 0.63 | 0.63 | 0.63 | 0.63 | 1.00 | 0.98 |
| vector (gemini) | 0.90 | 0.97 | 0.97 | 1.00 | 0.94 | 0.00 | 0.97 |
| hybrid, children ungrouped | 0.94 | 0.97 | 0.99 | 1.00 | 0.96 | 0.00 | 0.96 |
| hybrid (grouped by parent) | 0.94 | 0.97 | 1.00 | 1.00 | 0.96 | 0.00 | 0.96 |
| **hybrid + rerank** (default) | **0.96** | **1.00** | **1.00** | **1.00** | **0.98** | **0.80** | **0.97** |
| hybrid + rerank + MMR | 0.96 | 1.00 | 1.00 | 1.00 | 0.98 | 0.80 | 0.97 |

Recall@5 by query type (keyword-only → hybrid+rerank): semantic 0.46 → 0.96, typo 0.40 → 1.00,
speaker 0.83 → 1.00; entity, keyword and phrase are 1.00 for both. Not at rank 1: q23, q32 (rank 2),
q41 (rank 3). One false answer: n01 "Mars helicopter flights" (ep424 discusses Mars rover
autonomy, and the reranker scores it above the cut-off). This is reported, not tuned away.

### What evaluation caught (and how it was handled)
| Run | Finding | Action |
|---|---|---|
| 3 clips, labels = the anchor's 2–3 s | "affect **gene** expression" returned nothing | **real bug**: read as the speaker Gene Kranz, fixed (D10) |
| same | correct answers a sentence later in the same answer (q28) and the same quote by the host (q27) scored as misses | labels became answer passages (±30 s, same turn); phrase queries accept any occurrence |
| 6 clips | q62 "is there frozen water on the Moon…" → "a block of ice in the permanently shadowed regions" scored as a miss | passages now span backchannels ("Right."), mirroring the chunker |

Numbers before the label corrections: recall@5 0.93 (3 clips). All runs are in AGENT_LOG.md.

---

## 6. Latency

`python -m audiosearch bench`: 49 golden queries × 10 rounds after 1 warm-up, **5 requests in
flight**, over HTTP against the API, full pipeline (hybrid + rerank + threshold), warm caches.

| end-to-end | p50 | p90 | p99 | throughput | errors |
|---|---|---|---|---|---|
| 490 requests | **186 ms** | **274 ms** | **371 ms** | 24.9 req/s | 0 |

Per stage (server): retrieve p50 123 ms (at concurrency 1 the database work is ~7 ms, so the rest
is Python CPU queuing on one process), rerank 2 ms cached, post-processing < 1 ms. Per-query
p50/p90/p99 are in `eval/latency.md`. **Cold path** (from logs): Gemini query embedding 0.5–1.7 s,
Cohere rerank 0.5–0.8 s, summary answer 1–2 s; each is cached after the first call.
Every `/search` response includes `timings_ms` per stage. (Benchmarked before the last 3 clips
were added; re-run with `bench`.)

---

## 7. Success criteria and level of achievement

| Criterion (set before building) | Target | Achieved |
|---|---|---|
| Recall@5 of the full pipeline | ≥ 0.90 | **1.00** ✅ |
| Recall@10 of the full pipeline | ≥ 0.95 | **1.00** ✅ |
| Hybrid ≥ each single method (R@5, R@10, MRR) | not worse than −0.02 | ✅ (1.00 vs 0.76 keyword / 0.97 vector) |
| Every query type served (R@10 ≥ 0.8) | all types | ✅ |
| Off-topic queries answered with "no match" | ≥ 0.75 | **0.80** ✅ (4/5) |
| Speaker attribution of hits | ≥ 0.95 | **0.97** ✅ |
| Results show file, timestamp, speaker | always | ✅ plus highlighted words, episode time, audio link |
| Interactive latency (p95 of a single query) | < 1.5 s incl. hosted calls | ✅ warm p99 371 ms at concurrency 5; cold ~1.5–2.5 s |

---

## 8. Taking it to production: metrics and evaluation

**Retrieval quality:** recall@k, MRR / nDCG on a growing golden set labeled by people who didn't
build the system; **no-answer precision** (false confident answers); zero-result rate; per query
type and per collection. Online: click-through / play-from-timestamp rate, reformulation rate,
answer thumbs-up/down, and citation click-through on summary answers.

**Upstream audio quality** (it bounds everything downstream): word error rate on hand-corrected
samples per source (for example the *Kranz/Krantz* and *"tough incompetent"* errors seen here), diarization
error rate, speaker-name accuracy (identification can mis-spell or mis-assign), backchannel and
overlapping-speech handling.

**System:** p50/p90/p99 per stage (already in every response), throughput, pool saturation
(`/health`), cache hit rates, external API latency, error and rate-limit rates, cost per audio
hour (ASR) and per 1k queries (embedding + rerank + answer), ingestion lag, index size and HNSW
recall versus exact search.

**Evaluation process:** the recall gate runs in CI on every change (it already fails the build
below thresholds); nightly full evaluation with rerank; a blind-labeled golden set per collection;
A/B tests of ranker weights, thresholds and models; drift monitoring on score distributions; a
re-embedding plan per model version (the embeddings table is keyed by model, so versions can
coexist during migration).

**Scale-out path:** multiple API workers (CPU was the measured bottleneck); a durable job queue
(Postgres jobs table or a worker queue) instead of in-memory jobs; HNSW `m` / `ef_search` tuning and
per-collection partitioning; a self-hosted reranker for cost and rate limits; streaming ingestion
with idempotent file ids (already content-hashed).

---

## 9. Limitations

* **Embeddings are hosted (Gemini), while the task says "embedding generation and indexing should
  run locally".** Indexing (pgvector, full-text, trigram) is local; the query and document vectors
  come from the Gemini API by team decision. A local path exists and is implemented
  (`EMBEDDER=bge-base` / `bge-small` via sentence-transformers, `RERANK_PROVIDER=local` for a local
  cross-encoder); it has **not been evaluated** on the golden set in this submission.
* **Small, builder-written golden set.** 76 queries over 6 clips (152 chunks), written after reading
  the transcripts, so recall@5 saturates at 1.00. Recall@1, MRR and off-topic handling are the
  discriminating metrics. More audio and blind labeling are needed.
* **ASR errors propagate** into search and answers (e.g. *Krantz*, *"tough incompetent"* for *"failure is not an option"*).
* **Speaker identification is not verified automatically.** Names come from context (AssemblyAI) or
  the manifest; AssemblyAI occasionally adds artefacts ("Carlos Garcia-Galan - 1"), and the UI lets
  users correct them. ep399 is guest-dominated (the host has ~11% of the words).
* **Off-topic handling is 4/5.** Topically adjacent queries can pass the reranker cut-off. The
  thresholds (0.40 absolute / 0.5 relative) were calibrated on a small sample.
* **Latency figures are warm-cache.** They exclude external API time; single process; benchmarked on a laptop.
* **Operational gaps:** in-memory background jobs (lost on restart); no collection deletion;
  uploads of full episodes also appear next to their golden clips in the UI; trial-tier rate
  limits (Cohere 10 calls/min).

---

## 10. Coding-agent disclosure

**How the agent was steered: the artifacts**

| Artifact | Role |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Standing rules the owner gave the agent: architecture constraints, API budget rules, evaluation rules, definition of done |
| [`.claude/skills/`](.claude/skills) | Reusable workflows the agent follows step by step: [`golden-eval`](.claude/skills/golden-eval/SKILL.md) (add queries → build labels → evaluate → investigate misses → report), [`add-episode`](.claude/skills/add-episode/SKILL.md) (golden clip or upload) |
| [`docs/adr/`](docs/adr/README.md) | Architecture Decision Records: the owner's major decisions with context, options and evidence |
| [`AGENT_LOG.md`](AGENT_LOG.md) | Prompt-by-prompt trace: what was asked, what the agent proposed and did, how it was verified |
| `tests/` + `ruff` + `eval` | Guardrails: the agent's work counts as done only when lint, the tests and the recall@k gate pass |

`CLAUDE.md` and the skills codify the rules and workflows used throughout development; they were
written down as files at the end of the project.


This project was built with **Claude Code** (Anthropic) as a pair-programming agent. **The project
owner led the architecture and made every major design decision**: problem choice, providers,
chunking scheme, retrieval pipeline, multi-tenancy, evaluation approach and scope. The agent
proposed options with trade-offs, implemented what the owner decided, ran the measurements and
reported findings for the owner to act on. How it was directed:

* **Requirements came as prompts** at each step ("plan what is asked", "make it async and scalable",
  "child 128 / parent 512 tokens, embeddings on child text, dedup by parent_id", "use AssemblyAI to
  get speaker names", "collections like MongoDB", "add a summary answer", "p50/p90/p99 at
  concurrency 5"). Each prompt and the agent's response is logged in **[AGENT_LOG.md](AGENT_LOG.md)**.
* **Owner-led design decisions:** hybrid search with reranking and MMR; hosted Gemini embeddings
  (1536-d); Deepgram, then AssemblyAI for speaker names; the parent-child chunking scheme with its
  `parent_id` deduplication; the asynchronous backend/frontend split; collections per tenant; the
  episode timeline with `audio_url`; the summary answer; the golden-set scope and the latency targets;
  API budget limits on free-tier keys.
* **Agent analysis the owner evaluated before deciding:** trade-offs of Gemini Live versus batch ASR,
  chunk-size arithmetic for 9-minute files (which set the final 128/512 sizes), measured score
  distributions for the thresholds, the pipeline-mode measurement, and reporting the "Mars" false
  answer rather than tuning it away.
* **Verification discipline:** the agent was asked to test before claiming success. Unit, API,
  database and recall-gate tests (71 passing); logs with request ids; live checks kept to a
  minimum because the keys are free-tier.
