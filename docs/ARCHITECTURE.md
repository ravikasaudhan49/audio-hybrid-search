# Architecture

## 1. Components

```mermaid
flowchart LR
    subgraph Clients
        UI["Streamlit UI<br/>frontend/app.py"]
        CLI["CLI<br/>python -m audiosearch"]
        EVAL["Evaluation + tests<br/>eval/, tests/"]
    end
    subgraph Backend["FastAPI backend (async) · backend/"]
        API["REST API<br/>/search /files /uploads /collections /eval /health"]
        JOBS["Background jobs<br/>transcribe · index · eval"]
    end
    subgraph Core["audiosearch library (async core)"]
        TR["transcribe.py<br/>provider-neutral transcripts"]
        SEG["segment.py<br/>speaker clean-up · parent/child chunks"]
        EMB["embed.py<br/>Gemini / local BGE"]
        SRCH["search.py<br/>rankers · RRF · group · rerank · MMR"]
        QRY["query.py<br/>speaker / role detection"]
        ANS["answer.py<br/>grounded summary"]
    end
    subgraph Local["Local infrastructure (Docker)"]
        PG[("Postgres 17<br/>pgvector · pg_trgm · FTS<br/>one schema per collection")]
    end
    subgraph Hosted["Hosted APIs (cached)"]
        AAI["AssemblyAI<br/>ASR + diarization + speaker names"]
        DG["Deepgram Nova-3<br/>ASR + diarization"]
        GEM["Gemini<br/>embeddings · Flash-Lite answers"]
        COH["Cohere Rerank v4"]
    end
    UI -- HTTP --> API
    CLI --> Core
    EVAL --> Core
    API --> JOBS --> Core
    API --> Core
    TR --> AAI & DG
    EMB --> GEM
    SRCH --> COH
    ANS --> GEM
    Core --> PG
```

* The **UI is a pure HTTP client**: no database or model access, so it can be replaced.
* The **backend** owns the async connection pool (2–10 connections) and the loaded models; slow
  work (transcription, embedding, evaluation) runs as background jobs that clients poll.
* The **core library** is used directly by the CLI, evaluation and tests, so recall numbers measure
  retrieval, not HTTP.
* **Hosted calls are cached on disk** (`.cache/`: query embeddings, rerank scores, answers;
  `data/transcripts/`: raw and normalized ASR output), so re-runs are free and reproducible.
* **Logs:** one rotating text file per process in `logs/`; every line carries a request or job id.

## 2. Ingestion

```mermaid
flowchart TD
    A["audio file<br/>(UI upload · `add FILE` · manifest)"] --> B["ffmpeg → 16 kHz mono mp3<br/>(clip cut for golden files)"]
    B --> C{"transcript cached?"}
    C -- "no" --> D["ASR provider<br/>AssemblyAI: words + speakers + NAMES<br/>Deepgram: words + speaker numbers"]
    C -- "clip of a transcribed episode" --> S["slice full transcript by time<br/>(no API call)"]
    D --> N["normalized transcript<br/>words · seconds · speaker 0/1 · names"]
    S --> N
    N --> K["speaker clean-up<br/>2 speakers · flip smoothing"]
    K --> P["parents: whole turns ≤ 512 tokens"]
    P --> CH["children: ≤ 128 tokens, one speaker,<br/>25-token overlap, backchannels skipped"]
    CH --> DB["one transaction:<br/>files · speakers(role) · segments"]
    DB --> E["embed 'Speaker: child_text'<br/>batched, concurrent (Gemini 1536-d)"]
    E --> H["HNSW index per model"]
```

## 3. Query path

```mermaid
sequenceDiagram
    participant U as UI / client
    participant A as API /search
    participant Q as query.py
    participant G as Gemini (cached)
    participant P as Postgres (1 pooled connection)
    participant C as Cohere (cached)
    participant L as Gemini Flash-Lite
    U->>A: q, collection, options
    A->>Q: detect speaker / role ("what did Jordan say about X")
    Q-->>A: filter = Gary Jordan, text = "X"
    A->>G: embed query text (RETRIEVAL_QUERY)
    A->>P: SET search_path = col_<collection>
    A->>P: keyword (tsvector, coverage gate) · fuzzy (≤ 4 terms) · vector (HNSW, sim ≥ 0.50)
    A->>A: weighted RRF → top 30 → group by parent_id
    A->>P: hydrate: child/parent text, word timings, speaker, episode offset
    A->>C: rerank (question, parent dialogue)
    A->>A: threshold (≥ 0.40 and ≥ 50% of best) → optional MMR → top k
    A->>L: optional grounded answer from top 5 (cites [n])
    A-->>U: answer + results (file · speaker · episode mm:ss · highlighted text · timings_ms)
```

## 4. Data model (per collection schema `col_<name>`)

```mermaid
erDiagram
    FILES ||--o{ SPEAKERS : has
    FILES ||--o{ SEGMENTS : has
    SPEAKERS ||--o{ SEGMENTS : "spoke"
    SEGMENTS ||--o{ EMBEDDINGS : "vector per model"
    FILES {
        text id PK
        text title
        text audio_path "local clip"
        text audio_url "source episode"
        real source_offset_s "clip start in episode"
        real duration_s
        text origin "golden | upload"
    }
    SPEAKERS {
        int id PK
        text file_id FK
        int label "0 | 1"
        text name
        text role "host | guest"
    }
    SEGMENTS {
        int id PK
        text file_id FK
        int speaker_id FK
        real start_s "clip time"
        real end_s
        real episode_start_s "generated"
        text child_text "~128 tokens"
        jsonb words "word, start, end"
        text parent_id "file:pNN"
        text parent_text "~512 tokens dialogue"
        tsvector tsv "generated, GIN"
        text audio_url
    }
    EMBEDDINGS {
        int segment_id PK
        text model PK
        vector embedding "partial HNSW per model"
    }
```

`public.collections(name, description, created_at)` registers the collections; every collection is
created from the same `audiosearch/schema.sql`, so columns are identical. Queries select a
collection with `search_path`, and the path is reset when the connection returns to the pool.

## 5. Key parameters (`audiosearch/config.py`, overridable in `.env`)

| Parameter | Value | Why |
|---|---|---|
| `CHILD_TOKENS` / overlap / `PARENT_TOKENS` | 128 / 25 / 512 (tiktoken cl100k) | ~40–60 s children in one speaker's turn; 2–3 min parents keep a question and its answer together |
| `MIN_CHILD_WORDS` | 3 | backchannels ("Right.") stay in the parent, not as results |
| RRF k, weights | 60; keyword 1.0, vector 1.0, fuzzy 0.5 | rank-only fusion; typo ranker down-weighted |
| `FUZZY_MAX_TERMS` | 4 | trigram only for short lookups (measured ~30 ms scan, 0 semantic recall) |
| `VECTOR_MIN_SIM` | 0.50 | safety floor; cosine doesn't separate relevant from irrelevant (measured) |
| `RERANK_MIN_SCORE` / `_RELATIVE` | 0.40 / 0.5 × best | calibrated: answers 0.64–0.97, junk 0.26–0.30 |
| `RERANK_DEPTH` | 30 | candidates re-scored per query |
| `MMR_LAMBDA` | 0.6 | relevance versus diversity when MMR is on |
| `DB_POOL_MIN/MAX` | 2 / 10 | one connection per search, held for milliseconds |
| `EMBEDDER` / dims | gemini-embedding-001 / 1536 | HNSW supports ≤ 2000 dims; `bge-base` for local |
| `ANSWER_MODEL` | gemini-3.5-flash-lite | 2.5-flash-lite is no longer available to new users |

## 6. Multi-tenancy and scaling

* **Tenants = collections** (one Postgres schema each, identical tables, per-collection indexes).
  Users create their own knowledge base via UI / `POST /collections` / CLI, then upload and
  search inside it; isolation is enforced by `search_path` per connection and covered by a test.
* **Capacity levers:** `DB_POOL_MIN/MAX` (one connection per search, held for milliseconds),
  `HTTP_MAX_CONNECTIONS`, `EMBED_CONCURRENCY`, `INGEST_CONCURRENCY`, `JOB_CONCURRENCY`, and uvicorn
  workers or replicas. Current: p50 186 / p90 274 / p99 371 ms at concurrency 5 on one worker.
* **Next steps at scale:** durable job queue (Postgres jobs table or a worker queue) instead of
  in-memory jobs; PgBouncer and read replicas; HNSW tuning; moving large tenants to their own database;
  a self-hosted reranker to remove third-party rate limits.
