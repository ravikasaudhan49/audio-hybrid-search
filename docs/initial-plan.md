# Initial plan (superseded; kept for the record. See README.md and docs/ARCHITECTURE.md)

## 1. What is asked (deliverables)

| # | Requirement | What we ship |
|---|---|---|
| 1 | Golden dataset: 5–6 audio files, 8–10 min each, each with a **unique pair** of speakers | `data/audio/*.mp3` + `data/manifest.json` (source URL, license, speaker names) |
| 2 | Hybrid search strategy (exact words + semantic) | Design section in `README.md` |
| 3 | Local hybrid search; transcripts from audio; results show **file, timestamp, speaker** | Ingest pipeline + Postgres/pgvector + CLI (and small web UI) |
| 4 | Automated tests measuring **recall@k** on a labeled query set | `data/queries.json` + `pytest` eval suite |
| — | Write-up: design, rationale, success criteria, achievement vs criteria, limitations, **production metrics & evaluation** | `README.md` |
| — | Coding-agent disclosure: how we prompted / directed the agent | README §10 |
| — | Public GitHub repo with code, golden dataset, tests | GitHub |

Constraints: Python or TypeScript; RDBMS with vectors (Postgres + pgvector). **Embeddings + indexing must be local.** Transcription may be hosted.

## 2. Machine constraints (checked)

- CPU i5-8265U, 8 GB RAM, GeForce MX250 (2 GB) → too weak for fast local Whisper-large + pyannote.
- Not installed yet: Python, Docker, ffmpeg, Postgres.

**Decision:** hosted transcription + diarization (Deepgram Nova-3, `diarize=true`), everything else local on CPU.

## 3. What to install / sign up for

| Item | Why | How |
|---|---|---|
| Python 3.11 | Pipeline, tests | python.org installer, tick "Add to PATH" (or `winget install Python.Python.3.11`) |
| Docker Desktop | Runs Postgres + pgvector without compiling pgvector on Windows | docker.com (uses WSL2) |
| ffmpeg | Trim podcasts to 8–10 min, normalize to 16 kHz mono | `winget install Gyan.FFmpeg` |
| **Deepgram API key** (only API needed) | Transcript with word timestamps + speaker labels (Nova-3, diarize) | console.deepgram.com; free credit easily covers ~1 h of audio. ElevenLabs Scribe is a fallback |
| GitHub account/repo | Submission | github.com |

Optional: Hugging Face token (only if we add a local WhisperX + pyannote comparison run).
No OpenAI/Gemini key needed — embeddings run locally.

## 4. Architecture

```
audio (mp3) ──► Deepgram Nova-3 (words + timestamps + speaker A/B)
                     │  cache raw JSON in data/transcripts/
                     ▼
              segmenter: speaker turns → chunks (~20–45 s, split long turns,
                         small overlap), keep word-level timestamps
                     │  map A/B → real speaker names via manifest
                     ▼
              local embedder (sentence-transformers, BAAI/bge-small-en-v1.5, CPU)
                     ▼
Postgres 16 + pgvector
  files(id, title, source, duration)
  speakers(id, file_id, label, name)
  segments(id, file_id, speaker_id, start_ms, end_ms, text,
           tsv tsvector  -- GIN index (keyword)
           embedding vector(384)  -- HNSW index, cosine (semantic)
           + pg_trgm index (fuzzy / misspellings))
  words(segment_id, word, start_ms, end_ms)  -- exact-hit timestamps
                     ▼
query ──► keyword (websearch_to_tsquery + ts_rank_cd, phrase support)
      └─► semantic (embed query, HNSW k-NN)
          ──► Reciprocal Rank Fusion (k=60) ──► optional local cross-encoder rerank
          ──► result: file · [mm:ss] · speaker · highlighted snippet
```

Interfaces: `search "query"` CLI + minimal FastAPI page with an audio player that jumps to the timestamp.

## 5. Evaluation

- `data/queries.json`: ~40–60 labeled queries, each → (file, start–end span). Mix of types:
  exact keyword, exact phrase, rare named entity, paraphrase/semantic, speaker-scoped ("what did X say about Y"), misspelled.
- A hit = returned segment in the right file overlapping the labeled span.
- Metrics: recall@1/5/10, MRR, per query type, for **keyword-only vs vector-only vs hybrid vs hybrid+rerank**.
- `pytest` asserts hybrid recall@5 ≥ target and ≥ each single method.
- Also report transcript quality spot-check (WER on a hand-corrected 1-min excerpt per file) and diarization sanity.

Success criteria (draft): hybrid recall@5 ≥ 0.90, recall@10 ≥ 0.95, beats both single methods overall; p95 query latency < 300 ms locally.

## 6. Production metrics to discuss in write-up

Retrieval: recall@k, MRR/nDCG on growing golden set, zero-result rate, click-through / play-at-timestamp rate.
Upstream quality: WER, diarization error rate (DER), speaker-attribution accuracy.
System: p50/p95 latency, ingestion throughput & cost per audio hour, index size / memory, HNSW recall vs exact scan, reindex time on model change.
Scale notes: partitioning, HNSW params (m, ef_search), embedding-model versioning, BM25 via ParadeDB/pg_search if tsvector ranking is insufficient.

## 7. Build phases

1. Environment: install tools, `docker compose up` Postgres+pgvector, Python venv.
2. Dataset: choose 5–6 podcasts/interviews (unique speaker pairs, reusable license), trim with ffmpeg, manifest.
3. Transcribe + cache (Deepgram), segmenter, speaker naming.
4. Schema, embed, index.
5. Keyword, semantic, RRF, highlight; CLI.
6. Label queries; eval harness + pytest; tune chunk size / fusion / rerank with data.
7. Web UI (nice-to-have).
8. README write-up, push to GitHub.
