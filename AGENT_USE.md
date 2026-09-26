# How the coding agent was used

**Tool:** Claude Code (Anthropic, Claude Opus 5.5), used as a pair programmer in the terminal.

## Roles
* **Project owner:** defined the problem scope and requirements, led the architecture, chose every
  provider and key parameter, reviewed the output of each step and decided what to change next.
* **Claude Code:** wrote the code (library, backend, frontend, CLI, evaluation, tests), ran the
  commands, tests, evaluations and benchmarks, and reported results back to the owner.

## How the agent was directed
1. **Feature by feature, through written requirements.** Each step started with an explicit
   instruction from the owner, for example:
   - "Hybrid keyword + semantic search on Postgres with pgvector; results must show file, timestamp and speaker."
   - "Make everything asynchronous; maintain a connection pool that can be sized for throughput."
   - "Child chunks of 128 tokens and parent chunks of 512, embeddings on the child text, one
     `parent_id` per parent, deduplicate results by `parent_id`; count tokens with tiktoken."
   - "Use 1536-dimensional Gemini embeddings; Cohere reranking on by default; MMR as a UI toggle."
   - "Use AssemblyAI so results carry speaker names; answer queries like 'what did Jordan say'."
   - "Collections like MongoDB: every user can create their own knowledge base and search within it."
   - "Show episode time and an `audio_url` on every row."
   - "Add a grounded summary answer above the retrieved chunks."
   - "Build a golden dataset of 6 clips and measure recall@k; report latency p50/p90/p99 at concurrency 5."
2. **Standing rules in [`CLAUDE.md`](CLAUDE.md).** The layering, one DB connection per search, the
   chunking contract, API-budget rules for the free-tier keys (no paid calls in tests, cache every
   hosted call), evaluation rules (labels only from transcript anchors, never tune to a single
   query) and the definition of done (lint + tests + `doctor`).
3. **Repeatable workflows as skills** in [`.claude/skills/`](.claude/skills):
   - [`golden-eval`](.claude/skills/golden-eval/SKILL.md): write queries → build labels → estimate
     API cost → evaluate → investigate every miss → report.
   - [`add-episode`](.claude/skills/add-episode/SKILL.md): add an episode as a normal upload or as
     a golden 9-minute clip.
4. **Decisions recorded as ADRs** in [`docs/adr/`](docs/adr/README.md), one per major design choice.

## How the agent's output was verified
* **Tests as guardrails:** 71 automated tests (unit, API, database, and a recall@k gate that fails
  the build below the success criteria). A change counted as done only when `ruff` and `pytest` passed.
* **Evidence, not claims:** every reported number comes from a command that can be re-run
  (`python -m audiosearch eval --rerank`, `bench`, `report`), with the outputs stored in
  [`test-results/`](test-results/README.md).
* **Observability:** request-id logging and per-stage timings made each step inspectable.
* **I reviewed every step** in the UI and CLI before moving on, and redirected the work where
  needed (e.g. switching transcription provider for speaker names, limiting API usage, scoping
  the golden set).
