# Architecture Decision Records

Short records of the major design decisions (context, options, decision, consequences, evidence).
Each was decided by the project owner.

| ADR | Decision |
|---|---|
| [ADR-0001](0001-hybrid-search-on-postgres.md) | Hybrid search on a single Postgres store |
| [ADR-0002](0002-transcription-provider.md) | Transcription: hosted ASR, Deepgram then AssemblyAI |
| [ADR-0003](0003-parent-child-chunking.md) | Parent-child chunking (128 / 512 tokens) |
| [ADR-0004](0004-relevance-thresholds.md) | Relevance thresholds on reranker scores, not cosine |
| [ADR-0005](0005-embeddings.md) | Embeddings: Gemini (1536-d) behind a swappable interface |
| [ADR-0006](0006-one-connection-per-search.md) | One DB connection per search; no pipeline mode |
| [ADR-0007](0007-collections-as-schemas.md) | Collections as Postgres schemas |
| [ADR-0008](0008-anchored-golden-labels.md) | Golden labels anchored to transcript quotes |
