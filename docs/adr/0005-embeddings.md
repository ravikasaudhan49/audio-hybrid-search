# ADR-0005: Embeddings: Gemini (1536-d) behind a swappable interface

**Status:** accepted · **Decided by:** project owner
**Context.** The task text asks for local embedding generation; the owner chose a hosted model
for quality and speed on the dev laptop.

**Decision.** `gemini-embedding-001` with RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY task types,
truncated to 1536 dimensions and re-normalized. An embedder interface keeps local `bge-base` /
`bge-small` interchangeable; vectors of several models can coexist (one partial HNSW index per model).

**Consequences.** Query embedding is a network call (cached). Local-only operation is available
(`EMBEDDER=bge-base`, `RERANK_PROVIDER=local`) but has not been evaluated; listed as a limitation.
