# ADR-0001: Hybrid search on a single Postgres store

**Status:** accepted · **Decided by:** project owner
**Context.** Users search both for exact words (names, quotes, jargon) and for meaning. Vector
search blurs rare names and can't say "no match"; keyword search misses paraphrases.

**Options.** (a) a vector DB plus a separate text index; (b) Postgres with pgvector, tsvector and
pg_trgm in one engine.

**Decision.** (b). Three rankers (full-text, trigram, cosine) fused with weighted Reciprocal Rank
Fusion (k=60), then parent grouping, reranking, thresholds and optional MMR.

**Consequences.** One store, one transaction, one backup; rank-only fusion needs no score
calibration. Postgres HNSW is limited to ≤ 2000 dimensions.

**Evidence.** Golden set recall@5: keyword 0.76, vector 0.97, hybrid 1.00 (README §5).
