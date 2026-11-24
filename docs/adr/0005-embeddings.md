# ADR-0005: Embeddings: Gemini (1536-d), hosted only

**Status:** accepted · **Decided by:** project owner
**Context.** The task text asks for local embedding generation; the owner chose a hosted model
for quality and speed on the dev laptop.

**Decision.** `gemini-embedding-001` with RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY task types,
truncated to 1536 dimensions and re-normalized. The embedder is built once at startup
(`AppContainer`); the `Embedder` base class stays, so another provider can be added later.

*Revised:* the local `bge-base` / `bge-small` embedders and per-request model choice were removed
by owner decision ("we won't be using it; only APIs"). Gemini is the only embedder.

**Consequences.** Query embedding is a network call (cached). There is no local embedding path
any more, although the task text asks for local embedding generation; listed as a limitation.
