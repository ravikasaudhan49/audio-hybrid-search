# ADR-0004: Relevance thresholds on reranker scores, not cosine

**Status:** accepted · **Decided by:** project owner
**Context.** RRF uses ranks, so off-topic queries still returned confident-looking top-5 results.

**Evidence.** On cached sample queries: relevant cosine 0.59–0.78, irrelevant up to 0.76, an off-topic
query's best 0.58, so cosine cannot separate them. Cohere rerank scores: answers 0.64–0.97, junk
0.26–0.30.

**Decision.** A low vector floor (cosine ≥ 0.50) only as a safety net; after reranking, drop results
below 0.40 or below 50% of the query's best score; with none left, answer "no confident match".

**Consequences.** Off-topic queries → no answer: 4/5 on the golden set; topically adjacent queries
can pass. Never tuned per query.
