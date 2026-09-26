# ADR-0006: One DB connection per search; no pipeline mode

**Status:** accepted · **Decided by:** project owner
**Context.** First async version: each ranker took its own pooled connection (4 per search).
Load test at 25 concurrency: requests waited for pool connections.

**Decision.** Embed the query first (never while holding a connection), then run all rankers and
hydration on one connection. Later profiling showed psycopg pipeline mode at 48 ms versus 6.6 ms
sequential (TCP delayed-ACK on Windows → Docker), so queries run back to back. The trigram ranker
(~30 ms scan, 0 semantic recall) only runs for queries of ≤ 4 terms.

**Consequences.** Concurrency 5: p50 186 ms, p90 274 ms, p99 371 ms (warm). The remaining time is
Python CPU on one process; scale out with more workers.
