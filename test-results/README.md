# Test results

Generated 2026-09-26 14:56 by `python -m eval.report` (golden set: 76 queries,
collection `nasa`, golden files only).

**Default pipeline (hybrid + rerank):** recall@1 0.96 · recall@3 1.00 ·
recall@5 1.00 · recall@10 1.00 · MRR 0.98 ·
off-topic → no result 0.80 · speaker accuracy 0.97

**pytest:** `============================= 71 passed in 30.69s =============================`

| File | Contents |
|---|---|
| [evaluation_summary.md](evaluation_summary.md) | recall@1/3/5/10, MRR, no-answer and speaker accuracy, latency, per configuration and per query type |
| [evaluation_full.json](evaluation_full.json) | the same plus per-query first-hit ranks for every configuration |
| [per_query_evidence.md](per_query_evidence.md) | every query: expected passage, top-5 retrieved chunks marked hit/miss, verdict |
| [pytest_output.txt](pytest_output.txt) / [pytest_junit.xml](pytest_junit.xml) | the automated test suite, including the recall@k gate (`tests/test_recall.py`) |
| [latency.md](latency.md) / [latency.json](latency.json) | p50/p90/p99 at concurrency 5 (`python -m audiosearch bench`, needs `serve` running) |

How a hit is decided: a retrieved chunk counts if it is from the labeled file and its time span overlaps
the labeled answer passage (±2 s). Labels come from verbatim transcript anchors
(`data/golden_spec.json` → `python -m eval.build_golden` → `data/queries.json`).
