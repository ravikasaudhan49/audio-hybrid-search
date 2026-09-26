"""Write the evidence behind the reported numbers into test-results/.

  python -m eval.report            (also: python -m audiosearch report)

  test-results/
    README.md                    what each file is and how it was produced
    evaluation_summary.md        recall@k / MRR / no-answer / speaker accuracy per configuration and query type
    evaluation_full.json         the same, plus per-query ranks for every configuration
    per_query_evidence.md        for EVERY golden query (default pipeline): expected passage, top-5 retrieved
                                 chunks (file, episode time, speaker, text) marked hit/miss, first-hit rank
    pytest_output.txt            full `pytest -v` run (unit, API, DB, recall@k gate)
    pytest_junit.xml             the same in JUnit format
    latency.md / latency.json    last `python -m audiosearch bench` run (concurrency 5)

Uses the same code path as the evaluation; with warm caches it makes no API calls.
"""
import json
import shutil
import subprocess
import sys
from datetime import datetime

from audiosearch import aio, config, db
from audiosearch import log as logs
from audiosearch.golden import is_hit, load_queries
from audiosearch.search import Searcher, fmt_ts
from eval import build_golden
from eval.evaluate import RESULTS_JSON, RESULTS_MD, run_evaluation

OUT = config.ROOT / "test-results"
TOP = 5


def _passage(rel: dict) -> str:
    return f"{rel['file']} {fmt_ts(rel['start'])}–{fmt_ts(rel['end'])} ({rel.get('speaker') or '?'})"


async def per_query_evidence(pool) -> str:
    """Default pipeline (hybrid + rerank + threshold), golden files only."""
    s = Searcher(pool, origin="golden")
    lines = ["# Per-query evidence: default pipeline (hybrid + rerank + relevance threshold)", "",
             "For each golden query: the labeled answer passage (from the transcript anchor), then the top "
             f"{TOP} retrieved chunks. ✅ = same file and overlapping the labeled passage (±2 s).", ""]
    found_at = {}
    for q in load_queries():
        out = await s.search(q["query"], "hybrid", 10, patient_rerank=True)
        rels = q["relevant"]
        lines.append(f"### {q['id']} · {q['type']} · “{q['query']}”")
        if rels:
            lines.append("**Expected:** " + " **or** ".join(_passage(r) for r in rels)
                         + (f" · anchor: “{rels[0]['anchor']}”" if rels[0].get("anchor") else ""))
        else:
            lines.append("**Expected:** no result (off-topic query)")
        if out.speaker_detected or out.role_detected:
            lines.append(f"**Query understanding:** filter = {out.speaker_detected or out.role_detected}, "
                         f"searched “{out.search_text}”")
        first = None
        for i, r in enumerate(out.results, 1):
            hit = bool(rels) and any(is_hit(r, x) for x in rels)
            first = first or (i if hit else None)
            if i <= TOP:
                text = r.child_text.replace("|", "/")
                lines.append(f"{i}. {'✅' if hit else '▫️'} `{r.file_id}` {fmt_ts(r.episode_start_s)}–"
                             f"{fmt_ts(r.episode_end_s)} · {r.speaker} · score {r.score:.2f} · "
                             f"{text[:140]}{'…' if len(text) > 140 else ''}")
        if not out.results:
            lines.append("_(no results: " + ("; ".join(out.notes) or "empty") + ")_")
        verdict = ("✅ correct: no result" if not rels and not out.results else
                   "❌ returned results for an off-topic query" if not rels else
                   f"✅ first hit at rank {first}" if first else "❌ not found in top 10")
        lines += [f"**Verdict:** {verdict}", ""]
        found_at[q["id"]] = first if rels else (not out.results)
    pos = [v for k, v in found_at.items() if not k.startswith("n")]
    neg = [v for k, v in found_at.items() if k.startswith("n")]
    summary = [f"**Summary:** {sum(1 for v in pos if v == 1)}/{len(pos)} answerable queries at rank 1, "
               f"{sum(1 for v in pos if v and v <= 5)}/{len(pos)} in the top 5; "
               f"{sum(neg)}/{len(neg)} off-topic queries correctly returned nothing.", ""]
    return "\n".join(lines[:4] + summary + lines[4:])


async def build() -> None:
    OUT.mkdir(exist_ok=True)
    pool = await db.open_pool(min_size=1, max_size=4)
    try:
        print("1/4 evaluating all configurations …")
        await run_evaluation(pool, [config.EMBEDDER], rerank=True)
        shutil.copy(RESULTS_MD, OUT / "evaluation_summary.md")
        shutil.copy(RESULTS_JSON, OUT / "evaluation_full.json")
        print("2/4 per-query evidence …")
        (OUT / "per_query_evidence.md").write_text(await per_query_evidence(pool), encoding="utf-8")
    finally:
        await pool.close()
        await aio.close_http()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logs.setup("eval")
    if build_golden.main():
        return 1
    aio.run(build())
    print("3/4 pytest …")
    run = subprocess.run([sys.executable, "-m", "pytest", "-v", "--junitxml", str(OUT / "pytest_junit.xml")],
                         cwd=config.ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (OUT / "pytest_output.txt").write_text(run.stdout + run.stderr, encoding="utf-8")
    print("4/4 latency (copied from the last bench run) …")
    for name in ("latency.md", "latency.json"):
        if (config.EVAL_DIR / name).exists():
            shutil.copy(config.EVAL_DIR / name, OUT / name)
    summary = json.loads(RESULTS_JSON.read_text())
    default = summary.get(f"hybrid+rerank ({config.EMBEDDER})", {})
    o = default.get("overall", {})
    last_line = (run.stdout.strip().splitlines() or ["?"])[-1]
    (OUT / "README.md").write_text(f"""# Test results

Generated {datetime.now():%Y-%m-%d %H:%M} by `python -m eval.report` (golden set: {len(load_queries())} queries,
collection `{config.DEFAULT_COLLECTION}`, golden files only).

**Default pipeline (hybrid + rerank):** recall@1 {o.get('recall@1', 0):.2f} · recall@3 {o.get('recall@3', 0):.2f} ·
recall@5 {o.get('recall@5', 0):.2f} · recall@10 {o.get('recall@10', 0):.2f} · MRR {o.get('mrr', 0):.2f} ·
off-topic → no result {default.get('no_answer_accuracy') or 0:.2f} · speaker accuracy {default.get('speaker_accuracy') or 0:.2f}

**pytest:** `{last_line}`

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
""", encoding="utf-8")
    print(f"done → {OUT}")
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
