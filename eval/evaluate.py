"""Recall@k / MRR evaluation over the labeled query set.

queries.json:
  {"queries": [{"id": "q01", "query": "...", "type": "keyword|phrase|entity|semantic|speaker|typo",
                "filters": {"speaker": "..."},                       # optional
                "relevant": [{"file": "ep01", "start": 1413.0, "end": 1430.0, "speaker": "Name"}],
                "match": "any"}]}                                   # optional: any one passage suffices
  (start/end in seconds of the ORIGINAL EPISODE, i.e. the times the UI shows)

A retrieved segment counts as a hit for a relevant span if it is in the same file and
their time ranges overlap (with a small tolerance for boundary drift).
Queries run one at a time so the reported latency is single-user latency, not
latency under contention. Reranked configurations wait out API rate limits (and reuse
cached scores), so their numbers always reflect real reranking.

Run:  python -m eval.evaluate [--embedders gemini bge-base] [--rerank]
"""
import argparse
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Callable

from psycopg_pool import AsyncConnectionPool

from audiosearch import aio, config, db
from audiosearch import log as logs
from audiosearch.golden import is_hit, load_queries
from audiosearch.search import Searcher

KS = (1, 3, 5, 10)
RESULTS_JSON = config.EVAL_DIR / "results.json"
RESULTS_MD = config.EVAL_DIR / "results.md"


async def evaluate(searcher: Searcher, queries: list[dict], method: str,
                   rerank: bool = False, mmr: bool = False, group: bool = True) -> dict:
    per_query = []
    latencies = []
    speaker_checks = []
    rerank_skipped = 0
    for q in queries:
        f = q.get("filters", {})
        t0 = time.perf_counter()
        out = await searcher.search(q["query"], method, max(KS), f.get("speaker"), f.get("file"),
                                    rerank=rerank, mmr=mmr, group=group, patient_rerank=True)
        latencies.append((time.perf_counter() - t0) * 1000)
        results = out.results
        rerank_skipped += rerank and not out.reranked

        rels = q["relevant"]
        if not rels:
            # Negative query: correct only if nothing is returned ("no confident match").
            per_query.append({"id": q["id"], "type": q.get("type", "negative"), "negative": True,
                              "no_answer": not results, "returned": len(results)})
            continue
        first_rank = next((i for i, r in enumerate(results, 1) if any(is_hit(r, x) for x in rels)), None)
        row = {"id": q["id"], "type": q.get("type", "other"),
               "rr": 1 / first_rank if first_rank else 0.0, "first_rank": first_rank}
        for k in KS:
            found = sum(any(is_hit(r, x) for r in results[:k]) for x in rels)
            # "any": several acceptable passages (e.g. every place a quoted phrase is said).
            row[f"recall@{k}"] = min(found, 1) if q.get("match") == "any" else found / len(rels)
        per_query.append(row)
        # Speaker attribution: did the hit segment carry the labeled speaker?
        for x in rels:
            if "speaker" in x:
                hit = next((r for r in results if is_hit(r, x)), None)
                if hit:
                    speaker_checks.append(hit.speaker == x["speaker"])

    positives = [r for r in per_query if not r.get("negative")]
    negatives = [r for r in per_query if r.get("negative")]

    def agg(rows):
        return {**{f"recall@{k}": statistics.mean(r[f"recall@{k}"] for r in rows) for k in KS},
                "mrr": statistics.mean(r["rr"] for r in rows), "n": len(rows)}

    by_type = defaultdict(list)
    for r in positives:
        by_type[r["type"]].append(r)
    lat = sorted(latencies)
    return {
        "overall": agg(positives),
        # Share of negative queries answered with "no confident match" (needs rerank + threshold).
        "no_answer_accuracy": (sum(r["no_answer"] for r in negatives) / len(negatives)) if negatives else None,
        "by_type": {t: agg(rows) for t, rows in sorted(by_type.items())},
        "latency_ms": {"p50": lat[len(lat) // 2], "p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))]},
        "speaker_accuracy": (sum(speaker_checks) / len(speaker_checks)) if speaker_checks else None,
        "misses": [r["id"] for r in positives if r["recall@10"] < 1],
        "false_answers": [r["id"] for r in negatives if not r["no_answer"]],
        "rerank_skipped": rerank_skipped,
        "per_query": per_query,
    }


def configurations(embedders: list[str], rerank: bool) -> list[tuple[str, str, str | None, bool, bool, bool]]:
    """(label, method, embedder, rerank, mmr, group). All group by parent except the explicit
    'children' row, which shows what parent dedup adds."""
    cfgs = [("keyword (FTS)", "keyword", None, False, False, True),
            ("fuzzy (trigram)", "fuzzy", None, False, False, True)]
    for e in embedders:
        cfgs.append((f"vector ({e})", "vector", e, False, False, True))
        cfgs.append((f"hybrid, children ungrouped ({e})", "hybrid", e, False, False, False))
        cfgs.append((f"hybrid ({e})", "hybrid", e, False, False, True))
        if rerank:
            cfgs.append((f"hybrid+rerank ({e})", "hybrid", e, True, False, True))
            cfgs.append((f"hybrid+rerank+mmr ({e})", "hybrid", e, True, True, True))
    return cfgs


def to_markdown(report: dict) -> str:
    lines = ["| configuration | " + " | ".join(f"R@{k}" for k in KS)
             + " | MRR | no-answer on negatives | speaker acc. | p50 ms | p95 ms |",
             "|---|" + "---|" * (len(KS) + 5)]
    for label, r in report.items():
        o = r["overall"]
        na = r.get("no_answer_accuracy")
        sp = r.get("speaker_accuracy")
        na = "-" if na is None else f"{na:.2f}"
        sp = "-" if sp is None else f"{sp:.2f}"
        lines.append(f"| {label} | " + " | ".join(f"{o[f'recall@{k}']:.2f}" for k in KS)
                     + f" | {o['mrr']:.2f} | {na} | {sp}"
                     + f" | {r['latency_ms']['p50']:.0f} | {r['latency_ms']['p95']:.0f} |")
    types = sorted({t for r in report.values() for t in r["by_type"]})
    lines += ["", "Recall@5 by query type:", "",
              "| configuration | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for label, r in report.items():
        lines.append(f"| {label} | " + " | ".join(
            f"{r['by_type'][t]['recall@5']:.2f}" if t in r["by_type"] else "-" for t in types) + " |")
    return "\n".join(lines)


async def run_evaluation(pool: AsyncConnectionPool, embedders: list[str], rerank: bool = False,
                         log: Callable[[str], None] = logs.get("eval").info,
                         collection: str = config.DEFAULT_COLLECTION) -> dict:
    """Evaluate every configuration on the golden files and write eval/results.{json,md}."""
    queries = load_queries()
    log(f"evaluating {len(queries)} queries: embedders={embedders} rerank={rerank}")
    if not queries:
        raise RuntimeError(f"no labeled queries in {config.QUERIES_PATH}")
    report = {}
    for label, method, emb, rr, mmr, group in configurations(embedders, rerank):
        report[label] = await evaluate(Searcher(pool, emb, origin="golden", collection=collection),
                                       queries, method, rr, mmr, group)
        o = report[label]["overall"]
        log(f"{label:28s} R@5={o['recall@5']:.2f} MRR={o['mrr']:.2f}")
    RESULTS_JSON.write_text(json.dumps(report, indent=1))
    RESULTS_MD.write_text(to_markdown(report) + "\n", encoding="utf-8")
    log(f"wrote {RESULTS_JSON.name} and {RESULTS_MD.name}")
    return report


async def amain(embedders: list[str], rerank: bool) -> None:
    pool = await db.open_pool()
    try:
        report = await run_evaluation(pool, embedders, rerank)
        print("\n" + to_markdown(report))
    finally:
        await pool.close()
        await aio.close_http()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedders", nargs="*", default=[config.EMBEDDER])
    p.add_argument("--rerank", action="store_true")
    a = p.parse_args()
    logs.setup("eval")
    aio.run(amain(a.embedders, a.rerank))


if __name__ == "__main__":
    main()
