"""Latency benchmark: the golden queries against the running API at a fixed concurrency.

  python -m eval.bench [--concurrency 5] [--rounds 10] [--warmup 1] [--answer] [--url http://127.0.0.1:8000]

Every query is sent `rounds` times (after `warmup` unrecorded passes) with at most
`concurrency` requests in flight. Reported per query and overall: client-side end-to-end
latency p50 / p90 / p99, throughput, and the server's per-stage timings (understand, embed,
retrieve, rerank, post, answer). Writes eval/latency.md and eval/latency.json.

Query embeddings and rerank scores are cached after the evaluation run, so this measures the
system's own latency (HTTP + query understanding + Postgres + fusion + thresholds), not the
external APIs; the "cold" external-call costs are listed separately from the logs. Uses the
golden files only (origin=golden) so cached reranks match the evaluation's candidate sets.
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict

import httpx
import numpy as np

from audiosearch import config
from audiosearch.golden import load_queries

STAGES = ("understand", "embed", "retrieve", "rerank", "post", "answer", "total")
OUT_JSON = config.EVAL_DIR / "latency.json"
OUT_MD = config.EVAL_DIR / "latency.md"


def pct(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    return {"p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean()), "max": float(a.max()), "n": len(a)}


async def run(url: str, concurrency: int, rounds: int, warmup: int, answer: bool, collection: str) -> dict:
    queries = load_queries()
    sem = asyncio.Semaphore(concurrency)
    samples: dict[str, list[dict]] = defaultdict(list)
    errors: list[str] = []

    async with httpx.AsyncClient(base_url=url, timeout=120) as http:
        async def one(q: dict, record: bool):
            params = {"q": q["query"], "k": 10, "collection": collection, "origin": "golden",
                      "answer": str(answer).lower()}
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await http.get("/search", params=params)
                except httpx.HTTPError as e:
                    errors.append(f"{q['id']}: {type(e).__name__}")
                    return
                ms = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                errors.append(f"{q['id']}: HTTP {r.status_code}")
                return
            if record:
                body = r.json()
                samples[q["id"]].append({"client_ms": ms, "server": body["timings_ms"],
                                         "reranked": body["reranked"], "results": len(body["results"])})

        for _ in range(warmup):
            await asyncio.gather(*(one(q, False) for q in queries))
        t_start = time.perf_counter()
        for _ in range(rounds):
            await asyncio.gather(*(one(q, True) for q in queries))
        wall = time.perf_counter() - t_start

    all_client = [s["client_ms"] for ss in samples.values() for s in ss]
    stage = {st: pct(v) for st in STAGES
             if (v := [s["server"][st] for ss in samples.values() for s in ss if st in s["server"]])}
    by_query = {qid: {**pct([s["client_ms"] for s in ss]),
                      "server_total_p50": statistics.median(s["server"]["total"] for s in ss),
                      "reranked": all(s["reranked"] for s in ss), "results": ss[0]["results"]}
                for qid, ss in samples.items()}
    return {"config": {"url": url, "concurrency": concurrency, "rounds": rounds, "warmup": warmup,
                       "answer": answer, "queries": len(queries), "collection": collection},
            "overall": {**pct(all_client), "throughput_rps": len(all_client) / wall, "errors": errors},
            "stages": stage, "by_query": by_query,
            "query_text": {q["id"]: q["query"] for q in queries}}


def to_markdown(rep: dict) -> str:
    c, o = rep["config"], rep["overall"]
    lines = [f"Latency: {c['queries']} golden queries × {c['rounds']} rounds, concurrency {c['concurrency']}, "
             f"summary answer {'on' if c['answer'] else 'off'} (warm caches).", "",
             "| end-to-end (client) | p50 | p90 | p99 | mean | max | throughput | errors |",
             "|---|---|---|---|---|---|---|---|",
             f"| all requests (n={o['n']}) | {o['p50']:.0f} ms | {o['p90']:.0f} ms | {o['p99']:.0f} ms | "
             f"{o['mean']:.0f} ms | {o['max']:.0f} ms | {o['throughput_rps']:.1f} req/s | {len(o['errors'])} |",
             "", "| server stage | p50 | p90 | p99 |", "|---|---|---|---|"]
    for st, v in rep["stages"].items():
        lines.append(f"| {st} | {v['p50']:.1f} ms | {v['p90']:.1f} ms | {v['p99']:.1f} ms |")
    lines += ["", "| query | p50 | p90 | p99 | reranked | results |", "|---|---|---|---|---|---|"]
    for qid, v in sorted(rep["by_query"].items()):
        text = rep["query_text"][qid].replace("|", "\\|")[:60]
        lines.append(f"| {qid} {text} | {v['p50']:.0f} | {v['p90']:.0f} | {v['p99']:.0f} | "
                     f"{'yes' if v['reranked'] else 'no'} | {v['results']} |")
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--answer", action="store_true", help="include the Gemini summary answer (API calls)")
    p.add_argument("--collection", default=config.DEFAULT_COLLECTION)
    a = p.parse_args()
    rep = asyncio.run(run(a.url, a.concurrency, a.rounds, a.warmup, a.answer, a.collection))
    OUT_JSON.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    md = to_markdown(rep)
    OUT_MD.write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
