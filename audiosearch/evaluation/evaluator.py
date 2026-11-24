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
cached scores), so their numbers always reflect real reranking. Golden files only
(origin=golden), as CLAUDE.md requires.
"""
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.log import get
from ..models import SearchRequest
from ..retrieval import SearchService
from .golden import is_hit, load_queries

if TYPE_CHECKING:
    from ..container import AppContainer

KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class EvalConfig:
    label: str
    method: str
    rerank: bool
    mmr: bool
    group: bool


def configurations(embedder: str, rerank: bool) -> list[EvalConfig]:
    """All configurations group by parent except the explicit 'children' row, which shows
    what parent dedup adds. `embedder` (e.g. "gemini") only labels the rows."""
    e = embedder
    cfgs = [EvalConfig("keyword (FTS)", "keyword", False, False, True),
            EvalConfig("fuzzy (trigram)", "fuzzy", False, False, True),
            EvalConfig(f"vector ({e})", "vector", False, False, True),
            EvalConfig(f"hybrid, children ungrouped ({e})", "hybrid", False, False, False),
            EvalConfig(f"hybrid ({e})", "hybrid", False, False, True)]
    if rerank:
        cfgs += [EvalConfig(f"hybrid+rerank ({e})", "hybrid", True, False, True),
                 EvalConfig(f"hybrid+rerank+mmr ({e})", "hybrid", True, True, True)]
    return cfgs


async def evaluate(searcher: SearchService, queries: list[dict], method: str,
                   rerank: bool = False, mmr: bool = False, group: bool = True) -> dict:
    """Metrics for one configuration over `queries`."""
    per_query = []
    latencies = []
    speaker_checks = []
    rerank_skipped = 0
    for q in queries:
        f = q.get("filters", {})
        t0 = time.perf_counter()
        out = await searcher.search(SearchRequest(q["query"], method, max(KS), f.get("speaker"), f.get("file"),
                                                  rerank=rerank, mmr=mmr, group=group, patient_rerank=True))
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


class Evaluator:
    """Runs every configuration on the golden files and writes eval/results.{json,md}."""

    def __init__(self, app: "AppContainer", collection: str | None = None):
        """app: the AppContainer, which supplies the searcher (embedder + reranker)."""
        self.app = app
        self.settings = app.settings
        self.collection = collection or self.settings.default_collection

    @property
    def results_json(self) -> Path:
        return self.settings.eval_dir / "results.json"

    @property
    def results_md(self) -> Path:
        return self.settings.eval_dir / "results.md"

    async def run(self, rerank: bool = False, log: Callable[[str], None] | None = None) -> dict:
        log = log or get("eval").info
        queries = load_queries()
        log(f"evaluating {len(queries)} queries: embedder={self.settings.embedder} rerank={rerank}")
        if not queries:
            raise RuntimeError(f"no labeled queries in {self.settings.queries_path}")
        report = {}
        searcher = self.app.searcher(self.collection, origin="golden")
        for cfg in configurations(self.settings.embedder, rerank):
            report[cfg.label] = await evaluate(searcher, queries, cfg.method, cfg.rerank, cfg.mmr, cfg.group)
            o = report[cfg.label]["overall"]
            log(f"{cfg.label:28s} R@5={o['recall@5']:.2f} MRR={o['mrr']:.2f}")
        self.results_json.write_text(json.dumps(report, indent=1))
        self.results_md.write_text(to_markdown(report) + "\n", encoding="utf-8")
        log(f"wrote {self.results_json.name} and {self.results_md.name}")
        return report
