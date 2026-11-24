"""End-to-end retrieval quality gate against the golden query set.

Needs Postgres running with the golden dataset ingested; skipped otherwise.
Thresholds are the success criteria stated in the README.

Query embeddings and rerank scores are cached (.cache/), so after one `python -m eval.evaluate
--rerank` run this gate costs no API calls; on a fresh machine it spends ~1 Gemini call per
query and ~1 Cohere call per query (hybrid+rerank).
"""
import psycopg
import pytest

from audiosearch.container import AppContainer
from audiosearch.core import runtime
from audiosearch.db import FileRepository, collection_conn
from audiosearch.evaluation import evaluate, load_queries

THRESHOLDS = {"recall@5": 0.90, "recall@10": 0.95}
MIN_NO_ANSWER = 0.75   # share of negative queries that must return "no confident match"
MAX_P95_MS = 1500  # includes a hosted query-embedding call when EMBEDDER=gemini


async def _evaluate_all() -> dict | str:
    async with AppContainer.open(min_size=1, max_size=6) as app:
        async with collection_conn(app.pool, app.settings.default_collection) as conn:
            if not await FileRepository(conn).count_by_origin("golden"):
                return "no golden files ingested; run `python -m audiosearch ingest`"
        s = app.searcher(origin="golden")
        out = {m: await evaluate(s, load_queries(), m) for m in ("keyword", "vector", "hybrid")}  # no rerank
        out["hybrid+rerank"] = await evaluate(s, load_queries(), "hybrid", rerank=True)
        return out


@pytest.fixture(scope="module")
def reports():
    if not load_queries():
        pytest.skip("no labeled queries yet (data/queries.json)")
    try:
        out = runtime.run(_evaluate_all())
    except (OSError, psycopg.OperationalError) as e:  # DB not reachable
        pytest.skip(f"database unavailable: {e}")
    if isinstance(out, str):
        pytest.skip(out)
    return out


@pytest.mark.parametrize("metric,target", THRESHOLDS.items())
def test_hybrid_meets_recall_targets(reports, metric, target):
    got = reports["hybrid"]["overall"][metric]
    assert got >= target, f"hybrid {metric}={got:.2f} < {target}; misses: {reports['hybrid']['misses']}"


@pytest.mark.parametrize("metric", ["recall@5", "recall@10", "mrr"])
def test_hybrid_not_worse_than_either_single_method(reports, metric):
    h = reports["hybrid"]["overall"][metric]
    best_single = max(reports["keyword"]["overall"][metric], reports["vector"]["overall"][metric])
    assert h >= best_single - 0.02, f"hybrid {metric}={h:.2f} vs best single {best_single:.2f}"


def test_every_query_type_is_served(reports):
    weak = {t: r["recall@10"] for t, r in reports["hybrid"]["by_type"].items() if r["recall@10"] < 0.8}
    assert not weak, f"query types with recall@10 < 0.8: {weak}"


def test_latency(reports):
    assert reports["hybrid"]["latency_ms"]["p95"] <= MAX_P95_MS


@pytest.mark.parametrize("metric,target", THRESHOLDS.items())
def test_full_pipeline_meets_recall_targets(reports, metric, target):
    got = reports["hybrid+rerank"]["overall"][metric]
    assert got >= target, f"hybrid+rerank {metric}={got:.2f} < {target}; misses: {reports['hybrid+rerank']['misses']}"


def test_off_topic_queries_get_no_answer(reports):
    r = reports["hybrid+rerank"]
    assert r["no_answer_accuracy"] is not None and r["no_answer_accuracy"] >= MIN_NO_ANSWER, r["false_answers"]


def test_results_are_attributed_to_the_right_speaker(reports):
    assert reports["hybrid+rerank"]["speaker_accuracy"] >= 0.95
