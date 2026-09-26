import numpy as np

from audiosearch import aio, rerank, search
from audiosearch.search import fmt_ts, locate, mmr_select, query_terms, rrf


def test_rrf_rewards_agreement_between_methods():
    fused = rrf({"keyword": [1, 2, 3], "vector": [3, 4, 5]}, weights={"keyword": 1, "vector": 1})
    assert fused[0][0] == 3                       # in both lists beats rank-1 in only one
    assert fused[0][2] == {"keyword": 3, "vector": 1}


def test_rrf_weights():
    fused = rrf({"keyword": [1], "fuzzy": [2]}, weights={"keyword": 1.0, "fuzzy": 0.5})
    assert [sid for sid, _, _ in fused] == [1, 2]


def test_query_terms_drop_stopwords_and_punctuation():
    assert query_terms("What did they say about Vector-Databases?") == ["vectordatabases"]
    assert query_terms("the pricing of GPUs") == ["pricing", "gpus"]


def test_locate_finds_first_matching_word_time_and_highlights():
    words = [["We", 10.0, 10.2], ["trained", 10.2, 10.6], ["embeddings", 10.6, 11.2], ["today.", 11.2, 11.5]]
    hit, text = locate(words, ["embedding"], default_s=10.0)
    assert hit == 10.6
    assert text == "We trained [[embeddings]] today."


def test_locate_falls_back_to_segment_start():
    hit, text = locate([["hello", 5.0, 5.3]], ["zebra"], default_s=4.0)
    assert hit == 4.0 and "[[" not in text


def test_fmt_ts():
    assert fmt_ts(0) == "00:00" and fmt_ts(125.9) == "02:05"


def test_mmr_drops_near_duplicates():
    # 0 and 1 are near-identical (overlapping neighbour chunks); 2 is different but a bit less relevant.
    vecs = np.array([[1, 0, 0], [0.99, 0.141, 0], [0, 1, 0]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    rel = np.array([1.0, 0.95, 0.8], dtype=np.float32)
    assert mmr_select(rel, vecs, k=2, lam=1.0) == [0, 1]   # pure relevance keeps the duplicate
    assert mmr_select(rel, vecs, k=2, lam=0.7) == [0, 2]   # MMR swaps it for new information


def test_mmr_handles_missing_vectors_and_small_k():
    rel = np.array([0.2, 0.9, 0.5], dtype=np.float32)
    assert mmr_select(rel, np.zeros((3, 4), np.float32), k=5) == [1, 2, 0]


def test_rerank_falls_back_to_fused_order_when_unavailable(monkeypatch):
    class Down(rerank.Reranker):
        name = "down"

        async def scores(self, query, docs):
            raise rerank.RerankUnavailable("rate limit")

    monkeypatch.setattr(search, "get_reranker", lambda **kw: Down())
    results = [search.Result(i, "f", "t", "a", "s", 0, 1, 0, "x", "x", 1.0 - i / 10, {"keyword": i + 1},
                             "f:p00", 0, 1, "x")
               for i in range(3)]
    out = search.SearchOutput(list(results))
    aio.run(search.Searcher._rerank("q", out, patient=False))
    assert not out.reranked and [r.segment_id for r in out.results] == [0, 1, 2]
    assert "rate limit" in out.notes[0]
