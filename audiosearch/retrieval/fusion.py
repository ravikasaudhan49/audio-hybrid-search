"""Pure post-retrieval algorithms (no I/O): weighted RRF, parent grouping, MMR, thresholds."""
import numpy as np

from ..models import Result

RRF_K = 60
WEIGHTS = {"keyword": 1.0, "vector": 1.0, "fuzzy": 0.5}


def rrf(ranked: dict[str, list[int]], weights: dict[str, float] = WEIGHTS) -> list[tuple[int, float, dict]]:
    """Weighted Reciprocal Rank Fusion: score = sum(w / (RRF_K + rank)) over rankers."""
    scores: dict[int, float] = {}
    sources: dict[int, dict] = {}
    for method, ids in ranked.items():
        for rank, sid in enumerate(ids, 1):
            scores[sid] = scores.get(sid, 0.0) + weights.get(method, 1.0) / (RRF_K + rank)
            sources.setdefault(sid, {})[method] = rank
    return sorted(((sid, sc, sources[sid]) for sid, sc in scores.items()),
                  key=lambda x: (-x[1], x[0]))


def group_by_parent(results: list[Result]) -> list[Result]:
    """Keep the best-ranked child of each parent (results arrive best-first); the kept
    result records how many sibling children also matched."""
    best: dict[str, Result] = {}
    for r in results:
        if r.parent_id in best:
            best[r.parent_id].grouped += 1
        else:
            best[r.parent_id] = r
    return list(best.values())


def mmr_select(relevance: np.ndarray, vectors: np.ndarray, k: int, lam: float = 0.6) -> list[int]:
    """Maximal Marginal Relevance. Greedily picks the candidate maximizing
        lam * relevance(d) - (1 - lam) * max cosine(d, already picked)
    so near-duplicates (e.g. overlapping neighbour chunks) give way to new information.
    `relevance` is in [0, 1]; `vectors` are L2-normalized rows (zero rows = unknown)."""
    picked: list[int] = []
    remaining = list(range(len(relevance)))
    while remaining and len(picked) < k:
        if picked:
            redundancy = (vectors[remaining] @ vectors[picked].T).max(axis=1)
            scores = lam * relevance[remaining] - (1 - lam) * redundancy
        else:
            scores = relevance[remaining]
        best = remaining[int(np.argmax(scores))]
        picked.append(best)
        remaining.remove(best)
    return picked


def apply_thresholds(results: list[Result], min_score: float, min_relative: float) -> tuple[list[Result], float]:
    """Drop reranked results below an absolute score or a fraction of the query's best score,
    even if they made the top k (ADR-0004). Returns (kept, cut-off used)."""
    if not results:
        return results, min_score
    cutoff = max(min_score, min_relative * max(r.score for r in results))
    return [r for r in results if r.score >= cutoff], cutoff


def unit_range(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; all-equal input becomes all ones."""
    span = x.max() - x.min() if len(x) else 0
    return (x - x.min()) / span if span > 0 else np.ones_like(x)
