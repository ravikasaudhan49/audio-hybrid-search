"""Golden query set helpers shared by the evaluator, the tests and the API."""
import json

from . import config

TOLERANCE_S = 2.0
_cache: tuple[float, list[dict]] | None = None  # (file mtime, queries)


def load_queries() -> list[dict]:
    """Labeled queries, re-read only when data/queries.json changes."""
    global _cache
    if not config.QUERIES_PATH.exists():
        return []
    mtime = config.QUERIES_PATH.stat().st_mtime
    if _cache is None or _cache[0] != mtime:
        _cache = (mtime, json.loads(config.QUERIES_PATH.read_text(encoding="utf-8"))["queries"])
    return _cache[1]


def is_hit(result, rel: dict) -> bool:
    """A retrieved segment matches a labeled span if it is in the same file and the
    time ranges overlap, with a small tolerance for chunk-boundary drift.
    Labels are in EPISODE time, the timeline users see in results."""
    return (result.file_id == rel["file"]
            and result.episode_start_s <= rel["end"] + TOLERANCE_S
            and result.episode_end_s >= rel["start"] - TOLERANCE_S)


def relevant_spans(query: str) -> list[dict] | None:
    """Labeled spans if `query` is one of the golden queries, else None."""
    q = query.strip().lower()
    return next((item["relevant"] for item in load_queries() if item["query"].strip().lower() == q), None)
