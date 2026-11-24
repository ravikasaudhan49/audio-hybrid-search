"""The golden query set (data/queries.json), shared by the evaluator, the tests and the API."""
import json
from functools import lru_cache
from pathlib import Path

from ..core.settings import get_settings

TOLERANCE_S = 2.0


def is_hit(result, rel: dict) -> bool:
    """A retrieved segment matches a labeled span if it is in the same file and the
    time ranges overlap, with a small tolerance for chunk-boundary drift.
    Labels are in EPISODE time, the timeline users see in results."""
    return (result.file_id == rel["file"]
            and result.episode_start_s <= rel["end"] + TOLERANCE_S
            and result.episode_end_s >= rel["start"] - TOLERANCE_S)


class GoldenSet:
    """Labeled queries, re-read only when the file changes."""

    def __init__(self, path: Path):
        self.path = path
        self._cache: tuple[float, list[dict]] | None = None   # (file mtime, queries)

    def queries(self) -> list[dict]:
        if not self.path.exists():
            return []
        mtime = self.path.stat().st_mtime
        if self._cache is None or self._cache[0] != mtime:
            self._cache = (mtime, json.loads(self.path.read_text(encoding="utf-8"))["queries"])
        return self._cache[1]

    def relevant_spans(self, query: str) -> list[dict] | None:
        """Labeled spans if `query` is one of the golden queries, else None."""
        q = query.strip().lower()
        return next((item["relevant"] for item in self.queries() if item["query"].strip().lower() == q), None)


@lru_cache(maxsize=1)
def golden_set() -> GoldenSet:
    """The configured golden set (Singleton: one mtime cache per process)."""
    return GoldenSet(get_settings().queries_path)


def load_queries() -> list[dict]:
    return golden_set().queries()
