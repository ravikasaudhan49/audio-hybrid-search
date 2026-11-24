"""On-disk JSON cache for hosted API results (query embeddings, rerank scores, answers).

CLAUDE.md: every hosted call is cached on disk, so repeated queries and evaluation runs cost
no API quota. All three caches share this one class; the key layouts are unchanged from
earlier versions so existing `.cache/` contents stay valid.
"""
import hashlib
import json
from pathlib import Path
from typing import Any


class JsonDiskCache:
    """One JSON file per key under `directory`."""

    def __init__(self, directory: Path):
        self.directory = directory

    @staticmethod
    def key_of(*parts: Any) -> str:
        """Stable key for a JSON-serialisable tuple of parts."""
        return hashlib.sha1(json.dumps(list(parts)).encode()).hexdigest()

    @staticmethod
    def key_of_text(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()

    def path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self.path(key)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def put(self, key: str, value: Any) -> None:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
