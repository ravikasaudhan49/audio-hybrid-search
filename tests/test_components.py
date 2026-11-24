"""Settings, disk cache, job manager and helpers (no API, no DB)."""
import asyncio
import dataclasses

import pytest

from audiosearch.core import runtime
from audiosearch.core.cache import JsonDiskCache
from audiosearch.core.settings import get_settings
from audiosearch.core.text import snippet, split_terms, to_seconds, upload_file_id
from audiosearch.ingestion.chunking import host_roles
from backend.jobs import JobManager


def test_settings_is_a_singleton_and_immutable():
    s = get_settings()
    assert get_settings() is s
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.embedder = "other"


def test_json_disk_cache_roundtrip_and_stable_keys(tmp_path):
    cache = JsonDiskCache(tmp_path / "x")
    key = JsonDiskCache.key_of("model", "query", ["a", "b"])
    assert key == JsonDiskCache.key_of("model", "query", ["a", "b"]) != JsonDiskCache.key_of("model", "q", [])
    assert cache.get(key) is None
    cache.put(key, [0.5, 1])
    assert cache.get(key) == [0.5, 1]


def test_text_helpers():
    assert to_seconds("00:17:30") == to_seconds("17:30") == to_seconds(1050) == 1050.0
    assert split_terms(" a, b,,c ") == ["a", "b", "c"]
    assert upload_file_id(b"x") == upload_file_id(b"x") != upload_file_id(b"y")
    assert snippet("a b [[c]] d", width=40) == "a b [[c]] d"
    assert host_roles({0: "A", 1: "B"}, 1) == {0: "guest", 1: "host"} and host_roles({0: "A"}, None) is None


def test_job_manager_runs_reports_and_evicts_finished_jobs():
    async def go():
        jobs = JobManager(concurrency=1, keep_finished=2)

        async def work(h):
            h.stage("step", 0.5)
            h.log("note")
            return {"ok": True}

        async def boom(h):
            raise ValueError("bad")

        started = [jobs.start("t", work) for _ in range(3)] + [jobs.start("t", boom)]
        while any(jobs.get(j.id) and jobs.get(j.id).status == "running" for j in started):
            await asyncio.sleep(0.01)
        return [jobs.get(j.id) for j in started]

    first, second, third, failed = runtime.run(go())
    assert first is None and second is None                 # evicted: only the 2 newest finished are kept
    assert third.status == "done" and third.result == {"ok": True} and "note" in third.log
    assert failed.status == "error" and "ValueError: bad" in failed.error
