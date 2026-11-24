"""CPU-heavy steps must not freeze the event loop (or hold a DB connection while they run).

Each test makes one real step slower (the real function plus a 0.3 s sleep) and runs a
heartbeat beside it that ticks every 10 ms. Run on the event loop, the slow step lets the
heartbeat tick about once; run in a worker thread, about 30 times. No paid API calls.
"""
import asyncio
import dataclasses
import time

import psycopg
import pytest

from audiosearch.container import AppContainer
from audiosearch.core import runtime
from audiosearch.core.settings import get_settings
from audiosearch.db import Database, schema_name
from audiosearch.ingestion import IngestionService
from audiosearch.ingestion import pipeline as ingestion_pipeline
from audiosearch.ingestion.chunking import chunk_transcript, speaker_turns
from audiosearch.ingestion.transcription import Transcriber, TranscriptStore
from backend.routers import files as files_router

SLOW_S = 0.3
MIN_TICKS = 10           # a blocked loop gets ~1 tick in 0.3 s; a free one ~30
COLLECTION = "test_nonblocking"


def slowed(fn):
    """The real function, plus a blocking sleep: stands in for a long episode."""
    def wrapper(*args, **kwargs):
        time.sleep(SLOW_S)
        return fn(*args, **kwargs)
    return wrapper


async def heartbeat_during(coro) -> tuple[object, int]:
    """Run `coro` while a 10 ms heartbeat ticks; return (result, ticks)."""
    ticks, done = 0, False

    async def beat():
        nonlocal ticks
        while not done:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(beat())
    try:
        result = await coro
    finally:
        done = True
        await task
    return result, ticks


def transcript() -> dict:
    words, t = [], 0.0
    for spk, line in [(0, "So how do you combine keyword search and vector search in one system?"),
                      (1, "We use reciprocal rank fusion and it needs no score calibration at all.")]:
        for w in line.split():
            words.append({"text": w, "start": t, "end": t + 0.35, "speaker": spk})
            t += 0.4
    return {"provider": "test", "duration": t, "speakers": {}, "words": words}


# ---------- no database ----------

class InstantTranscriber(Transcriber):
    name = "instant-test"

    async def request(self, file_id, audio_path, keyterms):
        return transcript()

    @staticmethod
    def normalize(raw):
        return raw


class NoAudio:
    async def normalize(self, src, out, start=None, duration=None):
        return None


def test_upload_chunk_preview_does_not_block_the_event_loop(tmp_path, monkeypatch):
    # Break caught: describe_transcript (turns + chunk plan, ~250 ms per hour of audio) run on the loop.
    ingestion = IngestionService(pool=None, embedder=None, transcriber=InstantTranscriber("unused"),
                                 transcripts=TranscriptStore(tmp_path), audio=NoAudio())
    monkeypatch.setattr(IngestionService, "describe_transcript",
                        staticmethod(slowed(IngestionService.describe_transcript)))
    (_, plan), ticks = runtime.run(heartbeat_during(ingestion.transcribe_upload("up_1", tmp_path / "a.mp3", [])))
    assert plan["children"]                       # the real preview was still produced
    assert ticks >= MIN_TICKS


# ---------- with the database ----------

@pytest.fixture(scope="module")
def db_available():
    try:
        runtime.run(Database().init_schema())
    except (OSError, psycopg.OperationalError) as e:
        pytest.skip(f"database unavailable: {e}")


async def _with_app(tmp_path, body):
    """A started container with ONE pooled connection and a scratch collection."""
    app = AppContainer(dataclasses.replace(get_settings(), transcript_dir=tmp_path))
    await app.start(min_size=1, max_size=1)
    await app.collections.create(COLLECTION)
    try:
        return await body(app)
    finally:
        async with app.pool.connection() as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema_name(COLLECTION)} CASCADE")
            await conn.execute("DELETE FROM public.collections WHERE name=%s", (COLLECTION,))
        await app.close()


def test_save_file_chunks_without_blocking_or_holding_the_connection(db_available, tmp_path, monkeypatch):
    # Break caught: chunking inside the transaction (holds the only connection; CLAUDE.md rule 4)
    # or on the event loop (freezes every other request).
    monkeypatch.setattr(ingestion_pipeline, "chunk_transcript", slowed(chunk_transcript))

    async def body(app):
        ingestion = app.ingestion(COLLECTION)
        save = asyncio.create_task(ingestion.save_file("f1", "t", "f1.mp3", transcript(), {0: "Maya", 1: "Omar"},
                                                       origin="upload"))
        await asyncio.sleep(0.05)                 # save_file is now chunking
        t0 = time.perf_counter()
        async with app.pool.connection():         # the pool's only connection must be free
            waited = time.perf_counter() - t0
        n = await save
        return n, waited

    (n, waited), ticks = runtime.run(heartbeat_during(_with_app(tmp_path, body)))
    assert n >= 1                                 # chunks were still stored
    assert waited < SLOW_S / 2, f"waited {waited:.2f}s for a connection held during chunking"
    assert ticks >= MIN_TICKS


def test_file_transcript_view_does_not_block_the_event_loop(db_available, tmp_path, monkeypatch):
    # Break caught: speaker_turns (~80 ms per hour of audio) run on the loop for every GET /files/{id}.
    monkeypatch.setattr(files_router, "speaker_turns", slowed(speaker_turns))

    async def body(app):
        t = transcript()
        app.transcripts.save("f1", t)
        await app.ingestion(COLLECTION).save_file("f1", "t", "f1.mp3", t, {0: "Maya", 1: "Omar"}, origin="upload")
        return await heartbeat_during(files_router.get_file(app, COLLECTION, "f1"))

    detail, ticks = runtime.run(_with_app(tmp_path, body))
    assert [turn.speaker_name for turn in detail.turns] == ["Maya", "Omar"]   # the real view was still built
    assert ticks >= MIN_TICKS
