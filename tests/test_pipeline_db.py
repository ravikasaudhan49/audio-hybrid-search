"""End-to-end pipeline on the real Postgres with a FAKE local embedder and rerank off,
so it spends no API quota: chunk -> store (parent + child rows) -> embed -> search ->
parent grouping. Runs in its own throwaway collection (dropped afterwards), which also
checks that collections are isolated. Skipped when the database is unreachable."""
import dataclasses
import hashlib

import numpy as np
import psycopg
import pytest

from audiosearch.container import AppContainer
from audiosearch.core import runtime
from audiosearch.core.settings import get_settings
from audiosearch.db import Database, collection_conn, schema_name
from audiosearch.embeddings import Embedder
from audiosearch.ingestion.chunking import CHUNKING
from audiosearch.models import SearchRequest

FILE_ID = "test_pipeline"
COLLECTION = "test_pipeline"


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words hashing into config-sized vectors: similar words -> similar vectors."""
    name = "fake-test"
    dim = 1536
    min_similarity = 0.0   # bag-of-words cosines are small; the Gemini floor doesn't apply

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        for w in text.lower().replace(".", " ").replace(":", " ").split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        return v / (np.linalg.norm(v) or 1)

    async def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    async def _embed_queries(self, texts):
        return np.stack([self._vec(t) for t in texts])


def transcript() -> dict:
    lines = [
        (1, "So how do you combine keyword search and vector search in one system?"),
        (0, "We use reciprocal rank fusion. Each method produces a ranked list and we add one over sixty "
            "plus the rank. Documents that both methods like float to the top of the list every time."),
        (1, "Right."),
        (0, "It is simple and it needs no score calibration between the two methods at all."),
        (1, "Let's switch topics. You started your career at a bakery in Lisbon?"),
        (0, "I did. I baked sourdough bread every morning at four and it taught me patience."),
    ]
    words, t = [], 0.0
    for spk, line in lines:
        for w in line.split():
            words.append({"text": w, "start": t, "end": t + 0.35, "speaker": spk})
            t += 0.4
    return {"provider": "test", "duration": t, "speakers": {}, "words": words}


async def _run(cache_dir):
    # A container on a copy of the settings with its own cache dir, and a fake embedder
    # installed in the factory: no monkeypatching, no API calls.
    fake = FakeEmbedder()
    app = AppContainer(dataclasses.replace(get_settings(), cache_dir=cache_dir), embedder=fake)
    await app.start(min_size=1, max_size=4)
    pool = app.pool
    await app.collections.create(COLLECTION, "pytest scratch collection")
    try:
        ingestion = app.ingestion(COLLECTION)
        n = await ingestion.save_file(FILE_ID, "pipeline test", f"{FILE_ID}.mp3", transcript(),
                                      {0: "Maya Lin", 1: "Omar Diaz"}, origin="upload",  # labels renumber
                                      audio_url="https://media.example.com/podcasts/test.mp3", offset_s=1290.0,
                                      roles={0: "host", 1: "guest"})
        await ingestion.embed_missing()
        async with collection_conn(pool, COLLECTION) as conn:
            cur = await conn.execute(
                "SELECT count(DISTINCT parent_id), min(child_tokens), max(child_tokens), max(parent_tokens) "
                "FROM segments WHERE file_id=%s", (FILE_ID,))
            parents, min_ct, max_ct, max_pt = await cur.fetchone()
            cur = await conn.execute(
                "SELECT vector_dims(e.embedding) FROM embeddings e JOIN segments s ON s.id=e.segment_id "
                "WHERE s.file_id=%s LIMIT 1", (FILE_ID,))
            dims = (await cur.fetchone())[0]
        s = app.searcher(COLLECTION)

        def q(text, **kw):
            return SearchRequest(text, rerank=False, **kw)

        # isolation: the same search in the golden collection never sees the scratch file
        other = await app.searcher().search(q("reciprocal rank fusion", k=50))
        leaked = [r for r in other.results if r.file_id == FILE_ID]
        grouped = await s.search(q("reciprocal rank fusion", k=5, file_id=FILE_ID, group=True))
        ungrouped = await s.search(q("reciprocal rank fusion", k=5, file_id=FILE_ID, group=False))
        bakery = await s.search(q("sourdough bakery", k=1, file_id=FILE_ID))
        by_role = await s.search(q("what did the guest say about rank fusion", k=5, file_id=FILE_ID))
        by_name = await s.search(q("what did Omar say about sourdough", k=5, file_id=FILE_ID))
        return n, parents, min_ct, max_ct, max_pt, dims, grouped, ungrouped, bakery, by_role, by_name, leaked
    finally:
        async with pool.connection() as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema_name(COLLECTION)} CASCADE")
            await conn.execute("DELETE FROM public.collections WHERE name=%s", (COLLECTION,))
        await app.close()


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    try:
        runtime.run(Database().init_schema())
    except (OSError, psycopg.OperationalError) as e:
        pytest.skip(f"database unavailable: {e}")
    return runtime.run(_run(tmp_path_factory.mktemp("cache")))


def test_rows_have_child_and_parent_and_1536_dim_vectors(run):
    n, parents, _, max_ct, max_pt, dims, *_ = run
    assert n >= 3 and parents >= 1
    assert max_ct <= CHUNKING.child_tokens + 20 and max_pt <= CHUNKING.parent_tokens + 20
    assert dims == 1536


def test_result_carries_speaker_timestamp_child_and_parent(run):
    *_, grouped, _, _, _, _, _ = run
    top = grouped.results[0]
    assert top.speaker == "Omar Diaz" and "reciprocal" in top.child_text.lower()
    assert top.parent_id.startswith(f"{FILE_ID}:p") and "Maya Lin: So how do you combine" in top.parent_text
    assert "[[" in top.highlighted and top.hit_s >= top.start_s


def test_rows_carry_audio_url_and_episode_timeline(run):
    *_, grouped, _, _, _, _, _ = run
    top = grouped.results[0]
    assert top.audio_url == "https://media.example.com/podcasts/test.mp3"
    assert top.source_offset_s == 1290.0
    assert top.episode_start_s == top.start_s + 1290.0 and top.episode_hit_s == top.hit_s + 1290.0
    assert top.timestamp.startswith("21:")          # shown in episode time, not clip time


def test_grouping_returns_one_result_per_parent(run):
    *_, grouped, ungrouped, _, _, _, _ = run
    assert len({r.parent_id for r in grouped.results}) == len(grouped.results)
    assert len(ungrouped.results) >= len(grouped.results)


def test_backchannel_is_not_a_result_but_semantic_topic_is_found(run):
    *_, _, ungrouped, bakery, _, _, _ = run
    assert all(r.child_text != "Right." for r in ungrouped.results)
    assert "sourdough" in bakery.results[0].child_text


def test_role_and_name_in_query_filter_to_that_speaker(run):
    *_, by_role, by_name, _ = run
    assert by_role.role_detected == "guest" and by_role.search_text == "rank fusion"
    assert by_role.results and {r.speaker for r in by_role.results} == {"Omar Diaz"}
    assert by_name.speaker_detected == "Omar Diaz" and {r.speaker for r in by_name.results} == {"Omar Diaz"}


def test_collections_are_isolated(run):
    *_, leaked = run
    assert leaked == []
