"""End-to-end pipeline on the real Postgres with a FAKE local embedder and rerank off,
so it spends no API quota: chunk -> store (parent + child rows) -> embed -> search ->
parent grouping. Runs in its own throwaway collection (dropped afterwards), which also
checks that collections are isolated. Skipped when the database is unreachable."""
import hashlib

import numpy as np
import pytest

from audiosearch import aio, config, db, embed
from audiosearch.ingest import embed_missing, save_file
from audiosearch.search import Searcher

FILE_ID = "test_pipeline"
COLLECTION = "test_pipeline"


class FakeEmbedder(embed.Embedder):
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
    config.CACHE_DIR = cache_dir          # keep fake query vectors out of the real cache
    embed._embedders["fake"] = FakeEmbedder()
    pool = await db.open_pool(min_size=1, max_size=4)
    await db.create_collection(pool, COLLECTION, "pytest scratch collection")
    try:
        n = await save_file(pool, FILE_ID, "pipeline test", f"{FILE_ID}.mp3", transcript(),
                            {0: "Maya Lin", 1: "Omar Diaz"}, origin="upload",  # labels renumber by first appearance
                            audio_url="https://media.example.com/podcasts/test.mp3", offset_s=1290.0,
                            roles={0: "host", 1: "guest"}, collection=COLLECTION)
        await embed_missing(pool, "fake", collection=COLLECTION)
        async with db.collection_conn(pool, COLLECTION) as conn:
            cur = await conn.execute(
                "SELECT count(DISTINCT parent_id), min(child_tokens), max(child_tokens), max(parent_tokens) "
                "FROM segments WHERE file_id=%s", (FILE_ID,))
            parents, min_ct, max_ct, max_pt = await cur.fetchone()
            cur = await conn.execute(
                "SELECT vector_dims(e.embedding) FROM embeddings e JOIN segments s ON s.id=e.segment_id "
                "WHERE s.file_id=%s LIMIT 1", (FILE_ID,))
            dims = (await cur.fetchone())[0]
        s = Searcher(pool, "fake", collection=COLLECTION)
        # isolation: the same search in the golden collection never sees the scratch file
        other = await Searcher(pool, "fake").search("reciprocal rank fusion", k=50, rerank=False)
        leaked = [r for r in other.results if r.file_id == FILE_ID]
        grouped = await s.search("reciprocal rank fusion", k=5, file_id=FILE_ID, rerank=False, group=True)
        ungrouped = await s.search("reciprocal rank fusion", k=5, file_id=FILE_ID, rerank=False, group=False)
        bakery = await s.search("sourdough bakery", k=1, file_id=FILE_ID, rerank=False)
        by_role = await s.search("what did the guest say about rank fusion", k=5, file_id=FILE_ID, rerank=False)
        by_name = await s.search("what did Omar say about sourdough", k=5, file_id=FILE_ID, rerank=False)
        return n, parents, min_ct, max_ct, max_pt, dims, grouped, ungrouped, bakery, by_role, by_name, leaked
    finally:
        async with pool.connection() as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {db.schema_name(COLLECTION)} CASCADE")
            await conn.execute("DELETE FROM public.collections WHERE name=%s", (COLLECTION,))
        await pool.close()
        embed._embedders.pop("fake", None)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    try:
        aio.run(db.init_schema())
    except OSError as e:
        pytest.skip(f"database unavailable: {e}")
    real_cache = config.CACHE_DIR
    try:
        return aio.run(_run(tmp_path_factory.mktemp("cache")))
    finally:
        config.CACHE_DIR = real_cache


def test_rows_have_child_and_parent_and_1536_dim_vectors(run):
    n, parents, _, max_ct, max_pt, dims, *_ = run
    assert n >= 3 and parents >= 1
    assert max_ct <= config.CHILD_TOKENS + 20 and max_pt <= config.PARENT_TOKENS + 20
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
