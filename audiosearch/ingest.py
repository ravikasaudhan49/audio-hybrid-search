"""manifest -> (trim audio) -> transcribe -> chunk -> Postgres -> embeddings.

Each stage is a separate coroutine so the API can run and report them one by one.
Files are ingested concurrently (bounded); DB writes for a file happen in one transaction.
"""
import asyncio
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from . import aio, config, db
from .embed import get_embedder
from .log import get, timed
from .segment import Child, Parent, chunk_transcript, speaker_names
from .transcribe import has_transcript, slice_transcript, to_seconds, transcribe

log = get(__name__)

EMBED_BATCH = 50  # segments per embed + insert step (progress granularity)


def load_manifest() -> list[dict]:
    return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))["files"]


# ---------- audio ----------

def _ffmpeg(src: Path, out: Path, start=None, duration=None) -> None:
    import imageio_ffmpeg
    out.parent.mkdir(parents=True, exist_ok=True)
    trim = (["-ss", str(start)] if start is not None else []) + \
           (["-t", str(duration)] if duration is not None else [])
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", *trim, "-i", str(src),
         "-ac", "1", "-ar", "16000", "-b:a", "64k", str(out)],
        check=True,
    )


async def normalize_audio(src: Path, out: Path, start=None, duration=None) -> None:
    """Convert (and optionally trim) any ffmpeg-readable audio to 16 kHz mono mp3.
    Runs in a thread: asyncio subprocesses are unavailable on Windows' selector loop."""
    with timed(log, "ffmpeg normalize", src=src.name, out=out.name, start=start, duration=duration):
        await asyncio.to_thread(_ffmpeg, src, out, start, duration)


async def prepare_audio(entry: dict, force: bool = False) -> None:
    """Cut the 8-10 min golden clip from a longer raw recording (data/raw/...).
    Skipped when the entry has no `clip`."""
    clip = entry.get("clip")
    out = config.AUDIO_DIR / entry["audio"]
    if not clip:
        return
    if out.exists() and not force:
        log.info("audio clip already cut, skipping file=%s", entry["id"])
        return
    await normalize_audio(config.DATA_DIR / clip["source"], out, clip["start"], clip["duration"])


# ---------- storage ----------

async def upsert_file(conn: AsyncConnection, fid: str, title: str, audio: str, duration,
                      source: str | None = None, origin: str = "golden",
                      audio_url: str | None = None, offset_s: float = 0.0) -> None:
    await conn.execute(
        """INSERT INTO files (id, title, audio_path, duration_s, source, origin, audio_url, source_offset_s)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (id) DO UPDATE SET title=EXCLUDED.title, audio_path=EXCLUDED.audio_path,
             duration_s=EXCLUDED.duration_s, source=EXCLUDED.source, origin=EXCLUDED.origin,
             audio_url=EXCLUDED.audio_url, source_offset_s=EXCLUDED.source_offset_s""",
        (fid, title, audio, duration, source, origin, audio_url, offset_s),
    )


async def set_speakers(conn: AsyncConnection, fid: str, names: dict,
                       roles: dict | None = None) -> dict[int, int]:
    """Upsert speaker names and roles (host | guest); returns {diarization label: speakers.id}."""
    ids = {}
    roles = {int(k): v for k, v in (roles or {}).items()}
    for label, name in names.items():
        cur = await conn.execute(
            """INSERT INTO speakers (file_id, label, name, role) VALUES (%s,%s,%s,%s)
               ON CONFLICT (file_id, label) DO UPDATE SET name=EXCLUDED.name, role=EXCLUDED.role RETURNING id""",
            (fid, int(label), name, roles.get(int(label))),
        )
        ids[int(label)] = (await cur.fetchone())[0]
    return ids


def parent_key(fid: str, index: int) -> str:
    return f"{fid}:p{index:02d}"


async def store_chunks(conn: AsyncConnection, fid: str, parents: list[Parent], children: list[Child],
                       speaker_ids: dict[int, int], names: dict[int, str],
                       audio_url: str | None = None, offset_s: float = 0.0) -> int:
    """One row per child, each carrying its parent's id and text, the episode audio_url and the
    clip's offset in the episode (episode_* times are derived from it by Postgres)."""
    # Replacing a file's segments also drops their embeddings (ON DELETE CASCADE).
    await conn.execute("DELETE FROM segments WHERE file_id=%s", (fid,))
    parent_text = {p.index: p.render(names) for p in parents}
    by_index = {p.index: p for p in parents}
    async with conn.cursor() as cur:
        await cur.executemany(
            """INSERT INTO segments (file_id, speaker_id, seq, start_s, end_s, child_text, child_tokens, words,
                                     parent_id, parent_start_s, parent_end_s, parent_text, parent_tokens,
                                     audio_url, source_offset_s)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(fid, speaker_ids[c.speaker], i, c.start, c.end, c.text, c.tokens,
              Jsonb([[w.text, round(w.start, 2), round(w.end, 2)] for w in c.words]),
              parent_key(fid, c.parent), by_index[c.parent].start, by_index[c.parent].end,
              parent_text[c.parent], by_index[c.parent].tokens, audio_url, offset_s)
             for i, c in enumerate(children)],
        )
    return len(children)


async def save_file(pool: AsyncConnectionPool, fid: str, title: str, audio: str, resp: dict,
                    speakers: dict, source: str | None = None, origin: str = "golden",
                    rechunk: bool = True, audio_url: str | None = None, offset_s: float = 0.0,
                    roles: dict | None = None, collection: str = config.DEFAULT_COLLECTION) -> int:
    """File row, speakers and chunks in one transaction, so readers never see a half-written file."""
    async with db.collection_conn(pool, collection) as conn, conn.transaction():
        await upsert_file(conn, fid, title, audio, resp.get("duration"), source, origin, audio_url, offset_s)
        speaker_ids = await set_speakers(conn, fid, speakers, roles)
        cur = await conn.execute("SELECT count(*) FROM segments WHERE file_id=%s", (fid,))
        existing = (await cur.fetchone())[0]
        if existing and not rechunk:
            # Keep chunks and embeddings; refresh the row metadata that may have changed.
            await conn.execute("UPDATE segments SET audio_url=%s, source_offset_s=%s WHERE file_id=%s",
                               (audio_url, offset_s, fid))
            log.info("file=%s already has %d segments; kept them, refreshed audio_url/offset=%.0fs "
                     "(use rechunk to rebuild)", fid, existing, offset_s)
            return existing
        parents, children = chunk_transcript(resp)
        names = {int(label): name for label, name in speakers.items()}
        n = await store_chunks(conn, fid, parents, children, speaker_ids, names, audio_url, offset_s)
        ctok = [c.tokens for c in children] or [0]
        ptok = [p.tokens for p in parents] or [0]
        log.info("stored file=%s origin=%s speakers=%s parents=%d (avg %d tok, max %d) "
                 "children=%d (avg %d tok, max %d, avg %.0fs)",
                 fid, origin, list(speakers.values()), len(parents), sum(ptok) / len(ptok), max(ptok),
                 n, sum(ctok) / len(ctok), max(ctok),
                 sum(c.end - c.start for c in children) / max(1, len(children)))
        return n


async def ingest_entry(pool: AsyncConnectionPool, entry: dict, rechunk: bool = False,
                       collection: str = config.DEFAULT_COLLECTION) -> int:
    log.info("ingest start file=%s title=%r", entry["id"], entry["title"])
    await prepare_audio(entry)
    clip = entry.get("clip") or {}
    if clip.get("transcript_from") and not has_transcript(entry["id"]):
        # The full recording is already transcribed: cut the clip's words out of it (no API call).
        slice_transcript(clip["transcript_from"], entry["id"], clip["start"], clip["duration"],
                         clip.get("source_speakers"))
    t = await transcribe(entry["id"], config.AUDIO_DIR / entry["audio"], entry.get("keyterms"))
    # Names identified by the transcriber, overridden by any the manifest pins.
    names = {k: v or f"Speaker {k + 1}" for k, v in speaker_names(t).items()}
    names.update({int(k): v for k, v in (entry.get("speakers") or {}).items()})
    # The manifest names the host; everyone else in the pair is the guest.
    roles = {k: ("host" if name == entry.get("host") else "guest") for k, name in names.items()} \
        if entry.get("host") else None
    log.info("speakers file=%s %s roles=%s", entry["id"], names, roles)
    # Times are stored clip-relative; the clip's start in the episode maps them to episode time.
    offset_s = to_seconds(clip["start"]) if clip else 0.0
    audio_url = entry.get("audio_url") or config.AUDIO_URL_BASE + Path(clip.get("source", entry["audio"])).name
    return await save_file(pool, entry["id"], entry["title"], entry["audio"], t, names,
                           entry.get("source_url"), "golden", rechunk, audio_url, offset_s, roles, collection)


# ---------- embeddings ----------

async def embed_missing(pool: AsyncConnectionPool, embedder_name: str | None = None,
                        progress: Callable[[int, int], None] | None = None,
                        collection: str = config.DEFAULT_COLLECTION) -> int:
    """Embed a collection's segments that have no vector for this model yet. Batches run
    concurrently (bounded by the embedder); `progress(done, total)` is called as batches land."""
    emb = await get_embedder(embedder_name)
    async with db.collection_conn(pool, collection) as conn:
        cur = await conn.execute(
            # Embedded text = "Speaker: child_text", so speaker-scoped questions match on meaning.
            """SELECT s.id, sp.name || ': ' || s.child_text FROM segments s JOIN speakers sp ON sp.id = s.speaker_id
               WHERE NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.segment_id=s.id AND e.model=%s)
               ORDER BY s.id""",
            (emb.name,),
        )
        rows = await cur.fetchall()
    log.info("embedding %d new segments in collection %r with %s (batches of %d, concurrency %d)", len(rows),
             collection, emb.name, EMBED_BATCH, config.EMBED_CONCURRENCY)

    total, done = len(rows), 0
    sem = asyncio.Semaphore(config.EMBED_CONCURRENCY)

    async def batch(part: list[tuple[int, str]]):
        nonlocal done
        async with sem:
            vecs = await emb.embed_documents([t for _, t in part])
        async with db.collection_conn(pool, collection) as conn, conn.cursor() as cur:
            await cur.executemany(
                """INSERT INTO embeddings (segment_id, model, embedding) VALUES (%s,%s,%s)
                   ON CONFLICT (segment_id, model) DO NOTHING""",
                [(sid, emb.name, v) for (sid, _), v in zip(part, vecs, strict=True)],
            )
        done += len(part)
        log.info("embedded %d/%d segments (%s)", done, total, emb.name)
        if progress:
            progress(done, total)

    await asyncio.gather(*(batch(rows[i:i + EMBED_BATCH]) for i in range(0, total, EMBED_BATCH)))
    async with db.collection_conn(pool, collection) as conn:
        await db.ensure_vector_index(conn, emb.name, emb.dim)
    return total


# ---------- CLI entry ----------

async def run(embedders: list[str] | None = None, rechunk: bool = False, only: str | None = None,
              collection: str | None = None) -> None:
    pool = await db.open_pool()
    try:
        # The manifest may name its collection; --collection overrides; else the default.
        collection = collection or json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8")).get(
            "collection") or config.DEFAULT_COLLECTION
        await db.create_collection(pool, collection)
        entries = [e for e in load_manifest() if not only or e["id"] == only]
        log.info("ingest run: collection=%r, %d manifest entries, concurrency %d, embedders=%s, rechunk=%s",
                 collection, len(entries), config.INGEST_CONCURRENCY, embedders or [config.EMBEDDER], rechunk)
        sem = asyncio.Semaphore(config.INGEST_CONCURRENCY)

        async def one(entry: dict):
            async with sem:
                n = await ingest_entry(pool, entry, rechunk, collection)
                print(f"{entry['id']}: {n} segments")

        await asyncio.gather(*(one(e) for e in entries))
        for name in embedders or [config.EMBEDDER]:
            print(f"{name}: embedded {await embed_missing(pool, name, collection=collection)} new segments")
    finally:
        await pool.close()
        await aio.close_http()
