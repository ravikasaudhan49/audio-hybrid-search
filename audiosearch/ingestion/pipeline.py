"""Ingestion: audio -> transcript -> chunks -> Postgres -> embeddings.

Steps for one file (one method each below):
  1. audio       ffmpeg -> 16 kHz mono mp3 (golden clips are cut from the full episode)
  2. transcribe  provider called only if no cached transcript exists
  3. save_file   speakers (names + host/guest) and parent/child chunks, in one transaction
  4. embed       "Speaker: child_text" for every chunk without a vector, then the HNSW index

One service serves every entry point: the golden manifest (`python -m audiosearch ingest`),
a single file (`python -m audiosearch add`) and the API's two-stage upload (transcribe, then
index once the user has named the speakers).
"""
import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path

from psycopg_pool import AsyncConnectionPool

from ..core.log import get
from ..core.progress import NullReporter, Reporter
from ..core.settings import Settings, get_settings
from ..core.text import to_seconds
from ..db import (
    CollectionRepository,
    EmbeddingRepository,
    FileRepository,
    SegmentRepository,
    SpeakerRepository,
    collection_conn,
)
from ..embeddings import Embedder
from .audio import AudioNormalizer
from .chunking import (
    CHUNKING,
    chunk_transcript,
    display_names,
    host_roles,
    speaker_names,
    speaker_turns,
    words_from_transcript,
)
from .transcription import Transcriber, TranscriptStore

log = get(__name__)

EMBED_BATCH = 50  # segments per embed + insert step (progress granularity)
Progress = Callable[[int, int], None]


class Manifest:
    """data/manifest.json: the golden files, their clips, speakers and the target collection."""

    def __init__(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        self.collection: str | None = data.get("collection")
        self.entries: list[dict] = data["files"]

    def select(self, only: str | None = None) -> list[dict]:
        return [e for e in self.entries if not only or e["id"] == only]


class IngestionService:
    def __init__(self, pool: AsyncConnectionPool, *, embedder: Embedder, transcriber: Transcriber,
                 collection: str | None = None, settings: Settings | None = None,
                 transcripts: TranscriptStore | None = None, audio: AudioNormalizer | None = None):
        """embedder / transcriber: the app's, built once by AppContainer."""
        self.pool = pool
        self.embedder = embedder
        self.transcriber = transcriber
        self.settings = settings or get_settings()
        self.collection = collection or self.settings.default_collection
        self.transcripts = transcripts or TranscriptStore(self.settings.transcript_dir)
        self.audio = audio or AudioNormalizer()

    # ---------- transcription ----------

    async def transcribe(self, file_id: str, audio_path: Path, keyterms: list[str] | None = None,
                         force: bool = False) -> dict:
        """Normalized transcript; calls the provider only when nothing is cached (CLAUDE.md:
        never an uncached hosted call)."""
        if self.transcripts.has(file_id) and not force:
            log.info("transcript cache hit file=%s", file_id)
            return await asyncio.to_thread(self.transcripts.load, file_id)
        raw = await self.transcriber.request(file_id, audio_path, keyterms or [])
        await asyncio.to_thread(self.transcripts.save_raw, file_id, self.transcriber.name, raw)
        t = self.transcriber.normalize(raw)
        await asyncio.to_thread(self.transcripts.save, file_id, t)
        log.info("transcript file=%s provider=%s duration=%.1fs words=%d speakers=%s", file_id,
                 self.transcriber.name, t["duration"] or 0, len(t["words"]), t["speakers"])
        return t

    # ---------- storage ----------

    async def save_file(self, fid: str, title: str, audio: str, transcript: dict, speakers: dict, *,
                        source: str | None = None, origin: str = "golden", rechunk: bool = True,
                        audio_url: str | None = None, offset_s: float = 0.0, roles: dict | None = None) -> int:
        """File row, speakers and chunks in one transaction, so readers never see a half-written file."""
        # Chunking is CPU work (tokenizing every word): do it in a thread, before taking a connection,
        # so the transaction only holds the connection for the inserts (CLAUDE.md rule 4).
        parents, children = await asyncio.to_thread(chunk_transcript, transcript)
        async with collection_conn(self.pool, self.collection) as conn, conn.transaction():
            await FileRepository(conn).upsert(fid, title, audio, transcript.get("duration"), source, origin,
                                              audio_url, offset_s)
            speaker_ids = await SpeakerRepository(conn).upsert_many(fid, speakers, roles)
            segments = SegmentRepository(conn)
            existing = await segments.count_for_file(fid)
            if existing and not rechunk:
                # Keep chunks and embeddings; refresh the row metadata that may have changed.
                await segments.refresh_audio(fid, audio_url, offset_s)
                log.info("file=%s already has %d segments; kept them, refreshed audio_url/offset=%.0fs "
                         "(use rechunk to rebuild)", fid, existing, offset_s)
                return existing
            names ={int(label): name for label, name in speakers.items()}
            n = await segments.replace_for_file(fid, parents, children, speaker_ids, names, audio_url, offset_s)
        ctok = [c.tokens for c in children] or [0]
        ptok = [p.tokens for p in parents] or [0]
        log.info("stored file=%s origin=%s speakers=%s parents=%d (avg %d tok, max %d) "
                 "children=%d (avg %d tok, max %d, avg %.0fs)",
                 fid, origin, list(speakers.values()), len(parents), sum(ptok) / len(ptok), max(ptok),
                 n, sum(ctok) / len(ctok), max(ctok),
                 sum(c.end - c.start for c in children) / max(1, len(children)))
        return n

    async def embed_missing(self, progress: Progress | None = None) -> int:
        """Embed this collection's segments that have no vector yet. Batches run concurrently
        (bounded); `progress(done, total)` is called as batches land."""
        emb = self.embedder
        async with collection_conn(self.pool, self.collection) as conn:
            rows = await SegmentRepository(conn).texts_missing_embedding(emb.name)
        concurrency = self.settings.embed_concurrency
        log.info("embedding %d new segments in collection %r with %s (batches of %d, concurrency %d)", len(rows),
                 self.collection, emb.name, EMBED_BATCH, concurrency)

        total, done = len(rows), 0
        sem = asyncio.Semaphore(concurrency)

        async def batch(part: list[tuple[int, str]]):
            nonlocal done
            async with sem:
                vecs = await emb.embed_documents([t for _, t in part])
            async with collection_conn(self.pool, self.collection) as conn:
                await EmbeddingRepository(conn).insert_many(
                    emb.name, [(sid, v) for (sid, _), v in zip(part, vecs, strict=True)])
            done += len(part)
            log.info("embedded %d/%d segments (%s)", done, total, emb.name)
            if progress:
                progress(done, total)

        await asyncio.gather(*(batch(rows[i:i + EMBED_BATCH]) for i in range(0, total, EMBED_BATCH)))
        async with collection_conn(self.pool, self.collection) as conn:
            await EmbeddingRepository(conn).ensure_index(emb.name, emb.dim)
        return total

    # ---------- golden manifest ----------

    async def prepare_clip(self, entry: dict, force: bool = False) -> None:
        """Cut the 8-10 min golden clip from a longer raw recording (data/raw/...).
        Skipped when the entry has no `clip`."""
        clip = entry.get("clip")
        out = self.settings.audio_dir / entry["audio"]
        if not clip:
            return
        if out.exists() and not force:
            log.info("audio clip already cut, skipping file=%s", entry["id"])
            return
        await self.audio.normalize(self.settings.data_dir / clip["source"], out, clip["start"], clip["duration"])

    async def ingest_entry(self, entry: dict, rechunk: bool = False) -> int:
        log.info("ingest start file=%s title=%r", entry["id"], entry["title"])
        await self.prepare_clip(entry)
        clip = entry.get("clip") or {}
        if clip.get("transcript_from") and not self.transcripts.has(entry["id"]):
            # The full recording is already transcribed: cut the clip's words out of it (no API call).
            self.transcripts.slice(clip["transcript_from"], entry["id"], clip["start"], clip["duration"],
                                   clip.get("source_speakers"))
        t = await self.transcribe(entry["id"], self.settings.audio_dir / entry["audio"], entry.get("keyterms"))
        # Names identified by the transcriber, overridden by any the manifest pins.
        names = display_names(t)
        names.update({int(k): v for k, v in (entry.get("speakers") or {}).items()})
        # The manifest names the host; everyone else in the pair is the guest.
        roles = {k: ("host" if name == entry.get("host") else "guest") for k, name in names.items()} \
            if entry.get("host") else None
        log.info("speakers file=%s %s roles=%s", entry["id"], names, roles)
        # Times are stored clip-relative; the clip's start in the episode maps them to episode time.
        offset_s = to_seconds(clip["start"]) if clip else 0.0
        audio_url = entry.get("audio_url") or self.settings.audio_url_base + Path(
            clip.get("source", entry["audio"])).name
        return await self.save_file(entry["id"], entry["title"], entry["audio"], t, names,
                                    source=entry.get("source_url"), origin="golden", rechunk=rechunk,
                                    audio_url=audio_url, offset_s=offset_s, roles=roles)

    async def ingest_manifest(self, entries: list[dict], rechunk: bool = False,
                              on_file: Callable[[str, int], None] | None = None) -> None:
        """Ingest manifest entries concurrently (bounded by INGEST_CONCURRENCY)."""
        await CollectionRepository(self.pool).create(self.collection)
        sem = asyncio.Semaphore(self.settings.ingest_concurrency)

        async def one(entry: dict):
            async with sem:
                n = await self.ingest_entry(entry, rechunk)
                if on_file:
                    on_file(entry["id"], n)

        await asyncio.gather(*(one(e) for e in entries))

    # ---------- uploads ----------

    def upload_audio_path(self, fid: str) -> Path:
        return self.settings.audio_dir / f"{fid}.mp3"

    async def transcribe_upload(self, fid: str, src: Path, keyterms: list[str],
                                reporter: Reporter | None = None) -> tuple[dict, dict]:
        """Stage 1: normalize the audio, transcribe it with diarization (+ speaker names) and plan
        the chunks. Returns (transcript, describe_transcript(transcript))."""
        r = reporter or NullReporter()
        audio = self.upload_audio_path(fid)
        r.stage("Normalizing audio to 16 kHz mono mp3", 0.1)
        await self.audio.normalize(src, audio)
        r.stage(f"Transcribing + diarizing + identifying speakers ({self.transcriber.name})", 0.3)
        t0 = time.perf_counter()
        cached = self.transcripts.exists(fid)
        t = await self.transcribe(fid, audio, keyterms)
        r.log(f"Transcript {'loaded from cache' if cached else f'ready in {time.perf_counter() - t0:.1f}s'}"
              f" · audio {t.get('duration') or 0:.0f}s")
        r.stage("Chunking", 0.9)
        plan = await asyncio.to_thread(self.describe_transcript, t)   # CPU: turns + chunk plan
        r.log(f"{len(plan['parents'])} parents (~{CHUNKING.parent_tokens} tok), {len(plan['children'])} children "
              f"(~{CHUNKING.child_tokens} tok, {CHUNKING.child_overlap_tokens} overlap)")
        return t, plan

    async def index_upload(self, fid: str, title: str, names: dict[int, str], host: int | None,
                           audio_url: str, reporter: Reporter | None = None,
                           transcript: dict | None = None) -> tuple[int, int]:
        """Stage 2: store chunks with named speakers (one transaction), then embed and index them.
        Returns (segments stored, segments embedded)."""
        r = reporter or NullReporter()
        t = transcript or await asyncio.to_thread(self.transcripts.load, fid)
        r.stage(f"Storing file, speakers and chunks in collection '{self.collection}'", 0.05)
        n = await self.save_file(fid, title, self.upload_audio_path(fid).name, t, names, origin="upload",
                                 audio_url=audio_url, roles=host_roles(names, host))
        r.log(f"Stored {n} segments (full-text and trigram indexes update automatically)")
        r.stage(f"Embedding with {self.embedder.name}", 0.1)
        done = await self.embed_missing(lambda d, total: r.progress(0.1 + 0.9 * d / total, f"Embedding {d}/{total}"))
        r.log(f"Embedded {done} segments; HNSW index ready")
        return n, done

    @staticmethod
    def describe_transcript(t: dict) -> dict:
        """What the speaker-naming step shows: turns, a sample per speaker, the chunk plan."""
        turns = speaker_turns(t)
        parents, children = chunk_transcript(t)
        samples = {}
        for turn in turns:
            if turn["speaker"] not in samples and len(turn["text"].split()) > 8:
                samples[turn["speaker"]] = turn["text"][:200]
        return {
            "duration_s": t.get("duration") or 0,
            "raw_speaker_labels": sorted({w.speaker for w in words_from_transcript(t)}),
            "provider": t.get("provider"),
            "speaker_names": speaker_names(t),     # identified by the transcriber, or null
            "turns": turns, "samples": samples,
            "parents": [{"index": p.index, "start": p.start, "end": p.end, "tokens": p.tokens,
                         "turns": len(p.turns)} for p in parents],
            "children": [{"parent": c.parent, "speaker": c.speaker, "start": c.start, "end": c.end,
                          "tokens": c.tokens, "text": c.text} for c in children],
        }
