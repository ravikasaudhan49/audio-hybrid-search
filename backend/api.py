"""FastAPI backend (fully async).

Run:  uvicorn backend.api:app --port 8000 --loop asyncio:SelectorEventLoop     (docs at /docs)
      (the selector loop is required by psycopg's async driver on Windows)

Thin HTTP layer over the `audiosearch` library: it owns the async DB pool and the loaded
embedding models; slow work (transcription, embedding, eval) runs as background jobs.
"""
import asyncio
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from psycopg_pool import AsyncConnectionPool

from audiosearch import aio, config, db, golden
from audiosearch import log as logs
from audiosearch.answer import generate_answer
from audiosearch.ingest import embed_missing, normalize_audio, save_file
from audiosearch.search import METHODS, Searcher
from audiosearch.segment import chunk_transcript, speaker_names, speaker_turns, words_from_transcript
from audiosearch.transcribe import load_transcript, transcribe, transcript_path
from eval.evaluate import RESULTS_JSON, run_evaluation

from . import jobs
from .schemas import (
    AnswerOut,
    CollectionCreate,
    CollectionInfo,
    FileDetail,
    FileInfo,
    Health,
    IndexRequest,
    Job,
    SearchResponse,
    SearchResult,
    Turn,
)

EMBEDDERS = ["gemini", "bge-base", "bge-small"]
AUDIO_TYPES = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
QUIET_PATHS = ("/jobs/", "/health", "/files/")   # polled/streamed often: logged at DEBUG

logs.setup("api")
log = logs.get("backend.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("API starting: embedder=%s reranker=%s (default %s)", config.EMBEDDER,
             config.COHERE_RERANK_MODEL if config.RERANK_PROVIDER == "cohere" else config.RERANK_PROVIDER,
             "on" if config.RERANK_DEFAULT else "off")
    app.state.pool = await db.open_pool()
    try:
        yield
    finally:
        log.info("API shutting down")
        await jobs.shutdown()
        await app.state.pool.close()
        await aio.close_http()


app = FastAPI(title="Audio Transcript Search API", version="1.0", lifespan=lifespan,
              description="Hybrid (keyword + semantic) search over two-speaker audio transcripts.")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Tag each request with an id (also returned as X-Request-ID) and log method, path,
    status and duration. Everything logged while serving it carries the same id."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:8]
    token = logs.request_id.set(rid)
    t0 = time.perf_counter()
    target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    try:
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        if response.status_code >= 400:
            level = logs.logging.WARNING
        elif request.url.path.startswith(QUIET_PATHS):
            level = logs.logging.DEBUG
        else:
            level = logs.logging.INFO
        log.log(level, "%s %s -> %d in %.0f ms", request.method, target, response.status_code, ms)
        response.headers["X-Request-ID"] = rid
        return response
    except Exception:
        log.exception("%s %s -> 500 unhandled error after %.0f ms", request.method, target,
                      (time.perf_counter() - t0) * 1000)
        raise
    finally:
        logs.request_id.reset(token)


def pool_of(request: Request) -> AsyncConnectionPool:
    return request.app.state.pool


def stream_url(file_id: str, collection: str) -> str:
    return f"/files/{file_id}/audio?collection={collection}"


CollectionParam = Query(None, description="Collection (dataset/tenant); default: DEFAULT_COLLECTION")


async def resolve_collection(request: Request, name: str | None) -> str:
    """Validate a collection name and make sure it exists (400 / 404 otherwise)."""
    name = name or config.DEFAULT_COLLECTION
    try:
        db.schema_name(name)
    except db.CollectionError as e:
        raise HTTPException(400, str(e)) from None
    if not await db.collection_exists(pool_of(request), name):
        raise HTTPException(404, f"unknown collection {name!r}; create it with POST /collections")
    return name


def check_embedder(name: str | None) -> str:
    name = name or config.EMBEDDER
    if name not in EMBEDDERS:
        raise HTTPException(400, f"embedder must be one of {EMBEDDERS}")
    return name


# ---------- health ----------

@app.get("/health", response_model=Health)
async def health(request: Request, collection: str | None = CollectionParam):
    pool = pool_of(request)
    collection = await resolve_collection(request, collection)
    names = [c["name"] for c in await db.list_collections(pool)]
    async with db.collection_conn(pool, collection) as conn:
        cur = await conn.execute("SELECT (SELECT count(*) FROM files), (SELECT count(*) FROM segments)")
        files, segs = await cur.fetchone()
        cur = await conn.execute("SELECT model, count(*), max(vector_dims(embedding)) FROM embeddings GROUP BY model")
        rows = await cur.fetchall()
        emb, dims = {m: n for m, n, _ in rows}, {m: d for m, _, d in rows}
    stats = pool.get_stats()
    reranker = config.COHERE_RERANK_MODEL if config.RERANK_PROVIDER == "cohere" else "bge-reranker-base"
    return Health(database=True, collection=collection, collections=names,
                  files=files, segments=segs, embeddings=emb, embedding_dims=dims,
                  default_embedder=config.EMBEDDER, embedders=EMBEDDERS,
                  reranker=reranker, rerank_default=config.RERANK_DEFAULT,
                  pool={k: stats.get(k, 0) for k in ("pool_min", "pool_max", "pool_size", "pool_available",
                                                     "requests_waiting")})


# ---------- collections ----------

@app.get("/collections", response_model=list[CollectionInfo])
async def list_collections(request: Request):
    """All collections with their file and chunk counts."""
    return await db.list_collections(pool_of(request))


@app.post("/collections", response_model=CollectionInfo, status_code=201)
async def create_collection(request: Request, body: CollectionCreate):
    """Create a collection: its own files/speakers/segments/embeddings tables, identical columns."""
    pool = pool_of(request)
    if await db.collection_exists(pool, body.name):
        raise HTTPException(409, f"collection {body.name!r} already exists")
    await db.create_collection(pool, body.name, body.description)
    log.info("created collection %r", body.name)
    return next(c for c in await db.list_collections(pool) if c["name"] == body.name)


# ---------- search ----------

@app.get("/search", response_model=SearchResponse)
async def search(request: Request, q: str, method: str = "hybrid", k: int = Query(10, ge=1, le=50),
                 speaker: str | None = None, file_id: str | None = None,
                 rerank: bool = Query(config.RERANK_DEFAULT, description="Rerank top candidates (Cohere by default)"),
                 mmr: bool = Query(False, description="Diversify results with Maximal Marginal Relevance"),
                 mmr_lambda: float = Query(config.MMR_LAMBDA, ge=0, le=1,
                                           description="1.0 = pure relevance, 0.0 = pure diversity"),
                 group: bool = Query(True, description="One result per parent chunk (best-matching child)"),
                 role: str | None = Query(None, pattern="^(host|guest)$", description="Only this speaker role"),
                 auto_speaker: bool = Query(True, description="Detect 'what did <name>/the guest say' in q"),
                 threshold: bool = Query(True, description="Hide results below the relevance threshold"),
                 embedder: str | None = None, collection: str | None = CollectionParam,
                 answer: bool = Query(False, description="Add a grounded summary answer (Gemini) above the results"),
                 origin: str | None = Query(None, pattern="^(golden|upload)$",
                                            description="Only golden (evaluation) or uploaded files")):
    if method not in METHODS:
        raise HTTPException(400, f"method must be one of {list(METHODS)}")
    embedder = check_embedder(embedder)
    collection = await resolve_collection(request, collection)
    t0 = time.perf_counter()
    out = await Searcher(pool_of(request), embedder, origin, collection).search(q, method, k, speaker, file_id,
                                                            rerank=rerank, mmr=mmr, mmr_lambda=mmr_lambda,
                                                            group=group, role=role, auto_speaker=auto_speaker,
                                                            threshold=threshold)
    summary = None
    if answer and out.results:
        t_answer = time.perf_counter()
        try:
            a = await generate_answer(q, out.results)
            summary = AnswerOut(text=a.text, model=a.model, citations=a.citations, cached=a.cached) if a else None
        except Exception as e:  # the results are still useful without a summary
            log.warning("answer generation failed: %s: %s", type(e).__name__, e)
            out.notes.append(f"summary answer unavailable: {type(e).__name__}")
        out.timings_ms["answer"] = round((time.perf_counter() - t_answer) * 1000, 1)
    took = (time.perf_counter() - t0) * 1000
    spans = golden.relevant_spans(q)
    return SearchResponse(
        query=q, method=method, embedder=embedder, took_ms=round(took, 1), golden_query=spans is not None,
        reranked=out.reranked, reranker=out.reranker, mmr=out.mmr, grouped=out.grouped, notes=out.notes,
        search_text=out.search_text, speaker_detected=out.speaker_detected, role_detected=out.role_detected,
        below_threshold=out.below_threshold, answer=summary, timings_ms=out.timings_ms,
        results=[SearchResult(**asdict(r), timestamp=r.timestamp, stream_url=stream_url(r.file_id, collection),
                              episode_start_s=r.episode_start_s, episode_end_s=r.episode_end_s,
                              episode_hit_s=r.episode_hit_s, parent_episode_start_s=r.parent_episode_start_s,
                              parent_episode_end_s=r.parent_episode_end_s,
                              relevant=None if spans is None else any(golden.is_hit(r, x) for x in spans))
                 for r in out.results])


@app.get("/queries")
async def golden_queries():
    """The labeled evaluation queries (id, query text, type, relevant spans)."""
    return golden.load_queries()


# ---------- files ----------

FILE_SQL = """
    SELECT f.id, f.title, f.origin, f.duration_s, f.audio_url, f.source_offset_s,
           coalesce(array_agg(sp.name ORDER BY sp.label) FILTER (WHERE sp.id IS NOT NULL), '{}'),
           (SELECT count(*) FROM segments s WHERE s.file_id = f.id)
    FROM files f LEFT JOIN speakers sp ON sp.file_id = f.id
    WHERE %(id)s::text IS NULL OR f.id = %(id)s
    GROUP BY f.id ORDER BY f.origin, f.id"""


def _file_info(r, collection: str) -> FileInfo:
    return FileInfo(id=r[0], title=r[1], origin=r[2], duration_s=r[3], audio_url=r[4], source_offset_s=r[5],
                    speakers=list(r[6]), segments=r[7], stream_url=stream_url(r[0], collection))


@app.get("/files", response_model=list[FileInfo])
async def list_files(request: Request, collection: str | None = CollectionParam):
    collection = await resolve_collection(request, collection)
    async with db.collection_conn(pool_of(request), collection) as conn:
        cur = await conn.execute(FILE_SQL, {"id": None})
        return [_file_info(r, collection) for r in await cur.fetchall()]


@app.get("/files/{file_id}", response_model=FileDetail)
async def get_file(request: Request, file_id: str, collection: str | None = CollectionParam):
    collection = await resolve_collection(request, collection)
    async with db.collection_conn(pool_of(request), collection) as conn:
        cur = await conn.execute(FILE_SQL, {"id": file_id})
        row = await cur.fetchone()
        if not row:
            raise HTTPException(404, "file not found")
        cur = await conn.execute("SELECT label, name FROM speakers WHERE file_id=%s", (file_id,))
        names = dict(await cur.fetchall())
    turns = []
    if transcript_path(file_id).exists():
        resp = await asyncio.to_thread(load_transcript, file_id)
        offset = row[5]  # show episode time, like search results
        turns = [Turn(**{**t, "start": t["start"] + offset, "end": t["end"] + offset},
                      speaker_name=names.get(t["speaker"], f"Speaker {t['speaker']}"))
                 for t in speaker_turns(resp)]
    return FileDetail(**_file_info(row, collection).model_dump(), turns=turns)


@app.get("/files/{file_id}/audio")
async def get_audio(request: Request, file_id: str, collection: str | None = CollectionParam):
    """Streams the audio with HTTP range support, so players can seek to a hit."""
    collection = await resolve_collection(request, collection)
    async with db.collection_conn(pool_of(request), collection) as conn:
        cur = await conn.execute("SELECT audio_path FROM files WHERE id=%s", (file_id,))
        row = await cur.fetchone()
    path = config.AUDIO_DIR / (row[0] if row else f"{file_id}.mp3")
    if not path.exists():
        raise HTTPException(404, "audio not found")
    return FileResponse(path, media_type="audio/mpeg")


@app.delete("/files/{file_id}")
async def delete_file(request: Request, file_id: str, collection: str | None = CollectionParam):
    collection = await resolve_collection(request, collection)
    async with db.collection_conn(pool_of(request), collection) as conn:
        cur = await conn.execute("SELECT origin FROM files WHERE id=%s", (file_id,))
        row = await cur.fetchone()
        if not row:
            raise HTTPException(404, "file not found")
        if row[0] != "upload":
            raise HTTPException(403, "golden dataset files can't be deleted from the API")
        await conn.execute("DELETE FROM files WHERE id=%s", (file_id,))
    log.info("deleted uploaded file %s from collection %r (segments and embeddings cascade)", file_id, collection)
    return {"deleted": file_id, "collection": collection}


# ---------- upload pipeline (jobs) ----------

@app.post("/uploads", response_model=Job, status_code=202)
async def upload(request: Request, file: UploadFile = File(...), title: str = Form(""), keyterms: str = Form(""),
                 collection: str = Form("")):
    """Stage 1: store + normalize audio, transcribe with diarization. Poll /jobs/{id};
    the result holds the diarized turns and chunk preview for speaker naming.
    The target collection is checked up front; rows are written to it in stage 2."""
    collection = await resolve_collection(request, collection or None)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in AUDIO_TYPES:
        log.warning("upload rejected: unsupported type %r (%s)", ext, file.filename)
        raise HTTPException(400, f"unsupported type {ext!r}; use one of {sorted(AUDIO_TYPES)}")
    data = await file.read()
    fid = "up_" + hashlib.sha1(data).hexdigest()[:8]
    log.info("upload received name=%r bytes=%d -> file_id=%s keyterms=%d", file.filename, len(data), fid,
             len([t for t in keyterms.split(",") if t.strip()]))
    raw = config.RAW_DIR / f"{fid}{ext}"
    raw.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(raw.write_bytes, data)
    terms = [t.strip() for t in keyterms.split(",") if t.strip()]
    title = title or Path(file.filename or fid).stem

    async def work(h: jobs.JobHandle) -> dict:
        audio = config.AUDIO_DIR / f"{fid}.mp3"
        h.stage("Normalizing audio to 16 kHz mono mp3", 0.1)
        await normalize_audio(raw, audio)
        h.stage(f"Transcribing + diarizing + identifying speakers ({config.TRANSCRIBER})", 0.3)
        t0 = time.perf_counter()
        cached = transcript_path(fid).exists()
        resp = await transcribe(fid, audio, terms)
        dur = resp.get("duration") or 0
        h.log(f"Transcript {'loaded from cache' if cached else f'ready in {time.perf_counter() - t0:.1f}s'}"
              f" · audio {dur:.0f}s")
        h.stage("Chunking", 0.9)
        turns = speaker_turns(resp)
        parents, children = chunk_transcript(resp)
        h.log(f"{len(parents)} parents (~{config.PARENT_TOKENS} tok), {len(children)} children "
              f"(~{config.CHILD_TOKENS} tok, {config.CHILD_OVERLAP_TOKENS} overlap)")
        samples = {}
        for t in turns:
            if t["speaker"] not in samples and len(t["text"].split()) > 8:
                samples[t["speaker"]] = t["text"][:200]
        return {
            "file_id": fid, "title": title, "duration_s": dur, "collection": collection,
            "raw_speaker_labels": sorted({w.speaker for w in words_from_transcript(resp)}),
            "provider": resp.get("provider"),
            "speaker_names": speaker_names(resp),     # identified by the transcriber, or null
            "turns": turns, "samples": samples,
            "parents": [{"index": p.index, "start": p.start, "end": p.end, "tokens": p.tokens,
                         "turns": len(p.turns)} for p in parents],
            "children": [{"parent": c.parent, "speaker": c.speaker, "start": c.start, "end": c.end,
                          "tokens": c.tokens, "text": c.text} for c in children],
        }

    return jobs.start("transcribe", work)


@app.post("/uploads/{file_id}/index", response_model=Job, status_code=202)
async def index_upload(request: Request, file_id: str, req: IndexRequest):
    """Stage 2: store chunks with named speakers (one transaction), embed and index them."""
    if not transcript_path(file_id).exists():
        raise HTTPException(404, "no transcript for this file; POST /uploads first")
    embedder = check_embedder(req.embedder)
    collection = await resolve_collection(request, req.collection)
    pool = pool_of(request)

    async def work(h: jobs.JobHandle) -> dict:
        resp = await asyncio.to_thread(load_transcript, file_id)
        h.stage(f"Storing file, speakers and chunks in collection '{collection}'", 0.05)
        roles = None if req.host is None else {lbl: "host" if lbl == req.host else "guest" for lbl in req.speakers}
        n = await save_file(pool, file_id, req.title, f"{file_id}.mp3", resp, req.speakers, origin="upload",
                            audio_url=f"{config.AUDIO_URL_BASE}{file_id}.mp3", roles=roles,
                            collection=collection)
        h.log(f"Stored {n} segments (full-text and trigram indexes update automatically)")
        h.stage(f"Embedding with {embedder}", 0.1)
        done = await embed_missing(pool, embedder,
                                   progress=lambda d, t: h.progress(0.1 + 0.9 * d / t, f"Embedding {d}/{t}"),
                                   collection=collection)
        h.log(f"Embedded {done} segments; HNSW index ready")
        return {"file_id": file_id, "segments": n, "embedder": embedder, "collection": collection}

    return jobs.start("index", work)


@app.get("/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


# ---------- evaluation ----------

@app.get("/eval/results")
async def eval_results():
    if not RESULTS_JSON.exists():
        raise HTTPException(404, "no evaluation results yet")
    return json.loads(await asyncio.to_thread(RESULTS_JSON.read_text, encoding="utf-8"))


@app.post("/eval/run", response_model=Job, status_code=202)
async def eval_run(request: Request, embedders: list[str] | None = Query(None), rerank: bool = False,
                   collection: str | None = CollectionParam):
    """The golden set's labels belong to the default collection unless another is given."""
    names = [check_embedder(e) for e in (embedders or [config.EMBEDDER])]
    collection = await resolve_collection(request, collection)
    pool = pool_of(request)

    async def work(h: jobs.JobHandle) -> dict:
        h.stage(f"Evaluating all configurations on the golden set (collection '{collection}')", 0.1)
        report = await run_evaluation(pool, names, rerank, log=h.log, collection=collection)
        return {"configurations": list(report)}

    return jobs.start("eval", work)
