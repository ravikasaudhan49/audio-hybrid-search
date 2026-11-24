"""Two-stage upload: transcribe (stage 1), then index once the user has named the speakers
(stage 2). Both run as background jobs; the work itself is IngestionService's."""
import asyncio
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from audiosearch.core.log import get
from audiosearch.core.text import split_terms, upload_file_id

from ..dependencies import ContainerDep, JobsDep, resolve_collection
from ..jobs import JobHandle
from ..schemas import IndexRequest, Job

router = APIRouter(prefix="/uploads", tags=["uploads"])
log = get("backend.api")

AUDIO_TYPES = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


@router.post("", response_model=Job, status_code=202)
async def upload(app: ContainerDep, jobs: JobsDep, file: UploadFile = File(...), title: str = Form(""),
                 keyterms: str = Form(""), collection: str = Form("")):
    """Stage 1: store + normalize audio, transcribe with diarization. Poll /jobs/{id};
    the result holds the diarized turns and chunk preview for speaker naming.
    The target collection is checked up front; rows are written to it in stage 2."""
    collection = await resolve_collection(app, collection or None)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in AUDIO_TYPES:
        log.warning("upload rejected: unsupported type %r (%s)", ext, file.filename)
        raise HTTPException(400, f"unsupported type {ext!r}; use one of {sorted(AUDIO_TYPES)}")
    data = await file.read()
    fid = upload_file_id(data)
    terms = split_terms(keyterms)
    log.info("upload received name=%r bytes=%d -> file_id=%s keyterms=%d", file.filename, len(data), fid, len(terms))
    raw = app.settings.raw_dir / f"{fid}{ext}"
    raw.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(raw.write_bytes, data)
    title = title or Path(file.filename or fid).stem
    ingestion = app.ingestion(collection)

    async def work(h: JobHandle) -> dict:
        _, plan = await ingestion.transcribe_upload(fid, raw, terms, h)
        return {"file_id": fid, "title": title, "collection": collection, **plan}

    return jobs.start("transcribe", work)


@router.post("/{file_id}/index", response_model=Job, status_code=202)
async def index_upload(app: ContainerDep, jobs: JobsDep, file_id: str, req: IndexRequest):
    """Stage 2: store chunks with named speakers (one transaction), embed and index them."""
    if not app.transcripts.exists(file_id):
        raise HTTPException(404, "no transcript for this file; POST /uploads first")
    collection = await resolve_collection(app, req.collection)
    ingestion = app.ingestion(collection)

    async def work(h: JobHandle) -> dict:
        n, _ = await ingestion.index_upload(file_id, req.title, req.speakers, req.host,
                                            audio_url=f"{app.settings.audio_url_base}{file_id}.mp3",
                                            reporter=h)
        return {"file_id": file_id, "segments": n, "embedder": app.settings.embedder, "collection": collection}

    return jobs.start("index", work)
