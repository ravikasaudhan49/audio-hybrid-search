import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from audiosearch.core.log import get
from audiosearch.db import FileRepository, SpeakerRepository, collection_conn
from audiosearch.ingestion.chunking import speaker_turns

from ..dependencies import CollectionDep, ContainerDep
from ..schemas import FileDetail, FileInfo, Turn

router = APIRouter(prefix="/files", tags=["files"])
log = get("backend.api")


@router.get("", response_model=list[FileInfo])
async def list_files(app: ContainerDep, collection: CollectionDep):
    async with collection_conn(app.pool, collection) as conn:
        records = await FileRepository(conn).list_records()
    return [FileInfo.from_record(f, collection) for f in records]


@router.get("/{file_id}", response_model=FileDetail)
async def get_file(app: ContainerDep, collection: CollectionDep, file_id: str):
    async with collection_conn(app.pool, collection) as conn:
        record = await FileRepository(conn).get_record(file_id)
        if not record:
            raise HTTPException(404, "file not found")
        names = await SpeakerRepository(conn).names_by_label(file_id)
    turns = []
    if app.transcripts.exists(file_id):
        t = await asyncio.to_thread(app.transcripts.load, file_id)
        cleaned = await asyncio.to_thread(speaker_turns, t)   # CPU: tokenizes every word
        offset = record.source_offset_s  # show episode time, like search results
        turns = [Turn(**{**turn, "start": turn["start"] + offset, "end": turn["end"] + offset},
                      speaker_name=names.get(turn["speaker"], f"Speaker {turn['speaker']}"))
                 for turn in cleaned]
    return FileDetail(**FileInfo.from_record(record, collection).model_dump(), turns=turns)


@router.get("/{file_id}/audio")
async def get_audio(app: ContainerDep, collection: CollectionDep, file_id: str):
    """Streams the audio with HTTP range support, so players can seek to a hit."""
    async with collection_conn(app.pool, collection) as conn:
        audio = await FileRepository(conn).audio_path(file_id)
    path = app.settings.audio_dir / (audio or f"{file_id}.mp3")
    if not path.exists():
        raise HTTPException(404, "audio not found")
    return FileResponse(path, media_type="audio/mpeg")


@router.delete("/{file_id}")
async def delete_file(app: ContainerDep, collection: CollectionDep, file_id: str):
    async with collection_conn(app.pool, collection) as conn:
        files = FileRepository(conn)
        origin = await files.origin(file_id)
        if origin is None:
            raise HTTPException(404, "file not found")
        if origin != "upload":
            raise HTTPException(403, "golden dataset files can't be deleted from the API")
        await files.delete(file_id)
    log.info("deleted uploaded file %s from collection %r (segments and embeddings cascade)", file_id, collection)
    return {"deleted": file_id, "collection": collection}
