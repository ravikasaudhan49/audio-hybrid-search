from fastapi import APIRouter, HTTPException

from audiosearch.core.log import get

from ..dependencies import ContainerDep
from ..schemas import CollectionCreate, CollectionInfo

router = APIRouter(prefix="/collections", tags=["collections"])
log = get("backend.api")


@router.get("", response_model=list[CollectionInfo])
async def list_collections(app: ContainerDep):
    """All collections with their file and chunk counts."""
    return await app.collections.list_all()


@router.post("", response_model=CollectionInfo, status_code=201)
async def create_collection(app: ContainerDep, body: CollectionCreate):
    """Create a collection: its own files/speakers/segments/embeddings tables, identical columns."""
    repo = app.collections
    if await repo.exists(body.name):
        raise HTTPException(409, f"collection {body.name!r} already exists")
    await repo.create(body.name, body.description)
    log.info("created collection %r", body.name)
    return await repo.get(body.name)
