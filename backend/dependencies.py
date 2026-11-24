"""FastAPI dependencies: the app's container and job manager, and validated parameters."""
from typing import Annotated

from fastapi import Depends, HTTPException, Query, Request

from audiosearch.container import AppContainer
from audiosearch.core.errors import CollectionError
from audiosearch.db import schema_name

from .jobs import JobManager


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_jobs(request: Request) -> JobManager:
    return request.app.state.jobs


ContainerDep = Annotated[AppContainer, Depends(get_container)]
JobsDep = Annotated[JobManager, Depends(get_jobs)]


async def resolve_collection(app: AppContainer, name: str | None) -> str:
    """Validate a collection name and make sure it exists (400 / 404 otherwise)."""
    name = app.collection_or_default(name)
    try:
        schema_name(name)
    except CollectionError as e:
        raise HTTPException(400, str(e)) from None
    if not await app.collections.exists(name):
        raise HTTPException(404, f"unknown collection {name!r}; create it with POST /collections")
    return name


async def collection_param(
        request: Request,
        collection: str | None = Query(None, description="Collection (dataset/tenant); default: DEFAULT_COLLECTION"),
) -> str:
    return await resolve_collection(get_container(request), collection)


CollectionDep = Annotated[str, Depends(collection_param)]

