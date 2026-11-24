from fastapi import APIRouter

from audiosearch.db import EmbeddingRepository, FileRepository, collection_conn

from ..dependencies import CollectionDep, ContainerDep
from ..schemas import Health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health)
async def health(app: ContainerDep, collection: CollectionDep):
    names = [c["name"] for c in await app.collections.list_all()]
    async with collection_conn(app.pool, collection) as conn:
        files, segs = await FileRepository(conn).counts()
        rows = await EmbeddingRepository(conn).stats()
    stats = app.pool.get_stats()
    return Health(database=True, collection=collection, collections=names,
                  files=files, segments=segs,
                  embeddings={m: n for m, n, _ in rows}, embedding_dims={m: d for m, _, d in rows},
                  embedding_model=app.embedder.name,
                  reranker=app.reranker.name, rerank_default=app.settings.rerank_default,
                  pool={k: stats.get(k, 0) for k in ("pool_min", "pool_max", "pool_size", "pool_available",
                                                     "requests_waiting")})
