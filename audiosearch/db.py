"""Postgres: collections, schema management and the async connection pool.

A COLLECTION is an isolated set of tables for one tenant / dataset (like a MongoDB
database): collection "nasa" lives in the Postgres schema `col_nasa`, holding its own
files, speakers, segments and embeddings tables and indexes. Every collection is created
from the same schema.sql, so column names and types are identical across collections.
Code selects a collection per connection via search_path (`collection_conn`), so queries
stay unqualified and one collection's rows are never visible from another.
"""
import re
from contextlib import asynccontextmanager
from pathlib import Path

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection, sql
from psycopg_pool import AsyncConnectionPool

from . import config
from .log import get

log = get(__name__)

SCHEMA = Path(__file__).with_name("schema.sql")      # per-collection tables
SCHEMA_PREFIX = "col_"
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
GLOBAL_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE TABLE IF NOT EXISTS public.collections (
    name        TEXT PRIMARY KEY,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""
TABLES = ("files", "speakers", "segments", "embeddings")

# Session settings applied to every pooled connection once, instead of per query.
SESSION_SETTINGS = (
    "SET hnsw.ef_search = 100",                        # HNSW candidate list: recall vs speed
    "SET hnsw.iterative_scan = relaxed_order",         # keep recall when WHERE filters prune results
    "SET pg_trgm.word_similarity_threshold = 0.45",    # typo tolerance for the fuzzy ranker
)


class CollectionError(ValueError):
    """Invalid or unknown collection name."""


def schema_name(collection: str) -> str:
    """'nasa' -> 'col_nasa'. Names become SQL identifiers, so they are strictly validated."""
    if not NAME_RE.match(collection or ""):
        raise CollectionError(f"invalid collection name {collection!r}: use 1-40 lowercase letters, digits "
                              "or '_', starting with a letter")
    return SCHEMA_PREFIX + collection


async def use_collection(conn: AsyncConnection, collection: str) -> None:
    """Point unqualified table names at this collection (public stays on the path for the
    vector / pg_trgm types and operators)."""
    await conn.execute("SELECT set_config('search_path', %s, false)", (f"{schema_name(collection)}, public",))


@asynccontextmanager
async def collection_conn(pool: AsyncConnectionPool, collection: str):
    """A pooled connection scoped to one collection; the search_path is reset on release so a
    reused connection can never query the previous caller's collection by accident."""
    async with pool.connection() as conn:
        await use_collection(conn, collection)
        try:
            yield conn
        finally:
            await conn.execute("SELECT set_config('search_path', 'public', false)")


async def _create_collection(conn: AsyncConnection, name: str, description: str | None = None) -> None:
    schema = schema_name(name)
    await conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
    await use_collection(conn, name)
    await conn.execute(SCHEMA.read_text())
    await conn.execute("SELECT set_config('search_path', 'public', false)")
    await conn.execute(
        """INSERT INTO public.collections (name, description) VALUES (%s, %s)
           ON CONFLICT (name) DO UPDATE SET description = coalesce(EXCLUDED.description, collections.description)""",
        (name, description))


async def _migrate_legacy_tables(conn: AsyncConnection) -> None:
    """One-time move of pre-collection tables (public.files, ...) into the default collection's
    schema. ALTER TABLE ... SET SCHEMA moves rows, indexes, sequences and constraints as-is,
    so embeddings and the HNSW index are kept."""
    cur = await conn.execute("SELECT to_regclass('public.segments') IS NOT NULL")
    if not (await cur.fetchone())[0]:
        return
    schema = schema_name(config.DEFAULT_COLLECTION)
    await conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
    async with conn.transaction():
        for table in TABLES:
            await conn.execute(sql.SQL("ALTER TABLE IF EXISTS public.{} SET SCHEMA {}")
                               .format(sql.Identifier(table), sql.Identifier(schema)))
    log.warning("migrated existing tables %s from public into collection %r (schema %s)",
                TABLES, config.DEFAULT_COLLECTION, schema)


async def init_schema() -> None:
    """Extensions, the collections registry and the default collection. Uses a plain
    connection: the vector type must exist before pooled connections can register it."""
    async with await AsyncConnection.connect(config.DATABASE_URL, autocommit=True) as conn:
        await conn.execute(GLOBAL_SQL)
        await _migrate_legacy_tables(conn)
        await _create_collection(conn, config.DEFAULT_COLLECTION)
    log.info("schema ready (default collection %r)", config.DEFAULT_COLLECTION)


async def create_collection(pool: AsyncConnectionPool, name: str, description: str | None = None) -> None:
    async with pool.connection() as conn:
        await _create_collection(conn, name, description)
    log.info("collection %r ready (schema %s)", name, schema_name(name))


async def collection_exists(pool: AsyncConnectionPool, name: str) -> bool:
    if not NAME_RE.match(name or ""):
        return False
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT 1 FROM public.collections WHERE name = %s", (name,))
        return await cur.fetchone() is not None


async def require_collection(pool: AsyncConnectionPool, name: str) -> None:
    if not await collection_exists(pool, name):
        raise CollectionError(f"unknown collection {name!r}")


async def list_collections(pool: AsyncConnectionPool) -> list[dict]:
    """Registered collections with their file / chunk counts."""
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT name, description, created_at FROM public.collections ORDER BY name")
        rows = await cur.fetchall()
        out = []
        for name, description, created in rows:
            schema = sql.Identifier(schema_name(name))
            counts = sql.SQL("SELECT (SELECT count(*) FROM {}.files), (SELECT count(*) FROM {}.segments)")
            cur = await conn.execute(counts.format(schema, schema))
            files, segments = await cur.fetchone()
            out.append({"name": name, "description": description, "created_at": created.isoformat(),
                        "files": files, "segments": segments})
    return out


async def _configure(conn: AsyncConnection) -> None:
    await register_vector_async(conn)
    for stmt in SESSION_SETTINGS:
        await conn.execute(stmt)


async def open_pool(min_size: int | None = None, max_size: int | None = None) -> AsyncConnectionPool:
    await init_schema()
    pool = AsyncConnectionPool(
        config.DATABASE_URL,
        min_size=min_size or config.DB_POOL_MIN,
        max_size=max_size or config.DB_POOL_MAX,
        timeout=config.DB_POOL_TIMEOUT_S,   # wait this long for a free connection, then fail fast
        max_idle=300,                       # shrink back toward min_size after 5 idle minutes
        max_lifetime=3600,                  # recycle connections hourly
        kwargs={"autocommit": True},
        configure=_configure,
        name="audiosearch",
        open=False,
    )
    await pool.open(wait=True)
    log.info("connection pool open min=%d max=%d timeout=%.0fs", pool.min_size, pool.max_size, pool.timeout)
    return pool


def index_name(model: str) -> str:
    return "emb_hnsw_" + "".join(c if c.isalnum() else "_" for c in model)


async def ensure_vector_index(conn: AsyncConnection, model: str, dim: int) -> None:
    """Partial HNSW index per model (created in the connection's current collection).
    Queries must use the same cast + WHERE to hit it."""
    log.info("ensuring HNSW index %s (dim=%d)", index_name(model), dim)
    await conn.execute(
        f"CREATE INDEX IF NOT EXISTS {index_name(model)} ON embeddings "
        f"USING hnsw ((embedding::vector({dim})) vector_cosine_ops) "
        f"WHERE model = '{model}'"
    )
