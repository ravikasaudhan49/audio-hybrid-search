"""Postgres: collections, schema management and the async connection pool.

A COLLECTION is an isolated set of tables for one tenant / dataset (like a MongoDB
database): collection "nasa" lives in the Postgres schema `col_nasa`, holding its own
files, speakers, segments and embeddings tables and indexes. Every collection is created
from the same schema.sql, so column names and types are identical across collections.
Code selects a collection per connection via search_path (`collection_conn`), so queries
stay unqualified and one collection's rows are never visible from another (ADR-0007).
"""
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection, sql
from psycopg_pool import AsyncConnectionPool

from ..core.errors import CollectionError
from ..core.log import get
from ..core.settings import Settings, get_settings

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


def is_valid_name(collection: str | None) -> bool:
    return bool(NAME_RE.match(collection or ""))


def schema_name(collection: str) -> str:
    """'nasa' -> 'col_nasa'. Names become SQL identifiers, so they are strictly validated."""
    if not is_valid_name(collection):
        raise CollectionError(f"invalid collection name {collection!r}: use 1-40 lowercase letters, digits "
                              "or '_', starting with a letter")
    return SCHEMA_PREFIX + collection


async def use_collection(conn: AsyncConnection, collection: str) -> None:
    """Point unqualified table names at this collection (public stays on the path for the
    vector / pg_trgm types and operators)."""
    await conn.execute("SELECT set_config('search_path', %s, false)", (f"{schema_name(collection)}, public",))


async def reset_search_path(conn: AsyncConnection) -> None:
    await conn.execute("SELECT set_config('search_path', 'public', false)")


@asynccontextmanager
async def collection_conn(pool: AsyncConnectionPool, collection: str) -> AsyncIterator[AsyncConnection]:
    """A pooled connection scoped to one collection; the search_path is reset on release so a
    reused connection can never query the previous caller's collection by accident."""
    async with pool.connection() as conn:
        await use_collection(conn, collection)
        try:
            yield conn
        finally:
            await reset_search_path(conn)


async def create_collection_schema(conn: AsyncConnection, name: str, description: str | None = None) -> None:
    """Create (idempotently) a collection's schema, tables and registry row."""
    schema = schema_name(name)
    await conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
    await use_collection(conn, name)
    await conn.execute(SCHEMA.read_text())
    await reset_search_path(conn)
    await conn.execute(
        """INSERT INTO public.collections (name, description) VALUES (%s, %s)
           ON CONFLICT (name) DO UPDATE SET description = coalesce(EXCLUDED.description, collections.description)""",
        (name, description))


async def _configure(conn: AsyncConnection) -> None:
    await register_vector_async(conn)
    for stmt in SESSION_SETTINGS:
        await conn.execute(stmt)


class Database:
    """Schema bootstrap and pool construction for one database URL."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    async def _migrate_legacy_tables(self, conn: AsyncConnection) -> None:
        """One-time move of pre-collection tables (public.files, ...) into the default collection's
        schema. ALTER TABLE ... SET SCHEMA moves rows, indexes, sequences and constraints as-is,
        so embeddings and the HNSW index are kept."""
        cur = await conn.execute("SELECT to_regclass('public.segments') IS NOT NULL")
        if not (await cur.fetchone())[0]:
            return
        default = self.settings.default_collection
        schema = schema_name(default)
        await conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        async with conn.transaction():
            for table in TABLES:
                await conn.execute(sql.SQL("ALTER TABLE IF EXISTS public.{} SET SCHEMA {}")
                                   .format(sql.Identifier(table), sql.Identifier(schema)))
        log.warning("migrated existing tables %s from public into collection %r (schema %s)",
                    TABLES, default, schema)

    async def init_schema(self) -> None:
        """Extensions, the collections registry and the default collection. Uses a plain
        connection: the vector type must exist before pooled connections can register it."""
        async with await AsyncConnection.connect(self.settings.database_url, autocommit=True) as conn:
            await conn.execute(GLOBAL_SQL)
            await self._migrate_legacy_tables(conn)
            await create_collection_schema(conn, self.settings.default_collection)
        log.info("schema ready (default collection %r)", self.settings.default_collection)

    async def open_pool(self, min_size: int | None = None, max_size: int | None = None) -> AsyncConnectionPool:
        await self.init_schema()
        s = self.settings
        pool = AsyncConnectionPool(
            s.database_url,
            min_size=min_size or s.db_pool_min,
            max_size=max_size or s.db_pool_max,
            timeout=s.db_pool_timeout_s,        # wait this long for a free connection, then fail fast
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
