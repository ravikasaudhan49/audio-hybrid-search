"""Repositories: every SQL statement outside the search rankers lives here.

`CollectionRepository` works on the pool (the registry is in `public`). The per-table
repositories take a connection that is ALREADY scoped to one collection by
`collection_conn`, so callers decide the transaction boundary:

    async with collection_conn(pool, "nasa") as conn, conn.transaction():
        await FileRepository(conn).upsert(...)
        await SpeakerRepository(conn).upsert_many(...)
"""
from psycopg import AsyncConnection, sql
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from ..core.errors import CollectionError
from ..core.log import get
from ..models import Child, FileRecord, Parent
from .database import create_collection_schema, is_valid_name, schema_name

log = get(__name__)

Query = tuple[str, dict]


class CollectionRepository:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def create(self, name: str, description: str | None = None) -> None:
        async with self.pool.connection() as conn:
            await create_collection_schema(conn, name, description)
        log.info("collection %r ready (schema %s)", name, schema_name(name))

    async def exists(self, name: str) -> bool:
        if not is_valid_name(name):
            return False
        async with self.pool.connection() as conn:
            cur = await conn.execute("SELECT 1 FROM public.collections WHERE name = %s", (name,))
            return await cur.fetchone() is not None

    async def require(self, name: str) -> None:
        if not await self.exists(name):
            raise CollectionError(f"unknown collection {name!r}")

    async def list_all(self) -> list[dict]:
        """Registered collections with their file / chunk counts."""
        async with self.pool.connection() as conn:
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

    async def get(self, name: str) -> dict | None:
        return next((c for c in await self.list_all() if c["name"] == name), None)


class FileRepository:
    _INFO_SQL = """
        SELECT f.id, f.title, f.origin, f.duration_s, f.audio_url, f.source_offset_s,
               coalesce(array_agg(sp.name ORDER BY sp.label) FILTER (WHERE sp.id IS NOT NULL), '{}'),
               (SELECT count(*) FROM segments s WHERE s.file_id = f.id)
        FROM files f LEFT JOIN speakers sp ON sp.file_id = f.id
        WHERE %(id)s::text IS NULL OR f.id = %(id)s
        GROUP BY f.id ORDER BY f.origin, f.id"""

    def __init__(self, conn: AsyncConnection):
        self.conn = conn

    async def upsert(self, fid: str, title: str, audio: str, duration, source: str | None = None,
                     origin: str = "golden", audio_url: str | None = None, offset_s: float = 0.0) -> None:
        await self.conn.execute(
            """INSERT INTO files (id, title, audio_path, duration_s, source, origin, audio_url, source_offset_s)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (id) DO UPDATE SET title=EXCLUDED.title, audio_path=EXCLUDED.audio_path,
                 duration_s=EXCLUDED.duration_s, source=EXCLUDED.source, origin=EXCLUDED.origin,
                 audio_url=EXCLUDED.audio_url, source_offset_s=EXCLUDED.source_offset_s""",
            (fid, title, audio, duration, source, origin, audio_url, offset_s),
        )

    @staticmethod
    def _record(r) -> FileRecord:
        return FileRecord(id=r[0], title=r[1], origin=r[2], duration_s=r[3], audio_url=r[4],
                          source_offset_s=r[5], speakers=list(r[6]), segments=r[7])

    async def list_records(self) -> list[FileRecord]:
        cur = await self.conn.execute(self._INFO_SQL, {"id": None})
        return [self._record(r) for r in await cur.fetchall()]

    async def get_record(self, fid: str) -> FileRecord | None:
        cur = await self.conn.execute(self._INFO_SQL, {"id": fid})
        row = await cur.fetchone()
        return self._record(row) if row else None

    async def counts(self) -> tuple[int, int]:
        """(files, segments) in this collection."""
        cur = await self.conn.execute("SELECT (SELECT count(*) FROM files), (SELECT count(*) FROM segments)")
        return await cur.fetchone()

    async def count_by_origin(self, origin: str) -> int:
        cur = await self.conn.execute("SELECT count(*) FROM files WHERE origin = %s", (origin,))
        return (await cur.fetchone())[0]

    async def origin(self, fid: str) -> str | None:
        cur = await self.conn.execute("SELECT origin FROM files WHERE id=%s", (fid,))
        row = await cur.fetchone()
        return row[0] if row else None

    async def audio_path(self, fid: str) -> str | None:
        cur = await self.conn.execute("SELECT audio_path FROM files WHERE id=%s", (fid,))
        row = await cur.fetchone()
        return row[0] if row else None

    async def delete(self, fid: str) -> None:
        """Speakers, segments and embeddings cascade."""
        await self.conn.execute("DELETE FROM files WHERE id=%s", (fid,))


class SpeakerRepository:
    def __init__(self, conn: AsyncConnection):
        self.conn = conn

    async def upsert_many(self, fid: str, names: dict, roles: dict | None = None) -> dict[int, int]:
        """Upsert speaker names and roles (host | guest); returns {diarization label: speakers.id}."""
        ids = {}
        roles = {int(k): v for k, v in (roles or {}).items()}
        for label, name in names.items():
            cur = await self.conn.execute(
                """INSERT INTO speakers (file_id, label, name, role) VALUES (%s,%s,%s,%s)
                   ON CONFLICT (file_id, label) DO UPDATE SET name=EXCLUDED.name, role=EXCLUDED.role RETURNING id""",
                (fid, int(label), name, roles.get(int(label))),
            )
            ids[int(label)] = (await cur.fetchone())[0]
        return ids

    async def names_by_label(self, fid: str) -> dict[int, str]:
        cur = await self.conn.execute("SELECT label, name FROM speakers WHERE file_id=%s", (fid,))
        return dict(await cur.fetchall())

    async def distinct_names(self) -> list[str]:
        cur = await self.conn.execute("SELECT DISTINCT name FROM speakers")
        return [r[0] for r in await cur.fetchall()]


def parent_key(fid: str, index: int) -> str:
    """Corpus-unique parent id, the dedup key at query time."""
    return f"{fid}:p{index:02d}"


class SegmentRepository:
    def __init__(self, conn: AsyncConnection):
        self.conn = conn

    async def count_for_file(self, fid: str) -> int:
        cur = await self.conn.execute("SELECT count(*) FROM segments WHERE file_id=%s", (fid,))
        return (await cur.fetchone())[0]

    async def refresh_audio(self, fid: str, audio_url: str | None, offset_s: float) -> None:
        await self.conn.execute("UPDATE segments SET audio_url=%s, source_offset_s=%s WHERE file_id=%s",
                                (audio_url, offset_s, fid))

    async def replace_for_file(self, fid: str, parents: list[Parent], children: list[Child],
                               speaker_ids: dict[int, int], names: dict[int, str],
                               audio_url: str | None = None, offset_s: float = 0.0) -> int:
        """One row per child, each carrying its parent's id and text, the episode audio_url and the
        clip's offset in the episode (episode_* times are derived from it by Postgres)."""
        # Replacing a file's segments also drops their embeddings (ON DELETE CASCADE).
        await self.conn.execute("DELETE FROM segments WHERE file_id=%s", (fid,))
        parent_text = {p.index: p.render(names) for p in parents}
        by_index = {p.index: p for p in parents}
        async with self.conn.cursor() as cur:
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

    async def texts_missing_embedding(self, model: str) -> list[tuple[int, str]]:
        """(segment id, text to embed) for segments without a vector for `model`."""
        cur = await self.conn.execute(
            # Embedded text = "Speaker: child_text", so speaker-scoped questions match on meaning.
            """SELECT s.id, sp.name || ': ' || s.child_text FROM segments s JOIN speakers sp ON sp.id = s.speaker_id
               WHERE NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.segment_id=s.id AND e.model=%s)
               ORDER BY s.id""",
            (model,),
        )
        return await cur.fetchall()

    async def ranked_ids(self, query: Query) -> list[int]:
        """Run one ranker statement (see search/rankers.py); segment ids in rank order."""
        sql_text, params = query
        cur = await self.conn.execute(sql_text, params)
        return [r[0] for r in await cur.fetchall()]

    async def fetch_for_results(self, ids: list[int], vector_model: str | None = None) -> dict[int, dict]:
        """Segment details by id; with `vector_model`, also each segment's stored embedding
        (used by MMR, so diversification needs no extra API calls)."""
        if not ids:
            return {}
        cur = await self.conn.execute(
            """SELECT s.id, s.file_id, f.title, f.audio_path, sp.name, s.start_s, s.end_s, s.child_text, s.words,
                      s.parent_id, s.parent_start_s, s.parent_end_s, s.parent_text,
                      s.audio_url, s.source_offset_s, e.embedding
               FROM segments s JOIN files f ON f.id = s.file_id JOIN speakers sp ON sp.id = s.speaker_id
               LEFT JOIN embeddings e ON e.segment_id = s.id AND e.model = %s
               WHERE s.id = ANY(%s)""", (vector_model, ids))
        names = [d.name for d in cur.description]
        return {r[0]: dict(zip(names, r, strict=True)) for r in await cur.fetchall()}


def vector_index_name(model: str) -> str:
    # Lowercase: the name used to be interpolated unquoted (Postgres folds it), so existing
    # indexes keep matching now that it is passed as a quoted identifier.
    return ("emb_hnsw_" + "".join(c if c.isalnum() else "_" for c in model)).lower()


class EmbeddingRepository:
    def __init__(self, conn: AsyncConnection):
        self.conn = conn

    async def insert_many(self, model: str, rows: list[tuple[int, object]]) -> None:
        async with self.conn.cursor() as cur:
            await cur.executemany(
                """INSERT INTO embeddings (segment_id, model, embedding) VALUES (%s,%s,%s)
                   ON CONFLICT (segment_id, model) DO NOTHING""",
                [(sid, model, vec) for sid, vec in rows],
            )

    async def stats(self) -> list[tuple[str, int, int]]:
        """(model, vector count, dimensions) per embedding model."""
        cur = await self.conn.execute(
            "SELECT model, count(*), max(vector_dims(embedding)) FROM embeddings GROUP BY model")
        return await cur.fetchall()

    async def ensure_index(self, model: str, dim: int) -> None:
        """Partial HNSW index per model (created in the connection's current collection).
        Queries must use the same cast + WHERE to hit it."""
        name = vector_index_name(model)
        log.info("ensuring HNSW index %s (dim=%d)", name, dim)
        await self.conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON embeddings "
                    "USING hnsw ((embedding::vector({})) vector_cosine_ops) WHERE model = {}")
            .format(sql.Identifier(name), sql.Literal(int(dim)), sql.Literal(model)))
