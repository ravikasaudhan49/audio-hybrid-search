"""API contract tests, run in-process over ASGI (needs Postgres; skipped otherwise)."""
import httpx
import pytest

from audiosearch import aio, db
from backend.api import app, lifespan

try:
    aio.run(db.init_schema())
except Exception as e:  # DB not reachable
    pytest.skip(f"database unavailable: {e}", allow_module_level=True)


def call(fn):
    """Run `fn(client)` against the app with its lifespan (pool open/close) on the selector loop."""
    async def go():
        async with lifespan(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await fn(client)
    return aio.run(go())


def test_health_reports_pool():
    body = call(lambda c: c.get("/health")).json()
    assert body["database"] is True and "gemini" in body["embedders"]
    assert body["pool"]["pool_max"] >= body["pool"]["pool_min"] >= 1


def test_search_rejects_unknown_method_and_embedder():
    async def fn(c):
        return [(await c.get("/search", params={"q": "x", "method": "bogus"})).status_code,
                (await c.get("/search", params={"q": "x", "embedder": "bogus"})).status_code]
    assert call(fn) == [400, 400]


def test_keyword_search_with_no_match():
    r = call(lambda c: c.get("/search", params={"q": "zzzqqqxyzzy", "method": "keyword", "rerank": False}))
    assert r.status_code == 200
    assert r.json()["results"] == [] and r.json()["method"] == "keyword"


def test_concurrent_searches_share_the_pool():
    async def fn(c):
        import asyncio
        rs = await asyncio.gather(*(c.get("/search", params={"q": f"term{i}", "method": "keyword", "rerank": False})
                                    for i in range(20)))
        return [r.status_code for r in rs]
    assert call(fn) == [200] * 20


def test_upload_rejects_non_audio():
    r = call(lambda c: c.post("/uploads", files={"file": ("notes.txt", b"hello")}))
    assert r.status_code == 400


def test_unknown_ids_404():
    async def fn(c):
        return [(await c.get("/files/does-not-exist")).status_code,
                (await c.get("/jobs/does-not-exist")).status_code,
                (await c.delete("/files/does-not-exist")).status_code]
    assert call(fn) == [404, 404, 404]


def test_golden_files_cannot_be_deleted():
    async def fn(c):
        golden = [f for f in (await c.get("/files")).json() if f["origin"] == "golden"]
        if not golden:
            return None
        return (await c.delete(f"/files/{golden[0]['id']}")).status_code
    status = call(fn)
    if status is None:
        pytest.skip("no golden files ingested yet")
    assert status == 403


def test_collections_listed_and_validated():
    async def fn(c):
        listed = (await c.get("/collections")).json()
        bad = (await c.post("/collections", json={"name": "Bad Name!"})).status_code
        dup = (await c.post("/collections", json={"name": "nasa"})).status_code
        unknown = (await c.get("/search", params={"q": "x", "collection": "does_not_exist",
                                                  "method": "keyword", "rerank": False})).status_code
        invalid = (await c.get("/files", params={"collection": "DROP TABLE"})).status_code
        return listed, bad, dup, unknown, invalid
    listed, bad, dup, unknown, invalid = call(fn)
    assert "nasa" in {c["name"] for c in listed}
    assert (bad, dup, unknown, invalid) == (422, 409, 404, 400)
