"""Async runtime helpers shared by every entry point (API, CLI, eval, tests).

psycopg's async driver does not support Windows' default ProactorEventLoop, so all code runs
on a SelectorEventLoop: `run()` for scripts, `--loop asyncio:SelectorEventLoop` for uvicorn.
A selector loop on Windows can't spawn asyncio subprocesses, so ffmpeg runs via to_thread.
"""
import asyncio
import weakref
from collections.abc import Coroutine
from typing import Any

import httpx

from .settings import get_settings

_http_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = weakref.WeakKeyDictionary()


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """asyncio.run on a SelectorEventLoop (required by psycopg async on Windows)."""
    return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)


def http() -> httpx.AsyncClient:
    """Shared keep-alive HTTP client for external APIs: a Singleton per event loop, because an
    httpx client is bound to the loop it was first used on."""
    loop = asyncio.get_running_loop()
    client = _http_clients.get(loop)
    if client is None or client.is_closed:
        limit = get_settings().http_max_connections
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(120, connect=10),
            limits=httpx.Limits(max_connections=limit, max_keepalive_connections=limit // 2),
        )
        _http_clients[loop] = client
    return client


async def close_http() -> None:
    client = _http_clients.pop(asyncio.get_running_loop(), None)
    if client is not None:
        await client.aclose()
