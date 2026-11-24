"""HTTP middleware: request ids and access logging."""
import time
import uuid

from fastapi import Request

from audiosearch.core import log as logs

log = logs.get("backend.api")
QUIET_PATHS = ("/jobs/", "/health", "/files/")   # polled/streamed often: logged at DEBUG


async def log_requests(request: Request, call_next):
    """Tag each request with an id (also returned as X-Request-ID) and log method, path,
    status and duration. Everything logged while serving it carries the same id."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:8]
    token = logs.request_id.set(rid)
    t0 = time.perf_counter()
    target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    try:
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        if response.status_code >= 400:
            level = logs.logging.WARNING
        elif request.url.path.startswith(QUIET_PATHS):
            level = logs.logging.DEBUG
        else:
            level = logs.logging.INFO
        log.log(level, "%s %s -> %d in %.0f ms", request.method, target, response.status_code, ms)
        response.headers["X-Request-ID"] = rid
        return response
    except Exception:
        log.exception("%s %s -> 500 unhandled error after %.0f ms", request.method, target,
                      (time.perf_counter() - t0) * 1000)
        raise
    finally:
        logs.request_id.reset(token)
