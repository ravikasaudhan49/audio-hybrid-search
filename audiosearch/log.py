"""Logging setup shared by every process (API, UI, CLI, eval).

Each process writes a plain-text, rotating log file under logs/ (logs/api.log, logs/ui.log,
...) and mirrors it to the console. One file per process: Windows can't rotate a file
another process holds open. Every line carries a request/job id, so all the work done
for one search or upload can be followed with a single search in the file:

  2026-09-26 11:02:03.412 INFO    audiosearch.search     [a1b2c3d4] hybrid search done ...

Level via LOG_LEVEL in .env (DEBUG shows per-stage detail such as cache hits).
API keys are never logged.
"""
import contextvars
import logging
import logging.handlers
import time
from contextlib import contextmanager
from pathlib import Path

from . import config

# Correlation id for the current request / job; asyncio tasks inherit it automatically.
request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-24s [%(request_id)s] %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
QUIET = ("httpx", "httpcore", "psycopg", "psycopg.pool", "urllib3", "sentence_transformers",
         "huggingface_hub", "filelock", "watchdog", "multipart", "uvicorn.access", "asyncio")
_configured: Path | None = None


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True


def setup(component: str) -> Path:
    """Configure root logging for this process; idempotent. Returns the log file path."""
    global _configured
    if _configured:
        return _configured
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.LOG_DIR / f"{component}.log"
    formatter = logging.Formatter(FORMAT, DATEFMT)
    handlers = [
        logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"),
        logging.StreamHandler(),
    ]
    root = logging.getLogger()
    root.setLevel(config.LOG_LEVEL)
    for h in handlers:
        h.setFormatter(formatter)
        h.addFilter(_RequestIdFilter())
        root.addHandler(h)
    for name in QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)
    # uvicorn installs its own handlers; route its logs through ours instead.
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    _configured = path
    logging.getLogger("audiosearch").info("logging to %s (level %s)", path, config.LOG_LEVEL)
    return path


__all__ = ["get", "logging", "request_id", "setup", "timed"]


def get(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def timed(logger: logging.Logger, what: str, level: int = logging.INFO,
          expected: tuple[type[BaseException], ...] = (), **fields):
    """Log `what` with its duration (and extra key=value fields) when the block finishes.
    On failure logs ERROR, or WARNING for `expected` exceptions (handled fallbacks)."""
    t0 = time.perf_counter()
    extra = "".join(f" {k}={v}" for k, v in fields.items())
    try:
        yield
    except Exception as e:
        lvl = logging.WARNING if isinstance(e, expected) else logging.ERROR
        logger.log(lvl, "%s failed after %.0f ms%s: %s: %s", what, (time.perf_counter() - t0) * 1000,
                   extra, type(e).__name__, e)
        raise
    logger.log(level, "%s done in %.0f ms%s", what, (time.perf_counter() - t0) * 1000, extra)
