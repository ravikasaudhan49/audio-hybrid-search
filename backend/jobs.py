"""Background jobs for slow work (transcription, embedding, evaluation) as asyncio tasks.

The API returns a job id immediately; clients poll GET /jobs/{id} for stage, progress and log.
At most JOB_CONCURRENCY jobs run at once; the rest wait in "queued". Jobs live in memory,
which is fine for a single process; production would use a durable queue (a Postgres jobs
table, or Celery/RQ/Arq) so jobs survive restarts and can run on separate workers.
"""
import asyncio
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable

from audiosearch import config
from audiosearch.log import get, request_id

from .schemas import Job

log = get(__name__)

_jobs: dict[str, Job] = {}
_tasks: set[asyncio.Task] = set()   # strong refs so running tasks aren't garbage-collected
_slots: asyncio.Semaphore | None = None


class JobHandle:
    """What a running job coroutine uses to report progress. All calls happen on the
    event loop thread, so no locking is needed."""

    def __init__(self, job: Job):
        self.job = job

    def stage(self, name: str, progress: float | None = None) -> None:
        log.info("stage: %s", name)
        self.job.stage = name
        self.job.log.append(name)
        if progress is not None:
            self.job.progress = progress

    def progress(self, value: float, note: str | None = None) -> None:
        self.job.progress = value
        if note:
            self.job.stage = note

    def log(self, msg: str) -> None:
        log.info("%s", msg)
        self.job.log.append(msg)


JobFn = Callable[[JobHandle], Awaitable[dict | None]]


async def _run(job: Job, fn: JobFn) -> None:
    global _slots
    _slots = _slots or asyncio.Semaphore(config.JOB_CONCURRENCY)
    request_id.set(f"job:{job.id}")  # every log line of this job carries its id
    if _slots.locked():
        log.info("%s job queued (%d slots busy)", job.kind, config.JOB_CONCURRENCY)
    async with _slots:
        job.stage = "running"
        t0 = time.perf_counter()
        log.info("%s job started", job.kind)
        try:
            job.result = await fn(JobHandle(job))
            job.status, job.progress, job.stage = "done", 1.0, "done"
            log.info("%s job done in %.1fs", job.kind, time.perf_counter() - t0)
        except Exception as e:
            job.status, job.error = "error", f"{type(e).__name__}: {e}"
            job.log.append(traceback.format_exc(limit=3))
            log.exception("%s job failed after %.1fs", job.kind, time.perf_counter() - t0)


def start(kind: str, fn: JobFn) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], kind=kind, status="running", stage="queued", progress=0.0, log=[])
    _jobs[job.id] = job
    task = asyncio.create_task(_run(job, fn), name=f"job-{kind}-{job.id}")
    log.info("created %s job %s", kind, job.id)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return job


def get(job_id: str) -> Job | None:
    job = _jobs.get(job_id)
    return job.model_copy(deep=True) if job else None


async def shutdown() -> None:
    """Cancel running jobs on server shutdown."""
    for task in list(_tasks):
        task.cancel()
    await asyncio.gather(*_tasks, return_exceptions=True)
