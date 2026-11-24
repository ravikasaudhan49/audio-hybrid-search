"""Background jobs for slow work (transcription, embedding, evaluation) as asyncio tasks.

The API returns a job id immediately; clients poll GET /jobs/{id} for stage, progress and log.
At most JOB_CONCURRENCY jobs run at once; the rest wait in "queued". Jobs live in memory,
which is fine for a single process; production would use a durable queue (a Postgres jobs
table, or Celery/RQ/Arq) so jobs survive restarts and can run on separate workers.
One JobManager per app (on app.state); finished jobs beyond `keep_finished` are evicted, so a
long-running server doesn't grow without bound.
"""
import asyncio
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable

from audiosearch.core.log import get, request_id

from .schemas import Job

log = get(__name__)


class JobHandle:
    """What a running job coroutine uses to report progress (implements
    audiosearch.core.progress.Reporter). All calls happen on the event loop thread, so no
    locking is needed."""

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


class JobManager:
    def __init__(self, concurrency: int, keep_finished: int = 500):
        self._concurrency = concurrency
        self._keep_finished = keep_finished
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task] = set()   # strong refs so running tasks aren't garbage-collected
        self._slots: asyncio.Semaphore | None = None   # created lazily on the serving event loop

    async def _run(self, job: Job, fn: JobFn) -> None:
        self._slots = self._slots or asyncio.Semaphore(self._concurrency)
        request_id.set(f"job:{job.id}")  # every log line of this job carries its id
        if self._slots.locked():
            log.info("%s job queued (%d slots busy)", job.kind, self._concurrency)
        async with self._slots:
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
        self._evict()

    def _evict(self) -> None:
        finished = [jid for jid, j in self._jobs.items() if j.status != "running"]
        for jid in finished[:max(0, len(finished) - self._keep_finished)]:   # dicts keep insertion order
            del self._jobs[jid]

    def start(self, kind: str, fn: JobFn) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, status="running", stage="queued", progress=0.0, log=[])
        self._jobs[job.id] = job
        task = asyncio.create_task(self._run(job, fn), name=f"job-{kind}-{job.id}")
        log.info("created %s job %s", kind, job.id)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    def get(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        return job.model_copy(deep=True) if job else None

    async def shutdown(self) -> None:
        """Cancel running jobs on server shutdown."""
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
