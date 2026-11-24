from fastapi import APIRouter, HTTPException

from ..dependencies import JobsDep
from ..schemas import Job

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=Job)
async def get_job(jobs: JobsDep, job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job
