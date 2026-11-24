import asyncio
import json

from fastapi import APIRouter, HTTPException

from ..dependencies import CollectionDep, ContainerDep, JobsDep
from ..jobs import JobHandle
from ..schemas import Job

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/results")
async def eval_results(app: ContainerDep):
    path = app.evaluator().results_json
    if not path.exists():
        raise HTTPException(404, "no evaluation results yet")
    return json.loads(await asyncio.to_thread(path.read_text, encoding="utf-8"))


@router.post("/run", response_model=Job, status_code=202)
async def eval_run(app: ContainerDep, jobs: JobsDep, collection: CollectionDep, rerank: bool = False):
    """The golden set's labels belong to the default collection unless another is given."""
    evaluator = app.evaluator(collection)

    async def work(h: JobHandle) -> dict:
        h.stage(f"Evaluating all configurations on the golden set (collection '{collection}')", 0.1)
        report = await evaluator.run(rerank, log=h.log)
        return {"configurations": list(report)}

    return jobs.start("eval", work)
