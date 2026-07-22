"""In-memory job registry shared by all pipeline APIs (alignment, and future
hybrid) for their long-running steps.

Deliberately not a queue/worker system: single-user, single-machine, and the
dataflow contract (docs/UI/05-dataflow-api-contract.md) calls for
BackgroundTasks + polling, not a hand-rolled WebSocket.
"""
import uuid
from typing import Callable, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.schemas.common import JobStatus

router = APIRouter()

_UNFINISHED = ("pending", "running")

_jobs: Dict[str, dict] = {}


def active_job(key: str) -> Optional[str]:
    """job_id of the still-unfinished job registered under `key`, if any."""
    return next(
        (jid for jid, j in _jobs.items() if j["key"] == key and j["status"] in _UNFINISHED),
        None,
    )


def submit_job(
    background_tasks: BackgroundTasks,
    run: Callable[[], tuple[str, dict]],
    key: Optional[str] = None,
) -> str:
    """Register a job and schedule `run` to execute in the background.

    `run` must return (result_path, metadata) on success.

    `key` makes submission idempotent: while a job for the same key is still
    unfinished, its id is returned instead of starting a second one. Alignment
    passes the run_id, whose steps all write the same output directory -- two of
    them in flight at once would corrupt each other's artifacts.
    """
    # ponytail: keyed per run, not per step, so two browser tabs driving one run
    # share a job. Key on f"{run_id}:{step}" if steps ever become concurrent.
    if key is not None:
        existing = active_job(key)
        if existing is not None:
            return existing

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",
        "result_path": None,
        "metadata": None,
        "error": None,
        "key": key,
    }

    def _execute() -> None:
        _jobs[job_id]["status"] = "running"
        try:
            result_path, metadata = run()
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result_path"] = result_path
            _jobs[job_id]["metadata"] = metadata
        except Exception as e:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)

    background_tasks.add_task(_execute)
    return job_id


@router.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatus(job_id=job_id, **job)
