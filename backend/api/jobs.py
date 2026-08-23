"""In-memory job registry shared by all pipeline APIs (alignment, and future
hybrid) for their long-running steps.

Deliberately not a queue/worker system: single-user, single-machine, and the
dataflow contract (docs/UI/05-dataflow-api-contract.md) calls for
BackgroundTasks + polling, not a hand-rolled WebSocket.
"""
import uuid
from contextvars import ContextVar
from typing import Callable, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.schemas.common import JobProgress, JobStatus

router = APIRouter()

_UNFINISHED = ("pending", "running")

_jobs: Dict[str, dict] = {}

# The job whose `run` is executing on this thread. Set by `_execute` so code
# deep inside a pipeline call can report progress without the job id being
# threaded down through every function it passes through -- which would mean
# editing the algorithm layer, and progress reporting is not worth that.
_current_job: ContextVar[Optional[str]] = ContextVar("current_job", default=None)


def current_job_id() -> Optional[str]:
    """The job running on this thread, or None outside a background job."""
    return _current_job.get()


def active_job(key: str) -> Optional[str]:
    """job_id of the still-unfinished job registered under `key`, if any."""
    return next(
        (jid for jid, j in _jobs.items() if j["key"] == key and j["status"] in _UNFINISHED),
        None,
    )


def set_progress(job_id: str, phase: str, done: int, total: int, unit_label: str) -> None:
    """Publish how far `job_id` has got, for /api/jobs/{id} to hand back.

    Optional by design: a job that never calls this reports progress=None and
    its panel falls back to estimating. Silently ignores an unknown job_id --
    a progress update is never worth failing a running pipeline over.
    """
    job = _jobs.get(job_id)
    if job is None:
        return
    job["progress"] = JobProgress(phase=phase, done=done, total=total, unit_label=unit_label)


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
        "progress": None,
    }

    def _execute() -> None:
        _jobs[job_id]["status"] = "running"
        token = _current_job.set(job_id)
        try:
            result_path, metadata = run()
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result_path"] = result_path
            _jobs[job_id]["metadata"] = metadata
        except Exception as e:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)
        finally:
            _current_job.reset(token)

    background_tasks.add_task(_execute)
    return job_id


@router.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatus(job_id=job_id, **job)
