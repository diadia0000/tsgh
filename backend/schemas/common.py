"""Request/response models shared across all pipeline APIs (alignment, and
future hybrid). Kept separate from any one pipeline's schema file so adding a
second pipeline doesn't require reaching into another pipeline's module.
"""
from typing import Optional

from pydantic import BaseModel


class JobAccepted(BaseModel):
    job_id: str


class JobProgress(BaseModel):
    """How far a long job has got, when the job knows. Absent (None on
    JobStatus) whenever nothing is publishing it -- the alignment pipeline
    reports only step transitions, and its panel estimates from those.

    `done`/`total` count the units named by `unit_label`; `phase` says which
    stretch of the run is executing, because a pipeline's last phase can be
    long and have no counter of its own. A client must treat done == total as
    "this phase's counter is exhausted", not as "the job is finished" -- only
    status == "done" means that.
    """
    phase: str
    done: int
    total: int
    # What one unit is, in the user's words ("塊"), so the UI does not have to
    # keep its own table of phase -> noun.
    unit_label: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    result_path: Optional[str] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None
    # Idempotency key the job was submitted under (alignment sends the run_id).
    key: Optional[str] = None
    # None unless the running job publishes counts (hybrid analysis does).
    progress: Optional[JobProgress] = None
