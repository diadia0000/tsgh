"""tqdm bars inside a reporting job must land as job progress -- and nowhere else.

`backend/api/tqdm_progress.py` hooks tqdm.__init__/update/close so the bars the
alignment modules, VALIS and the hybrid pipeline already keep become
`/api/jobs/{id}` counts. Two things can silently break: the hook stops firing
(bar never moves), or it fires for jobs that did not opt in.
"""
from __future__ import annotations

from fastapi import BackgroundTasks
from tqdm import tqdm

from backend.api import jobs, tqdm_progress


def _job() -> str:
    return jobs.submit_job(BackgroundTasks(), lambda: ("", {}))


def _run_bar_in_job(job_id: str, run):
    token = jobs._current_job.set(job_id)
    try:
        return run()
    finally:
        jobs._current_job.reset(token)


def test_bar_inside_reporting_job_publishes_counts():
    tqdm_progress.install()
    job_id = _job()

    def run():
        tqdm(total=4, desc="處理進度", unit="區塊").update(3)
        return jobs.get_job(job_id).progress, {}

    seen = _run_bar_in_job(job_id, tqdm_progress.reporting(run))[0]
    assert (seen.phase, seen.done, seen.total, seen.unit_label) == ("處理進度", 3, 4, "區塊")


def test_bar_publishes_its_denominator_on_creation():
    """A stage's total is known the moment its bar opens; a panel should get
    0/total then, not sit empty until the first unit lands."""
    tqdm_progress.install()
    job_id = _job()
    _run_bar_in_job(job_id, tqdm_progress.reporting(lambda: (tqdm(total=25, desc="分析", unit="塊"), {})))
    progress = jobs.get_job(job_id).progress
    assert (progress.phase, progress.done, progress.total, progress.unit_label) == ("分析", 0, 25, "塊")


def test_close_publishes_the_final_count():
    """`for x in tqdm(...)` skips update() inside mininterval, so the last items
    of a loop only land when the bar closes -- that must still be published."""
    tqdm_progress.install()
    job_id = _job()

    def run():
        for _ in tqdm(range(4), desc="處理進度", mininterval=3600):
            pass
        return "", {}

    _run_bar_in_job(job_id, tqdm_progress.reporting(run))
    progress = jobs.get_job(job_id).progress
    assert (progress.done, progress.total) == (4, 4)


def test_default_unit_is_dropped():
    tqdm_progress.install()
    job_id = _job()
    _run_bar_in_job(job_id, tqdm_progress.reporting(lambda: (tqdm(total=2, desc="x").update(), {})))
    assert jobs.get_job(job_id).progress.unit_label == ""


def test_bar_in_job_that_did_not_opt_in_is_ignored():
    """A job that never opted in keeps progress=None, whatever bars the
    libraries it calls happen to open."""
    tqdm_progress.install()
    job_id = _job()
    _run_bar_in_job(job_id, lambda: tqdm(total=2, desc="cellpose").update())
    assert jobs.get_job(job_id).progress is None


def test_bar_outside_any_job_is_harmless():
    tqdm_progress.install()
    for _ in tqdm(range(3), desc="cli"):
        pass
