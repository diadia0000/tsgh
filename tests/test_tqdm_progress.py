"""tqdm bars inside an alignment job must land as job progress -- and nowhere else.

`backend/api/tqdm_progress.py` hooks tqdm.update/close so the counters the
alignment modules and VALIS already keep become `/api/jobs/{id}` counts. Two
things can silently break: the hook stops firing (bar never moves), or it fires
for jobs that did not opt in and tramples the hybrid pipeline's tile counter.
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
    """A hybrid job runs cellpose, which has tqdm bars of its own; they must not
    overwrite the tile counter hybrid_progress publishes."""
    tqdm_progress.install()
    job_id = _job()
    _run_bar_in_job(job_id, lambda: tqdm(total=2, desc="cellpose").update())
    assert jobs.get_job(job_id).progress is None


def test_bar_outside_any_job_is_harmless():
    tqdm_progress.install()
    for _ in tqdm(range(3), desc="cli"):
        pass
