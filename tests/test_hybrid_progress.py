"""Guards the one fragile joint in hybrid progress reporting.

`backend/api/hybrid_progress.py` reads the analysis loop's per-tile log record
to produce a progress number, because the pipeline exposes no callback and
`backend/algorithms/**` is not the API layer's to edit. That works, but it
means a constant in the API layer has to stay equal to a format string in
`hybrid_pipeline.py`, and nothing in either file would notice them drifting
apart -- the bar would just never move, which is exactly the kind of silent
breakage a user reports as "the progress bar is broken" months later.

So: one test that the two strings are still the same, and one that a matching
record actually lands as progress on the job.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import BackgroundTasks

from backend.api import hybrid_progress, jobs
from backend.api.hybrid_progress import (
    PHASE_ANALYZE,
    UNIT_TILE,
    _TILE_MSG,
    _TileProgressHandler,
)

PIPELINE_SRC = (
    Path(__file__).resolve().parent.parent
    / "backend" / "algorithms" / "hybrid" / "hybrid_pipeline.py"
)


def test_tile_message_still_matches_the_pipeline():
    """The format string the handler matches must still exist in the pipeline.

    If this fails, hybrid_pipeline.py's per-tile log line was reworded. Update
    `_TILE_MSG` to the new wording -- and check the new call still passes
    (index, total) as its first two args, which is what the handler reads.
    """
    source = PIPELINE_SRC.read_text(encoding="utf-8")
    assert repr(_TILE_MSG)[1:-1] in source or _TILE_MSG in source, (
        f"{_TILE_MSG!r} no longer appears in {PIPELINE_SRC.name}; hybrid "
        "progress reporting is silently dead until _TILE_MSG is updated."
    )


def _emit(record_args) -> None:
    handler = _TileProgressHandler()
    handler.emit(
        logging.LogRecord(
            name="backend.algorithms.hybrid.hybrid_pipeline",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=_TILE_MSG,
            args=record_args,
            exc_info=None,
        )
    )


def test_matching_record_becomes_job_progress():
    job_id = jobs.submit_job(BackgroundTasks(), lambda: ("", {}), key="test-progress")
    token = jobs._current_job.set(job_id)
    try:
        _emit((7, 42, "tile_x1024_y2048"))
    finally:
        jobs._current_job.reset(token)

    progress = jobs.get_job(job_id).progress
    assert progress is not None
    assert (progress.phase, progress.done, progress.total, progress.unit_label) == (
        PHASE_ANALYZE, 7, 42, UNIT_TILE,
    )


def test_record_outside_a_job_is_ignored():
    """CLI runs log the same line with no job in context; that must not raise."""
    _emit((1, 2, "tile_x0_y0"))  # no _current_job set


def test_fires_through_the_real_logger_when_root_filters_info():
    """The case that actually shipped broken.

    Under uvicorn the root logger is already configured by the time the app is
    imported, so `hybrid_pipeline`'s own `logging.basicConfig(level=INFO)` is a
    no-op and the pipeline logger inherits root's WARNING. `logger.info(...)`
    then creates no record at all, and a handler cannot observe what was never
    emitted -- a whole run reported 0/12 with an empty backend log.

    The other tests here call `handler.emit()` directly, which bypasses the
    level check and so cannot see this. This one goes through the real logger,
    with root set up the way uvicorn leaves it.
    """
    root = logging.getLogger()
    pipe = logging.getLogger("backend.algorithms.hybrid.hybrid_pipeline")
    saved = (root.level, list(root.handlers), pipe.level, list(pipe.handlers))
    try:
        # uvicorn's shape: root has a handler (so basicConfig would no-op) and
        # sits above INFO, and the pipeline logger has no level of its own.
        root.handlers = [logging.NullHandler()]
        root.setLevel(logging.WARNING)
        pipe.setLevel(logging.NOTSET)
        pipe.handlers = []
        assert not pipe.isEnabledFor(logging.INFO), "precondition: INFO is filtered out"

        hybrid_progress._installed = False
        hybrid_progress.install()

        job_id = jobs.submit_job(BackgroundTasks(), lambda: ("", {}), key="test-uvicorn")
        token = jobs._current_job.set(job_id)
        try:
            pipe.info(_TILE_MSG, 5, 12, "tile_x0_y0")   # exactly what the pipeline calls
        finally:
            jobs._current_job.reset(token)

        progress = jobs.get_job(job_id).progress
        assert progress is not None, "install() did not make the tile line observable"
        assert (progress.done, progress.total) == (5, 12)
    finally:
        root.level, root.handlers, pipe.level, pipe.handlers = (
            saved[0], saved[1], saved[2], saved[3],
        )
        hybrid_progress._installed = False


def test_unrelated_record_is_ignored():
    job_id = jobs.submit_job(BackgroundTasks(), lambda: ("", {}), key="test-progress-2")
    token = jobs._current_job.set(job_id)
    try:
        handler = _TileProgressHandler()
        handler.emit(
            logging.LogRecord(
                name="backend.algorithms.hybrid.hybrid_pipeline",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg="批次完成 — run_id=%s",
                args=("abc123",),
                exc_info=None,
            )
        )
    finally:
        jobs._current_job.reset(token)

    assert jobs.get_job(job_id).progress is None
