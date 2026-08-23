"""Turns the hybrid pipeline's per-tile log line into job progress.

The pipeline has no progress callback and `backend/algorithms/**` is not ours
to change, but it already announces every tile it starts:

    logger.info("[%d/%d] 處理 tile: %s", idx, remaining, dish_path.stem)

A logging handler is the observation point the logging module exists to
provide, so this reads that record -- `record.args`, never the formatted
string, so nothing here depends on how the message reads in any language --
and republishes it through the job registry.

**This couples to one log statement in `hybrid_pipeline.py`.** If that line's
format string changes, `_TILE_MSG` stops matching and progress simply goes
quiet: `/api/jobs/{id}` reports progress=None and the panel falls back to its
time estimate. Degrading to "no number" is the intended failure -- a stale
number would be worse. `tests/test_hybrid_progress.py` fails loudly if the
pipeline's message and this constant drift apart, so the breakage surfaces in
CI rather than as a bar that never moves.

Only the analysis loop is counted. The global merge and the overlay stitch
that follow it emit nothing until they finish, so there is no honest number to
publish for them; the panel presents that stretch as an unquantified 收尾中.
"""
import logging

from backend.api.jobs import current_job_id, set_progress

# The exact format string in hybrid_pipeline.py's per-tile log call. Matched by
# identity of the *template*, not the rendered text.
_TILE_MSG = "[%d/%d] 處理 tile: %s"

_PIPELINE_LOGGER = "backend.algorithms.hybrid.hybrid_pipeline"

PHASE_ANALYZE = "analyze"
UNIT_TILE = "塊"


class _TileProgressHandler(logging.Handler):
    """Republishes the pipeline's `[idx/total]` tile line as job progress."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.msg != _TILE_MSG or not record.args:
            return
        job_id = current_job_id()
        if job_id is None:
            return  # a CLI run, or a call outside any background job
        try:
            done, total = int(record.args[0]), int(record.args[1])
        except (TypeError, ValueError, IndexError):
            return
        set_progress(job_id, PHASE_ANALYZE, done, total, UNIT_TILE)


_installed = False


def install() -> None:
    """Attach the handler to the pipeline logger and make sure it can fire.

    Idempotent. Raises that one logger to INFO when something above it has
    filtered INFO out, because otherwise there is nothing to observe:

    `hybrid_pipeline.py` calls `logging.basicConfig(level=INFO)` at import, but
    basicConfig is a **no-op when the root logger already has handlers** -- and
    under uvicorn it always does, because uvicorn configures logging before it
    imports the app. The pipeline logger then inherits root's WARNING, so
    `logger.info(...)` never creates a record and the handler above never sees
    one. Under the CLI (root unconfigured) basicConfig does apply, which is why
    this only ever broke when served through the API.

    The visible consequence is that the pipeline's INFO lines now also reach
    whatever handlers root has -- i.e. they appear in the backend log, where
    previously they were silently dropped. That is the same output the CLI has
    always produced, and it is per tile, so a whole-slide run is verbose.
    """
    global _installed
    if _installed:
        return
    logger = logging.getLogger(_PIPELINE_LOGGER)
    logger.addHandler(_TileProgressHandler())
    if not logger.isEnabledFor(logging.INFO):
        logger.setLevel(logging.INFO)
    _installed = True
