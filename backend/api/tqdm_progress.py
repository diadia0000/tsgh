"""Republishes the tqdm bars a background job runs as that job's progress.

tqdm is already the alignment pipeline's progress primitive: module1 wraps its
strip pool in one, modules 3/4 count their stages with one, and VALIS wraps
every registration stage ("Converting images", "Finding rigid transforms", ...)
in one of its own. Hooking the class once, here, turns all of them into
`/api/jobs/{id}` counts -- the same shape as hybrid_progress: observe what the
pipeline already emits instead of threading a callback through
`backend/algorithms/**`.

Opt-in per job via `reporting(run)`, not global: cellpose emits tqdm bars too,
and inside a hybrid job they would overwrite the per-tile counter that
hybrid_progress publishes. Only bars on the job's own thread report (that is
where the job id contextvar lives); a bar inside a Pool worker publishes into
the forked copy of the registry and is never seen, which is the right outcome.
Outside any job (CLI runs, tests) the hook is a no-op.
"""
from contextvars import ContextVar
from typing import Callable

from tqdm import tqdm

from backend.api.jobs import current_job_id, set_progress

_reporting: ContextVar[bool] = ContextVar("tqdm_reporting", default=False)
_installed = False


def reporting(run: Callable[[], tuple[str, dict]]) -> Callable[[], tuple[str, dict]]:
    """Wrap a job's `run` so the tqdm bars inside it publish job progress."""
    def wrapped() -> tuple[str, dict]:
        token = _reporting.set(True)
        try:
            return run()
        finally:
            _reporting.reset(token)
    return wrapped


def _publish(bar: tqdm) -> None:
    job_id = current_job_id()
    if job_id is None or not _reporting.get() or not bar.total:
        return
    # tqdm's default unit "it" is noise on screen; a bar that named its unit keeps it.
    unit = bar.unit if bar.unit != "it" else ""
    set_progress(job_id, bar.desc or "", int(bar.n), int(bar.total), unit)


def install() -> None:
    """Hook tqdm.update/close once. Idempotent."""
    global _installed
    if _installed:
        return
    orig_update, orig_close = tqdm.update, tqdm.close

    def update(self, n=1):
        shown = orig_update(self, n)
        if not self.disable:
            _publish(self)
        return shown

    def close(self):
        # close() runs again from __del__; a second publish would overwrite a
        # newer bar's counts with this one's stale final state.
        was_open = not self.disable
        orig_close(self)
        if was_open:
            _publish(self)

    tqdm.update, tqdm.close = update, close
    _installed = True
