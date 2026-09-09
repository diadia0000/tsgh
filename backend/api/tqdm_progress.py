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
    """Hook tqdm.__init__/update/close once. Idempotent."""
    global _installed
    if _installed:
        return
    orig_init, orig_update, orig_close = tqdm.__init__, tqdm.update, tqdm.close

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if not self.disable:
            _publish(self)

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

    tqdm.__init__, tqdm.update, tqdm.close = __init__, update, close
    _installed = True
