"""Worker-side instrumentation hook for `run_batch(workers>1)` (DISCOVERED #40).

`perf_measure.py` instruments by monkeypatching the **parent** process's
`hybrid_pipeline` namespace. `spawn`ed workers re-import that module and get the
clean originals, so at `workers>1` every per-stage bucket (B1/B2/B3/...) came back
empty and the only honest number was end-to-end wall — you could see that four
workers were 2.06x faster but not *which stage* had changed.

This module is what `hybrid_pipeline._install_worker_probe()` loads when
`HYBRID_MP_WORKER_PROBE=mp_worker_probe:install` is set. It reuses
`perf_measure.install_wrappers()` verbatim so parent-side and worker-side buckets
are produced by exactly the same shims and are directly comparable; importing
`perf_measure` as a module (rather than as `__main__`) runs no measurement of its
own, since everything there is behind `if __name__ == "__main__"`.

Usage — the harness sets this up, you do not normally call it by hand:

    HYBRID_MP_WORKER_PROBE=mp_worker_probe:install \\
    PYTHONPATH=scripts \\
    .venv/bin/python scripts/perf_measure.py ... --mp-workers 4
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import perf_measure  # noqa: E402


def install():
    """Install the timing shims in this worker; return a collector for its TIMINGS.

    `hybrid_pipeline` calls this once per worker *before* the three `_init_*` model
    loads, so `init_unet` / `init_cellpose_*` are measured per worker too — that is
    how the `~3.14s x N` init claim in doc 20 §5 becomes checkable rather than
    inferred.
    """
    perf_measure.install_wrappers()
    return lambda: perf_measure.TIMINGS
