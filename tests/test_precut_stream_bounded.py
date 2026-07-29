"""Tests for `PrecutStream.__iter__`'s scheduling (doc 36 step 2).

`__iter__` used to submit **every** position to its thread pool before yielding
the first tile. On a 27,565-tile slide that is 27,565 queued cut jobs, and the
consumer's own abort has no effect on them: `_run_tiles_multiprocess._feed`
breaks out of the loop on fail-fast, the generator is abandoned mid-`with`, and
`ThreadPoolExecutor.__exit__` (or, later, `concurrent.futures`' interpreter-exit
hook) then blocks until the entire remaining queue has been cut to disk. That is
doc 35 §6.2's "correct fail-fast, then the parent sat alive for 49 minutes":
the process was not idle, it was finishing a precut nobody wanted.

The properties pinned here are the two that make an abandoned stream cheap
without changing what a fully-drained stream produces:

  1. **Abandoning the generator returns promptly**, having cut far fewer tiles
     than the grid holds.
  2. **A drained stream still yields every position exactly once**, and a cut
     failure still surfaces to the consumer (fail-fast unchanged).
"""
from __future__ import annotations

import time

from m0_module.m0_reader import PrecutStream


def _stream(n_positions: int, workers: int = 4) -> PrecutStream:
    """A PrecutStream with a grid but no images.

    `__init__` exists to open the two slides and derive the grid from their
    headers; these tests are about `__iter__`'s scheduling, so the grid is
    injected directly and `_cut` is replaced per test.
    """
    s = object.__new__(PrecutStream)
    s.positions = [(x, 0) for x in range(n_positions)]
    s.workers = workers
    s.tile_size = 1024
    return s


def test_abandoning_the_stream_does_not_cut_the_whole_slide(monkeypatch):
    """The consumer takes three tiles and gives up; the cutter must stop too."""
    cut: list = []

    def slow_cut(self, pos):
        time.sleep(0.02)
        cut.append(pos)
        return (f"ihc{pos}", f"dish{pos}", pos)

    monkeypatch.setattr(PrecutStream, "_cut", slow_cut)

    s = _stream(500)
    gen = iter(s)
    for _ in range(3):
        next(gen)
    t0 = time.perf_counter()
    gen.close()
    closed_in = time.perf_counter() - t0

    # 500 positions x 20 ms / 4 threads = 2.5 s if the whole grid is still queued.
    assert closed_in < 0.5, f"abandoning the stream took {closed_in:.2f}s"
    assert len(cut) < 100, f"{len(cut)}/500 tiles cut after taking 3"


def test_drained_stream_yields_every_position_once(monkeypatch):
    monkeypatch.setattr(PrecutStream, "_cut",
                        lambda self, pos: (f"ihc{pos}", f"dish{pos}", pos))
    s = _stream(37)
    got = [item[2] for item in s]
    assert sorted(got) == sorted(s.positions)
    assert len(got) == len(set(got))


def test_cut_failure_surfaces_to_the_consumer(monkeypatch):
    """A tile that cannot be cut must raise out of the iteration, not be skipped
    -- the consumer's fail-fast is the only thing standing between a bad cut and
    a slide with an undocumented hole."""
    def boom(self, pos):
        if pos == (5, 0):
            raise OSError("simulated cut failure")
        return (f"ihc{pos}", f"dish{pos}", pos)

    monkeypatch.setattr(PrecutStream, "_cut", boom)

    s = _stream(20, workers=2)
    raised = None
    try:
        for _ in s:
            pass
    except OSError as exc:
        raised = exc
    assert raised is not None and "simulated cut failure" in str(raised)
