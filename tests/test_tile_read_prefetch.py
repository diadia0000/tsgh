"""Tests for the one-tile-ahead read prefetch (doc 30 Option L, round 9).

`prefetch_tile_reads` moves each tile's two `_read_rgb` calls off the blocking
position they occupied at the top of `_process_precut_tile_gpu` and onto a
background thread, one tile ahead, so a tile's disk read overlaps the *previous*
tile's GPU forward.

Two properties carry real risk and neither is visible in a wall-clock number:

  1. **Every tile is yielded, exactly once, in order.** The prefetch is a
     pure reordering of *when bytes are decoded*, never of which tiles run.
  2. **A read failure still fail-fasts, at the point the tile is consumed.**
     Before Option L the read raised inside `_process_precut_tile_gpu`, which
     logged and returned `None`, and `run_batch` turned that into a batch abort.
     A prefetched read fails on another thread, one tile early -- if that
     exception were swallowed, or surfaced against the wrong tile, the batch
     would either continue past a hole or blame an innocent tile. Doc 30 §3
     calls this out as a correctness requirement of the design, not an
     implementation detail.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from m0_module import m0_tile_runner as TR


def _tiles(n: int):
    return [
        (Path(f"/ihc/tile_x{i}_y0.tiff"), Path(f"/dish/tile_x{i}_y0.tiff"), (i, 0))
        for i in range(n)
    ]


def _drain(tiles, depth=1):
    with ThreadPoolExecutor(max_workers=1) as pool:
        return [(item, fut) for item, fut in TR.prefetch_tile_reads(tiles, pool, depth)]


def test_yields_every_tile_once_in_order(monkeypatch):
    monkeypatch.setattr(TR, "_read_tile_pair", lambda i, d: (str(i), str(d)))
    tiles = _tiles(5)
    got = _drain(tiles)
    assert [item for item, _ in got] == tiles


def test_empty_stream_yields_nothing(monkeypatch):
    monkeypatch.setattr(TR, "_read_tile_pair", lambda i, d: (str(i), str(d)))
    assert _drain([]) == []


def test_single_tile_stream(monkeypatch):
    """Depth 1 must not swallow the only tile when the stream ends before the
    read-ahead window ever fills."""
    monkeypatch.setattr(TR, "_read_tile_pair", lambda i, d: (str(i), str(d)))
    tiles = _tiles(1)
    assert [item for item, _ in _drain(tiles)] == tiles


def test_read_is_issued_ahead_of_consumption(monkeypatch):
    """The whole point: tile N+1's read must already be submitted when tile N is
    handed over, otherwise nothing overlaps and the change buys nothing.

    Asserted on *submission* order, not execution order -- the pool is a real
    thread pool, so when a queued task actually runs is not deterministic.
    """
    monkeypatch.setattr(TR, "_read_tile_pair", lambda i, d: (str(i), str(d)))
    submitted: list[str] = []

    class RecordingPool:
        def __init__(self, inner):
            self._inner = inner

        def submit(self, fn, ihc, dish):
            submitted.append(ihc.name)
            return self._inner.submit(fn, ihc, dish)

    tiles = _tiles(4)
    with ThreadPoolExecutor(max_workers=1) as inner:
        gen = TR.prefetch_tile_reads(tiles, RecordingPool(inner), depth=1)
        first_item, _first_fut = next(gen)
        assert first_item == tiles[0]
        # tile 1's read was submitted before tile 0 was ever yielded
        assert submitted == ["tile_x0_y0.tiff", "tile_x1_y0.tiff"]
        list(gen)


def test_read_failure_surfaces_at_consumption(monkeypatch):
    """A failing read must re-raise out of `.result()` for the tile it belongs to,
    so run_batch's existing fail-fast path still triggers."""
    def boom(ihc, dish):
        if "tile_x2_" in ihc.name:
            raise OSError("simulated unreadable tile")
        return (str(ihc), str(dish))

    monkeypatch.setattr(TR, "_read_tile_pair", boom)
    got = _drain(_tiles(4))
    assert len(got) == 4
    for item, fut in got:
        if item[2] == (2, 0):
            with pytest.raises(OSError, match="simulated unreadable tile"):
                fut.result()
        else:
            assert fut.result() == (str(item[0]), str(item[1]))


def test_preread_bypasses_the_inline_read(monkeypatch):
    """`_process_precut_tile_gpu(preread=...)` must not touch disk; without it the
    prefetch would double every read instead of moving it."""
    called: list[Path] = []
    monkeypatch.setattr(TR, "_read_rgb", lambda p: called.append(p))

    class _Boom(Exception):
        pass

    def _stop(*a, **k):
        raise _Boom

    # Stop right after the read decision so no GPU/model is needed. The shape check
    # between the two arrays runs first, so preread must be real arrays.
    monkeypatch.setattr(TR, "core_crop_bounds", _stop)
    pair = (np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(_Boom):
        TR._process_precut_tile_gpu(
            Path("/ihc/a.tiff"), Path("/dish/a.tiff"), 0, 0,
            geometry=None, unet_inferencer=None, cellpose_segmenter=None,
            dish_cellpose_segmenter=None, output_dir=Path("/out"),
            preread=pair,
        )
    assert called == []
