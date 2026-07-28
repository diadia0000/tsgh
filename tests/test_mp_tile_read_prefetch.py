"""Tests for the one-tile-ahead read prefetch on the **multiprocess** path
(doc 33 follow-up 3 / doc 34 step 2).

`tests/test_tile_read_prefetch.py` pins `prefetch_tile_reads` itself and its wiring
into `run_batch`'s single-process loop. That loop is not what production runs:
`workers=4` is the shipped recommendation, and until this change `_mp_tile_worker`
read each tile inline on its own main thread, so none of Option L's benefit reached
the deployed configuration.

The worker loop is a different function from `run_batch`'s, with a different error
path (a `None` chunk becomes a `("error", ...)` message on the result queue rather
than a raise), so the two properties doc 30 §3 calls correctness requirements have to
be re-pinned *here* rather than inherited:

  1. **Every tile is processed exactly once, in order, and the read is prefetched.**
     Reading one tile ahead means the worker also pulls one task ahead off the shared
     queue; that must not drop a tile or process one twice.
  2. **A read failure still fail-fasts, against its own tile.** A prefetched read
     fails on another thread one tile early -- if the exception were swallowed the
     batch would ship a slide with an undocumented hole, and if it surfaced against
     the wrong tile the message would blame an innocent tile.

The ablation switch is pinned too: `HYBRID_MP_NO_PREFETCH=1` is the control arm the
`workers=4` measurement compares against, and a control arm that silently failed to
take effect would make the two arms identical and the measurement meaningless
(playbook anti-pattern #3).
"""
from __future__ import annotations

import queue
from pathlib import Path

import pytest

from m0_module import m0_multiprocess as MP
from m0_module import m0_tile_runner as TR


# ------------------------------------------------------------------
# _drain_task_queue -- the queue-to-iterable adapter
# ------------------------------------------------------------------

def test_drain_yields_prefetch_shape_and_stops_at_poison():
    q = queue.Queue()
    for i in range(3):
        q.put((Path(f"/ihc/t{i}.tiff"), Path(f"/dish/t{i}.tiff"), i, 0))
    q.put(MP._MP_POISON)
    q.put((Path("/ihc/never.tiff"), Path("/dish/never.tiff"), 9, 9))

    got = list(MP._drain_task_queue(q))
    assert got == [
        (Path(f"/ihc/t{i}.tiff"), Path(f"/dish/t{i}.tiff"), (i, 0)) for i in range(3)
    ]
    # the poison ends this worker; anything queued after it is another worker's problem
    assert q.qsize() == 1


# ------------------------------------------------------------------
# _inline_tile_reads -- the control arm, same generator contract
# ------------------------------------------------------------------

def _tiles(n):
    return [(Path(f"/ihc/t{i}.tiff"), Path(f"/dish/t{i}.tiff"), (i, 0)) for i in range(n)]


def test_inline_reads_match_the_prefetch_contract(monkeypatch):
    monkeypatch.setattr(MP, "_read_tile_pair", lambda i, d: (str(i), str(d)))
    got = list(MP._inline_tile_reads(_tiles(4), pool=None))
    assert [item for item, _ in got] == _tiles(4)
    assert [fut.result() for _, fut in got] == [
        (str(i), str(d)) for i, d, _ in _tiles(4)
    ]


def test_inline_read_failure_surfaces_at_consumption(monkeypatch):
    """The control arm must fail the same way as the prefetch arm, or the ablation
    would be comparing two different error semantics as well as two read schedules."""
    def boom(ihc, dish):
        if "t2" in ihc.name:
            raise OSError("simulated unreadable tile")
        return (str(ihc), str(dish))

    monkeypatch.setattr(MP, "_read_tile_pair", boom)
    got = list(MP._inline_tile_reads(_tiles(4), pool=None))
    assert len(got) == 4
    for item, fut in got:
        if item[2] == (2, 0):
            with pytest.raises(OSError, match="simulated unreadable tile"):
                fut.result()
        else:
            fut.result()


# ------------------------------------------------------------------
# _mp_tile_worker -- the real loop, with the models and GPU stubbed out
# ------------------------------------------------------------------

class _FakeTg:
    """Stands in for _TileGpuResult; only its identity matters to the loop."""

    def __init__(self, ax, ay):
        self.ax, self.ay = ax, ay


@pytest.fixture
def worker_env(monkeypatch):
    """Stub the three model loads, the GPU stage and the CPU tail, and record calls.

    Everything the worker does around the read schedule is replaced so the test
    observes exactly one thing: which tile got which pixels, and when.
    """
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for name in ("_init_unet_inferencer", "_init_cellpose_segmenter",
                 "_init_dish_cellpose_segmenter"):
        monkeypatch.setattr(MP, name, lambda: object())
    monkeypatch.setattr(MP, "compute_config_hash", lambda cfg: "HASH")

    gpu_calls: list[tuple] = []

    def fake_gpu(ihc_path, dish_path, ax, ay, geometry, unet, cp, dcp, out,
                 preread=None):
        gpu_calls.append((ax, ay, preread))
        return _FakeTg(ax, ay)

    monkeypatch.setattr(MP, "_process_precut_tile_gpu", fake_gpu)
    monkeypatch.setattr(MP, "_process_precut_tile_cpu",
                        lambda tg, geometry, out, merge: [(tg.ax, tg.ay)])
    monkeypatch.setattr(MP, "_read_tile_pair",
                        lambda i, d: (f"px:{i.name}", f"px:{d.name}"))
    monkeypatch.setattr(TR, "_read_tile_pair",
                        lambda i, d: (f"px:{i.name}", f"px:{d.name}"))
    return gpu_calls


def _run_worker(n_tiles: int):
    task_q, result_q = queue.Queue(), queue.Queue()
    for i in range(n_tiles):
        task_q.put((Path(f"/ihc/t{i}.tiff"), Path(f"/dish/t{i}.tiff"), i, 0))
    task_q.put(MP._MP_POISON)
    MP._mp_tile_worker(task_q, result_q, geometry=None, output_dir=Path("/out"),
                       merge_dir=None, parent_cfg_hash="HASH")
    msgs = []
    while not result_q.empty():
        msgs.append(result_q.get())
    return msgs


def test_worker_processes_every_tile_once_with_prefetched_pixels(worker_env, monkeypatch):
    monkeypatch.delenv(MP._MP_NO_PREFETCH_ENV, raising=False)
    msgs = _run_worker(4)

    assert msgs[0] == ("ready", None)
    assert [m[1][:2] for m in msgs[1:] if m[0] == "ok"] == [(i, 0) for i in range(4)]
    # every tile reached the GPU stage exactly once, in order, with pixels already read
    assert [(ax, ay) for ax, ay, _ in worker_env] == [(i, 0) for i in range(4)]
    for i, (_ax, _ay, preread) in enumerate(worker_env):
        assert preread == (f"px:t{i}.tiff", f"px:t{i}.tiff")


def test_worker_read_failure_fails_the_batch_for_its_own_tile(worker_env, monkeypatch):
    monkeypatch.delenv(MP._MP_NO_PREFETCH_ENV, raising=False)

    def boom(ihc, dish):
        if "t2" in ihc.name:
            raise OSError("simulated unreadable tile")
        return (f"px:{ihc.name}", f"px:{dish.name}")

    monkeypatch.setattr(TR, "_read_tile_pair", boom)
    msgs = _run_worker(4)

    errors = [m for m in msgs if m[0] == "error"]
    assert len(errors) == 1, "a failed read must abort the batch, not be skipped"
    assert "tile_x2_y0" in errors[0][1], "the failure must name the tile that failed"
    # tiles 0 and 1 completed; tile 2 never reached the GPU stage, and 3 never ran
    assert [(ax, ay) for ax, ay, _ in worker_env] == [(0, 0), (1, 0)]


def test_ablation_env_var_switches_to_inline_reads(worker_env, monkeypatch):
    """The `--no-prefetch` control arm must actually reach the spawned worker."""
    monkeypatch.setenv(MP._MP_NO_PREFETCH_ENV, "1")
    submitted: list[str] = []
    real = MP.prefetch_tile_reads

    def spy(tiles, pool, depth=1):
        submitted.append("prefetch")
        return real(tiles, pool, depth)

    monkeypatch.setattr(MP, "prefetch_tile_reads", spy)
    msgs = _run_worker(3)

    assert submitted == [], "control arm still went through the prefetch generator"
    assert [m[1][:2] for m in msgs if m[0] == "ok"] == [(i, 0) for i in range(3)]
    # the control arm still hands pixels over via `preread`; only the schedule differs
    for i, (_ax, _ay, preread) in enumerate(worker_env):
        assert preread == (f"px:t{i}.tiff", f"px:t{i}.tiff")
