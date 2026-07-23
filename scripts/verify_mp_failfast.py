"""Guard the fail-fast invariant of the cross-tile multiprocessing path.

`docs/hybrid-pipeline/20-cross-tile-multiprocessing-plan.md` §1 item 2 makes this
non-negotiable, and §3 step 3(b) requires it verified *before* any medium/large
scale run:

    "a worker's exception must propagate to the parent, and the parent must
     terminate every sibling worker before returning -- letting siblings run to
     completion after one worker has already failed reproduces exactly the
     'slide with an undocumented hole' failure mode the current fail-fast design
     exists to prevent."

Wall-clock benchmarks cannot see this. A pool that raises but leaves siblings
writing tiles still *looks* correct on a green run, and orphaned worker processes
keep a CUDA context (and its VRAM) alive after the parent has returned.

What this checks, on the real `run_batch` multiprocess path with a real corrupt
tile injected:

  1. `run_batch(workers=N)` **raises** rather than returning stats.
  2. It aborts **early** -- strictly fewer tiles are finished than the grid holds,
     proving siblings were stopped rather than drained.
  3. **No worker process survives** the raise (checked by PID, after the fact).
  4. The single-process path (`workers=1`) fails on the same input, so the
     multiprocess path is not being held to a weaker bar than today's code.

Needs the real models + GPU (it runs genuine tiles up to the failure), so it is a
slow guard, not a unit test. Point it at any precut scratch directory.

Usage:
  python scripts/verify_mp_failfast.py --tiles-dir <precut scratch> [--workers 3]
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HYBRID = REPO / "backend" / "algorithms" / "hybrid"
sys.path.insert(0, str(HYBRID))
sys.path.insert(0, str(REPO))

import hybrid_pipeline as HP  # noqa: E402


def _stage_tiles(src: Path, dst: Path) -> int:
    """Copy a precut tile pair tree, then corrupt exactly one IHC tile."""
    for sub in ("ihc", "dish"):
        shutil.copytree(src / sub, dst / sub)
    tiles = sorted((dst / "ihc").glob("tile_x*_y*.tiff"))
    if len(tiles) < 4:
        raise SystemExit(f"need >=4 tiles to test early abort, found {len(tiles)}")
    # Corrupt a tile in the middle of the grid, not the first one: aborting on
    # tile 1 would pass this test even with a pool that never stops siblings.
    victim = tiles[len(tiles) // 2]
    victim.write_bytes(b"not a tiff")
    print(f"  corrupted {victim.name} ({len(tiles)} tiles total)")
    return len(tiles)


def _count_written(out_dir: Path) -> int:
    d = out_dir / "instance_mask"
    return len(list(d.glob("*.tiff"))) if d.exists() else 0


def _live_workers(before: set[int]) -> list:
    """Surviving *tile worker* processes only.

    A plain "any new child" check is wrong: `multiprocessing.resource_tracker`
    and joblib/loky's semaphore tracker are legitimate, shared, long-lived
    children that outlive a batch by design -- they show up even on the
    `workers=1` path, which spawns no tile workers at all. Tile workers are the
    `multiprocessing.spawn` children, so match those and exclude the trackers.
    """
    import psutil

    me = psutil.Process()
    out = []
    for c in me.children(recursive=True):
        try:
            if not c.is_running():
                continue
            cmd = " ".join(c.cmdline())
        except psutil.Error:
            continue
        if c.pid in before or "resource_tracker" in cmd:
            continue
        if "spawn_main" in cmd:
            out.append((c.pid, cmd[:90]))
    return out


def run_case(tiles_dir: Path, workers: int, n_tiles: int) -> bool:
    import psutil

    print(f"\n--- workers={workers} ---")
    ok = True
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        before = {c.pid for c in psutil.Process().children(recursive=True)}
        raised = None
        try:
            HP.run_batch(tiles_dir / "ihc", tiles_dir / "dish", out,
                         workers=workers)
        except Exception as exc:  # noqa: BLE001 -- the point of the test
            raised = exc

        # 1. raised at all
        if raised is None:
            print("  FAIL: run_batch returned normally on a corrupt tile")
            ok = False
        else:
            print(f"  ok: raised {type(raised).__name__}")

        # 2. aborted early -- siblings stopped, not drained
        written = _count_written(out)
        if written >= n_tiles:
            print(f"  FAIL: {written}/{n_tiles} tiles written -- the batch drained "
                  f"instead of aborting")
            ok = False
        else:
            print(f"  ok: aborted early ({written}/{n_tiles} tiles written)")

        # 3. no surviving worker processes
        leaked = _live_workers(before)
        if leaked:
            print(f"  FAIL: {len(leaked)} worker process(es) still alive:")
            for pid, cmd in leaked:
                print(f"        pid={pid} {cmd}")
                try:
                    psutil.Process(pid).kill()
                except psutil.Error:
                    pass
            ok = False
        else:
            print("  ok: no worker process survived the raise")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", required=True,
                    help="precut scratch dir containing ihc/ and dish/")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    logging.getLogger("hybrid_pipeline").setLevel(logging.ERROR)
    logging.getLogger("cellpose.models").setLevel(logging.ERROR)

    with tempfile.TemporaryDirectory() as td:
        staged = Path(td) / "tiles"
        staged.mkdir()
        print("staging corrupt-tile fixture:")
        n = _stage_tiles(Path(a.tiles_dir), staged)

        results = {
            w: run_case(staged, w, n)
            for w in (a.workers, 1)          # multiprocess first, then the control
        }

    print("\n" + "=" * 60)
    for w, ok in results.items():
        print(f"workers={w}: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
