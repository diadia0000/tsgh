"""Isolate the two filesystem-side costs doc 24 raised, at P concurrent processes.

Candidate F (doc 24 §2.5): every background tile writes the same six placeholder
files a real tile writes, via the same `_save_tile_array`/`skimage.io.imsave` path.
Never separately measured, because at the 14%-background crops rounds 1-6 used it was
invisible in aggregate wall.

Candidate G (doc 24 §2.6): `_save_tile_array` ran `mkdir(parents=True, exist_ok=True)`
before *every* write -- six syscalls per tile against six fixed directories that exist
after the first tile, and under `workers>1` those syscalls land on the same directory
inodes from N processes at once. Only a concurrent probe can show whether that
contention is real; a single-process microbenchmark cannot.

The probe calls the pipeline's own `_write_blank_tile`, so what it times is the real
code, not a reimplementation. Three variants per iteration:

  mkdir      -- the six redundant mkdir calls alone (the round-6 hot-path cost)
  blank      -- one `_write_blank_tile` (six encodes)
  hardlink   -- the doc 24 §2.5 alternative: encode one blank tile set once, then
                `os.link` it per tile

All P processes share ONE output directory tree, as the real workers do, and start
their timed window on a barrier so the measured window is genuinely concurrent.

Usage:
  .venv/bin/python scripts/fs_write_probe.py --procs 1,4 --iters 60 --out out.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

HYBRID = Path(__file__).resolve().parent.parent / "backend" / "algorithms" / "hybrid"
sys.path.insert(0, str(HYBRID))
sys.path.insert(0, str(HYBRID.parent.parent.parent))

import hybrid_pipeline as HP  # noqa: E402
from config import config  # noqa: E402

TILE = 1024
CROP = 768                      # a core crop on the interior of the grid
# The six fixed per-tile output directories `_save_tile_array` writes into. Listed here
# rather than imported, because the pipeline (correctly, per doc 25 §3.5) has no such
# constant: it mkdirs per call, and the hoist that would have introduced one was
# stop-lossed on measurement.
SUBDIRS = (
    "core_mask", "masked_ihc", "dish_mask_overlay",
    "instance_mask", "dish_nucleus_mask", "overlay_annotated",
)
BLANK_SUFFIX = {
    "core_mask": ".png", "masked_ihc": ".png", "dish_mask_overlay": ".png",
    "instance_mask": ".tiff", "dish_nucleus_mask": ".tiff",
    "overlay_annotated": ".tiff",
}


def _child(rank: int, out_dir: str, iters: int, barrier, q) -> None:
    out = Path(out_dir)
    for name in SUBDIRS:
        (out / name).mkdir(parents=True, exist_ok=True)

    # warm: one full blank tile set, also the template the hardlink variant reuses
    HP._write_blank_tile(out, f"tmpl_{rank}", TILE, TILE, (CROP, CROP))

    barrier.wait()
    t0 = time.perf_counter()
    for i in range(iters):
        for name in SUBDIRS:
            (out / name).mkdir(parents=True, exist_ok=True)
    t_mkdir = time.perf_counter() - t0

    barrier.wait()
    t0 = time.perf_counter()
    for i in range(iters):
        HP._write_blank_tile(out, f"blank_{rank}_{i}", TILE, TILE, (CROP, CROP))
    t_blank = time.perf_counter() - t0

    barrier.wait()
    t0 = time.perf_counter()
    for i in range(iters):
        for name in SUBDIRS:
            suffix = BLANK_SUFFIX[name]
            src = out / name / f"tmpl_{rank}{suffix}"
            dst = out / name / f"link_{rank}_{i}{suffix}"
            os.link(src, dst)
    t_link = time.perf_counter() - t0

    q.put({
        "rank": rank,
        "mkdir_us_per_call": round(t_mkdir / (iters * len(SUBDIRS)) * 1e6, 3),
        "mkdir_ms_per_tile": round(t_mkdir / iters * 1e3, 4),
        "blank_ms_per_tile": round(t_blank / iters * 1e3, 3),
        "hardlink_ms_per_tile": round(t_link / iters * 1e3, 4),
    })


def run(procs: int, iters: int, root: Path) -> dict:
    out_dir = root / f"p{procs}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(procs)
    q = ctx.Queue()
    ps = [
        ctx.Process(target=_child, args=(r, str(out_dir), iters, barrier, q))
        for r in range(procs)
    ]
    for p in ps:
        p.start()
    rows = [q.get() for _ in ps]
    for p in ps:
        p.join()

    n = len(rows)
    agg = {
        "procs": procs,
        "iters_per_proc": iters,
        "per_proc": sorted(rows, key=lambda r: r["rank"]),
    }
    for k in ("mkdir_us_per_call", "mkdir_ms_per_tile",
              "blank_ms_per_tile", "hardlink_ms_per_tile"):
        agg[k + "_mean"] = round(sum(r[k] for r in rows) / n, 4)
    size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    agg["bytes_written"] = size
    shutil.rmtree(out_dir)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", default="1,4")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--root", default=None,
                    help="where to write (default: config.output_dir/_fs_probe)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(config.output_dir) / "_fs_probe"
    root.mkdir(parents=True, exist_ok=True)

    result = {"root": str(root), "tile": TILE, "crop": CROP,
              "background_fill_value": config.background_fill_value, "runs": []}
    for procs in [int(p) for p in args.procs.split(",")]:
        agg = run(procs, args.iters, root)
        result["runs"].append(agg)
        print(f"P={procs:2d}  mkdir {agg['mkdir_us_per_call_mean']:7.2f} us/call "
              f"({agg['mkdir_ms_per_tile_mean']:.3f} ms/tile)   "
              f"blank {agg['blank_ms_per_tile_mean']:7.2f} ms/tile   "
              f"hardlink {agg['hardlink_ms_per_tile_mean']:.3f} ms/tile")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
