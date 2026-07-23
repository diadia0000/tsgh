"""Step-1 probe for docs/hybrid-pipeline/20-cross-tile-multiprocessing-plan.md §3.1.

Answers the single question that gates Candidates B/C/D: **when N independent
`spawn`ed processes each open their own CUDA context on this GPU and run work
concurrently, do their kernels actually overlap in wall time, or do they
effectively serialize?**

Method -- throughput scaling, not a micro-benchmark. Every process does the
*same* fixed amount of work W. We compare the wall time of one process doing W
against N processes each doing W, with a `multiprocessing.Barrier` so the timed
sections genuinely coincide:

    speedup(N) = N * T(1) / T(N)

    speedup ~= N    -> full overlap (the device had idle capacity to fill)
    speedup ~= 1.0  -> full serialization (adding processes buys nothing)

Three workloads, because the answer depends on *why* the device is idle:

  `sm`      large matmuls -- SM-saturated. Control: a saturated device cannot
            overlap anything, so this should measure ~1.0 no matter what the
            driver does. If it does NOT measure ~1.0 the harness is wrong.
  `launch`  a Python loop of tiny kernels -- launch-bound, near-zero SM. This is
            the regime `18-...-implementation.md` §3 identified as the larger
            half of all device idle in the real pipeline (intra-forward,
            kernel-launch-bound Cellpose/SAM loops). Upper bound on what
            multiprocessing could recover.
  `models`  the real thing: `_process_precut_tile_gpu` over real precut tiles
            with all 3 real models loaded per process. The decisive number.

Also records what §5 lists as an open prerequisite: **per-process CUDA context
memory overhead** (device free-memory delta across `torch.cuda.init()`, before
any weights are loaded) and per-process total VRAM after model init.

Nothing here imports or modifies pipeline *control flow*; `models` mode calls
the existing per-tile GPU entry point unchanged.

Usage:
  python scripts/mp_concurrency_probe.py --workload launch --procs 1,2,3,4
  python scripts/mp_concurrency_probe.py --workload models --procs 1,2,3,4 \
      --tiles-dir <precut scratch> --out <metrics dir>
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

HYBRID = Path(__file__).resolve().parent.parent / "backend" / "algorithms" / "hybrid"


# ------------------------------------------------------------------
# workloads -- each runs entirely inside a child process
# ------------------------------------------------------------------
def _work_sm(units: int) -> None:
    """SM-saturating control: large matmuls, device kept busy by one process."""
    import torch

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(units):
        for _ in range(20):
            a = (a @ b) * 0.001
    torch.cuda.synchronize()


def _work_launch(units: int) -> None:
    """Launch-bound control: many tiny kernels, device mostly idle per process.

    Mirrors the shape of the Cellpose/SAM internal Python loops that
    `gil-contention-diag.md` traced: real kernels, but so small that the CPU
    cannot enqueue them fast enough to keep the SMs busy.
    """
    import torch

    x = torch.zeros(64, device="cuda")
    for _ in range(units):
        for _ in range(2000):
            x = x + 1.0
    torch.cuda.synchronize()


# `models` has no standalone function: it needs per-process model state, so it is
# built inline in `_child` (see there).
_WORKLOADS = {"sm": _work_sm, "launch": _work_launch}
_CHOICES = list(_WORKLOADS) + ["models"]


# ------------------------------------------------------------------
# child entry point
# ------------------------------------------------------------------
def _child(rank: int, workload: str, units: int, warmup: int, barrier, q,
           tiles: list, positions: list) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    free_before, total = torch.cuda.mem_get_info()
    torch.cuda.init()
    torch.zeros(1, device="cuda")           # force context creation
    torch.cuda.synchronize()
    free_after_ctx, _ = torch.cuda.mem_get_info()

    if workload == "models":
        # Model init is part of what a worker pays; time it and measure its VRAM,
        # but keep it OUT of the timed section (Candidate B/D pays it once at
        # worker startup, in parallel across workers).
        t_init = time.perf_counter()
        sys.path.insert(0, str(HYBRID))
        sys.path.insert(0, str(HYBRID.parent.parent.parent))
        import hybrid_pipeline as HP
        from config import config as CFG

        # Geometry must come from the COMPLETE grid -- compute_tile_geometry
        # fail-fasts on a partial one, and the probe deliberately runs only a
        # subset of tiles.
        geometry = HP.compute_tile_geometry(
            positions, CFG.default_tile_size, CFG.window_overlap_px,
        )
        unet = HP._init_unet_inferencer()
        cp = HP._init_cellpose_segmenter()
        dcp = HP._init_dish_cellpose_segmenter()
        init_s = time.perf_counter() - t_init

        def run(n, off=0):
            for u in range(n):
                ihc_p, dish_p, (ax, ay) = tiles[(rank * 7 + off + u) % len(tiles)]
                HP._process_precut_tile_gpu(
                    ihc_p, dish_p, ax, ay, geometry, unet, cp, dcp,
                    Path("/dev/null"),
                )
    else:
        init_s = 0.0
        fn = _WORKLOADS[workload]

        def run(n, off=0):
            fn(n)

    free_after_init, _ = torch.cuda.mem_get_info()

    run(warmup)                              # JIT / autotune / allocator warmup
    torch.cuda.synchronize()

    barrier.wait()                           # all processes start timed work together
    t0 = time.perf_counter()
    run(units, off=warmup)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    barrier.wait()                           # measure the *concurrent* window only
    dt_incl_stragglers = time.perf_counter() - t0

    q.put({
        "rank": rank,
        "wall_s": round(dt, 4),
        "wall_incl_barrier_s": round(dt_incl_stragglers, 4),
        "init_s": round(init_s, 3),
        "ctx_overhead_mb": round((free_before - free_after_ctx) / 1e6, 1),
        "model_vram_mb": round((free_after_ctx - free_after_init) / 1e6, 1),
        "proc_total_vram_mb": round((free_before - free_after_init) / 1e6, 1),
        "device_free_mb": round(free_after_init / 1e6, 1),
    })


# ------------------------------------------------------------------
def _dmon(path: Path):
    try:
        f = open(path, "w")
        p = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "um", "-d", "1", "-o", "DT"],
            stdout=f, stderr=subprocess.STDOUT,
        )
        return p, f
    except Exception:
        return None, None


def _dmon_stats(path: Path) -> dict:
    """Mean/peak SM% and peak FB over the dmon samples (skip the header lines)."""
    sm, fb = [], []
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        # #Date Time gpu sm mem enc dec jpg ofa fb bar1 ccpm
        try:
            sm.append(int(parts[3]))
            fb.append(int(parts[9]))
        except (IndexError, ValueError):
            continue
    if not sm:
        return {}
    return {
        "sm_mean": round(sum(sm) / len(sm), 1),
        "sm_peak": max(sm),
        "fb_peak_mb": max(fb),
        "n_samples": len(sm),
    }


def _discover_tiles(tiles_dir: Path, limit: int) -> tuple[list, list]:
    """Return (tile subset to process, ALL grid positions).

    The two are separate on purpose: `compute_tile_geometry` validates that the
    positions form a complete rectangular grid, so it must always see the whole
    grid even when the probe only runs a handful of tiles.
    """
    ihc_dir, dish_dir = tiles_dir / "ihc", tiles_dir / "dish"
    out = []
    for p in sorted(ihc_dir.glob("tile_x*_y*.tiff")):
        d = dish_dir / p.name
        if not d.exists():
            continue
        stem = p.stem                    # tile_x{X}_y{Y}
        x = int(stem.split("_x")[1].split("_y")[0])
        y = int(stem.split("_y")[1])
        out.append((p, d, (x, y)))
    if not out:
        raise SystemExit(f"no paired tiles under {tiles_dir}")
    positions = [pos for _i, _d, pos in out]
    return (out[:limit] if limit else out), positions


def run_trial(workload: str, n_procs: int, units: int, warmup: int,
              tiles: list, positions: list, metrics_dir: Path, tag: str) -> dict:
    ctx = mp.get_context("spawn")           # never fork -- see plan §2 Candidate E
    barrier = ctx.Barrier(n_procs)
    q = ctx.Queue()
    dmon_path = metrics_dir / f"{tag}_dmon.txt"
    dp, df = _dmon(dmon_path)

    procs = [
        ctx.Process(target=_child,
                    args=(r, workload, units, warmup, barrier, q, tiles,
                          positions))
        for r in range(n_procs)
    ]
    t0 = time.perf_counter()
    for p in procs:
        p.start()

    # Collect with a liveness check: a child that dies before reporting would
    # otherwise leave the parent blocked on q.get() forever (and a dead sibling
    # also deadlocks the survivors on the barrier).
    rows = []
    import queue as _queue
    while len(rows) < n_procs:
        try:
            rows.append(q.get(timeout=5))
        except _queue.Empty:
            dead = [p for p in procs if p.exitcode not in (None, 0)]
            if dead:
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                raise SystemExit(
                    f"child process died (exit codes "
                    f"{[p.exitcode for p in procs]}) -- see traceback above"
                )
    for p in procs:
        p.join()
    total = time.perf_counter() - t0

    if dp:
        dp.terminate()
        if df:
            df.close()

    fails = [p.exitcode for p in procs if p.exitcode != 0]
    walls = [r["wall_s"] for r in rows]
    return {
        "workload": workload,
        "n_procs": n_procs,
        "units_per_proc": units,
        "wall_max_s": round(max(walls), 4),
        "wall_min_s": round(min(walls), 4),
        "wall_mean_s": round(sum(walls) / len(walls), 4),
        "process_total_s": round(total, 3),
        "exit_failures": fails,
        "per_proc": rows,
        "dmon": _dmon_stats(dmon_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=_CHOICES, default="launch")
    ap.add_argument("--procs", default="1,2,3,4",
                    help="comma-separated process counts to trial")
    ap.add_argument("--units", type=int, default=20,
                    help="work units each process performs in the timed section")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tiles-dir", default=None,
                    help="precut scratch dir (has ihc/ and dish/) -- models mode")
    ap.add_argument("--tile-limit", type=int, default=24)
    ap.add_argument("--out", default=None, help="metrics dir for dmon + json")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    metrics_dir = Path(a.out) if a.out else Path("/tmp/mp_probe")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or a.workload

    tiles, positions = [], []
    if a.workload == "models":
        if not a.tiles_dir:
            raise SystemExit("--tiles-dir is required for --workload models")
        tiles, positions = _discover_tiles(Path(a.tiles_dir), a.tile_limit)
        print(f"[probe] {len(tiles)} tiles of a {len(positions)}-tile grid")

    counts = [int(x) for x in a.procs.split(",")]
    results = []
    for n in counts:
        print(f"\n[probe] {a.workload}: N={n} x {a.units} units ...", flush=True)
        r = run_trial(a.workload, n, a.units, a.warmup, tiles, positions,
                      metrics_dir, f"{tag}_n{n}")
        results.append(r)
        print(f"  wall max={r['wall_max_s']:.3f}s  mean={r['wall_mean_s']:.3f}s"
              f"  dmon={r['dmon']}")
        if r["exit_failures"]:
            print(f"  !! child exit codes {r['exit_failures']}")
        time.sleep(3)                       # let the device settle between trials

    base = next((r for r in results if r["n_procs"] == 1), None)
    print("\n" + "=" * 78)
    print(f"CONCURRENCY SCALING -- workload={a.workload}")
    print("=" * 78)
    hdr = (f"{'N':>3}{'wall max s':>12}{'throughput':>12}{'speedup':>10}"
           f"{'efficiency':>12}{'SM mean':>9}{'FB peak MB':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        thr = r["n_procs"] * r["units_per_proc"] / r["wall_max_s"]
        sp = (base["wall_max_s"] * r["n_procs"] / r["wall_max_s"]) if base else float("nan")
        print(f"{r['n_procs']:>3}{r['wall_max_s']:>12.3f}{thr:>12.2f}{sp:>10.2f}"
              f"{sp / r['n_procs'] * 100:>11.0f}%"
              f"{r['dmon'].get('sm_mean', 0):>9}{r['dmon'].get('fb_peak_mb', 0):>12}")
    print("\nspeedup ~= N -> kernels overlap;  speedup ~= 1.0 -> they serialize")

    if base and base["per_proc"]:
        p0 = base["per_proc"][0]
        print(f"\nper-process VRAM: CUDA context {p0['ctx_overhead_mb']} MB"
              f" + models {p0['model_vram_mb']} MB"
              f" = {p0['proc_total_vram_mb']} MB;"
              f" model init {p0['init_s']} s")

    outp = metrics_dir / f"{tag}_probe.json"
    outp.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
