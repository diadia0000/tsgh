"""A2 audit: does `detect_all_dots`'s `n_jobs=-1` oversubscribe cores at `workers=N`?

Executes docs/hybrid-pipeline/22-next-optimization-cycle-plan.md §2 (A2).
**Audit only — reads the pipeline, changes nothing.**

The hypothesis: `_finish_chunk_cpu` calls `detect_all_dots` without `n_jobs`
(hybrid_pipeline.py:631), so each worker process fans out `Parallel(n_jobs=-1,
prefer='threads')` over *every* core. Under cross-tile multiprocessing at
`workers=N` that is N processes each claiming all cores — an N-fold
oversubscription that would silently eat into the measured speedup.

Method (two phases so the CPU measurement never shares the box with GPU work):
  --prepare  runs the real GPU front (M1/M2/M3b) over K real tiles once and pickles
             the resulting `_ChunkGpuState`s, i.e. the exact inputs the BG arm sees.
  --run      replays the BG arm (`build_all_positive_results` + `enlarge_cell_instances`
             + `detect_all_dots` + merge) from those pickles in P concurrent processes,
             barrier-synchronised so the timed window is genuinely concurrent, and
             reports per-call wall time per stage.

Comparing P=1 against P=3/P=6 isolates contention; sweeping `--n-jobs` shows whether
capping the fan-out (`cpu_count // P`) recovers it.

Note this is a *lower* bound on real contention: in the pipeline each process also runs
a GPU/MAIN thread that holds a core, which this CPU-only replay does not reproduce.

Usage:
  .venv/bin/python scripts/cpu_contention_probe.py --prepare --tiles 8 --state-dir DIR
  .venv/bin/python scripts/cpu_contention_probe.py --run --state-dir DIR \
      --procs 1,3,6 --n-jobs -1 --passes 3 --out out.json
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import pickle
import statistics
import sys
import time
from pathlib import Path

logging.getLogger("pyvips").setLevel(logging.WARNING)

HYBRID = Path(__file__).resolve().parent.parent / "backend" / "algorithms" / "hybrid"
sys.path.insert(0, str(HYBRID))
sys.path.insert(0, str(HYBRID.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def prepare(ihc: Path, dish: Path, n_tiles: int, state_dir: Path) -> None:
    """Run the GPU front once and pickle the BG arm's real inputs."""
    import hybrid_pipeline as HP
    import torch
    from config import config
    from cellpose_batch_probe import _load_tiles

    pairs = _load_tiles(ihc, dish, n_tiles)
    unet = HP._init_unet_inferencer()
    m2 = HP._init_cellpose_segmenter()
    m3b = HP._init_dish_cellpose_segmenter()

    state_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for i, (ihc_img, dish_img) in enumerate(pairs):
        chunk = HP._process_one_chunk_gpu(
            ihc_img, dish_img, 0, 0, (True, True, True, True), unet, m2, m3b,
        )
        if chunk is None:
            continue
        with open(state_dir / f"chunk_{kept:03d}.pkl", "wb") as fh:
            pickle.dump(chunk, fh, protocol=pickle.HIGHEST_PROTOCOL)
        cells = int(chunk.instance_mask.max())
        print(f"  tile {i}: {cells} cells, saved as chunk_{kept:03d}.pkl")
        kept += 1
    del unet, m2, m3b
    torch.cuda.empty_cache()
    print(f"prepared {kept} chunk states in {state_dir} (config_tile={config.default_tile_size})")


def _bg_arm_worker(state_dir: str, n_jobs: int, passes: int, barrier, out_q, rank: int) -> None:
    """Replay the BG arm `passes` times over every saved chunk, timing each stage."""
    sys.path.insert(0, str(HYBRID))
    sys.path.insert(0, str(HYBRID.parent.parent.parent))
    from config import config
    from m3_cell_detection import (
        build_all_positive_results,
        detect_all_dots,
        enlarge_cell_instances,
        merge_dot_results_to_cell_analysis,
    )

    states = []
    for p in sorted(Path(state_dir).glob("chunk_*.pkl")):
        with open(p, "rb") as fh:
            states.append(pickle.load(fh))

    # warm up (imports, first-touch pages, joblib thread pool) before the barrier
    gs = states[0]
    detect_all_dots(gs.dish_mask_overlay, enlarge_cell_instances(gs.instance_mask, config),
                    config, dish_nucleus_mask=gs.dish_nucleus_mask, core_mask=gs.core_mask,
                    n_jobs=n_jobs)

    barrier.wait()
    t_start = time.perf_counter()
    detect_t, arm_t, n_cells = [], [], []
    for _ in range(passes):
        for gs in states:
            t0 = time.perf_counter()
            results_pre = build_all_positive_results(gs.instance_mask)
            matching_mask = enlarge_cell_instances(gs.instance_mask, config)
            t1 = time.perf_counter()
            all_dots, per_cell, _m = detect_all_dots(
                gs.dish_mask_overlay, matching_mask, config,
                dish_nucleus_mask=gs.dish_nucleus_mask, core_mask=gs.core_mask,
                n_jobs=n_jobs,
            )
            t2 = time.perf_counter()
            merge_dot_results_to_cell_analysis(results_pre, per_cell)
            t3 = time.perf_counter()
            detect_t.append(t2 - t1)
            arm_t.append(t3 - t0)
            n_cells.append(len(per_cell))
    out_q.put({
        "rank": rank,
        "pid": os.getpid(),
        "wall_s": time.perf_counter() - t_start,
        "detect_calls": len(detect_t),
        "detect_median_s": statistics.median(detect_t),
        "detect_mean_s": statistics.fmean(detect_t),
        "arm_median_s": statistics.median(arm_t),
        "cells_per_tile": n_cells[:len(n_cells) // passes] or n_cells,
    })


def run(state_dir: Path, procs: list[int], n_jobs_list: list[int], passes: int, out_path):
    ctx = mp.get_context("spawn")
    cores = len(os.sched_getaffinity(0))
    out = {"cpu_affinity_cores": cores, "os_cpu_count": os.cpu_count(),
           "state_dir": str(state_dir), "passes": passes, "results": {}}
    print(f"cores available: {cores} (os.cpu_count()={os.cpu_count()})")

    for p in procs:
        for n_jobs in n_jobs_list:
            eff = n_jobs if n_jobs > 0 else cores
            barrier = ctx.Barrier(p)
            q = ctx.Queue()
            workers = [
                ctx.Process(target=_bg_arm_worker,
                            args=(str(state_dir), n_jobs, passes, barrier, q, r))
                for r in range(p)
            ]
            t0 = time.perf_counter()
            for w in workers:
                w.start()
            recs = [q.get() for _ in range(p)]
            for w in workers:
                w.join()
            wall = time.perf_counter() - t0

            med = statistics.fmean(r["detect_median_s"] for r in recs)
            arm = statistics.fmean(r["arm_median_s"] for r in recs)
            calls = sum(r["detect_calls"] for r in recs)
            key = f"P{p}_njobs{n_jobs}"
            out["results"][key] = {
                "procs": p, "n_jobs_arg": n_jobs, "threads_per_proc": eff,
                "threads_total": eff * p, "wall_s": round(wall, 3),
                "detect_median_s": round(med, 4), "arm_median_s": round(arm, 4),
                "detect_calls_total": calls,
                "detect_throughput_calls_per_s": round(calls / wall, 3),
                "per_proc": [{k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in r.items() if k != "cells_per_tile"} for r in recs],
            }
            print(f"  P={p} n_jobs={n_jobs:>3} ({eff} thr/proc, {eff*p} total): "
                  f"detect median {med*1000:7.1f} ms/call, BG-arm {arm*1000:7.1f} ms/tile, "
                  f"wall {wall:6.2f}s, throughput {calls/wall:5.2f} calls/s")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))
        print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--ihc", default=None)
    ap.add_argument("--dish", default=None)
    ap.add_argument("--tiles", type=int, default=8)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--procs", default="1,3,6")
    ap.add_argument("--n-jobs", default="-1",
                    help="comma list; -1 = joblib default (all cores), or an explicit cap")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.prepare:
        from cellpose_batch_probe import ROI
        prepare(Path(args.ihc or ROI / "large_ihc.tiff"),
                Path(args.dish or ROI / "large_dish.tiff"),
                args.tiles, Path(args.state_dir))
    if args.run:
        run(Path(args.state_dir),
            [int(v) for v in args.procs.split(",")],
            [int(v) for v in args.n_jobs.split(",")],
            args.passes, args.out)


if __name__ == "__main__":
    main()
