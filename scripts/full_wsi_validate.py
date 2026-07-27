"""Drive the full real-WSI-scale end-to-end validation (doc 19 #7, DISCOVERED #3).

This is the item [09 §3.6](../docs/hybrid-pipeline/09-measurement-analysis-plan.md)
specified and every round since has descoped: **no round has ever run a complete
slide**. Every wall-clock number this project has is a linear extrapolation from a
25/121/441/576-tile crop, and doc 25 §1 already caught one of those extrapolations
being wrong by 1.8x (Phase D) and another (tissue share) being wrong outright. It is
also the binding gate on shipping `workers>1` — doc 20 §4 forbids shipping ahead of it.

What this adds over calling `perf_measure.py` by hand:

1. **Preflight.** A full slide is ~27,565 tiles; at the measured 12.3 MB of output per
   tile that is ~340 GB, plus the precut scratch. Discovering that at tile 24,000 is
   the same multi-hour loss the checkpoint exists to prevent, so free space, the
   `RLIMIT_NOFILE` the final stitch will need, and GPU/model availability are all
   checked *before* anything starts.
2. **Resume by default.** Every run goes through `run_batch(checkpoint=True)`, so an
   interrupted attempt restarts where it stopped instead of from tile 1.
3. **The two runs the plan asks for**, in the order it asks for: `workers=1`
   (production default) first, then `workers=4` (the round-6/7 recommendation).
4. **A correctness spot-check** between the two runs' `report.csv`, since a
   worker-count change must not change the cell table beyond GPU non-determinism.

Usage:
  .venv/bin/python scripts/full_wsi_validate.py \\
      --ihc /data/.../HER2_processed.tiff --dish /data/.../DISH_processed.tiff \\
      --output-root /data/nvmessd/full_wsi --workers 1,4

  # check the preflight without committing hours of GPU time
  .venv/bin/python scripts/full_wsi_validate.py --ihc ... --dish ... \\
      --output-root ... --preflight-only
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PERF = REPO / "scripts" / "perf_measure.py"
sys.path.insert(0, str(REPO / "backend" / "algorithms" / "hybrid"))
sys.path.insert(0, str(REPO))

# Measured per-tile disk cost, everything included. Two independent anchors:
#
#   round 7, 576-tile crop at 76.2% background : 7.085 GB total -> 12.30 MB/tile
#   round 8, 25-tile ROI at ~88% tissue        : 362.7 MB total -> 14.51 MB/tile
#
# The real slide is 44.2% tissue, between the two, which interpolates to ~13.0 MB/tile;
# 14.5 is used here so the check errs toward refusing a run that would fit rather than
# starting one that would not.
#
# Almost all of it is fixed per tile and does NOT scale with tissue content:
#   instance_mask       4.195 MB  (1024x1024 int32, uncompressed)
#   dish_nucleus_mask   4.195 MB  (same)
#   overlay_annotated   2.014 MB
#   _precut_scratch     2.370 MB  (IHC + DISH tile, deflate)
#   masked_ihc          0.621 MB
#   dish_mask_overlay   0.388 MB
#   core_mask/_resume   0.010 MB
#   cell_crops          0.006 MB per cell -- the only tissue-dependent term
BYTES_PER_TILE = 14.5e6

# Nothing in the pipeline ever deletes any of this: there is not one `rmtree`/`unlink`
# in `backend/algorithms/hybrid/`. The precut scratch in particular is written once and
# kept (`_run_single_tile_cli`: "保留供檢查，不自動清理"). Every byte below is therefore
# PERMANENT until somebody removes it by hand.


def conform_to_intersection(ihc: Path, dish: Path, dst_dir: Path) -> tuple[Path, Path]:
    """Write copies of both slides cropped to their common intersection.

    The registration stage emits a separate canvas per modality — on the reference
    case HER2 is 141818x114366 and DISH is 141658x114415 — because it aligns *cells*,
    not canvas extents. `PrecutStream` requires identical dimensions and fail-fasts
    otherwise, so a full-slide run needs the pair conformed once, upfront.

    Intersection rather than union: outside it, one modality has no pixels at all, so
    no IHC-DISH result there could be valid. Union with white fill would instead
    manufacture tiles where one arm is blank, and the pipeline would dutifully analyse
    them. On the reference case this drops a 160px right strip and a 49px bottom strip
    — 99.86% of the slide is retained.

    This is done in the runner, not in the pipeline: relaxing `PrecutStream`'s check
    would weaken a guard the API path relies on to reject genuinely mismatched input.

    **The output format is chosen to match how the source is actually read, not to make
    a nice archive file**, because this copy feeds the one measurement the whole run
    exists to produce and any format difference lands straight in the wall-clock:

    - `tile_width/height = 1024` mirrors the source's own tiling and, more importantly,
      `PrecutStream`'s access pattern — it crops 1024x1024 regions, so 1024px tiles mean
      ~1 tile decoded per crop. pyvips' 128px default would make every crop touch 64.
    - `pyramid=False` because `PrecutStream` opens with `access="random"` and only ever
      reads full resolution. Downsample levels are ~33% more bytes and write time for
      something nothing in this pipeline will read.
    - `compression="deflate"` (lossless) rather than re-encoding the source's JPEG. The
      source is already lossy; a second JPEG generation would add loss for no reason,
      and the size cost is irrelevant next to the ~400 GB the run itself writes.
    """
    import pyvips

    dst_dir.mkdir(parents=True, exist_ok=True)
    a = pyvips.Image.new_from_file(str(ihc), access="random")
    b = pyvips.Image.new_from_file(str(dish), access="random")
    w, h = min(a.width, b.width), min(a.height, b.height)

    out = []
    for src, img, name in ((ihc, a, "ihc"), (dish, b, "dish")):
        dst = dst_dir / f"{name}_conformed_{w}x{h}.tiff"
        if dst.exists():
            print(f"  reusing {dst}", flush=True)
        elif (img.width, img.height) == (w, h):
            # already the intersection size; a symlink avoids copying ~10 GB
            dst.symlink_to(Path(src).resolve())
            print(f"  {name}: already {w}x{h}, linked", flush=True)
        else:
            t0 = time.perf_counter()
            print(f"  {name}: {img.width}x{img.height} -> {w}x{h}, writing ...",
                  flush=True)
            img.crop(0, 0, w, h).tiffsave(
                str(dst), tile=True, tile_width=1024, tile_height=1024,
                pyramid=False, compression="deflate", bigtiff=True)
            print(f"    wrote {dst.stat().st_size / 1e9:.1f} GB in "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
        out.append(dst)
    return out[0], out[1]


def preflight(ihc: Path, dish: Path, out_root: Path, n_runs: int = 1) -> dict:
    """Fail before the run, not four hours into it."""
    import pyvips
    import resource

    from config import config
    from m0_reader import chunk_offsets

    problems: list[str] = []

    for name, p in (("IHC", ihc), ("DISH", dish)):
        if not p.exists():
            problems.append(f"{name} slide not found: {p}")
    if problems:
        return {"ok": False, "problems": problems}

    ihc_img = pyvips.Image.new_from_file(str(ihc), access="sequential")
    dish_img = pyvips.Image.new_from_file(str(dish), access="sequential")
    if (ihc_img.width, ihc_img.height) != (dish_img.width, dish_img.height):
        problems.append(
            f"IHC {ihc_img.width}x{ihc_img.height} != DISH "
            f"{dish_img.width}x{dish_img.height}; the two must be aligned and "
            f"identically sized before this stage")

    positions = chunk_offsets(ihc_img.height, ihc_img.width,
                              config.default_tile_size, config.window_overlap_px)
    n = len(positions)

    # Each worker count gets its own output directory, so the cost multiplies by the
    # number of runs -- they are not reusing one another's tiles, and they must not
    # (comparing two runs' report.csv is the correctness check).
    per_run = n * BYTES_PER_TILE
    need = per_run * n_runs
    out_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_root).free
    if free < need * 1.15:
        problems.append(
            f"free space {free / 1e9:.0f} GB at {out_root} is under the "
            f"{need * 1.15 / 1e9:.0f} GB this needs "
            f"({n_runs} run(s) x {n} tiles x {BYTES_PER_TILE / 1e6:.1f} MB "
            f"+ 15% margin). NOTE: nothing is deleted automatically -- see the "
            f"module docstring")

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed_fds = n + 256
    if hard != resource.RLIM_INFINITY and hard < needed_fds:
        problems.append(
            f"RLIMIT_NOFILE hard limit {hard} is below the {needed_fds} the final "
            f"stitch needs; raise it (`ulimit -Hn {needed_fds}`) or the run dies at "
            f"the very last step")

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda_ok else None
    except Exception as exc:                 # noqa: BLE001
        cuda_ok, gpu = False, f"import failed: {exc!r}"
    if not cuda_ok:
        problems.append("CUDA is not available; this run needs the GPU")

    for label, p in (("unet", config.unet_model_path),
                     ("cellpose_m2", config.cellpose_model_path),
                     ("cellpose_m3b", config.cellpose_dish_model_path)):
        if not Path(p).exists():
            problems.append(f"{label} model missing: {p}")

    return {
        "ok": not problems,
        "problems": problems,
        "slide_px": [ihc_img.width, ihc_img.height],
        "gigapixels": round(ihc_img.width * ihc_img.height / 1e9, 3),
        "n_tiles": n,
        "grid": [len({x for x, _ in positions}), len({y for _, y in positions})],
        "n_runs": n_runs,
        "projected_gb_per_run": round(per_run / 1e9, 1),
        "projected_gb_total": round(need / 1e9, 1),
        "projected_gb_breakdown": {
            "instance_mask": round(n * 4.195e6 / 1e9, 1),
            "dish_nucleus_mask": round(n * 4.195e6 / 1e9, 1),
            "overlay_annotated": round(n * 2.014e6 / 1e9, 1),
            "_precut_scratch": round(n * 2.370e6 / 1e9, 1),
            "masked_ihc": round(n * 0.621e6 / 1e9, 1),
            "dish_mask_overlay": round(n * 0.388e6 / 1e9, 1),
        },
        "free_gb": round(free / 1e9, 1),
        "free_gb_after": round((free - need) / 1e9, 1),
        "files_are_deleted_automatically": False,
        "rlimit_nofile": [soft, hard],
        "gpu": gpu,
    }


def run_one(python: str, ihc: Path, dish: Path, out_root: Path, metrics: Path,
            workers: int, worker_timings: bool) -> dict:
    label = f"fullwsi_w{workers}"
    out_dir = out_root / label
    cmd = [
        python, str(PERF),
        "--ihc", str(ihc), "--dish", str(dish),
        "--output", str(out_dir), "--label", label,
        "--mp-workers", str(workers), "--stream-precut", "--resume",
        "--metrics-dir", str(metrics), "--gpu-dmon",
    ]
    if worker_timings and workers > 1:
        cmd.append("--worker-timings")

    log_path = metrics / f"{label}_driver.log"
    print(f"\n=== {label}: {' '.join(cmd)}", flush=True)
    print(f"    live log: tail -f {log_path}", flush=True)

    # Stream to the log rather than buffering with capture_output: this run takes
    # hours, and a captured pipe means nobody can see progress (or a stall) until it
    # is over. The per-tile "[n/27565]" lines are the only progress signal there is.
    t0 = time.perf_counter()
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0

    row = {"label": label, "workers": workers, "driver_wall_s": round(wall, 1),
           "returncode": proc.returncode, "output_dir": str(out_dir),
           "log": str(log_path)}
    if proc.returncode != 0:
        row["ok"] = False
        row["tail"] = log_path.read_text().strip().splitlines()[-15:]
        print(f"    FAILED after {wall / 60:.1f} min — rerun to resume from the "
              f"checkpoint in {out_dir / '_resume'}", flush=True)
        return row

    row["ok"] = True
    row["metrics"] = json.loads((metrics / f"{label}_timings.json").read_text())
    print(f"    done in {wall / 3600:.2f} h", flush=True)
    return row


def compare_reports(a: Path, b: Path) -> dict:
    """Correctness veto: worker count must not change the cell table.

    GPU inference is non-deterministic, so this reports the deltas rather than
    asserting equality — prior rounds judged deltas against a same-code noise floor
    (doc 21 §4: zero delta at the small anchor, inside noise at the others).
    """
    def load(p: Path):
        with open(p, newline="") as fh:
            return list(csv.DictReader(fh))

    if not (a.exists() and b.exists()):
        return {"comparable": False, "why": f"missing {a if not a.exists() else b}"}

    ra, rb = load(a), load(b)
    amp_a = sum(1 for r in ra if str(r.get("is_amplified", "")).lower() in ("true", "1"))
    amp_b = sum(1 for r in rb if str(r.get("is_amplified", "")).lower() in ("true", "1"))
    return {
        "comparable": True,
        "n_cells": [len(ra), len(rb)],
        "n_cells_delta_pct": (round((len(rb) - len(ra)) / len(ra) * 100, 3)
                              if ra else None),
        "n_amplified": [amp_a, amp_b],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ihc", required=True)
    ap.add_argument("--dish", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--workers", default="1,4",
                    help="comma-separated worker counts to run, in order "
                         "(default 1,4 = the plan's two runs)")
    ap.add_argument("--worker-timings", action="store_true",
                    help="collect per-stage buckets inside the workers; adds shim "
                         "overhead, so the wall-clock of that run is not comparable")
    ap.add_argument("--python", default=str(REPO / ".venv" / "bin" / "python"))
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--conform", action="store_true",
                    help="first crop both slides to their common intersection (the "
                         "registration stage emits a different canvas per modality); "
                         "written once under <output-root>/_conformed and reused")
    ap.add_argument("--force", action="store_true",
                    help="run even if preflight found problems")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_root = Path(args.output_root)
    ihc_path, dish_path = Path(args.ihc), Path(args.dish)
    if args.conform:
        out_root.mkdir(parents=True, exist_ok=True)
        print("conforming slide pair to their intersection:", flush=True)
        ihc_path, dish_path = conform_to_intersection(
            ihc_path, dish_path, out_root / "_conformed")

    worker_counts = [int(x) for x in args.workers.split(",")]
    pf = preflight(ihc_path, dish_path, out_root, n_runs=len(worker_counts))
    print(json.dumps(pf, indent=2))
    if not pf["ok"]:
        print("\nPREFLIGHT PROBLEMS:")
        for p in pf["problems"]:
            print("  -", p)
        if not args.force:
            sys.exit(1)
    if args.preflight_only:
        return

    metrics = out_root / "_metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    result = {"preflight": pf, "runs": []}

    for w in worker_counts:
        result["runs"].append(
            run_one(args.python, ihc_path, dish_path,
                    out_root, metrics, w, args.worker_timings))
        # write after every run: this loop spans hours, and a partial record beats none
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2, default=str))

    ok = [r for r in result["runs"] if r["ok"]]
    if len(ok) >= 2:
        result["correctness"] = compare_reports(
            Path(ok[0]["output_dir"]) / "report.csv",
            Path(ok[1]["output_dir"]) / "report.csv")
        print("\ncorrectness:", json.dumps(result["correctness"], indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({r["label"]: r.get("driver_wall_s") for r in result["runs"]},
                     indent=2))


if __name__ == "__main__":
    main()
