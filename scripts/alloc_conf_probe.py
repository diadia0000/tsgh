"""Test `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` against the `workers>=6`
allocator balloon (doc 19 #7b, DISCOVERED #2, doc 23 §4.6/§6).

The defect: at `workers=6`, **2 of 6 runs** OOM-killed the batch, and in both the
victim worker had ballooned to **exactly 24.76 GiB** while its five siblings sat at
1.1-1.7 GB (steady state ~2.8 GB). Different victim tiles and different requested
sizes each time — the random victim looks like fragmentation, but a byte-identical
balloon does not. Round 7 saw the same signature at `workers=4` without an OOM
(one dmon sample at 26,687 MB of 32,607 against a 9,137 MB median), so the real
transient headroom at the recommended worker count is ~6 GB, not the ~12 GB a
steady-state reading suggests.

`expandable_segments:True` was named as the candidate first fix in doc 21 §4.7 and
again in doc 23 §6, and was never tried. This script tries it, the only way the
question can be settled: run the **same** sweep N times under each condition and
compare failure rate and peak VRAM. It changes nothing in the pipeline — the knob is
`config.cuda_alloc_conf`, applied to the environment just before the workers spawn
(the parent never allocates on the device on that path).

Note this is a *reliability* measurement, not a speed one. The decision it feeds is
"is `workers=6` safe", not "is it faster" — the curve is already flat past
`workers=5` (doc 23 §6), so a pass here buys headroom and confidence, not wall-clock.

Usage:
  .venv/bin/python scripts/alloc_conf_probe.py \\
      --ihc <crop_ihc.tiff> --dish <crop_dish.tiff> \\
      --workers 6 --repeats 6 --out alloc_conf.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PERF = REPO / "scripts" / "perf_measure.py"

# "" means: do not touch the environment at all -- today's behaviour, the control.
CONDITIONS = {
    "control": "",
    "expandable": "expandable_segments:True",
}


def one_run(python: str, ihc: Path, dish: Path, out_dir: Path, metrics: Path,
            label: str, workers: int, alloc_conf: str) -> dict:
    """One full perf_measure run; returns its parsed metrics or the failure reason."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        python, str(PERF),
        "--ihc", str(ihc), "--dish", str(dish),
        "--output", str(out_dir), "--label", label,
        "--mp-workers", str(workers), "--stream-precut",
        "--metrics-dir", str(metrics), "--gpu-dmon",
        "--cuda-alloc-conf", alloc_conf,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    (metrics / f"{label}_stdout.log").write_text(proc.stdout + "\n=== STDERR ===\n"
                                                 + proc.stderr)

    row = {"label": label, "alloc_conf": alloc_conf, "returncode": proc.returncode,
           "driver_wall_s": round(wall, 2)}
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
        row["ok"] = False
        row["oom"] = any("out of memory" in ln.lower() or "OutOfMemory" in ln
                         for ln in (proc.stderr or "").splitlines())
        row["tail"] = tail
        return row

    js = metrics / f"{label}_timings.json"
    data = json.loads(js.read_text())
    row.update({
        "ok": True,
        "end_to_end_s": data["wall"]["end_to_end_total_s"],
        "stats": data["stats"],
        "n_tiles": data["n_tiles"],
        "peak_rss_gb": data["peak_rss_gb"],
        "effective_alloc_conf": data.get("cuda_alloc_conf", ""),
    })
    row["peak_gpu_mb"] = peak_from_dmon(metrics / f"{label}_gpu_dmon.txt")
    return row


def peak_from_dmon(path: Path) -> float | None:
    """Peak framebuffer use (MB) from `nvidia-smi dmon -s um -o DT`.

    The parent-process `torch.cuda.memory_reserved` sampler cannot see this: at
    `workers>1` the allocations live in the *children*, which is exactly why the
    24.76 GiB balloon was only ever caught by dmon.

    The `fb` column is located from the header rather than by scanning for a
    plausible-looking number — `sm`/`mem` percentages and the `YYYYMMDD` date share
    the row, and a positional guess silently breaks if the column set ever changes.
    """
    if not path.exists():
        return None
    fb_idx = None
    peak = 0.0
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#"):
            if fb_idx is None and "fb" in line.split():
                # header names omit the Date/Time columns' own labels being split the
                # same way, so index against the same split the data rows use
                cols = line.lstrip("#").split()
                fb_idx = cols.index("fb")
            continue
        parts = line.split()
        if fb_idx is None or fb_idx >= len(parts):
            continue
        try:
            peak = max(peak, float(parts[fb_idx]))
        except ValueError:
            continue
    return peak or None


def gpu_is_clean(max_mb: float = 500.0) -> tuple[bool, str]:
    """Refuse to start on a GPU that already has someone else's memory on it.

    This measurement is *about* VRAM, so a stray process poisons every number in the
    sweep. It is not a hypothetical: killing a previous sweep's parent leaves its
    `spawn`ed tile workers alive, each still holding a ~1 GB CUDA context, and the next
    run then OOMs for a reason that has nothing to do with the condition under test.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
        used = float(out)
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
    except Exception as exc:                 # noqa: BLE001
        return True, f"could not query nvidia-smi ({exc!r}); proceeding"
    if used > max_mb:
        return False, (f"GPU already has {used:.0f} MB in use before the sweep starts "
                       f"(limit {max_mb:.0f} MB). Compute apps:\n{apps or '  (none reported)'}")
    return True, f"GPU clean: {used:.0f} MB in use"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ihc", required=True)
    ap.add_argument("--dish", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=6,
                    help="runs per condition; the defect is intermittent (2-in-6), so "
                         "a single pass proves nothing either way")
    ap.add_argument("--python", default=str(REPO / ".venv" / "bin" / "python"))
    ap.add_argument("--scratch", default="/tmp/alloc_conf_probe")
    ap.add_argument("--force", action="store_true",
                    help="start even if the GPU already has memory in use")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    clean, why = gpu_is_clean()
    print(why, flush=True)
    if not clean and not args.force:
        raise SystemExit("refusing to start on a dirty GPU (pass --force to override)")

    scratch = Path(args.scratch)
    metrics = scratch / "_metrics"
    metrics.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    # Interleave the conditions rather than running all of one then all of the other:
    # thermal state and page-cache warmth drift over a 20-minute sweep, and a blocked
    # design would hand the whole drift to whichever condition ran second.
    for i in range(args.repeats):
        for cond, conf in CONDITIONS.items():
            label = f"{cond}_w{args.workers}_r{i + 1}"
            print(f"[{len(rows) + 1}/{args.repeats * len(CONDITIONS)}] {label} ...",
                  flush=True)
            row = one_run(args.python, Path(args.ihc), Path(args.dish),
                          scratch / label, metrics, label, args.workers, conf)
            row["condition"] = cond
            rows.append(row)
            print(f"    ok={row['ok']} wall={row.get('end_to_end_s')} "
                  f"peak_gpu_mb={row.get('peak_gpu_mb')}", flush=True)
            shutil.rmtree(scratch / label, ignore_errors=True)

    summary = {}
    for cond in CONDITIONS:
        got = [r for r in rows if r["condition"] == cond]
        ok = [r for r in got if r["ok"]]
        peaks = [r["peak_gpu_mb"] for r in ok if r.get("peak_gpu_mb")]
        walls = [r["end_to_end_s"] for r in ok]
        summary[cond] = {
            "runs": len(got),
            "failures": len(got) - len(ok),
            "oom_failures": sum(1 for r in got if r.get("oom")),
            "median_wall_s": round(sorted(walls)[len(walls) // 2], 2) if walls else None,
            "max_peak_gpu_mb": max(peaks) if peaks else None,
            "median_peak_gpu_mb": (round(sorted(peaks)[len(peaks) // 2], 1)
                                   if peaks else None),
        }

    result = {"workers": args.workers, "repeats": args.repeats,
              "summary": summary, "runs": rows}
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
