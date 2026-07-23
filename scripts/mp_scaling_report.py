"""Summarise the cross-tile multiprocessing sweep (doc 20 §3 step 4).

`arm_report.py` cannot be used on these runs, and the reason is structural rather
than a missing feature. It decomposes a run into `wall ~= max(MAIN, BG) + outside`
from `perf_measure.py`'s monkeypatched per-function timers -- but those shims live
in the **parent** process, while under `--mp-workers N` every tile is processed in
a spawned child. The parent's `TIMINGS` therefore come back empty for every
worker-side bucket, and the two-arm model itself no longer describes the run:
there are now N MAIN arms and N BG arms, not one of each.

So this reports what is still measured and still decides the question, which the
playbook says is end-to-end wall-clock anyway:

  * wall-clock scaling vs the 1-worker control (the ablation),
  * the device-side view from `nvidia-smi dmon` (mean SM%, near-idle fraction,
    peak FB) -- doc 18 §7.1's caveat applies, so near-idle is reported at SM<=3
    rather than the knife-edge SM==0,
  * peak RSS across the whole process tree (the sampler already walks children),
  * cell/tile counts, as a first-pass correctness screen before the per-cell veto.

Usage:
  python scripts/mp_scaling_report.py --metrics-dir <dir> [--runs-dir <dir>]
      [--pattern med] [--baseline-workers 1]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


def load(metrics_dir: Path, pattern: str | None) -> list[dict]:
    out = []
    for f in sorted(metrics_dir.glob("*_timings.json")):
        d = json.load(open(f))
        if pattern and pattern not in d["label"]:
            continue
        d["_dmon"] = dmon_stats(metrics_dir / f"{d['label']}_gpu_dmon.txt")
        out.append(d)
    return out


def dmon_stats(path: Path) -> dict:
    """Mean SM%, near-idle fraction (SM<=3) and peak FB from an nvidia-smi dmon log.

    doc 18 §7.1: the historical `idle_frac` counted *exactly* SM==0, which is
    knife-edge -- a cluster of SM=1-3% samples collapsing onto 0 made idle_frac
    rise while wall fell. SM<=3 is the stable form and is what this reports.
    """
    if not path.exists():
        return {}
    sm, fb = [], []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        try:
            sm.append(int(p[3]))
            fb.append(int(p[9]))
        except (IndexError, ValueError):
            continue
    if not sm:
        return {}
    return {
        "sm_mean": round(statistics.fmean(sm), 1),
        "near_idle": round(sum(1 for v in sm if v <= 3) / len(sm), 3),
        "fb_peak": max(fb),
        "n": len(sm),
    }


def n_cells(runs_dir: Path | None, label: str) -> int | None:
    if runs_dir is None:
        return None
    p = runs_dir / label / "report.csv"
    if not p.exists():
        return None
    with open(p) as f:
        return sum(1 for _ in csv.DictReader(f))


def group_key(label: str) -> str:
    """Strip a trailing repeat suffix (`_r1`, `_r2`, ...) to group repeats."""
    return re.sub(r"_r\d+$", "", label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--pattern", default=None,
                    help="only labels containing this substring (e.g. 'med')")
    ap.add_argument("--baseline-workers", type=int, default=1)
    a = ap.parse_args()

    runs = load(Path(a.metrics_dir), a.pattern)
    if not runs:
        raise SystemExit("no matching *_timings.json")
    runs_dir = Path(a.runs_dir) if a.runs_dir else None

    groups: dict[str, list[dict]] = {}
    for d in runs:
        groups.setdefault(group_key(d["label"]), []).append(d)

    rows = []
    for key, ds in sorted(groups.items(), key=lambda kv: kv[1][0].get("mp_workers", 1)):
        walls = [d["wall"]["end_to_end_total_s"] for d in ds]
        cells = [c for c in (n_cells(runs_dir, d["label"]) for d in ds) if c]
        rows.append({
            "key": key,
            "w": ds[0].get("mp_workers", 1),
            "n": len(ds),
            "tiles": ds[0]["n_tiles"],
            "wall": statistics.fmean(walls),
            "lo": min(walls), "hi": max(walls),
            "rss": max(d.get("peak_rss_gb", 0) for d in ds),
            "fb": max((d["_dmon"].get("fb_peak", 0) for d in ds), default=0),
            "sm": statistics.fmean([d["_dmon"]["sm_mean"] for d in ds
                                    if d["_dmon"]]) if any(d["_dmon"] for d in ds) else 0,
            "idle": statistics.fmean([d["_dmon"]["near_idle"] for d in ds
                                      if d["_dmon"]]) if any(d["_dmon"] for d in ds) else 0,
            "cells": f"{min(cells)}-{max(cells)}" if cells else "-",
            "stats": ds[0]["stats"],
        })

    base = next((r for r in rows if r["w"] == a.baseline_workers), None)

    print("\n" + "=" * 104)
    print("CROSS-TILE MULTIPROCESSING SCALING  (end-to-end wall is the decision metric)")
    print("=" * 104)
    hdr = (f"{'workers':>8}{'n':>3}{'tiles':>7}{'wall s':>10}{'spread':>16}"
           f"{'speedup':>9}{'eff':>6}{'SM%':>7}{'idle≤3':>8}{'FB MB':>8}"
           f"{'RSS GB':>8}{'cells':>13}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        sp = base["wall"] / r["wall"] if base else float("nan")
        spread = f"{r['lo']:.1f}-{r['hi']:.1f}"
        print(f"{r['w']:>8}{r['n']:>3}{r['tiles']:>7}{r['wall']:>10.1f}{spread:>16}"
              f"{sp:>9.2f}{sp / r['w'] * 100:>5.0f}%{r['sm']:>7.1f}{r['idle']:>8.2f}"
              f"{r['fb']:>8}{r['rss']:>8.2f}{r['cells']:>13}")

    print("\nmarginal gain per added worker (where the knee is):")
    prev = None
    for r in sorted(rows, key=lambda x: x["w"]):
        sp = base["wall"] / r["wall"] if base else float("nan")
        if prev is not None:
            print(f"  {prev[0]} -> {r['w']} workers: speedup {prev[1]:.2f} -> {sp:.2f}"
                  f"  (+{sp - prev[1]:.2f}), wall {prev[2]:.1f} -> {r['wall']:.1f} s")
        prev = (r["w"], sp, r["wall"])

    print("\ntile outcomes (must be identical across worker counts):")
    for r in rows:
        print(f"  workers={r['w']}: {r['stats']}  cells={r['cells']}")

    print("\nNOTE: per-function TIMINGS are parent-only and therefore empty for "
          "workers>1;\n      arm_report.py does not apply to these runs (see this "
          "script's docstring).")


if __name__ == "__main__":
    main()
