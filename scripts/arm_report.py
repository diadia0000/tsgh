"""Decompose perf_measure.py runs into the two-arm model (doc 13 §2 / doc 17 §5).

`run_batch` runs two concurrent arms, so "self-time ÷ wall" is not a critical-path
share and cannot be used to rank. This script computes what can:

  wall ≈ max(MAIN, BG) + outside

per label, plus the margin BG/MAIN (how much MAIN must shed before the background CPU
arm becomes the new critical path) and, when a `*_gpu_dmon.txt` is present, the GPU
idle fraction the margin is competing for.

Usage:
  python scripts/arm_report.py --metrics-dir <dir> [--group]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

# Arm membership is a property of the code, not of the timings -- confirmed by reading
# hybrid_pipeline.py (main thread vs the single `tile-cpu` ThreadPoolExecutor worker).
MAIN = {
    "B1_unet_coremask": "GPU forward (UNet++)",
    "B1_m3b_cellpose": "GPU forward (Cellpose M2+M3b)",
    "B2r_tile_read": "_read_rgb",
    "BM1_apply_mask": "M1 overlay glue",
    "BM1_overlay_dish": "M1 overlay glue",
    "BM1_fuse": "M1 overlay glue",
    "Bs_clear_edge": "clear_slide_edge_cells",
    "B3_build_results": "(8) build_all_positive_results",
    "B3_enlarge_cells": "(8) enlarge_cell_instances",
    "B4_gc_collect": "gc.collect",
    "B4_empty_cache": "empty_cache",
}
BG = {
    "B3_detect_dots": "detect_all_dots",
    "B3_merge_dots": "merge dots",
    "Bs_filter_absolutize": "filter_and_absolutize",
    "B2_png_encode": "PNG encode",
    "B2_tiff_encode": "TIFF encode",
    "B2_render_overlay": "render_overlay_image",
    "B2_percell_crops": "per-cell crops",
}
OUTSIDE = {
    "D_stitch_overlay": "stitch D",
    "init_unet": "model init",
    "init_cellpose_m2": "model init",
    "init_cellpose_m3b": "model init",
    "C_export_csv": "global CSV",
    "C_export_summary": "global summary",
}


def t(d: dict, key: str) -> float:
    return (d["timings"].get(key) or {}).get("t", 0.0)


def dmon_stats(path: Path) -> tuple[float, float, int, float] | None:
    """idle_frac, mean SM%, n samples, near-idle frac -- from `nvidia-smi dmon -s um`.

    `idle_frac` counts only samples reading exactly SM==0, which is how every previous
    round recorded it. That definition is knife-edge: a 1-second sample that reads SM=1%
    is idle in substance but not by this count, so the metric can swing several points
    between configurations that merely redistributed near-zero samples. `near_idle`
    (SM<=3) is reported alongside it as the stable version -- prefer it when comparing
    runs, and prefer the cuda-Event gaps over either when the question is "how long was
    the device provably doing nothing".
    """
    if not path.exists():
        return None
    sm = []
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        f = line.split()
        if len(f) < 5:
            continue
        try:
            sm.append(int(f[3]))
        except ValueError:
            continue
    if not sm:
        return None
    return (sum(1 for v in sm if v == 0) / len(sm), statistics.fmean(sm), len(sm),
            sum(1 for v in sm if v <= 3) / len(sm))


def arms(d: dict) -> dict:
    main = sum(t(d, k) for k in MAIN)
    bg = sum(t(d, k) for k in BG)
    outside = sum(t(d, k) for k in OUTSIDE) + d["wall"]["phaseA_precut_s"]
    e2e = d["wall"]["end_to_end_total_s"]
    pred = max(main, bg) + outside
    return {
        "e2e": e2e,
        "runbatch": d["wall"]["runbatch_BCD_s"],
        "main": main,
        "bg": bg,
        "outside": outside,
        "pred": pred,
        "err_pct": (pred - e2e) / e2e * 100.0,
        "ratio": bg / main if main else 0.0,
        "shed_pct": (1 - bg / main) * 100.0 if main else 0.0,
    }


def breakdown(d: dict, members: dict, wall: float) -> list[tuple[str, float, float]]:
    agg: dict[str, float] = {}
    for k, label in members.items():
        agg[label] = agg.get(label, 0.0) + t(d, k)
    return sorted(
        ((lab, s, s / wall * 100) for lab, s in agg.items() if s > 0),
        key=lambda r: -r[1],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--group", action="store_true",
                    help="average labels sharing a `_r<N>` repeat suffix")
    ap.add_argument("--detail", action="store_true", help="per-arm component tables")
    ap.add_argument("--moved", default="",
                    help="comma-separated buckets to reclassify MAIN->BG for labels "
                         "matching --moved-labels (arm membership is a property of the "
                         "code version, so it must be stated, not guessed)")
    ap.add_argument("--moved-labels", default="",
                    help="comma-separated label prefixes the --moved reclassification "
                         "applies to (an over-broad prefix silently mis-attributes other "
                         "configs, so list them exactly)")
    args = ap.parse_args()

    moved = [b for b in args.moved.split(",") if b]
    if moved and not args.moved_labels:
        raise SystemExit("--moved requires --moved-labels")

    md = Path(args.metrics_dir)
    runs = {}
    for f in sorted(md.glob("*_timings.json")):
        d = json.load(open(f))
        runs[d["label"]] = d
    if not runs:
        raise SystemExit(f"no *_timings.json in {md}")

    print(f"\n{'label':<18}{'tiles':>6}{'e2e s':>9}{'MAIN s':>9}{'BG s':>9}"
          f"{'out s':>8}{'pred s':>9}{'err%':>7}{'BG/MAIN':>9}{'shed%':>7}"
          f"{'idle':>7}{'nidle':>7}{'SM%':>6}")
    print("-" * 113)
    rows = {}
    for label, d in runs.items():
        if moved and label.startswith(tuple(
                x for x in args.moved_labels.split(",") if x)):
            for b in moved:
                BG[b] = MAIN.pop(b, b)
            a = arms(d)
            for b in moved:
                MAIN[b] = BG.pop(b)
        else:
            a = arms(d)
        rows[label] = a
        g = dmon_stats(md / f"{label}_gpu_dmon.txt")
        idle = f"{g[0]:.3f}" if g else "-"
        nidle = f"{g[3]:.3f}" if g else "-"
        smv = f"{g[1]:.1f}" if g else "-"
        print(f"{label:<18}{d['n_tiles']:>6}{a['e2e']:>9.1f}{a['main']:>9.1f}"
              f"{a['bg']:>9.1f}{a['outside']:>8.1f}{a['pred']:>9.1f}"
              f"{a['err_pct']:>7.2f}{a['ratio']:>9.3f}{a['shed_pct']:>7.1f}"
              f"{idle:>7}{nidle:>7}{smv:>6}")

    if args.group:
        groups: dict[str, list[str]] = {}
        for label in rows:
            groups.setdefault(re.sub(r"_r\d+$", "", label), []).append(label)
        print(f"\n{'group':<18}{'n':>3}{'e2e mean':>10}{'e2e range':>18}"
              f"{'MAIN mean':>11}{'BG mean':>10}{'shed% mean':>12}")
        print("-" * 82)
        for g, labels in sorted(groups.items()):
            e = [rows[x]["e2e"] for x in labels]
            print(f"{g:<18}{len(labels):>3}{statistics.fmean(e):>10.1f}"
                  f"{f'{min(e):.1f}-{max(e):.1f}':>18}"
                  f"{statistics.fmean([rows[x]['main'] for x in labels]):>11.1f}"
                  f"{statistics.fmean([rows[x]['bg'] for x in labels]):>10.1f}"
                  f"{statistics.fmean([rows[x]['shed_pct'] for x in labels]):>12.1f}")

    if args.detail:
        for label, d in runs.items():
            a = rows[label]
            print(f"\n=== {label} (wall {a['e2e']:.1f} s) ===")
            for arm_name, members, tot in (
                ("MAIN", MAIN, a["main"]), ("BG", BG, a["bg"]),
            ):
                print(f"  {arm_name} arm")
                for lab, s, pct in breakdown(d, members, a["e2e"]):
                    print(f"    {lab:<34}{s:>9.2f} s{pct:>8.2f}%")
                print(f"    {'TOTAL':<34}{tot:>9.2f} s{tot / a['e2e'] * 100:>8.2f}%")

    # cuda-Event buckets, when the run was made with --cuda-events
    for label, d in runs.items():
        ev = {k: v for k, v in d["timings"].items() if k.startswith("E_")}
        if not ev:
            continue
        e2e = d["wall"]["end_to_end_total_s"]
        print(f"\n=== {label}: GPU-timeline events (device-side, {e2e:.1f} s wall) ===")
        span = sum(v["t"] for k, v in ev.items() if k.startswith("E_busy_"))
        gaps = sum(v["t"] for k, v in ev.items() if k.startswith("E_gap_"))
        for k, v in sorted(ev.items(), key=lambda r: -r[1]["t"]):
            print(f"  {k:<46}{v['t']:>9.2f} s{v['t'] / e2e * 100:>8.2f}%"
                  f"{v['t'] / max(v['n'], 1) * 1000:>10.1f} ms/call  n={v['n']}")
        # E_busy_* is the device-timeline SPAN of a forward, not its utilisation: each
        # call is kernel-launch-bound internally, so the device idles inside it too.
        # Only E_gap_* is provably idle -- no kernels are enqueued between forwards.
        print(f"  {'-- in-forward span (NOT utilisation)':<46}{span:>9.2f} s"
              f"{span / e2e * 100:>8.2f}%")
        print(f"  {'-- inter-forward idle (provable)':<46}{gaps:>9.2f} s"
              f"{gaps / e2e * 100:>8.2f}%")
        g = dmon_stats(md / f"{label}_gpu_dmon.txt")
        if g:
            tot_idle = g[3] * e2e
            print(f"  {'-- total device idle (dmon SM<=3)':<46}{tot_idle:>9.2f} s"
                  f"{g[3] * 100:>8.2f}%")
            print(f"  {'   of which intra-forward (launch-bound)':<46}"
                  f"{tot_idle - gaps:>9.2f} s{(tot_idle - gaps) / e2e * 100:>8.2f}%")


if __name__ == "__main__":
    main()
