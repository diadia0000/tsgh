"""Analyse the gc-cadence ablation runs (docs/hybrid-pipeline/14-gc-collect-frequency-plan.md §4).

Reads the *_timings.json / *_resource.csv that scripts/perf_measure.py emits for each
experiment cell and prints three things the plan requires before anything is adoptable:

  1. the wall-clock ablation table (end-to-end and run_batch, plus the B4 gc buckets),
  2. the two-part RSS check from §2 -- absolute peak, and the growth-shape evidence,
  3. the correctness veto -- per-cell reddot/blackdot/score against a reference
     report.csv, matched by nearest centroid.

Usage:
  python scripts/gc_ablation_report.py --metrics-dir <dir> [--runs-dir <dir>]
      [--reference <report.csv>] [--baseline <label>]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


# ---------------------------------------------------------------- timings
def load_runs(metrics_dir: Path) -> dict[str, dict]:
    runs = {}
    for f in sorted(metrics_dir.glob("*_timings.json")):
        d = json.load(open(f))
        runs[d["label"]] = d
    return runs


def bucket(d: dict, name: str) -> tuple[int, float]:
    b = d["timings"].get(name) or {}
    return b.get("n", 0), b.get("t", 0.0)


def cadence(d: dict) -> str:
    """Label a run's GC configuration.

    `gc_cadence` was emitted only while the Option A knobs existed (see doc 16 §5);
    they were deleted after the ablation, so current runs have no such field and are
    labelled by their gc.collect call count instead.
    """
    gc_ = d.get("gc_cadence")
    if gc_:
        return (f"N={gc_['gc_collect_every_n_tiles']} "
                f"freeze={'on' if gc_['gc_freeze_after_init'] else 'off'}")
    n, _ = bucket(d, "B4_gc_collect")
    per_tile = "every tile" if n >= d.get("n_tiles", 0) else f"{n} sweeps"
    return f"{per_tile}"


def timing_table(runs: dict[str, dict], baseline: str | None) -> None:
    print("\n" + "=" * 100)
    print("WALL-CLOCK ABLATION  (judged end-to-end, per playbook step 4)")
    print("=" * 100)
    base = runs.get(baseline) if baseline else None
    hdr = (f"{'label':<20}{'cadence':<24}{'tiles':>6}{'e2e s':>9}{'runbatch s':>12}"
           f"{'gc n':>6}{'gc s':>8}{'ec s':>7}{'Δe2e':>9}{'Δrunb':>9}")
    print(hdr)
    print("-" * len(hdr))
    for label, d in runs.items():
        gn, gt = bucket(d, "B4_gc_collect")
        _, et = bucket(d, "B4_empty_cache")
        e2e = d["wall"]["end_to_end_total_s"]
        rb = d["wall"]["runbatch_BCD_s"]
        de = dr = ""
        if base is not None and d is not base:
            de = f"{e2e - base['wall']['end_to_end_total_s']:+.1f}"
            dr = f"{rb - base['wall']['runbatch_BCD_s']:+.1f}"
        print(f"{label:<20}{cadence(d):<24}{d['n_tiles']:>6}{e2e:>9.1f}{rb:>12.1f}"
              f"{gn:>6}{gt:>8.1f}{et:>7.1f}{de:>9}{dr:>9}")


# ---------------------------------------------------------------- RSS
def rss_series(path: Path) -> list[tuple[float, float]]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append((float(row["t_s"]), float(row["rss_gb"])))
    return out


def rss_check(runs: dict[str, dict], metrics_dir: Path, baseline: str | None) -> None:
    print("\n" + "=" * 100)
    print("RSS CHECK (plan §2): 1) absolute peak  2) growth shape -- sawtooth vs monotonic ramp")
    print("=" * 100)
    base_peak = runs[baseline]["peak_rss_gb"] if baseline in runs else None
    hdr = (f"{'label':<20}{'peak GB':>9}{'Δpeak':>8}{'end GB':>8}"
           f"{'max drop GB':>13}{'n drops>50MB':>14}{'ramp frac':>11}")
    print(hdr)
    print("-" * len(hdr))
    for label, d in runs.items():
        s = rss_series(metrics_dir / f"{label}_resource.csv")
        peak = d.get("peak_rss_gb", 0.0)
        dp = f"{peak - base_peak:+.3f}" if base_peak is not None else ""
        if not s:
            print(f"{label:<20}{peak:>9.3f}{dp:>8}{'-':>8}{'-':>13}{'-':>14}{'-':>11}")
            continue
        vals = [v for _t, v in s]
        # Drawdowns: a sawtooth reclaims memory repeatedly; a monotonic ramp does not.
        drops, maxdrop, rising = [], 0.0, 0
        for a, b in zip(vals, vals[1:]):
            d_ = a - b
            if d_ > 0.05:
                drops.append(d_)
            maxdrop = max(maxdrop, d_)
            if b >= a:
                rising += 1
        ramp = rising / max(1, len(vals) - 1)
        print(f"{label:<20}{peak:>9.3f}{dp:>8}{vals[-1]:>8.3f}"
              f"{maxdrop:>13.3f}{len(drops):>14}{ramp:>11.2f}")
    print("\n  'max drop' / 'n drops' = memory actually handed back during the run (sawtooth "
          "evidence).\n  'ramp frac' = fraction of samples that did not decrease; ~1.00 with zero "
          "drops is the\n  monotonic-ramp signature the plan says to reject. Note per_tile_owned "
          "accumulates by\n  design, so some upward trend is expected regardless of gc cadence.")


# ---------------------------------------------------------------- correctness
def load_report(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def fnum(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def compare_report(ref_path: Path, got_path: Path, tol_px: float = 3.0) -> str:
    ref, got = load_report(ref_path), load_report(got_path)
    # Bucketed nearest-centroid match, tolerance tol_px.
    cell = tol_px
    grid: dict[tuple[int, int], list[dict]] = {}
    for r in ref:
        x, y = float(r["centroid_x"]), float(r["centroid_y"])
        grid.setdefault((int(x // cell), int(y // cell)), []).append(r)

    matched = 0
    diffs = {"reddot": [], "blackdot": [], "score": []}
    nan_mismatch = 0
    for g in got:
        x, y = float(g["centroid_x"]), float(g["centroid_y"])
        gx, gy = int(x // cell), int(y // cell)
        best, bestd = None, tol_px
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for r in grid.get((gx + dx, gy + dy), ()):
                    d = math.hypot(float(r["centroid_x"]) - x, float(r["centroid_y"]) - y)
                    if d < bestd:
                        best, bestd = r, d
        if best is None:
            continue
        matched += 1
        for k in diffs:
            a, b = fnum(best[k]), fnum(g[k])
            if (a is None) != (b is None):
                nan_mismatch += 1
            elif a is not None and b is not None:
                diffs[k].append(abs(a - b))

    parts = [f"cells ref={len(ref)} got={len(got)} matched={matched}"]
    for k, v in diffs.items():
        parts.append(f"{k}: max|Δ|={max(v) if v else 0:.3g} n≠0={sum(1 for x in v if x > 1e-9)}")
    parts.append(f"excluded(X)-flips={nan_mismatch}")
    return "  " + "\n  ".join(parts)


def correctness(runs: dict[str, dict], runs_dir: Path, reference: Path | None) -> None:
    if reference is None or not reference.exists():
        print("\n(correctness: no --reference given, skipped)")
        return
    print("\n" + "=" * 100)
    print(f"CORRECTNESS VETO vs {reference}")
    print("=" * 100)
    for label in runs:
        p = runs_dir / label / "report.csv"
        if not p.exists():
            print(f"\n{label}: report.csv missing")
            continue
        print(f"\n{label}:")
        print(compare_report(reference, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--baseline", default=None,
                    help="label to use as the delta reference in the tables")
    a = ap.parse_args()

    md = Path(a.metrics_dir)
    runs = load_runs(md)
    if not runs:
        raise SystemExit(f"no *_timings.json under {md}")
    baseline = a.baseline or next(iter(runs))

    timing_table(runs, baseline)
    rss_check(runs, md, baseline)
    if a.runs_dir:
        correctness(runs, Path(a.runs_dir), Path(a.reference) if a.reference else None)


if __name__ == "__main__":
    main()
