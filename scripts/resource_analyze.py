"""Summarize GPU/RAM/VRAM time series from perf_measure outputs.

Reads:
  <label>_gpu_dmon.txt   nvidia-smi dmon (-s um): sm% (compute), mem% (mem ctrl), fb MB
  <label>_resource.csv   process RSS + torch cuda alloc/reserved, sampled ~0.5s
Emits per-label stats to <label>_resource_summary.json: GPU idle fraction,
mean/median busy, VRAM peak, RAM peak/curve — i.e. the "utilization timeline"
rather than a single average (plan sec 4.1/4.3).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def parse_dmon(p: Path):
    sm, mem, fb = [], [], []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        c = line.split()
        # cols: Date Time gpu sm mem enc dec jpg ofa fb bar1 ccpm
        if len(c) < 12:
            continue
        try:
            sm.append(int(c[3])); mem.append(int(c[4])); fb.append(int(c[9]))
        except ValueError:
            continue
    return sm, mem, fb


def pct(xs, q):
    if not xs:
        return 0
    s = sorted(xs)
    i = int(q * (len(s) - 1))
    return s[i]


def analyze(label: str, mdir: Path):
    out = {"label": label}
    dmon = mdir / f"{label}_gpu_dmon.txt"
    if dmon.exists():
        sm, mem, fb = parse_dmon(dmon)
        n = len(sm) or 1
        out["gpu"] = {
            "samples_1s": len(sm),
            "sm_mean_pct": round(sum(sm) / n, 1),
            "sm_median_pct": pct(sm, 0.5),
            "sm_p90_pct": pct(sm, 0.9),
            "sm_idle_frac": round(sum(1 for v in sm if v == 0) / n, 3),
            "sm_busy_ge50_frac": round(sum(1 for v in sm if v >= 50) / n, 3),
            "memctrl_mean_pct": round(sum(mem) / n, 1),
            "vram_peak_mb": max(fb) if fb else 0,
            "vram_mean_mb": round(sum(fb) / n) if fb else 0,
        }
    res = mdir / f"{label}_resource.csv"
    if res.exists():
        rows = [l.split(",") for l in res.read_text().splitlines()[1:] if l.strip()]
        rss = [float(r[1]) for r in rows]
        alloc = [float(r[2]) for r in rows]
        reserved = [float(r[3]) for r in rows]
        out["mem"] = {
            "samples": len(rows),
            "rss_peak_gb": round(max(rss), 3) if rss else 0,
            "rss_start_gb": round(rss[0], 3) if rss else 0,
            "rss_end_gb": round(rss[-1], 3) if rss else 0,
            "cuda_alloc_peak_gb": round(max(alloc), 3) if alloc else 0,
            "cuda_reserved_peak_gb": round(max(reserved), 3) if reserved else 0,
        }
    return out


if __name__ == "__main__":
    mdir = Path(sys.argv[1])
    labels = sorted({f.name.rsplit("_gpu_dmon", 1)[0].rsplit("_resource", 1)[0]
                     for f in mdir.glob("*_gpu_dmon.txt")}
                    | {f.name.rsplit("_resource.csv", 1)[0] for f in mdir.glob("*_resource.csv")})
    allout = {}
    for lb in labels:
        a = analyze(lb, mdir)
        allout[lb] = a
        (mdir / f"{lb}_resource_summary.json").write_text(json.dumps(a, indent=2))
        print(json.dumps(a, indent=2))
    (mdir / "all_resource_summary.json").write_text(json.dumps(allout, indent=2))
