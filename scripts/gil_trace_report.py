"""B4: aggregate a py-spy raw (collapsed-stack) profile into the buckets doc 22 §3 asks about.

Executes docs/hybrid-pipeline/22-next-optimization-cycle-plan.md §3 (B4): the
"launch-bound, not placement-bound" diagnosis and the ~1.23x ceiling behind
open-backlog item 5 were traced against `cellpose==4.0.8` + the SAM ViT backbone.
Round 3 moved to `cellpose==4.2.1.1` with the `cpdino`/DINOv3 backbone, so both the
line numbers and the backbone's internal shape may have moved. This re-derives the
same numbers from a fresh trace instead of assuming they carried over.

Collect the trace with (GIL-holding samples, and separately all-thread wall samples):
  .venv/bin/py-spy record --gil    --format raw --rate 100 -o gil.raw  -- <workload>
  .venv/bin/py-spy record --idle   --format raw --rate 100 -o wall.raw -- <workload>

Then:
  .venv/bin/python scripts/gil_trace_report.py --raw gil.raw [--raw wall.raw] --out out.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# The five functions the 4.0.8 trace named (gil-contention-diag.md "追加深挖").
TRACKED = [
    "_extend_centers_gpu",
    "get_masks_torch",
    "steps_interp",
    "get_rel_pos",
    "fill_holes_and_remove_small_masks",
]

BUCKETS = [
    ("cellpose.dynamics", re.compile(r"cellpose[/\\]dynamics\.py")),
    ("cellpose.core", re.compile(r"cellpose[/\\]core\.py")),
    ("cellpose.models", re.compile(r"cellpose[/\\]models\.py")),
    ("cellpose.other", re.compile(r"cellpose[/\\]")),
    ("dinov3", re.compile(r"dinov3")),
    ("segment_anything", re.compile(r"segment_anything")),
    ("timm", re.compile(r"timm[/\\]")),
    ("torch", re.compile(r"torch[/\\]")),
    ("project.m3_dots", re.compile(r"m3_dot_")),
    ("project.hybrid", re.compile(r"algorithms[/\\]hybrid")),
    ("joblib", re.compile(r"joblib")),
    ("skimage/scipy/numpy", re.compile(r"(skimage|scipy|numpy)[/\\]")),
    ("pyvips/imageio", re.compile(r"(pyvips|imageio|tifffile)")),
]


def bucket_of(frame: str) -> str:
    for name, rx in BUCKETS:
        if rx.search(frame):
            return name
    return "other"


def parse(raw_path: Path):
    """py-spy raw = one collapsed stack per line, ending in a sample count."""
    self_c, incl_c, bucket_self = Counter(), Counter(), Counter()
    tracked_incl, tracked_self = Counter(), Counter()
    total = 0
    for line in raw_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        stack, count = parts[0], int(parts[1])
        frames = stack.split(";")
        total += count
        leaf = frames[-1]
        self_c[leaf] += count
        bucket_self[bucket_of(leaf)] += count
        seen = set()
        for f in frames:
            b = bucket_of(f)
            if b not in seen:                 # inclusive: count a bucket once per stack
                incl_c[b] += count
                seen.add(b)
        for name in TRACKED:
            if any(name in f for f in frames):
                tracked_incl[name] += count
            if name in leaf:
                tracked_self[name] += count
    return {
        "total_samples": total,
        "self_by_bucket": dict(bucket_self.most_common()),
        "inclusive_by_bucket": dict(incl_c.most_common()),
        "top_leaves": dict(self_c.most_common(25)),
        "tracked_inclusive": dict(tracked_incl),
        "tracked_self": dict(tracked_self),
    }


def report(name: str, d: dict) -> None:
    tot = d["total_samples"]
    print(f"\n===== {name}: {tot} samples =====")
    print("  self time by bucket (where the interpreter actually was):")
    for b, c in list(d["self_by_bucket"].items())[:12]:
        print(f"    {b:<22} {c:>8}  {c/tot:6.2%}")
    print("  inclusive by bucket (stack passed through it):")
    for b, c in list(d["inclusive_by_bucket"].items())[:8]:
        print(f"    {b:<22} {c:>8}  {c/tot:6.2%}")
    print(f"  the five functions named in the 4.0.8 trace:")
    for fn in TRACKED:
        i, s = d["tracked_inclusive"].get(fn, 0), d["tracked_self"].get(fn, 0)
        flag = "" if i else "   <-- NOT PRESENT in this trace"
        print(f"    {fn:<36} incl {i:>7} ({i/tot:6.2%})  self {s:>7} ({s/tot:6.2%}){flag}")
    print("  top leaf frames:")
    for f, c in list(d["top_leaves"].items())[:12]:
        print(f"    {c:>7} {c/tot:6.2%}  {f[:110]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", required=True,
                    help="py-spy raw file; pass twice (gil.raw, wall.raw)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = {}
    for p in args.raw:
        path = Path(p)
        d = parse(path)
        out[path.stem] = d
        report(path.stem, d)

    # The ceiling the backlog item quotes: what if every cellpose-internal Python
    # second went to zero? Computed from the wall trace's cellpose self share.
    for name, d in out.items():
        if "wall" not in name:
            continue
        tot = d["total_samples"]
        cp = sum(c for b, c in d["self_by_bucket"].items() if b.startswith("cellpose"))
        p = cp / tot if tot else 0
        print(f"\n  [{name}] cellpose-internal self share = {p:.2%} "
              f"-> Amdahl ceiling if driven to zero = {1/(1-p):.3f}x")
        d["cellpose_self_share"] = round(p, 4)
        d["amdahl_ceiling_if_zero"] = round(1 / (1 - p), 3) if p < 1 else None

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
