"""Guard the invariants of the gc.freeze() optimisation in run_batch.

Adopted change (docs/hybrid-pipeline/16-gc-collect-frequency-result.md): run_batch
wraps its tile loop in `_frozen_gc_generation()`, moving the resident model object
graph into the GC's permanent generation so each per-tile `gc.collect()` stops
re-scanning it (83.2 ms -> 1.2 ms per call at the 441-tile anchor).

Two things must never regress, and neither is visible in a wall-clock number:

  1. **Cadence is unchanged.** The optimisation reduces per-call *scan cost*, not
     call frequency. gc.collect() must still run once per tile, so the
     memory-bounded invariant is untouched. Verified against the pre-change
     hybrid_pipeline.py read out of git, as a control group.
  2. **freeze is always paired with unfreeze**, including when the batch
     fail-fasts. run_batch is called repeatedly inside a long-lived API server
     (backend/api/hybrid.py), and gc.freeze() without unfreeze would permanently
     pin whatever was tracked at that moment -- RSS growth unbounded in request
     count, invisible to any single-batch benchmark.

Drives the real run_batch loop with only the heavy stages stubbed, so no GPU,
model weights, or slide data are needed. Not a pipeline correctness test.

Usage:
  python scripts/verify_gc_freeze.py [--control-ref GIT_REF]
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HYBRID = REPO / "backend" / "algorithms" / "hybrid"
PIPELINE_REL = "backend/algorithms/hybrid/hybrid_pipeline.py"

sys.path.insert(0, str(HYBRID))
sys.path.insert(0, str(REPO))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_control(ref: str, workdir: Path):
    """Load hybrid_pipeline.py as of `ref` (must predate the freeze change)."""
    src = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{ref}:{PIPELINE_REL}"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "_frozen_gc_generation" in src:
        sys.exit(f"--control-ref {ref} already contains the freeze change; it cannot "
                 f"serve as the pre-change control. Pass the SHA before the change.")
    path = workdir / "hp_control.py"
    path.write_text(src)
    return load("hp_control", path)


def grid(n_cols: int, n_rows: int, stride: int = 768):
    """Positions on the real precut grid (tile 1024 / overlap 256 -> stride 768)."""
    return [(c * stride, r * stride) for r in range(n_rows) for c in range(n_cols)]


def stub(HP, positions):
    """Replace the heavy stages; the tile loop and geometry math stay real."""
    pairs = [
        (Path(f"/ihc/tile_x{x}_y{y}.tiff"), Path(f"/dish/tile_x{x}_y{y}.tiff"))
        for x, y in positions
    ]
    HP.find_paired_tiles = lambda *a, **k: pairs
    HP._init_unet_inferencer = lambda: object()
    HP._init_cellpose_segmenter = lambda: object()
    HP._init_dish_cellpose_segmenter = lambda: object()
    # Must be non-None or run_batch fail-fasts; the loop never inspects the contents.
    HP._process_precut_tile_gpu = lambda *a, **k: object()
    HP._process_precut_tile_cpu = lambda *a, **k: []
    HP.export_tile_csv = lambda *a, **k: None
    HP.export_summary_statistics = lambda *a, **k: None
    HP._stitch_overlay_slide = lambda *a, **k: None


class GcCounter:
    """Count gc.collect/freeze/unfreeze, mirroring perf_measure.py's shim."""

    def __init__(self):
        self.n = self.freezes = self.unfreezes = 0

    def __enter__(self):
        self._orig = (gc.collect, gc.freeze, gc.unfreeze)

        def collect(*a, **k):
            self.n += 1
            return self._orig[0](*a, **k)

        def freeze(*a, **k):
            self.freezes += 1
            return self._orig[1](*a, **k)

        def unfreeze(*a, **k):
            self.unfreezes += 1
            return self._orig[2](*a, **k)

        gc.collect, gc.freeze, gc.unfreeze = collect, freeze, unfreeze
        return self

    def __exit__(self, *exc):
        gc.collect, gc.freeze, gc.unfreeze = self._orig


def run(HP, positions, out: Path) -> GcCounter:
    stub(HP, positions)
    with GcCounter() as c:
        HP.run_batch(out / "ihc", out / "dish", out)
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-ref", default="HEAD",
                    help="git ref whose hybrid_pipeline.py predates the freeze "
                         "change (default HEAD: correct only while uncommitted)")
    args = ap.parse_args()

    # The loop logs one INFO line per tile; several 441-tile runs would bury the report.
    logging.disable(logging.INFO)
    failures: list[str] = []

    def check(label, got, want):
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got}, want {want}")
        if not ok:
            failures.append(label)

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        out = work / "out"
        out.mkdir()

        HP = load("hybrid_pipeline", HYBRID / "hybrid_pipeline.py")
        HP_ctl = load_control(args.control_ref, work)

        # 21x21 = 441 tiles: the large anchor the measurements were taken at.
        positions = grid(21, 21)
        total = len(positions)

        print(f"\n=== 1. cadence unchanged vs control ({args.control_ref}) ===")
        base = run(HP_ctl, positions, out)
        cur = run(HP, positions, out)
        check("control collects once per tile", base.n, total)
        check("current collects once per tile (freeze changes cost, not cadence)",
              cur.n, base.n)

        print("\n=== 2. freeze paired with unfreeze (API-server leak guard) ===")
        check("freeze called exactly once", cur.freezes, 1)
        check("unfreeze called exactly once", cur.unfreezes, 1)
        check("nothing left frozen after run_batch", gc.get_freeze_count(), 0)

        print("\n=== 3. unfreeze survives a mid-loop fail-fast ===")
        stub(HP, positions)
        HP._process_precut_tile_gpu = lambda *a, **k: None  # trigger fail-fast
        raised = False
        try:
            HP.run_batch(out / "ihc", out / "dish", out)
        except RuntimeError:
            raised = True
        check("fail-fast still raises", raised, True)
        check("nothing left frozen after fail-fast", gc.get_freeze_count(), 0)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
