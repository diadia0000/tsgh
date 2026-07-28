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

Round 9 (docs/hybrid-pipeline/28-gc-collect-round2-plan.md Option H) added a
*periodic* re-freeze: `_frozen_gc_generation()` now yields a `_PeriodicFreezer`
that calls `gc.freeze()` again every `_REFREEZE_EVERY_CELLS` accumulated cells,
so results appended after the first freeze stop being rescanned. Freezing is
additive and needs no matching unfreeze, so invariant 2 becomes an asymmetric
cardinality -- **freeze N>=1, unfreeze exactly once** -- and that is what this
script now asserts. Invariant 1 (one collect per tile) is unchanged, which is
the whole point: Option H moves scan *scope*, never cadence.

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

from hybrid_data_types import CellAnalysisResult  # noqa: E402
# _REFREEZE_EVERY_CELLS is read from the pipeline rather than duplicated, so retuning
# the cadence cannot silently invalidate the expected freeze count below.
from m0_module import m0_tile_runner as TR  # noqa: E402


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


def stub(HP, positions, cells_per_tile=0):
    """Replace the heavy stages; the tile loop and geometry math stay real.

    `cells_per_tile` fabricates the per-tile result list run_batch accumulates. It
    must be non-zero to exercise the round-9 periodic re-freeze at all, since that
    cadence is keyed off accumulated cell count -- a batch of empty tiles never
    crosses the threshold, which is correct behaviour and also a silent no-op.
    """
    pairs = [
        (Path(f"/ihc/tile_x{x}_y{y}.tiff"), Path(f"/dish/tile_x{x}_y{y}.tiff"))
        for x, y in positions
    ]
    HP.find_paired_tiles = lambda *a, **k: pairs
    # Round 9 (doc 30 Option L) moved the two _read_rgb calls out of
    # _process_precut_tile_gpu and onto a one-tile-ahead prefetch, so stubbing the GPU
    # stage no longer stops the loop from touching disk -- the fabricated paths must be
    # stubbed at the read too, or every tile fails to open.
    TR._read_tile_pair = lambda ihc, dish: (None, None)
    HP._init_unet_inferencer = lambda: object()
    HP._init_cellpose_segmenter = lambda: object()
    HP._init_dish_cellpose_segmenter = lambda: object()
    # Must be non-None or run_batch fail-fasts; the loop never inspects the contents.
    HP._process_precut_tile_gpu = lambda *a, **k: object()
    HP._process_precut_tile_cpu = lambda *a, **k: [
        CellAnalysisResult(cell_id=i, centroid_x=0.0, centroid_y=0.0,
                           is_her2_positive=False)
        for i in range(cells_per_tile)
    ]
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


def run(HP, positions, out: Path, cells_per_tile=0) -> GcCounter:
    stub(HP, positions, cells_per_tile)
    with GcCounter() as c:
        HP.run_batch(out / "ihc", out / "dish", out, workers=1)
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-ref", default=None,
                    help="git ref whose hybrid_pipeline.py predates the freeze change, "
                         "run as a control group for invariant 1. Optional: no such ref "
                         "survives in this repo's history any more (it was squashed "
                         "away), and the invariant -- one collect per tile -- is "
                         "asserted directly against the tile count regardless")
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
        HP_ctl = load_control(args.control_ref, work) if args.control_ref else None

        # 21x21 = 441 tiles: the large anchor the measurements were taken at.
        positions = grid(21, 21)
        total = len(positions)

        # Enough cells per tile that the round-9 cell-count-keyed cadence fires
        # several times across the anchor, i.e. the real slide's 29.24 (doc 27 §6.4).
        cells = 29
        n_expected_refreezes = total * cells // TR._REFREEZE_EVERY_CELLS

        print(f"\n=== 1. cadence unchanged (control: {args.control_ref or 'none'}) ===")
        cur = run(HP, positions, out, cells)
        check("current collects once per tile (freeze changes scope, not cadence)",
              cur.n, total)
        if HP_ctl is not None:
            base = run(HP_ctl, positions, out, cells)
            check("control collects once per tile", base.n, total)
            check("current matches control", cur.n, base.n)

        print("\n=== 2. freeze/unfreeze cardinality (API-server leak guard) ===")
        # Round 9: freezing is additive and needs no matching unfreeze, so the
        # invariant is asymmetric -- N freezes, exactly one unfreeze.
        check("freeze called once at entry plus once per cadence crossing",
              cur.freezes, 1 + n_expected_refreezes)
        check("periodic re-freeze actually fired", cur.freezes > 1, True)
        check("unfreeze called exactly once", cur.unfreezes, 1)
        check("nothing left frozen after run_batch", gc.get_freeze_count(), 0)

        print("\n=== 2b. no cells accumulated -> no re-freeze (cadence is cell-keyed) ===")
        empty = run(HP, positions, out, 0)
        check("all-background batch freezes only at entry", empty.freezes, 1)
        check("unfreeze still exactly once", empty.unfreezes, 1)

        print("\n=== 3. unfreeze survives a mid-loop fail-fast ===")
        # Fail *after* several re-freezes rather than on tile 1, so the fail-fast path
        # is exercised with the round-9 state actually built up.
        stub(HP, positions, cells)
        seen = {"n": 0}

        def _fail_midway(*a, **k):
            seen["n"] += 1
            return None if seen["n"] > total // 2 else object()

        HP._process_precut_tile_gpu = _fail_midway
        raised = False
        with GcCounter() as fc:
            try:
                HP.run_batch(out / "ihc", out / "dish", out, workers=1)
            except RuntimeError:
                raised = True
        check("fail-fast still raises", raised, True)
        check("re-freezes happened before the failure", fc.freezes > 1, True)
        check("unfreeze called exactly once on the fail path", fc.unfreezes, 1)
        check("nothing left frozen after fail-fast", gc.get_freeze_count(), 0)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
