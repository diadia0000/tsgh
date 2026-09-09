"""Golden-output regression for the hybrid pipeline on the bundled test pair.

Runs `_run_single_tile_cli` -- the `--test` path, and with a PrecutStream the
same path `/api/hybrid/tile` takes -- and compares the three artifacts against
a baseline recorded on this machine. The pipeline is not bit-reproducible on a
GPU: between two runs of *unchanged* code ~0.06% of cells land a few pixels
away or flip class, so the comparison is tolerant exactly where that jitter
lives and strict everywhere else:

- report.csv: cell_id is 1..N; row count within 1% of baseline; >= 99.5% of
  baseline cells have a produced centroid within 2 px with identical dot
  counts / score / class.
- summary.txt: same lines; each number within 2 (absolute) or 2%.
- overlay_slide.tiff: same shape and pyramid depth; <= 1% of pixels differ.

The baseline lives in tests/output_baseline_hybrid/ (gitignored: another GPU
segments slightly differently). The first run records it and skips; every run
after that compares. Delete the directory to re-baseline on purpose.

The same run is also the check that every stage reports progress through
tqdm (backend/api/tqdm_progress.py), in the order the frontend can rely on.

Skips when the test pair or the models are not on this machine.
"""
from __future__ import annotations

import csv
import itertools
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy.spatial import cKDTree

from backend.algorithms.hybrid.config import config   # not bare `config`: alignment has one too
from backend.api import tqdm_progress
from hybrid_pipeline import _run_single_tile_cli

BASELINE = Path(__file__).resolve().parent / "output_baseline_hybrid"
ARTIFACTS = ("report.csv", "summary.txt", "overlay_slide.tiff")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (
        config.ihc_test_path, config.dish_test_path, config.unet_model_path,
        config.cellpose_model_path, config.cellpose_dish_model_path,
    )),
    reason="bundled test pair / models are not on this machine",
)


@pytest.fixture(scope="module", params=[1, 2], ids=["workers=1", "workers=2"])
def run(request, tmp_path_factory):
    """One pipeline run per worker count -> (workers, output_dir, progress publishes).

    Runs under the tqdm hook the API installs, with the job lookups stubbed, so
    every bar the pipeline shows is captured as (phase, done, total, unit)."""
    workers = request.param
    out = tmp_path_factory.mktemp(f"hybrid_w{workers}")
    seen: list[tuple[str, int, int, str]] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tqdm_progress, "current_job_id", lambda: "golden")
        mp.setattr(tqdm_progress, "set_progress",
                   lambda _job, phase, done, total, unit: seen.append((phase, done, total, unit)))
        tqdm_progress.install()
        tqdm_progress.reporting(lambda: _run_single_tile_cli(
            str(config.ihc_test_path), str(config.dish_test_path), out, workers=workers,
        ))()
    return workers, out, seen


def _report(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    xy = np.array([[float(r["centroid_x"]), float(r["centroid_y"])] for r in rows]).reshape(-1, 2)
    return rows, xy


def _overlay(path: Path):
    with tifffile.TiffFile(path) as tif:
        return len(tif.series[0].levels), tif.series[0].levels[0].asarray()


def test_outputs_match_baseline(run):
    _workers, out, _seen = run
    for name in ARTIFACTS:
        assert (out / name).is_file(), f"{name} was not produced"
    if not BASELINE.exists():
        BASELINE.mkdir()
        for name in ARTIFACTS:
            shutil.copy(out / name, BASELINE / name)
        pytest.skip(f"baseline recorded in {BASELINE}; run again to compare")

    base_rows, base_xy = _report(BASELINE / "report.csv")
    rows, xy = _report(out / "report.csv")
    assert [int(r["cell_id"]) for r in rows] == list(range(1, len(rows) + 1))
    assert abs(len(rows) - len(base_rows)) <= 0.01 * len(base_rows), (len(rows), len(base_rows))
    dist, idx = cKDTree(xy).query(base_xy)
    attrs = [k for k in base_rows[0] if k not in ("cell_id", "centroid_x", "centroid_y")]
    same = [
        d <= 2 and all(base_rows[j][k] == rows[i][k] for k in attrs)
        for j, (d, i) in enumerate(zip(dist, idx))
    ]
    assert np.mean(same) >= 0.995, f"{same.count(False)} of {len(same)} cells differ from baseline"

    base_lines = (BASELINE / "summary.txt").read_text(encoding="utf-8").splitlines()
    lines = (out / "summary.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(base_lines)
    for got, want in zip(lines, base_lines):
        assert _NUM.sub("#", got) == _NUM.sub("#", want), (got, want)
        for g, w in zip(_NUM.findall(got), _NUM.findall(want)):
            assert abs(float(g) - float(w)) <= max(2, 0.02 * abs(float(w))), (got, want)

    base_levels, base_px = _overlay(BASELINE / "overlay_slide.tiff")
    levels, px = _overlay(out / "overlay_slide.tiff")
    assert (levels, px.shape) == (base_levels, base_px.shape)
    assert np.mean(np.any(px != base_px, axis=-1)) <= 0.01


def test_every_stage_reports_progress(run):
    """Phases arrive in a fixed order, each one ends at done == total, and the
    tile counter is published with its denominator before the first tile is
    done (so a panel has a total to draw against from the start)."""
    workers, _out, seen = run
    phases = [phase for phase, _ in itertools.groupby(p for p, *_ in seen)]
    # workers > 1 loads the models inside each worker, so no parent-side bar for it.
    expected = ["載入模型", "分析", "匯出報表", "縫合影像"] if workers == 1 else ["分析", "匯出報表", "縫合影像"]
    assert phases == expected, phases
    for phase in expected:
        _p, done, total, _u = [s for s in seen if s[0] == phase][-1]
        assert done == total > 0, f"{phase} ended at {done}/{total}"
    first_tile = next(s for s in seen if s[0] == "分析")
    assert (first_tile[1], first_tile[3]) == (0, "塊"), first_tile
