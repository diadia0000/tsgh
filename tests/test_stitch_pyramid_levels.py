"""Every pyramid level of `overlay_slide.tiff` must decode correctly, not just level 0.

Round 9 (doc 32 §5.1) found that the shipped stitch produced a file whose *reduced*
pyramid levels decode to noise. libvips 8.15.1's default `predictor="horizontal"` writes
tag 317 (Predictor=2) on the full-resolution IFD only, while still horizontally
differencing the reduced levels' pixel data. A reader that honours the tag — which is
every reader tried, tifffile and libtiff-via-Pillow alike — then fails to un-difference
levels 1..N and renders 88-99% black.

This was invisible to every existing check because they all read level 0, which is
tagged correctly. It is exactly the failure a pathologist hits first, though: QuPath
opens the file fine and only goes wrong when you zoom out.

The guard is the *shape* of the content, not exact pixels: a correctly decoded reduced
level of a mostly-white overlay stays mostly white. A mis-decoded one collapses to near
zero, because un-differenced data is dominated by the zeros of flat regions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.algorithms.hybrid.m0_slide import (  # noqa: E402
    _STITCH_SCRATCH,
    _stitch_overlay_slide,
    compute_tile_geometry,
    core_crop_bounds,
)

TILE = 512
OVERLAP = 128


def _build_slide(tmp_path: Path) -> Path:
    """Stitch a 3x3 grid of overlay-like tiles through the real pipeline function."""
    step = TILE - OVERLAP
    positions = [(x, y) for y in (0, step, 2 * step) for x in (0, step, 2 * step)]
    geometry = compute_tile_geometry(positions, TILE, OVERLAP)

    scratch = tmp_path / _STITCH_SCRATCH
    scratch.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for ax, ay in positions:
        lx0, lx1, ly0, ly1 = core_crop_bounds(geometry, ax, ay, TILE)
        h, w = ly1 - ly0, lx1 - lx0
        # Overlay-like: mostly white with sparse dark annotation, which is what makes a
        # mis-decoded level collapse to near-black and a correct one stay bright.
        crop = np.full((h, w, 3), 255, dtype=np.uint8)
        ys = rng.integers(0, h, size=max(1, h * w // 200))
        xs = rng.integers(0, w, size=max(1, h * w // 200))
        crop[ys, xs] = 0
        io.imsave(str(scratch / f"tile_x{ax}_y{ay}.tiff"), crop, check_contrast=False)

    _stitch_overlay_slide(tmp_path, geometry)
    return tmp_path / "overlay_slide.tiff"


def test_every_pyramid_level_decodes_bright(tmp_path: Path) -> None:
    slide = _build_slide(tmp_path)
    assert slide.exists()

    with tifffile.TiffFile(str(slide)) as tf:
        assert len(tf.pages) > 1, "no pyramid was written; this test would be vacuous"
        for i, page in enumerate(tf.pages):
            level = page.asarray()
            mean = float(level.mean())
            zero_frac = float((level == 0).mean())
            assert mean > 200.0, (
                f"pyramid level {i} decoded to mean {mean:.1f} — a mostly-white overlay "
                f"cannot average that dark. This is the predictor-tag bug: the level was "
                f"horizontally differenced but not tagged Predictor=2."
            )
            assert zero_frac < 0.2, (
                f"pyramid level {i} is {zero_frac:.1%} exactly-zero pixels; "
                f"mis-decoded differenced data looks exactly like this."
            )


def test_predictor_tag_is_consistent_across_levels(tmp_path: Path) -> None:
    """Whatever the predictor is, every IFD must agree — a tag present on level 0 and
    absent on the reduced levels is the specific inconsistency that caused the bug."""
    slide = _build_slide(tmp_path)
    with tifffile.TiffFile(str(slide)) as tf:
        preds = [
            page.tags["Predictor"].value if "Predictor" in page.tags else 1
            for page in tf.pages
        ]
    assert len(set(preds)) == 1, (
        f"Predictor differs across pyramid levels: {preds}. Readers apply each IFD's own "
        f"tag, so a level whose data is differenced but whose tag says otherwise decodes "
        f"as noise."
    )
