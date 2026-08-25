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

Round 13 (doc 40 §3 item 2) added a second encoder behind `config.stitch_backend`
(candidate B: band-streamed `tifffile` + Predictor 2). Every check here runs against
both backends, because the failure this file exists to catch is *exactly* the one the
new path is most exposed to: `tifffile` only declares the Predictor tag for
pre-compressed tiles, so the differencing is applied by our own code and a mistake
there reproduces doc 32 §5.1's silent corruption byte for byte.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from skimage import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.algorithms.hybrid.config import config  # noqa: E402
from backend.algorithms.hybrid.m0_slide import (  # noqa: E402
    _STITCH_SCRATCH,
    _stitch_overlay_slide,
    compute_tile_geometry,
    core_crop_bounds,
)

TILE = 512
OVERLAP = 128

BACKENDS = ("pyvips", "tifffile")


@pytest.fixture(params=BACKENDS)
def backend(request, monkeypatch) -> str:
    monkeypatch.setattr(config, "stitch_backend", request.param, raising=False)
    return request.param


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


def test_every_pyramid_level_decodes_bright(tmp_path: Path, backend: str) -> None:
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


def test_predictor_tag_is_consistent_across_levels(tmp_path: Path, backend: str) -> None:
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


def _build_with(tmp_path: Path, name: str, monkeypatch, backend: str) -> Path:
    monkeypatch.setattr(config, "stitch_backend", backend, raising=False)
    d = tmp_path / name
    d.mkdir()
    return _build_slide(d)


def test_backends_agree_on_layout_and_level0_pixels(tmp_path, monkeypatch) -> None:
    """The two encoders must produce the same *picture*, only different bytes.

    This is doc 32 §3's non-comparability guard turned into a test: a candidate that
    wrote larger container tiles or a shallower pyramid would simply have done less
    work, and its measured speedup would mean nothing.

    *Every* level is compared exactly, not just level 0. That is stronger than it may
    look: it pins `_shrink2_cpu` to pyvips's own `region_shrink` kernel. Measured
    directly on the real 141658x114366 shipped overlay, every level L3..L11 is a
    bit-exact 2x2 box shrink of the level above (`maxdelta=0`), so the two encoders
    agree on the downsample and any future drift in either is a real defect, not a
    rounding preference.
    """
    a = _build_with(tmp_path, "pyvips", monkeypatch, "pyvips")
    b = _build_with(tmp_path, "tifffile", monkeypatch, "tifffile")

    with tifffile.TiffFile(str(a)) as fa, tifffile.TiffFile(str(b)) as fb:
        assert len(fa.pages) == len(fb.pages), (
            f"pyramid depth differs: {len(fa.pages)} vs {len(fb.pages)} pages — a "
            f"shallower pyramid is less work, so the two are not comparable."
        )
        assert len(fa.pages) > 1, "no pyramid was written; this test would be vacuous"
        assert ((fa.pages[0].tilewidth, fa.pages[0].tilelength)
                == (fb.pages[0].tilewidth, fb.pages[0].tilelength))
        for i, (pa, pb) in enumerate(zip(fa.pages, fb.pages)):
            assert pa.shape == pb.shape, f"level {i} shape {pa.shape} vs {pb.shape}"
            assert np.array_equal(pa.asarray(), pb.asarray()), (
                f"level {i} differs between backends; both encoders are lossless and "
                f"share the same 2x2 box shrink, so the stitched picture must be "
                f"identical no matter which one wrote it."
            )


def test_unknown_backend_fails_loudly(tmp_path: Path, monkeypatch) -> None:
    """A typo'd backend must not silently fall through to the shipped path — a run that
    quietly ignored the switch would make every measurement of it a lie."""
    monkeypatch.setattr(config, "stitch_backend", "tiffile", raising=False)
    with pytest.raises(ValueError, match="stitch_backend"):
        _build_slide(tmp_path)
