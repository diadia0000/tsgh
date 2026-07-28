"""Tests for `PrecutStream(region=...)` — analysing only a ROI of a slide.

A whole slide is ~27,500 tiles and hours of GPU; the UI sends a ROI instead.
The invariant that makes this cheap is that the ROI grid is the ordinary grid
computed on the ROI's dimensions and then translated to the ROI's origin, so
tile filenames — and therefore every downstream coordinate: the cut lines,
core-ownership dedup, and the absolutised cell centroids — stay in **slide**
pixels. These tests pin that translation, and that no tile ever reads outside
the ROI (which would analyse tissue the pathologist did not select).
"""
from __future__ import annotations

import numpy as np
import pyvips
import pytest

from m0_module.m0_reader import PrecutStream, chunk_offsets
from m0_module.m0_stitch import compute_tile_geometry, core_crop_bounds

TILE = 256
OVERLAP = 64


@pytest.fixture
def slide_pair(tmp_path):
    """Two same-sized RGB slides whose pixel value encodes its own position, so a
    tile written from the wrong place is detectable rather than merely suspicious."""
    w, h = 1400, 1100
    xs = np.arange(w, dtype=np.uint8)
    ys = np.arange(h, dtype=np.uint8)
    arr = np.dstack([
        np.broadcast_to(xs, (h, w)),
        np.broadcast_to(ys[:, None], (h, w)),
        np.full((h, w), 7, dtype=np.uint8),
    ]).copy()

    paths = []
    for name in ("ihc", "dish"):
        p = tmp_path / f"{name}.tiff"
        pyvips.Image.new_from_array(arr).write_to_file(str(p))
        paths.append(p)
    return paths[0], paths[1], (w, h), arr


def _stream(slide_pair, tmp_path, region):
    ihc, dish, _, _ = slide_pair
    return PrecutStream(
        ihc, dish, tmp_path / "out_ihc", tmp_path / "out_dish",
        tile_size=TILE, overlap=OVERLAP, workers=2, region=region,
    )


def test_no_region_keeps_the_whole_slide_grid(slide_pair, tmp_path):
    _, _, (w, h), _ = slide_pair
    stream = _stream(slide_pair, tmp_path, None)
    assert stream.positions == chunk_offsets(h, w, TILE, OVERLAP)


def test_region_translates_the_grid_to_slide_coordinates(slide_pair, tmp_path):
    region = (384, 256, 768, 512)   # x, y, w, h
    stream = _stream(slide_pair, tmp_path, region)

    expected = [(x + 384, y + 256) for (x, y) in chunk_offsets(512, 768, TILE, OVERLAP)]
    assert stream.positions == expected


def test_every_tile_stays_inside_the_region(slide_pair, tmp_path):
    """A tile crossing the ROI border would analyse tissue outside the selection.
    The last window snaps back to the edge, so this must hold exactly."""
    x0, y0, rw, rh = 384, 256, 768, 512
    stream = _stream(slide_pair, tmp_path, (x0, y0, rw, rh))

    for x, y in stream.positions:
        assert x0 <= x and x + TILE <= x0 + rw
        assert y0 <= y and y + TILE <= y0 + rh


def test_region_covers_every_pixel_of_the_roi(slide_pair, tmp_path):
    """Snapping the last window back must not leave an unanalysed strip."""
    x0, y0, rw, rh = 384, 256, 768, 500
    stream = _stream(slide_pair, tmp_path, (x0, y0, rw, rh))

    covered = np.zeros((rh, rw), dtype=bool)
    for x, y in stream.positions:
        covered[y - y0:y - y0 + TILE, x - x0:x - x0 + TILE] = True
    assert covered.all()


def test_written_tiles_hold_the_pixels_at_their_named_coordinates(slide_pair, tmp_path):
    """The filename is the only record of where a tile came from; if it disagreed
    with the pixels, every cell centroid downstream would land in the wrong place."""
    _, _, _, arr = slide_pair
    region = (384, 256, 768, 512)
    stream = _stream(slide_pair, tmp_path, region)

    for ihc_tile, _dish_tile, (x, y) in stream:
        assert ihc_tile.name == f"tile_x{x}_y{y}.tiff"
        written = pyvips.Image.new_from_file(str(ihc_tile)).numpy()
        np.testing.assert_array_equal(written, arr[y:y + TILE, x:x + TILE])


def test_geometry_accepts_a_grid_starting_at_the_roi_origin(slide_pair, tmp_path):
    """The stitch geometry validates that the grid starts where the analysed area
    starts. For a ROI that is the ROI's corner, not the slide's."""
    x0, y0 = 384, 256
    stream = _stream(slide_pair, tmp_path, (x0, y0, 768, 512))

    geometry = compute_tile_geometry(stream.positions, TILE, OVERLAP, origin=(x0, y0))
    assert geometry.x_min == x0 and geometry.y_min == y0


def test_geometry_still_rejects_a_grid_missing_its_first_column(slide_pair, tmp_path):
    """Passing the origin must not weaken the check into "whatever starts[0] is".
    Dropping the first column has to stay an error."""
    x0, y0 = 384, 256
    stream = _stream(slide_pair, tmp_path, (x0, y0, 768, 512))
    without_first_col = [p for p in stream.positions if p[0] != x0]

    with pytest.raises(ValueError, match="起點必須從"):
        compute_tile_geometry(without_first_col, TILE, OVERLAP, origin=(x0, y0))


def test_core_regions_tile_the_roi_exactly(slide_pair, tmp_path):
    """Core-ownership regions must cover the ROI once and only once -- overlaps
    would double-count cells, gaps would drop them."""
    x0, y0, rw, rh = 384, 256, 768, 512
    stream = _stream(slide_pair, tmp_path, (x0, y0, rw, rh))
    geometry = compute_tile_geometry(stream.positions, TILE, OVERLAP, origin=(x0, y0))

    hits = np.zeros((rh, rw), dtype=int)
    for x, y in stream.positions:
        lx0, lx1, ly0, ly1 = core_crop_bounds(geometry, x, y, TILE)
        hits[y - y0 + ly0:y - y0 + ly1, x - x0 + lx0:x - x0 + lx1] += 1
    assert hits.min() == 1 and hits.max() == 1


def test_region_outside_the_slide_is_rejected(slide_pair, tmp_path):
    with pytest.raises(ValueError, match="超出切片範圍"):
        _stream(slide_pair, tmp_path, (1000, 0, 600, 512))


def test_region_smaller_than_one_tile_is_rejected(slide_pair, tmp_path):
    with pytest.raises(ValueError, match="小於最小允許尺寸"):
        _stream(slide_pair, tmp_path, (0, 0, TILE - 1, 512))
