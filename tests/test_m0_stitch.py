"""Correctness tests for `m0_stitch` — the tile-geometry / core-ownership layer.

Why this module first (doc 19 §3, DISCOVERED #37): it is pure data reorganization
with no model, no GPU and no I/O, so every claim in its docstrings is checkable
against synthetic numpy input in milliseconds. It is also the layer that decides
**which tile owns which cell** and **which pixels each tile contributes to the
stitched slide** — a silent error here produces a slide with duplicated or missing
cells that no downstream stage can detect.

The grid convention under test (shared with `m2_segmentation._overlap_window_coords`
and `m0_reader.chunk_offsets`): starts are `0, stride, 2*stride, ...` with the last
one snapped flush to `length - tile_size`, so the final gap may be short.
"""
from __future__ import annotations

import numpy as np
import pytest

from hybrid_data_types import CellAnalysisResult
from m0_module.m0_stitch import (
    ChunkResult,
    clear_slide_edge_cells,
    compute_tile_geometry,
    core_crop_bounds,
    filter_and_absolutize,
)

TILE = 1024
OVERLAP = 256
STRIDE = TILE - OVERLAP  # 768


def grid(cols: int, rows: int, last_x: int | None = None, last_y: int | None = None):
    """Build a well-formed `positions` list; `last_*` override the flush-snapped edge."""
    xs = [i * STRIDE for i in range(cols)]
    ys = [i * STRIDE for i in range(rows)]
    if last_x is not None:
        xs[-1] = last_x
    if last_y is not None:
        ys[-1] = last_y
    return [(x, y) for y in ys for x in xs]


# ------------------------------------------------------------------
# compute_tile_geometry — grid validation is the fail-fast the analysis
# stage relies on to catch a partially-completed precut job.
# ------------------------------------------------------------------

def test_geometry_cuts_sit_half_an_overlap_into_the_later_tile():
    g = compute_tile_geometry(grid(3, 2), TILE, OVERLAP)
    # cut between tile i-1 and i is at starts[i] + overlap//2
    assert g.cuts_x == [STRIDE + OVERLAP // 2, 2 * STRIDE + OVERLAP // 2]
    assert g.cuts_y == [STRIDE + OVERLAP // 2]
    assert g.col_of == {0: 0, STRIDE: 1, 2 * STRIDE: 2}
    assert (g.x_min, g.x_max) == (0, 2 * STRIDE)


def test_geometry_accepts_a_short_final_gap():
    """The last column is snapped flush to the slide edge, so its gap is < stride."""
    g = compute_tile_geometry(grid(3, 1, last_x=2 * STRIDE - 100), TILE, OVERLAP)
    assert g.x_max == 2 * STRIDE - 100


def test_single_tile_grid_has_no_cuts():
    g = compute_tile_geometry([(0, 0)], TILE, OVERLAP)
    assert g.cuts_x == [] and g.cuts_y == []
    assert g.edge_flags(0, 0) == (True, True, True, True)


@pytest.mark.parametrize("positions, why", [
    ([], "empty"),
    ([(0, 0), (0, 0)], "duplicate position"),
    ([(STRIDE, 0)], "does not start at 0"),
    ([(0, 0), (3 * STRIDE, 0), (6 * STRIDE, 0)], "interior gap != stride"),
    ([(0, 0), (STRIDE, 0), (0, STRIDE)], "not a full cartesian product (hole)"),
])
def test_geometry_rejects_malformed_grids(positions, why):
    with pytest.raises(ValueError):
        compute_tile_geometry(positions, TILE, OVERLAP)


def test_geometry_rejects_overlap_swallowing_the_tile():
    with pytest.raises(ValueError):
        compute_tile_geometry([(0, 0)], TILE, TILE)


def test_edge_flags_only_fire_on_real_slide_edges():
    g = compute_tile_geometry(grid(3, 3), TILE, OVERLAP)
    mid = STRIDE
    assert g.edge_flags(mid, mid) == (False, False, False, False)   # interior
    assert g.edge_flags(0, mid) == (False, False, True, False)      # left column
    assert g.edge_flags(2 * STRIDE, 2 * STRIDE) == (False, True, False, True)


# ------------------------------------------------------------------
# core_crop_bounds — the crops must tile the slide exactly: no overlap,
# no gap. That is the whole contract `_stitch_overlay_slide` rests on.
# ------------------------------------------------------------------

def test_core_crops_partition_the_slide_exactly():
    cols, rows = 4, 3
    positions = grid(cols, rows)
    g = compute_tile_geometry(positions, TILE, OVERLAP)

    full_w = (cols - 1) * STRIDE + TILE
    full_h = (rows - 1) * STRIDE + TILE
    covered = np.zeros((full_h, full_w), dtype=np.int32)

    for ax, ay in positions:
        lx0, lx1, ly0, ly1 = core_crop_bounds(g, ax, ay, TILE)
        assert 0 <= lx0 < lx1 <= TILE
        assert 0 <= ly0 < ly1 <= TILE
        covered[ay + ly0:ay + ly1, ax + lx0:ax + lx1] += 1

    # exactly once everywhere: 0 would be a gap (missing pixels in the slide),
    # 2 would be an overlap (a seam drawn twice / wrong dimensions after join)
    assert covered.min() == 1 and covered.max() == 1


def test_core_crop_widths_are_uniform_per_column_and_heights_per_row():
    """`_join_overlay_tiles` joins rows horizontally then vertically, which is only
    pixel-exact if same-row tiles share a height and same-column tiles share a width."""
    positions = grid(4, 3)
    g = compute_tile_geometry(positions, TILE, OVERLAP)
    widths, heights = {}, {}
    for ax, ay in positions:
        lx0, lx1, ly0, ly1 = core_crop_bounds(g, ax, ay, TILE)
        widths.setdefault(ax, set()).add(lx1 - lx0)
        heights.setdefault(ay, set()).add(ly1 - ly0)
    assert all(len(v) == 1 for v in widths.values())
    assert all(len(v) == 1 for v in heights.values())


def test_single_tile_core_crop_is_the_whole_tile():
    g = compute_tile_geometry([(0, 0)], TILE, OVERLAP)
    assert core_crop_bounds(g, 0, 0, TILE) == (0, TILE, 0, TILE)


# ------------------------------------------------------------------
# filter_and_absolutize — centroid core-ownership dedup
# ------------------------------------------------------------------

def _cell(cell_id: int, cx: float, cy: float) -> CellAnalysisResult:
    return CellAnalysisResult(
        cell_id=cell_id, centroid_x=cx, centroid_y=cy,
        is_her2_positive=False,
    )


def _chunk(ax: int, ay: int, cells) -> ChunkResult:
    z = np.zeros((TILE, TILE), dtype=np.int32)
    return ChunkResult(
        abs_x=ax, abs_y=ay, instance_mask=z, dish_nucleus_mask=z,
        dish_mask_overlay=np.zeros((TILE, TILE, 3), np.uint8),
        results=list(cells), all_dots=[], per_cell_dots={},
    )


def test_every_cell_in_the_overlap_is_claimed_by_exactly_one_tile():
    """The dedup invariant: a cell seen by two neighbouring tiles must survive once."""
    positions = grid(2, 1)
    g = compute_tile_geometry(positions, TILE, OVERLAP)
    cut = g.cuts_x[0]                       # 768 + 128 = 896

    # one global cell at every x across the whole overlap band of the two tiles
    for gx in range(STRIDE, TILE):
        left = filter_and_absolutize(
            _chunk(0, 0, [_cell(1, gx - 0, 10.0)]), g, 0, 0)
        right = filter_and_absolutize(
            _chunk(STRIDE, 0, [_cell(1, gx - STRIDE, 10.0)]), g, STRIDE, 0)
        assert len(left) + len(right) == 1, f"gx={gx} claimed {len(left)+len(right)}x"
        assert (len(left) == 1) == (gx < cut)


def test_absolutized_centroids_are_shifted_by_the_tile_origin():
    g = compute_tile_geometry(grid(2, 2), TILE, OVERLAP)
    owned = filter_and_absolutize(
        _chunk(STRIDE, STRIDE, [_cell(7, 500.0, 400.0)]), g, STRIDE, STRIDE)
    assert len(owned) == 1
    assert (owned[0].centroid_x, owned[0].centroid_y) == (STRIDE + 500.0, STRIDE + 400.0)


def test_filter_does_not_renumber_cell_ids():
    """Documented contract: global renumbering happens once, in `_finish_batch`."""
    g = compute_tile_geometry(grid(2, 2), TILE, OVERLAP)
    owned = filter_and_absolutize(
        _chunk(STRIDE, STRIDE, [_cell(42, 500.0, 400.0)]), g, STRIDE, STRIDE)
    assert owned[0].cell_id == 42


def test_single_tile_keeps_every_cell():
    g = compute_tile_geometry([(0, 0)], TILE, OVERLAP)
    cells = [_cell(i, float(i), float(i)) for i in range(1, 6)]
    assert len(filter_and_absolutize(_chunk(0, 0, cells), g, 0, 0)) == 5


# ------------------------------------------------------------------
# clear_slide_edge_cells — only the requested sides get cleared
# ------------------------------------------------------------------

def _mask_with_touching_cells() -> np.ndarray:
    m = np.zeros((20, 20), dtype=np.int32)
    m[0:3, 5:8] = 1        # touches top
    m[17:20, 5:8] = 2      # touches bottom
    m[5:8, 0:3] = 3        # touches left
    m[5:8, 17:20] = 4      # touches right
    m[9:12, 9:12] = 5      # interior, never cleared
    return m


def test_clear_only_removes_cells_on_the_requested_sides():
    out = clear_slide_edge_cells(_mask_with_touching_cells(),
                                 clear_top=True, clear_bottom=False,
                                 clear_left=False, clear_right=False)
    # the top cell is gone; the other four survive and are relabelled 1..4
    assert set(np.unique(out)) == {0, 1, 2, 3, 4}
    assert out[0:3, 5:8].max() == 0


def test_clearing_no_side_keeps_every_cell_but_relabels_sequentially():
    m = _mask_with_touching_cells()
    m[m == 3] = 9          # make the labels sparse
    out = clear_slide_edge_cells(m, False, False, False, False)
    assert set(np.unique(out)) == {0, 1, 2, 3, 4, 5}


def test_clearing_all_four_sides_leaves_only_interior_cells():
    out = clear_slide_edge_cells(_mask_with_touching_cells(), True, True, True, True)
    assert set(np.unique(out)) == {0, 1}
    assert out[9:12, 9:12].min() == 1


def test_clear_does_not_mutate_its_input():
    m = _mask_with_touching_cells()
    before = m.copy()
    clear_slide_edge_cells(m, True, True, False, False)
    np.testing.assert_array_equal(m, before)
