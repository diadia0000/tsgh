"""Tests for `run_batch`'s partial-resume checkpoint (doc 19 #1c, DISCOVERED #42).

These exercise the checkpoint store and the skip filter directly rather than
running `run_batch`, because the surrounding batch needs three GPU models. The
store *is* the whole feature: what gets persisted, what gets skipped on retry,
and — most importantly — the guards that stop a stale or foreign checkpoint from
being mixed into a fresh run.
"""
from __future__ import annotations

import pickle

import pytest

import hybrid_pipeline as HP
from hybrid_data_types import CellAnalysisResult


def _cell(cell_id: int) -> CellAnalysisResult:
    return CellAnalysisResult(
        cell_id=cell_id, centroid_x=1.0, centroid_y=2.0,
        is_her2_positive=True,
    )


POSITIONS = [(0, 0), (768, 0), (0, 768), (768, 768)]


def test_round_trip_preserves_results(tmp_path):
    HP._checkpoint_init(tmp_path, "abc12345")
    HP._checkpoint_save(tmp_path, 768, 0, [_cell(1), _cell(2)])

    done = HP._checkpoint_load(tmp_path, "abc12345", POSITIONS)
    assert list(done) == [(768, 0)]
    assert [c.cell_id for c in done[(768, 0)]] == [1, 2]
    assert done[(768, 0)][0].centroid_x == 1.0


def test_empty_result_list_round_trips_as_a_completed_tile(tmp_path):
    """A background tile legitimately owns zero cells — that is `skipped`, not `unrun`.
    Storing it as an empty list must not read back as 'not done'."""
    HP._checkpoint_init(tmp_path, "abc12345")
    HP._checkpoint_save(tmp_path, 0, 0, [])
    done = HP._checkpoint_load(tmp_path, "abc12345", POSITIONS)
    assert (0, 0) in done and done[(0, 0)] == []


def test_missing_directory_is_simply_not_a_resume(tmp_path):
    assert HP._checkpoint_load(tmp_path / "nope", "abc12345", POSITIONS) == {}


def test_config_hash_mismatch_discards_everything(tmp_path):
    """Tiles computed under a different config are not comparable; mixing them
    would produce a slide no config_hash in the CSV can describe."""
    HP._checkpoint_init(tmp_path, "OLDHASH1")
    HP._checkpoint_save(tmp_path, 0, 0, [_cell(1)])
    assert HP._checkpoint_load(tmp_path, "NEWHASH2", POSITIONS) == {}


def test_absent_hash_file_discards_everything(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    HP._checkpoint_save(tmp_path, 0, 0, [_cell(1)])
    assert HP._checkpoint_load(tmp_path, "abc12345", POSITIONS) == {}


def test_tiles_outside_this_grid_are_ignored(tmp_path):
    """Resuming into a *different* slide's grid must not import its tiles."""
    HP._checkpoint_init(tmp_path, "abc12345")
    HP._checkpoint_save(tmp_path, 999999, 999999, [_cell(1)])
    assert HP._checkpoint_load(tmp_path, "abc12345", POSITIONS) == {}


def test_corrupt_checkpoint_file_is_skipped_not_fatal(tmp_path):
    """A run killed mid-write should cost one tile of recompute, not the batch."""
    HP._checkpoint_init(tmp_path, "abc12345")
    HP._checkpoint_save(tmp_path, 0, 0, [_cell(1)])
    (tmp_path / "tile_x768_y0.pkl").write_bytes(b"not a pickle")

    done = HP._checkpoint_load(tmp_path, "abc12345", POSITIONS)
    assert list(done) == [(0, 0)]


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    HP._checkpoint_init(tmp_path, "abc12345")
    HP._checkpoint_save(tmp_path, 0, 0, [_cell(1)])
    assert not list(tmp_path.glob("*.tmp"))
    assert (tmp_path / "tile_x0_y0.pkl").exists()


def test_resaving_a_tile_overwrites_it(tmp_path):
    HP._checkpoint_init(tmp_path, "abc12345")
    HP._checkpoint_save(tmp_path, 0, 0, [_cell(1)])
    HP._checkpoint_save(tmp_path, 0, 0, [_cell(9), _cell(10)])
    with open(tmp_path / "tile_x0_y0.pkl", "rb") as fh:
        assert [c.cell_id for c in pickle.load(fh)] == [9, 10]


def test_skip_completed_filters_only_the_recorded_tiles():
    tiles = [(f"ihc{i}", f"dish{i}", pos) for i, pos in enumerate(POSITIONS)]
    done = {(0, 0): [], (768, 768): [_cell(1)]}
    kept = [pos for _i, _d, pos in HP._skip_completed(iter(tiles), done)]
    assert kept == [(768, 0), (0, 768)]


def test_skip_completed_is_a_no_op_without_a_checkpoint():
    tiles = [(f"ihc{i}", f"dish{i}", pos) for i, pos in enumerate(POSITIONS)]
    assert [pos for _i, _d, pos in HP._skip_completed(iter(tiles), {})] == POSITIONS


def test_checkpoint_is_off_by_default():
    """The API path issues single-tile requests; it must not start writing scratch
    into caller output directories because of this feature."""
    import inspect
    assert inspect.signature(HP.run_batch).parameters["checkpoint"].default is False
