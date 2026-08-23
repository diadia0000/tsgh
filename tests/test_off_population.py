"""`build_off_population_results` — the protein-negative (off) population.

Covers the two things that had no coverage before: a DISH nucleus nobody won
becomes its own row, and its `cell_id` is offset so it can never collide with
the tile's IHC cell ids (which are 1..N).

The image fed in must be the **raw, unmasked** dish tile — a synthetic one here
(pale-blue nucleus + real red/black dot signal), not `dish_mask_overlay`, whose
white fill outside the IHC core would blank this population's ROI to 0/0.
"""
from __future__ import annotations

import numpy as np

from m3_module.m3_dot_detection import build_off_population_results
from m4_module.csv import _classify_cell_type


class _Cfg:
    dot_background_l_threshold = 95.0
    score_cep17_min_count = 2
    dot_amplification_ratio = 2.0
    # remaining dot_* thresholds fall back to m3_dot_kernels defaults via getattr


def _mask() -> np.ndarray:
    mask = np.zeros((100, 100), dtype=np.int32)
    mask[10:50, 10:50] = 1   # never matched
    mask[60:90, 60:90] = 2   # already matched -- must not appear
    return mask


def _raw_dish() -> np.ndarray:
    """Raw-dish-style tile: white slide background, one pale-blue nucleus
    carrying 4 black (HER2) + 2 red (CEP17) dots -> ratio 2.0 -> amplified."""
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[10:50, 10:50] = (205, 205, 235)
    for y, x in ((16, 16), (16, 30)):
        image[y:y + 4, x:x + 4] = (200, 30, 30)
    for y, x in ((30, 16), (30, 30), (40, 16), (40, 30)):
        image[y:y + 4, x:x + 4] = (25, 25, 25)
    return image


def test_unmatched_nucleus_becomes_off_population_entry_with_offset_id():
    out, dots = build_off_population_results(
        _raw_dish(), _mask(), matched_dish_ids={2}, config=_Cfg(), id_offset=100,
    )
    assert len(out) == 1
    # The dots must come back too, id-offset to match the result, or the overlay
    # draws an orange (unmatched) nucleus with no red/black markers on it.
    assert len(dots) == 6
    assert {d.cell_id for d in dots} == {101}
    assert sorted(d.dot_type for d in dots).count("her2") == 4
    assert out[0].cell_id == 101
    assert out[0].is_her2_positive is False
    assert out[0].excluded is False
    assert (out[0].her2_dot_count, out[0].cep17_dot_count) == (4, 2)
    assert out[0].is_amplified is True
    assert _classify_cell_type(out[0]) == "non_classic_positive"


def test_core_masked_image_would_blank_the_off_population():
    """Why the caller must pass the raw dish: on a core-masked (white-filled)
    image the same nucleus scores 0/0 and can only ever come out negative."""
    masked = np.full((100, 100, 3), 255, dtype=np.uint8)
    out, dots = build_off_population_results(
        masked, _mask(), matched_dish_ids={2}, config=_Cfg(), id_offset=100,
    )
    assert dots == []
    assert (out[0].her2_dot_count, out[0].cep17_dot_count) == (0, 0)
    assert _classify_cell_type(out[0]) == "negative"


def test_all_nuclei_matched_yields_no_off_population():
    assert build_off_population_results(
        _raw_dish(), _mask(), matched_dish_ids={1, 2}, config=_Cfg(), id_offset=5,
    ) == ([], [])


def test_off_population_only_ever_gets_the_low_cep17_exclusion_reason():
    """drop_out / out_of_bounds are IHC-cell-specific (competition, mask edge);
    an unmatched DISH nucleus has neither concept, so a red-poor one must be
    tagged low_cep17 — otherwise the summary's alignment-quality breakdown is
    inflated by cells that were never mis-aligned."""
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[10:26, 10:26] = (205, 205, 235)
    image[16:20, 16:20] = (200, 30, 30)          # a single red dot → cep17 == 1 < 2
    mask = np.zeros((100, 100), dtype=np.int32)
    mask[10:26, 10:26] = 1

    out, _ = build_off_population_results(
        image, mask, matched_dish_ids=set(), config=_Cfg(), id_offset=0,
    )
    assert out[0].cep17_dot_count == 1
    assert out[0].excluded is True
    assert out[0].exclusion_reason == "low_cep17"
