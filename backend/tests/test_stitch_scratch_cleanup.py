"""縫合暫存夾的往返：拼出 overlay_slide.tiff 後暫存 tile 必須全數消失。

逐塊 overlay 已改為只寫進 ``_stitch_scratch/``（非產物，是 pyvips 惰性讀取的串流
緩衝）。這裡用合成 tile 驗證兩件事：拼回的整片版面逐像素正確、且拼完暫存夾被刪掉
——否則「跑圖不落地中間檔」這個保證就破了。

兩個縫合後端都要過（``config.stitch_backend``；round 13 / doc 40 §3 item 2 加的
candidate B 走的是完全不同的讀寫路徑，版面與清理這兩個保證卻必須一模一樣）。
"""
import sys
from pathlib import Path

import numpy as np
import pyvips
import pytest
from skimage import io

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.algorithms.hybrid.output.tile_x98304_y20480.config import config  # noqa: E402
from backend.algorithms.hybrid.m0_slide import (  # noqa: E402
    _STITCH_SCRATCH,
    _stitch_overlay_slide,
    compute_tile_geometry,
    core_crop_bounds,
)

TILE = 64
OVERLAP = 16


@pytest.mark.parametrize("backend", ["pyvips", "tifffile"])
def test_stitch_removes_scratch(tmp_path: Path, backend: str, monkeypatch) -> None:
    monkeypatch.setattr(config, "stitch_backend", backend, raising=False)
    # 2x2 格線，步距 = tile - overlap（與 precut 同一條格線規則）。
    step = TILE - OVERLAP
    positions = [(x, y) for y in (0, step) for x in (0, step)]
    geometry = compute_tile_geometry(positions, TILE, OVERLAP)

    scratch = tmp_path / _STITCH_SCRATCH
    expect = {}
    for i, (ax, ay) in enumerate(positions):
        lx0, lx1, ly0, ly1 = core_crop_bounds(geometry, ax, ay, TILE)
        # 每格填不同灰階，拼錯位會在下面的逐格比對露餡。
        crop = np.full((ly1 - ly0, lx1 - lx0, 3), 40 * (i + 1), dtype=np.uint8)
        expect[(ax, ay)] = crop
        scratch.mkdir(parents=True, exist_ok=True)
        io.imsave(str(scratch / f"tile_x{ax}_y{ay}.tiff"), crop,
                  check_contrast=False)

    _stitch_overlay_slide(tmp_path, geometry)

    slide_path = tmp_path / "overlay_slide.tiff"
    assert slide_path.exists()
    assert not scratch.exists(), "拼完必須刪掉暫存夾，否則中間檔仍留在硬碟"
    assert list(tmp_path.iterdir()) == [slide_path]

    # 版面：各格照 (欄, 列) 貼回原位，尺寸與內容都要對得上。
    slide = pyvips.Image.new_from_file(str(slide_path))
    arr = np.ndarray(
        buffer=slide.write_to_memory(),
        dtype=np.uint8,
        shape=(slide.height, slide.width, slide.bands),
    )
    oy = 0
    for ay in sorted(geometry.row_of):
        ox = 0
        for ax in sorted(geometry.col_of):
            crop = expect[(ax, ay)]
            h, w = crop.shape[:2]
            assert np.array_equal(arr[oy:oy + h, ox:ox + w], crop), (ax, ay)
            ox += w
        oy += crop.shape[0]
    assert (arr.shape[0], arr.shape[1]) == (oy, ox)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
