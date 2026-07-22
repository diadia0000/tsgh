#!/usr/bin/env python3
"""條狀切片邏輯的最小檢查：strips 必須無縫、無重疊地覆蓋整個 bbox 高度。

執行: python backend/tests/test_module1_strips.py
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithms" / "thriple_image_layer"))
import module1_preprocess as m1


def _strips(tmp: Path, bbox_y: int, bbox_h: int, strip_height: int):
    (tmp / "in").mkdir(parents=True)
    (tmp / "in" / "X.czi").touch()
    m1.aicspylibczi = SimpleNamespace(CziFile=lambda p: SimpleNamespace(
        get_mosaic_bounding_box=lambda: SimpleNamespace(x=7, y=bbox_y, w=100, h=bbox_h)))
    config = SimpleNamespace(
        czi_input_dir=tmp / "in",
        input_dir=tmp / "out",
        preprocess=SimpleNamespace(num_processes=1, strip_height=strip_height),
        modalities=[SimpleNamespace(name="X", filename="X.tiff",
                                    czi_filename="X.czi", scale_factor=1.0)],
    )
    return m1.CziPreprocessor(config).get_conversion_tasks()[0]["strips"]


def test_strips_tile_bbox_exactly():
    for bbox_y, bbox_h, sh in [(0, 1000, 250), (13, 1000, 300), (5, 100, 999)]:
        with tempfile.TemporaryDirectory() as d:
            strips = _strips(Path(d), bbox_y, bbox_h, sh)
        regions = [s["region"] for s in strips]
        assert regions[0][1] == bbox_y, regions
        assert sum(r[3] for r in regions) == bbox_h, regions        # 覆蓋完整高度
        assert all(r[3] > 0 for r in regions), regions              # 沒有空區塊
        for prev, cur in zip(regions, regions[1:]):
            assert prev[1] + prev[3] == cur[1], (prev, cur)         # 首尾相接不重疊
        assert [s["strip_index"] for s in strips] == list(range(len(strips)))


if __name__ == "__main__":
    test_strips_tile_bbox_exactly()
    print("OK")
