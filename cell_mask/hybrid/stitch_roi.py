"""
ROI 拼接實驗工具

把 ``test_picture/`` 內依座標連續的 1k tile 拼成單張大圖，
用來驗證「整塊 ROI 一次餵進去、不切 tile」能否自然解決跨縫細胞遺失
（Cellpose 內部 overlap-tiling + 縫合 vs. 現行 per-tile clear_border）。

用法:
    python stitch_roi.py

產出:
    test_picture/stitched/her2/tile_x{x0}_y{y0}.tiff
    test_picture/stitched/dish/tile_x{x0}_y{y0}.tiff
並印出最終尺寸與內部接縫座標（即原本 tile 邊界，比對細胞時看這幾條線）。
"""

import re
from pathlib import Path

import numpy as np
from skimage import io

_TILE_RE = re.compile(r"tile_x(\d+)_y(\d+)")
TILE = 1024


def _coord_map(tile_dir: Path) -> dict:
    """{(x, y): path}，掃描 tile_dir 內所有 tile_x{int}_y{int} 檔。"""
    out = {}
    for p in tile_dir.iterdir():
        m = _TILE_RE.search(p.name)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = p
    return out


def stitch(tile_dir: Path):
    """把 tile_dir 內的連續網格拼成單張圖。

    Returns:
        (image, seams_y, seams_x, (x0, y0))。
        image: ``(H, W, 3)`` uint8；缺格以背景值 255 填充。
        seams_*: 內部接縫像素座標（原 tile 邊界）。
        (x0, y0): 左上角 tile 的全域座標。
    """
    coords = _coord_map(tile_dir)
    xs = sorted({x for x, _ in coords})
    ys = sorted({y for _, y in coords})
    x0, y0 = xs[0], ys[0]
    cols = (xs[-1] - x0) // TILE + 1
    rows = (ys[-1] - y0) // TILE + 1

    canvas = np.full((rows * TILE, cols * TILE, 3), 255, dtype=np.uint8)
    for (x, y), p in coords.items():
        r = (y - y0) // TILE
        c = (x - x0) // TILE
        canvas[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = io.imread(str(p))[..., :3]

    seams_y = [r * TILE for r in range(1, rows)]
    seams_x = [c * TILE for c in range(1, cols)]
    return canvas, seams_y, seams_x, (x0, y0)


def main() -> None:
    base = Path(__file__).resolve().parent / "test_picture"
    for modality in ("her2", "dish"):
        img, seams_y, seams_x, (x0, y0) = stitch(base / modality)
        dst_dir = base / "stitched" / modality
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"tile_x{x0}_y{y0}.tiff"
        io.imsave(str(dst), img, check_contrast=False)
        print(f"[{modality}] {img.shape} -> {dst}")
        print(f"         內部接縫 y={seams_y} x={seams_x}")


if __name__ == "__main__":
    main()
