"""
M0: 預切塊器 (pre-cut tiler)

把 hybrid pipeline 的「整檔載入」讀取改成「以 pyvips 隨機存取預先切成重疊 tile 檔」，
讓醫師能丟比單一 tile 大很多的 ROI / WSI 而不整載爆記憶體（現況 20k² ROI 峰值 ≈
31GB）；切好的 tile 檔再交由分析階段逐塊（可平行）處理。

設計重點
--------
- 後端：``pyvips.Image.new_from_file(access="random")`` —— 已在
  ``scripts/tile_generator.py`` 的已配準 LZW/JPEG 檔上跑通。
- 視窗格線直接沿用 ``m2_segmentation._overlap_window_coords``（邊長 = ``tile_size``、
  ``stride = tile_size - overlap``，最後一格貼齊邊界覆蓋全圖），與既有「重疊視窗 +
  IoMin 去重」語義一致，不另造一套切塊規則。
- IHC / DISH 兩路以**相同** ``(abs_x, abs_y)`` 對齊切割，檔名同為
  ``tile_x{x}_y{y}.tiff``，供 ``m1_overlay.find_paired_tiles`` 依同名配對。
- 邊界不足整塊時以白底補滿（gravity/extend 範式），與 M1
  ``background_fill_value = 255`` 一致。

回歸基準
--------
tile 以無失真 ``deflate`` 壓縮寫出：本模組以 pyvips 解碼，實測對 JPEG-TIFF 與
``skimage.io.imread`` 逐位元相同，故無失真 tile 檔可維持與現行單圖路徑「逐位元
一致」的回歸基準。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import pyvips

# Disable pyvips operation cache: each crop() call on a WSI would otherwise be
# cached in C heap indefinitely, causing gradual RAM growth across thousands of tiles.
pyvips.cache_set_max(0)

# 中繼 tile 壓縮：無失真 deflate，保持與 skimage 解碼逐位元一致的回歸基準
# （沿用 ``scripts/tile_generator.py`` 慣例）。
_TILE_COMPRESSION = "deflate"

try:
    from .m2_segmentation import _overlap_window_coords
except ImportError:
    from m2_segmentation import _overlap_window_coords


def _open_rgb(path: Path) -> pyvips.Image:
    """以隨機存取開檔並正規化為 3 通道 uint8（對齊 ``_read_rgb`` 語義）。"""
    img = pyvips.Image.new_from_file(str(path), access="random")
    if img.bands == 4:           # RGBA → 去 alpha
        img = img.extract_band(0, n=3)
    elif img.bands == 1:         # 灰階 → 複製成 3 通道
        img = img.bandjoin([img, img])
    if img.format != "uchar":
        img = img.cast("uchar")
    return img


def _crop_to_tile(img: pyvips.Image, x: int, y: int, tile: int) -> pyvips.Image:
    """裁出 ``(tile, tile)`` 子塊；越界部分以白底補滿。"""
    w = min(tile, img.width - x)
    h = min(tile, img.height - y)
    crop = img.crop(x, y, w, h)
    if w < tile or h < tile:
        crop = crop.gravity("north-west", tile, tile, extend="white")
    return crop


def read_size(path: Path) -> Tuple[int, int]:
    """回傳影像 ``(height, width)``（只讀檔頭，不解碼像素）。"""
    img = pyvips.Image.new_from_file(str(path), access="random")
    return img.height, img.width


def chunk_offsets(
    height: int,
    width: int,
    tile_size: int = 1024,
    overlap: int = 256,
) -> List[Tuple[int, int]]:
    """回傳所有分塊的 ``(abs_x, abs_y)`` 左上角座標（供縫合層預先得知幾何）。"""
    return [
        (x0, y0)
        for (y0, x0, _y1, _x1) in _overlap_window_coords(height, width, tile_size, overlap)
    ]


def precut_paired_tiles(
    ihc_path: Path,
    dish_path: Path,
    ihc_out_dir: Path,
    dish_out_dir: Path,
    tile_size: int = 1024,
    overlap: int = 256,
    workers: int = 8,
) -> List[Tuple[int, int]]:
    """把對齊的 IHC / DISH 影像預切成重疊 tile 檔，寫入兩個輸出目錄。

    兩路以同一視窗格線、相同 offset 切割；每塊均為 ``tile_size × tile_size``
    （邊界以白底補滿），IHC 與 DISH 同名寫成 ``tile_x{x}_y{y}.tiff``，供
    ``m1_overlay.find_paired_tiles`` 依同名 zip 配對。以 ThreadPoolExecutor 平行
    處理裁切/寫檔（I/O bound；pyvips C 運算會釋放 GIL）。

    Args:
        ihc_path: IHC tile / ROI / WSI 影像路徑。
        dish_path: DISH 影像路徑（須與 IHC 同尺寸）。
        ihc_out_dir: IHC tile 輸出目錄（自動建立）。
        dish_out_dir: DISH tile 輸出目錄（自動建立）。
        tile_size: 分塊邊長（pixels）。
        overlap: 相鄰分塊重疊寬度（pixels）；``stride = tile_size - overlap``。
        workers: 平行裁切/寫檔的執行緒數。

    Returns:
        已寫出的所有 ``(abs_x, abs_y)`` 左上角座標（供呼叫端記錄/驗證）。

    Raises:
        ValueError: IHC/DISH 尺寸不一致，或任一邊 < ``tile_size``。
    """
    ihc_img = _open_rgb(Path(ihc_path))
    dish_img = _open_rgb(Path(dish_path))

    if (ihc_img.width, ihc_img.height) != (dish_img.width, dish_img.height):
        raise ValueError(
            f"IHC/DISH 尺寸不一致: ihc={(ihc_img.height, ihc_img.width)} "
            f"vs dish={(dish_img.height, dish_img.width)}"
        )

    h, w = ihc_img.height, ihc_img.width
    if min(h, w) < tile_size:
        raise ValueError(
            f"patch 邊長 {h}x{w} 小於最小允許尺寸 {tile_size}px——拒絕處理。"
        )

    ihc_out_dir = Path(ihc_out_dir)
    dish_out_dir = Path(dish_out_dir)
    ihc_out_dir.mkdir(parents=True, exist_ok=True)
    dish_out_dir.mkdir(parents=True, exist_ok=True)

    positions = chunk_offsets(h, w, tile_size, overlap)

    def _cut(pos: Tuple[int, int]) -> None:
        x, y = pos
        name = f"tile_x{x}_y{y}.tiff"
        _crop_to_tile(ihc_img, x, y, tile_size).write_to_file(
            str(ihc_out_dir / name), compression=_TILE_COMPRESSION
        )
        _crop_to_tile(dish_img, x, y, tile_size).write_to_file(
            str(dish_out_dir / name), compression=_TILE_COMPRESSION
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(_cut, positions))

    return positions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="預切：把對齊的 IHC/DISH ROI/WSI 切成重疊 tile 檔於磁碟。"
    )
    parser.add_argument("--ihc", required=True, type=Path, help="IHC 影像路徑")
    parser.add_argument("--dish", required=True, type=Path, help="DISH 影像路徑（須與 IHC 同尺寸）")
    parser.add_argument("--ihc-out", required=True, type=Path, help="IHC tile 輸出目錄")
    parser.add_argument("--dish-out", required=True, type=Path, help="DISH tile 輸出目錄")
    parser.add_argument("--tile-size", type=int, default=1024, help="分塊邊長 (px)")
    parser.add_argument("--overlap", type=int, default=256, help="相鄰分塊重疊寬度 (px)")
    parser.add_argument("--workers", type=int, default=8, help="平行裁切/寫檔執行緒數")
    args = parser.parse_args()

    written = precut_paired_tiles(
        args.ihc,
        args.dish,
        args.ihc_out,
        args.dish_out,
        tile_size=args.tile_size,
        overlap=args.overlap,
        workers=args.workers,
    )
    print(
        f"完成：共寫出 {len(written)} 組 tile "
        f"→ IHC={args.ihc_out} / DISH={args.dish_out}"
    )
