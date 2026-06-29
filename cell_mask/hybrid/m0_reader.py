"""
M0: 分塊讀取器 (chunked reader)

把 hybrid pipeline 的「整檔載入」讀取改成「以 pyvips 隨機存取逐塊讀」，讓醫師能丟
比單一 tile 大很多的 ROI / WSI 而不整載爆記憶體（現況 20k² ROI 峰值 ≈ 31GB）。

設計重點
--------
- 後端：``pyvips.Image.new_from_file(access="random")`` —— 已在
  ``thriple_image_layer/module5_tile_generator.py`` 的已配準 LZW/JPEG 檔上跑通。
- 視窗格線直接沿用 ``m2_segmentation._overlap_window_coords``（邊長 = ``tile_size``、
  ``stride = tile_size - overlap``，最後一格貼齊邊界覆蓋全圖），與既有「重疊視窗 +
  IoMin 去重」語義一致，不另造一套切塊規則。
- IHC / DISH 兩路以**相同** ``(abs_x, abs_y)`` 對齊讀取。
- 邊界不足整塊時以白底補滿（沿用 ``module5_tile_generator._crop_tile`` 的
  gravity/extend 範式），與 M1 ``background_fill_value = 255`` 一致。

回歸基準
--------
輸入 ≤ ``tile_size`` 時退化為單塊 = 讀整檔：本模組以 pyvips 解碼，實測對
JPEG-TIFF 與 ``skimage.io.imread`` 逐位元相同，故單塊輸出可作為與現行
``hybrid_pipeline._read_rgb`` 路徑「逐位元一致」的回歸基準。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import pyvips

from m2_segmentation import _overlap_window_coords


@dataclass
class Chunk:
    """單一分塊：對齊的 IHC / DISH RGB uint8 與其在全圖的絕對左上角座標。

    Attributes:
        ihc: ``(tile_size, tile_size, 3)`` uint8 RGB。
        dish: ``(tile_size, tile_size, 3)`` uint8 RGB（與 ``ihc`` 同 offset/尺寸）。
        abs_x: 此塊左上角在全圖的絕對 x（欄）座標。
        abs_y: 此塊左上角在全圖的絕對 y（列）座標。
    """

    ihc: np.ndarray
    dish: np.ndarray
    abs_x: int
    abs_y: int


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
    """裁出 ``(tile, tile)`` 子塊；越界部分以白底補滿（mirror module5 _crop_tile）。"""
    w = min(tile, img.width - x)
    h = min(tile, img.height - y)
    crop = img.crop(x, y, w, h)
    if w < tile or h < tile:
        crop = crop.gravity("north-west", tile, tile, extend="white")
    return crop


def _to_numpy(img: pyvips.Image) -> np.ndarray:
    """pyvips Image → 可寫的連續 ``(H, W, 3)`` uint8 numpy。"""
    arr = np.frombuffer(img.write_to_memory(), dtype=np.uint8)
    arr = arr.reshape(img.height, img.width, img.bands)
    return np.ascontiguousarray(arr[:, :, :3])


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


def iter_paired_chunks(
    ihc_path: Path,
    dish_path: Path,
    tile_size: int = 1024,
    overlap: int = 256,
) -> Iterator[Chunk]:
    """逐塊讀取對齊的 IHC / DISH，yield ``Chunk(ihc, dish, abs_x, abs_y)``。

    兩路以同一視窗格線、相同 offset 讀取；每塊均為 ``tile_size × tile_size``
    （邊界以白底補滿）。輸入 ≤ ``tile_size`` 時僅 yield 單塊（= 讀整檔）。

    Args:
        ihc_path: IHC tile / ROI / WSI 影像路徑。
        dish_path: DISH 影像路徑（須與 IHC 同尺寸）。
        tile_size: 分塊邊長（pixels）。
        overlap: 相鄰分塊重疊寬度（pixels）；``stride = tile_size - overlap``。

    Yields:
        ``Chunk``，依列優先（先 x 後 y）順序。

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

    for y0, x0, _y1, _x1 in _overlap_window_coords(h, w, tile_size, overlap):
        yield Chunk(
            ihc=_to_numpy(_crop_to_tile(ihc_img, x0, y0, tile_size)),
            dish=_to_numpy(_crop_to_tile(dish_img, x0, y0, tile_size)),
            abs_x=x0,
            abs_y=y0,
        )
