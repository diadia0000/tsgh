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

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pyvips

# Disable pyvips operation cache: each crop() call on a WSI would otherwise be
# cached in C heap indefinitely, causing gradual RAM growth across thousands of tiles.
pyvips.cache_set_max(0)

# 中繼 tile 壓縮：無失真 deflate，保持與 skimage 解碼逐位元一致的回歸基準
# （沿用 ``scripts/tile_generator.py`` 慣例）。
_TILE_COMPRESSION = "deflate"

# ``PrecutStream.__iter__`` 同時在飛的切檔工作上限，以每條切檔執行緒計。見該處 docstring。
_INFLIGHT_PER_WORKER = 8

try:
    from ..m2_segmentation import _overlap_window_coords
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


class PrecutStream:
    """格線先給、tile 檔邊切邊送的預切串流（讓分析迴圈不必空等整批切完）。

    ``precut_paired_tiles`` 要把整張圖切完才回傳，分析迴圈因此空等一整段：large 錨點
    實測 20.5 s = 3.8% wall，且與 tile 數線性成長，是「完全序列且完全在兩段式重疊之外」
    的兩個階段中較大的那個（另一個是縫合 D）。但**格線本身只要讀檔頭就能算出**
    （``read_size`` 不解碼像素），所以幾何可以先交出去，像素邊切邊送。

    ``positions`` 為完整格線，供 ``compute_tile_geometry`` 先建幾何；迭代則以**完成順序**
    產出 ``(ihc_tile, dish_tile, (abs_x, abs_y))``。完成順序不影響輸出：``run_batch``
    最後會把所有塊的細胞攤平、依 ``(abs_y, abs_x, cell_id)`` 全域排序後才重編號，縫合階段
    也是按座標讀檔，兩者都與處理順序無關。

    切檔仍走與 ``precut_paired_tiles`` 相同的 ``_crop_to_tile`` + deflate 路徑，故產出的
    tile 檔逐位元相同——這是「重疊」而非「換演算法」。
    """

    def __init__(
        self,
        ihc_path: Path,
        dish_path: Path,
        ihc_out_dir: Path,
        dish_out_dir: Path,
        tile_size: int = 1024,
        overlap: int = 256,
        workers: int = 8,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """``region`` = ``(x, y, w, h)`` 的分析範圍（ROI），單位為**全片像素**；
        ``None`` 表示整張片（原行為）。

        格線在 ROI 的尺寸上算，再整體平移回 ROI 原點，因此 ``positions`` 與 tile 檔名
        帶的仍是**全片絕對座標**——``compute_tile_geometry`` 的切線、核心區歸屬去重、
        ``filter_and_absolutize`` 的質心絕對化全部照舊，不需要任何額外的座標換算。

        ROI 邊界不會被切出去：``_overlap_window_coords`` 的最後一格會往回貼齊範圍
        邊界，所以每一塊都完整落在 ROI 內（前提是 ROI 邊長 ≥ ``tile_size``，下方檢查）。
        邊界上的細胞由 ``clear_slide_edge_cells`` 依 ``geometry.edge_flags()`` 清除——
        對 ROI 而言那正是「被範圍切一半的細胞不該計入」。
        """
        self._ihc_img = _open_rgb(Path(ihc_path))
        self._dish_img = _open_rgb(Path(dish_path))

        if (self._ihc_img.width, self._ihc_img.height) != (
            self._dish_img.width,
            self._dish_img.height,
        ):
            raise ValueError(
                f"IHC/DISH 尺寸不一致: "
                f"ihc={(self._ihc_img.height, self._ihc_img.width)} "
                f"vs dish={(self._dish_img.height, self._dish_img.width)}"
            )

        full_h, full_w = self._ihc_img.height, self._ihc_img.width
        origin_x, origin_y, h, w = 0, 0, full_h, full_w
        if region is not None:
            origin_x, origin_y, w, h = region
            if origin_x < 0 or origin_y < 0 or w <= 0 or h <= 0:
                raise ValueError(f"ROI 不合法: {region}")
            if origin_x + w > full_w or origin_y + h > full_h:
                raise ValueError(
                    f"ROI {region} 超出切片範圍 {full_w}x{full_h}（x+w / y+h 不得越界）"
                )

        if min(h, w) < tile_size:
            raise ValueError(
                f"patch 邊長 {h}x{w} 小於最小允許尺寸 {tile_size}px——拒絕處理。"
            )

        self.ihc_out_dir = Path(ihc_out_dir)
        self.dish_out_dir = Path(dish_out_dir)
        self.ihc_out_dir.mkdir(parents=True, exist_ok=True)
        self.dish_out_dir.mkdir(parents=True, exist_ok=True)
        self.tile_size = tile_size
        self.workers = workers
        self.region = region
        self.positions = [
            (x + origin_x, y + origin_y)
            for (x, y) in chunk_offsets(h, w, tile_size, overlap)
        ]

    def _cut(self, pos: Tuple[int, int]) -> Tuple[Path, Path, Tuple[int, int]]:
        x, y = pos
        name = f"tile_x{x}_y{y}.tiff"
        ihc_dst = self.ihc_out_dir / name
        dish_dst = self.dish_out_dir / name
        _crop_to_tile(self._ihc_img, x, y, self.tile_size).write_to_file(
            str(ihc_dst), compression=_TILE_COMPRESSION
        )
        _crop_to_tile(self._dish_img, x, y, self.tile_size).write_to_file(
            str(dish_dst), compression=_TILE_COMPRESSION
        )
        return ihc_dst, dish_dst, pos

    def __iter__(self) -> Iterator[Tuple[Path, Path, Tuple[int, int]]]:
        """以完成順序產出已落地的 tile 配對；任一塊切檔失敗在此 raise（fail-fast）。

        同時在飛的切檔工作以 ``workers * _INFLIGHT_PER_WORKER`` 為上限，而不是開場就把
        整片 submit 出去。全部 submit 的版本讓「消費端放棄」變成一句空話：fail-fast 之後
        ``_run_tiles_multiprocess._feed`` 會 break、本產生器被丟著，但佇列裡的切檔工作一件
        都不會少做——本輪實測在第 8 塊中止的 576 塊批次，仍然把 576 塊全部切完
        （``scripts/exit_latency_probe.py``）。整片規模等於在整批已放棄之後還要再切十幾分鐘、
        寫掉數十 GB，正是 ``_feed`` 裡那句「中止後必須真的停下來」要擋、卻擋不到的事。

        有界視窗不會讓切檔變成瓶頸：切一對塊約數十毫秒，而每塊的分析在 ``workers=1`` 是
        數百毫秒，切檔本來就領先分析一個數量級，視窗只是不讓它領先「整片」。放棄時
        ``cancel_futures=True`` 把還沒開跑的直接取消，只需等最多 ``workers`` 塊跑完。
        完整跑完的那條路徑產出與順序性質一字不變。
        """
        pool = ThreadPoolExecutor(max_workers=self.workers)
        try:
            todo = iter(self.positions)
            inflight = {
                pool.submit(self._cut, pos)
                for pos in islice(todo, self.workers * _INFLIGHT_PER_WORKER)
            }
            while inflight:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done:
                    nxt = next(todo, None)
                    if nxt is not None:
                        inflight.add(pool.submit(self._cut, nxt))
                    yield fut.result()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
