"""
M0 縫合層：分塊 (per-chunk) 幾何與「質心 core-ownership」去重的純函式。

純資料重組，不碰模型、不配置任何全圖陣列——可用合成資料單元測試。

背景（架構變更）
----------------
舊版 ``StitchAccumulator`` 會把每塊 M1–M3 結果貼進「六張全圖大小的 numpy 陣列」
（instance_mask / dish_nucleus_mask / core_mask / masked_ihc / dish_mask_overlay /
overlay_image）。對真實 WSI（~156222×134028）這是數百 GB 的記憶體，已完全移除。

新流程：
- 5 張 per-chunk 陣列（instance_mask / dish_nucleus_mask / core_mask / masked_ihc /
  dish_mask_overlay）由**別的任務**逐塊、以局部 ID 原樣寫檔，不做合併。
- ``overlay_image`` 由**別的任務**在最後用 pyvips arrayjoin+tiffsave 惰性拼接，
  同樣不需要全圖陣列。
- 唯一仍需「全域、去重、合併」的是**表格式**細胞結果（``CellAnalysisResult`` 清單，
  最終餵給 m4_module/csv 的 ``export_tile_csv`` / ``export_summary_statistics``）。
  因為是「列」不是「像素」，成本極低。

本模組即提供這條表格路徑所需的兩支純函式：

1. ``compute_tile_geometry(positions, tile_size, overlap)`` → ``TileGeometry``
   由整批分塊左上角座標算出核心區切線 (``cuts_x`` / ``cuts_y``)、欄列查表
   (``col_of`` / ``row_of``) 與四邊界成員資訊；並驗證分塊構成完整無缺格的格線。
2. ``filter_and_absolutize(cr, geometry, abs_x, abs_y)`` → ``List[CellAnalysisResult]``
   對單塊結果做「質心 core-ownership」去重並把質心平移成絕對座標。

兩支皆為**無共享可變狀態的純函式**，可安全地被平行 worker 逐塊獨立呼叫。

核心規則（不變）
----------------
質心 core-ownership 去重（交接 §5.4）：把每個分塊沿 ``overlap/2`` 切出互不重疊、
鋪滿全圖的「核心區」；一顆細胞只算在「其質心落在哪個分塊核心區」的那一塊。重疊帶
的重複偵測因此自動消除——不需再跑 IoMin。
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np

try:
    from .hybrid_data_types import CellAnalysisResult, CellDotResult, DetectedDot
    from .m2_segmentation import _relabel_sequential, _remove_border_cells
except ImportError:
    from hybrid_data_types import CellAnalysisResult, CellDotResult, DetectedDot
    from m2_segmentation import _relabel_sequential, _remove_border_cells


@dataclass
class ChunkResult:
    """單一分塊跑完 M1–M3 的完整產物（座標皆為分塊局部）。"""

    abs_x: int
    abs_y: int
    instance_mask: np.ndarray        # (th, tw) int32, 局部細胞 ID（已清真實 slide 邊、relabel）
    dish_nucleus_mask: np.ndarray    # (th, tw) int32, 局部 DISH 核 ID（detect_all_dots 過濾後）
    core_mask: np.ndarray            # (th, tw) uint8{0,1}
    masked_ihc: np.ndarray           # (th, tw, 3) uint8
    dish_mask_overlay: np.ndarray    # (th, tw, 3) uint8
    results: List[CellAnalysisResult]
    all_dots: List[DetectedDot]
    per_cell_dots: Dict[int, CellDotResult]


# ------------------------------------------------------------------
# 真實 slide 邊界清除（per-chunk，M3 之前）
# ------------------------------------------------------------------

def clear_slide_edge_cells(
    mask: np.ndarray,
    clear_top: bool,
    clear_bottom: bool,
    clear_left: bool,
    clear_right: bool,
) -> np.ndarray:
    """移除碰觸「指定」邊的細胞並重新編號；分塊內部接縫邊不清。

    四邊皆指定時 == ``m2_segmentation._remove_border_cells``（= skimage clear_border +
    relabel），故單塊（四邊皆真實 slide 邊）逐位元等同現行 M2 清邊行為。
    """
    if clear_top and clear_bottom and clear_left and clear_right:
        return _remove_border_cells(mask)
    if not (clear_top or clear_bottom or clear_left or clear_right):
        return _relabel_sequential(mask)

    remove: set = set()
    if clear_top:
        remove |= set(np.unique(mask[0, :]))
    if clear_bottom:
        remove |= set(np.unique(mask[-1, :]))
    if clear_left:
        remove |= set(np.unique(mask[:, 0]))
    if clear_right:
        remove |= set(np.unique(mask[:, -1]))
    remove.discard(0)

    if remove:
        mask = mask.copy()
        for rid in remove:
            mask[mask == rid] = 0
    return _relabel_sequential(mask)


# ------------------------------------------------------------------
# 分塊幾何
# ------------------------------------------------------------------

def _cut_lines(starts: List[int], overlap: int) -> List[int]:
    """相鄰分塊核心區的分界線：在後一塊起點再進 ``overlap//2`` 處切。"""
    return [starts[i] + overlap // 2 for i in range(1, len(starts))]


def _validate_axis(starts: List[int], stride: int, axis: str) -> None:
    """驗證單一軸的分塊起點構成完整無缺格的格線（對齊 ``_overlap_window_coords``）。

    約定（見 ``m2_segmentation._overlap_window_coords`` / ``m0_reader.chunk_offsets``）：
    起點為 ``0, stride, 2*stride, ...``，最後一格貼齊邊界 (``length - tile``)，故最後一段
    間距落在 ``[1, stride]``；其餘相鄰間距必等於 ``stride``。任一違反即代表分塊有缺格 /
    重複 / 對不上格線。
    """
    if starts[0] != 0:
        raise ValueError(
            f"{axis} 軸起點必須從 0 開始，實得 {starts[0]}——分塊未對齊格線。"
        )
    n = len(starts)
    for i in range(1, n):
        gap = starts[i] - starts[i - 1]
        if i == n - 1:
            if not (1 <= gap <= stride):
                raise ValueError(
                    f"{axis} 軸最後一段間距 {gap} 不在 [1, {stride}]——"
                    f"分塊未構成完整無缺格的格線：{starts}"
                )
        elif gap != stride:
            raise ValueError(
                f"{axis} 軸間距 {gap} != stride {stride}"
                f"（在 {starts[i - 1]}→{starts[i]}）——分塊有缺格：{starts}"
            )


@dataclass(frozen=True)
class TileGeometry:
    """整批分塊的縫合幾何（唯讀、可安全共享給平行 worker）。

    Attributes:
        cuts_x: x 方向核心區切線（``bisect_right(cuts_x, gx)`` 得該點所屬欄）。
        cuts_y: y 方向核心區切線。
        col_of: ``abs_x`` → 欄索引。
        row_of: ``abs_y`` → 列索引。
        x_min / x_max: 最左 / 最右欄的 ``abs_x``（碰觸真實 slide 左 / 右緣者）。
        y_min / y_max: 最上 / 最下列的 ``abs_y``（碰觸真實 slide 上 / 下緣者）。
    """

    cuts_x: List[int]
    cuts_y: List[int]
    col_of: Dict[int, int]
    row_of: Dict[int, int]
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    def edge_flags(self, abs_x: int, abs_y: int) -> Tuple[bool, bool, bool, bool]:
        """回傳該塊的 ``(clear_top, clear_bottom, clear_left, clear_right)``。

        供呼叫端直接餵給 ``clear_slide_edge_cells()``：只有位在最外欄 / 列的分塊
        才碰觸真實 slide 邊界，內部接縫邊不清（交由核心區去重）。
        """
        return (
            abs_y == self.y_min,
            abs_y == self.y_max,
            abs_x == self.x_min,
            abs_x == self.x_max,
        )


def compute_tile_geometry(
    positions: List[Tuple[int, int]],
    tile_size: int,
    overlap: int,
) -> TileGeometry:
    """由整批分塊左上角座標算出縫合幾何，並驗證格線完整。

    純函式、無副作用。呼叫端（``process_single_tile`` 或新的 batch driver）在跑分塊
    迴圈前呼叫一次，之後對每塊以 ``filter_and_absolutize`` / ``edge_flags`` 消費。

    Args:
        positions: 本批每塊的 ``(abs_x, abs_y)``（來自檔名解析或 ``chunk_offsets``）。
        tile_size: 分塊邊長（pixels）；與 ``config.default_tile_size`` 同值。
        overlap: 相鄰分塊重疊寬度（pixels）；與 ``config.window_overlap_px`` 同值。
            切線取在後一塊起點再進 ``overlap // 2`` 處（同舊 ``StitchAccumulator``）。

    Returns:
        ``TileGeometry``。

    Raises:
        ValueError: positions 為空、有重複、或不構成完整無缺格的矩形格線。
    """
    if not positions:
        raise ValueError("compute_tile_geometry: positions 為空。")

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(
            f"tile_size ({tile_size}) 必須大於 overlap ({overlap})，否則 stride<=0。"
        )

    x_starts = sorted({x for x, _y in positions})
    y_starts = sorted({y for _x, y in positions})
    _validate_axis(x_starts, stride, "x")
    _validate_axis(y_starts, stride, "y")

    # 完整矩形格線：無重複，且恰為 x_starts × y_starts 的笛卡兒積（否則有缺塊）。
    if len(set(positions)) != len(positions):
        raise ValueError(f"positions 含重複分塊座標：{positions}")
    expected = len(x_starts) * len(y_starts)
    if len(positions) != expected:
        raise ValueError(
            f"分塊不完整：得 {len(positions)} 塊，格線應為 "
            f"{len(x_starts)}(欄)×{len(y_starts)}(列)={expected} 塊——有缺格。"
        )

    return TileGeometry(
        cuts_x=_cut_lines(x_starts, overlap),
        cuts_y=_cut_lines(y_starts, overlap),
        col_of={x0: i for i, x0 in enumerate(x_starts)},
        row_of={y0: i for i, y0 in enumerate(y_starts)},
        x_min=x_starts[0],
        x_max=x_starts[-1],
        y_min=y_starts[0],
        y_max=y_starts[-1],
    )


def core_crop_bounds(
    geometry: TileGeometry,
    abs_x: int,
    abs_y: int,
    tile_size: int,
) -> Tuple[int, int, int, int]:
    """回傳本塊核心區在『本塊局部座標』下的裁切邊界 (lx0, lx1, ly0, ly1)。

    供呼叫端裁切本塊自己的局部陣列（例如已標註的 overlay），只留下此塊
    「core-ownership 擁有」的矩形範圍，之後可用 pyvips arrayjoin 無重疊、無縫隙地
    拼回全片——與舊 ``StitchAccumulator.add()`` 的 ``gx_lo/gx_hi/gy_lo/gy_hi``
    （全域座標，貼進全域陣列）邏輯等價，只是這裡改回傳局部座標，不需要全域陣列
    也不需要 full_h/full_w。

    邊界欄/列（``col == 0`` 或 ``col == len(cuts_x)``，row 同理）的核心區一路
    延伸到該塊自己的真實邊緣：因為 ``compute_tile_geometry`` 已驗證格線從 0 開始
    （``x_min == y_min == 0``）且依 ``_overlap_window_coords`` 慣例最後一欄/列必貼齊
    ``full_w/full_h - tile_size``（即 ``x_max + tile_size == full_w``），最外欄/列本身
    的 ``abs_x``/``abs_x + tile_size`` 就等於全域的 ``0``/``full_w``——不需額外傳入
    full_w/full_h。非邊界欄/列則以 ``cuts_x``/``cuts_y`` 為界。

    Args:
        geometry: ``compute_tile_geometry`` 的輸出。
        abs_x: 本塊左上角絕對 x。
        abs_y: 本塊左上角絕對 y。
        tile_size: 分塊邊長（pixels）；本塊局部陣列的 shape 即為
            ``(tile_size, tile_size, ...)``。

    Returns:
        ``(lx0, lx1, ly0, ly1)``——套用時取 ``local_array[ly0:ly1, lx0:lx1]``。
    """
    col = geometry.col_of[abs_x]
    row = geometry.row_of[abs_y]
    cuts_x, cuts_y = geometry.cuts_x, geometry.cuts_y

    gx_lo = cuts_x[col - 1] if col > 0 else abs_x
    gx_hi = cuts_x[col] if col < len(cuts_x) else abs_x + tile_size
    gy_lo = cuts_y[row - 1] if row > 0 else abs_y
    gy_hi = cuts_y[row] if row < len(cuts_y) else abs_y + tile_size

    return gx_lo - abs_x, gx_hi - abs_x, gy_lo - abs_y, gy_hi - abs_y


# ------------------------------------------------------------------
# 質心 core-ownership 去重 + 座標絕對化
# ------------------------------------------------------------------

def filter_and_absolutize(
    cr: ChunkResult,
    geometry: TileGeometry,
    abs_x: int,
    abs_y: int,
) -> List[CellAnalysisResult]:
    """對單塊結果做核心區去重、把質心平移成絕對座標，回傳表格用的細胞清單。

    對每顆 ``r in cr.results``：
      1. 絕對質心 ``gxc, gyc = abs_x + r.centroid_x, abs_y + r.centroid_y``。
      2. 由 ``geometry`` 查出本塊的 ``col`` / ``row``。
      3. **只保留**質心落在本塊核心區者，即
         ``bisect_right(cuts_x, gxc) == col and bisect_right(cuts_y, gyc) == row``。
         （與舊 ``StitchAccumulator.add`` 的 skip 條件同義，只是改寫成 keep 條件。）
      4. 以 ``dataclasses.replace`` 產生新結果，質心換成絕對座標。

    **cell_id 契約（重要）**：本函式**不重編號** ``cell_id``——回傳的每顆仍帶其
    *分塊局部* id。因此跨塊會有 id 碰撞，這是預期的。後續（另一支任務的）合併步驟
    需把所有分塊的清單串起來、以正典幾何順序 ``(abs_y, abs_x, cell_id)`` 排序後，
    重新編成全域 1..N 的 ``cell_id``。呼叫端請保留每塊回傳清單與其 ``(abs_x, abs_y)``
    的對應，供該排序使用（質心雖已絕對化，但排序鍵約定用分塊座標而非質心）。

    **不處理 per_cell_dots / all_dots / dish 核 ID**：``report.csv`` 欄位為
    ``cell_id, centroid_x, centroid_y, reddot, blackdot, score``、``summary.txt`` 只需
    彙總 count，兩者皆只讀 ``CellAnalysisResult`` 欄位（見 m4_module/csv.py），故表格
    路徑不需要點位 / 核 ID。點位與 DISH 核的視覺化（overlay、per-cell crop）改在
    per-chunk 層以局部 ID 產出（另一支任務負責），本函式不涉入。

    Args:
        cr: 單塊 M1–M3 產物。
        geometry: ``compute_tile_geometry`` 的輸出。
        abs_x: 本塊左上角絕對 x（= ``cr.abs_x``；顯式傳入以利平行呼叫）。
        abs_y: 本塊左上角絕對 y（= ``cr.abs_y``）。

    Returns:
        本塊「核心區擁有」的 ``CellAnalysisResult`` 清單，質心已絕對化，
        ``cell_id`` 仍為分塊局部值。
    """
    col = geometry.col_of[abs_x]
    row = geometry.row_of[abs_y]
    cuts_x, cuts_y = geometry.cuts_x, geometry.cuts_y

    owned: List[CellAnalysisResult] = []
    for r in cr.results:
        gxc = abs_x + r.centroid_x
        gyc = abs_y + r.centroid_y
        if bisect_right(cuts_x, gxc) == col and bisect_right(cuts_y, gyc) == row:
            owned.append(replace(r, centroid_x=gxc, centroid_y=gyc))
    return owned
