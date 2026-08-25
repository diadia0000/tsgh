"""
M0 縫合層：分塊 (per-chunk) 幾何與「質心 core-ownership」去重的純函式。

純資料重組，不碰模型、不配置任何全圖陣列——可用合成資料單元測試。

背景（架構變更）
----------------
舊版 ``StitchAccumulator`` 會把每塊 M1–M3 結果貼進「六張全圖大小的 numpy 陣列」
（instance_mask / dish_nucleus_mask / core_mask / masked_ihc / dish_mask_overlay /
overlay_image）。對真實 WSI（~156222×134028）這是數百 GB 的記憶體，已完全移除。

新流程：
- per-chunk 陣列（instance_mask / dish_nucleus_mask / core_mask / masked_ihc /
  dish_mask_overlay）全部留在記憶體、隨分塊釋放，一張都不寫檔。
- ``overlay_image`` 由 ``hybrid_pipeline._stitch_overlay_slide()`` 在最後用 pyvips
  惰性拼接（逐塊 overlay 先落 ``_stitch_scratch/``，拼完即刪），同樣不需要全圖陣列。
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

（2026-08 更新：上一句的假設只在「兩塊對同一顆細胞算出**完全相同**的質心」時成立。
兩塊各自獨立跑一次 M1/M2 前向，靠近接縫時同一顆細胞的質心常有幾個像素落差，兩邊
的 core-ownership 判定就可能都落在「自己那一塊」，於是各自合法保留一次 → ghost row。
真正的安全網是 ``dedup_cross_tile_duplicates()``，在 ``_finish_batch`` 的全域合併點
跑一次。）
"""
from __future__ import annotations

import logging
import shutil
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

try:  # 僅 POSIX 提供 resource/RLIMIT_NOFILE，見 _ensure_nofile_limit
    import resource
except ImportError:
    resource = None

import numpy as np
import pyvips
from scipy.spatial import cKDTree

try:
    from ..config import config
    from ..hybrid_data_types import CellAnalysisResult, CellDotResult, DetectedDot
    from ..m2_segmentation import _relabel_sequential, _remove_border_cells
except ImportError:
    from backend.algorithms.hybrid.config import config
    from hybrid_data_types import CellAnalysisResult, CellDotResult, DetectedDot
    from m2_segmentation import _relabel_sequential, _remove_border_cells

logger = logging.getLogger(__name__)

# 逐塊標註 overlay 的暫存夾。這不是產物而是縫合用的**串流緩衝**：pyvips 以
# ``access="sequential"`` 惰性讀這些檔再 ``tiffsave``，整片才不必同時在記憶體裡
# （純記憶體拼接就是當初被砍掉的全片 canvas 失效模式）。``_stitch_overlay_slide``
# 寫完 overlay_slide.tiff 後整夾刪除。
_STITCH_SCRATCH = "_stitch_scratch"


@dataclass
class ChunkResult:
    """單一分塊跑完 M1–M3 的完整產物（座標皆為分塊局部）。"""

    abs_x: int
    abs_y: int
    instance_mask: np.ndarray        # (th, tw) int32, 局部細胞 ID（已清真實 slide 邊、relabel）
    dish_nucleus_mask: np.ndarray    # (th, tw) int32, 局部 DISH 核 ID（detect_all_dots 過濾後）
    dish_mask_overlay: np.ndarray    # (th, tw, 3) uint8, M2/M3 計算用（UNet++ 核心遮罩後的 DISH）
    dish: np.ndarray                 # (th, tw, 3) uint8, 原始未遮罩 DISH，標註 overlay 的底圖
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


def _validate_axis(starts: List[int], stride: int, axis: str, origin: int = 0) -> None:
    """驗證單一軸的分塊起點構成完整無缺格的格線（對齊 ``_overlap_window_coords``）。

    約定（見 ``m2_segmentation._overlap_window_coords`` / ``m0_reader.chunk_offsets``）：
    起點為 ``origin, origin+stride, origin+2*stride, ...``，最後一格貼齊邊界
    (``origin + length - tile``)，故最後一段間距落在 ``[1, stride]``；其餘相鄰間距必等於
    ``stride``。任一違反即代表分塊有缺格 / 重複 / 對不上格線。

    ``origin`` 是這批分塊所涵蓋範圍的左上角：整片分析為 ``0``，只分析 ROI 時則為 ROI
    的起點（見 ``m0_reader.PrecutStream(region=...)``）。它由呼叫端明確傳入而非從
    ``starts[0]`` 推得——推得就等於放棄「第一格不見了」這個檢查。
    """
    if starts[0] != origin:
        raise ValueError(
            f"{axis} 軸起點必須從 {origin} 開始，實得 {starts[0]}——分塊未對齊格線。"
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
    origin: Tuple[int, int] = (0, 0),
) -> TileGeometry:
    """由整批分塊左上角座標算出縫合幾何，並驗證格線完整。

    純函式、無副作用。呼叫端（``process_single_tile`` 或新的 batch driver）在跑分塊
    迴圈前呼叫一次，之後對每塊以 ``filter_and_absolutize`` / ``edge_flags`` 消費。

    Args:
        positions: 本批每塊的 ``(abs_x, abs_y)``（來自檔名解析或 ``chunk_offsets``）。
        tile_size: 分塊邊長（pixels）；與 ``config.default_tile_size`` 同值。
        overlap: 相鄰分塊重疊寬度（pixels）；與 ``config.window_overlap_px`` 同值。
            切線取在後一塊起點再進 ``overlap // 2`` 處（同舊 ``StitchAccumulator``）。
        origin: 這批分塊涵蓋範圍的左上角 ``(x, y)``。整片分析是 ``(0, 0)``（預設）；
            只分析 ROI 時為 ROI 起點——``positions`` 一律是全片絕對座標，改變的只有
            格線從哪裡開始。切線、核心區與質心絕對化的算法本身與原點無關
            （``core_crop_bounds`` 對最外欄用的是該塊自己的 ``abs_x``，不是 0）。

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
    _validate_axis(x_starts, stride, "x", origin[0])
    _validate_axis(y_starts, stride, "y", origin[1])

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
    ``cell_id, centroid_x, centroid_y, reddot, blackdot, score, is_her2_positive,
    cell_type``、``summary.txt`` 只需
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


def dedup_cross_tile_duplicates(
    per_tile_owned: List[Tuple[int, int, List[CellAnalysisResult]]],
    max_distance_px: float,
) -> List[Tuple[int, int, List[CellAnalysisResult]]]:
    """Ghost-row 修復：同一顆物理細胞被兩個相鄰分塊各自獨立偵測、質心因各自的
    M1/M2 前向推論而些微不同，導致兩邊的 core-ownership 判定都落在「自己那一塊」，
    因而被 ``filter_and_absolutize`` 各自合法保留一次。

    在全域合併點（``filter_and_absolutize`` 之後、``cell_id`` 重新編號之前）補一道
    保險：只比對「不同分塊」來源的細胞，質心距離 < ``max_distance_px`` 者視為同一顆
    物理細胞。同一分塊內部的偵測完全不受影響——Cellpose 不會在同一次推論裡把同一顆
    細胞切成兩個 instance，只有不同分塊的獨立推論之間才有這種落差；真實相鄰但不同的
    兩顆細胞，中心距通常遠大於這個刻意設得很小的門檻，不會被誤合併。

    每組重複只保留 1 筆，取捨依序：未被排除(``excluded=False``) > 較高
    ``blue_region_count`` > 較多紅+黑點總數（訊號較完整、較可能是未被裁切的那份
    偵測）> 以 ``(abs_y, abs_x, cell_id)`` 決定性打平。

    Args:
        per_tile_owned: 每塊 ``(abs_x, abs_y, owned_results)``，來自
            ``filter_and_absolutize`` 的輸出。
        max_distance_px: 判定為同一顆物理細胞的最大質心距離；``<= 0`` 為 no-op。

    Returns:
        與輸入同型的 ``per_tile_owned``，重複組只保留 1 筆。
    """
    if max_distance_px <= 0:
        return per_tile_owned

    flat_entries: List[Tuple[int, int, CellAnalysisResult]] = [
        (ax, ay, r) for ax, ay, results in per_tile_owned for r in results
    ]
    if len(flat_entries) < 2:
        return per_tile_owned

    coords = np.array(
        [[e[2].centroid_y, e[2].centroid_x] for e in flat_entries], dtype=np.float64
    )
    pairs = cKDTree(coords).query_pairs(r=max_distance_px)

    parent = list(range(len(flat_entries)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    any_merged = False
    for i, j in pairs:
        if flat_entries[i][:2] != flat_entries[j][:2]:  # 僅跨塊才合併
            union(i, j)
            any_merged = True

    if not any_merged:
        return per_tile_owned

    groups: Dict[int, List[int]] = {}
    for idx in range(len(flat_entries)):
        groups.setdefault(find(idx), []).append(idx)

    def _rank(idx: int):
        ax, ay, r = flat_entries[idx]
        return (
            r.excluded,
            -r.blue_region_count,
            -(r.her2_dot_count + r.cep17_dot_count),
            ay, ax, r.cell_id,
        )

    drop: set = set()
    n_groups_merged = 0
    for idxs in groups.values():
        if len(idxs) == 1:
            continue
        keep = min(idxs, key=_rank)
        drop.update(i for i in idxs if i != keep)
        n_groups_merged += 1

    if not drop:
        return per_tile_owned

    logger.info(
        "Ghost-row dedup: %d 組跨塊重複偵測 → 各保留 1 顆，共移除 %d 列。",
        n_groups_merged, len(drop),
    )

    kept_by_tile: Dict[Tuple[int, int], List[CellAnalysisResult]] = {}
    for idx, (ax, ay, r) in enumerate(flat_entries):
        if idx in drop:
            continue
        kept_by_tile.setdefault((ax, ay), []).append(r)

    return [
        (ax, ay, kept_by_tile.get((ax, ay), []))
        for ax, ay, _results in per_tile_owned
    ]


def _ensure_nofile_limit(needed: int) -> None:
    """確保本行程的 open-file 軟上限足以讓縫合同時開著 ``needed`` 個 tile。

    ``_join_overlay_tiles`` 是**惰性**的：整張玻片的 27,565 個 overlay tile 會全部
    同時以 pyvips image 開著，一直到最後 ``tiffsave`` 拉資料時才真的逐一讀。本機的
    軟上限是 1,048,576，所以一路沒事；但 Linux 常見預設是 **1,024**，在那種機器上
    整片分析（數小時）跑完之後才會在最後一步炸掉——這是這條管線最貴的失敗方式，
    也正是 doc 25 §5.2 記下但沒補的缺口。

    故在開任何檔之前先檢查：軟上限不夠就自己往上提（soft→hard 一律允許，不需
    特權），硬上限也不夠才大聲失敗並講清楚要調多少。

    非 POSIX 平台沒有 per-process 的 RLIMIT_NOFILE（``resource`` 模組也不存在），
    沒有東西可檢查，直接跳過。
    """
    if resource is None:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY or soft >= needed:
        return
    if hard == resource.RLIM_INFINITY or hard >= needed:
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))
        logger.info(
            "RLIMIT_NOFILE 軟上限 %d 不足，已自行提高為 %d（縫合需同時開啟 %d 個 tile）",
            soft, needed, needed,
        )
        return
    raise RuntimeError(
        f"overlay 縫合需同時開啟約 {needed} 個檔案，但本行程的 RLIMIT_NOFILE "
        f"硬上限只有 {hard}（軟上限 {soft}）。請提高硬上限後重跑最後的縫合步驟"
        f"（例如 `ulimit -Hn {needed}`，或於 systemd unit 設 LimitNOFILE={needed}）"
        f"——整片的逐塊 overlay 已在 {_STITCH_SCRATCH}/ 落地，不需重跑。"
    )


def _join_overlay_tiles(output_dir: Path, geometry: TileGeometry) -> pyvips.Image:
    """把 ``_stitch_scratch/`` 內每格核心裁切的 tile 惰性拼成一張全片影像（尚未編碼）。

    每格 overlay 的尺寸依所在欄 / 列而異（邊界格較小），但**同欄同寬、同列同高**
    （由 ``core_crop_bounds`` 的建構保證）。``pyvips.Image.arrayjoin()`` 只做「等格montage」
    （以最大寬高為格、其餘留白），無法正確處理非均勻的每欄 / 每列尺寸（已用合成測試證實
    它會多出留白、尺寸錯誤）。故改為手動：每列內由左至右水平 join（同列高已相等），
    再把各列由上至下垂直 join（各列總寬已相等）——可逐像素還原原始版面。

    與 ``_stitch_overlay_slide`` 拆開，是為了讓 ``scripts/stitch_probe.py`` 能在**同一張
    已拼好的影像**上獨立消融 ``tiffsave`` 的各個參數（doc 26 Tier 1.1）：join 與 encode
    的成本必須分開讀，否則量到的差異分不清是哪一段造成的。
    """
    overlay_dir = output_dir / _STITCH_SCRATCH
    xs = sorted(geometry.col_of)  # abs_x，欄序（升冪即欄索引序）
    ys = sorted(geometry.row_of)  # abs_y，列序

    _ensure_nofile_limit(len(xs) * len(ys) + 256)

    row_images: List[pyvips.Image] = []
    for ay in ys:
        row_tiles: List[pyvips.Image] = []
        for ax in xs:
            path = overlay_dir / f"tile_x{ax}_y{ay}.tiff"
            if not path.exists():
                raise FileNotFoundError(
                    f"{_STITCH_SCRATCH} 缺少 tile: {path}——每格應恰有一檔。"
                )
            row_tiles.append(
                pyvips.Image.new_from_file(str(path), access="sequential")
            )
        row = row_tiles[0]
        for tile in row_tiles[1:]:
            row = row.join(tile, "horizontal", expand=True)
        row_images.append(row)

    slide = row_images[0]
    for row in row_images[1:]:
        slide = slide.join(row, "vertical", expand=True)
    return slide


def _stitch_overlay_slide(output_dir: Path, geometry: TileGeometry) -> None:
    """把 ``_stitch_scratch/`` 內的 tile 拼成一張全片 pyramid TIFF 並寫出，寫完刪暫存夾。

    兩個後端，由 ``config.stitch_backend`` 選（預設 ``"pyvips"`` = 今日出貨行為）：

    - ``"pyvips"`` — ``_join_overlay_tiles`` + 單次 ``tiffsave``。
    - ``"tifffile"`` — round 12/13 的 candidate B：逐 band 串流讀（讀搬到背景執行緒與
      編碼重疊）、CPU pyramid、per-tile LZW + Predictor 2、tifffile 組容器。
      量到 **1.365x**（doc 39 §4，4.055 GP）。

    兩條路徑產出的**版面**相同（同尺寸、同 128px tile、同 pyramid 層數），位元組則不同：
    candidate B 套 TIFF Predictor 2 而出貨版沒有，故檔案較小（4 GP 下 1.89 vs 2.44 GB）。
    這個差異已過人工關卡，見 ``config.stitch_backend`` 的註解。

    刪除暫存夾**不放 finally**（兩條路徑皆然）：拼合失敗時保留暫存夾，才能重跑縫合而
    不必重算整批。
    """
    backend = getattr(config, "stitch_backend", "pyvips")
    if backend == "pyvips":
        _stitch_overlay_slide_pyvips(output_dir, geometry)
    elif backend == "tifffile":
        _stitch_overlay_slide_tifffile(output_dir, geometry)
    else:
        raise ValueError(
            f"config.stitch_backend 只接受 'pyvips' 或 'tifffile'，實得 {backend!r}。"
        )


def _stitch_overlay_slide_pyvips(output_dir: Path, geometry: TileGeometry) -> None:
    """出貨後端：惰性 join 後單次 ``tiffsave``。

    壓縮採 **lzw（無失真）**：這是帶細胞邊界線 / 標籤文字 / 紅黑點的標註影像，JPEG
    的區塊假影會糊掉細線與小點，醫師判讀不宜；lzw 保真且仍可壓。

    ``tiffsave`` 的其餘參數（tile 大小、pyramid 深度、predictor）已在真實 16.22 GP
    規模上逐一消融過，沒有一個換得到值得留下的時間——見
    ``docs/hybrid-pipeline/27-remaining-work-implementation.md`` §3 與
    ``scripts/stitch_probe.py --ablate``。故此處維持 pyvips 預設，不加旋鈕：零貢獻的
    層一律不留（playbook step 4）。

    這條路徑**沒有** ``tifffile`` 後端那個「把讀搬到背景執行緒」的槓桿可用，而且不需要：
    ``tiffsave`` 把讀融進編碼裡，本來就免費重疊（doc 40 §1.2）。

    ``tiffsave`` 是同步的——回傳時惰性管線已把每格讀完寫出，故其後即可安全刪除暫存夾。
    """
    overlay_dir = output_dir / _STITCH_SCRATCH
    slide = _join_overlay_tiles(output_dir, geometry)
    slide.tiffsave(
        str(output_dir / "overlay_slide.tiff"),
        tile=True,
        pyramid=True,
        compression="lzw",
        bigtiff=True,
        # predictor="none" is a CORRECTNESS fix, not a tuning knob (round 9, doc 32 §5.1).
        # libvips 8.15.1's default (`horizontal`) writes tag 317 Predictor=2 on the full
        # -resolution IFD only, while still horizontally differencing the *reduced* pyramid
        # levels' data. Every reader tried — tifffile and libtiff via Pillow — therefore
        # decodes levels 1..N as noise (measured: 88–99% zero pixels), i.e. a pathologist
        # zooming out sees a black slide. Confirmed in QuPath on a real slide, then measured
        # directly on a real overlay_slide.tiff — not inferred from a synthetic one.
        #
        # Cost, measured on that same real overlay (18,688² L0): output grows
        # 93.35 → 102.08 MB (+9.4%), encode time 4.14 → 4.18 s (within noise). Paying ~9%
        # disk to make the zoomed-out view exist at all is the same trade doc 27 §3.1 made
        # when it vetoed `zstd`: an overlay a pathologist cannot read is worth zero.
        # **Do not "optimise" this back to horizontal without re-checking every pyramid
        # level** — `tests/test_stitch_pyramid_levels.py` will fail if you do.
        predictor="none",
    )
    logger.info(
        "overlay_slide.tiff 縫合完成: %d×%d px", slide.width, slide.height
    )
    if overlay_dir.exists():
        shutil.rmtree(overlay_dir)


# ------------------------------------------------------------------
# Phase D candidate B：band 串流 + 背景讀 + tifffile 容器（round 12/13）
# ------------------------------------------------------------------
# 為什麼是這個形狀：doc 35 §3.2 量到兩個 tifffile 候選都**比出貨的 pyvips 慢**
# （0.788x / 0.884x），因為它們序列地付「讀 + join」這半（佔 baseline 自己 48.7% 的
# wall），而 pyvips 把它融進 tiffsave 裡免費重疊掉了。doc 39 §4 把讀搬到背景執行緒後同
# 一批候選翻成 1.365x / 1.581x。所以這裡照抄的不是「tifffile 比較快」，而是「讀被藏起
# 來之後 tifffile 才比較快」——`_prefetch_bands` 是這條路徑存在的唯一理由。
#
# 全程只有「一個 band」等級的東西常駐（讀一條、編一條），與 `_join_overlay_tiles` 的
# 惰性讀同樣有界；這是 Phase D 能在 16 GP 玻片上跑得起來的前提，不要改成整片載入。

# 容器幾何必須與出貨版逐項相同，否則「比較快」可能只是「做得比較少」（doc 32 §3）：
# 以下三個都是 pyvips tiffsave 的預設值，也就是今天 overlay_slide.tiff 的實際規格。
_CONTAINER_TILE = 128    # tiffsave 預設 tile 邊長
_MIN_LEVEL = 128         # pyvips 一路對半縮到某層塞得進一個 tile 為止
_PYVIPS_DPI = 25.4       # pyvips 寫 1 px/mm；tifffile 預設不寫單位，對不上 QuPath 會
                         # 算出不同的實體像素大小、以不同比例顯示（doc 32 §5）
# per-tile LZW 的執行緒數。doc 39 §4 的 1.365x 就是在這個值下量到的
# （`stitch_probe.py --encode-workers` 的預設）；不另開 config 旋鈕：這條路徑目前唯一
# 有數字支撐的組態就是被量過的那一組。
_ENCODE_THREADS = 8


def _n_pyramid_levels(height: int, width: int) -> int:
    """pyvips 對這個尺寸會寫出的**縮圖**層數（不含 level 0）。

    規則是 ``max(h, w)`` 而非 ``min``：pyvips 一路對半縮，直到某層在**兩個**方向都塞得
    進一個 tile 為止。用 ``min`` 會在寬扁的玻片上早停兩層，而少寫兩層 pyramid 的候選只
    是做得比較少，速度不可比——doc 32 §3 第一次就踩到這個。
    """
    lv = 0
    while max(height, width) > _MIN_LEVEL:
        height, width, lv = height // 2, width // 2, lv + 1
    return lv


def _slide_dims(output_dir: Path, geometry: TileGeometry) -> Tuple[int, int]:
    """整片尺寸 ``(height, width)``，只讀第一列與第一欄的檔頭。

    同欄同寬、同列同高由 ``core_crop_bounds`` 的建構保證（見 ``_join_overlay_tiles``），
    所以整片寬 = 第一列各格寬之和、整片高 = 第一欄各格高之和。

    這裡刻意**不**走 ``_join_overlay_tiles``：那會為了問一個尺寸把全片 27,565 格同時開
    著（`scripts/stitch_probe.py` 的 spike 就是這樣做的，但它沒把這段計時）。逐欄逐列讀
    檔頭是 ``cols + rows`` 次開檔（整片約 332 次），且 pyvips 讀檔頭不解碼像素。
    """
    overlay_dir = output_dir / _STITCH_SCRATCH
    xs = sorted(geometry.col_of)
    ys = sorted(geometry.row_of)

    def _header(ax: int, ay: int) -> pyvips.Image:
        path = overlay_dir / f"tile_x{ax}_y{ay}.tiff"
        if not path.exists():
            raise FileNotFoundError(
                f"{_STITCH_SCRATCH} 缺少 tile: {path}——每格應恰有一檔。"
            )
        return pyvips.Image.new_from_file(str(path), access="sequential")

    width = sum(_header(ax, ys[0]).width for ax in xs)
    height = sum(_header(xs[0], ay).height for ay in ys)
    return height, width


def _band_source(output_dir: Path, geometry: TileGeometry) -> Iterator[np.ndarray]:
    """由上而下逐條吐出「整片寬」的水平 band，每列 tile 一條。

    水平 join 用的是 ``_join_overlay_tiles`` 的同一招（``arrayjoin`` 會對非均勻格線誤
    留白，見它的 docstring），差別只在這裡一次只實體化**一列**。
    """
    overlay_dir = output_dir / _STITCH_SCRATCH
    xs = sorted(geometry.col_of)
    for ay in sorted(geometry.row_of):
        row = None
        for ax in xs:
            path = overlay_dir / f"tile_x{ax}_y{ay}.tiff"
            if not path.exists():
                raise FileNotFoundError(
                    f"{_STITCH_SCRATCH} 缺少 tile: {path}——每格應恰有一檔。"
                )
            img = pyvips.Image.new_from_file(str(path), access="sequential")
            row = img if row is None else row.join(img, "horizontal", expand=True)
        buf = row.write_to_memory()
        yield np.ndarray(buffer=buf, dtype=np.uint8,
                         shape=(row.height, row.width, row.bands))


def _prefetch_bands(src: Iterator[np.ndarray], pool: ThreadPoolExecutor
                    ) -> Iterator[np.ndarray]:
    """在呼叫端編碼第 k 條時，於背景執行緒讀第 k+1 條。

    形狀與管線自己的 ``prefetch_tile_reads`` 相同：單執行緒、深度 1，所以最多兩條 band
    同時常駐（正在編的那條、正在讀的那條）。深度**刻意**維持 1：一條 band 在整片尺度是
    ~326 MB，而有界性正是 Phase D 跑得動 16 GP 玻片的前提（doc 21 §6 也在 tile 路徑上
    量到更深的 pipelining 是負收益）。

    只有一個執行緒會呼叫 ``next(src)``：下一次 fetch 是在前一個 future 解析**之後**才
    submit，故 generator 不會被同時重入。
    """
    fut = pool.submit(next, src, None)
    while True:
        band = fut.result()
        if band is None:
            return
        fut = pool.submit(next, src, None)
        yield band


def _shrink2_cpu(a: np.ndarray) -> np.ndarray:
    """2x2 box shrink；奇數的列 / 欄先截掉，與 pyvips 的作法一致。"""
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    return ((a[:h:2, :w:2].astype(np.uint16) + a[1:h:2, :w:2]
             + a[:h:2, 1:w:2] + a[1:h:2, 1:w:2] + 2) // 4).astype(np.uint8)


def _encode_tile_row(rows: np.ndarray, pool: ThreadPoolExecutor) -> List[bytes]:
    """把一整條 128px tile row 由左至右 LZW 編碼，**差分自己做**。

    tifffile 對「已壓縮的輸入」只會把 Predictor tag *宣告*上去，不會幫你轉換位元組。
    這件事做錯的產物是「開得起來但解碼成雜訊」——doc 32 §5.1 已經出貨過一次的無聲毀損。
    Predictor 2 是 ``out[0] = in[0]; out[k] = in[k] - in[k-1]``，所以第一欄保留絕對值
    （prepend 零，絕不是 prepend 第一欄自己）。
    """
    import imagecodecs

    h, w = rows.shape[:2]
    tiles = []
    for y in range(0, h, _CONTAINER_TILE):
        for x in range(0, w, _CONTAINER_TILE):
            t = rows[y:y + _CONTAINER_TILE, x:x + _CONTAINER_TILE]
            if t.shape[0] != _CONTAINER_TILE or t.shape[1] != _CONTAINER_TILE:
                pad = np.zeros((_CONTAINER_TILE, _CONTAINER_TILE, 3), dtype=rows.dtype)
                pad[:t.shape[0], :t.shape[1]] = t
                t = pad
            tiles.append(np.ascontiguousarray(t))

    def enc(t: np.ndarray) -> bytes:
        zero = np.zeros((t.shape[0], 1, t.shape[2]), dtype=t.dtype)
        d = np.diff(t, axis=1, prepend=zero)
        return imagecodecs.lzw_encode(np.ascontiguousarray(d).tobytes())

    return list(pool.map(enc, tiles))


def _stitch_overlay_slide_tifffile(output_dir: Path, geometry: TileGeometry) -> None:
    """candidate B 後端：band 串流讀（背景執行緒）+ CPU pyramid + tifffile 容器。

    每讀進一條 band 就餵給 ``feed(0, ...)``：湊滿 128 的整數倍高度就切出來編碼，同一塊
    順手 shrink 一次推進下一層的緩衝區，遞迴到最後一層。任何一層都不會有超過一條 band
    的資料常駐，這是與 ``_join_overlay_tiles`` 同等的有界性。
    """
    import tifffile

    overlay_dir = output_dir / _STITCH_SCRATCH
    xs = sorted(geometry.col_of)
    # 一次只開一列，不是整片——出貨路徑的 27,565 個 fd 這裡不需要。
    _ensure_nofile_limit(len(xs) + 256)

    height, width = _slide_dims(output_dir, geometry)
    n_lv = _n_pyramid_levels(height, width)

    buf: List[Optional[np.ndarray]] = []
    segs: List[List[bytes]] = []
    shapes: List[List[int]] = []

    def ensure(k: int) -> None:
        while len(buf) <= k:
            buf.append(None)
            segs.append([])
            shapes.append([0, 0])

    def emit(k: int, chunk: np.ndarray, pool: ThreadPoolExecutor) -> None:
        """編掉這層的一塊，並把它 shrink 一次餵進下一層。"""
        segs[k].extend(_encode_tile_row(chunk, pool))
        if k + 1 <= n_lv:
            feed(k + 1, _shrink2_cpu(chunk), pool)

    def feed(k: int, band: np.ndarray, pool: ThreadPoolExecutor) -> None:
        ensure(k)
        shapes[k][0] += band.shape[0]
        shapes[k][1] = band.shape[1]
        buf[k] = band if buf[k] is None else np.concatenate([buf[k], band], axis=0)
        n = (buf[k].shape[0] // _CONTAINER_TILE) * _CONTAINER_TILE
        if not n:
            return
        chunk, buf[k] = buf[k][:n], buf[k][n:]
        emit(k, chunk, pool)

    def flush(k: int, pool: ThreadPoolExecutor) -> None:
        """收尾：每層剩下不滿一個 tile row 的殘量也要寫出去（_encode_tile_row 會補零）。"""
        if k >= len(buf):
            return
        chunk, buf[k] = buf[k], None
        if chunk is not None and chunk.shape[0]:
            emit(k, chunk, pool)
        flush(k + 1, pool)

    with ThreadPoolExecutor(max_workers=_ENCODE_THREADS) as pool, \
            ThreadPoolExecutor(max_workers=1,
                               thread_name_prefix="stitch-band-read") as rpool:
        for band in _prefetch_bands(_band_source(output_dir, geometry), rpool):
            feed(0, band, pool)
        flush(0, pool)

    with tifffile.TiffWriter(str(output_dir / "overlay_slide.tiff"),
                             bigtiff=True) as tif:
        for i, (seg, (h, w)) in enumerate(zip(segs, shapes)):
            tif.write(
                iter(seg), shape=(h, w, 3), dtype=np.uint8,
                tile=(_CONTAINER_TILE, _CONTAINER_TILE),
                # predictor=True 只寫 tag；差分已由 _encode_tile_row 套用。出貨的 pyvips
                # 路徑則是 predictor="none"——兩者都正確，差別只在檔案大小，重點是**每層
                # IFD 的 tag 與資料一致**（doc 32 §5.1 壞掉的正是這個一致性）。
                compression="lzw", predictor=True, photometric="rgb",
                subfiletype=1 if i else 0,
                resolution=(_PYVIPS_DPI, _PYVIPS_DPI), resolutionunit="inch",
            )

    logger.info(
        "overlay_slide.tiff 縫合完成: %d×%d px（tifffile 後端，%d 層）",
        shapes[0][1], shapes[0][0], len(segs),
    )
    if overlay_dir.exists():
        shutil.rmtree(overlay_dir)
