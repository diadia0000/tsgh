"""
M0 縫合層：把逐塊 (per-chunk) 的 M1–M3 結果縫回單一 slide-level 輸出。

純資料重組，不碰模型——可用合成資料單元測試。

核心規則
--------
1. **質心 core-ownership 去重**（交接 §5.4）：把每個分塊沿 ``overlap/2`` 切出互不重疊、
   鋪滿全圖的「核心區」；一顆細胞只算在「其質心落在哪個分塊核心區」的那一塊。重疊帶
   的重複偵測因此自動消除——不需再跑 IoMin。
2. **全域重編號**：被認領的細胞依序給 1..N 全域 ID，畫進全圖 instance mask；DISH 核同理
   給 1..M 全域 ID，``assigned_dish_ids`` 一併改寫，確保 M4 的 per-cell 裁切 / 飄移箭頭 /
   粉色輪廓三者 ID 對得上。
3. **座標絕對化**：質心與每個 dot 的 ``(x, y)`` 平移 ``+(abs_x, abs_y)``。
4. **M1 影像縫合**：core_mask / masked_ihc / dish_mask_overlay / overlay 每塊只取其核心區
   貼進全圖（互不重疊、不重複塗），重現整圖 artifact 供 M4 沿用既有匯出。

單塊（輸入 ≤ tile_size）時核心區 = 整塊、無接縫、全域 ID == 局部 ID，本層退化為「原樣複製」，
配合 M1/M2/M3 不變 → 與現行整圖路徑逐位元一致（回歸基準）。
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import center_of_mass

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
    overlay_image: np.ndarray        # (th, tw, 3) uint8
    results: List[CellAnalysisResult]
    all_dots: List[DetectedDot]
    per_cell_dots: Dict[int, CellDotResult]


@dataclass
class StitchedTile:
    """縫合後的 slide-level 整圖輸出，欄位對齊 M4 既有匯出介面。"""

    instance_mask: np.ndarray
    dish_nucleus_mask: np.ndarray
    core_mask: np.ndarray
    masked_ihc: np.ndarray
    dish_mask_overlay: np.ndarray
    overlay_image: np.ndarray
    results: List[CellAnalysisResult] = field(default_factory=list)
    all_dots: List[DetectedDot] = field(default_factory=list)
    per_cell_dots: Dict[int, CellDotResult] = field(default_factory=dict)


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
# 縫合
# ------------------------------------------------------------------

def _cut_lines(starts: List[int], overlap: int) -> List[int]:
    """相鄰分塊核心區的分界線：在後一塊起點再進 ``overlap//2`` 處切。"""
    return [starts[i] + overlap // 2 for i in range(1, len(starts))]


def stitch_chunks(
    chunks: List[ChunkResult],
    full_h: int,
    full_w: int,
    overlap: int,
    background_fill_value: int = 255,
) -> StitchedTile:
    """把所有 ``ChunkResult`` 縫成單一 ``StitchedTile``。"""
    x_starts = sorted({c.abs_x for c in chunks})
    y_starts = sorted({c.abs_y for c in chunks})
    cuts_x = _cut_lines(x_starts, overlap)
    cuts_y = _cut_lines(y_starts, overlap)
    col_of = {x0: i for i, x0 in enumerate(x_starts)}
    row_of = {y0: i for i, y0 in enumerate(y_starts)}

    fill = background_fill_value
    instance_mask = np.zeros((full_h, full_w), np.int32)
    dish_nucleus_mask = np.zeros((full_h, full_w), np.int32)
    core_mask = np.zeros((full_h, full_w), np.uint8)
    masked_ihc = np.full((full_h, full_w, 3), fill, np.uint8)
    dish_mask_overlay = np.full((full_h, full_w, 3), fill, np.uint8)
    overlay_image = np.full((full_h, full_w, 3), fill, np.uint8)

    g_results: List[CellAnalysisResult] = []
    g_all_dots: List[DetectedDot] = []
    g_per_cell: Dict[int, CellDotResult] = {}
    g_cid = 0
    g_nid = 0

    for c in chunks:
        th, tw = c.instance_mask.shape
        x0, y0 = c.abs_x, c.abs_y
        col, row = col_of[x0], row_of[y0]

        # --- 此塊核心區（全圖座標 → 局部）---
        gx_lo = cuts_x[col - 1] if col > 0 else 0
        gx_hi = cuts_x[col] if col < len(cuts_x) else full_w
        gy_lo = cuts_y[row - 1] if row > 0 else 0
        gy_hi = cuts_y[row] if row < len(cuts_y) else full_h
        lx0, lx1 = max(0, gx_lo - x0), min(tw, gx_hi - x0)
        ly0, ly1 = max(0, gy_lo - y0), min(th, gy_hi - y0)

        # --- M1 影像：只貼核心區（互不重疊）---
        gy0, gx0 = y0 + ly0, x0 + lx0
        core_mask[gy0:y0 + ly1, gx0:x0 + lx1] = c.core_mask[ly0:ly1, lx0:lx1]
        masked_ihc[gy0:y0 + ly1, gx0:x0 + lx1] = c.masked_ihc[ly0:ly1, lx0:lx1]
        dish_mask_overlay[gy0:y0 + ly1, gx0:x0 + lx1] = c.dish_mask_overlay[ly0:ly1, lx0:lx1]
        overlay_image[gy0:y0 + ly1, gx0:x0 + lx1] = c.overlay_image[ly0:ly1, lx0:lx1]

        inst_sub = instance_mask[y0:y0 + th, x0:x0 + tw]
        nuc_sub = dish_nucleus_mask[y0:y0 + th, x0:x0 + tw]
        nuc_map: Dict[int, int] = {}

        def _assign_nucleus(local_did: int) -> int:
            nonlocal g_nid
            if local_did in nuc_map:
                return nuc_map[local_did]
            g_nid += 1
            nuc_map[local_did] = g_nid
            paint = (nuc_sub == 0) & (c.dish_nucleus_mask == local_did)
            nuc_sub[paint] = g_nid
            return g_nid

        # --- 被本塊認領的細胞（質心落在核心區）---
        for r in c.results:
            gxc, gyc = x0 + r.centroid_x, y0 + r.centroid_y
            if bisect_right(cuts_x, gxc) != col or bisect_right(cuts_y, gyc) != row:
                continue  # 質心不在本塊核心區 → 留給擁有它的鄰塊

            g_cid += 1
            new_id = g_cid
            paint = (inst_sub == 0) & (c.instance_mask == r.cell_id)
            inst_sub[paint] = new_id
            g_results.append(replace(r, cell_id=new_id, centroid_x=gxc, centroid_y=gyc))

            cdr = c.per_cell_dots.get(r.cell_id)
            if cdr is None:
                g_per_cell[new_id] = CellDotResult(cell_id=new_id)
                continue
            new_assigned = [_assign_nucleus(int(did)) for did in cdr.assigned_dish_ids]
            new_her2 = [replace(d, x=d.x + x0, y=d.y + y0, cell_id=new_id) for d in cdr.her2_dots]
            new_cep17 = [replace(d, x=d.x + x0, y=d.y + y0, cell_id=new_id) for d in cdr.cep17_dots]
            g_per_cell[new_id] = replace(
                cdr, cell_id=new_id, assigned_dish_ids=new_assigned,
                her2_dots=new_her2, cep17_dots=new_cep17,
            )
            g_all_dots.extend(new_her2)
            g_all_dots.extend(new_cep17)

        # --- 未配對但質心落在核心區的 DISH 核：補畫（overlay 橘色輪廓相容）---
        _paint_unmatched_core_nuclei(
            c.dish_nucleus_mask, nuc_map, _assign_nucleus,
            lx0, lx1, ly0, ly1,
        )

    return StitchedTile(
        instance_mask=instance_mask,
        dish_nucleus_mask=dish_nucleus_mask,
        core_mask=core_mask,
        masked_ihc=masked_ihc,
        dish_mask_overlay=dish_mask_overlay,
        overlay_image=overlay_image,
        results=g_results,
        all_dots=g_all_dots,
        per_cell_dots=g_per_cell,
    )


def _paint_unmatched_core_nuclei(
    nucleus_mask: np.ndarray,
    nuc_map: Dict[int, int],
    assign_fn,
    lx0: int,
    lx1: int,
    ly0: int,
    ly1: int,
) -> None:
    """把「質心落在核心區、尚未配給任何細胞」的 DISH 核也畫進全圖（僅供 overlay 輪廓）。"""
    ids = [int(v) for v in np.unique(nucleus_mask) if v != 0 and int(v) not in nuc_map]
    if not ids:
        return
    centroids = center_of_mass(np.ones(nucleus_mask.shape, np.uint8), labels=nucleus_mask, index=ids)
    for did, (cy, cx) in zip(ids, centroids):
        if np.isnan(cy) or np.isnan(cx):
            continue
        if ly0 <= cy < ly1 and lx0 <= cx < lx1:
            assign_fn(did)
