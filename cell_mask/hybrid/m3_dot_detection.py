"""M3b: DISH 紅點 (CEP17) / 黑點 (HER2) 偵測模組。

演算法核心：
    1. 先以 elastic matching 把 DISH 核 instance 分配給 IHC 細胞
       （競爭消解最佳指派，每顆 DISH 核最多被一顆 IHC 認領）。
    2. 建立 effective_instance_mask：IHC strict 先寫入，matched DISH
       區只填還是 0 的像素，不搶其他細胞。
    3. 對每顆 cell 逐顆 crop 出 region，在局部 LAB patch 上做紅/黑點
       偵測 → 直接以 cell 為 ROI，不再做全圖偵測 + 最近鄰指派。
    4. 排除規則：核數 1=valid；>= ``dot_blue_exclude_threshold``=多核（排除）；
       核數 0 再分兩種——曾有可行候選卻在競爭中落敗=drop-out（排除、打 X）；
       從頭就沒有可行候選=忽略配對、照常計入。被排除者不計入分析。

像素級偵測核心（紅/黑點、環統計、合併）在 m3_dot_kernels.py；
DISH 核彈性匹配在 m3_elastic_matching.py。詳見 docs/sdd-elastic-dish-matching.md。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import find_objects

from cell_mask.hybrid.m3_cells_generator import CellAnalysisResult
from cell_mask.hybrid.m3_dot_kernels import (
    DetectedDot,
    _detect_black_dots,
    _detect_red_dots,
    _merge_close_dots,
    _rgb_to_lab,
)
from cell_mask.hybrid.m3_elastic_matching import elastic_dish_nucleus_matching

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 資料結構
# ------------------------------------------------------------------

@dataclass
class CellDotResult:
    """單一細胞的點位計數結果。"""

    cell_id: int
    her2_dot_count: int = 0
    cep17_dot_count: int = 0
    her2_cep17_ratio: float = 0.0        # float("inf") 當 cep17_dot_count == 0
    is_amplified: bool = False
    blue_region_count: int = 0
    excluded: bool = False               # drop-out(0 核) 或 多核(≥ threshold) → 排除
    exclude_reason: str = ""             # "" | "drop_out" | "multi_nucleus"（打 X 用）
    her2_dots: List[DetectedDot] = field(default_factory=list)
    cep17_dots: List[DetectedDot] = field(default_factory=list)
    # elastic matching 認領到的 DISH 核 ID（用於視覺化飄移箭頭與粉色輪廓）
    assigned_dish_ids: List[int] = field(default_factory=list)


# ------------------------------------------------------------------
# 主要入口
# ------------------------------------------------------------------

def detect_all_dots(
    dish_image: np.ndarray,
    instance_mask: np.ndarray,
    config: object,
    dish_nucleus_mask: np.ndarray,
    dish_ids_by_cell: Optional[Dict[int, List[int]]] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[List[DetectedDot], Dict[int, CellDotResult]]:
    """逐顆 matched cell 偵測 HER2 黑點與 CEP17 紅點。

    Args:
        dish_image: (H, W, 3) uint8 RGB，白背景(255)，已套用 core_mask。
        instance_mask: (H, W) 整數，0=背景，1..N=IHC 細胞 ID（strict 分割）。
        config: 具有 ``dot_*`` / ``dish_elastic_*`` 欄位的配置物件。
        dish_nucleus_mask: (H, W) int，0=背景，1..M=DISH 細胞核 ID。
        dish_ids_by_cell: 可選，預先算好的 elastic matching 結果
            ``{ihc_cell_id: [dish_id, ...]}``。未提供時內部會自行呼叫。
        n_jobs: per-cell 偵測的平行度（joblib）。None=用滿所有核心 (-1)，
            1=序列，其他正整數=指定行程數。

    Returns:
        (all_dots, per_cell_results)
    """
    if dish_image.ndim != 3 or dish_image.shape[2] != 3:
        raise ValueError(f"dish_image 必須為 (H,W,3) RGB，實際 {dish_image.shape}")
    if dish_image.shape[:2] != instance_mask.shape:
        raise ValueError(
            f"shape 不一致: dish={dish_image.shape[:2]} vs mask={instance_mask.shape}"
        )
    if dish_nucleus_mask is None:
        raise ValueError("dish_nucleus_mask 為必填參數，未提供。")
    if dish_nucleus_mask.shape != instance_mask.shape:
        raise ValueError(
            "shape 不一致: "
            f"dish_nucleus_mask={dish_nucleus_mask.shape} vs mask={instance_mask.shape}"
        )

    instance_mask_i32 = instance_mask.astype(np.int32, copy=False)
    dish_nucleus_mask_i32 = dish_nucleus_mask.astype(np.int32, copy=False)

    if dish_ids_by_cell is None:
        dish_ids_by_cell, drop_out_ids = elastic_dish_nucleus_matching(
            dish_nucleus_mask=dish_nucleus_mask_i32,
            strict_instance_mask=instance_mask_i32,
            cfg=config,
        )
    else:
        # 預先算好的指派沒有「可行候選」資訊，無從判定 drop-out → 一律不排除。
        drop_out_ids = set()

    effective_mask = _build_effective_instance_mask(
        strict_instance_mask=instance_mask_i32,
        dish_nucleus_mask=dish_nucleus_mask_i32,
        dish_ids_by_cell=dish_ids_by_cell,
    )

    L, a, b = _rgb_to_lab(dish_image)
    bg_threshold = float(getattr(config, "dot_background_l_threshold", 95.0))
    bg_mask_global = L > bg_threshold

    default_merge_distance = float(getattr(config, "dot_merge_distance", 3.0))
    red_merge_distance = float(
        getattr(config, "dot_red_merge_distance", default_merge_distance)
    )
    black_merge_distance = float(
        getattr(config, "dot_black_merge_distance", default_merge_distance)
    )

    all_dots: List[DetectedDot] = []
    per_cell: Dict[int, CellDotResult] = {}

    # find_objects 一次取得每個 label 的 bbox，取代逐顆對整張影像掃 (== cid)。
    # slices[k] 對應 label k+1 的 (slice_y, slice_x)；label 不存在則為 None。
    slices = find_objects(effective_mask)
    cell_ids = sorted({int(v) for v in np.unique(effective_mask) if v != 0})
    tasks = [
        (cid, slices[cid - 1])
        for cid in cell_ids
        if cid - 1 < len(slices) and slices[cid - 1] is not None
    ]

    # 每顆 cell 完全獨立（只讀傳入陣列的 bbox 切片），多核平行。
    # joblib 對大陣列自動 memmap，每個只 dump 一次供所有 worker 共享唯讀。
    n_jobs_eff = -1 if n_jobs is None else n_jobs
    cell_results = Parallel(n_jobs=n_jobs_eff)(
        delayed(_detect_one_cell)(
            cid, sl, effective_mask, L, a, b, bg_mask_global,
            config, red_merge_distance, black_merge_distance,
        )
        for cid, sl in tasks
    )

    for cid, red_dots, black_dots in cell_results:
        cdr = CellDotResult(cell_id=cid)
        cdr.cep17_dots = red_dots
        cdr.her2_dots = black_dots
        per_cell[cid] = cdr
        all_dots.extend(red_dots)
        all_dots.extend(black_dots)

    _finalize_per_cell(per_cell, dish_ids_by_cell, drop_out_ids, config)

    n_red = sum(1 for d in all_dots if d.dot_type == "cep17")
    n_black = sum(1 for d in all_dots if d.dot_type == "her2")
    logger.info(
        "detect_all_dots: 紅點=%d, 黑點=%d, 涉及細胞=%d",
        n_red, n_black, len(per_cell),
    )
    return all_dots, per_cell


def _detect_one_cell(
    cid: int,
    sl: Tuple[slice, slice],
    effective_mask: np.ndarray,
    L: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    bg_mask_global: np.ndarray,
    cfg: object,
    red_merge_distance: float,
    black_merge_distance: float,
) -> Tuple[int, List[DetectedDot], List[DetectedDot]]:
    """在單顆 cell 的 bbox patch 上偵測紅/黑點，回傳全 tile 座標的點。

    供 joblib 平行呼叫：只讀傳入陣列在 ``sl`` 的切片，無共用寫入，
    因此各 cell 之間完全獨立。

    Returns:
        ``(cell_id, red_dots, black_dots)``；無 ROI 時兩串列皆空。
    """
    y0 = sl[0].start
    x0 = sl[1].start

    region_local = effective_mask[sl] == cid
    bg_local = bg_mask_global[sl]
    cell_roi_local = region_local & (~bg_local)
    if not cell_roi_local.any():
        return cid, [], []

    L_local = L[sl]
    a_local = a[sl]
    b_local = b[sl]

    red_dots = _detect_red_dots(
        a=a_local,
        cell_roi=cell_roi_local,
        bg_mask=bg_local,
        cfg=cfg,
        cell_id=cid,
    )
    black_dots = _detect_black_dots(
        L=L_local,
        a=a_local,
        b=b_local,
        cell_roi=cell_roi_local,
        bg_mask=bg_local,
        cfg=cfg,
        cell_id=cid,
    )
    red_dots = _merge_close_dots(red_dots, red_merge_distance)
    black_dots = _merge_close_dots(black_dots, black_merge_distance)

    # 局部 patch 座標 → 全 tile 座標
    for d in red_dots:
        d.y += y0
        d.x += x0
    for d in black_dots:
        d.y += y0
        d.x += x0
    return cid, red_dots, black_dots


# ------------------------------------------------------------------
# Effective instance mask（IHC strict + matched DISH 補位）
# ------------------------------------------------------------------

def _build_effective_instance_mask(
    strict_instance_mask: np.ndarray,
    dish_nucleus_mask: np.ndarray,
    dish_ids_by_cell: Dict[int, List[int]],
) -> np.ndarray:
    """Hard-assign 每個像素到一顆 IHC 細胞。

    優先序：
        Pass 1: IHC strict mask 先寫入（嚴格分割優先）。
        Pass 2: matched DISH 核區只填還是 0 的像素，等於把 cell footprint
                往 DISH 核位置擴張，但絕不覆蓋其他細胞的 strict 區。
    """
    out = strict_instance_mask.astype(np.int32, copy=True)
    if dish_nucleus_mask.size == 0:
        return out

    dish_to_ihc: Dict[int, int] = {}
    for cid, dish_ids in dish_ids_by_cell.items():
        for did in dish_ids:
            dish_to_ihc[int(did)] = int(cid)
    if not dish_to_ihc:
        return out

    max_dish = int(dish_nucleus_mask.max())
    if max_dish <= 0:
        return out

    lookup = np.zeros(max_dish + 1, dtype=np.int32)
    for did, cid in dish_to_ihc.items():
        if 0 < did <= max_dish:
            lookup[did] = cid
    extended = lookup[dish_nucleus_mask]
    fill_mask = (out == 0) & (extended > 0)
    out[fill_mask] = extended[fill_mask]
    return out


# ------------------------------------------------------------------
# 細胞層級統計 + 擴增判定
# ------------------------------------------------------------------

def _finalize_per_cell(
    per_cell: Dict[int, CellDotResult],
    dish_ids_by_cell: Dict[int, List[int]],
    drop_out_ids: Set[int],
    cfg: object,
) -> None:
    """填入計數、ratio、藍區數量、excluded、is_amplified（in-place）。"""
    amp_ratio = float(getattr(cfg, "dot_amplification_ratio", 2.0))
    amp_count = int(getattr(cfg, "dot_her2_count_threshold", 6))
    exclude_thr = int(getattr(cfg, "dot_blue_exclude_threshold", 2))
    exclude_zero = bool(getattr(cfg, "dish_elastic_exclude_zero", False))

    for cdr in per_cell.values():
        cdr.her2_dot_count = len(cdr.her2_dots)
        cdr.cep17_dot_count = len(cdr.cep17_dots)
        if cdr.cep17_dot_count > 0:
            cdr.her2_cep17_ratio = cdr.her2_dot_count / cdr.cep17_dot_count
        else:
            cdr.her2_cep17_ratio = float("inf") if cdr.her2_dot_count > 0 else 0.0

        assigned_ids = dish_ids_by_cell.get(cdr.cell_id, [])
        cdr.assigned_dish_ids = list(assigned_ids)
        cdr.blue_region_count = len(cdr.assigned_dish_ids)
        # 核數分類：>=threshold=多核（排除）；==1=valid；==0 再分兩種——
        # 曾有可行候選卻競爭落敗（cell_id in drop_out_ids）=drop-out（排除、打 X）；
        # 從頭就沒有可行候選=忽略配對、照常計入（不排除）。
        if cdr.blue_region_count >= exclude_thr:
            cdr.excluded = True
            cdr.exclude_reason = "multi_nucleus"
        elif (
            cdr.blue_region_count == 0
            and exclude_zero
            and cdr.cell_id in drop_out_ids
        ):
            cdr.excluded = True
            cdr.exclude_reason = "drop_out"
        else:
            cdr.excluded = False
            cdr.exclude_reason = ""
        if cdr.excluded:
            cdr.is_amplified = False
        else:
            cdr.is_amplified = (
                cdr.her2_dot_count >= amp_count
                or (cdr.cep17_dot_count > 0 and cdr.her2_cep17_ratio >= amp_ratio)
            )


# ------------------------------------------------------------------
# 合併至 CellAnalysisResult
# ------------------------------------------------------------------

def merge_dot_results_to_cell_analysis(
    cell_results: List[CellAnalysisResult],
    per_cell_results: Dict[int, CellDotResult],
) -> List[CellAnalysisResult]:
    """將 CellDotResult 的 4 個統計欄位合併回 CellAnalysisResult。

    若 cell_results 中的項目不存在於 per_cell_results，保留預設值 0 / 0.0 / False。
    """
    for res in cell_results:
        cdr = per_cell_results.get(res.cell_id)
        if cdr is None:
            continue
        res.her2_dot_count = cdr.her2_dot_count
        res.cep17_dot_count = cdr.cep17_dot_count
        res.her2_cep17_ratio = cdr.her2_cep17_ratio
        res.is_amplified = cdr.is_amplified
        res.blue_region_count = cdr.blue_region_count
        res.excluded = cdr.excluded
    return cell_results
