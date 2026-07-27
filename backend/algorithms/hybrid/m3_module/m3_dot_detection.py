"""M3b: DISH 紅點 (CEP17) / 黑點 (HER2) 偵測模組。

演算法核心（以細胞為中心）：
    1. 先以 elastic matching 讓每顆 IHC 細胞（綠框面積放大 factor 倍）認領
       「離它質心最近的 DISH 核」（一對一、最近優先、lock；每顆細胞至多 1 核）。
    2. 建立 nucleus_owner_mask：每個 matched DISH 核 pixel 標上擁有者 IHC
       細胞 id，其餘為 0。這就是紅黑點的計數 ROI。
    3. 對每顆認到核的 cell，在其「配對核區域」的局部 LAB patch 上做紅/黑點
       偵測——整顆核的點全記給擁有者，不沿 IHC 領地邊界把跨界核的點切給鄰居；
       沒配到核的細胞計 0 點。
    4. 排除規則：認到 1 核=valid（合格核優先）；0 核再分三種——曾有候選核卻
       競爭落敗=drop-out（排除、打 X）；與「出界核」（碰到 UNet++ mask 外、已被
       core_mask 過濾掉者）重疊=污染細胞（排除、打 X）；其餘從頭就沒有候選
       =照常計入（顯示 0/0）。被排除者不計入分析。一對一配對下每顆細胞至多
       1 核，故不再有「多核排除」。

像素級偵測核心（紅/黑點、環統計、合併）在 m3_dot_kernels.py；
DISH 核彈性匹配在 m3_elastic_matching.py（該檔的 docstring 即該演算法的權威說明）。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import find_objects

try:
    from ..hybrid_data_types import CellAnalysisResult, CellDotResult, DetectedDot
except ImportError:
    from hybrid_data_types import CellAnalysisResult, CellDotResult, DetectedDot
from .m3_dot_kernels import (
    _detect_black_dots,
    _detect_red_dots,
    _merge_close_dots,
    _rgb_to_lab,
)
from .m3_elastic_matching import elastic_dish_nucleus_matching

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# DISH 核 core_mask 過濾
# ------------------------------------------------------------------

def _filter_dish_nucleus_by_core_mask(
    dish_nucleus_mask: np.ndarray,
    core_mask: np.ndarray,
    min_inside_ratio: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """移除「未完全落在 IHC core_mask 內」的 DISH 核 instance。

    Returns:
        ``(kept_mask, out_of_bounds_mask)``——前者把出界核設為 0；後者只保留
        被丟棄的「出界核」原始 label，供下游判定「壓在邊界、對應到出界核」
        的 IHC 細胞並打 X。
    """
    if dish_nucleus_mask.size == 0:
        return dish_nucleus_mask, np.zeros_like(dish_nucleus_mask)
    mask_i32 = dish_nucleus_mask.astype(np.int32, copy=False)
    max_id = int(mask_i32.max())
    if max_id <= 0:
        return mask_i32, np.zeros_like(mask_i32)
    core_bool = core_mask.astype(bool, copy=False)
    flat = mask_i32.ravel()
    total = np.bincount(flat, minlength=max_id + 1)
    inside = np.bincount(
        flat,
        weights=core_bool.ravel().astype(np.int64),
        minlength=max_id + 1,
    ).astype(np.int64)
    outside = total - inside
    if min_inside_ratio >= 1.0:
        drop = outside > 0
    else:
        drop = inside < (min_inside_ratio * total)
    drop[0] = False
    out_of_bounds_mask = (
        np.where(drop[mask_i32], mask_i32, 0).astype(np.int32)
        if drop.any()
        else np.zeros_like(mask_i32)
    )
    if drop.any():
        remap = np.arange(max_id + 1, dtype=np.int32)
        remap[drop] = 0
        mask_i32 = remap[mask_i32]
    return mask_i32, out_of_bounds_mask


# ------------------------------------------------------------------
# 主要入口
# ------------------------------------------------------------------

def detect_all_dots(
    dish_image: np.ndarray,
    instance_mask: np.ndarray,
    config: object,
    dish_nucleus_mask: np.ndarray,
    core_mask: np.ndarray,
    n_jobs: Optional[int] = None,
) -> Tuple[List[DetectedDot], Dict[int, CellDotResult], np.ndarray]:
    """逐顆 matched cell 偵測 HER2 黑點與 CEP17 紅點。

    Args:
        dish_image: (H, W, 3) uint8 RGB，白背景(255)，已套用 core_mask。
        instance_mask: (H, W) 整數，0=背景，1..N=IHC 細胞 ID（strict 分割）。
        config: 具有 ``dot_*`` / ``dish_elastic_*`` 欄位的配置物件。
        dish_nucleus_mask: (H, W) int，0=背景，1..M=DISH 細胞核 ID（原始，未過濾）。
        core_mask: (H, W) uint8{0,1}，IHC UNet++ 核心遮罩；用於過濾出界 DISH 核。
        n_jobs: per-cell 偵測的平行度（joblib）。None=用滿所有核心 (-1)，
            1=序列，其他正整數=指定行程數。

    Returns:
        (all_dots, per_cell_results, filtered_dish_nucleus_mask)
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

    min_inside_ratio = float(
        getattr(config, "dish_nucleus_core_min_inside_ratio", 1.0)
    )
    dish_nucleus_mask, out_of_bounds_nucleus_mask = _filter_dish_nucleus_by_core_mask(
        dish_nucleus_mask, core_mask, min_inside_ratio=min_inside_ratio
    )

    instance_mask_i32 = instance_mask.astype(np.int32, copy=False)
    dish_nucleus_mask_i32 = dish_nucleus_mask.astype(np.int32, copy=False)

    dish_ids_by_cell, drop_out_ids = elastic_dish_nucleus_matching(
        dish_nucleus_mask=dish_nucleus_mask_i32,
        strict_instance_mask=instance_mask_i32,
        cfg=config,
    )

    # 與「出界核」有像素重疊的 IHC 細胞：這些細胞壓在 UNet++ mask 邊界上。
    # 若最終仍沒配到任何合格核，視為污染細胞打 X（合格核優先：有配到合格核
    # 代表已飄移到核上、不在邊界，保留不打 X——於 _finalize_per_cell 判定）。
    oob_i32 = out_of_bounds_nucleus_mask.astype(np.int32, copy=False)
    both_oob = (instance_mask_i32 > 0) & (oob_i32 > 0)
    oob_overlap_cells: Set[int] = (
        {int(v) for v in np.unique(instance_mask_i32[both_oob])}
        if both_oob.any()
        else set()
    )

    # 紅黑點只在「配對到的 DISH 核區域」內計算——整顆核的點都記給贏得它的細胞，
    # 不再沿 IHC strict 領地邊界把跨界核的點切給鄰居；沒配到核的細胞不計任何點。
    count_mask = _build_nucleus_owner_mask(
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

    # 偵測 ROI = 配對核區域；slices[k] 對應 owner cell k+1 的核 bbox（無則 None）。
    slices = find_objects(count_mask)
    all_ihc_ids = sorted({int(v) for v in np.unique(instance_mask_i32) if v != 0})
    tasks = [
        (cid, slices[cid - 1])
        for cid in all_ihc_ids
        if cid - 1 < len(slices) and slices[cid - 1] is not None
    ]

    # 每顆 cell 完全獨立（只讀傳入陣列的 bbox 切片），多核平行。
    # joblib 對大陣列自動 memmap，每個只 dump 一次供所有 worker 共享唯讀。
    n_jobs_eff = -1 if n_jobs is None else n_jobs
    cell_results = Parallel(n_jobs=n_jobs_eff, prefer='threads')(
        delayed(_detect_one_cell)(
            cid, sl, count_mask, L, a, b, bg_mask_global,
            config, red_merge_distance, black_merge_distance,
        )
        for cid, sl in tasks
    )

    # 先給所有 IHC 細胞建空結果（沒配到核者維持 0/0、照常計入），再填有偵測到的。
    for cid in all_ihc_ids:
        per_cell[cid] = CellDotResult(cell_id=cid)
    for cid, red_dots, black_dots in cell_results:
        cdr = per_cell[cid]
        cdr.cep17_dots = red_dots
        cdr.her2_dots = black_dots
        all_dots.extend(red_dots)
        all_dots.extend(black_dots)

    _finalize_per_cell(per_cell, dish_ids_by_cell, drop_out_ids, oob_overlap_cells, config)

    n_red = sum(1 for d in all_dots if d.dot_type == "cep17")
    n_black = sum(1 for d in all_dots if d.dot_type == "her2")
    logger.info(
        "detect_all_dots: 紅點=%d, 黑點=%d, 涉及細胞=%d",
        n_red, n_black, len(per_cell),
    )
    return all_dots, per_cell, dish_nucleus_mask


def _detect_one_cell(
    cid: int,
    sl: Tuple[slice, slice],
    count_mask: np.ndarray,
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

    region_local = count_mask[sl] == cid  # 該細胞贏得的 DISH 核區域
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
# Nucleus owner mask（紅黑點計數 ROI：matched DISH 核 → 擁有者 IHC 細胞）
# ------------------------------------------------------------------

def _build_nucleus_owner_mask(
    dish_nucleus_mask: np.ndarray,
    dish_ids_by_cell: Dict[int, List[int]],
) -> np.ndarray:
    """把每個 matched DISH 核 pixel 標上其擁有者 IHC 細胞 id；其餘為 0。

    紅黑點只在這個遮罩內計算：整顆核的點都記給贏得它的細胞，不沿 IHC strict
    領地邊界把跨界核的點切給鄰居（那會讓鄰居偷算到別人核裡的點）。沒被任何
    核配到的細胞不出現在此遮罩中 → 計 0 點。
    """
    out = np.zeros(dish_nucleus_mask.shape, dtype=np.int32)
    if dish_nucleus_mask.size == 0:
        return out
    max_dish = int(dish_nucleus_mask.max())
    if max_dish <= 0:
        return out

    lookup = np.zeros(max_dish + 1, dtype=np.int32)
    for cid, dish_ids in dish_ids_by_cell.items():
        for did in dish_ids:
            did = int(did)
            if 0 < did <= max_dish:
                lookup[did] = int(cid)
    return lookup[dish_nucleus_mask]


# ------------------------------------------------------------------
# 細胞層級統計 + 擴增判定
# ------------------------------------------------------------------

def _finalize_per_cell(
    per_cell: Dict[int, CellDotResult],
    dish_ids_by_cell: Dict[int, List[int]],
    drop_out_ids: Set[int],
    oob_overlap_cells: Set[int],
    cfg: object,
) -> None:
    """填入計數、ratio、score、藍區數量、excluded、is_amplified（in-place）。

    Score(r,b)：r=HER2 黑點、b=CEP17 紅點。b < score_cep17_min_count（預設 2）且
    「有訊號」（非 0/0）→ 紅點不足、無法計算 Score，直接排除打 X（low_cep17）；
    完全無訊號的 0/0 細胞不打 X、照常計入（顯示 0/0）。否則 score=r/b，
    且 score < amp_ratio 時歸 0。is_amplified = score > 0（即 score ≥ amp_ratio）。
    """
    amp_ratio = float(getattr(cfg, "dot_amplification_ratio", 2.0))
    cep17_min = int(getattr(cfg, "score_cep17_min_count", 2))
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
        # 排除（打 X）判定，依序：
        #   1. 0 核且曾有候選卻競爭落敗（drop_out_ids）= drop-out；
        #   2. 0 核且壓在邊界、與出界核重疊（oob_overlap_cells）= 污染細胞；
        #   3. 紅點不足（cep17 < cep17_min 且非 0/0 完全無訊號）= 無法計算 Score，直接打 X。
        # 其餘（含 0/0 完全無訊號）照常計入。
        if (
            cdr.blue_region_count == 0
            and exclude_zero
            and cdr.cell_id in drop_out_ids
        ):
            cdr.excluded = True
        elif (
            cdr.blue_region_count == 0
            and cdr.cell_id in oob_overlap_cells
        ):
            cdr.excluded = True
        elif cdr.cep17_dot_count < cep17_min and not (
            cdr.her2_dot_count == 0 and cdr.cep17_dot_count == 0
        ):
            cdr.excluded = True
        else:
            cdr.excluded = False

        if cdr.excluded:
            cdr.score = 0.0
            cdr.is_amplified = False
        else:
            # 此分支為 cep17 ≥ cep17_min 或 0/0 細胞，ratio 必為有限（0紅有黑的 inf 已排除）。
            cdr.score = (
                cdr.her2_cep17_ratio
                if cdr.her2_cep17_ratio >= amp_ratio
                else 0.0
            )
            cdr.is_amplified = cdr.score > 0.0


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
        res.score = cdr.score
        res.blue_region_count = cdr.blue_region_count
        res.excluded = cdr.excluded
    return cell_results
