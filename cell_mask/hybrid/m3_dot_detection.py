"""M3b: DISH 紅點 (CEP17) / 黑點 (HER2) 偵測模組。

演算法核心：
    1. 先以 elastic matching 把 DISH 核 instance 分配給 IHC 細胞
       （greedy exclusive，每顆 DISH 核最多被一顆 IHC 認領）。
    2. 建立 effective_instance_mask：IHC strict 先寫入，matched DISH
       區只填還是 0 的像素，不搶其他細胞。
    3. 對每顆 cell 逐顆 crop 出 region，在局部 LAB patch 上做紅/黑點
       偵測 → 直接以 cell 為 ROI，不再做全圖偵測 + 最近鄰指派。
    4. 多核排除：matched DISH 核數 ≥ ``dot_blue_exclude_threshold`` → 排除。

詳見 docs/sdd-elastic-dish-matching.md。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import center_of_mass, find_objects
from scipy.spatial import cKDTree
from skimage.color import rgb2lab
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_dilation,
    disk,
    h_maxima,
    h_minima,
)

from cell_mask.hybrid.m3_cells_generator import CellAnalysisResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 資料結構
# ------------------------------------------------------------------

@dataclass
class DetectedDot:
    """單一偵測點（全 tile 座標）。"""

    y: float
    x: float
    radius: float
    dot_type: str          # "her2" | "cep17"
    cell_id: int           # 0 表示不在任何 Cellpose 細胞內
    area: int
    circularity: float
    solidity: float
    contrast: float        # 紅: mean_a_dot - mean_a_ring; 黑: mean_L_ring - mean_L_dot
    score: float           # 排序用（紅: mean_a; 黑: -mean_L）


@dataclass
class CellDotResult:
    """單一細胞的點位計數結果。"""

    cell_id: int
    her2_dot_count: int = 0
    cep17_dot_count: int = 0
    her2_cep17_ratio: float = 0.0        # float("inf") 當 cep17_dot_count == 0
    is_amplified: bool = False
    blue_region_count: int = 0
    excluded: bool = False               # 多核（藍色區塊 ≥ threshold）→ 排除
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
        dish_ids_by_cell = elastic_dish_nucleus_matching(
            dish_nucleus_mask=dish_nucleus_mask_i32,
            strict_instance_mask=instance_mask_i32,
            cfg=config,
        )

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

    _finalize_per_cell(per_cell, dish_ids_by_cell, config)

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
# 色彩空間
# ------------------------------------------------------------------

def _rgb_to_lab(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB uint8 → LAB float32，回傳 (L, a, b)。"""
    lab = rgb2lab(img_rgb).astype(np.float32)
    return lab[..., 0], lab[..., 1], lab[..., 2]


# ------------------------------------------------------------------
# 紅點 (CEP17) 偵測 — 在單顆 cell 的局部 patch 上執行
# ------------------------------------------------------------------

def _detect_red_dots(
    a: np.ndarray,
    cell_roi: np.ndarray,
    bg_mask: np.ndarray,
    cfg: object,
    cell_id: int,
) -> List[DetectedDot]:
    """紅點偵測主流程（patch-local 座標）。"""
    h_depth = float(getattr(cfg, "dot_red_h", 12.0))
    a_min = float(getattr(cfg, "dot_red_a_min", 25.0))
    min_area = int(getattr(cfg, "dot_red_min_area", 7))
    max_area = int(getattr(cfg, "dot_red_max_area", 400))
    min_circ = float(getattr(cfg, "dot_red_min_circularity", 0.55))
    min_sol = float(getattr(cfg, "dot_red_min_solidity", 0.65))
    ring_gap = int(getattr(cfg, "dot_red_ring_gap", 2))
    ring_width = int(getattr(cfg, "dot_red_ring_width", 5))
    min_contrast = float(getattr(cfg, "dot_red_min_contrast", 10.0))

    seed_dilate = int(getattr(cfg, "dot_seed_dilate_radius", 3))

    a_masked = np.where(cell_roi, a, 0.0).astype(np.float32)

    peak_mask = h_maxima(a_masked, h=h_depth).astype(bool) & cell_roi
    if not peak_mask.any():
        return []

    dot_region = binary_dilation(peak_mask, disk(seed_dilate))
    dot_region &= (a_masked >= a_min) & cell_roi

    if not dot_region.any():
        return []

    label_img = label(dot_region, connectivity=2)
    props = regionprops(label_img, intensity_image=a_masked)

    dots: List[DetectedDot] = []
    for p in props:
        if p.area < min_area or p.area > max_area:
            continue
        circ = _circularity(p.area, p.perimeter)
        if circ < min_circ:
            continue
        if p.solidity < min_sol:
            continue

        rows, cols = p.coords[:, 0], p.coords[:, 1]
        mean_a_dot = float(a[rows, cols].mean())

        ring_stats = _compute_ring_stats(
            bbox=p.bbox,
            blob_pixels_rc=(rows, cols),
            ring_gap=ring_gap,
            ring_width=ring_width,
            cell_roi=cell_roi,
            bg_mask=bg_mask,
            intensity_imgs=(a,),
        )
        if ring_stats is None:
            continue
        mean_a_ring = ring_stats[0]
        contrast = mean_a_dot - mean_a_ring
        if contrast < min_contrast:
            continue

        cy, cx = p.centroid
        dots.append(DetectedDot(
            y=float(cy),
            x=float(cx),
            radius=math.sqrt(p.area / math.pi),
            dot_type="cep17",
            cell_id=cell_id,
            area=int(p.area),
            circularity=float(circ),
            solidity=float(p.solidity),
            contrast=float(contrast),
            score=mean_a_dot,
        ))

    return dots


# ------------------------------------------------------------------
# 黑點 (HER2) 偵測 — 在單顆 cell 的局部 patch 上執行
# ------------------------------------------------------------------

def _detect_black_dots(
    L: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    cell_roi: np.ndarray,
    bg_mask: np.ndarray,
    cfg: object,
    cell_id: int,
) -> List[DetectedDot]:
    """黑點偵測主流程（patch-local 座標）。"""
    h_depth = float(getattr(cfg, "dot_black_h", 15.0))
    l_max = float(getattr(cfg, "dot_black_l_max", 50.0))
    min_area = int(getattr(cfg, "dot_black_min_area", 5))
    max_area = int(getattr(cfg, "dot_black_max_area", 300))
    min_circ = float(getattr(cfg, "dot_black_min_circularity", 0.50))
    min_sol = float(getattr(cfg, "dot_black_min_solidity", 0.60))
    ring_gap = int(getattr(cfg, "dot_black_ring_gap", 2))
    ring_width = int(getattr(cfg, "dot_black_ring_width", 5))
    min_contrast = float(getattr(cfg, "dot_black_min_contrast", 18.0))
    min_ring_l = float(getattr(cfg, "dot_black_min_ring_l", 35.0))
    max_chroma = float(getattr(cfg, "dot_black_max_chroma", 18.0))
    max_median_chroma = float(getattr(cfg, "dot_black_max_median_chroma", 15.0))
    max_p90_chroma = float(getattr(cfg, "dot_black_max_p90_chroma", 22.0))
    p20_l_max = float(getattr(cfg, "dot_black_p20_l_max", 45.0))
    max_radius = float(getattr(cfg, "dot_black_max_radius", 7.0))
    very_dark_l_max = float(getattr(cfg, "dot_black_very_dark_l_max", 38.0))
    very_dark_min_contrast = float(
        getattr(cfg, "dot_black_very_dark_min_contrast", 12.0)
    )

    seed_dilate = int(
        getattr(
            cfg,
            "dot_black_seed_dilate_radius",
            getattr(cfg, "dot_seed_dilate_radius", 3),
        )
    )

    L_masked = np.where(cell_roi, L, 100.0).astype(np.float32)

    pit_mask = h_minima(L_masked, h=h_depth).astype(bool) & cell_roi
    if not pit_mask.any():
        return []

    dot_region = binary_dilation(pit_mask, disk(seed_dilate))
    dot_region &= (L_masked <= l_max) & cell_roi

    if not dot_region.any():
        return []

    label_img = label(dot_region, connectivity=2)
    props = regionprops(label_img, intensity_image=L_masked)

    def _log_reject(prop, gate: str, detail: str) -> None:
        """Debug-log 一個被某道閘門擋下的黑點候選（供下界調參用）。"""
        logger.debug(
            "黑點排除 cell=%d c=(%.0f,%.0f) gate=%s %s",
            cell_id, prop.centroid[0], prop.centroid[1], gate, detail,
        )

    dots: List[DetectedDot] = []
    for p in props:
        if p.area < min_area or p.area > max_area:
            _log_reject(p, "area", f"area={p.area} range=[{min_area},{max_area}]")
            continue
        radius = math.sqrt(p.area / math.pi)
        if max_radius > 0 and radius > max_radius:
            _log_reject(p, "radius", f"radius={radius:.1f} max={max_radius:.1f}")
            continue
        circ = _circularity(p.area, p.perimeter)
        if circ < min_circ:
            _log_reject(p, "circularity", f"circ={circ:.2f} min={min_circ:.2f}")
            continue
        if p.solidity < min_sol:
            _log_reject(p, "solidity", f"solidity={p.solidity:.2f} min={min_sol:.2f}")
            continue

        rows, cols = p.coords[:, 0], p.coords[:, 1]
        mean_L_dot = float(L[rows, cols].mean())
        mean_a_dot = float(a[rows, cols].mean())
        mean_b_dot = float(b[rows, cols].mean())
        p20_l_dot = float(np.percentile(L[rows, cols], 20))
        if p20_l_max > 0 and p20_l_dot > p20_l_max:
            _log_reject(p, "p20_l", f"p20_L={p20_l_dot:.1f} max={p20_l_max:.1f}")
            continue

        chroma_px = np.hypot(a[rows, cols], b[rows, cols])
        median_chroma = float(np.median(chroma_px))
        p90_chroma = float(np.percentile(chroma_px, 90))
        chroma_gate_failed = False
        if max_median_chroma > 0 and median_chroma > max_median_chroma:
            chroma_gate_failed = True
        if max_p90_chroma > 0 and p90_chroma > max_p90_chroma:
            chroma_gate_failed = True

        chroma = math.hypot(mean_a_dot, mean_b_dot)
        if max_chroma > 0 and chroma > max_chroma:
            chroma_gate_failed = True

        ring_stats = _compute_ring_stats(
            bbox=p.bbox,
            blob_pixels_rc=(rows, cols),
            ring_gap=ring_gap,
            ring_width=ring_width,
            cell_roi=cell_roi,
            bg_mask=bg_mask,
            intensity_imgs=(L,),
        )
        if ring_stats is None:
            _log_reject(p, "ring_empty", "no valid ring pixels")
            continue
        mean_L_ring = ring_stats[0]
        contrast = mean_L_ring - mean_L_dot
        if contrast < min_contrast:
            _log_reject(p, "contrast", f"contrast={contrast:.1f} min={min_contrast:.1f}")
            continue

        if mean_L_ring < min_ring_l:
            _log_reject(p, "ring_l", f"ring_L={mean_L_ring:.1f} min={min_ring_l:.1f}")
            continue

        if chroma_gate_failed:
            chroma_detail = (
                f"chroma med/p90/mean={median_chroma:.1f}/{p90_chroma:.1f}/{chroma:.1f}"
            )
            if mean_L_dot > very_dark_l_max:
                _log_reject(
                    p, "chroma+bright",
                    f"{chroma_detail} L={mean_L_dot:.1f}>{very_dark_l_max:.1f}",
                )
                continue
            if contrast < very_dark_min_contrast:
                _log_reject(
                    p, "chroma+lowcontrast",
                    f"{chroma_detail} contrast={contrast:.1f}<{very_dark_min_contrast:.1f}",
                )
                continue

        cy, cx = p.centroid
        dots.append(DetectedDot(
            y=float(cy),
            x=float(cx),
            radius=radius,
            dot_type="her2",
            cell_id=cell_id,
            area=int(p.area),
            circularity=float(circ),
            solidity=float(p.solidity),
            contrast=float(contrast),
            score=-mean_L_dot,
        ))

    if props:
        logger.debug(
            "黑點偵測 cell=%d 接受=%d / 候選blob=%d",
            cell_id, len(dots), len(props),
        )
    return dots


# ------------------------------------------------------------------
# DISH 細胞核 彈性匹配 — 用於排除多核細胞 + 擴張偵測 ROI
# 詳見 docs/sdd-elastic-dish-matching.md
# ------------------------------------------------------------------

def elastic_dish_nucleus_matching(
    dish_nucleus_mask: np.ndarray,
    strict_instance_mask: np.ndarray,
    cfg: object,
) -> Dict[int, List[int]]:
    """彈性匹配：IHC region 等向膨脹 → 候選收集 → centroid greedy 分配。

    三步驟流程（詳見 docs/sdd-elastic-dish-matching.md）：
        Step 1: 對每個 IHC 細胞 region 計算等向 dilation radius，使面積放大至
                ``dish_elastic_expand_factor`` 倍（近似圓形推導：
                r = (sqrt(A*f) - sqrt(A)) / sqrt(π)）。
        Step 2: 找出膨脹後 region 與 DISH 核的重疊 pixel，收集 candidate DISH ID。
        Step 3: 對 candidate pair 依 centroid 歐氏距離排序，greedy exclusive 分配
                ——每個 DISH 核最多被一個 IHC 細胞認領，IHC 可認領多顆（多核）。
                超過 ``dish_elastic_max_dist_px`` 的 pair 直接排除。

    Returns:
        ``{ihc_cell_id: [assigned_dish_id, ...]}``。
    """
    expand_factor = float(getattr(cfg, "dish_elastic_expand_factor", 1.5))
    max_dist_px = float(getattr(cfg, "dish_elastic_max_dist_px", 50.0))

    ihc_ids: List[int] = [int(v) for v in np.unique(strict_instance_mask) if v != 0]
    dish_ids: List[int] = [int(v) for v in np.unique(dish_nucleus_mask) if v != 0]
    result: Dict[int, List[int]] = {cid: [] for cid in ihc_ids}

    if not ihc_ids or not dish_ids:
        return result

    ihc_centroids: Dict[int, Tuple[float, float]] = {}
    cell_candidates: Dict[int, List[int]] = {cid: [] for cid in ihc_ids}

    # 用 find_objects 一次取得每顆 label 的 bbox，把所有 per-cell 運算侷限到
    # 自己的小視窗，避免在整張 strip 大小的陣列上反覆掃描 / dilation。
    H, W = strict_instance_mask.shape
    ihc_slices = find_objects(strict_instance_mask)
    for cid in ihc_ids:
        sl = ihc_slices[cid - 1] if cid - 1 < len(ihc_slices) else None
        if sl is None:
            continue
        local = (strict_instance_mask[sl] == cid)
        cy, cx = center_of_mass(local)
        ihc_centroids[cid] = (float(cy) + sl[0].start, float(cx) + sl[1].start)

        area = int(local.sum())
        if area == 0:
            continue
        r = max(1, round(
            (math.sqrt(area * expand_factor) - math.sqrt(area)) / math.sqrt(math.pi)
        ))
        # dilation 至多外擴 r：取 bbox 外加 r padding 的視窗，侷部 dilate 後
        # 結果與全圖 dilation 完全相同，但成本只與單顆細胞大小相關。
        r0 = max(0, sl[0].start - r)
        r1 = min(H, sl[0].stop + r)
        c0 = max(0, sl[1].start - r)
        c1 = min(W, sl[1].stop + r)
        win = (strict_instance_mask[r0:r1, c0:c1] == cid)
        expanded = binary_dilation(win, disk(r))
        overlap_vals = dish_nucleus_mask[r0:r1, c0:c1][expanded]
        cell_candidates[cid] = [int(v) for v in np.unique(overlap_vals) if v != 0]

    dish_centroids: Dict[int, Tuple[float, float]] = {}
    dish_slices = find_objects(dish_nucleus_mask)
    for did in dish_ids:
        sl = dish_slices[did - 1] if did - 1 < len(dish_slices) else None
        if sl is None:
            continue
        local = (dish_nucleus_mask[sl] == did)
        cy, cx = center_of_mass(local)
        dish_centroids[did] = (float(cy) + sl[0].start, float(cx) + sl[1].start)

    pairs: List[Tuple[float, int, int]] = []
    for cid, candidates in cell_candidates.items():
        iy, ix = ihc_centroids[cid]
        for did in candidates:
            dy, dx = dish_centroids[did]
            dist = math.hypot(iy - dy, ix - dx)
            if dist <= max_dist_px:
                pairs.append((dist, cid, did))
    pairs.sort(key=lambda x: x[0])

    assigned_dish_ids: set = set()
    for _dist, cid, did in pairs:
        if did in assigned_dish_ids:
            continue
        result[cid].append(did)
        assigned_dish_ids.add(did)

    exclude_thr = int(getattr(cfg, "dot_blue_exclude_threshold", 2))
    n_matched_any = sum(1 for v in result.values() if len(v) > 0)
    n_multi = sum(1 for v in result.values() if len(v) >= exclude_thr)
    logger.info(
        "elastic_dish_nucleus_matching: IHC=%d, DISH=%d, 有匹配=%d, "
        "多核候選(>=%d)=%d, expand_factor=%.1f, max_dist=%.1fpx",
        len(ihc_ids), len(dish_ids), n_matched_any, exclude_thr, n_multi,
        expand_factor, max_dist_px,
    )
    return result


# ------------------------------------------------------------------
# 形狀與對比環
# ------------------------------------------------------------------

def _circularity(area: float, perimeter: float) -> float:
    """圓形度 = 4π·area / perimeter²，圓=1.0。"""
    if perimeter <= 0:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)


def _compute_ring_stats(
    bbox: Tuple[int, int, int, int],
    blob_pixels_rc: Tuple[np.ndarray, np.ndarray],
    ring_gap: int,
    ring_width: int,
    cell_roi: np.ndarray,
    bg_mask: np.ndarray,
    intensity_imgs: Tuple[np.ndarray, ...],
) -> Optional[Tuple[float, ...]]:
    """在 blob 局部 bbox 內計算環形遮罩與 intensity 統計。

    所有座標皆為傳入 `cell_roi` / `bg_mask` 所在的座標系（一般為單顆細胞的 patch）。
    """
    r0, c0, r1, c1 = bbox
    pad = max(ring_gap + ring_width, 1)
    lr0 = max(0, r0 - pad)
    lc0 = max(0, c0 - pad)
    lr1 = min(cell_roi.shape[0], r1 + pad)
    lc1 = min(cell_roi.shape[1], c1 + pad)

    local_h = lr1 - lr0
    local_w = lc1 - lc0
    local_blob = np.zeros((local_h, local_w), dtype=bool)
    rows_abs, cols_abs = blob_pixels_rc
    local_blob[rows_abs - lr0, cols_abs - lc0] = True

    inner = binary_dilation(local_blob, disk(max(ring_gap, 0)))
    outer = binary_dilation(local_blob, disk(max(ring_gap + ring_width, 1)))
    ring = outer & (~inner)
    ring &= cell_roi[lr0:lr1, lc0:lc1]
    ring &= ~bg_mask[lr0:lr1, lc0:lc1]
    if not ring.any():
        return None

    out = []
    for img in intensity_imgs:
        out.append(float(img[lr0:lr1, lc0:lc1][ring].mean()))
    return tuple(out)


# ------------------------------------------------------------------
# 群聚合併
# ------------------------------------------------------------------

def _merge_close_dots(
    dots: List[DetectedDot],
    merge_distance: float,
) -> List[DetectedDot]:
    """距離 < merge_distance 的同類型點保留 score 較高者。"""
    if merge_distance <= 0 or len(dots) < 2:
        return dots

    coords = np.array([[d.y, d.x] for d in dots], dtype=np.float64)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=merge_distance)

    parent = list(range(len(dots)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, j in pairs:
        union(i, j)

    groups: Dict[int, List[int]] = {}
    for idx in range(len(dots)):
        groups.setdefault(find(idx), []).append(idx)

    kept: List[DetectedDot] = []
    for group_idx in groups.values():
        best = max(group_idx, key=lambda k: dots[k].score)
        kept.append(dots[best])
    return kept


# ------------------------------------------------------------------
# 細胞層級統計 + 擴增判定
# ------------------------------------------------------------------

def _finalize_per_cell(
    per_cell: Dict[int, CellDotResult],
    dish_ids_by_cell: Dict[int, List[int]],
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
        if cdr.blue_region_count >= exclude_thr:
            cdr.excluded = True
        elif cdr.blue_region_count == 0 and exclude_zero:
            cdr.excluded = True
        else:
            cdr.excluded = False
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
