"""M3b: DISH 紅點 (CEP17) / 黑點 (HER2) 偵測模組。

演算法核心：
    紅點 / 黑點：LAB 色彩空間 + H-形態重建 + 多準則閘控 (per-cell ROI)
    多核排除：Cellpose DISH 模型對純 DISH 圖偵測細胞核 instance，
              對每個 IHC 細胞先等向膨脹 region（容忍切片對齊誤差），
              再以重疊收集 candidate DISH 核，最後依 centroid 距離 greedy
              exclusive 分配；匹配到的 DISH 核數 ≥
              dot_blue_exclude_threshold → 排除（多核細胞）。
              詳見 docs/sdd-elastic-dish-matching.md。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    center_of_mass,
    distance_transform_edt,
)
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
) -> Tuple[List[DetectedDot], Dict[int, CellDotResult]]:
    """偵測 dish_image 中所有 HER2 黑點與 CEP17 紅點，並分配至各細胞。

    Args:
        dish_image: (H, W, 3) uint8 RGB，白背景(255)，已套用 core_mask。
        instance_mask: (H, W) 整數，0=背景，1..N=細胞ID。
        config: 具有 ``dot_*`` 欄位的配置物件。
        dish_nucleus_mask: (H, W) int，0=背景，1..M=DISH 細胞核 ID（Cellpose 輸出）。
            用於計算多核重疊以排除多核細胞。

    Returns:
        (all_dots, per_cell_results)
    """
    if dish_image.ndim != 3 or dish_image.shape[2] != 3:
        raise ValueError(f"dish_image 必須為 (H,W,3) RGB，實際 {dish_image.shape}")
    if dish_image.shape[:2] != instance_mask.shape:
        raise ValueError(
            f"shape 不一致: dish={dish_image.shape[:2]} vs mask={instance_mask.shape}"
        )

    instance_mask_i32 = instance_mask.astype(np.int32, copy=False)

    L, a, b = _rgb_to_lab(dish_image)
    bg_mask, cell_roi = _build_masks(L, instance_mask_i32, config)

    if not cell_roi.any():
        logger.info("detect_all_dots: cell_roi 為空，回傳空結果")
        return [], {}

    # 為 ROI 膨脹區內的 dot 提供最近鄰 cell_id 查找表
    nearest_cell_id, nearest_cell_dist = _build_nearest_cell_lookup(
        instance_mask_i32,
        cell_roi,
    )
    boundary_dist = _compute_boundary_distance(instance_mask_i32)

    red_dots = _detect_red_dots(
        a=a,
        cell_roi=cell_roi,
        bg_mask=bg_mask,
        nearest_instance_mask=nearest_cell_id,
        strict_instance_mask=instance_mask_i32,
        nearest_cell_dist=nearest_cell_dist,
        boundary_dist=boundary_dist,
        cfg=config,
    )
    black_dots = _detect_black_dots(
        L=L,
        a=a,
        b=b,
        cell_roi=cell_roi,
        bg_mask=bg_mask,
        nearest_instance_mask=nearest_cell_id,
        strict_instance_mask=instance_mask_i32,
        nearest_cell_dist=nearest_cell_dist,
        boundary_dist=boundary_dist,
        cfg=config,
    )

    # 群聚合併（同類型）
    default_merge_distance = float(getattr(config, "dot_merge_distance", 3.0))
    red_merge_distance = float(
        getattr(config, "dot_red_merge_distance", default_merge_distance)
    )
    black_merge_distance = float(
        getattr(config, "dot_black_merge_distance", default_merge_distance)
    )
    red_dots = _merge_close_dots(red_dots, red_merge_distance)
    black_dots = _merge_close_dots(black_dots, black_merge_distance)

    # 多核排除計數：以 Cellpose DISH 核 instance 與 IHC 細胞做彈性匹配
    dish_ids_by_cell = _elastic_dish_nucleus_matching(
        dish_nucleus_mask=dish_nucleus_mask.astype(np.int32, copy=False),
        strict_instance_mask=instance_mask_i32,
        cfg=config,
    )

    all_dots = red_dots + black_dots
    per_cell = _group_dots_by_cell(
        all_dots, instance_mask_i32, dish_ids_by_cell, config
    )

    logger.info(
        "detect_all_dots: 紅點=%d, 黑點=%d, 涉及細胞=%d",
        len(red_dots), len(black_dots), len(per_cell),
    )
    return all_dots, per_cell


# ------------------------------------------------------------------
# 色彩空間與遮罩
# ------------------------------------------------------------------

def _rgb_to_lab(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB uint8 → LAB float32，回傳 (L, a, b)。"""
    lab = rgb2lab(img_rgb).astype(np.float32)
    return lab[..., 0], lab[..., 1], lab[..., 2]


def _build_masks(
    L: np.ndarray,
    instance_mask: np.ndarray,
    cfg: object,
) -> Tuple[np.ndarray, np.ndarray]:
    """建立 (bg_mask, cell_roi)。

    cell_roi 會以 ``dot_cell_roi_dilate`` (px) 往外擴，
    以涵蓋 Cellpose 邊界外緊鄰的 dot（典型值 2–4 px）。
    背景像素 (L>閾值) 仍會從 ROI 中扣除。
    """
    bg_threshold = float(getattr(cfg, "dot_background_l_threshold", 95.0))
    roi_dilate = int(getattr(cfg, "dot_cell_roi_dilate", 0))
    bg_mask = L > bg_threshold

    cell_base = instance_mask > 0
    if roi_dilate > 0:
        cell_base = binary_dilation(cell_base, disk(roi_dilate))
    cell_roi = cell_base & (~bg_mask)
    return bg_mask, cell_roi


# ------------------------------------------------------------------
# 紅點 (CEP17) 偵測
# ------------------------------------------------------------------

def _detect_red_dots(
    a: np.ndarray,
    cell_roi: np.ndarray,
    bg_mask: np.ndarray,
    nearest_instance_mask: np.ndarray,
    strict_instance_mask: np.ndarray,
    nearest_cell_dist: np.ndarray,
    boundary_dist: np.ndarray,
    cfg: object,
) -> List[DetectedDot]:
    """紅點偵測主流程。"""
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

    # a* 限制在 cell_roi，其餘填 0（中性）
    a_masked = np.where(cell_roi, a, 0.0).astype(np.float32)

    # Step 1: H-maxima 找局部紅色極值種子（突出鄰域 ≥ h 階）
    peak_mask = h_maxima(a_masked, h=h_depth).astype(bool) & cell_roi
    if not peak_mask.any():
        return []

    # Step 2: 種子膨脹 disk(seed_dilate)，再與 a*>=a_min 交集形成 dot 連通區
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

        rows_abs, cols_abs = p.coords[:, 0], p.coords[:, 1]
        mean_a_dot = float(a[rows_abs, cols_abs].mean())

        ring_stats = _compute_ring_stats(
            bbox=p.bbox,
            blob_pixels_rc=(rows_abs, cols_abs),
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
        cell_id = _assign_cell_id(
            strict_instance_mask=strict_instance_mask,
            nearest_instance_mask=nearest_instance_mask,
            nearest_cell_dist=nearest_cell_dist,
            boundary_dist=boundary_dist,
            rows_abs=rows_abs,
            cols_abs=cols_abs,
            cy=cy,
            cx=cx,
            cfg=cfg,
        )
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
# 黑點 (HER2) 偵測
# ------------------------------------------------------------------

def _detect_black_dots(
    L: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    cell_roi: np.ndarray,
    bg_mask: np.ndarray,
    nearest_instance_mask: np.ndarray,
    strict_instance_mask: np.ndarray,
    nearest_cell_dist: np.ndarray,
    boundary_dist: np.ndarray,
    cfg: object,
) -> List[DetectedDot]:
    """黑點偵測主流程。"""
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

    # L* 限制在 cell_roi，其餘填 100（中性亮值，避免成為極小值）
    L_masked = np.where(cell_roi, L, 100.0).astype(np.float32)

    # Step 1: H-minima 找局部暗區種子
    pit_mask = h_minima(L_masked, h=h_depth).astype(bool) & cell_roi
    if not pit_mask.any():
        return []

    # Step 2: 種子膨脹 disk(seed_dilate)，限於 L*<=l_max 的候選暗區
    dot_region = binary_dilation(pit_mask, disk(seed_dilate))
    dot_region &= (L_masked <= l_max) & cell_roi

    if not dot_region.any():
        return []

    label_img = label(dot_region, connectivity=2)
    props = regionprops(label_img, intensity_image=L_masked)

    dots: List[DetectedDot] = []
    for p in props:
        if p.area < min_area or p.area > max_area:
            continue
        radius = math.sqrt(p.area / math.pi)
        if max_radius > 0 and radius > max_radius:
            continue
        circ = _circularity(p.area, p.perimeter)
        if circ < min_circ:
            continue
        if p.solidity < min_sol:
            continue

        rows_abs, cols_abs = p.coords[:, 0], p.coords[:, 1]
        mean_L_dot = float(L[rows_abs, cols_abs].mean())
        mean_a_dot = float(a[rows_abs, cols_abs].mean())
        mean_b_dot = float(b[rows_abs, cols_abs].mean())
        p20_l_dot = float(np.percentile(L[rows_abs, cols_abs], 20))
        if p20_l_max > 0 and p20_l_dot > p20_l_max:
            continue

        chroma_px = np.hypot(a[rows_abs, cols_abs], b[rows_abs, cols_abs])
        median_chroma = float(np.median(chroma_px))
        p90_chroma = float(np.percentile(chroma_px, 90))
        chroma_gate_failed = False
        if max_median_chroma > 0 and median_chroma > max_median_chroma:
            chroma_gate_failed = True
        if max_p90_chroma > 0 and p90_chroma > max_p90_chroma:
            chroma_gate_failed = True

        # 色彩中性：blob 內部 a/b 的彩度不能太高（否則是彩色暗點，例如暗紫核）。
        # 但若點位非常暗且局部對比足夠，允許視為可疑 HER2 點保留召回率。
        chroma = math.hypot(mean_a_dot, mean_b_dot)
        if max_chroma > 0 and chroma > max_chroma:
            chroma_gate_failed = True

        ring_stats = _compute_ring_stats(
            bbox=p.bbox,
            blob_pixels_rc=(rows_abs, cols_abs),
            ring_gap=ring_gap,
            ring_width=ring_width,
            cell_roi=cell_roi,
            bg_mask=bg_mask,
            intensity_imgs=(L,),
        )
        if ring_stats is None:
            continue
        mean_L_ring = ring_stats[0]
        contrast = mean_L_ring - mean_L_dot
        if contrast < min_contrast:
            continue

        # 非暗核核心：ring 本身不能太暗
        if mean_L_ring < min_ring_l:
            continue

        if chroma_gate_failed:
            if mean_L_dot > very_dark_l_max:
                continue
            if contrast < very_dark_min_contrast:
                continue

        cy, cx = p.centroid
        cell_id = _assign_cell_id(
            strict_instance_mask=strict_instance_mask,
            nearest_instance_mask=nearest_instance_mask,
            nearest_cell_dist=nearest_cell_dist,
            boundary_dist=boundary_dist,
            rows_abs=rows_abs,
            cols_abs=cols_abs,
            cy=cy,
            cx=cx,
            cfg=cfg,
        )
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

    return dots


# ------------------------------------------------------------------
# DISH 細胞核 彈性匹配 — 用於排除多核細胞（主要方法）
# 詳見 docs/sdd-elastic-dish-matching.md
# ------------------------------------------------------------------

def _elastic_dish_nucleus_matching(
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

    Args:
        dish_nucleus_mask: (H, W) int32，0=背景，1..M=DISH 細胞核 ID。
        strict_instance_mask: (H, W) int32，0=背景，1..N=IHC 細胞 ID。
        cfg: 須提供：
            ``dish_elastic_expand_factor`` (float, 預設 1.5) — 面積膨脹倍數
            ``dish_elastic_max_dist_px`` (float, 預設 50.0) — centroid 最大距離閾值

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

    # Step 1 & 2: 逐顆 IHC 細胞膨脹，收集與之重疊的 DISH 核 candidate
    ihc_centroids: Dict[int, Tuple[float, float]] = {}
    cell_candidates: Dict[int, List[int]] = {cid: [] for cid in ihc_ids}

    for cid in ihc_ids:
        cell_mask = (strict_instance_mask == cid)
        cy, cx = center_of_mass(cell_mask)
        ihc_centroids[cid] = (float(cy), float(cx))

        area = int(cell_mask.sum())
        if area == 0:
            continue
        r = max(1, round(
            (math.sqrt(area * expand_factor) - math.sqrt(area)) / math.sqrt(math.pi)
        ))
        expanded = binary_dilation(cell_mask, disk(r))
        overlap_vals = dish_nucleus_mask[expanded]
        cell_candidates[cid] = [int(v) for v in np.unique(overlap_vals) if v != 0]

    # 預先計算所有 DISH 核 centroid
    dish_centroids: Dict[int, Tuple[float, float]] = {}
    for did in dish_ids:
        cy, cx = center_of_mass(dish_nucleus_mask == did)
        dish_centroids[did] = (float(cy), float(cx))

    # Step 3: 依 centroid 距離排序，greedy exclusive 分配
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
        "_elastic_dish_nucleus_matching: IHC=%d, DISH=%d, 有匹配=%d, "
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

    Args:
        bbox: regionprops 的 bbox = (min_row, min_col, max_row, max_col)
        blob_pixels_rc: (rows, cols) 的絕對座標（全圖）。
        intensity_imgs: 需要統計的 intensity 影像們。

    Returns:
        (mean_i1, mean_i2, ...) 對應 intensity_imgs；若 ring 空則回傳 None。
    """
    r0, c0, r1, c1 = bbox
    pad = max(ring_gap + ring_width, 1)
    lr0 = max(0, r0 - pad)
    lc0 = max(0, c0 - pad)
    lr1 = min(cell_roi.shape[0], r1 + pad)
    lc1 = min(cell_roi.shape[1], c1 + pad)

    # 在局部區域內建立 blob 遮罩
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


def _build_nearest_cell_lookup(
    instance_mask: np.ndarray,
    cell_roi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """為 cell_roi（可能已膨脹超出 instance_mask）內的每個像素，
    建立「最近 instance_mask 細胞 ID」的查找表。

    - instance_mask > 0 的像素，id 不變
    - cell_roi 內但 instance_mask == 0 的像素，指派最近的 cell id
    - 其他像素保持 0
    """
    base = (instance_mask > 0)
    # 若 cell_roi 已完全位於 base 內，無須計算最近鄰
    need_lookup = cell_roi & (~base)
    dist_to_cell = distance_transform_edt(~base).astype(np.float32)
    if not need_lookup.any():
        return instance_mask.astype(np.int32), dist_to_cell

    # distance_transform_edt(background=True 的 mask) 會回傳到最近前景的距離。
    # return_indices=True 回傳每個位置「最近前景像素」的座標。
    # 我們要以 base（真細胞像素）為前景，找 need_lookup 內每個位置最近的 base id。
    _, indices = distance_transform_edt(~base, return_indices=True)
    out = instance_mask.astype(np.int32, copy=True)
    yy, xx = indices
    out_vals = instance_mask[yy[need_lookup], xx[need_lookup]]
    out[need_lookup] = out_vals
    return out, dist_to_cell


def _compute_boundary_distance(instance_mask: np.ndarray) -> np.ndarray:
    """回傳每個像素到最近 label 邊界的距離（px）。"""
    m = instance_mask
    boundary = np.zeros(m.shape, dtype=bool)

    boundary[1:, :] |= (m[1:, :] != m[:-1, :])
    boundary[:-1, :] |= (m[:-1, :] != m[1:, :])
    boundary[:, 1:] |= (m[:, 1:] != m[:, :-1])
    boundary[:, :-1] |= (m[:, :-1] != m[:, 1:])

    # 將貼邊細胞視為邊界，避免邊界附近誤歸屬。
    boundary[0, :] |= (m[0, :] > 0)
    boundary[-1, :] |= (m[-1, :] > 0)
    boundary[:, 0] |= (m[:, 0] > 0)
    boundary[:, -1] |= (m[:, -1] > 0)

    return distance_transform_edt(~boundary).astype(np.float32)


def _assign_cell_id(
    strict_instance_mask: np.ndarray,
    nearest_instance_mask: np.ndarray,
    nearest_cell_dist: np.ndarray,
    boundary_dist: np.ndarray,
    rows_abs: np.ndarray,
    cols_abs: np.ndarray,
    cy: float,
    cx: float,
    cfg: object,
) -> int:
    """依 blob 與細胞重疊、距離與邊界緩衝區分配 cell_id。"""
    min_overlap_ratio = float(getattr(cfg, "dot_assignment_min_overlap_ratio", 0.20))
    max_assignment_dist = float(getattr(cfg, "dot_assignment_max_distance", 2.0))
    boundary_margin = float(getattr(cfg, "dot_assignment_boundary_margin", 1.0))

    iy = int(round(cy))
    ix = int(round(cx))
    H, W = strict_instance_mask.shape
    if not (0 <= iy < H and 0 <= ix < W):
        return 0

    # 優先使用 blob 與原始 instance mask 的實際重疊來決定 cell_id。
    strict_ids = strict_instance_mask[rows_abs, cols_abs]
    inside_ids = strict_ids[strict_ids > 0]
    cell_id = 0
    if inside_ids.size > 0:
        overlap_ratio = inside_ids.size / float(strict_ids.size)
        if overlap_ratio < min_overlap_ratio:
            return 0
        votes = np.bincount(inside_ids)
        cell_id = int(np.argmax(votes))
    else:
        # 若 blob 全在膨脹 ROI 外圍，僅允許非常靠近真細胞時才最近鄰指派。
        if nearest_cell_dist[iy, ix] > max_assignment_dist:
            return 0
        cell_id = int(nearest_instance_mask[iy, ix])

    if cell_id <= 0:
        return 0

    # 邊界附近 (含細胞互相貼合處) 視為不確定區，避免誤判歸屬。
    if boundary_dist[iy, ix] < boundary_margin:
        return 0

    return cell_id


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

    # 聯合查找（Union-Find）
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

    # 每組保留 score 最高者
    kept: List[DetectedDot] = []
    for group_idx in groups.values():
        best = max(group_idx, key=lambda k: dots[k].score)
        kept.append(dots[best])
    return kept


# ------------------------------------------------------------------
# 依細胞分組 + 擴增判定
# ------------------------------------------------------------------

def _group_dots_by_cell(
    all_dots: List[DetectedDot],
    instance_mask: np.ndarray,
    dish_ids_by_cell: Dict[int, List[int]],
    cfg: object,
) -> Dict[int, CellDotResult]:
    """依 cell_id 分組，計算計數與擴增判定，並標記多核排除細胞。

    ``dish_ids_by_cell`` 來自 :func:`_elastic_dish_nucleus_matching`：
    ``{ihc_cell_id: [assigned_dish_nucleus_id, ...]}``。
    """
    amp_ratio = float(getattr(cfg, "dot_amplification_ratio", 2.0))
    amp_count = int(getattr(cfg, "dot_her2_count_threshold", 6))
    exclude_thr = int(getattr(cfg, "dot_blue_exclude_threshold", 2))
    exclude_zero = bool(getattr(cfg, "dish_elastic_exclude_zero", False))

    per_cell: Dict[int, CellDotResult] = {}
    cell_ids = sorted(set(np.unique(instance_mask)) - {0})
    for cid in cell_ids:
        cid_int = int(cid)
        per_cell[cid_int] = CellDotResult(cell_id=cid_int)

    for dot in all_dots:
        if dot.cell_id <= 0:
            continue
        cdr = per_cell.get(dot.cell_id)
        if cdr is None:
            continue
        if dot.dot_type == "her2":
            cdr.her2_dots.append(dot)
        elif dot.dot_type == "cep17":
            cdr.cep17_dots.append(dot)

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

    return per_cell


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
