"""M3b 偵測核心：紅點 (CEP17) / 黑點 (HER2) 的像素級偵測與幾何/合併工具。

從 m3_dot_detection.py 拆出，專注於「單顆 cell 局部 patch → DetectedDot 串列」，
不涉及 elastic matching 或 cell 層級編排。
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from skimage.color import rgb2lab
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_dilation,
    disk,
    h_maxima,
    h_minima,
)

from cell_mask.hybrid.hybrid_data_types import DetectedDot  # noqa: F401 (re-exported)

logger = logging.getLogger(__name__)


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
