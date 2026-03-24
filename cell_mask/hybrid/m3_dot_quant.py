"""
M3: 紅/黑 Dot 定量模組

流程:
  1. 影像正規化 — pixel / 255 → [0, 1]。
  2. Gamma 調整 — alpha × input^sigma，強化訊號對比。
  3. RGB 顏色閾值分離黑色 / 紅色 dot。
  4. 雜訊移除 — 丟棄面積過小的 connected component。
  5. 形態學後處理 — close + open。
  6. Connected component 叢集補償計數。
  7. 彙整每個細胞的 black / red dot 計數與 ratio。
"""

import logging
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 資料結構
# ------------------------------------------------------------------

@dataclass
class CellAnalysisResult:
    """單一細胞的定量結果。

    Attributes:
        cell_id: 細胞實例 ID (>0)。
        centroid_x: 細胞質心 X 座標 (pixels)。
        centroid_y: 細胞質心 Y 座標 (pixels)。
        is_her2_positive: 是否屬於 Her2+ 區域。
        black_dot_count: 黑色 dot 計數。
        red_dot_count: 紅色 dot 計數。
        ratio: black / red 比值 (遵循 SDD ratio 規則)。
    """

    cell_id: int
    centroid_x: float
    centroid_y: float
    is_her2_positive: bool
    black_dot_count: int
    red_dot_count: int
    ratio: float


# ------------------------------------------------------------------
# 影像前處理
# ------------------------------------------------------------------

def normalize_image(image: np.ndarray) -> np.ndarray:
    """線性正規化至 [0, 1]。

    Args:
        image: shape ``(H, W, 3)``、``uint8`` RGB 影像。

    Returns:
        shape ``(H, W, 3)``、``float64`` 正規化影像。
    """
    return image.astype(np.float64) / 255.0


def gamma_adjust(
    image: np.ndarray,
    sigma: float = 0.7,
    alpha: float = 1.0,
) -> np.ndarray:
    """Gamma 亮度對比調整: ``output = alpha * input^sigma``。

    Args:
        image: shape ``(H, W, 3)``、``float64``、[0, 1] 正規化影像。
        sigma: Gamma 指數。< 1 提亮暗區, > 1 壓暗。
        alpha: 亮度倍率。

    Returns:
        shape ``(H, W, 3)``、``float64`` 調整後影像，仍在 [0, 1]。
    """
    adjusted = alpha * np.power(np.clip(image, 0.0, 1.0), sigma)
    return np.clip(adjusted, 0.0, 1.0)


# ------------------------------------------------------------------
# 顏色閾值分離
# ------------------------------------------------------------------

def threshold_black(
    image: np.ndarray,
    brightness_thresh: float = 0.30,
) -> np.ndarray:
    """分離黑色 dot (HER2 銀增強): 低亮度區域。

    Args:
        image: shape ``(H, W, 3)``、``float64``、[0, 1]。
        brightness_thresh: 平均亮度上限。

    Returns:
        shape ``(H, W)``、``uint8``、{0, 1} 二值遮罩。
    """
    brightness = image.mean(axis=2)
    return (brightness < brightness_thresh).astype(np.uint8)


def threshold_red(
    image: np.ndarray,
    r_min: float = 0.45,
    b_max: float = 0.35,
    diff_min: float = 0.10,
) -> np.ndarray:
    """分離紅色 dot (CEP17 Fast Red): 高 R、低 B 區域。

    Args:
        image: shape ``(H, W, 3)``、``float64``、[0, 1]。
        r_min: R 通道下限。
        b_max: B 通道上限。
        diff_min: (R - B) 最小差值。

    Returns:
        shape ``(H, W)``、``uint8``、{0, 1} 二值遮罩。
    """
    r_ch = image[:, :, 0]
    b_ch = image[:, :, 2]
    mask = (r_ch > r_min) & (b_ch < b_max) & ((r_ch - b_ch) > diff_min)
    return mask.astype(np.uint8)


def threshold_red_hsv(
    image: np.ndarray,
    hue_ranges: list = None,
    sat_min: float = 0.40,
    val_min: float = 0.35,
) -> np.ndarray:
    """使用 HSV 色彩空間分離紅/品紅色 dot (CEP17 Fast Red)。

    Fast Red 染劑偏品紅色調 (Hue ~325-355°)，HSV 對染色強度變異
    比 RGB 閾值更穩健。

    Args:
        image: shape ``(H, W, 3)``、``float64``、[0, 1]。
        hue_ranges: Hue 區間列表 (0-360°)，預設 ``[(0, 20), (325, 360)]``。
        sat_min: 最低飽和度。
        val_min: 最低明度。

    Returns:
        shape ``(H, W)``、``uint8``、{0, 1} 二值遮罩。
    """
    if hue_ranges is None:
        hue_ranges = [(0, 20), (325, 360)]

    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

    # OpenCV HSV: H=0-180, S=0-255, V=0-255
    h = hsv[:, :, 0].astype(np.float64) * 2.0   # → 0-360
    s = hsv[:, :, 1].astype(np.float64) / 255.0  # → 0-1
    v = hsv[:, :, 2].astype(np.float64) / 255.0  # → 0-1

    hue_mask = np.zeros(h.shape, dtype=bool)
    for lo, hi in hue_ranges:
        hue_mask |= (h >= lo) & (h <= hi)

    mask = hue_mask & (s >= sat_min) & (v >= val_min)
    return mask.astype(np.uint8)


# ------------------------------------------------------------------
# 雜訊移除與形態學後處理
# ------------------------------------------------------------------

def remove_small_components(
    binary: np.ndarray,
    min_area: int = 3,
) -> np.ndarray:
    """移除面積 < min_area 的 connected component。

    Args:
        binary: shape ``(H, W)``、``uint8``、{0, 1}。
        min_area: 最小有效 dot 面積 (pixels)。

    Returns:
        shape ``(H, W)``、``uint8``、{0, 1}。
    """
    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        return binary

    areas = ndimage.sum(binary, labeled, range(1, num_features + 1))
    for idx, area in enumerate(areas, start=1):
        if area < min_area:
            binary[labeled == idx] = 0
    return binary


def morphological_postprocess(
    binary: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """形態學後處理: close (合併鄰近 cluster) → open (消除碎片)。

    Args:
        binary: shape ``(H, W)``、``uint8``、{0, 1}。
        kernel_size: 形態學核直徑。

    Returns:
        shape ``(H, W)``、``uint8``、{0, 1}。
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    return result


def filter_by_circularity(
    binary: np.ndarray,
    min_circularity: float = 0.4,
) -> np.ndarray:
    """過濾非圓形的 connected component，排除邊緣碎片與拉長 artifact。

    circularity = 4π × area / perimeter²，完美圓 = 1.0。
    DISH 的 dot 通常 > 0.4。

    Args:
        binary: shape ``(H, W)``、``uint8``、{0, 1}。
        min_circularity: 最低圓度閾值。

    Returns:
        shape ``(H, W)``、``uint8``、{0, 1}。
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    filtered = np.zeros_like(binary)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity >= min_circularity:
            cv2.drawContours(filtered, [cnt], -1, 1, -1)
    return filtered


# ------------------------------------------------------------------
# 叢集補償計數
# ------------------------------------------------------------------

def cluster_compensate(
    binary_mask: np.ndarray,
    cluster_area_factor: float = 2.5,
    min_dot_area: int = 3,
) -> int:
    """根據連通元件面積進行叢集補償計數。

    面積 > ``cluster_area_factor * median_area`` 的元件
    拆分為 ``round(area / median_area)`` 顆 dot。

    Args:
        binary_mask: 經後處理的二值遮罩 ``(H, W)``、{0, 1}。
        cluster_area_factor: 叢集面積倍率閾值。
        min_dot_area: 有效 dot 的最小面積。

    Returns:
        經叢集補償後的 dot 總數。
    """
    labeled, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return 0

    areas = ndimage.sum(
        binary_mask, labeled, range(1, num_features + 1)
    )
    valid_areas = [a for a in areas if a >= min_dot_area]
    if not valid_areas:
        return 0

    median_area = float(np.median(valid_areas))
    if median_area < 1.0:
        median_area = 1.0

    total = 0
    for area in areas:
        if area < min_dot_area:
            continue
        if area > cluster_area_factor * median_area:
            total += max(1, round(area / median_area))
        else:
            total += 1
    return total


# ------------------------------------------------------------------
# 單通道 Dot 計數 (region-level)
# ------------------------------------------------------------------

def _count_dots_in_region(
    binary: np.ndarray,
    region_mask: np.ndarray,
    cluster_area_factor: float,
    min_dot_area: int,
) -> int:
    """在指定細胞區域中計算 dot 數量。

    Args:
        binary: 全圖二值遮罩 ``(H, W)``、{0, 1}。
        region_mask: 細胞區域 ``(H, W)``、``bool``。
        cluster_area_factor: 叢集補償因子。
        min_dot_area: 最小 dot 面積。

    Returns:
        dot 計數。
    """
    rows, cols = np.where(region_mask)
    if rows.size == 0:
        return 0

    r_min, r_max = rows.min(), rows.max() + 1
    c_min, c_max = cols.min(), cols.max() + 1

    local_binary = binary[r_min:r_max, c_min:c_max].copy()
    local_mask = region_mask[r_min:r_max, c_min:c_max]
    local_binary[~local_mask] = 0

    if local_binary.sum() == 0:
        return 0

    return cluster_compensate(local_binary, cluster_area_factor, min_dot_area)


# ------------------------------------------------------------------
# Ratio 計算
# ------------------------------------------------------------------

def compute_ratio(black: int, red: int) -> float:
    """計算 black/red 比值，遵循 SDD ratio 規則。

    Args:
        black: 黑色 dot 計數。
        red: 紅色 dot 計數。

    Returns:
        ratio: black / red。
            red=0 且 black>0 → ``inf``；
            red=0 且 black=0 → ``nan``。
    """
    if red == 0:
        return float("inf") if black > 0 else float("nan")
    return black / red


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

def quantify_overlay_signals(
    masked_overlay_image: np.ndarray,
    cell_instance_mask: np.ndarray,
    gamma_sigma: float = 0.7,
    gamma_alpha: float = 1.0,
    black_brightness_thresh: float = 0.30,
    red_r_min: float = 0.45,
    red_b_max: float = 0.35,
    red_diff_min: float = 0.10,
    min_dot_area: int = 3,
    morph_kernel_size: int = 3,
    cluster_area_factor: float = 2.5,
) -> List[CellAnalysisResult]:
    """對每個分割出的細胞進行紅/黑 dot 計數。

    Args:
        masked_overlay_image: shape ``(H, W, 3)``、``uint8``。
        cell_instance_mask: shape ``(H, W)``、``int32``。背景=0。
        gamma_sigma: Gamma 指數。
        gamma_alpha: Gamma 亮度倍率。
        black_brightness_thresh: 黑色 dot 亮度上限 (正規化)。
        red_r_min: 紅色 R 通道下限。
        red_b_max: 紅色 B 通道上限。
        red_diff_min: 紅色 (R-B) 最小差值。
        min_dot_area: 最小有效 dot 面積 (pixels)。
        morph_kernel_size: 形態學核直徑。
        cluster_area_factor: 叢集補償因子。

    Returns:
        每個細胞的 ``CellAnalysisResult`` 列表。
    """
    # Step 1–2: 正規化 + Gamma
    img_norm = normalize_image(masked_overlay_image)
    img_gamma = gamma_adjust(img_norm, sigma=gamma_sigma, alpha=gamma_alpha)

    # Step 3: 顏色閾值分離 (RGB + HSV 聯集)
    black_binary = threshold_black(img_gamma, black_brightness_thresh)
    red_binary_rgb = threshold_red(img_gamma, red_r_min, red_b_max, red_diff_min)
    red_binary_hsv = threshold_red_hsv(img_gamma)
    red_binary = np.maximum(red_binary_rgb, red_binary_hsv)

    # Step 4: 雜訊移除
    black_binary = remove_small_components(black_binary, min_dot_area)
    red_binary = remove_small_components(red_binary, min_dot_area)

    # Step 5: 形態學後處理
    black_binary = morphological_postprocess(black_binary, morph_kernel_size)
    red_binary = morphological_postprocess(red_binary, morph_kernel_size)

    # Step 5.5: 圓度過濾 — 排除非圓形 artifact
    black_binary = filter_by_circularity(black_binary, min_circularity=0.4)
    red_binary = filter_by_circularity(red_binary, min_circularity=0.4)

    # Step 6–7: 逐細胞計數
    cell_ids = sorted(set(np.unique(cell_instance_mask)) - {0})
    results: List[CellAnalysisResult] = []

    for cid in cell_ids:
        result = _quantify_single_cell(
            cell_id=cid,
            cell_instance_mask=cell_instance_mask,
            black_binary=black_binary,
            red_binary=red_binary,
            cluster_area_factor=cluster_area_factor,
            min_dot_area=min_dot_area,
        )
        results.append(result)

    logger.info(
        "Dot 定量完成: %d 個細胞, "
        "平均 black=%.1f, 平均 red=%.1f",
        len(results),
        _safe_mean([r.black_dot_count for r in results]),
        _safe_mean([r.red_dot_count for r in results]),
    )
    return results


def _quantify_single_cell(
    cell_id: int,
    cell_instance_mask: np.ndarray,
    black_binary: np.ndarray,
    red_binary: np.ndarray,
    cluster_area_factor: float,
    min_dot_area: int,
) -> CellAnalysisResult:
    """量化單一細胞的 dot 計數。"""
    region_mask = (cell_instance_mask == cell_id)
    cy, cx = ndimage.center_of_mass(region_mask)

    black_count = _count_dots_in_region(
        black_binary, region_mask, cluster_area_factor, min_dot_area,
    )
    red_count = _count_dots_in_region(
        red_binary, region_mask, cluster_area_factor, min_dot_area,
    )
    ratio = compute_ratio(black_count, red_count)

    return CellAnalysisResult(
        cell_id=cell_id,
        centroid_x=float(cx),
        centroid_y=float(cy),
        is_her2_positive=True,
        black_dot_count=black_count,
        red_dot_count=red_count,
        ratio=ratio,
    )


def _safe_mean(values: list) -> float:
    """安全計算平均值，空列表回傳 0.0。"""
    if not values:
        return 0.0
    return sum(values) / len(values)
