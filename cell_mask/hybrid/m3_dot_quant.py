"""
M3: 紅/黑 Dot 定量模組

流程:
  1. 色彩解卷積 (Color Deconvolution) 分離紅色與黑色通道。
  2. LoG / TopHat blob 偵測定位 dot 候選。
  3. 連通元件面積基準叢集拆分 (cluster compensation)。
  4. 彙整每個細胞的 black / red dot 計數與 ratio。
"""

import logging
import math
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import blob_log

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
# 色彩解卷積
# ------------------------------------------------------------------

def color_deconvolution(
    image: np.ndarray,
    od_matrix: np.ndarray,
) -> np.ndarray:
    """RGB → 光學密度 → 色彩通道分離。

    Args:
        image: shape ``(H, W, 3)``、``uint8`` RGB 影像。
        od_matrix: shape ``(3, 3)`` 光學密度矩陣。
            各列為一種染色劑的 OD 向量。

    Returns:
        shape ``(H, W, 3)``、``float64`` 解卷積後各通道濃度。
        channel 0 = red stain, channel 1 = black stain, channel 2 = residual。
    """
    od_norm = _normalize_od_matrix(od_matrix)
    od_inv = np.linalg.inv(od_norm)

    img_float = image.astype(np.float64) / 255.0
    img_float = np.clip(img_float, 1e-6, 1.0)

    optical_density = -np.log(img_float)
    deconv = np.dot(optical_density, od_inv.T)
    return deconv


def _normalize_od_matrix(od_matrix: np.ndarray) -> np.ndarray:
    """正規化 OD 矩陣的每列向量為單位長度。"""
    norms = np.linalg.norm(od_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return od_matrix / norms


# ------------------------------------------------------------------
# Blob 偵測
# ------------------------------------------------------------------

def detect_blobs(
    channel: np.ndarray,
    min_sigma: float = 1.0,
    max_sigma: float = 5.0,
    num_sigma: int = 5,
    threshold: float = 0.02,
) -> np.ndarray:
    """使用 LoG (Laplacian of Gaussian) 偵測 blob 候選位置。

    Args:
        channel: shape ``(H, W)``、``float64`` 單通道濃度圖。
        min_sigma: LoG 最小 sigma。
        max_sigma: LoG 最大 sigma。
        num_sigma: sigma 離散取樣數。
        threshold: 偵測閾值。

    Returns:
        shape ``(N, 3)`` 陣列，每列為 ``(row, col, sigma)``。
    """
    blobs = blob_log(
        channel,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
    )
    return blobs


def cluster_compensate(
    blobs: np.ndarray,
    binary_mask: np.ndarray,
    cluster_area_factor: float = 2.5,
    min_blob_area: int = 3,
) -> int:
    """根據連通元件面積進行叢集補償計數。

    當一個連通元件面積大於 ``cluster_area_factor * avg_single_dot_area``
    時，將其拆分為多顆 dot。

    Args:
        blobs: LoG 偵測到的 blob 座標 ``(N, 3)``。
        binary_mask: 與 blob channel 對應的二值化遮罩。
        cluster_area_factor: 叢集面積倍率閾值。
        min_blob_area: 有效 dot 的最小面積。

    Returns:
        經叢集補償後的 dot 總數。
    """
    if blobs.shape[0] == 0:
        return 0

    labeled, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return blobs.shape[0]

    component_areas = ndimage.sum(
        binary_mask, labeled, range(1, num_features + 1)
    )
    valid_areas = [a for a in component_areas if a >= min_blob_area]
    if not valid_areas:
        return blobs.shape[0]

    avg_area = float(np.median(valid_areas))
    if avg_area < 1.0:
        avg_area = 1.0

    total = 0
    for area in component_areas:
        if area < min_blob_area:
            continue
        if area > cluster_area_factor * avg_area:
            total += max(1, round(area / avg_area))
        else:
            total += 1

    return total


# ------------------------------------------------------------------
# 單通道 Dot 計數 (含叢集補償)
# ------------------------------------------------------------------

def _count_dots_in_region(
    channel: np.ndarray,
    region_mask: np.ndarray,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    threshold: float,
    min_blob_area: int,
    cluster_area_factor: float,
) -> int:
    """在指定區域中計算 dot 數量。

    Args:
        channel: 解卷積後的單通道 ``(H, W)``。
        region_mask: 細胞區域二值遮罩 ``(H, W)``、``bool``。
        min_sigma: LoG min sigma。
        max_sigma: LoG max sigma。
        num_sigma: sigma 取樣數。
        threshold: LoG 閾值。
        min_blob_area: 最小 blob 面積。
        cluster_area_factor: 叢集補償因子。

    Returns:
        dot 計數 (int)。
    """
    masked_channel = channel * region_mask.astype(np.float64)

    blobs = detect_blobs(
        masked_channel,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
    )

    if blobs.shape[0] == 0:
        return 0

    # 產生二值化遮罩用於叢集補償
    binary = _threshold_channel(masked_channel)
    return cluster_compensate(
        blobs, binary, cluster_area_factor, min_blob_area
    )


def _threshold_channel(channel: np.ndarray) -> np.ndarray:
    """Otsu 自動閾值化解卷積通道。"""
    normalized = np.clip(channel / (channel.max() + 1e-8), 0, 1)
    img_uint8 = (normalized * 255).astype(np.uint8)
    _, binary = cv2.threshold(
        img_uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary.astype(np.uint8)


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

def quantify_dish_signals(
    masked_dish_image: np.ndarray,
    cell_instance_mask: np.ndarray,
    od_matrix: np.ndarray,
    min_sigma: float = 1.0,
    max_sigma: float = 5.0,
    num_sigma: int = 5,
    log_threshold: float = 0.02,
    min_blob_area: int = 3,
    cluster_area_factor: float = 2.5,
) -> List[CellAnalysisResult]:
    """對每個分割出的細胞進行紅/黑 dot 計數。

    Args:
        masked_dish_image: shape ``(H, W, 3)``、``uint8``。
        cell_instance_mask: shape ``(H, W)``、``int32``。背景=0。
        od_matrix: shape ``(3, 3)`` OD 矩陣。
        min_sigma: LoG min sigma。
        max_sigma: LoG max sigma。
        num_sigma: sigma 數量。
        log_threshold: LoG 偵測閾值。
        min_blob_area: 最小 blob 面積。
        cluster_area_factor: 叢集補償因子。

    Returns:
        每個細胞的 ``CellAnalysisResult`` 列表。
    """
    deconv = color_deconvolution(masked_dish_image, od_matrix)
    red_channel = deconv[:, :, 0]
    black_channel = deconv[:, :, 1]

    cell_ids = sorted(set(np.unique(cell_instance_mask)) - {0})
    results: List[CellAnalysisResult] = []

    for cid in cell_ids:
        result = _quantify_single_cell(
            cell_id=cid,
            cell_instance_mask=cell_instance_mask,
            red_channel=red_channel,
            black_channel=black_channel,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            log_threshold=log_threshold,
            min_blob_area=min_blob_area,
            cluster_area_factor=cluster_area_factor,
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
    red_channel: np.ndarray,
    black_channel: np.ndarray,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    log_threshold: float,
    min_blob_area: int,
    cluster_area_factor: float,
) -> CellAnalysisResult:
    """量化單一細胞的 dot 計數。"""
    region_mask = (cell_instance_mask == cell_id)
    cy, cx = ndimage.center_of_mass(region_mask)

    dot_params = dict(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=log_threshold,
        min_blob_area=min_blob_area,
        cluster_area_factor=cluster_area_factor,
    )

    black_count = _count_dots_in_region(
        black_channel, region_mask, **dot_params
    )
    red_count = _count_dots_in_region(
        red_channel, region_mask, **dot_params
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
