"""M3: 單細胞資料準備。

簡化後的 M3 主流程:
    1. 接收 M2 的 Cellpose cell instance mask。
    2. 將 mask 套用至 dish_mask_overlay，產生逐細胞結果。
    3. 所有 Cellpose 分割出的細胞標記為陽性。
"""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 資料結構
# ------------------------------------------------------------------

@dataclass
class CellAnalysisResult:
    """單一細胞的分析結果。

    Attributes:
        cell_id: 細胞實例 ID (>0)。
        centroid_x: 細胞質心 X 座標 (pixels)。
        centroid_y: 細胞質心 Y 座標 (pixels)。
        is_her2_positive: 是否為 HER2 陽性。
        hematoxylin_ratio: 細胞區域中 Hematoxylin 陽性像素佔比。
    """

    cell_id: int
    centroid_x: float
    centroid_y: float
    is_her2_positive: bool
    hematoxylin_ratio: float
    # --- M3b DISH 點位偵測結果（預設值允許舊流程零變動沿用）---
    her2_dot_count: int = 0
    cep17_dot_count: int = 0
    her2_cep17_ratio: float = 0.0
    is_amplified: bool = False
    blue_region_count: int = 0
    excluded: bool = False


# ------------------------------------------------------------------
# 逐細胞分析
# ------------------------------------------------------------------

def build_all_positive_results(
    cell_instance_mask: np.ndarray,
) -> List[CellAnalysisResult]:
    """將所有 Cellpose 分割細胞標記為陽性。

    Args:
        cell_instance_mask: shape ``(H, W)``、``int32``，背景=0。

    Returns:
        每個細胞對應一筆 ``CellAnalysisResult``，且皆為陽性。
    """
    cell_ids = sorted(int(cid) for cid in np.unique(cell_instance_mask) if cid != 0)
    if not cell_ids:
        logger.info("All-positive 標註完成: 0 個細胞")
        return []

    # 一次掃完整張 label mask 取得所有質心，避免逐細胞建立全圖 boolean mask。
    centroids = ndimage.center_of_mass(
        np.ones(cell_instance_mask.shape, dtype=np.uint8),
        labels=cell_instance_mask,
        index=cell_ids,
    )
    results: List[CellAnalysisResult] = []

    for cid, (cy, cx) in zip(cell_ids, centroids):
        if np.isnan(cy) or np.isnan(cx):
            continue
        results.append(
            CellAnalysisResult(
                cell_id=int(cid),
                centroid_x=float(cx),
                centroid_y=float(cy),
                is_her2_positive=True,
                hematoxylin_ratio=1.0,
            )
        )

    logger.info("All-positive 標註完成: %d 個細胞", len(results))
    return results
