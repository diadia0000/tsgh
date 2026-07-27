"""M3: 單細胞資料準備。

簡化後的 M3 主流程:
    1. 接收 M2 的 Cellpose cell instance mask。
    2. 將 mask 套用至 dish_mask_overlay，產生逐細胞結果。
    3. 所有 Cellpose 分割出的細胞標記為陽性。
"""

import logging
import math
from typing import List

import numpy as np
from scipy import ndimage
from skimage.segmentation import expand_labels

try:
    from ..hybrid_data_types import CellAnalysisResult  # noqa: F401 (re-exported)
except ImportError:
    from hybrid_data_types import CellAnalysisResult  # noqa: F401 (re-exported)

logger = logging.getLogger(__name__)


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
            )
        )

    logger.info("All-positive 標註完成: %d 個細胞", len(results))
    return results


# ------------------------------------------------------------------
# 細胞放大（M3 配對前處理）
# ------------------------------------------------------------------

def enlarge_cell_instances(
    cell_instance_mask: np.ndarray,
    cfg: object,
) -> np.ndarray:
    """把每顆細胞 instance 往背景膨脹，使其等效「面積」放大 ``cell_enlarge_area_factor`` 倍。

    用 skimage ``expand_labels`` 做 Voronoi 式外擴：所有 label 同步往背景長大、彼此
    不重疊（碰到鄰近細胞自動停在分水嶺），label 集合與數量完全不變、只是每顆變大。
    目的是讓綠色細胞蓋到更多 DISH 核（overlap 候選），提高 M3 配對成功率。

    ``expand_labels`` 只接受單一外擴距離，故以「面積放大倍數」換算：對等效圓半徑
    ``r = sqrt(area/π)``，面積 ×factor ⇔ 半徑 ×√factor，外擴距離 ``d = r*(√factor − 1)``。
    這裡取全部細胞等效半徑的「中位數」估 ``d``（典型細胞剛好 ×factor，偏小者放大略多、
    偏大者略少）。``factor <= 1.0`` 或無細胞時原樣回傳（等同停用）。

    Args:
        cell_instance_mask: ``(H, W)`` int，0=背景，1..N=細胞 ID。
        cfg: 具 ``cell_enlarge_area_factor``（面積放大倍數）的配置物件。

    Returns:
        膨脹後的 instance mask；dtype/shape/label 集合與輸入一致。
    """
    factor = float(getattr(cfg, "cell_enlarge_area_factor", 1.0))
    if factor <= 1.0:
        return cell_instance_mask

    counts = np.bincount(cell_instance_mask.ravel().astype(np.int64))
    areas = counts[1:][counts[1:] > 0]  # 去掉背景(0) 與標號空隙
    if areas.size == 0:
        return cell_instance_mask

    median_radius = float(np.median(np.sqrt(areas / math.pi)))
    distance = median_radius * (math.sqrt(factor) - 1.0)
    if distance <= 0:
        return cell_instance_mask

    enlarged = expand_labels(cell_instance_mask, distance=distance)
    logger.info(
        "enlarge_cell_instances: %d 顆細胞, factor=%.2f(面積), 中位半徑=%.1fpx, 外擴=%.1fpx",
        int(areas.size), factor, median_radius, distance,
    )
    return enlarged
