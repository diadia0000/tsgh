"""
M2: Cellpose 實例分割模組

在遮罩後的 DISH 影像上執行 Cellpose 推論，
產出 ``cell_instance_mask`` (背景=0, 細胞ID=1..N)。
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from skimage.segmentation import clear_border

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Cellpose 推論器
# ------------------------------------------------------------------

class CellposeSegmenter:
    """封裝 Cellpose 模型載入與推論。

    Attributes:
        model: Cellpose 模型物件。
        diameter: 預估細胞直徑 (pixels); None 時自動估計。
        flow_threshold: flow error 閾值。
        cellprob_threshold: 細胞機率閾值。
    """

    def __init__(
        self,
        model_path: Path,
        diameter: Optional[float] = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        gpu: bool = True,
    ) -> None:
        """初始化 Cellpose 推論器。

        Args:
            model_path: 預訓練 Cellpose 模型路徑。
            diameter: 細胞直徑 (pixels), None 則自動估計。
            flow_threshold: flow 誤差閾值。
            cellprob_threshold: 細胞機率閾值。
            gpu: 是否使用 GPU。
        """
        from cellpose.models import CellposeModel  # noqa: WPS433

        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold

        self.model = CellposeModel(
            gpu=gpu,
            pretrained_model=str(model_path),
        )
        logger.info(
            "Cellpose 模型載入完成: %s (diameter=%s)",
            model_path.name,
            diameter,
        )

    def predict(self, image: np.ndarray) -> np.ndarray:
        """執行單張影像的實例分割。

        Args:
            image: shape ``(H, W, 3)``、``uint8`` RGB 影像。

        Returns:
            shape ``(H, W)``、``int32`` 實例遮罩。
            背景=0, 細胞ID=1..N。
        """
        masks, _, _ = self.model.eval(
            image,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            channels=[0, 0],
        )
        return masks.astype(np.int32)


# ------------------------------------------------------------------
# 分割入口
# ------------------------------------------------------------------

def segment_masked_dish(
    masked_dish_image: np.ndarray,
    segmenter: CellposeSegmenter,
    remove_border: bool = True,
) -> np.ndarray:
    """在遮罩後的 DISH 影像上執行 Cellpose 分割。

    Args:
        masked_dish_image: shape ``(H, W, 3)``、``uint8``。
            非 ROI 區域應為 ``background_fill_value`` (通常 0)。
        segmenter: 已初始化的 ``CellposeSegmenter``。
        remove_border: 是否移除碰觸邊界的細胞。

    Returns:
        shape ``(H, W)``、``int32`` 實例遮罩 (背景=0)。
    """
    instance_mask = segmenter.predict(masked_dish_image)

    if remove_border:
        instance_mask = _remove_border_cells(instance_mask)

    num_cells = len(np.unique(instance_mask)) - 1  # 扣除背景 0
    logger.info("Cellpose 分割完成: %d 個有效細胞", num_cells)
    return instance_mask


def _remove_border_cells(instance_mask: np.ndarray) -> np.ndarray:
    """移除碰觸影像邊界的細胞，並重新編號。

    Args:
        instance_mask: ``int32`` 實例遮罩。

    Returns:
        移除邊界細胞後的 ``int32`` 實例遮罩（ID 連續化）。
    """
    before_ids = set(np.unique(instance_mask)) - {0}

    binary = (instance_mask > 0).astype(np.uint8)
    cleared = clear_border(binary)
    keep_mask = cleared > 0

    cleaned = np.where(keep_mask, instance_mask, 0)

    after_ids = set(np.unique(cleaned)) - {0}
    removed_count = len(before_ids) - len(after_ids)
    if removed_count > 0:
        logger.info("移除 %d 個邊界細胞", removed_count)

    return _relabel_sequential(cleaned)


def _relabel_sequential(mask: np.ndarray) -> np.ndarray:
    """將非連續 ID 重新標記為 1..N 連續整數。"""
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]

    relabeled = np.zeros_like(mask, dtype=np.int32)
    for new_id, old_id in enumerate(sorted(unique_ids), start=1):
        relabeled[mask == old_id] = new_id

    return relabeled
