"""
M2: Cellpose 實例分割模組

在 IHC-DISH 50/50 alpha blending 疊合影像上執行 Cellpose 推論，
產出 ``cell_instance_mask`` (背景=0, 細胞ID=1..N)。

模型已在醫師標註的 IHC-DISH 疊合影像上重新訓練。
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

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
        )
        return masks.astype(np.int32)

    def predict_batch(
        self,
        images: list,
        batch_size: int = 8,
    ) -> list:
        """批次執行多張影像的實例分割（真批次：np.stack → ndarray）。

        Cellpose v4 的 ``eval(list)`` 走 serial per-image for-loop
        (``models.py:221``)；只有當輸入是 ``ndim==4`` 的 ndarray 時才會
        進入真批次的 ``_run_net`` (``models.py:267+``)。
        因此這裡先 ``np.stack`` 成 ``(B, H, W, 3)``。邊緣 window 比
        ``window_size`` 小時，以白色 (255) pad 到 batch 內最大尺寸；
        取回 mask 後再 crop 回原始大小。

        Fill 必須是白色而非黑色——Cellpose 會以 1st/99th percentile 做
        per-image normalization，黑色 pad 會壓低 1st percentile、扭曲
        正常組織區的歸一化結果。

        Args:
            images: list of ``(H, W, 3)`` uint8 RGB 影像。
            batch_size: cellpose 內部 tile batch 大小。

        Returns:
            list of ``int32`` 實例遮罩，順序與輸入對齊。
        """
        if not images:
            return []

        # 單張時 _compute_masks 回傳 2D (H, W)，不是 3D；走 single-image
        # 路徑可避免 3D 索引邏輯，與舊版 predict() 行為一致。
        if len(images) == 1:
            masks, _, _ = self.model.eval(
                images[0],
                diameter=self.diameter,
                flow_threshold=self.flow_threshold,
                cellprob_threshold=self.cellprob_threshold,
                batch_size=batch_size,
            )
            return [np.asarray(masks).astype(np.int32)]

        original_shapes = [(img.shape[0], img.shape[1]) for img in images]
        target_h = max(h for h, _ in original_shapes)
        target_w = max(w for _, w in original_shapes)

        padded: list = []
        for img, (h, w) in zip(images, original_shapes):
            if h == target_h and w == target_w:
                padded.append(img)
            else:
                canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
                canvas[:h, :w] = img
                padded.append(canvas)

        # (B, H, W, 3) ndarray → cellpose transforms.convert_image ndim==4 batch path
        stacked = np.stack(padded, axis=0)

        masks, _, _ = self.model.eval(
            stacked,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            batch_size=batch_size,
        )
        # nimg > 1 時 cellpose 回傳 (B, H, W) 的 ndarray 或長度 B 的 list；
        # 兩種型態都用 index 取出後 crop 回原本大小即可。
        return [
            np.asarray(masks[i])[:h, :w].astype(np.int32)
            for i, (h, w) in enumerate(original_shapes)
        ]
