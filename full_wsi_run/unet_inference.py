"""
UNet++ 推論 + 膜預測後處理 — hybrid pipeline 專用模組

整合功能:
  1. UNetPPInference: UNet++ 細胞膜語義分割推論器
        (FP32 + cudnn.benchmark + inference_mode 加速)
  2. postprocess_membrane_mask: 輕量後處理（形態學閉合）

本模組為 hybrid 資料夾的在地版本，不依賴 unet_mask 目錄。

Author: TSGH AI Team
Date: 2026-03-06
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io

logger = logging.getLogger(__name__)


# =====================================================================
# Part 1: UNet++ 推論器
# =====================================================================

class UNetPPInference:
    """
    UNet++ 推論器

    封裝模型載入、前處理、後處理和推論邏輯。
    支援 FP32 + cuDNN benchmark + torch.inference_mode 加速。

    Attributes:
        model: UNet++ 模型
        device: 計算設備
        image_size: 輸入影像尺寸
        transform: 前處理轉換
    """

    def __init__(
        self,
        model_path: Path,
        encoder_name: str = "efficientnet-b4",
        num_classes: int = 2,
        image_size: Tuple[int, int] = (1024, 1024),
        device: Optional[torch.device] = None,
    ) -> None:
        """
        初始化推論器

        Args:
            model_path: 模型權重路徑 (.pth)
            encoder_name: 編碼器名稱
            num_classes: 類別數量
            image_size: 輸入影像尺寸 (H, W)，最小會被限制為 1024x1024
            device: 計算設備 (None 則自動偵測)
        """
        self.image_size = self._sanitize_window_size(image_size)
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # 啟用 cuDNN 自動調優 (首次推論稍慢，後續推論加速)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # 建立模型
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,  # 推論時不需要 ImageNet 權重
            in_channels=3,
            classes=num_classes,
        )

        # 載入訓練好的權重
        self._load_weights(model_path)

        self.model = self.model.to(self.device)
        self.model.eval()

        # 建立前處理轉換
        self.transform = self._build_transform()

        logger.info("模型載入完成，設備: %s", self.device)

    @staticmethod
    def _sanitize_window_size(image_size: Tuple[int, int]) -> Tuple[int, int]:
        """強制最小輸入尺寸為 1024x1024。"""
        raw_h = max(1, int(image_size[0]))
        raw_w = max(1, int(image_size[1]))
        win_h = max(raw_h, 1024)
        win_w = max(raw_w, 1024)

        if (win_h, win_w) != (raw_h, raw_w):
            logger.warning(
                "image_size=%s 低於最小視窗，已自動調整為 (%d, %d)",
                image_size,
                win_h,
                win_w,
            )

        return (win_h, win_w)

    # ----------------------------------------------------------
    # 內部方法
    # ----------------------------------------------------------

    def _load_weights(self, model_path: Path) -> None:
        """載入模型權重。"""
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型權重: {model_path}")

        state_dict = torch.load(
            model_path, map_location=self.device, weights_only=True
        )

        # 處理 checkpoint 格式 (包含 optimizer 等其他資訊)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        self.model.load_state_dict(state_dict)
        logger.info("成功載入權重: %s", model_path)

    def _build_transform(self) -> A.Compose:
        """建立推論時的前處理轉換。"""
        return A.Compose([
            A.PadIfNeeded(
                min_height=self.image_size[0],
                min_width=self.image_size[1],
                border_mode=0,  # cv2.BORDER_CONSTANT
                fill=255,       # 白色填充 (背景)
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    # ----------------------------------------------------------
    # 前 / 後處理
    # ----------------------------------------------------------

    def preprocess(
        self, image: np.ndarray,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        影像前處理

        Args:
            image: 輸入影像 (H, W, C) RGB 格式

        Returns:
            tensor: 處理後的 Tensor (1, C, H, W)
            original_size: 原始影像尺寸 (H, W)
        """
        original_size = image.shape[:2]

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        transformed = self.transform(image=image)
        tensor = transformed["image"].unsqueeze(0)
        return tensor, original_size

    def postprocess(
        self,
        output: torch.Tensor,
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        模型輸出後處理 → 二值 Mask (H, W)，值為 0 或 1
        """
        pred = output.argmax(dim=1).squeeze(0)
        mask = pred.cpu().numpy().astype(np.uint8)
        h, w = original_size
        return mask[:h, :w]

    # ----------------------------------------------------------
    # 推論
    # ----------------------------------------------------------

    @torch.inference_mode()
    def predict_single(
        self,
        image: Union[np.ndarray, Path, str],
    ) -> np.ndarray:
        """
        單張影像推論 (FP32 + inference_mode)

        Args:
            image: 輸入影像 (ndarray 或 路徑)

        Returns:
            mask: 二值 Mask (H, W)
        """
        if isinstance(image, (str, Path)):
            image = io.imread(str(image))

        # 確保 RGB 3 通道
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        h, w = image.shape[:2]
        win_h, win_w = self.image_size

        if h > win_h or w > win_w:
            raise ValueError(
                f"image shape {image.shape[:2]} exceeds image_size "
                f"{self.image_size}; sliding-window inference disabled"
            )

        # 直接推論
        return self._predict_direct(image)

    def _predict_direct(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """直接推論 (影像 ≤ image_size)。"""
        tensor, original_size = self.preprocess(image)
        tensor = tensor.to(self.device)

        output = self.model(tensor)

        return self.postprocess(output, original_size)

    @torch.inference_mode()
    def predict_batch_arrays(
        self,
        images: List[np.ndarray],
    ) -> List[np.ndarray]:
        """批次推論一組 ndarray patch，回傳對應的 argmax mask 列表。

        所有 patch 會 padding 至 ``self.image_size``，組成單一 batch tensor，
        一次送進模型，再依各自原始尺寸切回。每張 patch 的 H/W 必須 ≤
        ``self.image_size``（pipeline 端已保證這點）。
        """
        if not images:
            return []

        win_h, win_w = self.image_size
        tensors: List[torch.Tensor] = []
        original_sizes: List[Tuple[int, int]] = []
        for img in images:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            if img.shape[0] > win_h or img.shape[1] > win_w:
                raise ValueError(
                    f"patch shape {img.shape[:2]} exceeds image_size "
                    f"{self.image_size}; sliding-window not supported in batch path"
                )
            tensor, original_size = self.preprocess(img)
            tensors.append(tensor)
            original_sizes.append(original_size)

        batch = torch.cat(tensors, dim=0).to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            output = self.model(batch)  # (B, C, H, W)
        pred = output.argmax(dim=1).cpu().numpy().astype(np.uint8)

        masks: List[np.ndarray] = []
        for i, (h, w) in enumerate(original_sizes):
            masks.append(pred[i, :h, :w])
        return masks



# =====================================================================
# Part 2: 膜預測後處理
# =====================================================================

def postprocess_membrane_mask(
    raw_mask: np.ndarray,
    close_kernel_size: int = 7,
    min_area: int = 550,
) -> np.ndarray:
    """對 UNet++ 的原始膜預測做輕量後處理。

    模型已直接輸出 filled blob（整塊棕色細胞膜區域），
    僅需連接微小斷裂，並移除面積過小的碎片。

    Args:
        raw_mask: UNet++ 輸出的二值 mask (H, W)，值域 {0, 1}。
        close_kernel_size: 閉合核大小 (pixels)，越大可跨越越寬的斷裂。
        min_area: 最小連通區域面積 (pixels)，低於此值的碎片會被移除。

    Returns:
        後處理後的二值 mask (H, W)，uint8 {0, 1}。
    """
    mask = (raw_mask > 0).astype(np.uint8)

    if close_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel_size, close_kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8,
        )
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                mask[labels == i] = 0

    return mask
