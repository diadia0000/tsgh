"""
UNet++ 推論 + 細胞核心萃取 — hybrid pipeline 專用模組

整合功能:
  1. UNetPPInference: UNet++ 細胞膜語義分割推論器
     (FP16 + cudnn.benchmark + inference_mode 加速)
  2. extract_cell_cores: 從膜遮罩萃取被包圍的細胞內部核心

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
from scipy import ndimage
from skimage import io, measure

logger = logging.getLogger(__name__)


# =====================================================================
# Part 1: UNet++ 推論器
# =====================================================================

class UNetPPInference:
    """
    UNet++ 推論器

    封裝模型載入、前處理、後處理和推論邏輯。
    支援 FP16 半精度 + cuDNN benchmark + torch.inference_mode 加速。

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
            image_size: 輸入影像尺寸 (H, W)
            device: 計算設備 (None 則自動偵測)
        """
        self.image_size = image_size
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._use_fp16 = (self.device.type == "cuda")

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

        # FP16 半精度加速 (僅 GPU)
        if self._use_fp16:
            self.model = self.model.half()
            logger.info("已啟用 FP16 半精度推論")

        # 建立前處理轉換
        self.transform = self._build_transform()

        logger.info("模型載入完成，設備: %s", self.device)

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
                value=255,      # 白色填充 (背景)
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

    def predict_proba(
        self,
        output: torch.Tensor,
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        取得機率預測 (Softmax) → (H, W, C)
        """
        proba = torch.softmax(output, dim=1).squeeze(0)
        proba = proba.permute(1, 2, 0).cpu().numpy()
        h, w = original_size
        return proba[:h, :w, :]

    # ----------------------------------------------------------
    # 推論
    # ----------------------------------------------------------

    @torch.inference_mode()
    def predict_single(
        self,
        image: Union[np.ndarray, Path, str],
        return_proba: bool = False,
        overlap: int = 128,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        單張影像推論 (FP16 + inference_mode 加速)

        當影像尺寸大於 ``image_size`` 時自動啟用滑動視窗推論，
        對重疊區域以機率平均後再 argmax，確保接縫無痕。

        Args:
            image: 輸入影像 (ndarray 或 路徑)
            return_proba: 是否回傳機率圖
            overlap: 滑動視窗重疊像素 (僅大圖時生效)

        Returns:
            mask: 二值 Mask (H, W)
            proba: 機率圖 (H, W, C)，僅當 return_proba=True
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

        # 大圖 → 滑動視窗
        if h > win_h or w > win_w:
            return self._predict_sliding_window(
                image, overlap=overlap, return_proba=return_proba
            )

        # 小圖 → 直接推論
        return self._predict_direct(image, return_proba=return_proba)

    def _predict_direct(
        self,
        image: np.ndarray,
        return_proba: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """直接推論 (影像 ≤ image_size)。"""
        tensor, original_size = self.preprocess(image)
        tensor = tensor.to(self.device)

        if self._use_fp16:
            tensor = tensor.half()

        output = self.model(tensor)

        if self._use_fp16:
            output = output.float()

        mask = self.postprocess(output, original_size)

        if return_proba:
            proba = self.predict_proba(output, original_size)
            return mask, proba

        return mask

    # ----------------------------------------------------------
    # 滑動視窗推論
    # ----------------------------------------------------------

    def _predict_sliding_window(
        self,
        image: np.ndarray,
        overlap: int = 128,
        return_proba: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """滑動視窗推論，重疊區域以機率平均融合。

        Args:
            image: shape ``(H, W, 3)``、``uint8`` RGB 影像。
            overlap: 相鄰視窗重疊像素數。
            return_proba: 是否回傳機率圖。

        Returns:
            mask ``(H, W)`` 或 ``(mask, proba)``。
        """
        h, w = image.shape[:2]
        win_h, win_w = self.image_size
        stride_h = max(1, win_h - overlap)
        stride_w = max(1, win_w - overlap)

        windows = self._generate_window_coords(h, w, win_h, win_w, stride_h, stride_w)

        logger.info(
            "滑動視窗推論: 影像 (%d, %d), 視窗 (%d, %d), "
            "overlap=%d, 共 %d 個視窗",
            h, w, win_h, win_w, overlap, len(windows),
        )

        # 機率累加緩衝 (float64 避免精度損失)
        num_classes = self.model.classes if hasattr(self.model, 'classes') else 2
        proba_sum = np.zeros((h, w, num_classes), dtype=np.float64)
        count_map = np.zeros((h, w), dtype=np.float64)

        for y0, x0, y1, x1 in windows:
            patch = image[y0:y1, x0:x1]
            _, patch_proba = self._predict_direct(patch, return_proba=True)
            proba_sum[y0:y1, x0:x1] += patch_proba.astype(np.float64)
            count_map[y0:y1, x0:x1] += 1.0

        # 平均融合
        count_map = np.maximum(count_map, 1.0)
        proba_avg = proba_sum / count_map[:, :, np.newaxis]
        mask = proba_avg.argmax(axis=2).astype(np.uint8)

        if return_proba:
            return mask, proba_avg.astype(np.float32)
        return mask

    @staticmethod
    def _generate_window_coords(
        img_h: int,
        img_w: int,
        win_h: int,
        win_w: int,
        stride_h: int,
        stride_w: int,
    ) -> List[Tuple[int, int, int, int]]:
        """產生滑動視窗的 (y0, x0, y1, x1) 座標列表。

        確保最後一列/行的視窗貼齊影像右/下邊界，
        不會漏掉任何像素。
        """
        coords: List[Tuple[int, int, int, int]] = []

        y_starts = list(range(0, img_h - win_h + 1, stride_h))
        if y_starts[-1] + win_h < img_h:
            y_starts.append(img_h - win_h)

        x_starts = list(range(0, img_w - win_w + 1, stride_w))
        if x_starts[-1] + win_w < img_w:
            x_starts.append(img_w - win_w)

        for y0 in y_starts:
            for x0 in x_starts:
                coords.append((y0, x0, y0 + win_h, x0 + win_w))

        return coords

    @torch.inference_mode()
    def predict_batch(
        self,
        image_paths: List[Path],
        output_dir: Path,
        save_proba: bool = False,
        save_overlay: bool = True,
        overlay_alpha: float = 0.5,
    ) -> List[Path]:
        """
        批次影像推論

        Args:
            image_paths: 影像路徑列表
            output_dir: 輸出目錄
            save_proba: 是否儲存機率熱圖
            save_overlay: 是否儲存疊加圖
            overlay_alpha: 疊加透明度

        Returns:
            saved_paths: 儲存的 Mask 路徑列表
        """
        from tqdm import tqdm  # noqa: WPS433

        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[Path] = []

        for img_path in tqdm(image_paths, desc="推論中"):
            if save_proba:
                mask, proba = self.predict_single(img_path, return_proba=True)
            else:
                mask = self.predict_single(img_path)

            mask_name = f"{img_path.stem}_pred.png"
            mask_path = output_dir / mask_name
            io.imsave(str(mask_path), (mask * 255).astype(np.uint8))
            saved_paths.append(mask_path)

            if save_proba:
                proba_name = f"{img_path.stem}_proba.png"
                proba_path = output_dir / proba_name
                membrane_proba = (proba[:, :, 1] * 255).astype(np.uint8)
                io.imsave(str(proba_path), membrane_proba)

            if save_overlay:
                original = io.imread(str(img_path))
                if original.ndim == 2:
                    original = np.stack([original] * 3, axis=-1)
                elif original.shape[2] == 4:
                    original = original[:, :, :3]
                overlay = self._create_overlay(original, mask, overlay_alpha)
                overlay_name = f"{img_path.stem}_overlay.png"
                overlay_path = output_dir / overlay_name
                io.imsave(str(overlay_path), overlay)

        return saved_paths

    @staticmethod
    def _create_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """建立膜區域紅色疊加視覺化圖。"""
        overlay = image.copy()
        membrane_mask = mask == 1
        overlay[membrane_mask] = (
            overlay[membrane_mask] * (1 - alpha)
            + np.array([255, 0, 0]) * alpha
        ).astype(np.uint8)
        return overlay


# =====================================================================
# Part 2: 細胞核心萃取
# =====================================================================

def extract_cell_cores(
    membrane_mask: np.ndarray,
    dilate_kernel_size: int = 7,
    close_kernel_size: int = 20,
    max_boundary_gap: int = 400,
) -> np.ndarray:
    """
    從細胞膜遮罩中萃取出被包圍的細胞內部 (Cell Cores)。
    支援封閉邊界缺口，可萃取邊緣細胞。

    演算法步驟：
    1. 形態學閉合 (Closing)：以較大 Kernel 將膜上斷裂的中小型缺口縫合起來。
    2. 膨脹 (Dilation)：將修補好的膜稍微加粗，確保絕對 watertight 防止填充外漏。
    3. 封閉邊界 (Close Edge Gaps)：尋找接觸影像邊界的短缺口並閉合。
    4. 填充 (Fill Holes)：把被膜包圍的內部孔洞填滿。
    5. 邏輯相減：填充區域 - 閉合後的細胞膜 = 內部核心。
       (使用閉合後的膜而非原始膜，避免 Closing 橋接像素殘留為白線)

    Args:
        membrane_mask: UNet++ 輸出的細胞膜二值化遮罩 (H, W)，值域 0 或 1。
        dilate_kernel_size: 膨脹核大小，確保水密性。
        close_kernel_size: 閉合核大小，縫合破裂的膜。
        max_boundary_gap: 允許閉合的邊界最大缺口長度 (pixels)。

    Returns:
        乾淨的細胞內部核心二值化遮罩 (H, W)，值域 {0, 1}。
    """
    membrane_uint8 = (membrane_mask > 0).astype(np.uint8)

    # 1. 形態學閉合：縫合斷裂缺口
    if close_kernel_size > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel_size, close_kernel_size),
        )
        closed_membrane = cv2.morphologyEx(
            membrane_uint8, cv2.MORPH_CLOSE, close_kernel
        )
    else:
        closed_membrane = membrane_uint8.copy()

    # 2. 膨脹細胞膜
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilate_kernel_size, dilate_kernel_size),
    )
    dilated_membrane = cv2.dilate(closed_membrane, dilate_kernel, iterations=1)

    # 3. 封閉邊緣缺口 (針對邊緣細胞)
    boundary_mask = np.zeros_like(dilated_membrane)
    boundary_mask[0, :] = 1
    boundary_mask[-1, :] = 1
    boundary_mask[:, 0] = 1
    boundary_mask[:, -1] = 1

    zero_boundary = np.logical_and(boundary_mask == 1, dilated_membrane == 0)
    labeled_boundary, num = measure.label(
        zero_boundary, connectivity=2, return_num=True
    )

    for i in range(1, num + 1):
        gap_mask = labeled_boundary == i
        if np.sum(gap_mask) <= max_boundary_gap:
            dilated_membrane[gap_mask] = 1

    # 4. 拓樸孔洞填充
    filled_mask = ndimage.binary_fill_holes(dilated_membrane).astype(np.uint8)

    # 5. 移除膨脹造成的外部光暈 (Halo)
    exterior_background = (filled_mask == 0).astype(np.uint8)
    restored_exterior = cv2.dilate(
        exterior_background, dilate_kernel, iterations=1
    )

    true_core_region = (restored_exterior == 0).astype(np.uint8)

    # 邏輯相減：使用 closed_membrane (非原始 membrane_uint8)
    # 閉合產生的橋接像素也必須被排除，否則會殘留白線
    core_mask = np.logical_and(
        true_core_region, np.logical_not(closed_membrane)
    ).astype(np.uint8)

    return core_mask
