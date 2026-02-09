"""
UNet++ HER2 細胞膜語義分割推論腳本

載入訓練完成的模型進行單張或批次影像推論

Author: TSGH AI Team
Date: 2026-02-05
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io
from tqdm import tqdm

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """
    載入配置檔案
    
    Returns:
        Config: 配置物件
    
    Raises:
        ImportError: 若 config.py 不存在
    """
    try:
        from config import config
        return config
    except ImportError:
        raise ImportError(
            "找不到 config.py！\n"
            "請複製 config_example.py 為 config.py 並設定參數"
        )


class UNetPPInference:
    """
    UNet++ 推論器
    
    封裝模型載入、前處理、後處理和推論邏輯
    
    Attributes:
        model: UNet++ 模型
        device: 計算設備
        image_size: 輸入影像尺寸
        transform: 前處理轉換
    """
    
    def __init__(
        self,
        model_path: Path,
        encoder_name: str = "efficientnet-b0",
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
        
        logger.info(f"模型載入完成，設備: {self.device}")
    
    def _load_weights(self, model_path: Path) -> None:
        """
        載入模型權重
        
        Args:
            model_path: 權重檔案路徑
        """
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型權重: {model_path}")
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # 處理 checkpoint 格式 (包含 optimizer 等其他資訊)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        self.model.load_state_dict(state_dict)
        logger.info(f"成功載入權重: {model_path}")
    
    def _build_transform(self) -> A.Compose:
        """
        建立推論時的前處理轉換
        
        Returns:
            Albumentations Compose 物件
        """
        return A.Compose([
            # 確保最小尺寸 (處理邊緣 tile)
            A.PadIfNeeded(
                min_height=self.image_size[0],
                min_width=self.image_size[1],
                border_mode=0,  # cv2.BORDER_CONSTANT
                value=255,  # 白色填充 (背景)
            ),
            # ImageNet 正規化
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        影像前處理
        
        Args:
            image: 輸入影像 (H, W, C) RGB 格式
            
        Returns:
            tensor: 處理後的 Tensor (1, C, H, W)
            original_size: 原始影像尺寸 (H, W)
        """
        original_size = image.shape[:2]
        
        # 確保 RGB 格式
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # 應用轉換
        transformed = self.transform(image=image)
        tensor = transformed['image'].unsqueeze(0)  # 增加 batch 維度
        
        return tensor, original_size
    
    def postprocess(
        self,
        output: torch.Tensor,
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        模型輸出後處理
        
        Args:
            output: 模型輸出 (B, C, H, W)
            original_size: 原始影像尺寸 (H, W)
            
        Returns:
            mask: 二值 Mask (H, W)，值為 0 或 1
        """
        # 取得預測類別
        pred = output.argmax(dim=1).squeeze(0)  # (H, W)
        mask = pred.cpu().numpy().astype(np.uint8)
        
        # 裁剪回原始尺寸
        h, w = original_size
        mask = mask[:h, :w]
        
        return mask
    
    def predict_proba(
        self,
        output: torch.Tensor,
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        取得機率預測 (Softmax)
        
        Args:
            output: 模型輸出 (B, C, H, W)
            original_size: 原始影像尺寸 (H, W)
            
        Returns:
            proba: 機率圖 (H, W, C)
        """
        proba = torch.softmax(output, dim=1).squeeze(0)  # (C, H, W)
        proba = proba.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        
        # 裁剪回原始尺寸
        h, w = original_size
        proba = proba[:h, :w, :]
        
        return proba
    
    @torch.no_grad()
    def predict_single(
        self,
        image: Union[np.ndarray, Path, str],
        return_proba: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        單張影像推論
        
        Args:
            image: 輸入影像 (ndarray 或 路徑)
            return_proba: 是否回傳機率圖
            
        Returns:
            mask: 二值 Mask (H, W)
            proba: 機率圖 (H, W, C)，僅當 return_proba=True
        """
        # 讀取影像
        if isinstance(image, (str, Path)):
            image = io.imread(str(image))
        
        # 前處理
        tensor, original_size = self.preprocess(image)
        tensor = tensor.to(self.device)
        
        # 推論
        output = self.model(tensor)
        
        # 後處理
        mask = self.postprocess(output, original_size)
        
        if return_proba:
            proba = self.predict_proba(output, original_size)
            return mask, proba
        
        return mask
    
    @torch.no_grad()
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
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        
        for img_path in tqdm(image_paths, desc="推論中"):
            # 推論
            if save_proba:
                mask, proba = self.predict_single(img_path, return_proba=True)
            else:
                mask = self.predict_single(img_path)
            
            # 儲存 Mask
            mask_name = f"{img_path.stem}_pred.png"
            mask_path = output_dir / mask_name
            io.imsave(str(mask_path), (mask * 255).astype(np.uint8))
            saved_paths.append(mask_path)
            
            # 儲存機率熱圖
            if save_proba:
                proba_name = f"{img_path.stem}_proba.png"
                proba_path = output_dir / proba_name
                # 取膜類別機率 (class 1)
                membrane_proba = (proba[:, :, 1] * 255).astype(np.uint8)
                io.imsave(str(proba_path), membrane_proba)
            
            # 儲存疊加圖
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
    
    def _create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        建立疊加視覺化圖
        
        Args:
            image: 原始影像 (H, W, 3)
            mask: 預測 Mask (H, W)
            alpha: 疊加透明度
            
        Returns:
            overlay: 疊加圖 (H, W, 3)
        """
        overlay = image.copy()
        
        # 膜區域標記為紅色
        membrane_mask = mask == 1
        overlay[membrane_mask] = (
            overlay[membrane_mask] * (1 - alpha) + 
            np.array([255, 0, 0]) * alpha
        ).astype(np.uint8)
        
        return overlay


def collect_images(
    input_path: Path,
    extensions: List[str] = None,
) -> List[Path]:
    """
    收集影像路徑
    
    Args:
        input_path: 輸入路徑 (檔案或目錄)
        extensions: 支援的副檔名
        
    Returns:
        image_paths: 影像路徑列表
    """
    if extensions is None:
        extensions = ['.tiff', '.tif', '.png', '.jpg', '.jpeg']
    
    if input_path.is_file():
        return [input_path]
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_path.glob(f'*{ext}'))
        image_paths.extend(input_path.glob(f'*{ext.upper()}'))
    
    return sorted(set(image_paths))


def main() -> None:
    """主程式入口"""
    config = load_config()
    
    # ========== 設定推論參數 ==========
    # 模型路徑
    model_path = config.model_save_dir / "best_model.pth"
    
    # 輸入路徑 (可以是單張影像或目錄)
    input_path = config.base_dir / "train" / "test"  # 測試影像目錄
    
    # 輸出目錄
    output_dir = config.base_dir / "output" / "inference_results"
    
    # ========== 執行推論 ==========
    logger.info("=" * 60)
    logger.info("UNet++ HER2 細胞膜分割推論")
    logger.info("=" * 60)
    logger.info(f"模型路徑: {model_path}")
    logger.info(f"輸入路徑: {input_path}")
    logger.info(f"輸出目錄: {output_dir}")
    logger.info("=" * 60)
    
    # 建立推論器
    inferencer = UNetPPInference(
        model_path=model_path,
        encoder_name=config.encoder_name,
        num_classes=config.num_classes,
        image_size=config.image_size,
    )
    
    # 收集影像
    image_paths = collect_images(input_path)
    logger.info(f"找到 {len(image_paths)} 張影像")
    
    if not image_paths:
        logger.warning("沒有找到任何影像！")
        return
    
    # 批次推論
    saved_paths = inferencer.predict_batch(
        image_paths=image_paths,
        output_dir=output_dir,
        save_proba=True,
        save_overlay=True,
        overlay_alpha=0.5,
    )
    
    logger.info("=" * 60)
    logger.info(f"推論完成！共處理 {len(saved_paths)} 張影像")
    logger.info(f"結果已儲存至: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
