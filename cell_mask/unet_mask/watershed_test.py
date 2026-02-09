"""
Watershed 細胞分割測試腳本 (使用 UNet++ 模型推論)

流程:
1. 讀取 HER2 原圖
2. 使用訓練好的 UNet++ 模型推論細胞膜 Mask (支援批次推論)
3. 用 HED H 通道偵測藍色細胞核
4. Marker-Controlled Watershed 分割

輸入: train/test 目錄的測試圖
輸出: 分割結果視覺化

Author: TSGH AI Team
Date: 2026-02-09
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
import torch
from skimage import io
from skimage.color import rgb2hed
from skimage.measure import label
from skimage.morphology import remove_small_objects, disk, binary_opening
from skimage.segmentation import watershed

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


# ============================================================
# UNet++ 推論器 (支援批次推論)
# ============================================================
class UNetPPMembraneDetector:
    """
    UNet++ 細胞膜偵測器
    
    封裝模型載入與推論邏輯，支援批次推論
    
    Attributes:
        model: UNet++ 模型
        device: 計算設備
        image_size: 輸入影像尺寸
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
        import segmentation_models_pytorch as smp
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        self.image_size = image_size
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() 
            else torch.device("cpu")
        )
        
        # 建立模型
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
        )
        
        # 載入訓練好的權重
        self._load_weights(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 建立前處理轉換
        self.transform = A.Compose([
            A.PadIfNeeded(
                min_height=self.image_size[0],
                min_width=self.image_size[1],
                border_mode=0,
                value=255,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
        
        logger.info(f"UNet++ 模型載入完成，設備: {self.device}")
    
    def _load_weights(self, model_path: Path) -> None:
        """載入模型權重"""
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型權重: {model_path}")
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        self.model.load_state_dict(state_dict)
        logger.info(f"成功載入權重: {model_path}")
    
    def _preprocess_single(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        單張影像前處理
        
        Args:
            image: 輸入影像 (H, W, 3)
            
        Returns:
            tensor: 處理後的 Tensor (C, H, W)
            original_size: 原始影像尺寸 (H, W)
        """
        original_size = image.shape[:2]
        
        # 確保 RGB 格式
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        transformed = self.transform(image=image)
        return transformed['image'], original_size
    
    @torch.no_grad()
    def predict_single(self, image: np.ndarray) -> np.ndarray:
        """
        單張影像推論
        
        Args:
            image: 輸入影像 (H, W, 3) RGB 格式
            
        Returns:
            membrane_mask: 二值 Mask (H, W)，bool 類型
        """
        original_size = image.shape[:2]
        
        # 確保 RGB 格式
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # 前處理
        transformed = self.transform(image=image)
        tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # 推論
        output = self.model(tensor)
        
        # 後處理
        pred = output.argmax(dim=1).squeeze(0)
        mask = pred.cpu().numpy().astype(np.uint8)
        
        # 裁剪回原始尺寸
        h, w = original_size
        mask = mask[:h, :w]
        
        return mask.astype(bool)
    
    @torch.no_grad()
    def predict_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        批次影像推論
        
        Args:
            images: 輸入影像列表，每張 (H, W, 3) RGB 格式
            
        Returns:
            masks: 二值 Mask 列表，每張 (H, W)，bool 類型
        """
        if not images:
            return []
        
        # 前處理所有影像
        tensors = []
        original_sizes = []
        
        for image in images:
            tensor, orig_size = self._preprocess_single(image)
            tensors.append(tensor)
            original_sizes.append(orig_size)
        
        # 堆疊成批次
        batch_tensor = torch.stack(tensors, dim=0).to(self.device)
        
        # 批次推論
        outputs = self.model(batch_tensor)
        
        # 後處理每張影像
        masks = []
        predictions = outputs.argmax(dim=1).cpu().numpy().astype(np.uint8)
        
        for i, (pred, orig_size) in enumerate(zip(predictions, original_sizes)):
            h, w = orig_size
            mask = pred[:h, :w]
            masks.append(mask.astype(bool))
        
        return masks


# ============================================================
# 全域模型實例 (延遲初始化)
# ============================================================
_membrane_detector: Optional[UNetPPMembraneDetector] = None


def get_membrane_detector(config) -> UNetPPMembraneDetector:
    """
    取得或初始化細胞膜偵測器 (Singleton 模式)
    
    Args:
        config: 配置物件
    
    Returns:
        UNetPPMembraneDetector 實例
    """
    global _membrane_detector
    if _membrane_detector is None:
        model_path = config.model_save_dir / "best_model.pth"
        _membrane_detector = UNetPPMembraneDetector(
            model_path=model_path,
            encoder_name=config.encoder_name,
            num_classes=config.num_classes,
            image_size=config.image_size,
        )
    return _membrane_detector


def extract_nucleus_mask(
    image: np.ndarray,
    min_nucleus_size: int = 50,
) -> np.ndarray:
    """
    從 HER2 影像提取藍色細胞核
    
    使用 HED 色彩分離的 Hematoxylin 通道 + Otsu 自動閾值
    
    Args:
        image: RGB 影像 (H, W, 3)
        min_nucleus_size: 最小細胞核大小 (像素)
        
    Returns:
        nucleus_mask: 二值遮罩 (H, W), bool
    """
    # 轉換為 float
    if image.dtype == np.uint8:
        image_float = image.astype(np.float64) / 255.0
    else:
        image_float = image.astype(np.float64)
    
    # HED 色彩分離
    hed = rgb2hed(image_float)
    hematoxylin = hed[:, :, 0]  # H 通道 = 藍紫色
    
    # CLAHE 增強
    h_uint8 = cv2.normalize(hematoxylin, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    h_enhanced = clahe.apply(h_uint8)
    
    # Otsu 自動閾值
    _, nucleus_mask = cv2.threshold(h_enhanced, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nucleus_mask = nucleus_mask.astype(bool)
    
    # 形態學開運算 (去除小雜訊)
    nucleus_mask = binary_opening(nucleus_mask, disk(2))
    
    # 移除太小的區域
    nucleus_mask = remove_small_objects(nucleus_mask, min_size=min_nucleus_size)
    
    return nucleus_mask


def watershed_cell_segmentation(
    nucleus_mask: np.ndarray,
    membrane_mask: np.ndarray,
) -> np.ndarray:
    """
    使用 Marker-Controlled Watershed 分割細胞
    
    Args:
        nucleus_mask: 細胞核 mask (bool)
        membrane_mask: 細胞膜 mask (bool)
    
    Returns:
        cell_labels: 每個像素標記屬於哪個細胞 (0=邊界/膜)
    """
    # 1. 標記細胞核 (每個核一個 ID)
    markers = label(nucleus_mask)
    num_nuclei = markers.max()
    logger.info(f"  偵測到 {num_nuclei} 個細胞核")
    
    # 2. 計算距離圖 (從膜越遠的地方值越高)
    non_membrane = (~membrane_mask).astype(np.uint8)
    distance = cv2.distanceTransform(non_membrane, cv2.DIST_L2, 5)
    
    # 3. 反轉距離圖作為地形 (膜=山脊, 遠離膜=谷底)
    elevation = -distance
    
    # 4. Watershed 分割
    cell_labels = watershed(elevation, markers)
    
    # 5. 將膜區域標記為邊界 (0)
    cell_labels[membrane_mask] = 0
    
    return cell_labels


def create_visualization(
    image: np.ndarray,
    membrane_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    cell_labels: np.ndarray,
    cell_boundary_overlap_ratio: float = 0.87,
    membrane_dilation_radius: int = 3,
) -> np.ndarray:
    """
    建立視覺化結果
    
    六宮格:
    - 左上: 原圖
    - 中上: UNet++ 細胞膜預測
    - 右上: 細胞核 Mask
    - 左下: Watershed 結果
    - 中下: 細胞邊界疊加
    - 右下: 最終結果
    """
    h, w = image.shape[:2]
    
    # 確保 uint8
    if image.dtype != np.uint8:
        img_display = (image * 255).astype(np.uint8)
    else:
        img_display = image.copy()
    
    # 1. 原圖
    panel1 = img_display.copy()
    
    # 2. UNet++ 細胞膜 Mask (紅色)
    panel2 = img_display.copy()
    panel2[membrane_mask] = [255, 0, 0]
    
    # 3. 細胞核 Mask (藍色)
    panel3 = img_display.copy()
    panel3[nucleus_mask] = [0, 0, 255]
    
    # 4. Watershed 結果 (隨機色彩)
    panel4 = np.zeros((h, w, 3), dtype=np.uint8)
    np.random.seed(42)
    colors = np.random.randint(50, 255, (cell_labels.max() + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # 背景/邊界為黑色
    for i in range(cell_labels.max() + 1):
        panel4[cell_labels == i] = colors[i]
    
    # 5. 細胞邊界疊加
    panel5 = img_display.copy()
    contours, _ = cv2.findContours(
        (cell_labels > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(panel5, contours, -1, (0, 255, 0), 2)
    
    # 6. 最終結果: 結合 Watershed 與 邊界檢查
    panel6 = np.zeros_like(img_display)
    valid_mask = np.zeros_like(membrane_mask, dtype=np.uint8)
    
    # 擴張膜 Mask 以便檢查邊界重疊
    membrane_dilated = cv2.dilate(
        membrane_mask.astype(np.uint8), 
        disk(membrane_dilation_radius)
    )
    
    for i in range(1, cell_labels.max() + 1):
        cell_mask = (cell_labels == i).astype(np.uint8)
        
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        boundary_mask = np.zeros_like(membrane_mask, dtype=np.uint8)
        cv2.drawContours(boundary_mask, contours, -1, 1, 1)
        
        total_boundary_pixels = np.sum(boundary_mask)
        if total_boundary_pixels == 0:
            continue
            
        overlap_pixels = np.sum(boundary_mask & membrane_dilated)
        overlap_ratio = overlap_pixels / total_boundary_pixels
        
        # 如果邊界與膜重疊超過閾值，表示細胞被膜包圍得很好
        if overlap_ratio > cell_boundary_overlap_ratio:
            valid_mask[cell_mask == 1] = 1
            
    # 細胞內部 = 有效區域 - 膜本身
    cell_interior = (valid_mask > 0) & (~membrane_mask)
    panel6[cell_interior] = img_display[cell_interior]
    
    # 組合六宮格
    top_row = np.hstack([panel1, panel2, panel3])
    bottom_row = np.hstack([panel4, panel5, panel6])
    vis = np.vstack([top_row, bottom_row])
    
    # 添加標籤
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [
        (10, 25, "Original"),
        (w + 10, 25, "UNet++ Membrane"),
        (2 * w + 10, 25, "Nucleus (Blue)"),
        (10, h + 25, "Watershed Labels"),
        (w + 10, h + 25, "Cell Contours"),
        (2 * w + 10, h + 25, "Final Result"),
    ]
    for x, y, text in labels:
        cv2.putText(vis, text, (x + 1, y + 1), font, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, text, (x, y), font, 0.6, (255, 255, 255), 2)
    
    return vis


def process_batch(
    image_paths: List[Path],
    images: List[np.ndarray],
    membrane_masks: List[np.ndarray],
    output_dir: Path,
    config,
) -> int:
    """
    處理一個批次的影像（後處理 + 儲存）
    
    Args:
        image_paths: 影像路徑列表
        images: 原始影像列表
        membrane_masks: 細胞膜 Mask 列表
        output_dir: 輸出目錄
        config: 配置物件
        
    Returns:
        成功處理的影像數量
    """
    success_count = 0
    
    for image_path, image, membrane_mask in zip(image_paths, images, membrane_masks):
        try:
            logger.info(f"  後處理: {image_path.name}")
            
            # 提取細胞核
            nucleus_mask = extract_nucleus_mask(
                image, 
                min_nucleus_size=config.min_nucleus_size,
            )
            
            # Watershed 分割
            cell_labels = watershed_cell_segmentation(nucleus_mask, membrane_mask)
            num_cells = cell_labels.max()
            logger.info(f"  分割出 {num_cells} 個細胞")
            
            # 視覺化
            vis = create_visualization(
                image, membrane_mask, nucleus_mask, cell_labels,
                cell_boundary_overlap_ratio=config.cell_boundary_overlap_ratio,
                membrane_dilation_radius=config.membrane_dilation_radius,
            )
            
            # 儲存結果
            vis_path = output_dir / f"{image_path.stem}_watershed_unetpp.png"
            io.imsave(str(vis_path), vis)
            logger.info(f"  已儲存: {vis_path.name}")
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"  處理失敗: {image_path.name} | {e}")
            import traceback
            traceback.print_exc()
    
    return success_count


def main() -> None:
    """主程式"""
    # 載入配置
    config = load_config()
    
    # 路徑設定
    input_dir = config.watershed_input_dir
    output_dir = config.watershed_output_dir
    model_path = config.model_save_dir / "best_model.pth"
    batch_size = config.inference_batch_size
    
    logger.info("=" * 60)
    logger.info("Watershed 細胞分割測試 (UNet++ 模型推論版)")
    logger.info("=" * 60)
    logger.info(f"輸入目錄: {input_dir}")
    logger.info(f"輸出目錄: {output_dir}")
    logger.info(f"模型路徑: {model_path}")
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"編碼器: {config.encoder_name}")
    logger.info(f"細胞核最小尺寸: {config.min_nucleus_size}")
    logger.info(f"邊界重疊閾值: {config.cell_boundary_overlap_ratio}")
    
    # 檢查模型是否存在
    if not model_path.exists():
        logger.error(f"找不到模型檔案: {model_path}")
        logger.error("請確認已完成模型訓練並儲存 best_model.pth")
        return
    
    # 建立輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集影像
    image_paths = []
    for ext in config.supported_extensions:
        image_paths.extend(input_dir.glob(f"*{ext}"))
        image_paths.extend(input_dir.glob(f"*{ext.upper()}"))
    image_paths = sorted(set(image_paths))
    
    logger.info(f"找到 {len(image_paths)} 張影像")
    
    if not image_paths:
        logger.warning("沒有找到任何影像！")
        return
    
    # 取得推論器
    detector = get_membrane_detector(config)
    
    # 批次處理
    success_count = 0
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        
        logger.info(f"\n批次 {batch_idx + 1}/{total_batches} ({len(batch_paths)} 張)")
        
        # 讀取影像
        images = []
        valid_paths = []
        for img_path in batch_paths:
            try:
                image = io.imread(str(img_path))
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.shape[2] == 4:
                    image = image[:, :, :3]
                images.append(image)
                valid_paths.append(img_path)
            except Exception as e:
                logger.error(f"讀取失敗: {img_path.name} | {e}")
        
        if not images:
            continue
        
        # 批次推論細胞膜
        logger.info(f"  UNet++ 批次推論 ({len(images)} 張)...")
        membrane_masks = detector.predict_batch(images)
        
        # 後處理每張影像
        success_count += process_batch(
            valid_paths, images, membrane_masks, output_dir, config
        )
    
    logger.info("\n" + "=" * 60)
    logger.info(f"完成! 成功處理 {success_count}/{len(image_paths)} 張影像")
    logger.info(f"結果已儲存至: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
