"""
推論模組：使用訓練好的 UNet 模型對大型影像進行滑動視窗推論
"""
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
from skimage import io
from torch.cuda.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import (
    MODEL_DIR, DEVICE, CROP_SIZE, ENCODER_NAME, 
    ENCODER_WEIGHTS, IN_CHANNELS, NUM_CLASSES,
    IMAGENET_MEAN, IMAGENET_STD
)


class TileInference:
    """滑動視窗推論類別"""
    
    def __init__(self, model_path, tile_size=512, overlap=128, device=DEVICE):
        """
        Args:
            model_path: 模型檔案路徑
            tile_size: 滑動視窗大小
            overlap: 重疊區域大小
            device: 運算裝置
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.device = device
        
        # 載入模型
        self.model = self._load_model(model_path)
        
        # 定義預處理轉換
        self.transform = A.Compose([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    
    def _load_model(self, model_path):
        """載入訓練好的模型"""
        model = smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES,
        )
        
        # 載入權重
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"✓ 模型載入成功: {model_path}")
        return model
    
    def _preprocess_tile(self, image_tile):
        """預處理單個 tile"""
        # 確保是 RGB 格式
        if image_tile.shape[2] == 4:  # RGBA
            image_tile = cv2.cvtColor(image_tile, cv2.COLOR_RGBA2RGB)
        
        # 應用轉換
        augmented = self.transform(image=image_tile)
        image_tensor = augmented['image'].unsqueeze(0)  # 加入 batch 維度
        
        return image_tensor
    
    def _extract_tiles(self, image):
        """
        提取所有需要推論的 tiles
        
        Args:
            image: numpy array, shape (H, W, C)
            
        Returns:
            tiles: list of (tile_image, x, y)
        """
        h, w = image.shape[:2]
        tiles = []
        
        for y in range(0, h - self.tile_size + 1, self.stride):
            for x in range(0, w - self.tile_size + 1, self.stride):
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                tiles.append((tile, x, y))
        
        return tiles
    
    def _merge_predictions(self, predictions, image_shape):
        """
        合併所有 tile 的預測結果
        
        Args:
            predictions: list of (pred_mask, x, y)
            image_shape: (H, W) 原始影像大小
            
        Returns:
            merged_mask: numpy array, shape (H, W)
        """
        h, w = image_shape
        
        # 建立權重地圖和累積預測
        weight_map = np.zeros((h, w), dtype=np.float32)
        accumulated = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
        
        # 建立高斯權重矩陣（中心權重高，邊緣權重低）
        tile_weight = self._create_tile_weights()
        
        for pred_mask, x, y in predictions:
            # pred_mask shape: (NUM_CLASSES, tile_size, tile_size)
            y_end = min(y + self.tile_size, h)
            x_end = min(x + self.tile_size, w)
            
            tile_h = y_end - y
            tile_w = x_end - x
            
            # 裁切權重以匹配 tile 大小
            weight = tile_weight[:tile_h, :tile_w]
            
            # 累積預測和權重
            accumulated[:, y:y_end, x:x_end] += pred_mask[:, :tile_h, :tile_w] * weight
            weight_map[y:y_end, x:x_end] += weight
        
        # 避免除以零
        weight_map = np.maximum(weight_map, 1e-8)
        
        # 正規化
        for c in range(NUM_CLASSES):
            accumulated[c] = accumulated[c] / weight_map
        
        # 取得最大機率的類別
        merged_mask = np.argmax(accumulated, axis=0).astype(np.uint8)
        
        return merged_mask
    
    def _create_tile_weights(self):
        """建立 tile 權重矩陣（高斯分布，中心權重高）"""
        center = self.tile_size // 2
        y, x = np.ogrid[:self.tile_size, :self.tile_size]
        
        # 高斯權重
        sigma = self.tile_size / 4
        weights = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
        
        return weights
    
    @torch.no_grad()
    def predict_image(self, image_path, save_path=None, save_class1_only=True, 
                     result_dir=None, visualize=True):
        """
        對大型影像進行推論
        
        Args:
            image_path: 輸入影像路徑
            save_path: 儲存預測結果的路徑（可選）
            save_class1_only: 是否儲存只有類別1的原始圖片（其他區域變白色）
            result_dir: 結果輸出目錄（預設為 process/result/）
            visualize: 是否視覺化結果
            
        Returns:
            prediction_mask: numpy array, shape (H, W)
            image: 原始影像
        """
        print(f"\n開始處理影像: {image_path}")
        
        # 讀取影像
        image = io.imread(image_path)
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        
        original_shape = image.shape[:2]
        print(f"影像大小: {original_shape}")
        
        # 提取 tiles
        tiles = self._extract_tiles(image)
        print(f"總共 {len(tiles)} 個 tiles")
        
        # 推論所有 tiles
        predictions = []
        for tile_img, x, y in tqdm(tiles, desc="推論中"):
            # 預處理
            tensor = self._preprocess_tile(tile_img).to(self.device)
            
            # 推論
            with autocast():
                output = self.model(tensor)
            
            # 轉換為機率 (softmax)
            pred_probs = torch.softmax(output, dim=1)
            pred_probs = pred_probs.cpu().numpy()[0]  # shape: (NUM_CLASSES, H, W)
            
            predictions.append((pred_probs, x, y))
        
        # 合併預測結果
        print("合併預測結果...")
        merged_mask = self._merge_predictions(predictions, original_shape)
        
        # 設定預設結果目錄
        if result_dir is None:
            result_dir = Path("process/result")
        else:
            result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 取得檔案名稱
        image_name = Path(image_path).stem
        
        # 儲存預測 mask
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), merged_mask)
            print(f"✓ 預測 mask 已儲存: {save_path}")
        
        # 儲存只保留類別1的原始圖片
        if save_class1_only:
            class1_image = self._extract_class1_region(image, merged_mask)
            class1_path = result_dir / f"{image_name}_class1.png"
            cv2.imwrite(str(class1_path), cv2.cvtColor(class1_image, cv2.COLOR_RGB2BGR))
            print(f"✓ 類別1區域圖片已儲存: {class1_path}")
        
        # 視覺化
        if visualize:
            self._visualize_result(image, merged_mask)
        
        return merged_mask, image
    
    def _extract_class1_region(self, image, mask):
        """
        提取細胞區域的原始圖片（類別1+類別2），其他區域設為白色
        
        註：由於 pseudo mask 生成策略，類別1只包含細胞膜內的空白區域，
        而類別2（細胞膜）包含了大部分細胞結構，因此輸出兩者合併的區域。
        
        Args:
            image: 原始影像 (H, W, 3)
            mask: 預測mask (H, W)
            
        Returns:
            result: 處理後的影像 (H, W, 3)
        """
        # 建立白色背景
        result = np.ones_like(image) * 255
        
        # 保留類別1（細胞內部）和類別2（細胞膜）的區域
        cell_mask = (mask == 1) | (mask == 2)
        result[cell_mask] = image[cell_mask]
        
        return result   
    def _visualize_result(self, image, mask):
        """視覺化原始影像和預測結果（3分類）"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 確保影像是 RGB 格式（3 通道）
        if len(image.shape) == 2:  # 灰階
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # 原始影像
        axes[0].imshow(image)
        axes[0].set_title('原始影像', fontsize=14)
        axes[0].axis('off')
        
        # 預測 mask（使用自定義顏色）
        colored_mask_display = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored_mask_display[mask == 0] = [0, 0, 128]      # 背景 - 深藍色
        colored_mask_display[mask == 1] = [255, 0, 0]      # 細胞內 - 紅色
        colored_mask_display[mask == 2] = [0, 255, 0]      # 細胞膜 - 綠色
        
        axes[1].imshow(colored_mask_display)
        axes[1].set_title('預測 Mask\n(藍=背景, 紅=細胞內, 綠=細胞膜)', fontsize=12)
        axes[1].axis('off')
        
        # 疊加顯示（只顯示細胞內和細胞膜）
        overlay = image.copy()
        colored_overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored_overlay[mask == 1] = [255, 100, 100]  # 細胞內 - 淺紅色
        colored_overlay[mask == 2] = [100, 255, 100]  # 細胞膜 - 淺綠色
        
        # 確保兩張圖片尺寸和類型一致
        assert overlay.shape == colored_overlay.shape, f"Shape mismatch: {overlay.shape} vs {colored_overlay.shape}"
        
        # Alpha blending
        overlay = cv2.addWeighted(overlay, 0.6, colored_overlay, 0.4, 0)
        
        axes[2].imshow(overlay)
        axes[2].set_title('疊加顯示\n(紅=細胞內, 綠=細胞膜)', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def batch_predict(self, image_dir, output_dir=None, pattern="*.tiff"):
        """
        批次處理資料夾中的所有影像
        
        Args:
            image_dir: 輸入影像資料夾
            output_dir: 輸出資料夾（預設為 process/result/）
            pattern: 檔案匹配模式
        """
        image_dir = Path(image_dir)
        
        # 設定輸出目錄
        if output_dir is None:
            output_dir = Path("process/result")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = sorted(image_dir.glob(pattern))
        print(f"找到 {len(image_files)} 個影像檔案")
        
        for img_path in image_files:
            # 檢查是否已處理
            class1_output = output_dir / f"{img_path.stem}_class1.png"
            if class1_output.exists():
                print(f"跳過已處理: {img_path.name}")
                continue
            
            try:
                self.predict_image(
                    img_path, 
                    save_class1_only=True,
                    result_dir=output_dir,
                    visualize=False
                )
            except Exception as e:
                print(f"✗ 處理失敗 {img_path.name}: {str(e)}")
                continue
        
        print("\n✓ 批次處理完成！")
        print(f"結果已儲存至: {output_dir}")


def main():
    """主程式：示範如何使用"""
    # 設定路徑
    model_path = MODEL_DIR / "unet1.pt"
    test_image = Path("/home/sec312/tsgh/unet_mask/process/tile/tile_x153600_y59392.tiff")
    result_dir = Path("process/result")
    
    # 檢查模型是否存在
    if not model_path.exists():
        print(f"✗ 找不到模型檔案: {model_path}")
        print("請先執行 train.py 訓練模型")
        return
    
    # 建立推論器
    inferencer = TileInference(
        model_path=model_path,
        tile_size=512,
        overlap=128
    )
    
    # 單張影像推論
    if test_image.exists():
        prediction, original = inferencer.predict_image(
            image_path=test_image,
            save_class1_only=True,
            result_dir=result_dir,
            visualize=True
        )
        print(f"\n✓ 完成！請查看 {result_dir} 資料夾")
    else:
        print(f"測試影像不存在: {test_image}")
    
    # 批次推論（如果有資料夾）
    tile_dir = Path("process/tile")
    if tile_dir.exists() and len(list(tile_dir.glob("*.tiff"))) > 0:
        print("\n發現更多影像，是否批次處理？")
        inferencer.batch_predict(
            image_dir=tile_dir,
            output_dir=result_dir
        )


if __name__ == "__main__":
    main()
