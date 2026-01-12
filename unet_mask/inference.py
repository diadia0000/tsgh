"""
HER2 語義分割推論腳本

用於載入訓練好的模型進行預測和視覺化
"""
import os
import sys
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path
from skimage import io
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config


def get_inference_transform():
    """獲取推論時的預處理"""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def load_model(checkpoint_path: str, device: torch.device = None, encoder_name: str = None):
    """
    載入訓練好的模型
    
    Args:
        checkpoint_path: checkpoint 檔案路徑
        device: 計算設備
        encoder_name: encoder 名稱 (若未指定則從 checkpoint 讀取)
        
    Returns:
        model: 載入權重後的模型
    """
    device = device or config.device
    
    print(f"載入 Checkpoint: {checkpoint_path}")
    
    # 先載入 checkpoint 以讀取模型配置
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 從 checkpoint 讀取模型配置 (優先使用 checkpoint 中的設定)
    model_config = checkpoint.get("model_config", {})
    
    if encoder_name is None:
        encoder_name = model_config.get("encoder_name", config.encoder_name)
    
    num_classes = model_config.get("num_classes", config.num_classes)
    
    print(f"使用 Encoder: {encoder_name}")
    print(f"類別數量: {num_classes}")
    
    # 建立模型
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,  # 不使用預訓練權重，使用 checkpoint
        in_channels=3,
        classes=num_classes
    )
    
    # 載入權重
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get("epoch", "N/A")
    metrics = checkpoint.get("metrics", {})
    print(f"Epoch: {epoch}")
    print(f"驗證 mIoU: {metrics.get('val_miou', 'N/A')}")
    
    return model


@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device = None
) -> np.ndarray:
    """
    對單張影像進行預測
    
    Args:
        model: 模型
        image: RGB 影像 (H, W, 3)
        device: 計算設備
        
    Returns:
        prediction: 預測標籤圖 (H, W)
    """
    device = device or config.device
    transform = get_inference_transform()
    
    # 預處理
    augmented = transform(image=image)
    input_tensor = augmented["image"].unsqueeze(0).to(device)
    
    # 推論
    output = model(input_tensor)
    prediction = output.argmax(dim=1).squeeze(0).cpu().numpy()
    
    return prediction.astype(np.uint8)


def create_visualization(
    image: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    建立視覺化結果
    
    Args:
        image: 原始 RGB 影像
        prediction: 預測標籤圖
        alpha: 疊加透明度
        
    Returns:
        visualization: 視覺化結果
    """
    # 顏色定義
    colors = {
        0: [0, 0, 0],       # 背景: 透明 (不疊加)
        1: [0, 255, 0],     # 細胞內部: 綠色
        2: [255, 0, 0]      # 細胞膜: 紅色
    }
    
    vis = image.copy().astype(np.float32)
    
    for class_id, color in colors.items():
        if class_id == 0:  # 背景不疊加
            continue
        mask = prediction == class_id
        vis[mask] = vis[mask] * (1 - alpha) + np.array(color) * alpha
    
    return vis.astype(np.uint8)


def create_colored_mask(prediction: np.ndarray) -> np.ndarray:
    """
    建立彩色 mask
    
    Args:
        prediction: 預測標籤圖
        
    Returns:
        colored_mask: 彩色 mask (RGB)
    """
    # 顏色定義
    colors = {
        0: [255, 255, 255],  # 背景: 白色
        1: [0, 255, 0],      # 細胞內部: 綠色
        2: [255, 0, 0]       # 細胞膜: 紅色
    }
    
    h, w = prediction.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in colors.items():
        colored[prediction == class_id] = color
    
    return colored


def process_single_image(
    model: torch.nn.Module,
    image_path: str,
    output_dir: str,
    device: torch.device = None,
    save_visualization: bool = True,
    save_mask: bool = True,
    save_colored_mask: bool = True
):
    """
    處理單張影像
    
    Args:
        model: 模型
        image_path: 輸入影像路徑
        output_dir: 輸出目錄
        device: 計算設備
        save_visualization: 是否保存視覺化結果
        save_mask: 是否保存原始 mask
        save_colored_mask: 是否保存彩色 mask
    """
    # 讀取影像
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取影像: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 預測
    prediction = predict_single(model, image, device)
    
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 輸出檔名
    filename = Path(image_path).stem
    
    # 保存結果
    if save_mask:
        mask_path = os.path.join(output_dir, f"{filename}_mask.png")
        cv2.imwrite(mask_path, prediction)
    
    if save_colored_mask:
        colored = create_colored_mask(prediction)
        colored_path = os.path.join(output_dir, f"{filename}_colored.png")
        cv2.imwrite(colored_path, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
    
    if save_visualization:
        vis = create_visualization(image, prediction)
        vis_path = os.path.join(output_dir, f"{filename}_overlay.png")
        cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    # 統計
    total_pixels = prediction.size
    stats = {
        "background": np.sum(prediction == 0) / total_pixels * 100,
        "interior": np.sum(prediction == 1) / total_pixels * 100,
        "membrane": np.sum(prediction == 2) / total_pixels * 100
    }
    
    return stats


def batch_inference(
    model: torch.nn.Module,
    input_dir: str,
    output_dir: str,
    device: torch.device = None,
    extensions: list = None
):
    """
    批量推論
    
    Args:
        model: 模型
        input_dir: 輸入目錄
        output_dir: 輸出目錄
        device: 計算設備
        extensions: 支援的副檔名
    """
    extensions = extensions or ['.tiff', '.tif', '.png', '.jpg', '.jpeg']
    
    # 收集影像檔案
    input_path = Path(input_dir)
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    image_files = sorted(set(image_files))
    
    print(f"找到 {len(image_files)} 張影像")
    
    if len(image_files) == 0:
        print("沒有找到任何影像")
        return
    
    # 統計
    total_stats = {"background": 0, "interior": 0, "membrane": 0}
    
    for img_file in tqdm(image_files, desc="推論中"):
        stats = process_single_image(
            model, str(img_file), output_dir, device
        )
        if stats:
            for key in total_stats:
                total_stats[key] += stats[key]
    
    # 平均統計
    n = len(image_files)
    print(f"\n平均像素分佈:")
    print(f"  背景: {total_stats['background']/n:.2f}%")
    print(f"  細胞內部: {total_stats['interior']/n:.2f}%")
    print(f"  細胞膜: {total_stats['membrane']/n:.2f}%")


def find_best_checkpoint(model_dir: str) -> str:
    """
    找到最佳 checkpoint
    
    Args:
        model_dir: 模型目錄
        
    Returns:
        checkpoint_path: 最佳 checkpoint 路徑
    """
    model_path = Path(model_dir)
    checkpoints = list(model_path.glob("checkpoint_*.pth"))
    
    if len(checkpoints) == 0:
        raise ValueError(f"在 {model_dir} 中沒有找到任何 checkpoint")
    
    # 按照 mIoU 排序 (從檔名解析)
    def extract_miou(path):
        try:
            name = path.stem
            parts = name.split("_")
            for i, p in enumerate(parts):
                if p == "val":
                    return float(parts[i + 2])
        except:
            return 0.0
        return 0.0
    
    checkpoints = sorted(checkpoints, key=extract_miou, reverse=True)
    return str(checkpoints[0])


def main():
    parser = argparse.ArgumentParser(description="HER2 語義分割推論")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint 檔案路徑 (若未指定則使用最佳 checkpoint)")
    parser.add_argument("--input", type=str, required=True,
                        help="輸入影像或目錄")
    parser.add_argument("--output", type=str, default="./inference_output",
                        help="輸出目錄")
    parser.add_argument("--device", type=str, default="cuda",
                        help="計算設備 (cuda 或 cpu)")
    
    args = parser.parse_args()
    
    # 設置設備
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 載入模型
    if args.checkpoint is None:
        checkpoint_path = find_best_checkpoint(config.model_save_dir)
    else:
        checkpoint_path = args.checkpoint
    
    model = load_model(checkpoint_path, device)
    
    # 推論
    input_path = Path(args.input)
    
    if input_path.is_file():
        print(f"\n處理單張影像: {input_path}")
        stats = process_single_image(model, str(input_path), args.output, device)
        if stats:
            print(f"像素分佈:")
            print(f"  背景: {stats['background']:.2f}%")
            print(f"  細胞內部: {stats['interior']:.2f}%")
            print(f"  細胞膜: {stats['membrane']:.2f}%")
    elif input_path.is_dir():
        print(f"\n批量處理目錄: {input_path}")
        batch_inference(model, str(input_path), args.output, device)
    else:
        print(f"錯誤: 找不到 {input_path}")
        return
    
    print(f"\n輸出目錄: {args.output}")


if __name__ == "__main__":
    main()
