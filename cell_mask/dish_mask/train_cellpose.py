#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cellpose 訓練腳本
使用 Cellpose GUI 標註的 10 個 1024x1024 tiles 進行訓練
"""

import os
import sys
import logging
import numpy as np
import albumentations as A
from glob import glob
from pathlib import Path
from tifffile import imread
from cellpose import models, train

# 設定 logging 讓訓練進度可見
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# 確保 cellpose 的 logger 也輸出
logging.getLogger('cellpose').setLevel(logging.INFO)

# ==================== 配置參數 ====================

BASE_DIR = Path(__file__).resolve().parent

# 訓練資料目錄列表 (可加入多個目錄)
TRAIN_DIRS = [
    str(BASE_DIR / "train"),
]
MODEL_DIR = str(BASE_DIR / "models")  # 模型輸出目錄

# 訓練參數
MODEL_NAME = "cellpose"   # 模型名稱
# 使用之前訓練的模型繼續訓練 (設為 None 或 "cyto3" 則從頭開始)
INITIAL_MODEL = "cyto3"  # 繼續訓練
N_EPOCHS = 200                    # 更多資料，增加訓練輪數
LEARNING_RATE = 5e-6              # 繼續訓練時使用較小學習率
WEIGHT_DECAY = 0.1                # 權重衰減
BATCH_SIZE = 8                    # 批次大小
MIN_TRAIN_MASKS = 1               # 最少訓練 mask 數量
TEST_SPLIT = 0.2                  # 測試集比例 (20%)

# 資料增強參數
SCALE_RANGE = 1.0                 # 縮放範圍: 圖片會縮放 (1-range/2) ~ (1+range/2)，即 0.5x ~ 1.5x
RESCALE = True                    # 正規化細胞到統一大小
USE_COLOR_AUGMENTATION = True     # 是否使用顏色增強

# 顏色增強設定 (適合組織染色圖像)
COLOR_AUGMENTATION = A.Compose([
    A.ColorJitter(
        brightness=0.15,          # 亮度變化 ±15%
        contrast=0.15,            # 對比度變化 ±15%
        saturation=0.1,           # 飽和度變化 ±10%
        hue=0.05,                 # 色調變化 ±5%
        p=0.5                     # 50% 機率套用
    ),
    A.GaussNoise(
        std_range=(0.02, 0.1),    # 高斯雜訊 (正規化範圍 0-1)
        p=0.3                     # 30% 機率套用
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.1,     # 額外亮度調整
        contrast_limit=0.1,       # 額外對比度調整  
        p=0.3                     # 30% 機率套用
    ),
])

# ==================== 資料增強函數 ====================

def apply_color_augmentation(images):
    """對圖片列表套用顏色增強"""
    if not USE_COLOR_AUGMENTATION:
        return images
    
    augmented_images = []
    for img in images:
        # Albumentations 需要 HWC 格式的 uint8 圖片
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
        
        augmented = COLOR_AUGMENTATION(image=img)
        augmented_images.append(augmented['image'])
    
    return augmented_images


# ==================== 資料準備 ====================

def prepare_training_data():
    """準備訓練資料：從多個目錄載入圖片和 masks"""
    print("=" * 60)
    print("準備訓練資料...")
    print("=" * 60)
    
    images = []
    masks = []
    file_names = []
    
    for train_dir in TRAIN_DIRS:
        if not os.path.exists(train_dir):
            print(f"  跳過不存在的目錄: {train_dir}")
            continue
        
        seg_files = sorted(glob(os.path.join(train_dir, "*_seg.npy")))
        print(f"\n{train_dir}: {len(seg_files)} 個標註")
        
        for seg_file in seg_files:
            base_name = Path(seg_file).stem.replace("_seg", "")
            
            # 嘗試在同目錄或 tile 目錄找圖片
            image_file = os.path.join(train_dir, f"{base_name}.tiff")
            if not os.path.exists(image_file):
                image_file = str(BASE_DIR / "tile" / "1024" / "dish" / f"{base_name}.tiff")
            if not os.path.exists(image_file):
                image_file = str(BASE_DIR / "test_picture" / "dish" / f"{base_name}.tiff")
            
            if not os.path.exists(image_file):
                print(f"    ⚠️ 找不到圖片: {base_name}")
                continue
            
            img = imread(image_file)
            seg_data = np.load(seg_file, allow_pickle=True).item()
            mask = seg_data.get('masks', None)
            
            if mask is None:
                continue
            
            n_objects = len(np.unique(mask)) - 1
            if n_objects < MIN_TRAIN_MASKS:
                continue
            
            images.append(img)
            masks.append(mask)
            file_names.append(base_name)
            print(f"    ✓ {base_name}: {img.shape}, {n_objects} 物件")
    
    print(f"\n總計: {len(images)} 組訓練資料")
    return images, masks, file_names


# ==================== 訓練函數 ====================

def train_cellpose_model(images, masks, file_names):
    """
    訓練 Cellpose 模型
    """
    print("\n" + "=" * 60)
    print("開始訓練 Cellpose 模型...")
    print("=" * 60)
    
    # 確保模型輸出目錄存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 初始化模型
    print(f"\n基礎模型: {INITIAL_MODEL}")
    print(f"學習率: {LEARNING_RATE}")
    print(f"訓練輪數: {N_EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    
    # 使用 GPU 如果可用
    # 判斷是使用預設模型還是自訓練模型
    if os.path.exists(str(INITIAL_MODEL)):
        print(f"載入已訓練模型: {INITIAL_MODEL}")
        model = models.CellposeModel(
            gpu=True,
            pretrained_model=INITIAL_MODEL  # 使用自訓練模型
        )
    else:
        print(f"使用預設模型: {INITIAL_MODEL}")
        model = models.CellposeModel(
            gpu=True,
            model_type=INITIAL_MODEL  # 使用預設模型類型
        )
    
    # 開始訓練
    print("\n開始訓練...")
    
    # 分割訓練/測試集
    n_total = len(images)
    n_test = max(1, int(n_total * TEST_SPLIT))  # 至少 1 個測試樣本
    n_train = n_total - n_test
    
    # 隨機打亂索引
    np.random.seed(42)  # 固定隨機種子以便復現
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_images = [images[i] for i in train_indices]
    train_masks = [masks[i] for i in train_indices]
    test_images = [images[i] for i in test_indices]
    test_masks = [masks[i] for i in test_indices]
    
    print(f"訓練集: {len(train_images)} 張, 測試集: {len(test_images)} 張")
    
    # 注意: train_seg 會在 save_path 下建立 models 子目錄
    # 所以這裡用父目錄，避免 models/models 的問題
    save_base = str(Path(MODEL_DIR).parent)
    
    # 套用顏色增強
    if USE_COLOR_AUGMENTATION:
        print(f"\n套用顏色增強...")
        train_images_aug = apply_color_augmentation(train_images)
        print(f"  ✓ 已增強 {len(train_images_aug)} 張訓練圖片")
    else:
        train_images_aug = train_images
    
    print(f"\n縮放範圍: {SCALE_RANGE}")
    print(f"重新縮放: {RESCALE}")
    
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_images_aug,
        train_labels=train_masks,
        test_data=test_images,      # 加入測試集
        test_labels=test_masks,     # 加入測試集
        save_path=save_base,        # 改用父目錄
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        model_name=MODEL_NAME,
        min_train_masks=MIN_TRAIN_MASKS,
        save_every=10,              # 每 10 個 epoch 儲存一次
        scale_range=SCALE_RANGE,    # 縮放增強
        rescale=RESCALE,            # 正規化細胞大小
    )
    
    # 輸出訓練曲線摘要
    print(f"\n訓練 Loss 範圍: {train_losses.min():.4f} ~ {train_losses.max():.4f}")
    print(f"最終 Loss: {train_losses[-1]:.4f}")
    
    print("\n" + "=" * 60)
    print("訓練完成!")
    print("=" * 60)
    print(f"模型已儲存至: {model_path}")
    
    return model_path, train_losses, test_losses


# ==================== 主程式 ====================

def main():
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "Cellpose 訓練腳本" + " " * 25 + "║")
    print("║" + " " * 15 + "Dish Mask Training" + " " * 24 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # 1. 準備訓練資料
    images, masks, file_names = prepare_training_data()
    
    if len(images) == 0:
        print("❌ 沒有找到有效的訓練資料！")
        return
    
    # 2. 訓練模型
    model_path, train_losses, test_losses = train_cellpose_model(images, masks, file_names)
    
    print("\n" + "=" * 60)
    print("全部完成! 🎉")
    print("=" * 60)
    print(f"\n模型位置: {model_path}")
    print(f"可以使用以下命令進行推論:")
    print(f"  python -m cellpose --pretrained_model {model_path} --dir <IMAGE_DIR>")


if __name__ == "__main__":
    main()
