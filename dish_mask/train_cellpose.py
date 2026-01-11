#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cellpose 訓練腳本
使用 Cellpose GUI 標註的 10 個 1024x1024 tiles 進行訓練
"""

import os
import numpy as np
from glob import glob
from pathlib import Path
from tifffile import imread
from cellpose import models, train

# ==================== 配置參數 ====================

# 訓練資料目錄列表 (可加入多個目錄)
TRAIN_DIRS = [
    "/home/sec312/tsgh/dish_mask/train",         # 第一輪標註
    "/home/sec312/tsgh/dish_mask/train_round2",  # 第二輪修正 (如果存在)
]
MODEL_DIR = "/home/sec312/tsgh/dish_mask/models"  # 模型輸出目錄

# 訓練參數
MODEL_NAME = "dish_cellpose_v2"   # 模型名稱 (新版本)
INITIAL_MODEL = "cyto3"           # 基礎模型 (可選: cyto, cyto2, cyto3, nuclei)
N_EPOCHS = 100                    # 訓練輪數
LEARNING_RATE = 1e-5              # 學習率
WEIGHT_DECAY = 0.1                # 權重衰減
BATCH_SIZE = 8                    # 批次大小
MIN_TRAIN_MASKS = 1               # 最少訓練 mask 數量

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
                image_file = os.path.join("/home/sec312/tsgh/dish_mask/tile/1024/dish", f"{base_name}.tiff")
            if not os.path.exists(image_file):
                image_file = os.path.join("/home/sec312/tsgh/dish_mask/test_picture/dish", f"{base_name}.tiff")
            
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
    model = models.CellposeModel(
        gpu=True,
        model_type=INITIAL_MODEL
    )
    
    # 開始訓練
    print("\n開始訓練...")
    
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=masks,
        save_path=MODEL_DIR,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        model_name=MODEL_NAME,
        min_train_masks=MIN_TRAIN_MASKS,
        save_every=10,  # 每 10 個 epoch 儲存一次
    )
    
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
