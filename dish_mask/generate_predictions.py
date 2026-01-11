#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
產生 Cellpose GUI 可修正的預測檔案
輸出 _seg.npy 格式，可在 GUI 中載入並修正
"""

import os
import random
import warnings
import numpy as np
from glob import glob
from pathlib import Path
from tifffile import imread
from cellpose import models

warnings.filterwarnings('ignore', message='.*shaped series shape.*')

# 配置
INPUT_DIR = "/home/sec312/tsgh/dish_mask/test_picture/dish"  # 要預測的圖片
OUTPUT_DIR = "/home/sec312/tsgh/dish_mask/train_round2"       # 輸出目錄 (待修正)
MODEL_DIR = "/home/sec312/tsgh/dish_mask/models"
MODEL_NAME = "dish_cellpose"
N_IMAGES = 10  # 要產生多少張預測


def find_model_path():
    patterns = [
        os.path.join(MODEL_DIR, f"{MODEL_NAME}*"),
        os.path.join(MODEL_DIR, "models", f"{MODEL_NAME}*"),
    ]
    for pattern in patterns:
        matches = glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
    return None


def main():
    # 載入模型
    model_path = find_model_path()
    if not model_path:
        print(f"找不到模型: {MODEL_DIR}")
        return
    
    print(f"模型: {Path(model_path).name}")
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    
    # 取得已經訓練過的圖片名稱 (排除)
    trained_dir = "/home/sec312/tsgh/dish_mask/train"
    trained_files = set()
    for f in glob(os.path.join(trained_dir, "*_seg.npy")):
        trained_files.add(Path(f).stem.replace("_seg", ""))
    print(f"排除已訓練的 {len(trained_files)} 張圖片")
    
    # 選取新圖片
    all_images = glob(os.path.join(INPUT_DIR, "*.tiff"))
    new_images = [f for f in all_images if Path(f).stem not in trained_files]
    
    if not new_images:
        print("沒有新圖片可以預測！")
        return
    
    n_samples = min(N_IMAGES, len(new_images))
    selected = random.sample(new_images, n_samples)
    print(f"選取 {n_samples} 張新圖片進行預測\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 預測並儲存
    for i, img_path in enumerate(selected):
        stem = Path(img_path).stem
        img = imread(img_path)
        
        # 預測
        masks, flows, _ = model.eval(img, diameter=None)
        n_obj = len(np.unique(masks)) - 1
        
        # 複製原圖到輸出目錄
        from shutil import copy2
        copy2(img_path, os.path.join(OUTPUT_DIR, f"{stem}.tiff"))
        
        # 儲存 _seg.npy (Cellpose GUI 格式)
        seg_data = {
            'masks': masks,
            'outlines': None,  # GUI 會自動計算
            'chan_choose': [0, 0],
            'ismanual': np.zeros(masks.max() + 1, dtype=bool),
            'filename': f"{stem}.tiff",
            'flows': flows,
            'est_diam': 30.0,
        }
        
        seg_path = os.path.join(OUTPUT_DIR, f"{stem}_seg.npy")
        np.save(seg_path, seg_data)
        
        print(f"[{i+1}/{n_samples}] {stem}: {n_obj} 物件")
    
    print(f"\n完成！")
    print(f"輸出目錄: {OUTPUT_DIR}")
    print(f"\n下一步:")
    print(f"  1. 開啟 Cellpose GUI")
    print(f"  2. File -> Load image/folder -> 選擇 {OUTPUT_DIR}")
    print(f"  3. 修正錯誤的標記 (按住 Alt 刪除, 按住 Ctrl 繪製)")
    print(f"  4. 儲存後將 _seg.npy 移到訓練目錄")
    print(f"  5. 重新訓練模型")


if __name__ == "__main__":
    main()
