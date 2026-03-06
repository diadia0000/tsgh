#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cellpose 模型測試腳本"""

import os
import random
import warnings
import numpy as np
from glob import glob
from pathlib import Path
from tifffile import imread, imwrite
from cellpose import models

warnings.filterwarnings('ignore', message='.*shaped series shape.*')

BASE_DIR = Path(__file__).resolve().parent

# 配置
TEST_IMAGE_DIR = str(BASE_DIR / "test_picture" / "dish")
MODEL_DIR = str(BASE_DIR / "models")
OUTPUT_DIR = str(BASE_DIR / "test_output")
MODEL_NAME = "cellpose"
N_TEST_IMAGES = 5


def find_model_path():
    """尋找最新的模型檔案"""
    patterns = [
        os.path.join(MODEL_DIR, f"{MODEL_NAME}*"),
        os.path.join(MODEL_DIR, "models", f"{MODEL_NAME}*"),
    ]
    for pattern in patterns:
        matches = glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
    return None


def save_result(img, masks, img_path, output_dir):
    """儲存結果：原圖疊加 mask 輪廓"""
    from skimage.segmentation import find_boundaries
    from PIL import Image
    stem = Path(img_path).stem
    
    # 儲存彩色 mask (每個物件不同顏色)
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(masks.max() + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # 背景為黑色
    colored_mask = colors[masks]
    
    mask_path = os.path.join(output_dir, f"{stem}_masks.png")
    Image.fromarray(colored_mask).save(mask_path)
    
    # 儲存疊加輪廓的圖片
    overlay = img.copy()
    if overlay.dtype != np.uint8:
        overlay = (overlay / overlay.max() * 255).astype(np.uint8)
    if overlay.ndim == 2:
        overlay = np.stack([overlay] * 3, axis=-1)
    
    # 繪製紅色輪廓
    boundaries = find_boundaries(masks, mode='outer')
    overlay[boundaries] = [255, 0, 0]
    
    overlay_path = os.path.join(output_dir, f"{stem}_overlay.png")
    Image.fromarray(overlay).save(overlay_path)


def main():
    # 載入模型
    model_path = find_model_path()
    if not model_path:
        print(f"找不到模型: {MODEL_DIR}")
        return
    
    print(f"模型: {Path(model_path).name}")
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    
    # 隨機選取測試圖片
    all_images = glob(os.path.join(TEST_IMAGE_DIR, "*.tiff"))
    if not all_images:
        print(f"找不到圖片: {TEST_IMAGE_DIR}")
        return
    
    n_samples = min(N_TEST_IMAGES, len(all_images))
    test_images = random.sample(all_images, n_samples)
    print(f"測試 {n_samples} 張圖片 (2048x2048)\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 測試每張圖片
    total_objects = 0
    for i, img_path in enumerate(test_images):
        img = imread(img_path)
        masks, flows, _ = model.eval(img, diameter=None, augment=True, tile_overlap=0.1, bsize=256)
        n_obj = len(np.unique(masks)) - 1
        total_objects += n_obj
        
        save_result(img, masks, img_path, OUTPUT_DIR)
        print(f"[{i+1}/{n_samples}] {Path(img_path).name}: {n_obj} 物件")
    
    print(f"\n完成! 平均 {total_objects/n_samples:.1f} 物件/張")
    print(f"輸出: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
