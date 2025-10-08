#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Her2 病理影像數據清洗工作流程

使用 OpenSlide 從 WSI TIFF 中提取 Her2 (DAB) 陽性訊號的二值化遮罩

輸入：picture/WSI/HER2_20X_ED7.tiff (5層級)
輸出：Her2 陽性區域的二值化遮罩
"""

import gc
import json
from pathlib import Path
import numpy as np
import openslide
from skimage import morphology, exposure
import cv2
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


# 參數配置
class Her2ProcessingConfig:
    def __init__(self):
        # 形態學處理參數
        self.MORPH_OPEN_KERNEL_SIZE = 3
        self.MORPH_OPEN_ITERATIONS = 2
        self.MORPH_CLOSE_KERNEL_SIZE = 3
        self.MORPH_CLOSE_ITERATIONS = 2

        # 處理參數
        self.H_THRESHOLD = 127
        self.SOURCE_IMAGE = "E:/Class/tsgh/picture/WSI/HER2_20X_ED7.tiff"
        self.OUTPUT_DIR = "E:/Class/tsgh/Her2/picture/"
        self.PROCESS_LEVEL = 4
        self.TILE_SIZE = 2048


def to_uint8(img: np.ndarray) -> np.ndarray:
    """將影像轉為uint8以利PNG輸出，使用CLAHE增強對比"""
    if img.dtype == np.uint8:
        return img

    # 正規化到0-255
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(img, dtype=np.uint8)
    
    # 使用CLAHE增強局部對比
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
    return enhanced





def color_deconvolution_her2(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """執行Her2色彩分解，提取Hematoxylin和DAB通道"""
    from skimage.color import hed_from_rgb, separate_stains

    # 使用 skimage 的標準 HED 分解矩陣，因為 Her2 使用 DAB 染色
    rgb_norm = rgb.astype(np.float32) / 255.0
    hed_stains = separate_stains(rgb_norm, hed_from_rgb)

    # 提取 H (Hematoxylin) 和 D (DAB) 通道
    H_channel = hed_stains[:, :, 0]  # Hematoxylin 通道
    DAB_channel = hed_stains[:, :, 2]  # DAB 通道 (第三個通道)

    return H_channel, DAB_channel


def extract_her2_signal(dab_channel: np.ndarray, h_channel: np.ndarray, config: Her2ProcessingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """從DAB通道提取Her2陽性訊號並進行數據清洗，使用H通道過濾避免Hematoxylin誤判"""
    # 反轉並正規化DAB強度
    dab_inverted = 1.0 - dab_channel
    dab_normalized = to_uint8(dab_inverted)
    
    # 正規化H通道用於過濾
    h_normalized = to_uint8(h_channel)
    
    # 使用 Otsu 全域閾值進行 DAB 信號二值化
    otsu_threshold, binary_mask_u8 = cv2.threshold(
        dab_normalized, 
        0, 
        255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary_mask = binary_mask_u8 > 0  # 轉換為布林遮罩
    
    # H通道過濾：排除強烈的Hematoxylin染色區域
    # Hematoxylin染色區域在H通道中會有較高的值
    h_filter_mask = h_normalized < config.H_THRESHOLD
    
    # 整合「Otsu 閾值 + H 通道排除」
    binary_mask = binary_mask & h_filter_mask

    # 數據清洗 - 去噪 (形態學開運算)
    open_kernel = morphology.disk(config.MORPH_OPEN_KERNEL_SIZE)
    cleaned_mask = binary_mask.copy()
    for _ in range(config.MORPH_OPEN_ITERATIONS):
        cleaned_mask = morphology.binary_opening(cleaned_mask, open_kernel)

    # 數據清洗 - 填補 (形態學閉運算)
    close_kernel = morphology.disk(config.MORPH_CLOSE_KERNEL_SIZE)
    for _ in range(config.MORPH_CLOSE_ITERATIONS):
        cleaned_mask = morphology.binary_closing(cleaned_mask, close_kernel)

    # 轉換為uint8格式 (255表示Her2陽性)
    raw_mask = (binary_mask * 255).astype(np.uint8)
    final_mask = (cleaned_mask * 255).astype(np.uint8)

    return dab_normalized, raw_mask, final_mask


def save_png(path: Path, img: np.ndarray):
    """保存PNG圖片"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)

def process_her2_workflow(config: Optional[Her2ProcessingConfig] = None,
                          max_tiles: Optional[int] = None,
                          batch_size: int = 500):
    """主要的Her2處理工作流程"""
    if config is None:
        config = Her2ProcessingConfig()
    wsi_path = Path(config.SOURCE_IMAGE)
    output_dir = Path(config.OUTPUT_DIR)
    print(f"開始處理Her2影像: {wsi_path.name}")
    print(f"處理層級: {config.PROCESS_LEVEL}")
    print(f"批量處理大小: {batch_size} 個圖塊/批次")

    processing_results = []

    try:
        slide = openslide.OpenSlide(str(wsi_path))
        level = config.PROCESS_LEVEL
        level_w, level_h = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]
        
        print(f"層級 {level} 尺寸: {level_w} x {level_h}")
        
        cols = (level_w + config.TILE_SIZE - 1) // config.TILE_SIZE
        rows = (level_h + config.TILE_SIZE - 1) // config.TILE_SIZE
        total_tiles = rows * cols
        if max_tiles:
            total_tiles = min(total_tiles, max_tiles)
        print(f"總共要處理 {total_tiles} 個圖塊 ({rows}x{cols})")
        print(f"Downsample: {downsample:.2f}")
        final_mask = np.ones((level_h, level_w), dtype=np.uint8) * 255
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for tile_idx in range(total_tiles):
                future = executor.submit(process_single_tile, str(wsi_path), level, config.TILE_SIZE, 
                                       tile_idx, cols, level_w, level_h, downsample, config)
                futures.append((tile_idx, future))
            
            for tile_idx, future in futures:
                try:
                    result = future.result()
                    processing_results.append(result)
                    
                    row = tile_idx // cols
                    col = tile_idx % cols
                    y = row * config.TILE_SIZE
                    x = col * config.TILE_SIZE
                    
                    w_expected = min(config.TILE_SIZE, level_w - x)
                    h_expected = min(config.TILE_SIZE, level_h - y)
                    
                    mask_path = output_dir / "masks" / f"tile_{tile_idx:04d}_final_mask.png"
                    if mask_path.exists():
                        tile_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if tile_mask is not None and tile_mask.size > 0:
                            h, w = tile_mask.shape
                            end_y = min(y + h, level_h)
                            end_x = min(x + w, level_w)
                            h_actual = end_y - y
                            w_actual = end_x - x
                            if h_actual > 0 and w_actual > 0:
                                final_mask[y:end_y, x:end_x] = tile_mask[:h_actual, :w_actual]
                        else:
                            final_mask[y:y+h_expected, x:x+w_expected] = 255
                        mask_path.unlink()
                    else:
                        final_mask[y:y+h_expected, x:x+w_expected] = 255
                    
                    if (tile_idx + 1) % 100 == 0:
                        success = len([r for r in processing_results if r['status'] == 'success'])
                        errors = len([r for r in processing_results if r['status'] == 'error'])
                        print(f"已處理 {tile_idx + 1}/{total_tiles} (成功: {success}, 錯誤: {errors})")
                except Exception as e:
                    row = tile_idx // cols
                    col = tile_idx % cols
                    y = row * config.TILE_SIZE
                    x = col * config.TILE_SIZE
                    w_fill = min(config.TILE_SIZE, level_w - x)
                    h_fill = min(config.TILE_SIZE, level_h - y)
                    final_mask[y:y+h_fill, x:x+w_fill] = 255
                    print(f"圖塊 {tile_idx} 拼接失敗: {e}")
        # 保存最終拼接結果
        output_path = output_dir / "DISH_final_complete_mask.png"
        save_png(output_path, final_mask)
        print(f"最終拼接遮罩已保存至: {output_path}")

        # 保存處理統計
        stats = {
            "total_tiles": total_tiles,
            "successful_tiles": len([r for r in processing_results if r["status"] == "success"]),
            "empty_tiles": len([r for r in processing_results if r["status"] == "empty"]),
            "failed_tiles": len([r for r in processing_results if r["status"] == "error"]),
            "batch_size": batch_size,
            "total_batches": (total_tiles + batch_size - 1) // batch_size,
            "config": {
                "process_level": config.PROCESS_LEVEL,
                "h_threshold": config.H_THRESHOLD,
                "threshold_method": "otsu_with_h_filter",
                "morph_open_kernel": config.MORPH_OPEN_KERNEL_SIZE,
                "morph_open_iterations": config.MORPH_OPEN_ITERATIONS,
                "morph_close_kernel": config.MORPH_CLOSE_KERNEL_SIZE,
                "morph_close_iterations": config.MORPH_CLOSE_ITERATIONS
            }
        }

        stats_path = output_dir / "processing_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"處理統計已保存至: {stats_path}")
        print("Her2工作流程完成!")

        return final_mask, processing_results

    finally:
        if 'slide' in locals():
            slide.close()
        gc.collect()


def process_single_tile(wsi_path: str, level: int, tile_size: int, tile_idx: int, 
                       cols: int, level_w: int, level_h: int, downsample: float,
                       config: Her2ProcessingConfig) -> dict:
    output_dir = Path(config.OUTPUT_DIR)
    slide = openslide.OpenSlide(wsi_path)
    
    try:
        row = tile_idx // cols
        col = tile_idx % cols
        x_level = col * tile_size
        y_level = row * tile_size
        w = min(tile_size, level_w - x_level)
        h = min(tile_size, level_h - y_level)
        
        if w <= 0 or h <= 0:
            raise ValueError("Invalid tile size")
        
        x_level0 = int(x_level * downsample)
        y_level0 = int(y_level * downsample)
        tile = slide.read_region((x_level0, y_level0), level, (w, h))
        tile_rgb = np.array(tile.convert('RGB'))
        
        tile_gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
        mean_val = np.mean(tile_gray)
        std_val = np.std(tile_gray)
        
        if mean_val < 35 and std_val < 15:
            raise ValueError("Defective tile detected")
        
        H_channel, DAB_channel = color_deconvolution_her2(tile_rgb)
        _, _, final_mask = extract_her2_signal(DAB_channel, H_channel, config)
        save_png(output_dir / "masks" / f"tile_{tile_idx:04d}_final_mask.png", final_mask)
        return {"tile_idx": tile_idx, "status": "success", "shape": final_mask.shape}
        
    except Exception as e:
        try:
            h_safe = h if h > 0 else tile_size
            w_safe = w if w > 0 else tile_size
            white_mask = np.ones((h_safe, w_safe), dtype=np.uint8) * 255
            save_png(output_dir / "masks" / f"tile_{tile_idx:04d}_final_mask.png", white_mask)
        except:
            pass
        return {"tile_idx": tile_idx, "status": "error", "error": str(e)}
    
    finally:
        slide.close()

if __name__ == "__main__":
    # 執行Her2處理工作流程
    config = Her2ProcessingConfig()
    final_mask, results = process_her2_workflow(config)
