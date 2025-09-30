#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Her2 病理影像數據清洗工作流程 (Her2 Pathology Image Data Cleaning Workflow)

目標：從 Her2 染色的全掃描數位病理影像 (WSI) 中，提取並清洗出 Her2 (DAB) 陽性訊號的二值化遮罩。

記憶體安全策略:
- 一次性讀取大區域包含多個圖塊，處理後切分並寫檔
- 使用 CziFile.read_mosaic(..., scale_factor=0.125) 在讀取時縮小
- 批量處理後明確釋放記憶體

輸入：picture/whole_size/HER2_20X_ED7-002.czi
輸出：Her2 陽性區域的二值化遮罩（批量拼接格式）
"""

import gc
import json
from pathlib import Path
import numpy as np
from aicspylibczi import CziFile
from skimage import morphology, exposure
import cv2
from typing import Tuple, Optional


# 參數配置
class Her2ProcessingConfig:
    def __init__(self):
        # 形態學處理參數
        self.MORPH_OPEN_KERNEL_SIZE = 3
        self.MORPH_OPEN_ITERATIONS = 2
        self.MORPH_CLOSE_KERNEL_SIZE = 3
        self.MORPH_CLOSE_ITERATIONS = 2

        # 處理參數
        self.SCALE_FACTOR = 0.20
        self.H_THRESHOLD = 127      # H通道過濾閾值，用於排除Hematoxylin區域
        self.SOURCE_IMAGE = "E:/Class/tsgh/picture/whole_size/HER2_20X_ED7-002.czi"
        self.OUTPUT_DIR = "E:/Class/tsgh/Her2/picture/"


def to_uint8(img: np.ndarray) -> np.ndarray:
    """將影像轉為uint8以利PNG輸出"""
    if img.dtype == np.uint8:
        return img

    low, high = np.percentile(img, (1, 99))
    if high <= low:
        high, low = img.max(), img.min()

    scaled = exposure.rescale_intensity(img, in_range=(low, high), out_range=(0, 255))
    return scaled.astype(np.uint8)


def bgr_to_rgb_if_needed(img: np.ndarray, pixel_type: str) -> np.ndarray:
    """根據 CZI 的 pixel_type 轉換 BGR → RGB"""
    if img.ndim == 3 and img.shape[-1] == 3:
        if pixel_type and pixel_type.lower().startswith("bgr"):
            return img[..., ::-1]
    return img


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


def stitch_final_masks(config: Her2ProcessingConfig) -> Optional[np.ndarray]:
    """從批次拼接的圖片創建最終的完整遮罩"""
    output_dir = Path(config.OUTPUT_DIR)
    batch_files = sorted(output_dir.glob("batch_*_stitched_mask.png"))

    if not batch_files:
        print("沒有找到批次拼接的遮罩圖片")
        return None

    print(f"找到 {len(batch_files)} 個批次拼接圖片")

    # 讀取所有批次圖片並獲取各自的尺寸
    batch_masks = []
    max_width = 0
    max_height = 0

    for batch_file in batch_files:
        batch_mask = cv2.imread(str(batch_file), cv2.IMREAD_GRAYSCALE)
        batch_masks.append(batch_mask)
        max_height = max(max_height, batch_mask.shape[0])
        max_width = max(max_width, batch_mask.shape[1])

    # 計算最終大圖的網格佈局
    num_batches = len(batch_files)
    grid_cols = int(np.ceil(np.sqrt(num_batches)))
    grid_rows = int(np.ceil(num_batches / grid_cols))

    # 創建最終的大圖，使用最大尺寸作為每個網格的大小
    final_height = grid_rows * max_height
    final_width = grid_cols * max_width
    final_mask = np.zeros((final_height, final_width), dtype=np.uint8)

    print(f"最終拼接尺寸: {final_width} x {final_height}")

    # 拼接所有批次圖片，處理不同尺寸
    for i, batch_mask in enumerate(batch_masks):
        row = i // grid_cols
        col = i % grid_cols

        # 計算在最終大圖中的位置
        start_row = row * max_height
        start_col = col * max_width

        # 獲取當前批次圖片的實際尺寸
        actual_height, actual_width = batch_mask.shape

        # 只拼接實際存在的區域
        end_row = start_row + actual_height
        end_col = start_col + actual_width

        final_mask[start_row:end_row, start_col:end_col] = batch_mask

    # 保存最終拼接結果
    output_path = output_dir / "DISH_final_complete_mask.png"
    cv2.imwrite(str(output_path), final_mask)

    print(f"最終拼接完成: {output_path}")
    print(f"最終尺寸: {final_mask.shape}")

    return final_mask


def process_her2_workflow(config: Optional[Her2ProcessingConfig] = None,
                          max_tiles: Optional[int] = None,
                          batch_size: int = 500):
    """主要的Her2處理工作流程"""
    if config is None:
        config = Her2ProcessingConfig()

    czi_path = Path(config.SOURCE_IMAGE)
    output_dir = Path(config.OUTPUT_DIR)

    print(f"開始處理Her2影像: {czi_path.name}")
    print(f"縮放倍率: {config.SCALE_FACTOR}")
    print(f"批量處理大小: {batch_size} 個圖塊/批次")

    processing_results = []

    try:
        czi = CziFile(str(czi_path))
        bboxes = czi.get_all_mosaic_tile_bounding_boxes()
        bbox_items = list(bboxes.items())

        # --- 第一階段：計算全局座標 ---
        print("第一階段：計算全局座標以確定最終畫布大小...")
        all_coords = [bbox for _, bbox in bbox_items]
        global_min_x = min(b.x for b in all_coords)
        global_min_y = min(b.y for b in all_coords)
        global_max_x = max(b.x + b.w for b in all_coords)
        global_max_y = max(b.y + b.h for b in all_coords)

        final_width = int((global_max_x - global_min_x) * config.SCALE_FACTOR)
        final_height = int((global_max_y - global_min_y) * config.SCALE_FACTOR)

        # --- 創建最終畫布 ---
        final_mask = np.zeros((final_height, final_width), dtype=np.uint8)
        print(f"最終畫布尺寸已確定: {final_width} x {final_height}")

        # 創建 tile_idx 到 bbox 的映射
        tile_idx_to_bbox = {item[0].m_index: item[1] for item in bbox_items}

        total_tiles = len(bbox_items)
        if max_tiles:
            total_tiles = min(total_tiles, max_tiles)

        print(f"總共要處理 {total_tiles} 個圖塊")

        # --- 第二階段：批量處理並精確放置 ---
        for batch_start in range(0, total_tiles, batch_size):
            batch_end = min(batch_start + batch_size, total_tiles)
            current_batch_size = batch_end - batch_start

            print(f"處理批次 {batch_start//batch_size + 1}: 圖塊 {batch_start + 1}-{batch_end} ({current_batch_size} 個)")

            current_batch = bbox_items[batch_start:batch_end]
            batch_results = process_batch_tiles(czi, current_batch, config)
            processing_results.extend(batch_results)

            successful_tiles = [r for r in batch_results if r["status"] == "success"]
            print(f"完成 (成功: {len(successful_tiles)}, 空白: {len([r for r in batch_results if r['status'] == 'empty'])}, 失敗: {len([r for r in batch_results if r['status'] == 'error'])})")

            # 直接將成功處理的圖塊遮罩放置到最終畫布上
            if successful_tiles:
                print(f"將 {len(successful_tiles)} 個遮罩放置到最終畫布上")
                for result in successful_tiles:
                    tile_idx = result['tile_idx']
                    bbox = tile_idx_to_bbox[tile_idx]

                    mask_path = output_dir / "masks" / f"tile_{tile_idx:04d}_final_mask.png"
                    if mask_path.exists():
                        tile_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                        # 計算在最終畫布上的相對位置
                        relative_x = int((bbox.x - global_min_x) * config.SCALE_FACTOR)
                        relative_y = int((bbox.y - global_min_y) * config.SCALE_FACTOR)

                        # 計算實際要拼接的區域大小
                        h, w = tile_mask.shape
                        end_y, end_x = relative_y + h, relative_x + w

                        # 確保不越界
                        if end_y > final_height: end_y = final_height
                        if end_x > final_width: end_x = final_width
                        h = end_y - relative_y
                        w = end_x - relative_x

                        # 放置遮罩
                        final_mask[relative_y:end_y, relative_x:end_x] = tile_mask[:h, :w]

                        # 刪除臨時的單個遮罩文件
                        mask_path.unlink()

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
                "scale_factor": config.SCALE_FACTOR,
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
        if 'czi' in locals():
            del czi
        gc.collect()


def process_batch_tiles(czi: CziFile, tile_batch_info, config: Her2ProcessingConfig) -> list:
    """批量處理：計算圖塊的大區域座標，一次性讀取大圖片，再執行處理流程"""
    batch_results = []

    if not tile_batch_info:
        return batch_results

    print(f"計算 {len(tile_batch_info)} 個圖塊的大區域座標")

    # 提取所有圖塊的座標和索引
    tile_coords = []
    tile_idx_list = []

    for tile_info in tile_batch_info:
        if hasattr(tile_info, '__iter__') and len(tile_info) == 2:
            tile_idx_obj, bbox = tile_info
            tile_idx = tile_idx_obj.m_index
        else:
            tile_idx = getattr(tile_info, 'm_index', 0)
            bbox = tile_info

        tile_coords.append((bbox.x, bbox.y, bbox.w, bbox.h))
        tile_idx_list.append(tile_idx)

    # 計算包含所有圖塊的最小邊界框
    min_x = min(x for x, y, w, h in tile_coords)
    min_y = min(y for x, y, w, h in tile_coords)
    max_x = max(x + w for x, y, w, h in tile_coords)
    max_y = max(y + h for x, y, w, h in tile_coords)

    # 大區域的座標和尺寸
    big_region = (min_x, min_y, max_x - min_x, max_y - min_y)
    print(f"大區域: {big_region}")

    # 一次性讀取大區域
    print("讀取大區域影像")
    big_image_data = czi.read_mosaic(big_region, scale_factor=config.SCALE_FACTOR, C=0)
    # 處理影像維度
    big_image_data = big_image_data.squeeze()
    if big_image_data.ndim == 2:
        big_image_data = np.stack([big_image_data] * 3, axis=-1)
    elif big_image_data.ndim == 3:
        if big_image_data.shape[-1] == 1:
            big_image_data = np.repeat(big_image_data, 3, axis=-1)
        elif big_image_data.shape[-1] > 3:
            big_image_data = big_image_data[..., :3]
    # BGR轉RGB
    pixel_type = getattr(czi, 'pixel_type', 'rgb24')
    big_image = bgr_to_rgb_if_needed(big_image_data, pixel_type)
    big_image = to_uint8(big_image)
    print(f"讀取完成，尺寸: {big_image.shape}")

    # 對整張大圖片執行顏色解構
    print("執行顏色解構")
    big_H_channel, big_DAB_channel = color_deconvolution_her2(big_image)

    # 對整張大圖片執行Her2訊號提取和清洗
    print("執行數據清洗")
    big_dab_normalized, big_raw_mask, big_final_mask = extract_her2_signal(big_DAB_channel, big_H_channel, config)

    # 從大圖片中切出個別圖塊並保存
    print("切分並保存個別圖塊結果")
    output_dir = Path(config.OUTPUT_DIR)

    for i, (tile_coord, tile_idx) in enumerate(zip(tile_coords, tile_idx_list)):
        try:
            x, y, w, h = tile_coord

            # 計算在大圖片中的相對座標（考慮縮放因子）
            relative_x = int((x - min_x) * config.SCALE_FACTOR)
            relative_y = int((y - min_y) * config.SCALE_FACTOR)
            relative_w = int(w * config.SCALE_FACTOR)
            relative_h = int(h * config.SCALE_FACTOR)

            # 確保座標在大圖片範圍內
            relative_x = max(0, min(relative_x, big_image.shape[1]))
            relative_y = max(0, min(relative_y, big_image.shape[0]))
            relative_w = min(relative_w, big_image.shape[1] - relative_x)
            relative_h = min(relative_h, big_image.shape[0] - relative_y)

            if relative_w <= 0 or relative_h <= 0:
                batch_results.append({
                    "tile_idx": tile_idx,
                    "status": "empty",
                    "shape": None
                })
                continue

            # 從大圖片中切出對應的區域
            end_x = relative_x + relative_w
            end_y = relative_y + relative_h

            tile_final_mask = big_final_mask[relative_y:end_y, relative_x:end_x]

            # 保存清洗後的遮罩
            save_png(output_dir / "masks" / f"tile_{tile_idx:04d}_final_mask.png", tile_final_mask)

            batch_results.append({
                "tile_idx": tile_idx,
                "status": "success",
                "shape": tile_final_mask.shape
            })

        except Exception as e:
            batch_results.append({
                "tile_idx": tile_idx,
                "status": "error",
                "error": str(e),
                "shape": None
            })

    print("切分完成")

    # 清理大圖片記憶體
    del big_image_data, big_image, big_H_channel, big_DAB_channel
    del big_dab_normalized, big_raw_mask, big_final_mask
    gc.collect()

    return batch_results


def stitch_batch_masks(batch_results: list, config: Her2ProcessingConfig, batch_idx: int,
                       tile_coords: list) -> Optional[np.ndarray]:
    """根據圖塊在CZI檔案中的實際座標位置拼接遮罩"""
    mask_dir = Path(config.OUTPUT_DIR) / "masks"
    successful_tiles = [r for r in batch_results if r["status"] == "success"]

    if not successful_tiles:
        return None

    # 從座標計算拼接後的大圖尺寸
    min_x = min(x for x, y, w, h in tile_coords)
    min_y = min(y for x, y, w, h in tile_coords)
    max_x = max(x + w for x, y, w, h in tile_coords)
    max_y = max(y + h for x, y, w, h in tile_coords)

    # 計算拼接後大圖的尺寸（考慮縮放因子）
    stitched_width = int((max_x - min_x) * config.SCALE_FACTOR)
    stitched_height = int((max_y - min_y) * config.SCALE_FACTOR)
    stitched_mask = np.zeros((stitched_height, stitched_width), dtype=np.uint8)

    # 創建 tile_idx 到座標的映射
    idx_to_coord = {}
    for result, coord in zip(batch_results, tile_coords):
        if result["status"] == "success":
            idx_to_coord[result["tile_idx"]] = coord

    for result in successful_tiles:
        tile_idx = result['tile_idx']

        if tile_idx not in idx_to_coord:
            continue

        x, y, w, h = idx_to_coord[tile_idx]

        # 計算在拼接圖中的位置（相對於最小座標，考慮縮放）
        relative_x = int((x - min_x) * config.SCALE_FACTOR)
        relative_y = int((y - min_y) * config.SCALE_FACTOR)

        # 讀取遮罩圖片
        mask_path = mask_dir / f"tile_{tile_idx:04d}_final_mask.png"
        if mask_path.exists():
            tile_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # 確保座標在範圍內
            if (relative_x >= 0 and relative_y >= 0 and
                relative_x < stitched_width and relative_y < stitched_height):

                # 計算實際要拼接的區域大小
                actual_w = min(tile_mask.shape[1], stitched_width - relative_x)
                actual_h = min(tile_mask.shape[0], stitched_height - relative_y)

                # 拼接到對應位置
                stitched_mask[relative_y:relative_y+actual_h, relative_x:relative_x+actual_w] = \
                    tile_mask[:actual_h, :actual_w]

            # 處理完後刪除單個圖塊檔案以節省空間
            mask_path.unlink()

    # 保存批次拼接結果
    batch_output_path = Path(config.OUTPUT_DIR) / f"batch_{batch_idx:03d}_stitched_mask.png"
    cv2.imwrite(str(batch_output_path), stitched_mask)

    return stitched_mask


if __name__ == "__main__":
    # 執行Her2處理工作流程
    config = Her2ProcessingConfig()
    final_mask, results = process_her2_workflow(config)
