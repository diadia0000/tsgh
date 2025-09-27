#!/usr/bin/env python3
"""
CZI 馬賽克影像拼接腳本 (使用金字塔縮圖版本)

功能:
1. 遍歷指定目錄下的所有 CZI 檔案
2. 使用最低解析度層級 (scale_factor=0.0625) 進行拼接以節省記憶體
3. 使用互訊息 (Mutual Information, MI) 作為圖像對齊品質評估指標
4. 將拼接後的縮圖影像儲存為 TIFF 格式

記憶體管理策略:
- 使用金字塔最上層（最低解析度）避免記憶體爆炸
- 逐一處理圖塊，立即釋放記憶體
- 估算記憶體使用量並提供警告
"""

import os
import gc
import numpy as np
import tifffile
from pathlib import Path
from datetime import datetime
from aicspylibczi import CziFile
from sklearn.metrics import mutual_info_score
from scipy.ndimage import gaussian_filter

def format_bytes(bytes_count):
    """將位元組轉換為人類可讀的格式"""
    if bytes_count is None:
        return "N/A"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.2f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.2f} PB"

def calculate_mutual_information(img1, img2):
    """
    計算兩個圖像之間的互訊息 (Mutual Information)

    Args:
        img1, img2: 兩個numpy陣列，形狀相同

    Returns:
        float: 互訊息值，數值越大表示兩個圖像越相關
    """
    # 將圖像轉為灰階並平坦化
    if len(img1.shape) == 3:
        img1_gray = np.dot(img1, [0.299, 0.587, 0.114])
    else:
        img1_gray = img1

    if len(img2.shape) == 3:
        img2_gray = np.dot(img2, [0.299, 0.587, 0.114])
    else:
        img2_gray = img2

    # 正規化到0-255範圍並轉換為整數
    img1_norm = ((img1_gray - img1_gray.min()) / (img1_gray.max() - img1_gray.min()) * 255).astype(np.uint8)
    img2_norm = ((img2_gray - img2_gray.min()) / (img2_gray.max() - img2_gray.min()) * 255).astype(np.uint8)

    # 平坦化
    img1_flat = img1_norm.flatten()
    img2_flat = img2_norm.flatten()

    # 計算互訊息
    mi_score = mutual_info_score(img1_flat, img2_flat)
    return mi_score

def check_tile_overlap_quality(tile1, tile2, overlap_region1, overlap_region2):
    """
    使用互訊息評估兩個圖塊重疊區域的對齊品質

    Args:
        tile1, tile2: 圖塊數據
        overlap_region1, overlap_region2: 重疊區域的切片

    Returns:
        float: 互訊息分數，用於評估對齊品質
    """
    try:
        # 提取重疊區域
        region1 = tile1[overlap_region1]
        region2 = tile2[overlap_region2]

        # 確保區域大小相同
        if region1.shape != region2.shape:
            min_h = min(region1.shape[0], region2.shape[0])
            min_w = min(region1.shape[1], region2.shape[1])
            region1 = region1[:min_h, :min_w]
            region2 = region2[:min_h, :min_w]

        # 計算互訊息
        mi_score = calculate_mutual_information(region1, region2)
        return mi_score
    except Exception as e:
        print(f"      - 警告: 互訊息計算失敗: {e}")
        return 0.0

def stitch_single_czi(filepath, output_dir, scale_factor=0.0625):
    """
    拼接單個 CZI 馬賽克檔案 (使用低解析度層級)

    Args:
        filepath: CZI檔案路徑
        output_dir: 輸出目錄
        scale_factor: 縮放因子，預設0.0625 (約1/16解析度)
    """
    print(f"\n{'='*60}")
    print(f"開始處理檔案: {filepath.name}")
    print(f"使用縮放因子: {scale_factor} (約 {scale_factor*100:.1f}% 解析度)")
    print(f"{'='*60}")

    czi = None
    try:
        # 1. 載入 CZI 檔案
        print("  - 正在載入 CZI 檔案...")
        czi = CziFile(filepath)
        print("  - ✓ CZI 檔案載入成功")

        # 2. 檢查是否為馬賽克影像
        if not czi.is_mosaic():
            print(f"  - 警告: {filepath.name} 不是馬賽克檔案，跳過處理")
            return

        print(f"  - ✓ 檔案被識別為馬賽克影像")

        # 3. 獲取縮放後的邊界框
        print("  - 正在獲取馬賽克邊界框...")
        mosaic_bbox = czi.get_mosaic_bounding_box()

        # 根據縮放因子調整畫布大小
        canvas_height = int(mosaic_bbox.h * scale_factor)
        canvas_width = int(mosaic_bbox.w * scale_factor)

        print(f"  - 原始馬賽克尺寸: {mosaic_bbox.w} x {mosaic_bbox.h}")
        print(f"  - 縮放後畫布尺寸: {canvas_width} x {canvas_height}")

        # 獲取像素類型信息
        pixel_type = czi.pixel_type
        print(f"  - 檢測到像素類型: {pixel_type}")

        # 設定輸出格式
        channels = 3
        dtype = np.uint8
        print(f"  - 輸出格式: 通道數={channels}, 數據類型={dtype}")

        # 4. 估算記憶體使用量
        estimated_size = canvas_height * canvas_width * channels * np.dtype(dtype).itemsize
        print(f"  - 預估畫布記憶體使用: {format_bytes(estimated_size)}")

        if estimated_size > 8 * 1024 * 1024 * 1024:  # 8GB
            print(f"  - ⚠️  記憶體警告: 預估使用量超過8GB，建議使用更小的縮放因子")

        # 5. 建立畫布
        print("  - 正在建立畫布...")
        canvas = np.zeros((canvas_height, canvas_width, channels), dtype=dtype)
        canvas_mask = np.zeros((canvas_height, canvas_width), dtype=bool)  # 記錄已填充的區域
        print("  - ✓ 畫布建立完成")

        # 6. 獲取所有圖塊座標
        print("  - 正在獲取圖塊座標...")
        all_tile_bboxes = czi.get_all_mosaic_tile_bounding_boxes()
        num_tiles = len(all_tile_bboxes)
        print(f"  - ✓ 找到 {num_tiles} 個圖塊")

        # 7. 處理圖塊
        print("  - 開始拼接圖塊...")
        processed_tiles = 0
        total_mi_scores = []

        for i, (tile_id, bbox) in enumerate(all_tile_bboxes.items(), 1):
            if i % 100 == 0 or i <= 10:  # 只顯示前10個和每100個的進度
                print(f"    - 正在處理圖塊 {i}/{num_tiles}...")

            try:
                # 讀取縮放後的圖塊
                tile_region = (bbox.x, bbox.y, bbox.w, bbox.h)
                tile_data = czi.read_mosaic(tile_region, scale_factor=scale_factor, C=0)

                if tile_data is None or tile_data.size == 0:
                    continue

                # 處理圖塊數據
                tile_data = tile_data.squeeze()

                # 確保是3通道
                if len(tile_data.shape) == 2:
                    tile_data = np.stack([tile_data] * 3, axis=-1)
                elif len(tile_data.shape) == 3 and tile_data.shape[2] == 1:
                    tile_data = np.repeat(tile_data, 3, axis=2)
                elif len(tile_data.shape) == 3 and tile_data.shape[2] > 3:
                    tile_data = tile_data[:, :, :3]

                # 確保數據類型
                if tile_data.dtype != np.uint8:
                    if tile_data.dtype == np.uint16:
                        tile_data = (tile_data / 256).astype(np.uint8)
                    else:
                        tile_data = (tile_data * 255).astype(np.uint8)

                # 計算在縮放畫布上的位置
                y_start = int((bbox.y - mosaic_bbox.y) * scale_factor)
                x_start = int((bbox.x - mosaic_bbox.x) * scale_factor)
                y_end = y_start + tile_data.shape[0]
                x_end = x_start + tile_data.shape[1]

                # 邊界檢查
                y_start = max(0, y_start)
                x_start = max(0, x_start)
                y_end = min(canvas_height, y_end)
                x_end = min(canvas_width, x_end)

                if y_end <= y_start or x_end <= x_start:
                    continue

                # 調整圖塊大小以匹配畫布區域
                canvas_h = y_end - y_start
                canvas_w = x_end - x_start

                if tile_data.shape[0] > canvas_h or tile_data.shape[1] > canvas_w:
                    tile_data = tile_data[:canvas_h, :canvas_w, :]

                # 檢查重疊區域並計算互訊息
                canvas_region = canvas[y_start:y_end, x_start:x_end, :]
                existing_mask = canvas_mask[y_start:y_end, x_start:x_end]

                if np.any(existing_mask):
                    # 有重疊區域，計算互訊息
                    overlap_mask = existing_mask
                    if np.sum(overlap_mask) > 100:  # 只在重疊區域足夠大時計算MI
                        mi_score = calculate_mutual_information(
                            canvas_region[overlap_mask],
                            tile_data[overlap_mask]
                        )
                        total_mi_scores.append(mi_score)

                        if i <= 10:  # 只為前10個圖塊顯示詳細信息
                            print(f"      - 重疊區域互訊息: {mi_score:.4f}")

                # 將圖塊貼到畫布上
                canvas[y_start:y_end, x_start:x_end, :] = tile_data[:canvas_h, :canvas_w, :]
                canvas_mask[y_start:y_end, x_start:x_end] = True

                processed_tiles += 1

            except Exception as e:
                if i <= 10:  # 只為前10個圖塊顯示錯誤
                    print(f"      - 圖塊 {i} 處理失敗: {e}")
                continue

        print(f"  - ✓ 成功處理 {processed_tiles}/{num_tiles} 個圖塊")


        # 7. 儲存拼接後的影像
        output_filename = f"{filepath.stem}_stitched.tiff"
        output_path = output_dir / output_filename
        print(f"  - 正在儲存拼接影像至: {output_path}")

        tifffile.imwrite(output_path, canvas, imagej=True)

        print(f"  - ✓ 拼接影像儲存成功！")

    except Exception as e:
        print(f"  - ✗ 處理檔案 {filepath.name} 時發生錯誤: {e}")

    finally:
        # 8. 清理記憶體
        if 'canvas' in locals():
            del canvas
        if czi is not None:
            del czi
        gc.collect()
        print("  - ✓ 記憶體已清理。")


def main():
    """
    主程序
    """
    print("CZI 馬賽克影像拼接程序")
    print("="*60)

    # 設定目錄
    picture_dir = Path("E:/Class/tsgh/picture/whole_size/")
    output_dir = Path("E:/Class/tsgh/stitched_output/WSI/")

    # 建立輸出目錄
    output_dir.mkdir(exist_ok=True)
    print(f"輸入目錄: {picture_dir}")
    print(f"輸出目錄: {output_dir}")

    # 檢查輸入目錄
    if not picture_dir.exists():
        print(f"錯誤: 輸入目錄 {picture_dir} 不存在。")
        return

    # 尋找 CZI 檔案
    czi_files = list(picture_dir.glob("*.czi"))
    if not czi_files:
        print(f"錯誤: 在 {picture_dir} 中找不到 CZI 檔案。")
        return

    print(f"找到 {len(czi_files)} 個 CZI 檔案，準備開始處理...")

    # 逐一處理檔案
    for czi_file in czi_files:
        stitch_single_czi(czi_file, output_dir)

    print(f"\n{'='*60}")
    print("所有檔案處理完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
