"""
HER2 Pseudo Mask 批量生成腳本

根據 segment_membrane_interior 邏輯，將原始 HER2 影像批量處理為 0-1-2 標籤的 PNG 檔案

標籤定義:
    0: 背景 (Background)
    1: 細胞內部 (Interior/Cytoplasm/Nucleus)
    2: 細胞膜 (Membrane)
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from skimage import io
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 添加當前目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from her2_mask import generate_her2_mask
from membrane_interior_segmentation import segment_membrane_interior


def generate_dab_mask_bool(img):
    """
    從 HER2 影像生成 DAB mask（布林型）
    
    Args:
        img: RGB 影像 (H, W, 3)
        
    Returns:
        dab_region: 布林 mask (True = DAB 陽性)
    """
    mask, _ = generate_her2_mask(
        img, 
        min_dab_od=config.min_dab_od,
        dab_dominance=config.dab_dominance,
        min_total_od=config.min_total_od,
        closing_radius=config.closing_radius
    )
    # 轉換為布林型態（mask == 0 表示 DAB 陽性）
    return mask == 0


def process_single_image(args):
    """
    處理單張影像並生成 pseudo mask
    
    Args:
        args: (input_path, output_path, min_interior_size, min_hole_size)
        
    Returns:
        tuple: (成功/失敗, 輸入路徑, 統計資訊/錯誤訊息)
    """
    input_path, output_path, min_interior_size, min_hole_size = args
    
    try:
        # 讀取影像
        img = io.imread(input_path)
        
        # 確保是 RGB 格式
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        
        # 生成 DAB mask (布林型)
        dab_mask = generate_dab_mask_bool(img)
        
        # 分割細胞膜和細胞內部
        segmentation = segment_membrane_interior(
            dab_mask,
            min_interior_size=min_interior_size,
            min_hole_size=min_hole_size
        )
        
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存為 8-bit PNG (0, 1, 2 標籤)
        cv2.imwrite(output_path, segmentation.astype(np.uint8))
        
        # 統計各類別像素數
        total_pixels = segmentation.size
        stats = {
            "background": np.sum(segmentation == 0) / total_pixels * 100,
            "interior": np.sum(segmentation == 1) / total_pixels * 100,
            "membrane": np.sum(segmentation == 2) / total_pixels * 100
        }
        
        return True, input_path, stats
        
    except Exception as e:
        return False, input_path, str(e)


def batch_generate_masks(
    input_dir: str = None,
    output_dir: str = None,
    min_interior_size: int = None,
    min_hole_size: int = None,
    num_workers: int = None,
    image_extensions: list = None
):
    """
    批量生成 pseudo masks
    
    分割邏輯:
    - DAB 染色區域 = 細胞膜 (2)
    - 被膜包圍的空洞 = 細胞內部 (1)
    - 其他區域 = 背景 (0)
    
    Args:
        input_dir: 輸入影像資料夾
        output_dir: 輸出 mask 資料夾
        min_interior_size: 最小內部區域大小
        min_hole_size: 最小空洞大小
        num_workers: 並行處理的 worker 數量
        image_extensions: 支援的影像副檔名
    """
    # 使用配置檔案的預設值
    input_dir = input_dir or config.train_image_dir
    output_dir = output_dir or config.mask_dir
    min_interior_size = min_interior_size or config.min_interior_size
    min_hole_size = min_hole_size or config.min_hole_size
    num_workers = num_workers or max(1, multiprocessing.cpu_count() - 2)
    image_extensions = image_extensions or ['.tiff', '.tif', '.png', '.jpg', '.jpeg']
    
    print("=" * 60)
    print("HER2 Pseudo Mask 批量生成")
    print("=" * 60)
    print(f"分割邏輯: DAB=膜(2), 空洞=內部(1), 其他=背景(0)")
    print(f"輸入目錄: {input_dir}")
    print(f"輸出目錄: {output_dir}")
    print(f"最小內部區域: {min_interior_size} pixels")
    print(f"最小空洞大小: {min_hole_size} pixels")
    print(f"並行處理 workers: {num_workers}")
    print()
    
    # 收集所有影像檔案
    input_path = Path(input_dir)
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    image_files = sorted(set(image_files))
    print(f"找到 {len(image_files)} 張影像")
    
    if len(image_files) == 0:
        print("錯誤: 沒有找到任何影像檔案")
        return
    
    # 建立處理任務列表
    tasks = []
    for img_path in image_files:
        # 生成輸出路徑 (保持相同檔名，改為 .png)
        output_filename = img_path.stem + "_mask.png"
        output_path = os.path.join(output_dir, output_filename)
        tasks.append((str(img_path), output_path, min_interior_size, min_hole_size))
    
    # 統計變數
    success_count = 0
    fail_count = 0
    total_stats = {"background": 0, "interior": 0, "membrane": 0}
    failed_files = []
    
    # 使用進度條和並行處理
    print(f"\n開始處理 {len(tasks)} 張影像...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="生成 Masks") as pbar:
            for future in as_completed(futures):
                success, path, result = future.result()
                
                if success:
                    success_count += 1
                    for key in total_stats:
                        total_stats[key] += result[key]
                else:
                    fail_count += 1
                    failed_files.append((path, result))
                
                pbar.update(1)
    
    # 計算平均統計
    if success_count > 0:
        for key in total_stats:
            total_stats[key] /= success_count
    
    # 輸出結果
    print("\n" + "=" * 60)
    print("處理完成！")
    print("=" * 60)
    print(f"成功: {success_count} 張")
    print(f"失敗: {fail_count} 張")
    print(f"\n平均像素分佈:")
    print(f"  背景 (0): {total_stats['background']:.2f}%")
    print(f"  細胞內部 (1): {total_stats['interior']:.2f}%")
    print(f"  細胞膜 (2): {total_stats['membrane']:.2f}%")
    print(f"\n輸出目錄: {output_dir}")
    
    if failed_files:
        print(f"\n失敗的檔案:")
        for path, error in failed_files[:10]:
            print(f"  {Path(path).name}: {error}")
        if len(failed_files) > 10:
            print(f"  ... 還有 {len(failed_files) - 10} 個檔案失敗")


def verify_masks(mask_dir: str = None, sample_count: int = 5):
    """
    驗證生成的 masks 是否正確
    
    Args:
        mask_dir: mask 資料夾路徑
        sample_count: 隨機抽樣驗證的數量
    """
    mask_dir = mask_dir or config.mask_dir
    mask_path = Path(mask_dir)
    
    mask_files = list(mask_path.glob("*_mask.png"))
    
    if len(mask_files) == 0:
        print("錯誤: 沒有找到任何 mask 檔案")
        return
    
    print(f"\n驗證 {len(mask_files)} 個 mask 檔案...")
    
    # 隨機抽樣
    import random
    samples = random.sample(mask_files, min(sample_count, len(mask_files)))
    
    for mask_file in samples:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        
        if mask is None:
            print(f"  {mask_file.name}: 讀取失敗")
            continue
        
        unique_values = np.unique(mask)
        valid = all(v in [0, 1, 2] for v in unique_values)
        
        stats = {
            "shape": mask.shape,
            "dtype": mask.dtype,
            "unique_values": unique_values.tolist(),
            "valid": valid
        }
        
        print(f"  {mask_file.name}: shape={stats['shape']}, dtype={stats['dtype']}, "
              f"values={stats['unique_values']}, valid={stats['valid']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HER2 Pseudo Mask 批量生成")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="輸入影像資料夾")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="輸出 mask 資料夾")
    parser.add_argument("--min_interior_size", type=int, default=None,
                        help="最小內部區域大小 (pixels)")
    parser.add_argument("--min_hole_size", type=int, default=None,
                        help="最小空洞大小 (pixels)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="並行處理的 worker 數量")
    parser.add_argument("--verify", action="store_true",
                        help="驗證生成的 masks")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_masks(args.output_dir)
    else:
        batch_generate_masks(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            min_interior_size=args.min_interior_size,
            min_hole_size=args.min_hole_size,
            num_workers=args.num_workers
        )
