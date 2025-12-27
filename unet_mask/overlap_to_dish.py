"""
HER2 Mask 生成並疊加到 DISH 圖片

從 her2_mask.py 導入 mask 生成函數，確保參數一致
"""
import os
import cv2
import numpy as np
from skimage import io

# 從 her2_mask.py 導入
from her2_mask import (
    generate_her2_mask,
    create_overlay,
    DEFAULT_MIN_DAB_OD,
    DEFAULT_DAB_DOMINANCE,
    DEFAULT_CLOSING_RADIUS
)

# ========== 路徑設定 ==========
INPUT_DIR = "/home/sec312/tsgh/unet_mask/tile/"
OUTPUT_DIR = "/home/sec312/tsgh/unet_mask/test_output/overlap/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def overlay_mask_on_dish(dish_img, mask, color=(255, 0, 0), alpha=0.5):
    """
    將 HER2 mask 疊加到 DISH 圖片上
    
    Args:
        dish_img: DISH RGB 影像 (H, W, 3), uint8
        mask: 二值化 mask (H, W), 黑色 (0) = 要標記的區域
        color: 疊加顏色 (R, G, B)
        alpha: 透明度 (0-1)
        
    Returns:
        overlay: 疊加後的影像 (H, W, 3), uint8
    """
    overlay = dish_img.copy().astype(np.float64)
    mask_area = mask == 0  # 黑色區域 = DAB
    overlay[mask_area] = overlay[mask_area] * (1 - alpha) + np.array(color) * alpha
    return overlay.astype(np.uint8)


def process_tile_pair(her2_path, dish_path, output_prefix=None):
    """
    處理一對 HER2 和 DISH 圖片
    
    Args:
        her2_path: HER2 圖片路徑
        dish_path: DISH 圖片路徑
        output_prefix: 輸出檔名前綴 (可選)
        
    Returns:
        mask: 生成的 HER2 mask
        overlay: 疊加到 DISH 上的結果
    """
    # 讀取影像
    her2_img = io.imread(her2_path)[:, :, :3]
    dish_img = io.imread(dish_path)[:, :, :3]
    
    # 生成 HER2 mask（使用 her2_mask.py 的函數）
    mask, dab_gray = generate_her2_mask(her2_img)
    
    # 疊加到 DISH
    overlay = overlay_mask_on_dish(dish_img, mask, color=(255, 0, 0), alpha=0.5)
    
    # 自動輸出檔名
    if output_prefix is None:
        base_name = os.path.splitext(os.path.basename(her2_path))[0]
        output_prefix = base_name.replace("_her2", "")
    
    # 儲存結果
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{output_prefix}_mask.png"), mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{output_prefix}_dab_gray.png"), dab_gray)
    io.imsave(os.path.join(OUTPUT_DIR, f"{output_prefix}_dish_overlay.png"), overlay)
    
    return mask, overlay


def run_overlap_test():
    """測試疊加功能"""
    her2_file = os.path.join(INPUT_DIR, "tile_x60928_y30720_her2.tiff")
    dish_file = os.path.join(INPUT_DIR, "tile_x60928_y30720_dish.tiff")
    
    print("=" * 50)
    print("HER2 Mask 疊加到 DISH")
    print("=" * 50)
    print(f"參數 (繼承自 her2_mask.py): min_dab_od={DEFAULT_MIN_DAB_OD}, dab_dominance={DEFAULT_DAB_DOMINANCE}, closing_radius={DEFAULT_CLOSING_RADIUS}")
    print(f"\nHER2: {her2_file}")
    print(f"DISH: {dish_file}")
    
    mask, overlay = process_tile_pair(her2_file, dish_file, "tile_x60928_y30720")
    
    # 計算統計
    mask_coverage = np.sum(mask == 0) / mask.size * 100
    print(f"\n覆蓋率: {mask_coverage:.2f}%")
    print(f"輸出目錄: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_overlap_test()
