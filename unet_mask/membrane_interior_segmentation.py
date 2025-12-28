"""
HER2 Mask 細胞膜與細胞內部分割

從 her2_mask.py 導入 mask 生成函數，確保參數一致

將 DAB mask 進一步區分為：
- 細胞膜 (Membrane): HER2 染色的外圍
- 細胞內部 (Interior): 被膜包圍的區域
- 背景 (Background): 細胞外的空白區域
"""
import os
import cv2
import numpy as np
from skimage import io, morphology, measure
from scipy import ndimage

# 從 her2_mask.py 導入
from her2_mask import (
    generate_her2_mask,
    DEFAULT_MIN_DAB_OD,
    DEFAULT_DAB_DOMINANCE,
    DEFAULT_CLOSING_RADIUS
)

# ========== 路徑設定 ==========
INPUT_DIR = "/home/sec312/tsgh/unet_mask/tile/"
OUTPUT_DIR = "/home/sec312/tsgh/unet_mask/test_output/segmentation/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_dab_mask_bool(img):
    """
    從 HER2 影像生成 DAB mask（布林型）
    
    Returns:
        dab_region: 布林 mask (True = DAB 陽性)
    """
    mask, _ = generate_her2_mask(img)
    # 轉換為布林型態（mask == 0 表示 DAB 陽性）
    return mask == 0


def segment_membrane_interior(dab_mask, min_interior_size=100, min_hole_size=50):
    """
    將 DAB mask 分割為細胞膜和細胞內部
    
    修正邏輯：
    - 整個 DAB 區域 = 細胞膜 (Membrane)
    - 被膜包圍的空洞 = 細胞內部 (Interior)
    - 其他區域 = 背景 (Background)
    
    Args:
        dab_mask: 二值化 DAB mask (True = DAB 陽性)
        min_interior_size: 最小內部區域大小 (小於此值視為噪點)
        min_hole_size: 最小空洞大小 (小於此值不視為內部)
        
    Returns:
        segmentation: (H, W) 標籤圖
            0 = 背景 (Background)
            1 = 細胞內部 (Interior) - 被膜包圍的空洞
            2 = 細胞膜 (Membrane) - DAB 染色區域
    """
    dab_mask = dab_mask.astype(bool)
    
    # 初始化分割結果
    segmentation = np.zeros(dab_mask.shape, dtype=np.uint8)
    
    # Step 1: DAB 陽性區域 = 細胞膜 (2)
    segmentation[dab_mask] = 2
    
    # Step 2: 找出被膜包圍的空洞 (細胞內部)
    # 使用 binary_fill_holes 填充整個 DAB 區域，然後與原始 mask 做差集
    filled = ndimage.binary_fill_holes(dab_mask)
    holes = filled & ~dab_mask  # 被填充的區域 = 空洞 = 細胞內部
    
    # Step 3: 過濾太小的空洞 (可能是噪點)
    if min_hole_size > 0:
        holes = morphology.remove_small_objects(holes, min_size=min_hole_size)
    
    # Step 4: 標記細胞內部 (1)
    segmentation[holes] = 1
    
    # Step 5: 處理邊緣觸碰的情況
    # 如果一個區域接觸影像邊緣，它不應該被視為「內部」
    # 使用連通區域分析，移除接觸邊緣的「內部」區域
    interior_labeled = measure.label(segmentation == 1)
    
    for region in measure.regionprops(interior_labeled):
        # 檢查該區域是否接觸影像邊緣
        minr, minc, maxr, maxc = region.bbox
        touches_edge = (minr == 0 or minc == 0 or 
                       maxr == dab_mask.shape[0] or maxc == dab_mask.shape[1])
        
        if touches_edge:
            # 接觸邊緣的區域改為背景
            segmentation[interior_labeled == region.label] = 0
        elif region.area < min_interior_size:
            # 太小的內部區域也改為背景 (可能是噪點)
            segmentation[interior_labeled == region.label] = 0
    
    return segmentation


def create_visualization(original_img, segmentation):
    """
    創建視覺化結果
    
    顏色編碼:
    - 背景: 原始顏色
    - 內部: 綠色
    - 膜: 紅色
    """
    vis = original_img.copy().astype(np.float64)
    
    # 內部 = 綠色
    interior_mask = segmentation == 1
    vis[interior_mask] = vis[interior_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
    
    # 膜 = 紅色
    membrane_mask = segmentation == 2
    vis[membrane_mask] = vis[membrane_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    return vis.astype(np.uint8)


def create_3class_mask(segmentation):
    """
    創建三分類 mask (用於 UNet 訓練)
    
    輸出:
    - 0 = 背景
    - 127 = 細胞內部
    - 255 = 細胞膜
    """
    mask_3class = np.zeros(segmentation.shape, dtype=np.uint8)
    mask_3class[segmentation == 0] = 0    # 背景
    mask_3class[segmentation == 1] = 127  # 內部
    mask_3class[segmentation == 2] = 255  # 膜
    
    return mask_3class


def run_segmentation_test(membrane_thickness=5):
    """執行分割測試"""
    her2_path = os.path.join(INPUT_DIR, "tile_x49152_y98304_her2.tiff")
    
    print("=" * 50)
    print("HER2 細胞膜與內部分割")
    print("=" * 50)
    print(f"DAB 參數 (繼承自 her2_mask.py): min_dab_od={DEFAULT_MIN_DAB_OD}, dab_dominance={DEFAULT_DAB_DOMINANCE}, closing_radius={DEFAULT_CLOSING_RADIUS}")
    print(f"膜厚度: {membrane_thickness} pixels")
    
    # 讀取影像
    her2_img = io.imread(her2_path)[:, :, :3]
    print(f"輸入影像: {her2_path}")
    
    # 生成 DAB mask（使用 her2_mask.py 的函數）
    dab_mask = generate_dab_mask_bool(her2_img)
    
    # 分割膜和內部
    segmentation = segment_membrane_interior(
        dab_mask, 
        membrane_thickness=membrane_thickness,
        min_interior_size=100
    )
    
    # 創建視覺化
    vis = create_visualization(her2_img, segmentation)
    
    # 創建三分類 mask
    mask_3class = create_3class_mask(segmentation)
    
    # 儲存結果
    io.imsave(os.path.join(OUTPUT_DIR, "01_original.png"), her2_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "02_dab_mask.png"), 
                (~dab_mask).astype(np.uint8) * 255)
    io.imsave(os.path.join(OUTPUT_DIR, "03_segmentation_vis.png"), vis)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "04_3class_mask.png"), mask_3class)
    
    # 統計
    total_pixels = segmentation.size
    background_ratio = np.sum(segmentation == 0) / total_pixels * 100
    interior_ratio = np.sum(segmentation == 1) / total_pixels * 100
    membrane_ratio = np.sum(segmentation == 2) / total_pixels * 100
    
    print(f"\n=== 分割結果統計 ===")
    print(f"背景: {background_ratio:.2f}%")
    print(f"細胞內部: {interior_ratio:.2f}%")
    print(f"細胞膜: {membrane_ratio:.2f}%")
    print(f"\n輸出目錄: {OUTPUT_DIR}")
    
    return segmentation


if __name__ == "__main__":
    run_segmentation_test(membrane_thickness=5)
