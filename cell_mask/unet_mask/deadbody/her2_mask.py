"""
HER2 DAB Mask 生成

使用 Color Deconvolution 從 HER2 IHC 染色影像中提取 DAB（棕色）區域
- 排除淡色區域
- 排除藍色細胞核
- 排除白色背景

最佳參數 (tuned_1_c2):
- min_dab_od=0.13
- dab_dominance=1.05
- closing_radius=2
"""
import os
import cv2
import numpy as np
from skimage import io, morphology

# ========== 路徑設定 ==========
INPUT_DIR = "/home/sec312/tsgh/unet_mask/tile/"
OUTPUT_DIR = "/home/sec312/tsgh/unet_mask/test_output/her2_mask/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== QuPath 提取的參數 ==========
BACKGROUND_RGB = np.array([206.0, 206.0, 212.0])
HEMA_VECTOR = np.array([0.651, 0.701, 0.29])
DAB_VECTOR = np.array([0.269, 0.568, 0.778])

# ========== 最佳參數 (tuned_1_c2) ==========
DEFAULT_MIN_DAB_OD = 0.13
DEFAULT_DAB_DOMINANCE = 1.115
DEFAULT_CLOSING_RADIUS = 2
DEFAULT_MIN_TOTAL_OD = 0.08


def generate_her2_mask(img, 
                       min_dab_od=DEFAULT_MIN_DAB_OD,
                       dab_dominance=DEFAULT_DAB_DOMINANCE,
                       min_total_od=DEFAULT_MIN_TOTAL_OD,
                       closing_radius=DEFAULT_CLOSING_RADIUS):
    """
    從 HER2 影像生成 DAB（棕色）Mask
    
    Args:
        img: RGB 影像 (H, W, 3)
        min_dab_od: 最小 DAB OD 濃度 (越高越嚴格)
        dab_dominance: DAB/Hema 比例 (越高越嚴格)
        min_total_od: 最小總 OD (排除純白背景)
        closing_radius: closing 半徑（連接斷裂區域）
        
    Returns:
        mask: 二值化 mask (黑=DAB, 白=其他)
        dab_gray: DAB 濃度灰階圖 (用於診斷)
    """
    img = img.astype(np.float64)
    
    # 1. 轉換至 OD 空間
    img_clamped = np.clip(img, 1, 255)
    od = -np.log10(img_clamped / BACKGROUND_RGB)
    od = np.clip(od, 0, None)
    
    # 2. 染色向量（正規化）
    hema_vector = HEMA_VECTOR / np.linalg.norm(HEMA_VECTOR)
    dab_vector = DAB_VECTOR / np.linalg.norm(DAB_VECTOR)
    
    # 3. 計算濃度（OD 投影到染色向量）
    hema_conc = np.dot(od, hema_vector)
    dab_conc = np.dot(od, dab_vector)
    total_od = np.sum(od, axis=2)
    
    # 4. 降噪
    dab_smooth = cv2.medianBlur(dab_conc.astype(np.float32), 3)
    hema_smooth = cv2.medianBlur(hema_conc.astype(np.float32), 3)
    
    # 5. DAB 陽性條件
    high_dab = dab_smooth > min_dab_od
    dab_dominant = dab_smooth > hema_smooth * dab_dominance
    not_background = total_od > min_total_od
    dab_positive = high_dab & dab_dominant & not_background
    
    # 6. 排除藍色細胞核
    blue_nuclei = (hema_smooth > 0.25) & (hema_smooth > dab_smooth * 1.5)
    mask = dab_positive & ~blue_nuclei
    
    # 7. 形態學清理
    mask_cleaned = morphology.remove_small_objects(mask, min_size=30)
    mask_closed = morphology.binary_closing(mask_cleaned, morphology.disk(closing_radius))
    hole_threshold = 50 + closing_radius * 30
    mask_final = morphology.remove_small_holes(mask_closed, area_threshold=hole_threshold)
    
    # 8. 輸出格式
    mask_output = (~mask_final).astype(np.uint8) * 255
    dab_gray = (dab_smooth * 255 / (np.max(dab_smooth) + 1e-6)).astype(np.uint8)
    
    return mask_output, dab_gray


def create_overlay(original_img, mask, color=(255, 0, 0), alpha=0.4):
    """將 mask 疊加到原圖"""
    overlay = original_img.copy().astype(np.float64)
    mask_area = mask == 0  # 黑色區域 = DAB
    overlay[mask_area] = overlay[mask_area] * (1 - alpha) + np.array(color) * alpha
    return overlay.astype(np.uint8)


def run_test():
    """測試 mask 生成"""
    her2_path = os.path.join(INPUT_DIR, "tile_x2048_y16384_her2.tiff")
    
    print("=" * 50)
    print("HER2 DAB Mask 生成")
    print("=" * 50)
    print(f"參數: min_dab_od={DEFAULT_MIN_DAB_OD}, dab_dominance={DEFAULT_DAB_DOMINANCE}, closing_radius={DEFAULT_CLOSING_RADIUS}")
    
    # 讀取影像
    her2_img = io.imread(her2_path)[:, :, :3]
    print(f"輸入影像: {her2_path}")
    
    # 生成 mask
    mask, dab_gray = generate_her2_mask(her2_img)
    
    # 計算覆蓋率
    coverage = np.sum(mask == 0) / mask.size * 100
    print(f"覆蓋率: {coverage:.2f}%")
    
    # 儲存結果
    cv2.imwrite(os.path.join(OUTPUT_DIR, "her2_mask.png"), mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "dab_concentration.png"), dab_gray)
    
    overlay = create_overlay(her2_img, mask)
    io.imsave(os.path.join(OUTPUT_DIR, "her2_overlay.png"), overlay)
    
    print(f"\n輸出目錄: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_test()
