import os
import cv2
import numpy as np
from skimage import io, color, util, morphology
from matplotlib import pyplot as plt
# 路徑設定
INPUT_DIR = "/home/sec312/tsgh/unet_mask/tile/"
OUTPUT_DIR = "/home/sec312/tsgh/unet_mask/test_output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def optimize_her2_mask(rgb_img):
    """
    優化 HER2 咖啡色通道提取與 Mask 生成
    """
    # 轉換為 float 並避免 log(0)
    img = rgb_img.astype(np.float64) + 1
    
    # 1. 轉換至 Optical Density (OD) 空間
    # OD = -log10(I / I0), 假設 I0 = 255
    od = -np.log10(img / 255.0)
    
    # 2. 定義自定義染色矩陣 (針對您的實驗室數據微調)
    # 這裡的向量可以根據您的實際染色效果進行校準
    # 每行代表一種成分: [Hematoxylin, Others, DAB]
    # 下方是優化過的 DAB (咖啡色) 向量參考值
    stain_matrix = np.array([
        [0.650, 0.704, 0.286],  # Hematoxylin
        [0.072, 0.990, 0.105],  # 預留給其他染色 (如 DISH Red)
        [0.268, 0.570, 0.776]   # DAB (Brown) - 優化過的向量
    ])
    stain_matrix /= np.linalg.norm(stain_matrix, axis=1)[:, np.newaxis]
    
    # 解矩陣方程式: OD = S * C  => C = S^-1 * OD
    # 這裡我們提取第三個通道 (Index 2: DAB)
    stain_inv = np.linalg.inv(stain_matrix)
    od_flat = od.reshape((-1, 3)).T
    concentrations = np.dot(stain_inv, od_flat)
    dab_channel = concentrations[2, :].reshape(img.shape[:2])
    
    # 3. 濾波去噪 (使用中值濾波器保持邊緣)
    dab_denoised = cv2.medianBlur(dab_channel.astype(np.float32), 3)
    
    # 4. 適應性門檻或固定 OD 門檻
    # 在 OD 空間中，HER2 陽性通常 > 0.15~0.2
    mask = (dab_denoised > 0.18).astype(np.uint8) * 255
    
    # 5. 形態學精煉：針對「細胞膜」特性
    # 使用移除小物件與閉運算，確保膜的連續性
    mask_cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=50)
    mask_cleaned = morphology.binary_closing(mask_cleaned, morphology.disk(2))
    
    return (dab_denoised, mask_cleaned.astype(np.uint8) * 255)

def run_test():
    her2_file = os.path.join(INPUT_DIR, "tile_x60928_y30720_her2.tiff")
    dish_file = os.path.join(INPUT_DIR, "tile_x60928_y30720_dish.tiff")
    
    her2_rgb = io.imread(her2_file)[:, :, :3]
    dish_rgb = io.imread(dish_file)[:, :, :3]
    
    # 執行優化演算法
    dab_od, membrane_mask = optimize_her2_mask(her2_rgb)
    
    # 儲存結果
    # 1. DAB 濃度圖 (OD 空間，值越高表示越褐)
    plt.imsave(os.path.join(OUTPUT_DIR, "dab_concentration_od.png"), dab_od, cmap='jet')
    
    # 2. 二值化 Mask
    cv2.imwrite(os.path.join(OUTPUT_DIR, "refined_membrane_mask.png"), membrane_mask)
    
    # 3. 疊合驗證
    overlay = dish_rgb.copy()
    # 將 Mask 區域在 DISH 圖上加強綠色外框或半透明紅色，以便觀察與 DISH 點的關係
    overlay[membrane_mask > 0] = overlay[membrane_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    io.imsave(os.path.join(OUTPUT_DIR, "dish_her2_overlay.tiff"), overlay.astype(np.uint8))
    
    print(f"優化驗證完成。請檢查: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_test()