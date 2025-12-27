import os
import cv2
import numpy as np
from skimage import io, color, morphology

# 路徑設定
INPUT_DIR = "/home/sec312/tsgh/unet_mask/tile/"
OUTPUT_DIR = "/home/sec312/tsgh/unet_mask/test_output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_her2_membrane_mask(her2_path):
    # 1. 讀取 HER2 影像 (假設內含 Brown 與 Blue)
    img = io.imread(her2_path)[:, :, :3]
    
    # 2. 顏色解捲機：使用標準 HED 矩陣 (H:藍, E:無效, D:棕)
    # 在這裡我們只取 DAB 通道 (index 2)
    ihc_hed = color.separate_stains(img, color.hed_from_rgb)
    dab_channel = ihc_hed[:, :, 2]
    
    # 3. 轉為 0-255 方便 OpenCV 處理
    # 注意：這裡使用百分位數 (percentile) 縮放，可以避免極端亮點影響二值化
    v_min, v_max = np.percentile(dab_channel, (0, 99.5))
    dab_rescaled = np.clip((dab_channel - v_min) / (v_max - v_min), 0, 1)
    dab_8bit = (dab_rescaled * 255).astype(np.uint8)
    
    # 4. 二值化：提取細胞膜
    # 使用 Otsu 加上一個小的偏移量，確保只抓到比較深的咖啡色 (陽性膜)
    thresh, _ = cv2.threshold(dab_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (dab_8bit > (thresh * 0.9)).astype(np.uint8) * 255
    
    # 5. 形態學處理：連接斷裂的膜、去除雜訊
    # 閉運算：填補細胞膜的小細孔
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 移除過小的雜點 (非細胞結構)
    mask_final = morphology.remove_small_objects(mask_closed.astype(bool), min_size=100)
    
    return dab_8bit, mask_final.astype(np.uint8) * 255

def main():
    her2_name = "tile_x60928_y30720_her2.tiff"
    dish_name = "tile_x60928_y30720_dish.tiff"
    
    her2_path = os.path.join(INPUT_DIR, her2_name)
    dish_path = os.path.join(INPUT_DIR, dish_name)
    
    # 執行處理
    print(f"正在分析 {her2_name}...")
    dab_gray, membrane_mask = get_her2_membrane_mask(her2_path)
    
    # 讀取 DISH 影像用於疊合
    dish_rgb = io.imread(dish_path)[:, :, :3]
    
    # 儲存結果
    # 1. 解捲後的咖啡色灰階圖
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step1_dab_channel.png"), dab_gray)
    
    # 2. 細胞膜 Mask
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step2_membrane_mask.png"), membrane_mask)
    
    # 3. Final Overlay: 將 Mask 變成半透明紅色疊在 DISH 上
    overlay = dish_rgb.copy()
    overlay[membrane_mask > 0] = overlay[membrane_mask > 0] * 0.6 + np.array([255, 0, 0]) * 0.4
    
    io.imsave(os.path.join(OUTPUT_DIR, "final_overlay_on_dish.tiff"), overlay.astype(np.uint8))
    
    print(f"驗證圖已存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()