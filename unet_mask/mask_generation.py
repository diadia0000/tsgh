"""
Pseudo Mask 生成模組：使用分離的 HER2 和 DISH 圖像自動生成細胞分割的訓練標籤

此模組使用 Color Deconvolution 和形態學處理來生成 pseudo mask，
用於 UNet 模型的弱監督學習。
"""
import numpy as np
import cv2
from pathlib import Path
from scipy import ndimage


def color_deconvolution_hed(img_rgb: np.ndarray) -> tuple:
    """
    Color Deconvolution：分離 Hematoxylin、Eosin、DAB 染色通道
    
    使用 Ruifrok and Johnston (2001) 的標準 HED 染色向量
    
    參數：
        img_rgb: RGB 圖像 (uint8)
    
    回傳：
        (h_channel, e_channel, d_channel): 三個分離的染色通道
    """
    # HED 染色向量矩陣（Ruifrok標準）
    # 每列代表一種染色的 RGB 吸收係數
    hed_matrix = np.array([
        [0.65, 0.70, 0.29],  # Hematoxylin（藍紫色）
        [0.07, 0.99, 0.11],  # Eosin（粉紅色）
        [0.27, 0.57, 0.78],  # DAB（棕色）
    ])
    
    # 計算逆矩陣用於解卷積
    hed_matrix_inv = np.linalg.inv(hed_matrix)
    
    # 轉換到光學密度（OD）空間
    # OD = -log10(I/I0)，其中 I0 = 255（白光）
    img_float = img_rgb.astype(np.float32) + 1  # 避免 log(0)
    od = -np.log10(img_float / 255.0)
    
    # 應用逆矩陣進行解卷積
    od_flat = od.reshape(-1, 3)
    hed_channels = np.dot(od_flat, hed_matrix_inv.T)
    hed_channels = hed_channels.reshape(img_rgb.shape)
    
    # 分離三個通道
    h_channel = hed_channels[:, :, 0]  # Hematoxylin
    e_channel = hed_channels[:, :, 1]  # Eosin
    d_channel = hed_channels[:, :, 2]  # DAB
    
    return h_channel, e_channel, d_channel


def generate_pseudo_mask_v3(her2_path: Path, dish_path: Path) -> np.ndarray:
    """
    使用分離的 HER2 和 DISH 圖像生成精確的細胞分割 mask（推薦方法）
    
    處理策略：
    1. 使用 HER2 圖像做 Color Deconvolution 提取 DAB 通道（完全不受紅黑點干擾）
    2. 使用 DISH 圖像檢測紅黑點位置
    3. 結合兩者生成最終 mask
    
    參數：
        her2_path: HER2 圖像路徑（純棕色細胞膜，無紅黑點）
        dish_path: DISH 圖像路徑（含紅黑信號點）
    
    回傳：
        三類別 mask（numpy array）：
        - 0 = 背景（細胞外區域 + 紅黑信號點）
        - 1 = 細胞內部  
        - 2 = 細胞膜（棕色 DAB 染色區域）
    """
    # ========== 讀取 HER2 圖像 ==========
    her2_img = cv2.imread(str(her2_path))
    her2_rgb = cv2.cvtColor(her2_img, cv2.COLOR_BGR2RGB)
    
    # ========== 讀取 DISH 圖像 ==========
    dish_img = cv2.imread(str(dish_path))
    dish_rgb = cv2.cvtColor(dish_img, cv2.COLOR_BGR2RGB)
    dish_hsv = cv2.cvtColor(dish_rgb, cv2.COLOR_RGB2HSV)
    
    # ========== 步驟 1：從 DISH 圖像檢測紅色信號點 ==========
    lower_red1 = np.array([0, 80, 50], dtype=np.uint8)
    upper_red1 = np.array([12, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([155, 80, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    
    mask_red = cv2.bitwise_or(
        cv2.inRange(dish_hsv, lower_red1, upper_red1),
        cv2.inRange(dish_hsv, lower_red2, upper_red2)
    )
    
    # ========== 步驟 2：從 DISH 圖像檢測黑色信號點 ==========
    _, _, v_channel = cv2.split(dish_hsv)
    mask_black = (v_channel < 55).astype(np.uint8) * 255
    
    # 合併紅黑點 mask
    mask_red_black = cv2.bitwise_or(mask_red, mask_black)
    
    # ========== 步驟 3：從 HER2 圖像提取 DAB 通道 ==========
    # 關鍵：HER2 圖像沒有紅黑點，Color Deconvolution 結果會更準確
    h_channel, e_channel, d_channel = color_deconvolution_hed(her2_rgb)
    
    # DAB 通道正規化到 0-255
    d_normalized = np.clip(d_channel, 0, None)
    if d_normalized.max() > 0:
        d_normalized = (d_normalized / d_normalized.max() * 255).astype(np.uint8)
    else:
        d_normalized = np.zeros_like(d_channel, dtype=np.uint8)
    
    # ========== 步驟 4：使用 Otsu 閾值檢測 DAB 染色區域 ==========
    _, mask_dab_otsu = cv2.threshold(d_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 額外的閾值過濾
    dab_threshold = 20  # 降低閾值因為 HER2 圖像更乾淨
    mask_dab_strong = (d_normalized > dab_threshold).astype(np.uint8) * 255
    
    # 結合 Otsu 和強度閾值
    mask_brown = cv2.bitwise_and(mask_dab_otsu, mask_dab_strong)
    
    # ========== 步驟 5：形態學處理 ==========
    # 閉合操作連接細胞膜
    kernel_close = np.ones((7, 7), np.uint8)
    mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_CLOSE, kernel_close)
    
    # 開運算去除小雜訊
    kernel_open = np.ones((3, 3), np.uint8)
    mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_OPEN, kernel_open)
    
    # ========== 步驟 6：智能填充得到細胞內部 ==========
    mask_filled = ndimage.binary_fill_holes(mask_brown).astype(np.uint8) * 255
    mask_inside_raw = cv2.subtract(mask_filled, mask_brown)
    
    # 使用連通組件分析過濾填充區域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_inside_raw, connectivity=8
    )
    
    mask_inside = np.zeros_like(mask_inside_raw)
    img_area = mask_brown.shape[0] * mask_brown.shape[1]
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if 30 < area < img_area * 0.35:
            mask_inside[labels == label] = 255
    
    # ========== 步驟 7：生成最終三類別 mask ==========
    final_mask = np.zeros(her2_rgb.shape[:2], dtype=np.uint8)
    final_mask[mask_inside > 0] = 1   # 細胞內部
    final_mask[mask_brown > 0] = 2    # 細胞膜
    
    # ========== 步驟 8：強制保留紅黑點為背景 ==========
    final_mask[mask_red_black > 0] = 0
    
    return final_mask


def batch_generate_masks_v3(
    her2_dir: Path,
    dish_dir: Path,
    output_dir: Path,
    visualize_samples: int = 0
) -> None:
    """
    批次生成 masks（使用分離的 HER2 和 DISH tiles）
    
    參數：
        her2_dir: HER2 tiles 目錄
        dish_dir: DISH tiles 目錄
        output_dir: 輸出目錄
        visualize_samples: 隨機抽取幾張進行視覺化檢查（0 = 不檢查）
    """
    import time
    
    # 收集所有 tiles
    her2_tiles = sorted(list(her2_dir.glob("*.tiff")) + list(her2_dir.glob("*.tif")))
    dish_tiles = sorted(list(dish_dir.glob("*.tiff")) + list(dish_dir.glob("*.tif")))
    
    if len(her2_tiles) != len(dish_tiles):
        raise ValueError(f"HER2 和 DISH tiles 數量不同！HER2: {len(her2_tiles)}, DISH: {len(dish_tiles)}")
    
    print("=" * 60)
    print(f"使用 HER2 + DISH 圖像生成 Pseudo Masks (V3)")
    print("=" * 60)
    print(f"HER2 tiles: {her2_dir}")
    print(f"DISH tiles: {dish_dir}")
    print(f"輸出目錄: {output_dir}")
    print(f"總共: {len(her2_tiles)} 組 tiles")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    processed = 0
    failed = 0
    
    for her2_tile, dish_tile in zip(her2_tiles, dish_tiles):
        try:
            # 確保檔名對應
            if her2_tile.stem != dish_tile.stem:
                print(f"⚠️  警告：檔名不對應！{her2_tile.name} vs {dish_tile.name}")
                continue
            
            # 生成 mask
            mask = generate_pseudo_mask_v3(her2_tile, dish_tile)
            
            # 儲存 mask
            mask_filename = f"{her2_tile.stem}_mask.png"
            mask_path = output_dir / mask_filename
            cv2.imwrite(str(mask_path), mask)
            
            processed += 1
            
            # 進度顯示
            if processed % 50 == 0:
                elapsed = time.time() - start_time
                speed = processed / elapsed if elapsed > 0 else 0
                remaining = (len(her2_tiles) - processed) / speed if speed > 0 else 0
                print(f"進度: {processed}/{len(her2_tiles)} "
                      f"({processed/len(her2_tiles)*100:.1f}%) | "
                      f"速度: {speed:.1f} 張/秒 | "
                      f"預計剩餘: {remaining:.0f} 秒")
        
        except Exception as e:
            print(f"❌ 錯誤: {her2_tile.name} - {e}")
            failed += 1
    
    # 完成統計
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ 處理完成！")
    print(f"   成功: {processed} 張")
    print(f"   失敗: {failed} 張")
    print(f"   總耗時: {elapsed_time:.1f} 秒")
    print(f"   平均速度: {processed/elapsed_time:.2f} 張/秒")
    print(f"   輸出目錄: {output_dir.absolute()}")
    print("=" * 60)
    
    # 視覺化檢查
    if visualize_samples > 0 and processed > 0:
        import random
        print(f"\n隨機抽取 {visualize_samples} 張進行視覺化檢查...")
        sample_tiles = random.sample(her2_tiles[:processed], min(visualize_samples, processed))
        
        # 準備視覺化資料
        vis_data = []
        for her2_tile in sample_tiles:
            dish_tile = dish_dir / her2_tile.name
            mask_path = output_dir / f"{her2_tile.stem}_mask.png"
            
            if mask_path.exists():
                # 讀取圖像（使用 HER2 作為顯示）
                img = cv2.imread(str(her2_tile))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 讀取 mask
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                vis_data.append((img_rgb, mask, her2_tile.name))
        
        if vis_data:
            interactive_batch_visualize_with_masks(vis_data)


def interactive_batch_visualize_with_masks(data: list):
    """互動式視覺化（已有 mask 資料）"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.canvas.manager.set_window_title('Mask 視覺化檢查 (使用 ← → 或 p/n 切換，q 退出)')
    
    current_idx = [0]
    
    def update_display(idx):
        img_rgb, mask, name = data[idx]
        
        for ax_row in axes:
            for ax in ax_row:
                ax.clear()
                ax.axis('off')
        
        fig.suptitle(f'[{idx+1}/{len(data)}] {name}', fontsize=14, fontweight='bold')
        
        # 原圖
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('原始圖像 (HER2)', fontsize=12)
        
        # Mask
        axes[0, 1].imshow(mask, cmap='tab10', vmin=0, vmax=2)
        axes[0, 1].set_title(f'Mask (0=背景, 1=內部, 2=膜)', fontsize=12)
        
        # 疊加
        overlay = img_rgb.copy()
        overlay[mask == 1] = [100, 255, 100]
        overlay[mask == 2] = [255, 100, 100]
        axes[0, 2].imshow(overlay)
        axes[0, 2].set_title('疊加顯示', fontsize=12)
        
        # 三個類別
        axes[1, 0].imshow(mask == 0, cmap='gray')
        axes[1, 0].set_title('類別 0：背景', fontsize=11)
        
        axes[1, 1].imshow(mask == 1, cmap='gray')
        axes[1, 1].set_title('類別 1：細胞內部', fontsize=11)
        
        axes[1, 2].imshow(mask == 2, cmap='gray')
        axes[1, 2].set_title('類別 2：細胞膜', fontsize=11)
        
        plt.tight_layout()
        fig.canvas.draw_idle()
    
    def on_key(event):
        if event.key in ['right', 'n']:
            current_idx[0] = (current_idx[0] + 1) % len(data)
            update_display(current_idx[0])
        elif event.key in ['left', 'p']:
            current_idx[0] = (current_idx[0] - 1) % len(data)
            update_display(current_idx[0])
        elif event.key == 'q':
            plt.close(fig)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    update_display(0)
    
    print("操作提示：")
    print("  - 按 ← → 或 p/n 切換圖像")
    print("  - 按 q 退出\n")
    
    plt.show()


# ===== 主程式 =====
if __name__ == "__main__":
    print("=" * 80)
    print("請使用新的流程腳本：")
    print("  python scripts/generate_masks_from_separated_images.py")
    print()
    print("或直接導入函數使用：")
    print("  from unet_mask.mask_generation import batch_generate_masks_v3")
    print("=" * 80)
