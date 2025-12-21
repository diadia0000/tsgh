"""
Pseudo Mask 生成模組：自動生成細胞分割的訓練標籤

此模組使用色彩空間轉換和形態學處理來生成 pseudo mask，
用於 UNet 模型的弱監督學習。
"""
import numpy as np
import cv2
from pathlib import Path
from scipy import ndimage


def generate_pseudo_mask_v2(image_path: Path) -> np.ndarray:
    """
    改進版 pseudo mask 生成：精確檢測棕色細胞膜，不影響紅黑信號點
    
    處理策略：
    1. 排除紅色區域（SISH 紅色信號）
    2. 排除黑色區域（SISH 黑色信號）  
    3. 檢測棕色區域（HER2 DAB 染色的細胞膜）
    4. 區分細胞膜/細胞內部/背景
    
    參數：
        image_path: 輸入圖像路徑
    
    回傳：
        三類別 mask（numpy array）：
        - 0 = 背景（細胞外區域）
        - 1 = 細胞內部  
        - 2 = 細胞膜（棕色 DAB 染色區域）
    """
    # ========== 讀取並轉換圖像 ==========
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 轉換到多個色彩空間以便更準確地分析顏色
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    # ========== 步驟 1：排除紅色信號點 ==========
    # 紅色在 HSV 中：Hue 接近 0 或 180，高飽和度
    # 這是 SISH 染色的紅色信號，不應被誤判為細胞膜
    lower_red1 = np.array([0, 120, 70], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([165, 120, 70], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    
    # ========== 步驟 2：排除黑色信號點 ==========
    # 黑色：低亮度值，這是 SISH 染色的黑色信號
    _, _, v_channel = cv2.split(hsv)
    mask_black = (v_channel < 70).astype(np.uint8) * 255
    
    # 合併需要排除的區域
    mask_exclude = cv2.bitwise_or(mask_red, mask_black)
    
    # ========== 步驟 3：檢測棕色（DAB 染色區域）==========
    # 方法 1：HSV 色彩空間
    # 棕色特徵：色相在橙黃色範圍，中等飽和度和亮度
    lower_brown_hsv = np.array([6, 20, 38], dtype=np.uint8)
    upper_brown_hsv = np.array([32, 185, 205], dtype=np.uint8)
    mask_brown_hsv = cv2.inRange(hsv, lower_brown_hsv, upper_brown_hsv)
    
    # 方法 2：Lab 色彩空間
    # 棕色特徵：a* 偏正（偏紅），b* 偏正（偏黃）
    l_channel, a_channel, b_channel = cv2.split(lab)
    mask_brown_lab = (
        (a_channel > 127) &  # a* > 0（偏紅）
        (b_channel > 127) &  # b* > 0（偏黃）
        (l_channel > 28) &   # 不要太暗
        (l_channel < 205)    # 不要太亮
    ).astype(np.uint8) * 255
    
    # 結合 HSV 和 Lab 的結果（使用 AND 確保精準度）
    mask_brown = cv2.bitwise_and(mask_brown_hsv, mask_brown_lab)
    
    # ========== 步驟 4：處理棕色區域中的紅黑點 ==========
    # 策略：如果紅黑點位於咖啡色區域內，將它們也視為細胞膜的一部分
    # 這可以避免紅黑點破壞細胞膜的完整性
    
    # 先膨脹棕色區域，得到大致範圍
    kernel_dilate = np.ones((7, 7), np.uint8)
    mask_brown_dilated = cv2.dilate(mask_brown, kernel_dilate, iterations=1)
    
    # 找出位於棕色區域內的紅黑點
    red_black_on_brown = cv2.bitwise_and(mask_exclude, mask_brown_dilated)
    
    # 將這些紅黑點加入棕色 mask
    mask_brown = cv2.bitwise_or(mask_brown, red_black_on_brown)
    
    # ========== 步驟 5：形態學處理 ==========
    # 策略：先用大 kernel 閉合連接斷裂處，再用小 kernel 去除雜訊
    
    # 第一步：強力閉合，連接斷裂的細胞膜
    kernel_close_large = np.ones((11, 11), np.uint8)
    mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_CLOSE, kernel_close_large)
    
    # 第二步：開運算去除小雜訊
    kernel_open = np.ones((3, 3), np.uint8)
    mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_OPEN, kernel_open)
    
    # 第三步：再次輕度閉合，修補可能產生的小缺口
    kernel_close_small = np.ones((5, 5), np.uint8)
    mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_CLOSE, kernel_close_small)
    
    # ========== 步驟 6：填充得到細胞內部 ==========
    # 使用孔洞填充演算法找出被細胞膜包圍的區域
    mask_filled = ndimage.binary_fill_holes(mask_brown).astype(np.uint8) * 255
    
    # 細胞內部 = 填充後的區域 - 原始棕色區域
    mask_inside = cv2.subtract(mask_filled, mask_brown)
    
    # ========== 步驟 7：生成最終三類別 mask ==========
    final_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    final_mask[mask_inside > 0] = 1   # 細胞內部
    final_mask[mask_brown > 0] = 2    # 細胞膜（棕色）
    # 背景保持為 0
    
    return final_mask


def visualize_mask_quality(image_path: Path, mask: np.ndarray):
    """
    視覺化 mask 品質，檢查是否影響紅黑信號點
    
    參數：
        image_path: 原始圖像路徑
        mask: 生成的 mask
    """
    import matplotlib.pyplot as plt
    
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原圖
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('原始圖像')
    axes[0, 0].axis('off')
    
    # Mask（三類別）
    axes[0, 1].imshow(mask, cmap='jet', vmin=0, vmax=2)
    axes[0, 1].set_title('三類別 Mask\n(0=背景, 1=細胞內, 2=細胞膜)')
    axes[0, 1].axis('off')
    
    # 疊加顯示
    overlay = img_rgb.copy()
    overlay[mask == 1] = [100, 255, 100]  # 淺綠 = 細胞內部
    overlay[mask == 2] = [255, 100, 100]  # 淺紅 = 細胞膜
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('疊加顯示')
    axes[0, 2].axis('off')
    
    # 分別顯示三個類別
    axes[1, 0].imshow(mask == 0, cmap='gray')
    axes[1, 0].set_title('類別 0：背景')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask == 1, cmap='gray')
    axes[1, 1].set_title('類別 1：細胞內部')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(mask == 2, cmap='gray')
    axes[1, 2].set_title('類別 2：細胞膜（棕色）')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def interactive_batch_visualize(image_paths: list):
    """
    互動式批次視覺化：支援使用鍵盤或按鈕瀏覽多張圖像
    
    參數：
        image_paths: 圖像路徑列表
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    
    if not image_paths:
        print("沒有圖像需要視覺化")
        return
    
    # 預載入所有 masks（避免重複計算）
    print(f"正在預載入 {len(image_paths)} 張圖像的 masks...")
    data = []
    for i, img_path in enumerate(image_paths, 1):
        try:
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = generate_pseudo_mask_v2(img_path)
            data.append((img_path.name, img_rgb, mask))
            if i % 5 == 0:
                print(f"  已載入 {i}/{len(image_paths)}")
        except Exception as e:
            print(f"  跳過 {img_path.name}: {e}")
    
    if not data:
        print("沒有成功載入的圖像")
        return
    
    print(f"載入完成！共 {len(data)} 張圖像\n")
    
    # 建立 figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'', fontsize=16, y=0.98)
    
    # 當前索引（使用 list 以便在閉包中修改）
    current_idx = [0]
    
    def update_display(idx):
        """更新顯示內容"""
        name, img_rgb, mask = data[idx]
        
        # 更新標題
        fig.suptitle(f'[{idx+1}/{len(data)}] {name}', fontsize=14, y=0.98)
        
        # 清空所有 axes
        for ax in axes.flat:
            ax.clear()
            ax.axis('off')
        
        # 原圖
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('原始圖像', fontsize=12)
        
        # Mask（三類別）
        axes[0, 1].imshow(mask, cmap='jet', vmin=0, vmax=2)
        axes[0, 1].set_title('三類別 Mask\n(0=背景, 1=細胞內, 2=細胞膜)', fontsize=11)
        
        # 疊加顯示
        overlay = img_rgb.copy()
        overlay[mask == 1] = [100, 255, 100]
        overlay[mask == 2] = [255, 100, 100]
        axes[0, 2].imshow(overlay)
        axes[0, 2].set_title('疊加顯示', fontsize=12)
        
        # 三個類別分別顯示
        axes[1, 0].imshow(mask == 0, cmap='gray')
        axes[1, 0].set_title('類別 0：背景', fontsize=11)
        
        axes[1, 1].imshow(mask == 1, cmap='gray')
        axes[1, 1].set_title('類別 1：細胞內部', fontsize=11)
        
        axes[1, 2].imshow(mask == 2, cmap='gray')
        axes[1, 2].set_title('類別 2：細胞膜（棕色）', fontsize=11)
        
        plt.tight_layout()
        fig.canvas.draw_idle()
    
    def on_key(event):
        """鍵盤事件：左右箭頭切換圖像"""
        if event.key == 'right' or event.key == 'n':
            current_idx[0] = (current_idx[0] + 1) % len(data)
            update_display(current_idx[0])
        elif event.key == 'left' or event.key == 'p':
            current_idx[0] = (current_idx[0] - 1) % len(data)
            update_display(current_idx[0])
        elif event.key == 'q':
            plt.close(fig)
    
    # 連接鍵盤事件
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # 顯示第一張
    update_display(0)
    
    print("操作提示：")
    print("  - 使用工具列的 ← → 按鈕切換圖像")
    print("  - 或按鍵盤 ← → 或 p/n 切換")
    print("  - 按 q 退出\n")
    
    plt.show()


def batch_generate_masks(paths_dict: dict, output_dir: Path):
    """
    批次生成所有圖像的 pseudo masks
    
    參數：
        paths_dict: 字典，key 為類別名稱，value 為該類別的圖像路徑列表
        output_dir: mask 輸出目錄
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    total = sum(len(paths) for paths in paths_dict.values())
    processed = 0
    
    for category, paths in paths_dict.items():
        cat_dir = output_dir / category
        cat_dir.mkdir(exist_ok=True)
        
        for img_path in paths:
            try:
                # 生成 mask
                mask = generate_pseudo_mask_v2(img_path)
                
                # 儲存 mask（使用原檔名）
                mask_path = cat_dir / f"{img_path.stem}_mask.png"
                cv2.imwrite(str(mask_path), mask)
                
                processed += 1
                if processed % 10 == 0:
                    print(f"已處理 {processed}/{total} 張圖像...")
                    
            except Exception as e:
                print(f"處理錯誤 {img_path}: {e}")
    
    print(f"完成！已在 {output_dir} 生成 {processed} 個 masks")


# ===== 主程式：批次生成 masks =====
if __name__ == "__main__":
    from pathlib import Path
    import time
    
    # 設定路徑
    input_dir = Path("unet_mask/process/train-512-lv1")
    output_dir = Path("unet_mask/process/pseudo_masks")
    
    # 檢查輸入目錄是否存在
    if not input_dir.exists():
        print(f"錯誤：輸入目錄不存在: {input_dir}")
        print("請檢查路徑是否正確")
        exit(1)
    
    # 4 個子資料夾（HER2 分類）
    categories = ['blank', 'negative', 'strong', 'weak']
    
    # 收集所有圖像路徑
    all_images = {}
    total_count = 0
    
    for cat in categories:
        cat_dir = input_dir / cat
        if cat_dir.exists():
            # 支援多種圖像格式
            images = list(cat_dir.glob('*.tiff')) + \
                     list(cat_dir.glob('*.tif')) + \
                     list(cat_dir.glob('*.png')) + \
                     list(cat_dir.glob('*.jpg'))
            all_images[cat] = images
            total_count += len(images)
            print(f"找到 {cat}: {len(images)} 張圖像")
        else:
            print(f"警告：子目錄不存在: {cat_dir}")
            all_images[cat] = []
    
    print(f"\n總共 {total_count} 張圖像")
    print(f"輸出目錄: {output_dir}")
    print(f"開始批次生成 masks...\n")
    
    # 建立輸出目錄
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 批次處理
    start_time = time.time()
    processed = 0
    failed = 0
    
    for cat, images in all_images.items():
        if not images:
            continue
            
        # 建立類別輸出目錄
        cat_output_dir = output_dir / cat
        cat_output_dir.mkdir(exist_ok=True)
        
        print(f"\n處理類別: {cat} ({len(images)} 張)")
        
        for i, img_path in enumerate(images, 1):
            try:
                # 生成 mask
                mask = generate_pseudo_mask_v2(img_path)
                
                # 儲存 mask（保持原檔名，改副檔名為 .png）
                mask_filename = f"{img_path.stem}_mask.png"
                mask_path = cat_output_dir / mask_filename
                cv2.imwrite(str(mask_path), mask)
                
                processed += 1
                
                # 每 10 張顯示一次進度
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    speed = processed / elapsed if elapsed > 0 else 0
                    remaining = (total_count - processed) / speed if speed > 0 else 0
                    print(f"  {cat}: {i}/{len(images)} | "
                          f"總進度: {processed}/{total_count} | "
                          f"速度: {speed:.1f} 張/秒 | "
                          f"預計剩餘: {remaining:.0f} 秒")
                
            except Exception as e:
                print(f"  錯誤: {img_path.name} - {e}")
                failed += 1
    
    # 完成統計
    elapsed_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"處理完成！")
    print(f"  成功: {processed} 張")
    print(f"  失敗: {failed} 張")
    print(f"  總耗時: {elapsed_time:.1f} 秒")
    print(f"  平均速度: {processed/elapsed_time:.2f} 張/秒")
    print(f"  輸出目錄: {output_dir.absolute()}")
    print(f"{'='*50}")
    
    # 可選：生成所有樣本的視覺化檢查
    print("\n生成所有圖像的視覺化檢查...")
    all_sample_images = []
    for cat in categories:
        if cat in all_images and all_images[cat]:
            # 添加所有圖像
            all_sample_images.extend(all_images[cat])
    
    if all_sample_images:
        print(f"共 {len(all_sample_images)} 張圖像進行檢查")
        interactive_batch_visualize(all_sample_images)
