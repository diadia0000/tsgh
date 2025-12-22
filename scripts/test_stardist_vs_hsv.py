#!/usr/bin/env python3
"""
測試腳本：比較 HSV 和 StarDist 檢測 DISH 紅黑點的效果
"""
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def detect_signals_hsv(dish_img_rgb):
    """使用 HSV 閾值檢測紅黑點（當前方法）"""
    dish_hsv = cv2.cvtColor(dish_img_rgb, cv2.COLOR_RGB2HSV)
    
    # 紅色檢測
    lower_red1 = np.array([0, 80, 50], dtype=np.uint8)
    upper_red1 = np.array([12, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([155, 80, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    
    mask_red = cv2.bitwise_or(
        cv2.inRange(dish_hsv, lower_red1, upper_red1),
        cv2.inRange(dish_hsv, lower_red2, upper_red2)
    )
    
    # 黑色檢測
    _, _, v_channel = cv2.split(dish_hsv)
    mask_black = (v_channel < 55).astype(np.uint8) * 255
    
    # 合併
    mask = cv2.bitwise_or(mask_red, mask_black)
    
    return mask


def detect_signals_stardist(dish_img_rgb, use_gpu=True):
    """使用 StarDist 檢測紅黑點"""
    try:
        from stardist.models import StarDist2D
        from csbdeep.utils import normalize
    except ImportError:
        print("❌ 錯誤：未安裝 StarDist")
        print("請執行：pip install stardist tensorflow")
        return None
    
    # 載入預訓練模型
    try:
        print("載入 StarDist 模型...")
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")
        print("嘗試下載模型...")
        return None
    
    # 轉換為灰度圖
    gray = cv2.cvtColor(dish_img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 正規化
    img_normalized = normalize(gray, 1, 99.8)
    
    # 預測
    print("執行 StarDist 預測...")
    labels, details = model.predict_instances(img_normalized)
    
    # 創建 mask（所有檢測到的物體）
    mask = (labels > 0).astype(np.uint8) * 255
    
    return mask, labels


def compare_methods(dish_tile_path: Path):
    """比較兩種方法的效果"""
    print("=" * 60)
    print(f"測試檔案: {dish_tile_path.name}")
    print("=" * 60)
    
    # 讀取圖像
    dish_img = cv2.imread(str(dish_tile_path))
    dish_rgb = cv2.cvtColor(dish_img, cv2.COLOR_BGR2RGB)
    
    # 方法 1：HSV
    print("\n1. HSV 閾值檢測...")
    mask_hsv = detect_signals_hsv(dish_rgb)
    count_hsv = cv2.countNonZero(mask_hsv)
    
    # 方法 2：StarDist
    print("\n2. StarDist 檢測...")
    result = detect_signals_stardist(dish_rgb)
    
    if result is None:
        print("⚠️  StarDist 不可用，只顯示 HSV 結果")
        mask_stardist = None
        labels = None
        count_stardist = 0
    else:
        mask_stardist, labels = result
        count_stardist = cv2.countNonZero(mask_stardist)
        num_objects = labels.max()
        print(f"   檢測到 {num_objects} 個獨立物體")
    
    # 視覺化比較
    print("\n3. 生成比較圖...")
    
    if mask_stardist is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = [axes]  # 包裝成 2D 陣列格式
    
    fig.suptitle(f'紅黑點檢測比較: {dish_tile_path.name}', fontsize=14, fontweight='bold')
    
    # 第一行：原圖 + HSV
    axes[0][0].imshow(dish_rgb)
    axes[0][0].set_title('原始 DISH 圖像', fontsize=12)
    axes[0][0].axis('off')
    
    axes[0][1].imshow(mask_hsv, cmap='gray')
    axes[0][1].set_title(f'HSV 閾值檢測\n像素數: {count_hsv}', fontsize=12)
    axes[0][1].axis('off')
    
    # HSV 疊加
    overlay_hsv = dish_rgb.copy()
    overlay_hsv[mask_hsv > 0] = [255, 255, 0]  # 黃色
    axes[0][2].imshow(overlay_hsv)
    axes[0][2].set_title('HSV 疊加顯示', fontsize=12)
    axes[0][2].axis('off')
    
    if mask_stardist is not None:
        # 第二行：StarDist
        axes[1][0].imshow(labels, cmap='tab20')
        axes[1][0].set_title(f'StarDist 實例分割\n物體數: {labels.max()}', fontsize=12)
        axes[1][0].axis('off')
        
        axes[1][1].imshow(mask_stardist, cmap='gray')
        axes[1][1].set_title(f'StarDist Mask\n像素數: {count_stardist}', fontsize=12)
        axes[1][1].axis('off')
        
        # StarDist 疊加
        overlay_stardist = dish_rgb.copy()
        overlay_stardist[mask_stardist > 0] = [0, 255, 255]  # 青色
        axes[1][2].imshow(overlay_stardist)
        axes[1][2].set_title('StarDist 疊加顯示', fontsize=12)
        axes[1][2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 統計
    print("\n" + "=" * 60)
    print("統計結果:")
    print(f"  HSV 方法: {count_hsv} 像素")
    if mask_stardist is not None:
        print(f"  StarDist: {count_stardist} 像素, {labels.max()} 個物體")
        print(f"  差異: {abs(count_hsv - count_stardist)} 像素")
    print("=" * 60)


def main():
    """主程式"""
    print("=" * 60)
    print("HSV vs StarDist 紅黑點檢測比較測試")
    print("=" * 60)
    
    # 設定測試圖像路徑
    tiles_dir = Path("/home/sec312/tsgh/thriple_image_layer/output/tiles_lv1/dish")
    
    if not tiles_dir.exists():
        print(f"\n❌ 錯誤：tiles 目錄不存在: {tiles_dir}")
        print("請先運行 module5_tile_generator.py 生成 tiles")
        return
    
    # 找幾張測試圖像
    test_tiles = list(tiles_dir.glob("*.tiff"))[:5]
    
    if not test_tiles:
        print(f"\n❌ 錯誤：在 {tiles_dir} 中找不到 tiff 檔案")
        return
    
    print(f"\n找到 {len(test_tiles)} 張測試圖像")
    print("將依次顯示比較結果...\n")
    
    for i, tile_path in enumerate(test_tiles, 1):
        print(f"\n{'='*60}")
        print(f"測試 {i}/{len(test_tiles)}")
        compare_methods(tile_path)
        
        if i < len(test_tiles):
            input("\n按 Enter 繼續下一張...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用戶中斷")
    except Exception as e:
        print(f"\n\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
