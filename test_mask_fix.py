#!/usr/bin/env python3
"""測試修改後的 mask 生成效果"""
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent))

from unet_mask.mask_generation import generate_pseudo_mask_v2

def test_single_image(image_path):
    """測試單張圖片的 mask 生成"""
    print(f"測試圖片: {image_path}")
    
    # 讀取圖片
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 生成 mask
    mask = generate_pseudo_mask_v2(Path(image_path))
    
    # 統計各類別像素數量
    bg_pixels = np.sum(mask == 0)
    inside_pixels = np.sum(mask == 1)
    membrane_pixels = np.sum(mask == 2)
    total_pixels = mask.size
    
    print(f"\nMask 統計:")
    print(f"  背景 (0): {bg_pixels:,} ({bg_pixels/total_pixels*100:.1f}%)")
    print(f"  細胞內部 (1): {inside_pixels:,} ({inside_pixels/total_pixels*100:.1f}%)")
    print(f"  細胞膜 (2): {membrane_pixels:,} ({membrane_pixels/total_pixels*100:.1f}%)")
    
    # 視覺化
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
    axes[1, 0].set_title(f'類別 0：背景\n({bg_pixels/total_pixels*100:.1f}%)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask == 1, cmap='gray')
    axes[1, 1].set_title(f'類別 1：細胞內部\n({inside_pixels/total_pixels*100:.1f}%)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(mask == 2, cmap='gray')
    axes[1, 2].set_title(f'類別 2：細胞膜\n({membrane_pixels/total_pixels*100:.1f}%)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存結果
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_test_result.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n結果已保存到: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    # 從訓練數據中選一張圖片測試
    test_dir = Path("unet_mask/process/train-512-lv1/strong")
    
    if test_dir.exists():
        # 找第一張圖片
        test_images = list(test_dir.glob("*.tiff")) + list(test_dir.glob("*.png"))
        if test_images:
            test_single_image(str(test_images[0]))
        else:
            print(f"在 {test_dir} 中找不到測試圖片")
    else:
        print(f"測試目錄不存在: {test_dir}")
