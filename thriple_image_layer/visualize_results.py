#!/usr/bin/env python3
"""
PNG 配準結果視覺化工具
快速檢查 PNG 配準品質和參數
注意: 這是 PNG 層級的驗證，完整的 CZI 驗證請使用 validate_czi_registration.py
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_registration_result(ref_path, mov_path, aligned_path, json_path):
    """
    視覺化配準結果
    
    Args:
        ref_path: 參考影像路徑
        mov_path: 原始移動影像路徑
        aligned_path: 配準後影像路徑
        json_path: JSON 參數檔案路徑
    """
    # 載入影像
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    mov_img = cv2.imread(mov_path, cv2.IMREAD_GRAYSCALE)
    aligned_img = cv2.imread(aligned_path, cv2.IMREAD_GRAYSCALE)
    
    # 載入 JSON 參數
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 創建視覺化
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 原始影像對比
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(ref_img, cmap='gray')
    ax1.set_title('Reference (DISH)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(mov_img, cmap='gray')
    ax2.set_title('Moving (HER2)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(aligned_img, cmap='gray')
    ax3.set_title('Aligned (HER2→DISH)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 2. 差異圖
    ax4 = plt.subplot(2, 4, 4)
    diff = cv2.absdiff(ref_img, aligned_img)
    ax4.imshow(diff, cmap='hot')
    ax4.set_title('Absolute Difference', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(ax4.imshow(diff, cmap='hot'), ax=ax4, fraction=0.046)
    
    # 3. 棋盤格疊合
    ax5 = plt.subplot(2, 4, 5)
    checkerboard = create_checkerboard_overlay(ref_img, aligned_img, square_size=500)
    ax5.imshow(checkerboard, cmap='gray')
    ax5.set_title('Checkerboard Overlay', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 4. 色彩疊合
    ax6 = plt.subplot(2, 4, 6)
    color_overlay = create_color_overlay(ref_img, aligned_img)
    ax6.imshow(color_overlay)
    ax6.set_title('Color Overlay (Red: DISH, Green: HER2)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # 5. 邊緣疊合
    ax7 = plt.subplot(2, 4, 7)
    edge_overlay = create_edge_overlay(ref_img, aligned_img)
    ax7.imshow(edge_overlay)
    ax7.set_title('Edge Overlay', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    # 6. 參數資訊
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # 提取關鍵資訊
    method = results['metadata']['registration_method']
    ssim = results['quality_metrics']['SSIM']
    psnr = results['quality_metrics']['PSNR_dB']
    
    info_text = f"""
配準方法: {method}

品質指標:
  SSIM: {ssim:.4f}
  PSNR: {psnr:.2f} dB
  MSE: {results['quality_metrics']['MSE']:.2f}
  NCC: {results['quality_metrics']['NCC']:.4f}

影像資訊:
  DISH PNG: {results['image_info']['reference']['png_size']['width']}x{results['image_info']['reference']['png_size']['height']}
  DISH CZI: {results['image_info']['reference']['czi_size']['width']}x{results['image_info']['reference']['czi_size']['height']}
  縮放: {results['image_info']['reference']['scale_factor']:.2f}x

  HER2 PNG: {results['image_info']['moving']['png_size']['width']}x{results['image_info']['moving']['png_size']['height']}
  HER2 CZI: {results['image_info']['moving']['czi_size']['width']}x{results['image_info']['moving']['czi_size']['height']}
  縮放: {results['image_info']['moving']['scale_factor']:.2f}x
"""
    
    if 'good_matches' in results['feature_matching']:
        info_text += f"""
特徵匹配:
  良好匹配: {results['feature_matching']['good_matches']}
  RANSAC內點: {results['feature_matching']['ransac_inliers']}
  RANSAC外點: {results['feature_matching']['ransac_outliers']}
"""
    
    ax8.text(0.05, 0.95, info_text, 
             transform=ax8.transAxes,
             fontsize=10,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('registration_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ 視覺化結果已儲存: registration_visualization.png")
    plt.show()


def create_checkerboard_overlay(img1, img2, square_size=100):
    """創建棋盤格疊合"""
    h, w = img1.shape
    result = np.zeros_like(img1)
    
    for i in range(0, h, square_size):
        for j in range(0, w, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                result[i:i+square_size, j:j+square_size] = img1[i:i+square_size, j:j+square_size]
            else:
                result[i:i+square_size, j:j+square_size] = img2[i:i+square_size, j:j+square_size]
    
    return result


def create_color_overlay(img1, img2, alpha=0.5):
    """創建色彩疊合 (紅-綠)"""
    # 正規化到 0-255
    img1_norm = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img2_norm = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 創建 RGB 影像
    overlay = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
    overlay[:, :, 0] = (img1_norm * alpha).astype(np.uint8)  # Red channel
    overlay[:, :, 1] = (img2_norm * alpha).astype(np.uint8)  # Green channel
    overlay[:, :, 2] = 0  # Blue channel
    
    return overlay


def create_edge_overlay(img1, img2):
    """創建邊緣疊合"""
    # Canny 邊緣檢測
    edges1 = cv2.Canny(img1, 50, 150)
    edges2 = cv2.Canny(img2, 50, 150)
    
    # 創建彩色疊合
    overlay = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
    overlay[:, :, 0] = edges1  # Red: Reference edges
    overlay[:, :, 1] = edges2  # Green: Moving edges
    overlay[:, :, 2] = np.minimum(edges1, edges2)  # Blue: Common edges
    
    return overlay


def analyze_transformation_matrix(json_path):
    """分析變換矩陣的性質"""
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    H_png = np.array(results['transformation']['png_space']['matrix'])
    H_czi = np.array(results['transformation']['czi_space']['matrix'])
    
    print("\n" + "="*70)
    print("變換矩陣分析")
    print("="*70)
    
    print("\nPNG 空間變換矩陣:")
    print(H_png)
    
    print("\nCZI 空間變換矩陣:")
    print(H_czi)
    
    # 分解變換矩陣
    print("\n變換性質分析:")
    
    # 提取旋轉、縮放、平移
    # H = [sR | t]
    #     [0  | 1]
    
    for space, H in [("PNG", H_png), ("CZI", H_czi)]:
        print(f"\n{space} 空間:")
        
        # 正規化 (除以 H[2,2])
        H_norm = H / H[2, 2]
        
        # 提取平移
        tx = H_norm[0, 2]
        ty = H_norm[1, 2]
        print(f"  平移 (Translation): tx={tx:.2f}, ty={ty:.2f}")
        
        # 提取旋轉和縮放
        a = H_norm[0, 0]
        b = H_norm[0, 1]
        c = H_norm[1, 0]
        d = H_norm[1, 1]
        
        # 計算縮放
        sx = np.sqrt(a**2 + c**2)
        sy = np.sqrt(b**2 + d**2)
        print(f"  縮放 (Scale): sx={sx:.4f}, sy={sy:.4f}")
        
        # 計算旋轉角度
        theta = np.arctan2(c, a) * 180 / np.pi
        print(f"  旋轉 (Rotation): {theta:.2f}°")
        
        # 計算剪切
        shear = np.arctan2(-b, d) * 180 / np.pi - theta
        if abs(shear) > 0.01:
            print(f"  剪切 (Shear): {shear:.2f}°")
        
        # 行列式 (面積縮放因子)
        det = np.linalg.det(H_norm[:2, :2])
        print(f"  行列式 (Det): {det:.4f} (面積變化倍數)")
        
        # 條件數 (數值穩定性)
        cond = np.linalg.cond(H_norm)
        print(f"  條件數 (Condition): {cond:.2f}", end="")
        if cond < 100:
            print(" (良好)")
        elif cond < 1000:
            print(" (可接受)")
        else:
            print(" (可能不穩定)")


def validate_transformation(json_path):
    """驗證變換矩陣的一致性"""
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    H_png = np.array(results['transformation']['png_space']['matrix'])
    H_czi = np.array(results['transformation']['czi_space']['matrix'])
    
    ref_scale = results['image_info']['reference']['scale_factor']
    mov_scale = results['image_info']['moving']['scale_factor']
    
    print("\n" + "="*70)
    print("變換矩陣驗證")
    print("="*70)
    
    # 驗證: H_czi 應該等於 S_ref * H_png * S_mov_inv
    S_ref = np.array([
        [ref_scale, 0, 0],
        [0, ref_scale, 0],
        [0, 0, 1]
    ])
    
    S_mov_inv = np.array([
        [1/mov_scale, 0, 0],
        [0, 1/mov_scale, 0],
        [0, 0, 1]
    ])
    
    H_czi_computed = S_ref @ H_png @ S_mov_inv
    
    print("\n計算的 CZI 矩陣:")
    print(H_czi_computed)
    
    print("\n儲存的 CZI 矩陣:")
    print(H_czi)
    
    print("\n差異:")
    diff = np.abs(H_czi - H_czi_computed)
    print(diff)
    print(f"最大差異: {np.max(diff):.2e}")
    
    if np.max(diff) < 1e-6:
        print("✓ 驗證通過: 矩陣轉換正確")
    else:
        print("✗ 警告: 矩陣轉換可能有誤")


def check_transformation_on_points(json_path):
    """測試變換在特定點上的效果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    H_png = np.array(results['transformation']['png_space']['matrix'])
    H_czi = np.array(results['transformation']['czi_space']['matrix'])
    
    print("\n" + "="*70)
    print("點變換測試")
    print("="*70)
    
    # 測試 PNG 空間的幾個點
    test_points_png = np.array([
        [0, 0, 1],  # 左上角
        [31877, 0, 1],  # 右上角
        [0, 24444, 1],  # 左下角
        [31877, 24444, 1],  # 右下角
        [15938, 12222, 1]  # 中心
    ])
    
    print("\nPNG 空間變換 (HER2 -> DISH):")
    print(f"{'原始點 (x, y)':<20} -> {'變換後 (x, y)':<20}")
    print("-" * 42)
    
    for pt in test_points_png:
        transformed = H_png @ pt
        transformed /= transformed[2]  # 正規化
        print(f"({pt[0]:>6.0f}, {pt[1]:>6.0f})    -> ({transformed[0]:>8.2f}, {transformed[1]:>8.2f})")
    
    # 測試 CZI 空間的對應點
    test_points_czi = np.array([
        [0, 0, 1],
        [159388, 0, 1],
        [0, 122224, 1],
        [159388, 122224, 1],
        [79694, 61112, 1]
    ])
    
    print("\nCZI 空間變換 (HER2 -> DISH):")
    print(f"{'原始點 (x, y)':<20} -> {'變換後 (x, y)':<20}")
    print("-" * 42)
    
    for pt in test_points_czi:
        transformed = H_czi @ pt
        transformed /= transformed[2]
        print(f"({pt[0]:>6.0f}, {pt[1]:>6.0f})    -> ({transformed[0]:>8.2f}, {transformed[1]:>8.2f})")


def main():
    """主程序"""
    print("PNG 配準結果視覺化與驗證工具")
    print("="*70)
    print("注意: 這是 PNG 層級的快速驗證")
    print("完整的 CZI 驗證請使用: python validate_czi_registration.py")
    print("="*70)
    
    # 檢查檔案是否存在
    required_files = [
        "DISH_mask.png",
        "Her2_mask.png",
        "Her2_aligned_to_DISH.png",
        "registration_results.json"
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"✗ 缺少以下檔案: {', '.join(missing_files)}")
        print("請先執行 image_registration_pipeline.py")
        return
    
    print("✓ 所有必要檔案都存在\n")
    
    # 1. 視覺化配準結果
    print("1. 生成視覺化圖表...")
    try:
        visualize_registration_result(
            "DISH_mask.png",
            "Her2_mask.png",
            "Her2_aligned_to_DISH.png",
            "registration_results.json"
        )
    except Exception as e:
        print(f"視覺化失敗: {e}")
    
    # 2. 分析變換矩陣
    print("\n2. 分析變換矩陣...")
    analyze_transformation_matrix("registration_results.json")
    
    # 3. 驗證矩陣轉換
    print("\n3. 驗證矩陣轉換...")
    validate_transformation("registration_results.json")
    
    # 4. 測試點變換
    print("\n4. 測試點變換...")
    check_transformation_on_points("registration_results.json")
    
    print("\n" + "="*70)
    print("✓ 驗證完成")
    print("="*70)


if __name__ == "__main__":
    main()
