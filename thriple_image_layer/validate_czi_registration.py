#!/usr/bin/env python3
"""
CZI 影像配準驗證工具
使用真實的 CZI 檔案進行配準驗證

工作流程:
1. 從 PNG 配準獲得變換矩陣
2. 將變換矩陣轉換到 CZI 空間
3. 以 0.2x 縮放載入 CZI 影像
4. 調整變換矩陣以適應 0.2x 縮放
5. 應用變換並視覺化結果
"""

import cv2
import numpy as np
import json
from pathlib import Path
from aicspylibczi import CziFile
import matplotlib.pyplot as plt
import gc


class CZIRegistrationValidator:
    """CZI 配準驗證器"""
    
    def __init__(self, json_path, ref_czi_path, mov_czi_path, scale_factor=0.2):
        """
        初始化驗證器
        
        Args:
            json_path: registration_results.json 路徑
            ref_czi_path: 參考影像 CZI 檔案路徑 (DISH)
            mov_czi_path: 移動影像 CZI 檔案路徑 (HER2)
            scale_factor: CZI 載入縮放比例 (0.2 = 20%)
        """
        self.json_path = Path(json_path)
        self.ref_czi_path = Path(ref_czi_path)
        self.mov_czi_path = Path(mov_czi_path)
        self.scale_factor = scale_factor
        
        # 載入配準參數
        self.load_registration_params()
        
        print(f"CZI 驗證配置:")
        print(f"  參考 CZI: {self.ref_czi_path.name}")
        print(f"  移動 CZI: {self.mov_czi_path.name}")
        print(f"  載入縮放: {self.scale_factor}x ({self.scale_factor*100:.0f}%)")
        
    def load_registration_params(self):
        """載入 JSON 配準參數"""
        print(f"\n正在載入配準參數: {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        
        # 提取 CZI 空間的變換矩陣
        self.H_czi_full = np.array(self.params['transformation']['czi_space']['matrix'])
        
        # 提取影像尺寸資訊
        self.ref_czi_width = self.params['image_info']['reference']['czi_size']['width']
        self.ref_czi_height = self.params['image_info']['reference']['czi_size']['height']
        self.mov_czi_width = self.params['image_info']['moving']['czi_size']['width']
        self.mov_czi_height = self.params['image_info']['moving']['czi_size']['height']
        
        print(f"  ✓ 參考影像 CZI: {self.ref_czi_width} x {self.ref_czi_height}")
        print(f"  ✓ 移動影像 CZI: {self.mov_czi_width} x {self.mov_czi_height}")
        print(f"  ✓ 變換矩陣已載入")
        
    def compute_scaled_transformation(self):
        """
        計算適用於縮放 CZI 的變換矩陣
        
        當我們以 scale_factor (例如 0.2) 載入 CZI 時，
        需要調整變換矩陣以匹配新的尺度
        
        H_scaled = S × H_czi_full × S^(-1)
        其中 S = diag(scale_factor, scale_factor, 1)
        """
        print(f"\n正在計算 {self.scale_factor}x 縮放的變換矩陣...")
        
        # 縮放矩陣
        S = np.array([
            [self.scale_factor, 0, 0],
            [0, self.scale_factor, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        S_inv = np.array([
            [1/self.scale_factor, 0, 0],
            [0, 1/self.scale_factor, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 計算縮放後的變換矩陣
        # 從 mov_czi_scaled -> ref_czi_scaled
        self.H_czi_scaled = S @ self.H_czi_full @ S_inv
        
        # 計算縮放後的影像尺寸
        self.ref_czi_scaled_width = int(self.ref_czi_width * self.scale_factor)
        self.ref_czi_scaled_height = int(self.ref_czi_height * self.scale_factor)
        self.mov_czi_scaled_width = int(self.mov_czi_width * self.scale_factor)
        self.mov_czi_scaled_height = int(self.mov_czi_height * self.scale_factor)
        
        print(f"  ✓ 縮放後參考影像: {self.ref_czi_scaled_width} x {self.ref_czi_scaled_height}")
        print(f"  ✓ 縮放後移動影像: {self.mov_czi_scaled_width} x {self.mov_czi_scaled_height}")
        print(f"  ✓ 變換矩陣已調整")
        
        return self.H_czi_scaled
    
    def load_czi_mosaic(self, czi_path, scale_factor):
        """
        載入 CZI 馬賽克影像
        
        Args:
            czi_path: CZI 檔案路徑
            scale_factor: 縮放比例
            
        Returns:
            image: 灰階影像 numpy array
        """
        print(f"\n正在載入 CZI: {czi_path.name}")
        print(f"  使用縮放: {scale_factor}x")
        
        czi = CziFile(czi_path)
        
        try:
            # 獲取馬賽克邊界框
            bbox = czi.get_mosaic_bounding_box()
            print(f"  馬賽克邊界: ({bbox.x}, {bbox.y}), 尺寸: {bbox.w} x {bbox.h}")
            
            # 讀取整個馬賽克區域（使用縮放）
            # 注意: read_mosaic 需要 (x, y, w, h) 元組和 scale_factor
            region = (bbox.x, bbox.y, bbox.w, bbox.h)
            
            print(f"  正在讀取影像數據...")
            # 讀取第一個通道 (C=0)
            image = czi.read_mosaic(region, scale_factor=scale_factor, C=0)
            
            if image is None or image.size == 0:
                raise ValueError("讀取的影像為空")
            
            # 處理維度
            print(f"  原始形狀: {image.shape}")
            
            # 移除多餘的維度並取得灰階影像
            image = np.squeeze(image)
            
            # 如果是 RGB/BGR，轉換為灰階
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    # BGR to Gray
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif image.shape[2] == 4:
                    # BGRA to Gray
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            
            print(f"  ✓ 載入完成: {image.shape}, dtype: {image.dtype}")
            
            return image
            
        except Exception as e:
            raise RuntimeError(f"載入 CZI 失敗: {str(e)}")
        
        finally:
            del czi
            gc.collect()
    
    def apply_transformation_to_czi(self):
        """
        載入 CZI 影像並應用變換
        
        Returns:
            ref_image: 參考影像
            mov_image: 原始移動影像
            aligned_image: 配準後的移動影像
        """
        print("\n" + "="*70)
        print("開始 CZI 配準驗證")
        print("="*70)
        
        # 1. 計算縮放後的變換矩陣
        H_scaled = self.compute_scaled_transformation()
        
        # 2. 載入參考影像 (DISH)
        print(f"\n[1/3] 載入參考影像 (DISH)...")
        ref_image = self.load_czi_mosaic(self.ref_czi_path, self.scale_factor)
        
        # 3. 載入移動影像 (HER2)
        print(f"\n[2/3] 載入移動影像 (HER2)...")
        mov_image = self.load_czi_mosaic(self.mov_czi_path, self.scale_factor)
        
        # 4. 應用變換
        print(f"\n[3/3] 應用變換矩陣...")
        print(f"  目標尺寸: {ref_image.shape[1]} x {ref_image.shape[0]}")
        
        aligned_image = cv2.warpPerspective(
            mov_image,
            H_scaled,
            (ref_image.shape[1], ref_image.shape[0]),  # (width, height)
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        print(f"  ✓ 變換完成")
        
        return ref_image, mov_image, aligned_image
    
    def calculate_quality_metrics(self, ref_image, aligned_image):
        """計算配準品質指標"""
        print("\n正在計算品質指標...")
        
        # 確保影像尺寸相同
        if ref_image.shape != aligned_image.shape:
            print(f"  警告: 影像尺寸不匹配 {ref_image.shape} vs {aligned_image.shape}")
            return {}
        
        # 計算有效區域 (排除黑邊)
        mask = aligned_image > 0
        
        if not mask.any():
            print("  警告: 配準影像全為黑色")
            return {}
        
        # 1. MSE (只在有效區域)
        mse = np.mean((ref_image[mask].astype(float) - aligned_image[mask].astype(float)) ** 2)
        
        # 2. PSNR
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        # 3. SSIM
        from skimage.metrics import structural_similarity
        ssim = structural_similarity(ref_image, aligned_image, data_range=255)
        
        # 4. NCC (正規化互相關)
        ref_norm = (ref_image[mask] - np.mean(ref_image[mask])) / (np.std(ref_image[mask]) + 1e-8)
        aligned_norm = (aligned_image[mask] - np.mean(aligned_image[mask])) / (np.std(aligned_image[mask]) + 1e-8)
        ncc = np.mean(ref_norm * aligned_norm)
        
        # 5. 重疊百分比
        overlap_percent = (mask.sum() / mask.size) * 100
        
        metrics = {
            'MSE': float(mse),
            'PSNR_dB': float(psnr),
            'SSIM': float(ssim),
            'NCC': float(ncc),
            'Overlap_Percent': float(overlap_percent)
        }
        
        print(f"  MSE: {mse:.2f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")
        print(f"  NCC: {ncc:.4f}")
        print(f"  重疊區域: {overlap_percent:.1f}%")
        
        return metrics
    
    def visualize_results(self, ref_image, mov_image, aligned_image, metrics, output_path="czi_validation_results.png"):
        """視覺化 CZI 配準結果"""
        print(f"\n正在生成視覺化圖表...")
        
        fig = plt.figure(figsize=(20, 14))
        
        # 1. 參考影像
        ax1 = plt.subplot(3, 4, 1)
        ax1.imshow(ref_image, cmap='gray', vmin=0, vmax=255)
        ax1.set_title(f'Reference (DISH) CZI\n{ref_image.shape[1]}x{ref_image.shape[0]}', 
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. 移動影像
        ax2 = plt.subplot(3, 4, 2)
        ax2.imshow(mov_image, cmap='gray', vmin=0, vmax=255)
        ax2.set_title(f'Moving (HER2) CZI\n{mov_image.shape[1]}x{mov_image.shape[0]}', 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 3. 配準後影像
        ax3 = plt.subplot(3, 4, 3)
        ax3.imshow(aligned_image, cmap='gray', vmin=0, vmax=255)
        ax3.set_title(f'Aligned (HER2→DISH)\n{aligned_image.shape[1]}x{aligned_image.shape[0]}', 
                     fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # 4. 差異圖
        ax4 = plt.subplot(3, 4, 4)
        diff = cv2.absdiff(ref_image, aligned_image)
        im4 = ax4.imshow(diff, cmap='hot')
        ax4.set_title('Absolute Difference', fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        # 5. 棋盤格疊合 (大)
        ax5 = plt.subplot(3, 4, 5)
        checkerboard = self.create_checkerboard_overlay(ref_image, aligned_image, square_size=500)
        ax5.imshow(checkerboard, cmap='gray')
        ax5.set_title('Checkerboard (500px)', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        # 6. 棋盤格疊合 (小)
        ax6 = plt.subplot(3, 4, 6)
        checkerboard_small = self.create_checkerboard_overlay(ref_image, aligned_image, square_size=200)
        ax6.imshow(checkerboard_small, cmap='gray')
        ax6.set_title('Checkerboard (200px)', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        # 7. 色彩疊合
        ax7 = plt.subplot(3, 4, 7)
        color_overlay = self.create_color_overlay(ref_image, aligned_image)
        ax7.imshow(color_overlay)
        ax7.set_title('Color Overlay\n(Red: DISH, Green: HER2)', fontsize=12, fontweight='bold')
        ax7.axis('off')
        
        # 8. 邊緣疊合
        ax8 = plt.subplot(3, 4, 8)
        edge_overlay = self.create_edge_overlay(ref_image, aligned_image)
        ax8.imshow(edge_overlay)
        ax8.set_title('Edge Overlay', fontsize=12, fontweight='bold')
        ax8.axis('off')
        
        # 9-10. 局部放大 (左上角)
        crop_size = 1000
        ax9 = plt.subplot(3, 4, 9)
        ref_crop = ref_image[:crop_size, :crop_size]
        ax9.imshow(ref_crop, cmap='gray')
        ax9.set_title('Reference (Top-Left Crop)', fontsize=10)
        ax9.axis('off')
        
        ax10 = plt.subplot(3, 4, 10)
        aligned_crop = aligned_image[:crop_size, :crop_size]
        ax10.imshow(aligned_crop, cmap='gray')
        ax10.set_title('Aligned (Top-Left Crop)', fontsize=10)
        ax10.axis('off')
        
        # 11. 色彩疊合 (局部)
        ax11 = plt.subplot(3, 4, 11)
        color_crop = self.create_color_overlay(ref_crop, aligned_crop)
        ax11.imshow(color_crop)
        ax11.set_title('Overlay (Top-Left Crop)', fontsize=10)
        ax11.axis('off')
        
        # 12. 參數資訊
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        info_text = f"""
CZI 配準驗證結果

載入設定:
  縮放比例: {self.scale_factor}x ({self.scale_factor*100:.0f}%)
  
實際尺寸:
  DISH: {ref_image.shape[1]} x {ref_image.shape[0]}
  HER2: {mov_image.shape[1]} x {mov_image.shape[0]}

原始 CZI 尺寸:
  DISH: {self.ref_czi_width} x {self.ref_czi_height}
  HER2: {self.mov_czi_width} x {self.mov_czi_height}

品質指標:
  SSIM: {metrics.get('SSIM', 0):.4f}
  PSNR: {metrics.get('PSNR_dB', 0):.2f} dB
  MSE: {metrics.get('MSE', 0):.2f}
  NCC: {metrics.get('NCC', 0):.4f}
  重疊: {metrics.get('Overlap_Percent', 0):.1f}%

配準方法:
  {self.params['metadata']['registration_method']}

變換矩陣 (縮放後):
  [{self.H_czi_scaled[0,0]:>8.3f} {self.H_czi_scaled[0,1]:>8.3f} {self.H_czi_scaled[0,2]:>9.1f}]
  [{self.H_czi_scaled[1,0]:>8.3f} {self.H_czi_scaled[1,1]:>8.3f} {self.H_czi_scaled[1,2]:>9.1f}]
  [{self.H_czi_scaled[2,0]:>8.6f} {self.H_czi_scaled[2,1]:>8.6f} {self.H_czi_scaled[2,2]:>8.3f}]
"""
        
        ax12.text(0.05, 0.95, info_text,
                 transform=ax12.transAxes,
                 fontsize=9,
                 verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 視覺化結果已儲存: {output_path}")
        plt.show()
    
    def create_checkerboard_overlay(self, img1, img2, square_size=100):
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
    
    def create_color_overlay(self, img1, img2, alpha=0.6):
        """創建色彩疊合 (紅-綠)"""
        # 正規化
        img1_norm = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img2_norm = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 創建 RGB
        overlay = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
        overlay[:, :, 0] = (img1_norm * alpha).astype(np.uint8)  # Red
        overlay[:, :, 1] = (img2_norm * alpha).astype(np.uint8)  # Green
        overlay[:, :, 2] = 0  # Blue
        
        return overlay
    
    def create_edge_overlay(self, img1, img2):
        """創建邊緣疊合"""
        edges1 = cv2.Canny(img1, 50, 150)
        edges2 = cv2.Canny(img2, 50, 150)
        
        overlay = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
        overlay[:, :, 0] = edges1  # Red
        overlay[:, :, 1] = edges2  # Green
        overlay[:, :, 2] = np.minimum(edges1, edges2)  # Blue
        
        return overlay
    
    def save_aligned_czi_image(self, aligned_image, output_path="Her2_aligned_to_DISH_CZI_scaled.png"):
        """儲存配準後的 CZI 影像"""
        cv2.imwrite(output_path, aligned_image)
        print(f"✓ 配準後的 CZI 影像已儲存: {output_path}")
    
    def export_validation_results(self, metrics, output_json="czi_validation_results.json"):
        """導出驗證結果"""
        results = {
            'metadata': {
                'timestamp': self.params['metadata']['timestamp'],
                'validation_scale_factor': self.scale_factor,
                'ref_czi_file': str(self.ref_czi_path),
                'mov_czi_file': str(self.mov_czi_path)
            },
            'scaled_images': {
                'reference': {
                    'width': self.ref_czi_scaled_width,
                    'height': self.ref_czi_scaled_height
                },
                'moving': {
                    'width': self.mov_czi_scaled_width,
                    'height': self.mov_czi_scaled_height
                }
            },
            'transformation_scaled': {
                'matrix': self.H_czi_scaled.tolist(),
                'description': f'Homography matrix for {self.scale_factor}x scaled CZI images'
            },
            'quality_metrics': metrics,
            'note': 'This validation was performed on scaled CZI images for memory efficiency'
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 驗證結果已導出: {output_json}")


def main():
    """主程序"""
    print("="*70)
    print("CZI 配準驗證工具")
    print("="*70)
    
    # ========== 設定檔案路徑 ==========
    json_path = "registration_results.json"
    ref_czi_path = Path("../picture/P2525729F_DISH_region.czi")
    mov_czi_path = Path("../picture/P2525729F_HER2_region.czi")
    
    # 檢查檔案是否存在
    if not Path(json_path).exists():
        print(f"✗ 找不到配準參數檔案: {json_path}")
        print("請先執行 image_registration_pipeline.py")
        return
    
    if not ref_czi_path.exists():
        print(f"✗ 找不到參考 CZI 檔案: {ref_czi_path}")
        return
    
    if not mov_czi_path.exists():
        print(f"✗ 找不到移動 CZI 檔案: {mov_czi_path}")
        return
    
    print("✓ 所有必要檔案都存在\n")
    
    # ========== 設定縮放比例 ==========
    print("選擇 CZI 載入縮放比例:")
    print("1. 0.0625x (6.25%) - 最快，約 1GB 記憶體")
    print("2. 0.125x (12.5%) - 快速，約 2.5GB 記憶體")
    print("3. 0.2x (20%) - 平衡，約 6GB 記憶體 [推薦]")
    print("4. 0.25x (25%) - 較慢，約 10GB 記憶體")
    print("5. 0.5x (50%) - 很慢，約 40GB 記憶體")
    
    choice = input("請選擇 (1-5) [預設: 3]: ").strip() or "3"
    
    scale_map = {
        "1": 0.0625,
        "2": 0.125,
        "3": 0.2,
        "4": 0.25,
        "5": 0.5
    }
    
    scale_factor = scale_map.get(choice, 0.2)
    
    try:
        # ========== 初始化驗證器 ==========
        validator = CZIRegistrationValidator(
            json_path,
            ref_czi_path,
            mov_czi_path,
            scale_factor=scale_factor
        )
        
        # ========== 執行驗證 ==========
        ref_img, mov_img, aligned_img = validator.apply_transformation_to_czi()
        
        # ========== 計算品質指標 ==========
        metrics = validator.calculate_quality_metrics(ref_img, aligned_img)
        
        # ========== 儲存結果 ==========
        validator.save_aligned_czi_image(aligned_img)
        validator.export_validation_results(metrics)
        
        # ========== 視覺化 ==========
        validator.visualize_results(ref_img, mov_img, aligned_img, metrics)
        
        print("\n" + "="*70)
        print("✓ CZI 驗證完成！")
        print("="*70)
        print("輸出檔案:")
        print("  - Her2_aligned_to_DISH_CZI_scaled.png")
        print("  - czi_validation_results.png")
        print("  - czi_validation_results.json")
        
        if metrics:
            print(f"\n品質指標:")
            print(f"  SSIM: {metrics['SSIM']:.4f}")
            print(f"  PSNR: {metrics['PSNR_dB']:.2f} dB")
            print(f"  重疊: {metrics['Overlap_Percent']:.1f}%")
        
    except Exception as e:
        print(f"\n✗ 驗證失敗: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理記憶體
        gc.collect()
        print("\n✓ 記憶體已清理")


if __name__ == "__main__":
    main()
