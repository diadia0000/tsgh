#!/usr/bin/env python3
"""
醫療影像疊合 Pipeline (最小驗證版本)
功能：
1. 對兩張灰階 PNG 影像進行配準 (Registration)
2. 計算轉換參數 (Transformation Matrix)
3. 將 PNG 參數轉換為 CZI 原圖參數
4. 輸出 JSON 格式的配準結果

設計考量：
- 使用 OpenCV 進行特徵點檢測與匹配
- 支援多種配準方法 (SIFT, ORB, ECC)
- 便於轉換為 C++ (使用 OpenCV API)
- 記憶體效率優化
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import gc
from PIL import Image


class ImageRegistration:
    """影像配準類別"""
    
    def __init__(self, reference_image_path, moving_image_path, 
                 ref_czi_size, mov_czi_size, ref_png_size, mov_png_size,
                 target_size=None):
        """
        初始化配準器
        
        Args:
            reference_image_path: 參考影像路徑 (固定不動的影像, DISH)
            moving_image_path: 移動影像路徑 (需要對齊的影像, HER2)
            ref_czi_size: 參考影像 CZI 原圖尺寸 (width, height)
            mov_czi_size: 移動影像 CZI 原圖尺寸 (width, height)
            ref_png_size: 參考影像 PNG 尺寸 (width, height)
            mov_png_size: 移動影像 PNG 尺寸 (width, height)
            target_size: 統一縮放目標尺寸 (width, height)，若為None則使用參考影像尺寸
        """
        self.ref_path = Path(reference_image_path)
        self.mov_path = Path(moving_image_path)
        
        # CZI 原圖尺寸
        self.ref_czi_width, self.ref_czi_height = ref_czi_size
        self.mov_czi_width, self.mov_czi_height = mov_czi_size
        
        # PNG 原始尺寸
        self.ref_png_width, self.ref_png_height = ref_png_size
        self.mov_png_width, self.mov_png_height = mov_png_size
        
        # 統一縮放目標尺寸
        if target_size is None:
            # 預設使用參考影像的尺寸
            self.target_width = ref_png_size[0]
            self.target_height = ref_png_size[1]
        else:
            self.target_width, self.target_height = target_size
        
        # 計算縮放比例（從CZI到目標尺寸）
        self.ref_scale = self.ref_czi_width / self.target_width
        self.mov_scale = self.mov_czi_width / self.target_width
        
        print(f"目標影像尺寸: {self.target_width} x {self.target_height}")
        print(f"參考影像縮放比例 (CZI→目標): {self.ref_scale:.4f}x")
        print(f"移動影像縮放比例 (CZI→目標): {self.mov_scale:.4f}x")
        
        # 載入影像
        self.ref_image = None
        self.mov_image = None
        self.transformation_matrix = None
        self.registration_method = None
        self.keypoints_info = {}
        
    def load_images(self):
        """使用PIL載入影像並統一縮放到目標尺寸，然後轉換為灰階"""
        print("\n正在載入影像...")
        
        # 使用PIL載入影像
        ref_pil = Image.open(str(self.ref_path))
        mov_pil = Image.open(str(self.mov_path))
        
        print(f"  原始參考影像尺寸: {ref_pil.size}")
        print(f"  原始移動影像尺寸: {mov_pil.size}")
        
        # 統一縮放到目標尺寸（使用高品質LANCZOS重採樣）
        target_size = (self.target_width, self.target_height)
        ref_pil_resized = ref_pil.resize(target_size, Image.Resampling.LANCZOS)
        mov_pil_resized = mov_pil.resize(target_size, Image.Resampling.LANCZOS)
        
        # 轉換為灰階並轉為numpy array
        ref_gray = ref_pil_resized.convert('L')
        mov_gray = mov_pil_resized.convert('L')
        
        self.ref_image = np.array(ref_gray, dtype=np.uint8)
        self.mov_image = np.array(mov_gray, dtype=np.uint8)
        
        # 釋放PIL物件
        ref_pil.close()
        mov_pil.close()
        ref_pil_resized.close()
        mov_pil_resized.close()
        
        print(f"✓ 參考影像 (縮放後): {self.ref_image.shape}")
        print(f"✓ 移動影像 (縮放後): {self.mov_image.shape}")

        
    def register_feature_based(self, method='SIFT', max_features=10000):
        """
        基於特徵點的配準方法
        
        Args:
            method: 'SIFT' 或 'ORB'
            max_features: 最大特徵點數量
            
        Returns:
            transformation_matrix: 3x3 變換矩陣
        """
        print(f"\n正在使用 {method} 進行特徵點配準...")
        
        # 1. 特徵檢測器
        if method == 'SIFT':
            detector = cv2.SIFT_create(nfeatures=max_features)
        elif method == 'ORB':
            detector = cv2.ORB_create(nfeatures=max_features)
        else:
            raise ValueError(f"不支援的方法: {method}")
        
        # 2. 檢測特徵點和描述子
        print("  - 正在檢測特徵點...")
        kp1, des1 = detector.detectAndCompute(self.ref_image, None)
        kp2, des2 = detector.detectAndCompute(self.mov_image, None)
        
        print(f"  - 參考影像特徵點: {len(kp1)}")
        print(f"  - 移動影像特徵點: {len(kp2)}")
        
        self.keypoints_info = {
            'reference_keypoints': len(kp1),
            'moving_keypoints': len(kp2)
        }
        
        # 3. 特徵匹配
        print("  - 正在匹配特徵點...")
        if method == 'SIFT':
            # FLANN matcher for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            matches = matcher.knnMatch(des1, des2, k=2)
            
            # Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
        else:  # ORB
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)
            good_matches = sorted(matches, key=lambda x: x.distance)[:max_features//2]
        
        print(f"  - 良好匹配點數: {len(good_matches)}")
        self.keypoints_info['good_matches'] = len(good_matches)
        
        if len(good_matches) < 4:
            raise ValueError("匹配點數量不足，無法計算變換矩陣")
        
        # 4. 提取匹配點座標
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 5. 計算變換矩陣 (使用 RANSAC 濾除外點)
        print("  - 正在計算變換矩陣...")
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            raise ValueError("無法計算變換矩陣")
        
        inliers = mask.ravel().sum()
        print(f"  - RANSAC 內點數: {inliers}/{len(good_matches)}")
        
        self.keypoints_info['ransac_inliers'] = int(inliers)
        self.keypoints_info['ransac_outliers'] = len(good_matches) - int(inliers)
        
        self.transformation_matrix = M
        self.registration_method = method
        
        return M
    
    def register_intensity_based(self, iterations=5000):
        """
        基於灰度的配準方法 (ECC - Enhanced Correlation Coefficient)
        適用於已經大致對齊的影像
        
        Args:
            iterations: 最大迭代次數
            
        Returns:
            transformation_matrix: 3x3 變換矩陣
        """
        print(f"\n正在使用 ECC 進行灰度配準...")
        
        # 定義運動模型 (使用 Homography)
        warp_mode = cv2.MOTION_HOMOGRAPHY
        
        # 初始化變換矩陣為單位矩陣
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        
        # 定義終止條件
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-6)
        
        try:
            print("  - 正在執行 ECC 算法...")
            (cc, warp_matrix) = cv2.findTransformECC(
                self.ref_image, 
                self.mov_image, 
                warp_matrix, 
                warp_mode, 
                criteria,
                inputMask=None,
                gaussFiltSize=5
            )
            
            print(f"  - 相關係數: {cc:.6f}")
            
            self.transformation_matrix = warp_matrix
            self.registration_method = 'ECC'
            self.keypoints_info['correlation_coefficient'] = float(cc)
            
            return warp_matrix
            
        except cv2.error as e:
            raise ValueError(f"ECC 配準失敗: {str(e)}")
    
    def apply_transformation(self, save_result=True, output_path=None):
        """
        應用變換矩陣到移動影像
        
        Args:
            save_result: 是否儲存配準後的影像
            output_path: 輸出路徑
            
        Returns:
            aligned_image: 配準後的影像
        """
        print("\n正在應用變換...")
        
        if self.transformation_matrix is None:
            raise ValueError("尚未計算變換矩陣，請先執行配準")
        
        # 應用變換
        h, w = self.ref_image.shape
        aligned_image = cv2.warpPerspective(
            self.mov_image, 
            self.transformation_matrix, 
            (w, h),
            flags=cv2.INTER_LINEAR
        )
        
        print(f"✓ 變換完成，輸出尺寸: {aligned_image.shape}")
        
        if save_result:
            if output_path is None:
                output_path = Path("aligned_result.png")
            cv2.imwrite(str(output_path), aligned_image)
            print(f"✓ 配準結果已儲存: {output_path}")
        
        return aligned_image
    
    def scale_transformation_to_czi(self):
        """
        將 PNG 的變換矩陣縮放到 CZI 原圖尺度
        
        Returns:
            czi_transformation_matrix: 適用於 CZI 的 3x3 變換矩陣
        """
        if self.transformation_matrix is None:
            raise ValueError("尚未計算變換矩陣")
        
        print("\n正在將變換矩陣轉換到 CZI 尺度...")
        
        # 建立縮放矩陣
        # 1. 先縮放移動影像座標到 CZI 尺度
        # 2. 應用變換
        # 3. 結果已在參考影像的 CZI 尺度
        
        # S_mov: 移動影像的縮放矩陣 (PNG -> CZI)
        S_mov = np.array([
            [self.mov_scale, 0, 0],
            [0, self.mov_scale, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # S_ref_inv: 參考影像的逆縮放矩陣 (CZI -> PNG)
        S_ref_inv = np.array([
            [1/self.ref_scale, 0, 0],
            [0, 1/self.ref_scale, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 組合變換: S_ref * H * S_mov_inv
        # 其中 H 是 PNG 尺度的變換矩陣
        H_png = self.transformation_matrix
        
        # 完整變換: 從移動影像 CZI 座標 -> 參考影像 CZI 座標
        # 步驟: CZI_mov -> PNG_mov -> PNG_ref -> CZI_ref
        # H_czi = S_ref * H_png * S_mov_inv
        
        S_mov_inv = np.linalg.inv(S_mov)
        S_ref = np.linalg.inv(S_ref_inv)  # 即原始的縮放矩陣
        
        H_czi = S_ref @ H_png @ S_mov_inv
        
        print(f"✓ CZI 變換矩陣已計算")
        print(f"  參考影像 CZI 尺寸: {self.ref_czi_width} x {self.ref_czi_height}")
        print(f"  移動影像 CZI 尺寸: {self.mov_czi_width} x {self.mov_czi_height}")
        
        return H_czi
    
    def evaluate_registration(self, aligned_image):
        """
        評估配準品質
        
        Args:
            aligned_image: 配準後的影像
            
        Returns:
            metrics: 評估指標字典
        """
        print("\n正在評估配準品質...")
        
        # 1. 均方誤差 (MSE)
        mse = np.mean((self.ref_image.astype(float) - aligned_image.astype(float)) ** 2)
        
        # 2. 峰值信噪比 (PSNR)
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        # 3. 結構相似性 (SSIM)
        # 簡化版 SSIM 計算
        from skimage.metrics import structural_similarity
        ssim = structural_similarity(self.ref_image, aligned_image, data_range=255)
        
        # 4. 正規化互相關 (NCC)
        ref_norm = (self.ref_image - np.mean(self.ref_image)) / (np.std(self.ref_image) + 1e-8)
        aligned_norm = (aligned_image - np.mean(aligned_image)) / (np.std(aligned_image) + 1e-8)
        ncc = np.mean(ref_norm * aligned_norm)
        
        metrics = {
            'MSE': float(mse),
            'PSNR_dB': float(psnr),
            'SSIM': float(ssim),
            'NCC': float(ncc)
        }
        
        print(f"  MSE: {mse:.2f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")
        print(f"  NCC: {ncc:.4f}")
        
        return metrics
    
    def export_results(self, output_json="registration_results.json"):
        """
        導出所有配準結果為 JSON
        
        Args:
            output_json: JSON 輸出檔案路徑
        """
        print(f"\n正在導出結果到 {output_json}...")
        
        if self.transformation_matrix is None:
            raise ValueError("尚未完成配準")
        
        # 計算 CZI 尺度的變換矩陣
        czi_matrix = self.scale_transformation_to_czi()
        
        # 應用變換並評估
        aligned_image = self.apply_transformation(save_result=False)
        metrics = self.evaluate_registration(aligned_image)
        
        # 組織結果
        results = {
            'metadata': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'registration_method': self.registration_method,
                'reference_image': str(self.ref_path),
                'moving_image': str(self.mov_path)
            },
            'image_info': {
                'reference': {
                    'png_size': {
                        'width': int(self.ref_png_width),
                        'height': int(self.ref_png_height)
                    },
                    'czi_size': {
                        'width': int(self.ref_czi_width),
                        'height': int(self.ref_czi_height)
                    },
                    'scale_factor': float(self.ref_scale)
                },
                'moving': {
                    'png_size': {
                        'width': int(self.mov_png_width),
                        'height': int(self.mov_png_height)
                    },
                    'czi_size': {
                        'width': int(self.mov_czi_width),
                        'height': int(self.mov_czi_height)
                    },
                    'scale_factor': float(self.mov_scale)
                }
            },
            'transformation': {
                'png_space': {
                    'matrix': self.transformation_matrix.tolist(),
                    'description': 'Homography matrix for PNG images (3x3)'
                },
                'czi_space': {
                    'matrix': czi_matrix.tolist(),
                    'description': 'Homography matrix for CZI images (3x3)'
                }
            },
            'feature_matching': self.keypoints_info,
            'quality_metrics': metrics,
            'usage_instructions': {
                'opencv_cpp': 'cv::Mat H = (cv::Mat_<double>(3,3) << matrix[0][0], matrix[0][1], ...; cv::warpPerspective(src, dst, H, size);',
                'opencv_python': 'H = np.array(matrix); aligned = cv2.warpPerspective(moving_img, H, (width, height))',
                'note': 'Use czi_space matrix for full-resolution CZI images'
            }
        }
        
        # 儲存 JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 結果已導出: {output_json}")
        
        return results


def main():
    """主程序"""
    print("="*70)
    print("醫療影像配準 Pipeline - 最小驗證版本")
    print("="*70)
    
    # ========== 設定檔案路徑 ==========
    reference_image = "DISH_mask.png"
    moving_image = "Her2_mask.png"
    
    # ========== CZI 原圖尺寸 (從 analysis.txt 獲得) ==========
    # DISH: 馬賽克邊界框 161259 x 122176
    # HER2: 馬賽克邊界框 159388 x 122224
    ref_czi_size = (161259, 122176)  # (width, height)
    mov_czi_size = (159388, 122224)
    
    # ========== PNG 尺寸 (使用者提供) ==========
    ref_png_size = (40314, 30544)  # DISH_mask
    mov_png_size = (31877, 24444)  # Her2_mask
    
    # ========== 統一縮放目標尺寸 ==========
    # 可以設定為較小的尺寸以加速處理，例如 (8000, 6000)
    # 設定為None則使用參考影像的原始尺寸
    target_size = (8000, 6000)  # 統一縮放大小
    
    # ========== 初始化配準器 ==========
    registrator = ImageRegistration(
        reference_image,
        moving_image,
        ref_czi_size,
        mov_czi_size,
        ref_png_size,
        mov_png_size,
        target_size=target_size
    )
    
    # ========== 載入影像 ==========
    registrator.load_images()
    
    # ========== 執行配準 ==========
    print("\n選擇配準方法:")
    print("1. SIFT (特徵點配準 - 推薦)")
    print("2. ORB (特徵點配準 - 快速)")
    print("3. ECC (灰度配準 - 需要初始對齊)")
    
    method_choice = input("請選擇方法 (1-3) [預設: 1]: ").strip() or "1"
    
    try:
        if method_choice == "1":
            registrator.register_feature_based(method='SIFT', max_features=10000)
        elif method_choice == "2":
            registrator.register_feature_based(method='ORB', max_features=10000)
        elif method_choice == "3":
            registrator.register_intensity_based(iterations=5000)
        else:
            print("無效選擇，使用預設方法 SIFT")
            registrator.register_feature_based(method='SIFT', max_features=10000)
        
        # ========== 應用變換並儲存結果 ==========
        aligned = registrator.apply_transformation(
            save_result=True,
            output_path="Her2_aligned_to_DISH.png"
        )
        
        # ========== 導出 JSON 結果 ==========
        results = registrator.export_results("registration_results.json")
        
        print("\n" + "="*70)
        print("✓ 配準完成！")
        print("="*70)
        print(f"輸出檔案:")
        print(f"  - 配準影像: Her2_aligned_to_DISH.png")
        print(f"  - 參數檔案: registration_results.json")
        print(f"\n品質指標:")
        print(f"  - SSIM: {results['quality_metrics']['SSIM']:.4f}")
        print(f"  - PSNR: {results['quality_metrics']['PSNR_dB']:.2f} dB")
        
    except Exception as e:
        print(f"\n✗ 配準失敗: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理記憶體
        gc.collect()


if __name__ == "__main__":
    main()
