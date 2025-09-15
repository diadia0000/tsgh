#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HER2 染色細胞膜遮罩核心處理邏輯
Core Processing Logic for HER2 Cell Membrane Masking (Coffee-colored membrane staining)
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import json
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
from PIL import Image
from skimage.color import rgb2hed, rgb2lab
from skimage import exposure

@dataclass
class Her2MaskingParams:
    """簡化版 Her2 遮罩參數 - 只保留咖啡色提取需要的參數"""
    
    # LAB 通道閾值範圍 (咖啡色特徵)
    lab_a_lower: int = 125    # A通道下限 (0-255)
    lab_a_upper: int = 135    # A通道上限 (0-255)
    lab_b_lower: int = 120    # B通道下限 (0-255)  
    lab_b_upper: int = 140    # B通道上限 (0-255)
    
    # DAB 通道閾值範圍 (咖啡色強度)
    dab_lower: int = 15       # DAB下限 (0-255)
    dab_upper: int = 255      # DAB上限 (0-255)
    
    # 簡化的形態學參數 (去雜訊)
    membrane_kernel_size: int = 3
    membrane_open_iter: int = 1
    membrane_close_iter: int = 1
    
    # 基本過濾參數
    min_membrane_area: int = 50
    
    # 顯示參數
    alpha: int = 50  # 透明度百分比
    
    def to_dict(self) -> Dict[str, Any]:
        """轉為字典格式"""
        return {
            'lab_a_lower': self.lab_a_lower,
            'lab_a_upper': self.lab_a_upper,
            'lab_b_lower': self.lab_b_lower,
            'lab_b_upper': self.lab_b_upper,
            'dab_lower': self.dab_lower,
            'dab_upper': self.dab_upper,
            'membrane_kernel_size': self.membrane_kernel_size,
            'membrane_open_iter': self.membrane_open_iter,
            'membrane_close_iter': self.membrane_close_iter,
            'min_membrane_area': self.min_membrane_area,
            'alpha': self.alpha
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Her2MaskingParams':
        """從字典建立參數"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
@dataclass 
class Her2ProcessingResult:
    """簡化版 HER2 處理結果 - 只保留咖啡色提取相關資料"""
    
    # 基本通道 (簡化版)
    lab_channels: np.ndarray  # 3通道 LAB 影像  
    dab_channel: np.ndarray   # DAB 通道 (咖啡色)
    
    # 咖啡色遮罩
    mask_membrane_raw: np.ndarray
    mask_membrane_clean: np.ndarray
    
    # 疊加圖 (簡化版)
    overlay_membrane: np.ndarray
    
    # 提取結果 (簡化版)
    extract_membrane: np.ndarray
    
    # 參數與統計
    params: Her2MaskingParams
    processing_time: float
    
    # 簡化統計資訊
    total_pixels: int
    membrane_pixels: int
    membrane_coverage_percent: float
    
    # DAB 強度統計
    dab_mean: float
    dab_median: float

class Her2MaskProcessor:
    """HER2 染色細胞膜遮罩處理核心"""
    
    def __init__(self):
        self.original_image = None
        self.working_image = None
        self.working_scale = 0.15  # 15% 縮小顯示 (更快速處理, 從25%降到15%)
        self.image_path = None
        
    def load_image(self, image_path: str) -> bool:
        """載入影像"""
        try:
            path = Path(image_path)
            if not path.exists():
                print(f"錯誤: 檔案不存在 {image_path}")
                return False
            
            # 使用 OpenCV 讀取
            self.original_image = cv.imread(str(path))
            if self.original_image is None:
                print(f"錯誤: 無法讀取影像 {image_path}")
                return False
            
            # 建立工作影像 (縮小至 25%)
            h, w = self.original_image.shape[:2]
            new_h, new_w = int(h * self.working_scale), int(w * self.working_scale)
            self.working_image = cv.resize(self.original_image, (new_w, new_h), interpolation=cv.INTER_AREA)
            
            self.image_path = image_path
            print(f"已載入影像: {path.name}, 原始={w}×{h}, 工作={new_w}×{new_h}")
            return True
            
        except Exception as e:
            print(f"載入影像失敗: {e}")
            return False
    
    def load_her2_from_directory(self, base_dir: str = "picture/tiff") -> bool:
        """自動載入 HER2 影像"""
        try:
            base_path = Path(base_dir)
            if not base_path.exists():
                print(f"目錄不存在: {base_dir}")
                return False
            
            # 尋找 HER2 相關檔案
            her2_patterns = ["*_Her2_region*", "*_HER2_region*"]
            her2_file = None
            
            for pattern in her2_patterns:
                files = list(base_path.glob(pattern + ".tiff")) + list(base_path.glob(pattern + ".tif"))
                if files:
                    her2_file = files[0]
                    break
            
            if her2_file is None:
                # 備用搜尋
                all_files = list(base_path.glob("*.tiff")) + list(base_path.glob("*.tif"))
                for file in all_files:
                    filename = file.name.upper()
                    if "HER2" in filename and "REGION" in filename:
                        her2_file = file
                        break
            
            if her2_file is None:
                print("未找到 HER2 影像檔案")
                print(f"在目錄 {base_dir} 中搜尋的模式: {her2_patterns}")
                return False
            
            print(f"找到 HER2 檔案: {her2_file.name}")
            return self.load_image(str(her2_file))
            
        except Exception as e:
            print(f"自動載入 HER2 影像失敗: {e}")
            return False
    
    def compute_dab_8u(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        快速計算咖啡色強度 (簡化版本)
        
        Args:
            img_bgr: BGR 格式的輸入圖像
            
        Returns:
            8位元的咖啡色強度圖 (0-255)
        """
        # 轉換為 RGB
        rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        # 咖啡色特徵：R > G > B，且在合適的亮度範圍
        brown_mask = (r > g) & (g > b) & (r > 100) & (r < 200) & (g > 50) & (g < 150)
        brown_score = np.where(brown_mask, (r - b) + (r - g), 0)
        
        # 正規化到 0-255
        if brown_score.max() > 0:
            dab_8u = (brown_score / brown_score.max() * 255).astype(np.uint8)
        else:
            dab_8u = np.zeros_like(brown_score, dtype=np.uint8)
        
        return dab_8u
    
    def enhance_L_only(self, img_bgr: np.ndarray, clipLimit: float = 1.5, tileGridSize: tuple = (4, 4)) -> np.ndarray:
        """
        快速L通道強化 (優化版本)
        
        Args:
            img_bgr: BGR 格式的輸入圖像
            clipLimit: CLAHE 的限制值 (降低以加快速度)
            tileGridSize: CLAHE 的瓦片網格大小 (縮小以加快速度)
            
        Returns:
            L 通道強化後的 BGR 圖像
        """
        lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
        L, A, B = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        L2 = clahe.apply(L)
        lab2 = cv.merge([L2, A, B])
        return cv.cvtColor(lab2, cv.COLOR_LAB2BGR)
    
    def apply_lab_mask_with_dab_gate(self, img_bgr: np.ndarray, lab_lower: np.ndarray, lab_upper: np.ndarray,
                                   dab_threshold: int = 15, use_L_enhance: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        快速LAB遮罩與DAB門檻交集 (優化版本)
        
        Args:
            img_bgr: 輸入的 BGR 圖像
            lab_lower: LAB 下限值
            lab_upper: LAB 上限值  
            dab_threshold: DAB 門檻值
            use_L_enhance: 是否使用 L 通道強化
            
        Returns:
            (遮罩, 濾波後圖像, DAB強度圖)
        """
        # 簡化L通道增強 - 只在必要時使用
        if use_L_enhance:
            enhanced_bgr = self.enhance_L_only(img_bgr, clipLimit=1.5, tileGridSize=(4, 4))  # 降低CLAHE複雜度
            lab_src = cv.cvtColor(enhanced_bgr, cv.COLOR_BGR2LAB)
        else:
            lab_src = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
        
        # 快速計算DAB強度
        dab_img = self.compute_dab_8u(img_bgr)
        
        # 1) LAB 範圍篩選 - 使用OpenCV的快速方法
        lab_mask = cv.inRange(lab_src, lab_lower, lab_upper)
        
        # 2) DAB 門檻 - 使用numpy的快速比較
        dab_gate = (dab_img > dab_threshold).astype(np.uint8) * 255
        
        # 3) 快速遮罩交集
        final_mask = cv.bitwise_and(lab_mask, dab_gate)
        
        # 4) 快速產生濾波圖像 (只在需要時)
        filtered_img = cv.bitwise_and(img_bgr, img_bgr, mask=final_mask)
        
        return final_mask, filtered_img, dab_img

    def get_image_info(self) -> Dict[str, Any]:
        """取得影像資訊"""
        if self.original_image is None:
            return {}
        
        h, w = self.original_image.shape[:2]
        work_h, work_w = self.working_image.shape[:2] if self.working_image is not None else (0, 0)
        
        return {
            'filename': Path(self.image_path).name if self.image_path else 'Unknown',
            'original_size': (w, h),
            'working_size': (work_w, work_h),
            'working_scale': self.working_scale
        }
    
    def process_mask(self, params: Her2MaskingParams, use_original: bool = False) -> Her2ProcessingResult:
        """簡化處理 HER2 咖啡色區域"""
        start_time = time.time()
        
        try:
            print(f"開始簡化處理咖啡色區域，use_original={use_original}")
            
            # 選擇處理影像
            if use_original and self.original_image is not None:
                img = self.original_image
                scale_factor = 1.0
            else:
                img = self.working_image
                scale_factor = self.working_scale
            
            if img is None:
                raise ValueError("沒有可用的影像")
            
            # 核心處理：LAB + DAB 咖啡色提取
            print("進行咖啡色區域提取...")
            
            # LAB範圍參數
            lab_lower = np.array([0, params.lab_a_lower, params.lab_b_lower], dtype=np.uint8)
            lab_upper = np.array([255, params.lab_a_upper, params.lab_b_upper], dtype=np.uint8)
            
            # DAB閾值
            dab_threshold = int((params.dab_lower + params.dab_upper) / 2)
            
            # 簡化處理：只做咖啡色提取
            mask_membrane_raw, filtered_img, dab_img = self.apply_lab_mask_with_dab_gate(
                img, lab_lower, lab_upper, 
                dab_threshold=dab_threshold,
                use_L_enhance=False
            )
            
            # 簡單形態學處理
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            mask_membrane_clean = cv.morphologyEx(mask_membrane_raw, cv.MORPH_CLOSE, kernel, iterations=1)
            mask_membrane_clean = cv.morphologyEx(mask_membrane_clean, cv.MORPH_OPEN, kernel, iterations=1)
            
            # 創建簡化的結果 - 不需要複雜的核處理和邊緣檢測
            rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            lab_img = rgb2lab(rgb_img)
            
            # 創建疊加圖：只顯示咖啡色區域
            overlay_coffee = img.copy()
            overlay_coffee[mask_membrane_clean > 0] = [0, 0, 255]  # 紅色標記咖啡色區域
            
            # 提取咖啡色區域
            extract_coffee = cv.bitwise_and(img, img, mask=mask_membrane_clean)
            
            # 統計資訊
            total_pixels = img.shape[0] * img.shape[1]
            coffee_pixels = np.sum(mask_membrane_clean > 0)
            
            processing_time = time.time() - start_time
            print(f"簡化處理完成，耗時: {processing_time:.2f}秒")
            print(f"咖啡色區域覆蓋率: {(coffee_pixels/total_pixels)*100:.1f}%")
            
            # 創建簡化的結果物件
            print(f"簡化處理完成，耗時: {processing_time:.2f}秒")
            print(f"咖啡色區域覆蓋率: {(coffee_pixels/total_pixels)*100:.1f}%")
            
            # 創建簡化的結果物件
            return Her2ProcessingResult(
                lab_channels=(lab_img * 255).astype(np.uint8),
                dab_channel=dab_img,
                mask_membrane_raw=mask_membrane_raw,
                mask_membrane_clean=mask_membrane_clean,
                overlay_membrane=overlay_coffee,
                extract_membrane=extract_coffee,
                params=params,
                processing_time=processing_time,
                total_pixels=total_pixels,
                membrane_pixels=coffee_pixels,
                membrane_coverage_percent=100.0 * coffee_pixels / total_pixels,
                dab_mean=float(np.mean(dab_img[mask_membrane_clean > 0])) if coffee_pixels > 0 else 0.0,
                dab_median=float(np.median(dab_img[mask_membrane_clean > 0])) if coffee_pixels > 0 else 0.0
            )
            
        except Exception as e:
            print(f"處理遮罩時發生錯誤: {e}")
            raise
    
    def _morphology_cleanup(self, mask: np.ndarray, kernel_size: int, 
                          open_iter: int, close_iter: int) -> np.ndarray:
        """快速形態學去雜訊 (優化版本)"""
        if kernel_size <= 0:
            return mask
        
        # 縮小核心大小以提升速度
        fast_kernel_size = max(1, min(kernel_size, 3))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (fast_kernel_size, fast_kernel_size))
        
        # 減少迭代次數以提升速度
        fast_open_iter = max(0, min(open_iter, 1))
        fast_close_iter = max(0, min(close_iter, 1))
        
        # 開運算去雜訊
        if fast_open_iter > 0:
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=fast_open_iter)
        
        # 閉運算填補空洞
        if fast_close_iter > 0:
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=fast_close_iter)
        
        return mask
    
    def _area_filter(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """面積過濾移除小物件（優化版）"""
        if min_area <= 0:
            return mask
        
        print(f"進行面積過濾，最小面積: {min_area}")
        
        # 如果遮罩像素太多，先進行形態學預處理減少雜訊
        mask_pixels = np.sum(mask > 0)
        print(f"遮罩像素數: {mask_pixels}")
        
        if mask_pixels > 200000:  # 降低閾值到20萬像素，更快啟用快速模式
            print("使用快速面積過濾模式")
            # 使用更大的開運算來移除小碎片
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))  # 縮小核心大小
            filtered_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)  # 減少迭代次數
            print(f"快速過濾後像素數: {np.sum(filtered_mask > 0)}")
            return filtered_mask
        
        # 正常面積過濾
        print("使用標準連通元件分析")
        try:
            # 連通元件分析
            num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
            print(f"找到 {num_labels-1} 個連通元件")
            
            # 如果連通元件太多，改用快速模式
            if num_labels > 1000:  # 降低閾值，更快啟用快速模式
                print(f"連通元件過多 ({num_labels-1} 個)，改用快速模式")
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # 更小的核心
                filtered_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)  # 減少迭代
                print(f"快速過濾後像素數: {np.sum(filtered_mask > 0)}")
                return filtered_mask
            
            # 建立過濾後的遮罩
            filtered_mask = np.zeros_like(mask)
            
            kept_count = 0
            for i in range(1, num_labels):  # 跳過背景 (label 0)
                area = stats[i, cv.CC_STAT_AREA]
                if area >= min_area:
                    filtered_mask[labels == i] = 255
                    kept_count += 1
            
            print(f"保留 {kept_count} 個符合面積要求的元件")
            return filtered_mask
            
        except Exception as e:
            print(f"連通元件分析失敗: {e}，使用備用方法")
            # 備用方法：使用形態學操作
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            return cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    
    def _create_overlay(self, img: np.ndarray, mask: np.ndarray, alpha: int, 
                       color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """建立疊加圖"""
        overlay = img.copy()
        alpha_value = alpha / 100.0
        
        # 建立彩色遮罩
        colored_mask = np.zeros_like(img)
        colored_mask[mask > 0] = color
        
        # 混合
        result = cv.addWeighted(overlay, 1 - alpha_value, colored_mask, alpha_value, 0)
        return result
    
    def _create_extraction(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """建立提取結果 (RGBA)"""
        # 建立 4 通道圖像 (BGRA)
        result = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        result[:, :, :3] = img  # 複製 BGR
        result[:, :, 3] = mask  # Alpha 通道
        
        # 只保留遮罩區域的顏色
        mask_3d = np.expand_dims(mask, axis=2)
        result[:, :, :3] = img * (mask_3d / 255.0)
        
        return result
    
    def save_results(self, result: Her2ProcessingResult, output_dir: str) -> bool:
        """儲存處理結果"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 取得檔案基本名稱
            if self.image_path:
                basename = Path(self.image_path).stem
            else:
                basename = "her2_result"
            
            # 建立子資料夾
            membrane_dir = output_path / "membrane"
            nuclei_dir = output_path / "nuclei"
            combined_dir = output_path / "combined"
            channels_dir = output_path / "channels"
            
            for dir_path in [membrane_dir, nuclei_dir, combined_dir, channels_dir]:
                dir_path.mkdir(exist_ok=True)
            
            # 使用全尺寸處理 (如果需要)
            if self.original_image is not None:
                print("進行全尺寸處理以輸出高品質結果...")
                full_result = self.process_mask(result.params, use_original=True)
            else:
                full_result = result
            
            # 儲存通道圖像 (簡化版)
            self._save_png(channels_dir / f"{basename}_dab_channel.png", full_result.dab_channel)
            # 移除 hema_channel - 簡化版本不需要
            
            # 儲存遮罩 (簡化版)
            self._save_png(membrane_dir / f"{basename}_mask_membrane.png", full_result.mask_membrane_clean)
            # 移除其他複雜遮罩 - 簡化版本不需要
            
            # 儲存疊加圖 (簡化版)
            self._save_png(membrane_dir / f"{basename}_overlay_membrane.png", full_result.overlay_membrane)
            # 移除其他複雜疊加圖 - 簡化版本不需要
            
            # 儲存提取結果 (簡化版)
            self._save_png_rgba(membrane_dir / f"{basename}_extract_membrane.png", full_result.extract_membrane)
            # 移除 extract_nuclei - 簡化版本不需要
            
            # 移除邊緣檢測結果 - 簡化版本不需要
            
            # 儲存參數
            params_file = output_path / f"{basename}_params.json"
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(result.params.to_dict(), f, indent=2, ensure_ascii=False)
            
            # 儲存統計報告 (CSV)
            self._save_statistics_csv(output_path / f"{basename}_report.csv", full_result)
            
            print(f"結果已儲存至: {output_path}")
            return True
            
        except Exception as e:
            print(f"儲存結果失敗: {e}")
            return False
    
    def _save_statistics_csv(self, filepath: Path, result: Her2ProcessingResult):
        """儲存統計報告 CSV"""
        try:
            import csv
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # 標題行
                writer.writerow(['項目', '數值', '單位'])
                
                # 基本資訊
                writer.writerow(['檔案名稱', Path(self.image_path).name if self.image_path else 'Unknown', ''])
                writer.writerow(['處理時間', f"{result.processing_time:.2f}", '秒'])
                writer.writerow(['總像素數', result.total_pixels, '像素'])
                
                # 覆蓋率統計 (簡化版)
                writer.writerow(['咖啡色像素數', result.membrane_pixels, '像素'])
                writer.writerow(['咖啡色覆蓋率', f"{result.membrane_coverage_percent:.2f}", '%'])
                # 移除細胞核和ROI統計 - 簡化版本不需要
                
                # DAB 強度統計 (簡化版)
                writer.writerow(['DAB平均值', f"{result.dab_mean:.4f}", ''])
                writer.writerow(['DAB中位數', f"{result.dab_median:.4f}", ''])
                # 移除百分位統計 - 簡化版本不需要
            
            print(f"統計報告已儲存: {filepath}")
            
        except Exception as e:
            print(f"儲存統計報告失敗: {e}")
    
    def _save_png(self, filepath: Path, img: np.ndarray):
        """使用 Pillow 儲存 PNG"""
        try:
            if len(img.shape) == 3:
                # BGR -> RGB
                img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
            else:
                # 灰階
                pil_img = Image.fromarray(img)
            
            pil_img.save(str(filepath))
            
        except Exception as e:
            print(f"儲存 PNG 失敗 {filepath}: {e}")
    
    def _save_png_rgba(self, filepath: Path, img: np.ndarray):
        """使用 Pillow 儲存 RGBA PNG"""
        try:
            # BGRA -> RGBA
            img_rgba = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
            pil_img = Image.fromarray(img_rgba, 'RGBA')
            pil_img.save(str(filepath))
            
        except Exception as e:
            print(f"儲存 RGBA PNG 失敗 {filepath}: {e}")