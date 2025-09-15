#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HE 染色細胞遮罩核心處理邏輯
Core Processing Logic for HE Cell Masking (Cytoplasm/Nuclei Separation)
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import json
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
from PIL import Image

@dataclass
class HEMaskingParams:
    """HE 遮罩處理參數 - 專注於細胞膜分離"""
    # 細胞膜 (Eosin) HSV 閾值 - 雙區段 H 範圍
    membrane_h1_min: int = 0
    membrane_h1_max: int = 20
    membrane_h2_min: int = 160
    membrane_h2_max: int = 179
    membrane_s_min: int = 30
    membrane_s_max: int = 150
    membrane_v_min: int = 80
    membrane_v_max: int = 255
    
    # 形態學參數 - 細胞膜（設為1以跳過處理加速）
    membrane_kernel_size: int = 3
    membrane_open_iter: int = 1
    membrane_close_iter: int = 1
    
    # 平滑參數（關閉以加速處理）
    gaussian_kernel: int = 1
    median_kernel: int = 1
    
    # 面積過濾
    min_membrane_area: int = 50
    
    # 顯示參數
    alpha: int = 50  # 透明度百分比
    
    def to_dict(self) -> Dict[str, Any]:
        """轉為字典格式"""
        return {
            'membrane_h1_min': self.membrane_h1_min, 'membrane_h1_max': self.membrane_h1_max,
            'membrane_h2_min': self.membrane_h2_min, 'membrane_h2_max': self.membrane_h2_max,
            'membrane_s_min': self.membrane_s_min, 'membrane_s_max': self.membrane_s_max,
            'membrane_v_min': self.membrane_v_min, 'membrane_v_max': self.membrane_v_max,
            'membrane_kernel_size': self.membrane_kernel_size,
            'membrane_open_iter': self.membrane_open_iter,
            'membrane_close_iter': self.membrane_close_iter,
            'gaussian_kernel': self.gaussian_kernel,
            'median_kernel': self.median_kernel,
            'min_membrane_area': self.min_membrane_area,
            'alpha': self.alpha
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HEMaskingParams':
        """從字典建立參數"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
class HEProcessingResult:
    """HE 處理結果 - 專注於細胞膜"""
    # 原始遮罩
    mask_membrane_raw: np.ndarray
    
    # 清理後遮罩
    mask_membrane_clean: np.ndarray
    
    # 疊加與提取圖
    overlay_membrane: np.ndarray
    extract_membrane: np.ndarray
    
    # 處理參數與統計
    params: HEMaskingParams
    processing_time: float
    total_pixels: int
    membrane_pixels: int
    membrane_area_percent: float

class HEMaskProcessor:
    """HE 染色細胞遮罩處理核心"""
    
    def __init__(self):
        self.original_image = None
        self.working_image = None
        self.working_scale = 0.125  # 12.5% 縮小顯示 (更快速處理)
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
            
            # 建立工作影像 (縮小至 12.5%)
            h, w = self.original_image.shape[:2]
            new_h, new_w = int(h * self.working_scale), int(w * self.working_scale)
            self.working_image = cv.resize(self.original_image, (new_w, new_h), interpolation=cv.INTER_AREA)
            
            self.image_path = image_path
            print(f"已載入影像: {path.name}, 原始={w}×{h}, 工作={new_w}×{new_h}")
            return True
            
        except Exception as e:
            print(f"載入影像失敗: {e}")
            return False
    
    def load_he_from_directory(self, base_dir: str = "picture/tiff") -> bool:
        """自動載入 HE 影像"""
        try:
            base_path = Path(base_dir)
            if not base_path.exists():
                print(f"目錄不存在: {base_dir}")
                return False
            
            # 尋找 HE 相關檔案 (避免匹配到 HER2)
            he_patterns = ["*_HE_*", "*HE_region*", "*H&E*"]
            he_file = None
            
            for pattern in he_patterns:
                files = list(base_path.glob(pattern + ".tiff")) + list(base_path.glob(pattern + ".tif"))
                if files:
                    he_file = files[0]
                    break
            
            # 如果上面的模式都沒找到，嘗試更精確的搜尋
            if he_file is None:
                all_files = list(base_path.glob("*.tiff")) + list(base_path.glob("*.tif"))
                for file in all_files:
                    filename = file.name.upper()
                    # 確保是 HE 但不是 HER2
                    if ("_HE_" in filename or "HE_REGION" in filename) and "HER2" not in filename:
                        he_file = file
                        break
            
            if he_file is None:
                print("未找到 HE 影像檔案")
                print(f"在目錄 {base_dir} 中搜尋的模式: {he_patterns}")
                return False
            
            print(f"找到 HE 檔案: {he_file.name}")
            return self.load_image(str(he_file))
            
        except Exception as e:
            print(f"自動載入 HE 影像失敗: {e}")
            return False
    
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
    
    def process_mask(self, params: HEMaskingParams, use_original: bool = False) -> HEProcessingResult:
        """處理 HE 細胞膜遮罩 - 專注於細胞膜分離"""
        start_time = time.time()
        
        try:
            print(f"開始處理 HE 細胞膜遮罩，use_original={use_original}")
            
            # 選擇處理影像
            if use_original and self.original_image is not None:
                img = self.original_image
                scale_factor = 1.0
                print(f"使用原始影像: {img.shape}")
            else:
                img = self.working_image
                scale_factor = self.working_scale
                print(f"使用工作影像: {img.shape if img is not None else 'None'}")
            
            if img is None:
                raise ValueError("沒有可用的影像")
            
            # 色彩空間轉換
            print("轉換為 HSV 色彩空間")
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            
            # 平滑處理
            if params.gaussian_kernel > 1:
                print(f"套用 Gaussian 平滑: {params.gaussian_kernel}x{params.gaussian_kernel}")
                hsv = cv.GaussianBlur(hsv, (params.gaussian_kernel, params.gaussian_kernel), 0)
            if params.median_kernel > 1:
                print(f"套用 Median 平滑: {params.median_kernel}x{params.median_kernel}")
                hsv = cv.medianBlur(hsv, params.median_kernel)
            
            # 細胞膜遮罩 (雙 H 範圍)
            print(f"提取細胞膜遮罩: H1({params.membrane_h1_min}-{params.membrane_h1_max}), H2({params.membrane_h2_min}-{params.membrane_h2_max})")
            mask_membrane_raw = self._extract_membrane_mask(hsv, params)
            print(f"原始細胞膜遮罩像素數: {np.sum(mask_membrane_raw > 0)}")
            
            # 形態學清理
            print("進行形態學清理...")
            mask_membrane_clean = self._morphology_cleanup(
                mask_membrane_raw, params.membrane_kernel_size, 
                params.membrane_open_iter, params.membrane_close_iter
            )
            
            # 面積過濾
            print("進行面積過濾...")
            mask_membrane_clean = self._area_filter(mask_membrane_clean, int(params.min_membrane_area * scale_factor**2))
            
            # 建立疊加圖
            print("建立疊加圖...")
            overlay_membrane = self._create_overlay(img, mask_membrane_clean, params.alpha, color=(0, 255, 255))  # 黃色細胞膜
            
            # 建立提取結果 (RGBA)
            print("建立提取結果...")
            extract_membrane = self._create_extraction(img, mask_membrane_clean)
            
            # 統計資訊
            total_pixels = img.shape[0] * img.shape[1]
            membrane_pixels = np.sum(mask_membrane_clean > 0)
            
            processing_time = time.time() - start_time
            print(f"處理完成，耗時: {processing_time:.2f}秒")
            
            return HEProcessingResult(
                mask_membrane_raw=mask_membrane_raw,
                mask_membrane_clean=mask_membrane_clean,
                overlay_membrane=overlay_membrane,
                extract_membrane=extract_membrane,
                params=params,
                processing_time=processing_time,
                total_pixels=total_pixels,
                membrane_pixels=membrane_pixels,
                membrane_area_percent=100.0 * membrane_pixels / total_pixels
            )
            
        except Exception as e:
            print(f"處理遮罩時發生錯誤: {e}")
            raise
    
    def _extract_membrane_mask(self, hsv: np.ndarray, params: HEMaskingParams) -> np.ndarray:
        """提取細胞膜遮罩 (雙 H 範圍聯集)"""
        # 第一段 H 範圍
        lower1 = np.array([params.membrane_h1_min, params.membrane_s_min, params.membrane_v_min])
        upper1 = np.array([params.membrane_h1_max, params.membrane_s_max, params.membrane_v_max])
        mask1 = cv.inRange(hsv, lower1, upper1)
        
        # 第二段 H 範圍
        lower2 = np.array([params.membrane_h2_min, params.membrane_s_min, params.membrane_v_min])
        upper2 = np.array([params.membrane_h2_max, params.membrane_s_max, params.membrane_v_max])
        mask2 = cv.inRange(hsv, lower2, upper2)
        
        # 聯集
        return cv.bitwise_or(mask1, mask2)
    
    def _morphology_cleanup(self, mask: np.ndarray, kernel_size: int, open_iter: int, close_iter: int) -> np.ndarray:
        """形態學去雜訊"""
        if kernel_size <= 0:
            return mask
            
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 開運算去雜訊
        if open_iter > 0:
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=open_iter)
        
        # 閉運算填補空洞
        if close_iter > 0:
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=close_iter)
        
        return mask
    
    def _area_filter(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """面積過濾移除小物件（優化版）"""
        if min_area <= 0:
            return mask
        
        print(f"進行面積過濾，最小面積: {min_area}")
        
        # 如果遮罩像素太多，先進行形態學預處理減少雜訊
        mask_pixels = np.sum(mask > 0)
        print(f"遮罩像素數: {mask_pixels}")
        
        if mask_pixels > 100000:  # 超過 10 萬像素時使用快速模式
            print("使用快速面積過濾模式")
            # 使用更大的開運算來移除小碎片
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
            filtered_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=3)
            print(f"快速過濾後像素數: {np.sum(filtered_mask > 0)}")
            return filtered_mask
        
        # 正常面積過濾
        print("使用標準連通元件分析")
        try:
            # 連通元件分析
            num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
            print(f"找到 {num_labels-1} 個連通元件")
            
            # 如果連通元件太多，改用快速模式
            if num_labels > 5000:
                print(f"連通元件過多 ({num_labels-1} 個)，改用快速模式")
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
                filtered_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=3)
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
    
    def _create_overlay(self, img: np.ndarray, mask: np.ndarray, alpha: int, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
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
    
    def save_results(self, result: HEProcessingResult, output_dir: str) -> bool:
        """儲存處理結果"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 取得檔案基本名稱
            if self.image_path:
                basename = Path(self.image_path).stem
            else:
                basename = "he_result"
            
            # 建立子資料夾
            membrane_dir = output_path / "membrane"
            membrane_dir.mkdir(exist_ok=True)
            
            # 使用全尺寸處理 (如果需要)
            if self.original_image is not None:
                full_result = self.process_mask(result.params, use_original=True)
            else:
                full_result = result
            
            # 儲存遮罩 (PNG 8-bit)
            # 一律確保「白色(255)=細胞膜，黑色(0)=其他」
            mask_to_save = full_result.mask_membrane_clean
            if mask_to_save is None:
                raise ValueError("mask_membrane_clean 為 None，無法儲存遮罩")
            # 轉為0/255二值圖
            mask_to_save = ((mask_to_save > 0).astype(np.uint8)) * 255
            # 如果白色覆蓋率過高(>50%)，推測極性相反，將其反相
            white_ratio = float(np.mean(mask_to_save)) / 255.0
            if white_ratio > 0.5:
                mask_to_save = cv.bitwise_not(mask_to_save)
            self._save_png(membrane_dir / f"{basename}_mask_membrane.png", mask_to_save)
            
            # 儲存疊加圖
            self._save_png(membrane_dir / f"{basename}_overlay_membrane.png", full_result.overlay_membrane)
            
            # 儲存提取結果 (RGBA)
            self._save_png_rgba(membrane_dir / f"{basename}_extract_membrane.png", full_result.extract_membrane)
            
            # 儲存參數
            params_file = output_path / f"{basename}_params.json"
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(result.params.to_dict(), f, indent=2, ensure_ascii=False)
            
            print(f"結果已儲存至: {output_path}")
            return True
            
        except Exception as e:
            print(f"儲存結果失敗: {e}")
            return False
    
    def _save_png(self, filepath: Path, img: np.ndarray):
        """使用 Pillow 儲存 PNG"""
        if len(img.shape) == 3:
            # BGR -> RGB
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
        else:
            # 灰階
            pil_img = Image.fromarray(img)
        
        pil_img.save(str(filepath))
    
    def _save_png_rgba(self, filepath: Path, img: np.ndarray):
        """使用 Pillow 儲存 RGBA PNG"""
        # BGRA -> RGBA
        img_rgba = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
        pil_img = Image.fromarray(img_rgba, 'RGBA')
        pil_img.save(str(filepath))