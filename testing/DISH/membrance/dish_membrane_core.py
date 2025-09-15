#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DISH 染色細胞膜遮罩核心處理邏輯
Core Processing Logic for DISH Cell Membrane Masking
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
class DishMembraneParams:
    """DISH 細胞膜遮罩參數 - 採用與 Her2 類似的方法"""
    
    # LAB 通道閾值範圍 (DISH 特徵顏色)
    lab_a_lower: int = 115    # A通道下限 (0-255)
    lab_a_upper: int = 140    # A通道上限 (0-255)
    lab_b_lower: int = 115    # B通道下限 (0-255)  
    lab_b_upper: int = 145    # B通道上限 (0-255)
    
    # DAB 通道閾值範圍 (DISH 染色強度)
    dab_lower: int = 10       # DAB下限 (0-255)
    dab_upper: int = 255      # DAB上限 (0-255)
    
    # HSV 輔助閾值 (可選)
    h_min: int = 0
    h_max: int = 179
    s_min: int = 0
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255
    
    # 形態學參數
    membrane_kernel_size: int = 3
    membrane_open_iter: int = 1
    membrane_close_iter: int = 1
    membrane_dilate_iter: int = 1
    
    # 基本過濾參數
    min_membrane_area: int = 50
    
    # 膜擬合參數 (針對DISH無真實膜染色的近似方法)
    membrane_approx_radius: int = 5  # 膜擬合膨脹半徑 (0-50)
    
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
            'h_min': self.h_min,
            'h_max': self.h_max,
            's_min': self.s_min,
            's_max': self.s_max,
            'v_min': self.v_min,
            'v_max': self.v_max,
            'membrane_kernel_size': self.membrane_kernel_size,
            'membrane_open_iter': self.membrane_open_iter,
            'membrane_close_iter': self.membrane_close_iter,
            'membrane_dilate_iter': self.membrane_dilate_iter,
            'min_membrane_area': self.min_membrane_area,
            'membrane_approx_radius': self.membrane_approx_radius,
            'alpha': self.alpha
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DishMembraneParams':
        """從字典建立參數"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass 
class DishMembraneResult:
    """DISH 細胞膜處理結果"""
    
    # 基本通道
    lab_channels: np.ndarray  # 3通道 LAB 影像  
    dab_channel: np.ndarray   # DAB 通道 (DISH 染色)
    hsv_channels: np.ndarray  # HSV 通道
    
    # DISH 細胞膜遮罩
    mask_membrane_raw: np.ndarray
    mask_membrane_clean: np.ndarray
    mask_hsv: np.ndarray  # HSV 輔助遮罩
    
    # 膜擬合結果 (核擴張近似膜區域)
    mask_membrane_approx: np.ndarray
    
    # 疊加圖
    overlay_membrane: np.ndarray
    
    # 提取結果
    extract_membrane: np.ndarray
    
    # 參數與統計
    params: DishMembraneParams
    processing_time: float
    
    # 統計資訊
    total_pixels: int
    membrane_pixels: int
    membrane_coverage_percent: float
    
    # DAB 強度統計
    dab_mean: float
    dab_median: float

class DishMembraneProcessor:
    """DISH 染色細胞膜遮罩處理核心"""
    
    def __init__(self):
        self.original_image = None
        self.working_image = None
        self.image_filename = ""
        
        # 縮放設定
        self.working_scale = 0.5  # 運算用縮放比例
        self.max_working_size = 4000  # 運算圖最大邊長
        
    def load_image(self, image_path: str) -> bool:
        """載入影像"""
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"影像檔案不存在: {image_path}")
            
            # 讀取影像
            img = cv.imread(str(path), cv.IMREAD_COLOR)
            if img is None:
                # 嘗試 16-bit
                img = cv.imread(str(path), cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
                if img is not None and img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)
                    
            if img is None:
                raise ValueError("無法讀取影像檔案")
                
            self.original_image = img
            self.image_filename = path.stem
            self._prepare_working_image()
            return True
            
        except Exception as e:
            print(f"載入影像失敗: {e}")
            return False
    
    def load_dish_from_directory(self, tiff_dir: str = "picture/tiff") -> bool:
        """從目錄自動載入 DISH 影像"""
        try:
            tiff_path = Path(tiff_dir)
            if not tiff_path.exists():
                raise FileNotFoundError(f"目錄不存在: {tiff_dir}")
                
            # 尋找 DISH 檔案
            dish_files = (list(tiff_path.glob("*DISH*.tiff")) + 
                         list(tiff_path.glob("*DISH*.tif")) +
                         list(tiff_path.glob("*dish*.tiff")) + 
                         list(tiff_path.glob("*dish*.tif")))
            
            if not dish_files:
                raise FileNotFoundError("找不到 DISH 染色影像")
                
            # 載入第一個找到的檔案
            return self.load_image(str(dish_files[0]))
            
        except Exception as e:
            print(f"自動載入 DISH 影像失敗: {e}")
            return False
    
    def _prepare_working_image(self):
        """準備工作用影像"""
        if self.original_image is None:
            return
            
        height, width = self.original_image.shape[:2]
        
        # 計算縮放比例
        if max(height, width) > self.max_working_size:
            scale = self.max_working_size / max(height, width)
        else:
            scale = self.working_scale
            
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        self.working_image = cv.resize(
            self.original_image, 
            (new_width, new_height), 
            interpolation=cv.INTER_AREA
        )
    
    def process_membrane(self, params: DishMembraneParams, use_original: bool = False) -> DishMembraneResult:
        """執行DISH細胞膜遮罩處理"""
        start_time = time.time()
        
        # 選擇處理圖像
        img = self.original_image if use_original else self.working_image
        if img is None:
            raise ValueError("尚未載入影像")
        
        try:
            print(f"DEBUG: 開始處理膜分離，影像尺寸: {img.shape}")
            
            # 1. 顏色空間轉換
            lab_img = rgb2lab(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            lab_img = (lab_img * 255 / 100).astype(np.uint8)  # 正規化到 0-255
            print(f"DEBUG: LAB轉換完成，A通道範圍: {lab_img[:,:,1].min()}-{lab_img[:,:,1].max()}")
            print(f"DEBUG: LAB轉換完成，B通道範圍: {lab_img[:,:,2].min()}-{lab_img[:,:,2].max()}")
            
            # 2. H&E 分離獲得 DAB 通道
            rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            hed_img = rgb2hed(rgb_img)
            
            # DAB 通道 (第3通道，索引2)
            dab_channel = hed_img[:, :, 2]
            dab_channel = exposure.rescale_intensity(dab_channel, out_range=(0, 255))
            dab_channel = dab_channel.astype(np.uint8)
            print(f"DEBUG: DAB通道提取完成，範圍: {dab_channel.min()}-{dab_channel.max()}")
            
            # 3. LAB 閾值處理 - 提取DISH特徵顏色
            a_channel = lab_img[:, :, 1]  # A 通道
            b_channel = lab_img[:, :, 2]  # B 通道
            
            # LAB 顏色範圍遮罩 - 改用OR邏輯
            mask_a = cv.inRange(a_channel, params.lab_a_lower, params.lab_a_upper)
            mask_b = cv.inRange(b_channel, params.lab_b_lower, params.lab_b_upper)
            mask_lab = cv.bitwise_or(mask_a, mask_b)  # 改為OR邏輯
            print(f"DEBUG: LAB遮罩創建完成，A遮罩像素數: {cv.countNonZero(mask_a)}")
            print(f"DEBUG: LAB遮罩創建完成，B遮罩像素數: {cv.countNonZero(mask_b)}")
            print(f"DEBUG: LAB遮罩創建完成，OR組合遮罩像素數: {cv.countNonZero(mask_lab)}")
            
            # 4. DAB 通道閾值處理
            mask_dab = cv.inRange(dab_channel, params.dab_lower, params.dab_upper)
            print(f"DEBUG: DAB遮罩創建完成，像素數: {cv.countNonZero(mask_dab)}")
            
            # 5. HSV 輔助遮罩（可選）
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            lower_hsv = np.array([params.h_min, params.s_min, params.v_min], dtype=np.uint8)
            upper_hsv = np.array([params.h_max, params.s_max, params.v_max], dtype=np.uint8)
            mask_hsv = cv.inRange(hsv, lower_hsv, upper_hsv)
            print(f"DEBUG: HSV遮罩創建完成，像素數: {cv.countNonZero(mask_hsv)}")
            
            # 6. 結合所有遮罩
            mask_combined = cv.bitwise_and(mask_lab, mask_dab)
            print(f"DEBUG: LAB+DAB組合遮罩像素數: {cv.countNonZero(mask_combined)}")
            # 只有當HSV不是全選狀態時才使用HSV輔助
            # 檢查HSV是否為預設的全選狀態
            is_hsv_default = (params.h_min == 0 and params.h_max == 179 and 
                            params.s_min == 0 and params.s_max == 255 and 
                            params.v_min == 0 and params.v_max == 255)
            
            if not is_hsv_default and np.sum(mask_hsv) > 0:
                mask_combined = cv.bitwise_and(mask_combined, mask_hsv)
                print(f"DEBUG: 加入HSV後組合遮罩像素數: {cv.countNonZero(mask_combined)}")
            else:
                print(f"DEBUG: HSV為預設狀態，跳過HSV濾波")
            
            # 7. 形態學清理
            kernel_size = params.membrane_kernel_size
            if not use_original and kernel_size > 3:
                kernel_size = max(1, int(kernel_size * 0.5))
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            print(f"DEBUG: 形態學處理 - kernel大小: {kernel_size}")
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # 開運算去雜訊
            mask_clean = cv.morphologyEx(mask_combined, cv.MORPH_OPEN, kernel, 
                                       iterations=params.membrane_open_iter)
            print(f"DEBUG: 開運算後遮罩像素數: {cv.countNonZero(mask_clean)}")
            
            # 閉運算補洞
            mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_CLOSE, kernel, 
                                       iterations=params.membrane_close_iter)
            print(f"DEBUG: 閉運算後遮罩像素數: {cv.countNonZero(mask_clean)}")
            
            # 輕微膨脹強化細胞膜
            if params.membrane_dilate_iter > 0:
                mask_clean = cv.dilate(mask_clean, kernel, iterations=params.membrane_dilate_iter)
                print(f"DEBUG: 膨脹後遮罩像素數: {cv.countNonZero(mask_clean)}")
            
            # 8. 面積過濾
            mask_filtered = self._filter_by_area(mask_clean, params.min_membrane_area)
            print(f"DEBUG: 面積過濾後遮罩像素數: {cv.countNonZero(mask_filtered)}")
            
            # 9. 膜擬合處理 (核擴張近似膜區域)
            mask_membrane_approx = self._create_membrane_approximation(mask_filtered, params.membrane_approx_radius)
            print(f"DEBUG: 膜擬合後遮罩像素數: {cv.countNonZero(mask_membrane_approx)}")
            
            # 10. 產生疊加圖和提取圖
            overlay = self._create_overlay(img, mask_membrane_approx, params.alpha)
            extract = self._create_extract(img, mask_membrane_approx)
            print(f"DEBUG: 疊加圖生成完成，透明度: {params.alpha}%")
            
            # 11. 計算統計
            stats = self._calculate_statistics(mask_membrane_approx, dab_channel)
            
            processing_time = time.time() - start_time
            
            return DishMembraneResult(
                lab_channels=lab_img,
                dab_channel=dab_channel,
                hsv_channels=hsv,
                mask_membrane_raw=mask_combined,
                mask_membrane_clean=mask_filtered,
                mask_hsv=mask_hsv,
                mask_membrane_approx=mask_membrane_approx,
                overlay_membrane=overlay,
                extract_membrane=extract,
                params=params,
                processing_time=processing_time,
                **stats
            )
            
        except Exception as e:
            raise RuntimeError(f"DISH細胞膜處理失敗: {e}")
    
    def _filter_by_area(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """依面積過濾連通區域"""
        if min_area <= 0:
            return mask
            
        # 尋找連通元件
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
        
        # 建立過濾後的遮罩
        filtered_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):  # 跳過背景標籤0
            area = stats[i, cv.CC_STAT_AREA]
            if area >= min_area:
                filtered_mask[labels == i] = 255
                
        return filtered_mask
    
    def _create_membrane_approximation(self, mask: np.ndarray, radius: int) -> np.ndarray:
        """建立膜擬合區域 - 透過核擴張近似細胞膜"""
        if radius <= 0:
            return mask
        
        # 創建圓形擴張核心
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        
        # 膨脹操作來模擬從細胞核擴張到細胞膜
        membrane_approx = cv.dilate(mask, kernel, iterations=1)
        
        return membrane_approx
    
    def _create_overlay(self, img: np.ndarray, mask: np.ndarray, alpha_percent: int) -> np.ndarray:
        """建立彩色疊圖"""
        alpha = alpha_percent / 100.0
        
        # 建立彩色遮罩
        color = np.zeros_like(img)
        color[mask == 255] = (0, 255, 0)  # BGR 綠色 (DISH膜)
        
        # 正確的透明度邏輯：
        # alpha = 0%: 只看原圖 (img權重1.0, color權重0.0)
        # alpha = 100%: 只看遮罩 (img權重0.0, color權重1.0)
        img_weight = 1.0 - alpha
        color_weight = alpha
        
        return cv.addWeighted(img, img_weight, color, color_weight, 0)
    
    def _create_extract(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """建立提取圖 - 只顯示細胞膜區域"""
        extract = img.copy()
        extract[mask == 0] = [255, 255, 255]  # 白色背景
        return extract
    
    def _calculate_statistics(self, mask: np.ndarray, dab_channel: np.ndarray) -> Dict[str, Any]:
        """計算統計資訊"""
        total_pixels = mask.shape[0] * mask.shape[1]
        membrane_pixels = int(np.sum(mask > 0))
        membrane_coverage_percent = (membrane_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # DAB 強度統計
        if membrane_pixels > 0:
            dab_values = dab_channel[mask > 0]
            dab_mean = float(np.mean(dab_values))
            dab_median = float(np.median(dab_values))
        else:
            dab_mean = 0.0
            dab_median = 0.0
        
        return {
            'total_pixels': total_pixels,
            'membrane_pixels': membrane_pixels,
            'membrane_coverage_percent': membrane_coverage_percent,
            'dab_mean': dab_mean,
            'dab_median': dab_median
        }
    
    def save_results(self, result: DishMembraneResult, output_dir: str) -> bool:
        """儲存處理結果"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 如果結果是基於縮圖，可選擇重新處理原圖以獲得高品質結果
            # 但由於膜擬合操作對大圖像很慢，暫時跳過自動重新處理
            if result.mask_membrane_clean.shape[:2] != self.original_image.shape[:2]:
                print("DEBUG: 檢測到縮圖結果，跳過原圖重新處理以避免長時間等待")
                # result = self.process_membrane(result.params, use_original=True)  # 暫時禁用
            
            prefix = self.image_filename
            
            # 儲存圖像 - PNG格式
            save_items = [
                (result.lab_channels[:,:,1], '01_LAB_A通道'),
                (result.lab_channels[:,:,2], '02_LAB_B通道'),
                (result.dab_channel, '03_DAB通道'),
                (result.mask_hsv, '04_HSV輔助遮罩'),
                (result.mask_membrane_raw, '05_原始細胞膜遮罩'),
                (result.mask_membrane_clean, '06_清理後細胞膜遮罩'),
                (result.mask_membrane_approx, '07_膜擬合遮罩'),
                (result.overlay_membrane, '08_細胞膜疊加圖'),
                (result.extract_membrane, '09_細胞膜提取圖')
            ]
            
            saved_files = []
            
            for img_data, desc in save_items:
                if img_data is not None:
                    filename = f"{prefix}_{desc}.png"
                    filepath = output_path / filename
                    
                    # 確保影像格式正確
                    save_img = img_data.copy()
                    if save_img.dtype != np.uint8:
                        save_img = save_img.astype(np.uint8)
                    
                    # 使用PIL儲存PNG
                    try:
                        if len(save_img.shape) == 3:
                            pil_img = Image.fromarray(save_img[:, :, ::-1])  # BGR -> RGB
                        else:
                            pil_img = Image.fromarray(save_img, mode='L')
                        pil_img.save(str(filepath), format='PNG')
                        saved_files.append(filename)
                    except Exception as e:
                        print(f"儲存 {filename} 失敗: {e}")
                        continue
            
            # 儲存參數
            params_file = output_path / f"{prefix}_細胞膜參數設定.json"
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(result.params.to_dict(), f, ensure_ascii=False, indent=2)
            saved_files.append(params_file.name)
            
            # 儲存統計報告
            stats_file = output_path / f"{prefix}_細胞膜統計報告.txt"
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(self._generate_report(result))
            saved_files.append(stats_file.name)
            
            print(f"已儲存 {len(saved_files)} 個檔案至: {output_path}")
            return len(saved_files) > 0
            
        except Exception as e:
            print(f"儲存失敗: {e}")
            return False
    
    def _generate_report(self, result: DishMembraneResult) -> str:
        """產生統計報告"""
        orig_h, orig_w = self.original_image.shape[:2]
        
        return f"""
DISH 染色細胞膜遮罩處理報告
=========================

檔案資訊:
• 原始檔名: {self.image_filename}
• 影像尺寸: {orig_w} × {orig_h}
• 處理時間: {result.processing_time:.3f} 秒

處理參數:
• LAB A 範圍: {result.params.lab_a_lower}-{result.params.lab_a_upper}
• LAB B 範圍: {result.params.lab_b_lower}-{result.params.lab_b_upper}
• DAB 範圍: {result.params.dab_lower}-{result.params.dab_upper}
• HSV 範圍: H({result.params.h_min}-{result.params.h_max}) S({result.params.s_min}-{result.params.s_max}) V({result.params.v_min}-{result.params.v_max})
• Kernel 大小: {result.params.membrane_kernel_size}
• 開運算次數: {result.params.membrane_open_iter}
• 閉運算次數: {result.params.membrane_close_iter}
• 膨脹次數: {result.params.membrane_dilate_iter}
• 最小面積: {result.params.min_membrane_area}
• 透明度: {result.params.alpha}%

統計結果:
• 總像素數: {result.total_pixels:,}
• 細胞膜像素: {result.membrane_pixels:,} ({result.membrane_coverage_percent:.2f}%)
• DAB 平均強度: {result.dab_mean:.2f}
• DAB 中位數強度: {result.dab_median:.2f}

品質評估:
• 細胞膜覆蓋率: {'適中' if 1 <= result.membrane_coverage_percent <= 30 else ('過少' if result.membrane_coverage_percent < 1 else '過多')}
• DAB 染色強度: {'良好' if result.dab_mean > 20 else '較弱'}

生成時間: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

    def get_image_info(self) -> Dict[str, Any]:
        """取得影像資訊"""
        if self.original_image is None:
            return {}
            
        orig_h, orig_w = self.original_image.shape[:2]
        work_h, work_w = self.working_image.shape[:2] if self.working_image is not None else (0, 0)
        
        return {
            'filename': self.image_filename,
            'original_size': (orig_w, orig_h),
            'working_size': (work_w, work_h),
            'working_scale': work_w / orig_w if orig_w > 0 else 0
        }
