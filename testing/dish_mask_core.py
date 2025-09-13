#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DISH 染色細胞遮罩核心處理邏輯
Core Processing Logic for DISH Cell Masking
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
class MaskingParams:
    """遮罩處理參數"""
    # HSV 閾值
    h_min: int = 0
    h_max: int = 179
    s_min: int = 0
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255
    
    # 形態學參數
    kernel_size: int = 3
    open_iter: int = 1
    close_iter: int = 1
    dilate_iter: int = 2
    
    # Watershed 參數
    dist_threshold: float = 0.35
    
    # 顯示參數
    alpha: int = 50  # 透明度百分比
    
    # 遮罩模式
    invert_mask: bool = False  # True=保留細胞核遮掉細胞質, False=保留細胞質遮掉背景
    
    def to_dict(self) -> Dict[str, Any]:
        """轉為字典格式"""
        return {
            'Hmin': self.h_min, 'Hmax': self.h_max,
            'Smin': self.s_min, 'Smax': self.s_max,
            'Vmin': self.v_min, 'Vmax': self.v_max,
            'kernel_size': self.kernel_size,
            'open_iter': self.open_iter,
            'close_iter': self.close_iter,
            'dilate_iter': self.dilate_iter,
            'dist_threshold': self.dist_threshold,
            'alpha': self.alpha
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MaskingParams':
        """從字典建立參數"""
        return cls(
            h_min=data.get('Hmin', 0),
            h_max=data.get('Hmax', 179),
            s_min=data.get('Smin', 0),
            s_max=data.get('Smax', 255),
            v_min=data.get('Vmin', 0),
            v_max=data.get('Vmax', 255),
            kernel_size=data.get('kernel_size', 3),
            open_iter=data.get('open_iter', 1),
            close_iter=data.get('close_iter', 1),
            dilate_iter=data.get('dilate_iter', 2),
            dist_threshold=data.get('dist_threshold', 0.35),
            alpha=data.get('alpha', 50)
        )

@dataclass
class ProcessingResult:
    """處理結果"""
    mask: np.ndarray
    clean: np.ndarray
    sure_bg: np.ndarray
    sure_fg: np.ndarray
    dist: np.ndarray
    cells: np.ndarray
    overlay: np.ndarray
    params: MaskingParams
    processing_time: float
    
    # 統計資訊
    total_pixels: int
    mask_pixels: int
    cell_pixels: int
    cell_count: int
    mask_area_percent: float
    cell_area_percent: float

class DishMaskProcessor:
    """DISH 染色細胞遮罩處理核心"""
    
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
    
    def process_mask(self, params: MaskingParams, use_original: bool = False) -> ProcessingResult:
        """執行遮罩處理"""
        start_time = time.time()
        
        # 選擇處理圖像
        img = self.original_image if use_original else self.working_image
        if img is None:
            raise ValueError("尚未載入影像")
        
        try:
            # 1. HSV 顏色遮罩 - 識別 DISH 染色區域
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            lower = np.array([params.h_min, params.s_min, params.v_min], dtype=np.uint8)
            upper = np.array([params.h_max, params.s_max, params.v_max], dtype=np.uint8)
            mask = cv.inRange(hsv, lower, upper)
            
            # 2. 形態學清理
            kernel_size = params.kernel_size
            if not use_original and kernel_size > 3:
                # 縮圖時調整 kernel 大小
                kernel_size = max(1, int(kernel_size * 0.5))
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # 開運算去雜訊
            clean = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, 
                                  iterations=params.open_iter)
            
            # 閉運算補洞
            clean = cv.morphologyEx(clean, cv.MORPH_CLOSE, kernel, 
                                  iterations=params.close_iter)
            
            # 3. Watershed 分離
            if np.sum(clean) > 0:
                sure_bg, sure_fg, dist, cells = self._watershed_separation(
                    clean, kernel, params, img
                )
            else:
                # 無前景的情況
                sure_bg = mask.copy()
                sure_fg = mask.copy()
                dist = np.zeros_like(mask, dtype=np.uint8)
                cells = mask.copy()
            
            # 4. 產生彩色疊圖
            overlay = self._create_overlay(img, cells, params.alpha)
            
            # 5. 計算統計
            stats = self._calculate_statistics(mask, cells)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                mask=mask,
                clean=clean,
                sure_bg=sure_bg,
                sure_fg=sure_fg,
                dist=dist,
                cells=cells,
                overlay=overlay,
                params=params,
                processing_time=processing_time,
                **stats
            )
            
        except Exception as e:
            raise RuntimeError(f"遮罩處理失敗: {e}")
    
    def _watershed_separation(self, clean: np.ndarray, kernel: np.ndarray, 
                            params: MaskingParams, img: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Watershed 分離處理"""
        # 確定背景
        sure_bg = cv.dilate(clean, kernel, iterations=params.dilate_iter)
        
        # 距離變換找確定前景
        dist = cv.distanceTransform(clean, cv.DIST_L2, 3)
        
        if dist.max() > 0:
            _, sure_fg = cv.threshold(
                dist, params.dist_threshold * dist.max(), 255, cv.THRESH_BINARY
            )
            sure_fg = sure_fg.astype(np.uint8)
            
            # 未知區域
            unknown = cv.subtract(sure_bg, sure_fg)
            
            # 標記連通元件
            _, markers = cv.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # 套用 Watershed
            markers = cv.watershed(img.copy(), markers)
            
            # 建立細胞遮罩
            cells = (markers > 1).astype(np.uint8) * 255
            
            # 應用遮罩模式選擇
            if params.invert_mask:
                cells = cv.bitwise_not(cells)
            
            # 距離圖正規化
            dist_norm = ((dist / dist.max()) * 255).astype(np.uint8)
        else:
            cells = clean.copy()
            sure_fg = clean.copy()
            dist_norm = np.zeros_like(clean, dtype=np.uint8)
        
        return sure_bg, sure_fg, dist_norm, cells
    
    def _create_overlay(self, img: np.ndarray, cells: np.ndarray, alpha_percent: int) -> np.ndarray:
        """建立彩色疊圖"""
        alpha = alpha_percent / 100.0
        color = np.zeros_like(img)
        color[cells == 255] = (0, 0, 255)  # BGR 紅色
        return cv.addWeighted(img, 1.0, color, alpha, 0)
    
    def _calculate_statistics(self, mask: np.ndarray, cells: np.ndarray) -> Dict[str, Any]:
        """計算統計資訊"""
        total_pixels = mask.shape[0] * mask.shape[1]
        mask_pixels = int(np.sum(mask > 0))
        cell_pixels = int(np.sum(cells > 0))
        
        mask_area_percent = (mask_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        cell_area_percent = (cell_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # 估算細胞數量
        if cell_pixels > 0:
            num_labels, _ = cv.connectedComponents(cells)
            cell_count = num_labels - 1
        else:
            cell_count = 0
        
        return {
            'total_pixels': total_pixels,
            'mask_pixels': mask_pixels,
            'cell_pixels': cell_pixels,
            'cell_count': cell_count,
            'mask_area_percent': mask_area_percent,
            'cell_area_percent': cell_area_percent
        }
    
    def save_results(self, result: ProcessingResult, output_dir: str) -> bool:
        """儲存處理結果"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 如果結果是基於縮圖，重新處理原圖以獲得高品質結果
            if result.mask.shape[:2] != self.original_image.shape[:2]:
                result = self.process_mask(result.params, use_original=True)
            
            prefix = self.image_filename
            
            # 儲存圖像 - 恢復完整功能
            save_items = [
                (result.mask, '01_HSV遮罩'),
                (result.clean, '02_形態學清理'),
                (result.sure_bg, '03_確定背景'),
                (result.sure_fg, '04_確定前景'),
                (result.dist, '05_距離變換'),
                (result.cells, '06_細胞分離'),
                (result.overlay, '07_彩色疊圖')
            ]
            
            saved_files = []
            
            for img_data, desc in save_items:
                if img_data is not None:
                    filename = f"{prefix}_{desc}.tiff"
                    filepath = output_path / filename
                    
                    # 確保影像格式正確
                    save_img = img_data.copy()
                    if save_img.dtype != np.uint8:
                        save_img = save_img.astype(np.uint8)
                    
                    # 嘗試 OpenCV，失敗則用 PIL
                    success = cv.imwrite(str(filepath), save_img, [cv.IMWRITE_TIFF_COMPRESSION, 1])
                    
                    if not success:
                        try:
                            if len(save_img.shape) == 3:
                                pil_img = Image.fromarray(save_img[:, :, ::-1])  # BGR -> RGB
                            else:
                                pil_img = Image.fromarray(save_img, mode='L')
                            pil_img.save(str(filepath), format='TIFF', compression='lzw')
                            success = True
                        except Exception:
                            # 最後嘗試 JPG
                            jpg_path = filepath.with_suffix('.jpg')
                            try:
                                if len(save_img.shape) == 3:
                                    pil_img = Image.fromarray(save_img[:, :, ::-1])
                                else:
                                    pil_img = Image.fromarray(save_img, mode='L')
                                pil_img.save(str(jpg_path), format='JPEG', quality=95)
                                saved_files.append(jpg_path.name)
                                continue
                            except Exception:
                                continue
                    
                    if success:
                        saved_files.append(filename)
            
            # 儲存參數
            params_file = output_path / f"{prefix}_參數設定.json"
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(result.params.to_dict(), f, ensure_ascii=False, indent=2)
            saved_files.append(params_file.name)
            
            # 儲存統計報告
            stats_file = output_path / f"{prefix}_統計報告.txt"
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(self._generate_report(result))
            saved_files.append(stats_file.name)
            
            print(f"已儲存 {len(saved_files)} 個檔案至: {output_path}")
            return len(saved_files) > 0
            
        except Exception as e:
            print(f"儲存失敗: {e}")
            return False
    
    def _generate_report(self, result: ProcessingResult) -> str:
        """產生統計報告"""
        orig_h, orig_w = self.original_image.shape[:2]
        
        return f"""
DISH 染色細胞遮罩處理報告
========================

檔案資訊:
• 原始檔名: {self.image_filename}
• 影像尺寸: {orig_w} × {orig_h}
• 處理時間: {result.processing_time:.3f} 秒

處理參數:
• HSV 範圍: H({result.params.h_min}-{result.params.h_max}) S({result.params.s_min}-{result.params.s_max}) V({result.params.v_min}-{result.params.v_max})
• Kernel 大小: {result.params.kernel_size}
• 開運算次數: {result.params.open_iter}
• 閉運算次數: {result.params.close_iter}
• 膨脹次數: {result.params.dilate_iter}
• 距離閾值: {result.params.dist_threshold:.2f}
• 透明度: {result.params.alpha}%

統計結果:
• 總像素數: {result.total_pixels:,}
• HSV 遮罩像素: {result.mask_pixels:,} ({result.mask_area_percent:.2f}%)
• 細胞區域像素: {result.cell_pixels:,} ({result.cell_area_percent:.2f}%)
• 估算細胞數量: {result.cell_count}

品質評估:
• 遮罩覆蓋率: {'適中' if 5 <= result.mask_area_percent <= 50 else ('過少' if result.mask_area_percent < 5 else '過多')}
• 細胞分離效果: {'良好' if result.cell_count > 0 else '需調整'}

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