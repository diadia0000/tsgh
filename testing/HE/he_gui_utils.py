#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HE 遮罩 GUI 通用工具模組
Common utilities for HE mask GUI
"""

import numpy as np
import cv2 as cv
from PyQt5.QtWidgets import QLabel, QSlider, QGridLayout, QGroupBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from typing import Dict, List, Tuple, Callable, Any

class SliderGroup:
    """滑桿組管理類"""
    
    def __init__(self):
        self.sliders = {}
    
    def create_slider_with_label(self, key: str, label: str, min_val: int, max_val: int, 
                               default: int, callback: Callable = None, 
                               value_formatter: Callable = None) -> Tuple[QLabel, QSlider, QLabel]:
        """建立帶標籤的滑桿"""
        label_widget = QLabel(f"{label}:")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setTracking(False)  # 防抖動
        
        value_label = QLabel(str(default))
        value_label.setMinimumWidth(40)
        value_label.setAlignment(Qt.AlignCenter)
        
        # 值更新回調
        def update_value(value):
            if value_formatter:
                formatted_value = value_formatter(value)
            else:
                formatted_value = str(value)
            value_label.setText(formatted_value)
            
        slider.valueChanged.connect(update_value)
        
        if callback:
            slider.valueChanged.connect(callback)
        
        self.sliders[key] = (slider, value_label)
        
        return label_widget, slider, value_label
    
    def add_sliders_to_layout(self, layout: QGridLayout, configs: List[Tuple], 
                            callback: Callable = None, value_formatters: Dict = None):
        """批量添加滑桿到佈局"""
        for i, config in enumerate(configs):
            if len(config) == 5:
                key, label, min_val, max_val, default = config
                formatter = None
            elif len(config) == 6:
                key, label, min_val, max_val, default, formatter = config
            else:
                raise ValueError(f"Invalid config format: {config}")
            
            # 使用自訂格式器或全域格式器
            if value_formatters and key in value_formatters:
                formatter = value_formatters[key]
            
            label_widget, slider, value_label = self.create_slider_with_label(
                key, label, min_val, max_val, default, callback, formatter
            )
            
            layout.addWidget(label_widget, i, 0)
            layout.addWidget(slider, i, 1)
            layout.addWidget(value_label, i, 2)
    
    def get_value(self, key: str) -> int:
        """取得滑桿值"""
        if key in self.sliders:
            return self.sliders[key][0].value()
        return 0
    
    def set_value(self, key: str, value: int):
        """設定滑桿值"""
        if key in self.sliders:
            slider, value_label = self.sliders[key]
            slider.setValue(value)
            value_label.setText(str(value))
    
    def reset_values(self, defaults: Dict[str, int]):
        """重置所有滑桿值"""
        for key, default_val in defaults.items():
            self.set_value(key, default_val)

class ImageDisplayHelper:
    """影像顯示輔助類"""
    
    @staticmethod
    def numpy_to_qpixmap(img: np.ndarray) -> QPixmap:
        """將 NumPy 陣列轉換為 QPixmap"""
        try:
            if img is None:
                return QPixmap()
                
            # 確保資料連續性
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img)
                
            # BGR 轉 RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                rgb = cv.cvtColor(img, cv.COLOR_BGRA2RGB)
            else:
                rgb = img
                
            height, width = rgb.shape[:2]
            
            # 確保是 uint8 格式
            if rgb.dtype != np.uint8:
                if rgb.dtype == np.uint16:
                    rgb = (rgb / 256).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)
            
            if len(rgb.shape) == 3:
                bytes_per_line = 3 * width
                format_type = QImage.Format_RGB888
            else:
                bytes_per_line = width
                format_type = QImage.Format_Grayscale8
            
            q_image = QImage(rgb.data.tobytes(), width, height, bytes_per_line, format_type)
            
            if q_image.isNull():
                return QPixmap()
                
            return QPixmap.fromImage(q_image)
            
        except Exception as e:
            print(f"影像轉換錯誤: {e}")
            return QPixmap()
    
    @staticmethod
    def scale_pixmap_to_fit(pixmap: QPixmap, max_width: int, max_height: int) -> QPixmap:
        """縮放 QPixmap 以適應指定尺寸"""
        if pixmap.isNull():
            return pixmap
            
        if (pixmap.width() > max_width or pixmap.height() > max_height):
            return pixmap.scaled(
                max_width, max_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        return pixmap

class GroupBoxFactory:
    """QGroupBox 工廠類"""
    
    @staticmethod
    def create_slider_group(title: str, configs: List[Tuple], callback: Callable = None,
                          value_formatters: Dict = None) -> Tuple[QGroupBox, SliderGroup]:
        """建立包含滑桿的群組框"""
        group = QGroupBox(title)
        layout = QGridLayout()
        
        slider_group = SliderGroup()
        slider_group.add_sliders_to_layout(layout, configs, callback, value_formatters)
        
        group.setLayout(layout)
        return group, slider_group

class ParameterManager:
    """參數管理器"""
    
    def __init__(self):
        self.slider_groups = {}
    
    def register_slider_group(self, name: str, slider_group: SliderGroup):
        """註冊滑桿組"""
        self.slider_groups[name] = slider_group
    
    def get_all_values(self) -> Dict[str, Any]:
        """取得所有參數值"""
        values = {}
        for group_name, slider_group in self.slider_groups.items():
            for key, (slider, value_label) in slider_group.sliders.items():
                values[key] = slider.value()
        return values
    
    def set_all_values(self, values: Dict[str, Any]):
        """設定所有參數值"""
        for group_name, slider_group in self.slider_groups.items():
            for key, value in values.items():
                if key in slider_group.sliders:
                    slider, value_label = slider_group.sliders[key]
                    slider.setValue(value)
                    value_label.setText(str(value))
    
    def reset_to_defaults(self, defaults: Dict[str, Any]):
        """重置為預設值"""
        self.set_all_values(defaults)

class ValueFormatters:
    """值格式化器集合"""
    
    @staticmethod
    def percentage(value: int) -> str:
        """百分比格式"""
        return f"{value}%"
    
    @staticmethod
    def decimal_percentage(value: int) -> str:
        """小數百分比格式 (value/100)"""
        return f"{value/100:.2f}"
    
    @staticmethod
    def ensure_odd(value: int) -> str:
        """確保奇數格式"""
        actual_value = value if value % 2 == 1 else value + 1
        return str(actual_value)
    
    @staticmethod
    def with_unit(unit: str):
        """帶單位格式化器工廠"""
        def formatter(value: int) -> str:
            return f"{value} {unit}"
        return formatter

# 預定義的滑桿配置
class SliderConfigs:
    """預定義滑桿配置"""
    
    # 細胞質 HSV 配置
    CYTOPLASM_HSV = [
        ('cyto_h1_min', 'H1最小', 0, 179, 0),
        ('cyto_h1_max', 'H1最大', 0, 179, 20),
        ('cyto_h2_min', 'H2最小', 0, 179, 160),
        ('cyto_h2_max', 'H2最大', 0, 179, 179),
        ('cyto_s_min', 'S最小', 0, 255, 30),
        ('cyto_s_max', 'S最大', 0, 255, 150),
        ('cyto_v_min', 'V最小', 0, 255, 80),
        ('cyto_v_max', 'V最大', 0, 255, 255),
    ]
    
    # 細胞核 HSV 配置
    NUCLEI_HSV = [
        ('nuclei_h_min', 'H最小', 0, 179, 100),
        ('nuclei_h_max', 'H最大', 0, 179, 140),
        ('nuclei_s_min', 'S最小', 0, 255, 50),
        ('nuclei_s_max', 'S最大', 0, 255, 255),
        ('nuclei_v_min', 'V最小', 0, 255, 20),
        ('nuclei_v_max', 'V最大', 0, 255, 255),
    ]
    
    # 細胞質形態學配置
    CYTOPLASM_MORPH = [
        ('cyto_kernel_size', 'Kernel 大小', 1, 15, 3),
        ('cyto_open_iter', '開運算次數', 0, 10, 1),
        ('cyto_close_iter', '閉運算次數', 0, 10, 1),
        ('min_area_cyto', '最小面積', 10, 500, 50),
    ]
    
    # 細胞核形態學配置
    NUCLEI_MORPH = [
        ('nuclei_kernel_size', 'Kernel 大小', 1, 15, 3),
        ('nuclei_open_iter', '開運算次數', 0, 10, 1),
        ('nuclei_close_iter', '閉運算次數', 0, 10, 1),
        ('min_area_nuclei', '最小面積', 5, 200, 20),
    ]
    
    # 平滑處理配置
    SMOOTHING = [
        ('gaussian_kernel', 'Gaussian Kernel', 1, 15, 5),
        ('median_kernel', 'Median Kernel', 1, 15, 3),
    ]

# 預定義的值格式化器
DEFAULT_VALUE_FORMATTERS = {
    'cyto_kernel_size': ValueFormatters.ensure_odd,
    'nuclei_kernel_size': ValueFormatters.ensure_odd,
    'gaussian_kernel': ValueFormatters.ensure_odd,
    'median_kernel': ValueFormatters.ensure_odd,
}

class StyleSheets:
    """樣式表集合"""
    
    SAVE_BUTTON = """
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px;
            font-size: 14px;
            font-weight: bold;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:disabled {
            background-color: #cccccc;
        }
    """
    
    IMAGE_LABEL = """
        QLabel {
            border: 2px solid #cccccc;
            border-radius: 5px;
            background-color: #f5f5f5;
            color: #666666;
            font-size: 16px;
        }
    """
    
    INFO_LABEL = "QLabel { font-family: 'Microsoft JhengHei'; }"
    
    STATS_LABEL = "QLabel { font-family: 'Microsoft JhengHei'; color: #0066cc; }"