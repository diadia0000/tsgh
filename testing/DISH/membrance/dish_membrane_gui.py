#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DISH 染色細胞膜遮罩工具 - GUI 介面
DISH Cell Membrane Masking Tool - GUI Interface
"""

import sys
import os
import numpy as np
import cv2 as cv
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QSlider, QPushButton,
                            QGroupBox, QTextEdit, QMessageBox, QProgressBar,
                            QFileDialog, QSplitter, QCheckBox, QTabWidget)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont

# 導入核心處理邏輯
from dish_membrane_core import DishMembraneProcessor, DishMembraneParams, DishMembraneResult

class MembraneProcessingThread(QThread):
    """背景處理執行緒"""
    result_ready = pyqtSignal(object)  # DishMembraneResult
    error_occurred = pyqtSignal(str)   # 錯誤訊息
    
    def __init__(self, processor: DishMembraneProcessor, params: DishMembraneParams):
        super().__init__()
        self.processor = processor
        self.params = params
        
    def run(self):
        try:
            print("DEBUG: 背景處理執行緒開始")
            result = self.processor.process_membrane(self.params, use_original=False)
            print("DEBUG: 背景處理完成，發送結果")
            self.result_ready.emit(result)
        except Exception as e:
            print(f"DEBUG: 背景處理發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))

class DishMembraneGUI(QMainWindow):
    """DISH 染色細胞膜遮罩 GUI 介面"""
    
    def __init__(self):
        super().__init__()
        self.processor = DishMembraneProcessor()
        self.current_result = None
        self.processing_thread = None
        
        # 防抖動計時器
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.request_processing)
        self.debounce_delay = 200  # 200ms 防抖動
        
        # UI 元件
        self.sliders = {}
        self.pending_reprocess = False  # 用於追蹤是否需要重新處理
        
        self.initUI()
        self.auto_load_dish_image()
        
    def initUI(self):
        """初始化使用者介面"""
        self.setWindowTitle('DISH 染色細胞膜遮罩工具')
        self.setGeometry(100, 50, 1600, 1000)
        
        # 主要 widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主要佈局 - 分割器
        main_splitter = QSplitter(Qt.Horizontal)
        
        # 左側控制面板
        control_panel = self.create_control_panel()
        control_panel.setMaximumWidth(380)
        
        # 右側圖像顯示
        image_panel = self.create_image_panel()
        
        main_splitter.addWidget(control_panel)
        main_splitter.addWidget(image_panel)
        main_splitter.setStretchFactor(0, 0)  # 固定寬度
        main_splitter.setStretchFactor(1, 1)  # 可擴展
        
        # 主佈局
        main_layout = QHBoxLayout()
        main_layout.addWidget(main_splitter)
        central_widget.setLayout(main_layout)
        
        # 狀態列
        self.statusBar().showMessage('準備就緒')
        
    def create_control_panel(self):
        """建立左側控制面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 檔案操作組
        file_group = QGroupBox("檔案操作")
        file_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("載入影像...")
        self.load_btn.clicked.connect(self.load_custom_image)
        
        self.reload_btn = QPushButton("重新載入 DISH")
        self.reload_btn.clicked.connect(self.auto_load_dish_image)
        
        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(self.reload_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # LAB 顏色閾值控制組
        lab_group = QGroupBox("LAB 顏色閾值 (主要)")
        lab_layout = QGridLayout()
        
        lab_configs = [
            ('lab_a_lower', 'A通道最小', 0, 255, 100),   # 稍微降低A通道下限
            ('lab_a_upper', 'A通道最大', 0, 255, 150),   # 擴大A通道上限
            ('lab_b_lower', 'B通道最小', 0, 255, 90),    # 擴大B通道範圍
            ('lab_b_upper', 'B通道最大', 0, 255, 160),   # 擴大B通道範圍
        ]
        
        for i, (key, label, min_val, max_val, default) in enumerate(lab_configs):
            label_widget = QLabel(f"{label}:")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default)
            slider.setTracking(True)  # 改為即時跟蹤
            
            value_label = QLabel(str(default))
            value_label.setMinimumWidth(40)
            value_label.setAlignment(Qt.AlignCenter)
            
            # 修復閉包變數綁定問題
            def make_value_updater(label):
                def update_value(value):
                    print(f"DEBUG: 滑桿值變更: {value}")
                    label.setText(str(value))
                return update_value
            
            slider.valueChanged.connect(make_value_updater(value_label))
            slider.valueChanged.connect(self.schedule_update)
            
            self.sliders[key] = (slider, value_label)
            
            lab_layout.addWidget(label_widget, i, 0)
            lab_layout.addWidget(slider, i, 1)
            lab_layout.addWidget(value_label, i, 2)
            
        lab_group.setLayout(lab_layout)
        layout.addWidget(lab_group)
        
        # DAB 通道控制組
        dab_group = QGroupBox("DAB 染色強度")
        dab_layout = QGridLayout()
        
        dab_configs = [
            ('dab_lower', 'DAB 最小值', 0, 255, 10),
            ('dab_upper', 'DAB 最大值', 0, 255, 255),
        ]
        
        for i, (key, label, min_val, max_val, default) in enumerate(dab_configs):
            label_widget = QLabel(f"{label}:")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default)
            slider.setTracking(True)  # 改為即時跟蹤
            
            value_label = QLabel(str(default))
            value_label.setMinimumWidth(40)
            value_label.setAlignment(Qt.AlignCenter)
            
            # 修復閉包變數綁定問題
            def make_value_updater(label):
                def update_value(value):
                    print(f"DEBUG: 滑桿值變更: {value}")
                    label.setText(str(value))
                return update_value
            
            slider.valueChanged.connect(make_value_updater(value_label))
            slider.valueChanged.connect(self.schedule_update)
            
            self.sliders[key] = (slider, value_label)
            
            dab_layout.addWidget(label_widget, i, 0)
            dab_layout.addWidget(slider, i, 1)
            dab_layout.addWidget(value_label, i, 2)
            
        dab_group.setLayout(dab_layout)
        layout.addWidget(dab_group)
        
        # HSV 輔助閾值控制組 (摺疊)
        hsv_group = QGroupBox("HSV 輔助閾值 (可選)")
        hsv_layout = QGridLayout()
        
        hsv_configs = [
            ('h_min', 'H最小', 0, 255, 0),
            ('h_max', 'H最大', 0, 255, 179),
            ('s_min', 'S最小', 0, 255, 0),
            ('s_max', 'S最大', 0, 255, 255),
            ('v_min', 'V最小', 0, 255, 0),
            ('v_max', 'V最大', 0, 255, 255),
        ]
        
        for i, (key, label, min_val, max_val, default) in enumerate(hsv_configs):
            label_widget = QLabel(f"{label}:")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default)
            slider.setTracking(True)  # 改為即時跟蹤
            
            value_label = QLabel(str(default))
            value_label.setMinimumWidth(40)
            value_label.setAlignment(Qt.AlignCenter)
            
            # 修復閉包變數綁定問題
            def make_value_updater(label):
                def update_value(value):
                    print(f"DEBUG: 滑桿值變更: {value}")
                    label.setText(str(value))
                return update_value
            
            slider.valueChanged.connect(make_value_updater(value_label))
            slider.valueChanged.connect(self.schedule_update)
            
            self.sliders[key] = (slider, value_label)
            
            hsv_layout.addWidget(label_widget, i, 0)
            hsv_layout.addWidget(slider, i, 1)
            hsv_layout.addWidget(value_label, i, 2)
            
        hsv_group.setLayout(hsv_layout)
        layout.addWidget(hsv_group)
        
        # 形態學處理控制組
        morph_group = QGroupBox("形態學處理")
        morph_layout = QGridLayout()
        
        # 定義形態學參數配置
        morph_list = [
            ('membrane_kernel_size', 'Kernel 大小', 1, 15, 3),
            ('membrane_open_iter', '開運算次數', 0, 5, 1),
            ('membrane_close_iter', '閉運算次數', 0, 5, 1),
            ('membrane_dilate_iter', '膨脹次數', 0, 5, 1),
            ('min_membrane_area', '最小面積', 10, 500, 50),
            ('membrane_approx_radius', '膜擬合膨脹半徑', 0, 50, 5),
        ]
        
        for i, config in enumerate(morph_list):
            key, label, min_val, max_val, default = config
            
            label_widget = QLabel(f"{label}:")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default)
            slider.setTracking(True)  # 改為即時跟蹤
            
            value_label = QLabel(str(default))
            value_label.setMinimumWidth(40)
            value_label.setAlignment(Qt.AlignCenter)
            
            # 為 kernel_size 建立特殊處理
            if 'kernel_size' in key:
                def make_kernel_updater(label):
                    def update_kernel_value(value):
                        actual_value = value if value % 2 == 1 else value + 1
                        print(f"DEBUG: Kernel滑桿值變更: {value} -> {actual_value}")
                        label.setText(str(actual_value))
                    return update_kernel_value
                
                slider.valueChanged.connect(make_kernel_updater(value_label))
            else:
                def make_updater(label):
                    def update_value(value):
                        print(f"DEBUG: 形態學滑桿值變更: {value}")
                        label.setText(str(value))
                    return update_value
                
                slider.valueChanged.connect(make_updater(value_label))
            
            slider.valueChanged.connect(self.schedule_update)
            
            self.sliders[key] = (slider, value_label)
            
            morph_layout.addWidget(label_widget, i, 0)
            morph_layout.addWidget(slider, i, 1)
            morph_layout.addWidget(value_label, i, 2)
            
        morph_group.setLayout(morph_layout)
        layout.addWidget(morph_group)
        
        # 顯示控制組
        display_group = QGroupBox("顯示設定")
        display_layout = QGridLayout()
        
        alpha_label = QLabel("遮罩透明度:")
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.setTracking(True)  # 啟用即時跟蹤
        self.alpha_value_label = QLabel("50%")
        
        self.alpha_slider.valueChanged.connect(
            lambda v: (print(f"DEBUG: 透明度滑桿值變更: {v}%"), self.alpha_value_label.setText(f"{v}%"))[-1]
        )
        self.alpha_slider.valueChanged.connect(self.update_display_only)
        
        display_layout.addWidget(alpha_label, 0, 0)
        display_layout.addWidget(self.alpha_slider, 0, 1)
        display_layout.addWidget(self.alpha_value_label, 0, 2)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # 統計資訊顯示
        info_group = QGroupBox("影像資訊")
        info_layout = QVBoxLayout()
        
        self.info_label = QLabel("等待載入影像...")
        self.info_label.setStyleSheet("QLabel { font-family: 'Microsoft JhengHei'; }")
        info_layout.addWidget(self.info_label)
        
        # 即時統計資訊
        self.stats_label = QLabel("處理統計：等待處理...")
        self.stats_label.setStyleSheet("QLabel { font-family: 'Microsoft JhengHei'; color: #0066cc; }")
        info_layout.addWidget(self.stats_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # 操作按鈕
        button_layout = QVBoxLayout()
        
        self.save_btn = QPushButton("儲存所有結果")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("""
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
        """)
        
        self.reset_btn = QPushButton("重置參數")
        self.reset_btn.clicked.connect(self.reset_parameters)
        
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.reset_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
        
    def create_image_panel(self):
        """建立右側圖像顯示面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 圖像顯示標籤
        self.image_label = QLabel("尚未載入影像...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                border-radius: 5px;
                background-color: #f5f5f5;
                color: #666666;
                font-size: 16px;
            }
        """)
        
        layout.addWidget(self.image_label)
        panel.setLayout(layout)
        
        return panel
    
    def get_current_params(self) -> DishMembraneParams:
        """取得目前的參數設定"""
        params = DishMembraneParams()
        
        # LAB 參數
        params.lab_a_lower = self.sliders['lab_a_lower'][0].value()
        params.lab_a_upper = self.sliders['lab_a_upper'][0].value()
        params.lab_b_lower = self.sliders['lab_b_lower'][0].value()
        params.lab_b_upper = self.sliders['lab_b_upper'][0].value()
        
        # DAB 參數
        params.dab_lower = self.sliders['dab_lower'][0].value()
        params.dab_upper = self.sliders['dab_upper'][0].value()
        
        # HSV 參數
        params.h_min = self.sliders['h_min'][0].value()
        params.h_max = self.sliders['h_max'][0].value()
        params.s_min = self.sliders['s_min'][0].value()
        params.s_max = self.sliders['s_max'][0].value()
        params.v_min = self.sliders['v_min'][0].value()
        params.v_max = self.sliders['v_max'][0].value()
        
        # 形態學參數
        kernel_value = self.sliders['membrane_kernel_size'][0].value()
        params.membrane_kernel_size = kernel_value if kernel_value % 2 == 1 else kernel_value + 1
        params.membrane_open_iter = self.sliders['membrane_open_iter'][0].value()
        params.membrane_close_iter = self.sliders['membrane_close_iter'][0].value()
        params.membrane_dilate_iter = self.sliders['membrane_dilate_iter'][0].value()
        params.min_membrane_area = self.sliders['min_membrane_area'][0].value()
        
        # 顯示參數
        params.alpha = self.alpha_slider.value()
        
        print(f"DEBUG: 讀取參數 - LAB A: {params.lab_a_lower}-{params.lab_a_upper}, LAB B: {params.lab_b_lower}-{params.lab_b_upper}")
        print(f"DEBUG: 讀取參數 - DAB: {params.dab_lower}-{params.dab_upper}, HSV: H{params.h_min}-{params.h_max}")
        
        return params
    
    def schedule_update(self):
        """排程更新 (防抖動)"""
        # 如果沒有載入影像，不需要排程處理
        if self.processor.working_image is None:
            print("DEBUG: 沒有載入影像，跳過更新排程")
            return
            
        print("DEBUG: 排程更新，停止現有計時器並重新開始")
        self.update_timer.stop()
        self.update_timer.start(self.debounce_delay)
        
    def request_processing(self):
        """請求背景處理"""
        print("DEBUG: request_processing 被調用")
        
        if self.processor.working_image is None:
            print("GUI: 沒有載入工作影像，跳過處理")
            return
            
        # 如果有線程在運行，標記需要重新處理，但不等待
        if self.processing_thread and self.processing_thread.isRunning():
            print("GUI: 前一個處理還在進行中，將在完成後重新處理")
            self.pending_reprocess = True
            return
            
        params = self.get_current_params()
        print(f"GUI: 開始背景處理，LAB A: {params.lab_a_lower}-{params.lab_a_upper}")
        print(f"DEBUG: 參數詳情 - DAB: {params.dab_lower}-{params.dab_upper}, Kernel: {params.membrane_kernel_size}")
        
        # 建立背景處理執行緒
        self.processing_thread = MembraneProcessingThread(self.processor, params)
        self.processing_thread.result_ready.connect(self.on_processing_complete)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        
        self.statusBar().showMessage("處理中...")
        self.processing_thread.start()
    
    @pyqtSlot(object)
    def on_processing_complete(self, result: DishMembraneResult):
        """處理完成回調"""
        try:
            print("DEBUG: 收到處理結果，開始更新顯示")
            self.current_result = result
            self.update_display(result)
            self.save_btn.setEnabled(True)
            self.statusBar().showMessage(f"處理完成 ({result.processing_time:.2f}s)")
            print("DEBUG: 顯示更新完成")
            
            # 檢查是否有待處理的重新處理請求
            if self.pending_reprocess:
                print("DEBUG: 檢測到待處理的重新處理請求")
                self.pending_reprocess = False
                QTimer.singleShot(50, self.request_processing)  # 短暫延遲後重新處理
                
        except Exception as e:
            print(f"DEBUG: 顯示更新錯誤: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"顯示錯誤: {e}")
    
    @pyqtSlot(str)
    def on_processing_error(self, error_message: str):
        """處理錯誤回調"""
        print(f"DEBUG: 收到處理錯誤: {error_message}")
        self.statusBar().showMessage(f"處理錯誤: {error_message}")
        QMessageBox.critical(self, "處理錯誤", error_message)
    
    def update_display_only(self):
        """僅更新顯示 (透明度變更)"""
        print("DEBUG: update_display_only 被調用")
        if self.current_result is None:
            print("DEBUG: current_result 為 None，跳過透明度更新")
            return
            
        # 重新產生疊圖
        alpha = self.alpha_slider.value()
        print(f"DEBUG: 透明度更新，alpha = {alpha}")
        overlay = self.processor._create_overlay(
            self.processor.working_image, 
            self.current_result.mask_membrane_clean, 
            alpha
        )
        
        print("DEBUG: 透明度疊圖產生完成，開始顯示")
        self.display_image(overlay)
    
    def update_display(self, result):
        """更新顯示結果"""
        try:
            if result and hasattr(result, 'overlay_membrane'):
                # 顯示疊圖結果
                self.display_image(result.overlay_membrane)
                
                # 更新即時統計資訊
                self.update_stats_display(result)
                
        except Exception as e:
            QMessageBox.warning(self, "錯誤", f"更新顯示失敗: {str(e)}")

    def update_stats_display(self, result):
        """更新統計資訊顯示"""
        try:
            # 更新影像資訊
            info = self.processor.get_image_info()
            info_text = f"""影像資訊:
• 檔案: {info.get('filename', 'N/A')}
• 原始尺寸: {info.get('original_size', (0, 0))[0]}×{info.get('original_size', (0, 0))[1]}
• 工作尺寸: {info.get('working_size', (0, 0))[0]}×{info.get('working_size', (0, 0))[1]}
• 縮放比例: {info.get('working_scale', 0):.1%}"""
            
            self.info_label.setText(info_text)
            
            # 更新統計資訊
            stats_text = f"""處理統計:
• 細胞膜覆蓋率: {result.membrane_coverage_percent:.1f}%
• DAB 平均強度: {result.dab_mean:.1f}
• DAB 中位數: {result.dab_median:.1f}
• 處理時間: {result.processing_time:.2f}秒"""
            
            self.stats_label.setText(stats_text)
            
        except Exception as e:
            self.stats_label.setText(f"統計資訊更新失敗: {str(e)}")
    
    def display_image(self, img):
        """顯示圖像"""
        try:
            if img is None:
                return
                
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
                return
                
            pixmap = QPixmap.fromImage(q_image)
            
            if pixmap.isNull():
                return
            
            # 縮放適應標籤
            label_size = self.image_label.size()
            if (pixmap.width() > label_size.width() - 20 or 
                pixmap.height() > label_size.height() - 20):
                pixmap = pixmap.scaled(
                    label_size.width() - 20,
                    label_size.height() - 20,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            
            self.image_label.setPixmap(pixmap)
            
            # 強制更新顯示
            self.image_label.update()
            self.update()
            QApplication.processEvents()
            
        except Exception as e:
            print(f"顯示影像錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    def auto_load_dish_image(self):
        """自動載入 DISH 影像"""
        try:
            print("DEBUG: 嘗試自動載入 DISH 影像")
            if self.processor.load_dish_from_directory("picture/tiff"):
                info = self.processor.get_image_info()
                print(f"DEBUG: 成功載入影像: {info}")
                self.statusBar().showMessage(f"已載入: {info['filename']}")
                
                if self.processor.working_image is not None:
                    print(f"DEBUG: working_image 尺寸: {self.processor.working_image.shape}")
                    self.display_image(self.processor.working_image)
                else:
                    print("DEBUG: working_image 為 None")
                
                self.image_label.setText("")
                self.schedule_update()
            else:
                print("DEBUG: 自動載入失敗")
                QMessageBox.warning(self, "警告", "無法自動載入 DISH 影像\n請使用「載入影像」按鈕手動選擇")
        except Exception as e:
            print(f"DEBUG: 自動載入異常: {e}")
            QMessageBox.critical(self, "錯誤", f"載入影像時發生錯誤: {e}")
    
    def load_custom_image(self):
        """載入自訂影像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇影像檔案", "picture/tiff",
            "影像檔案 (*.tiff *.tif *.png *.jpg *.jpeg *.bmp);;所有檔案 (*)"
        )
        
        if file_path:
            try:
                if self.processor.load_image(file_path):
                    info = self.processor.get_image_info()
                    self.statusBar().showMessage(f"已載入: {info['filename']}")
                    
                    if self.processor.working_image is not None:
                        self.display_image(self.processor.working_image)
                    
                    self.image_label.setText("")
                    self.schedule_update()
                else:
                    QMessageBox.critical(self, "錯誤", "載入影像失敗")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"載入影像失敗: {e}")
    
    def save_results(self):
        """儲存處理結果"""
        if self.current_result is None:
            QMessageBox.warning(self, "警告", "沒有可儲存的結果")
            return
        
        # 選擇輸出目錄
        output_dir = QFileDialog.getExistingDirectory(
            self, "選擇輸出目錄", "testing/DISH/membrance/output"
        )
        
        if output_dir:
            self.statusBar().showMessage("儲存中...")
            if self.processor.save_results(self.current_result, output_dir):
                QMessageBox.information(self, "儲存成功", f"結果已儲存至:\n{output_dir}")
                self.statusBar().showMessage("儲存完成")
            else:
                QMessageBox.critical(self, "儲存失敗", "無法儲存結果")
                self.statusBar().showMessage("儲存失敗")
    
    def schedule_update(self):
        """排程更新 - 防抖動"""
        print("DEBUG: 排程更新，停止現有計時器並重新開始")
        self.update_timer.stop()
        self.update_timer.start(self.debounce_delay)
    
    def request_processing(self):
        """請求處理"""
        print("DEBUG: request_processing 被調用")
        
        if self.processor.working_image is None:
            print("GUI: 沒有載入工作影像，跳過處理")
            return
            
        # 如果有線程在運行，標記需要重新處理，但不等待
        if self.processing_thread and self.processing_thread.isRunning():
            print("GUI: 前一個處理還在進行中，將在完成後重新處理")
            self.pending_reprocess = True
            return
            
        params = self.get_current_params()
        print(f"GUI: 開始背景處理，LAB A: {params.lab_a_lower}-{params.lab_a_upper}")
        print(f"DEBUG: 參數詳情 - DAB: {params.dab_lower}-{params.dab_upper}, Kernel: {params.membrane_kernel_size}")
        
        # 建立背景處理執行緒
        self.processing_thread = MembraneProcessingThread(self.processor, params)
        self.processing_thread.result_ready.connect(self.on_processing_complete)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        
        self.statusBar().showMessage("處理中...")
        self.processing_thread.start()
    
    @pyqtSlot(object)
    def on_processing_complete(self, result: DishMembraneResult):
        """處理完成回調"""
        try:
            print("DEBUG: 收到處理結果，開始更新顯示")
            self.current_result = result
            self.update_display(result)
            self.save_btn.setEnabled(True)
            self.statusBar().showMessage(f"處理完成 ({result.processing_time:.2f}s)")
            print("DEBUG: 顯示更新完成")
            
            # 檢查是否有待處理的重新處理請求
            if self.pending_reprocess:
                print("DEBUG: 檢測到待處理的重新處理請求")
                self.pending_reprocess = False
                QTimer.singleShot(50, self.request_processing)  # 短暫延遲後重新處理
                
        except Exception as e:
            print(f"DEBUG: 顯示更新錯誤: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"顯示錯誤: {e}")
    
    @pyqtSlot(str)
    def on_processing_error(self, error_message: str):
        """處理錯誤回調"""
        print(f"DEBUG: 收到處理錯誤: {error_message}")
        self.statusBar().showMessage(f"處理錯誤: {error_message}")
    
    def update_display(self, result: DishMembraneResult):
        """更新圖像顯示和統計資訊"""
        try:
            # 顯示疊加圖
            self.display_image(result.overlay_membrane)
            
            # 更新統計資訊
            self.update_statistics(result)
            
        except Exception as e:
            print(f"DEBUG: 更新顯示錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    def update_statistics(self, result: DishMembraneResult):
        """更新統計資訊顯示"""
        try:
            # 影像資訊
            info = self.processor.get_image_info()
            info_text = f"""影像檔案: {info.get('filename', 'N/A')}
原始尺寸: {info.get('original_size', (0,0))[0]} x {info.get('original_size', (0,0))[1]}
工作尺寸: {info.get('working_size', (0,0))[0]} x {info.get('working_size', (0,0))[1]}
縮放比例: {info.get('working_scale', 0):.3f}"""
            
            self.info_label.setText(info_text)
            
            # 處理統計
            stats_text = f"""處理結果統計:
• 總像素數: {result.total_pixels:,}
• 膜區域像素: {result.membrane_pixels:,} 
• 細胞膜覆蓋率: {result.membrane_coverage_percent:.1f}%
• DAB 平均強度: {result.dab_mean:.1f}
• DAB 中位數: {result.dab_median:.1f}
• 處理時間: {result.processing_time:.2f}秒"""
            
            self.stats_label.setText(stats_text)
            
        except Exception as e:
            self.stats_label.setText(f"統計資訊更新失敗: {str(e)}")
    
    def update_display_only(self):
        """僅更新顯示（透明度變更時）"""
        print("DEBUG: update_display_only 被調用")
        if self.current_result is None:
            return
            
        # 獲取當前透明度
        alpha = self.alpha_slider.value()
        print(f"DEBUG: 透明度更新，alpha = {alpha}")
        
        # 重新生成疊加圖
        try:
            overlay = self.processor._create_overlay(
                self.processor.working_image,
                self.current_result.mask_membrane_approx,
                alpha
            )
            self.display_image(overlay)
            print("DEBUG: 透明度疊圖產生完成，開始顯示")
        except Exception as e:
            print(f"DEBUG: 透明度更新錯誤: {e}")
    
    def get_current_params(self) -> DishMembraneParams:
        """從滑桿獲取當前參數"""
        try:
            print("DEBUG: 讀取參數 - LAB A: {}-{}, LAB B: {}-{}".format(
                self.sliders['lab_a_lower'][0].value(), 
                self.sliders['lab_a_upper'][0].value(),
                self.sliders['lab_b_lower'][0].value(), 
                self.sliders['lab_b_upper'][0].value()
            ))
            print("DEBUG: 讀取參數 - DAB: {}-{}, HSV: H{}-{}".format(
                self.sliders['dab_lower'][0].value(), 
                self.sliders['dab_upper'][0].value(),
                self.sliders['h_min'][0].value(), 
                self.sliders['h_max'][0].value()
            ))
            
            # 特殊處理kernel_size確保為奇數
            kernel_size = self.sliders['membrane_kernel_size'][0].value()
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            return DishMembraneParams(
                lab_a_lower=self.sliders['lab_a_lower'][0].value(),
                lab_a_upper=self.sliders['lab_a_upper'][0].value(),
                lab_b_lower=self.sliders['lab_b_lower'][0].value(),
                lab_b_upper=self.sliders['lab_b_upper'][0].value(),
                dab_lower=self.sliders['dab_lower'][0].value(),
                dab_upper=self.sliders['dab_upper'][0].value(),
                h_min=self.sliders['h_min'][0].value(),
                h_max=self.sliders['h_max'][0].value(),
                s_min=self.sliders['s_min'][0].value(),
                s_max=self.sliders['s_max'][0].value(),
                v_min=self.sliders['v_min'][0].value(),
                v_max=self.sliders['v_max'][0].value(),
                membrane_kernel_size=kernel_size,
                membrane_open_iter=self.sliders['membrane_open_iter'][0].value(),
                membrane_close_iter=self.sliders['membrane_close_iter'][0].value(),
                membrane_dilate_iter=self.sliders['membrane_dilate_iter'][0].value(),
                min_membrane_area=self.sliders['min_membrane_area'][0].value(),
                membrane_approx_radius=self.sliders['membrane_approx_radius'][0].value(),
                alpha=self.alpha_slider.value()
            )
        except Exception as e:
            print(f"DEBUG: 獲取參數時發生錯誤: {e}")
            return DishMembraneParams()  # 返回預設參數
    
    def reset_parameters(self):
        """重置參數"""
        # LAB 預設值
        lab_defaults = {
            'lab_a_lower': 115, 'lab_a_upper': 140,
            'lab_b_lower': 115, 'lab_b_upper': 145,
        }
        
        # DAB 預設值
        dab_defaults = {
            'dab_lower': 10, 'dab_upper': 255,
        }
        
        # HSV 預設值
        hsv_defaults = {
            'h_min': 0, 'h_max': 179,
            's_min': 0, 's_max': 255,
            'v_min': 0, 'v_max': 255,
        }
        
        # 形態學預設值
        morph_defaults = {
            'membrane_kernel_size': 3,
            'membrane_open_iter': 1,
            'membrane_close_iter': 1,
            'membrane_dilate_iter': 1,
            'min_membrane_area': 50,
            'membrane_approx_radius': 5,
        }
        
        all_defaults = {**lab_defaults, **dab_defaults, **hsv_defaults, **morph_defaults}
        
        for key, default_val in all_defaults.items():
            if key in self.sliders:
                slider, value_label = self.sliders[key]
                slider.setValue(default_val)
                value_label.setText(str(default_val))
        
        # 透明度
        self.alpha_slider.setValue(50)
        self.alpha_value_label.setText("50%")
        
        self.schedule_update()
        self.statusBar().showMessage("參數已重置")
    
    def auto_load_dish_image(self):
        """自動載入DISH影像"""
        try:
            print("DEBUG: 嘗試自動載入 DISH 影像")
            dish_path = "picture/tiff/P2525729F_DISH_region.tiff"
            
            if Path(dish_path).exists():
                if self.processor.load_image(dish_path):
                    info = self.processor.get_image_info()
                    print(f"DEBUG: 成功載入影像: {info}")
                    
                    # 更新顯示
                    if self.processor.working_image is not None:
                        print(f"DEBUG: working_image 尺寸: {self.processor.working_image.shape}")
                        self.display_image(self.processor.working_image)
                    
                    self.image_label.setText("")
                    self.schedule_update()
                    self.statusBar().showMessage(f"已自動載入: {info['filename']}")
                else:
                    print("DEBUG: 載入影像失敗")
            else:
                print(f"DEBUG: DISH 影像檔案不存在: {dish_path}")
                
        except Exception as e:
            print(f"DEBUG: 自動載入影像錯誤: {e}")

def main():
    """主程式進入點"""
    app = QApplication(sys.argv)
    app.setApplicationName('DISH 染色細胞膜遮罩工具')
    app.setStyle('Fusion')
    
    # 建立並顯示主視窗
    window = DishMembraneGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()