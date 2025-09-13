#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DISH 染色細胞遮罩工具 - GUI 介面
DISH Cell Masking Tool - GUI Interface
"""

import sys
import os
import numpy as np
import cv2 as cv
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QSlider, QPushButton,
                            QGroupBox, QTextEdit, QMessageBox, QProgressBar,
                            QFileDialog, QSplitter, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont

# 導入核心處理邏輯
from dish_mask_core import DishMaskProcessor, MaskingParams, ProcessingResult

class ProcessingThread(QThread):
    """背景處理執行緒"""
    result_ready = pyqtSignal(object)  # ProcessingResult
    error_occurred = pyqtSignal(str)   # 錯誤訊息
    
    def __init__(self, processor: DishMaskProcessor, params: MaskingParams):
        super().__init__()
        self.processor = processor
        self.params = params
        
    def run(self):
        try:
            result = self.processor.process_mask(self.params, use_original=False)
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

class DishMaskGUI(QMainWindow):
    """DISH 染色細胞遮罩 GUI 介面"""
    
    def __init__(self):
        super().__init__()
        self.processor = DishMaskProcessor()
        self.current_result = None
        self.processing_thread = None
        
        # 防抖動計時器
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.request_processing)
        self.debounce_delay = 1  # 1ms 防抖動
        
        # UI 元件
        self.sliders = {}
        
        self.initUI()
        self.auto_load_dish_image()
        
    def initUI(self):
        """初始化使用者介面"""
        self.setWindowTitle('DISH 染色細胞遮罩工具 (架構分離版)')
        self.setGeometry(100, 50, 1600, 1000)
        
        # 主要 widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主要佈局 - 分割器
        main_splitter = QSplitter(Qt.Horizontal)
        
        # 左側控制面板
        control_panel = self.create_control_panel()
        control_panel.setMaximumWidth(350)
        
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
        
        # HSV 閾值控制組
        hsv_group = QGroupBox("HSV 顏色閾值")
        hsv_layout = QGridLayout()
        
        slider_configs = [
            ('h_min', 'H最小', 0, 179, 0),
            ('h_max', 'H最大', 0, 179, 179),
            ('s_min', 'S最小', 0, 255, 0),
            ('s_max', 'S最大', 0, 255, 255),
            ('v_min', 'V最小', 0, 255, 0),
            ('v_max', 'V最大', 0, 255, 255),
        ]
        
        for i, (key, label, min_val, max_val, default) in enumerate(slider_configs):
            label_widget = QLabel(f"{label}:")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default)
            slider.setTracking(False)  # 防抖動
            
            value_label = QLabel(str(default))
            value_label.setMinimumWidth(40)
            value_label.setAlignment(Qt.AlignCenter)
            
            slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(str(v)))
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
        
        morph_configs = [
            ('kernel_size', 'Kernel 大小', 1, 15, 3),
            ('open_iter', '開運算次數', 0, 10, 1),
            ('close_iter', '閉運算次數', 0, 10, 1),
            ('dilate_iter', '膨脹次數', 1, 10, 2),
        ]
        
        for i, (key, label, min_val, max_val, default) in enumerate(morph_configs):
            label_widget = QLabel(f"{label}:")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default)
            slider.setTracking(False)
            
            value_label = QLabel(str(default))
            value_label.setMinimumWidth(40)
            value_label.setAlignment(Qt.AlignCenter)
            
            # 為 kernel_size 建立特殊處理
            if key == 'kernel_size':
                def update_kernel_value(value, label=value_label, s=slider):
                    # 確保為奇數，但不直接修改滑桿值
                    actual_value = value if value % 2 == 1 else value + 1
                    label.setText(str(actual_value))
                
                slider.valueChanged.connect(update_kernel_value)
            else:
                slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(str(v)))
            
            slider.valueChanged.connect(self.schedule_update)
            
            self.sliders[key] = (slider, value_label)
            
            morph_layout.addWidget(label_widget, i, 0)
            morph_layout.addWidget(slider, i, 1)
            morph_layout.addWidget(value_label, i, 2)
            
        morph_group.setLayout(morph_layout)
        layout.addWidget(morph_group)
        
        # Watershed 控制組
        watershed_group = QGroupBox("Watershed 分離")
        watershed_layout = QGridLayout()
        
        dist_label = QLabel("距離閾值:")
        self.dist_slider = QSlider(Qt.Horizontal)
        self.dist_slider.setRange(10, 80)  # 0.1 到 0.8
        self.dist_slider.setValue(35)      # 預設 0.35
        self.dist_slider.setTracking(False)
        self.dist_value_label = QLabel("0.35")
        
        self.dist_slider.valueChanged.connect(
            lambda v: self.dist_value_label.setText(f"{v/100:.2f}")
        )
        self.dist_slider.valueChanged.connect(self.schedule_update)
        
        watershed_layout.addWidget(dist_label, 0, 0)
        watershed_layout.addWidget(self.dist_slider, 0, 1)
        watershed_layout.addWidget(self.dist_value_label, 0, 2)
        
        watershed_group.setLayout(watershed_layout)
        layout.addWidget(watershed_group)
        
        # 顯示控制組
        display_group = QGroupBox("顯示設定")
        display_layout = QGridLayout()
        
        alpha_label = QLabel("遮罩透明度:")
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_value_label = QLabel("50%")
        
        self.alpha_slider.valueChanged.connect(
            lambda v: self.alpha_value_label.setText(f"{v}%")
        )
        self.alpha_slider.valueChanged.connect(self.update_display_only)
        
        # 遮罩模式選擇
        mask_mode_label = QLabel("遮罩模式:")
        self.invert_mask_checkbox = QCheckBox("反轉遮罩 (保留細胞核)")
        self.invert_mask_checkbox.stateChanged.connect(self.schedule_update)
        
        display_layout.addWidget(alpha_label, 0, 0)
        display_layout.addWidget(self.alpha_slider, 0, 1)
        display_layout.addWidget(self.alpha_value_label, 0, 2)
        display_layout.addWidget(mask_mode_label, 1, 0)
        display_layout.addWidget(self.invert_mask_checkbox, 1, 1, 1, 2)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # 統計資訊
        stats_group = QGroupBox("統計資訊")
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setReadOnly(True)
        self.stats_text.setPlainText("尚未載入影像...")
        
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
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
    
    def get_current_params(self) -> MaskingParams:
        """取得目前的參數設定"""
        params = MaskingParams()
        
        # HSV 參數
        if 'h_min' in self.sliders:
            params.h_min = self.sliders['h_min'][0].value()
            params.h_max = self.sliders['h_max'][0].value()
            params.s_min = self.sliders['s_min'][0].value()
            params.s_max = self.sliders['s_max'][0].value()
            params.v_min = self.sliders['v_min'][0].value()
            params.v_max = self.sliders['v_max'][0].value()
        
        # 形態學參數
        if 'kernel_size' in self.sliders:
            kernel_value = self.sliders['kernel_size'][0].value()
            # 確保 kernel_size 為奇數
            params.kernel_size = kernel_value if kernel_value % 2 == 1 else kernel_value + 1
            params.open_iter = self.sliders['open_iter'][0].value()
            params.close_iter = self.sliders['close_iter'][0].value()
            params.dilate_iter = self.sliders['dilate_iter'][0].value()
        
        # Watershed 參數
        params.dist_threshold = self.dist_slider.value() / 100.0
        
        # 顯示參數
        params.alpha = self.alpha_slider.value()
        
        # 遮罩模式
        params.invert_mask = self.invert_mask_checkbox.isChecked()
        
        return params
    
    def schedule_update(self):
        """排程更新 (防抖動)"""
        self.update_timer.stop()
        self.update_timer.start(self.debounce_delay)
        
    def request_processing(self):
        """請求背景處理"""
        if self.processor.working_image is None:
            print("GUI: 沒有載入工作影像，跳過處理")
            return
            
        if self.processing_thread and self.processing_thread.isRunning():
            print("GUI: 已有處理執行中，跳過")
            return  # 已有處理在進行中
            
        params = self.get_current_params()
        print(f"GUI: 開始背景處理，參數: H({params.h_min}-{params.h_max})")
        
        # 建立背景處理執行緒
        self.processing_thread = ProcessingThread(self.processor, params)
        self.processing_thread.result_ready.connect(self.on_processing_complete)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        
        self.statusBar().showMessage("處理中...")
        self.processing_thread.start()
    
    @pyqtSlot(object)
    def on_processing_complete(self, result: ProcessingResult):
        """處理完成回調"""
        try:
            self.current_result = result
            self.update_display()
            self.update_statistics()
            self.save_btn.setEnabled(True)
            self.statusBar().showMessage(f"處理完成 ({result.processing_time:.2f}s)")
        except Exception as e:
            self.statusBar().showMessage(f"顯示錯誤: {e}")
    
    @pyqtSlot(str)
    def on_processing_error(self, error_message: str):
        """處理錯誤回調"""
        self.statusBar().showMessage(f"處理錯誤: {error_message}")
        QMessageBox.critical(self, "處理錯誤", error_message)
    
    def update_display_only(self):
        """僅更新顯示 (透明度變更)"""
        if self.current_result is None:
            return
            
        # 重新產生疊圖
        alpha = self.alpha_slider.value()
        overlay = self.processor._create_overlay(
            self.processor.working_image, 
            self.current_result.cells, 
            alpha
        )
        
        self.display_image(overlay)
    
    def update_display(self):
        """更新圖像顯示"""
        if self.current_result is None:
            return
            
        self.display_image(self.current_result.overlay)
    
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
    
    def update_statistics(self):
        """更新統計資訊"""
        if self.current_result is None:
            return
            
        try:
            result = self.current_result
            info = self.processor.get_image_info()
            
            stats_text = f"""
影像資訊:
• 檔案: {info.get('filename', 'N/A')}
• 原始尺寸: {info.get('original_size', (0, 0))[0]}×{info.get('original_size', (0, 0))[1]}
• 工作尺寸: {info.get('working_size', (0, 0))[0]}×{info.get('working_size', (0, 0))[1]}
• 縮放比例: {info.get('working_scale', 0):.1%}

統計結果:
• 總像素: {result.total_pixels:,}
• 遮罩像素: {result.mask_pixels:,} ({result.mask_area_percent:.1f}%)
• 細胞像素: {result.cell_pixels:,} ({result.cell_area_percent:.1f}%)
• 細胞數量: {result.cell_count}

處理時間: {result.processing_time:.3f} 秒
"""
            self.stats_text.setPlainText(stats_text.strip())
            
        except Exception as e:
            self.stats_text.setPlainText(f"統計錯誤: {e}")
    
    def auto_load_dish_image(self):
        """自動載入 DISH 影像"""
        try:
            if self.processor.load_dish_from_directory("picture/tiff"):
                info = self.processor.get_image_info()
                self.statusBar().showMessage(f"已載入: {info['filename']}")
                
                if self.processor.working_image is not None:
                    self.display_image(self.processor.working_image)
                
                self.image_label.setText("")
                self.schedule_update()
            else:
                QMessageBox.warning(self, "警告", "無法自動載入 DISH 影像\n請使用「載入影像」按鈕手動選擇")
        except Exception as e:
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
            self, "選擇輸出目錄", "testing/output/middle-gen"
        )
        
        if output_dir:
            self.statusBar().showMessage("儲存中...")
            if self.processor.save_results(self.current_result, output_dir):
                QMessageBox.information(self, "儲存成功", f"結果已儲存至:\n{output_dir}")
                self.statusBar().showMessage("儲存完成")
            else:
                QMessageBox.critical(self, "儲存失敗", "無法儲存結果")
                self.statusBar().showMessage("儲存失敗")
    
    def reset_parameters(self):
        """重置參數"""
        # HSV 預設值
        defaults = {
            'h_min': 0, 'h_max': 179,
            's_min': 0, 's_max': 255,
            'v_min': 0, 'v_max': 255,
            'kernel_size': 3,
            'open_iter': 1,
            'close_iter': 1,
            'dilate_iter': 2
        }
        
        for key, default_val in defaults.items():
            if key in self.sliders:
                slider, value_label = self.sliders[key]
                slider.setValue(default_val)
                value_label.setText(str(default_val))
        
        # 其他滑桿
        self.dist_slider.setValue(35)
        self.dist_value_label.setText("0.35")
        self.alpha_slider.setValue(50)
        self.alpha_value_label.setText("50%")
        
        # 遮罩模式
        self.invert_mask_checkbox.setChecked(False)
        
        self.schedule_update()
        self.statusBar().showMessage("參數已重置")

def main():
    """主程式進入點"""
    app = QApplication(sys.argv)
    app.setApplicationName('DISH 染色細胞遮罩工具')
    app.setStyle('Fusion')
    
    # 建立並顯示主視窗
    window = DishMaskGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()