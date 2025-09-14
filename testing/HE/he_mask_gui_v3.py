#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HE 染色細胞遮罩工具 - 簡化 GUI 介面
HE Cell Masking Tool - Simplified GUI Interface
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import cv2 as cv
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QSlider, QPushButton,
                            QGroupBox, QMessageBox, QFileDialog, QSplitter, 
                            QComboBox, QTabWidget)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

# 導入核心處理邏輯和工具模組
from he_mask_core import HEMaskProcessor, HEMaskingParams, HEProcessingResult
from he_gui_utils import (SliderGroup, ImageDisplayHelper, GroupBoxFactory, 
                         ParameterManager, ValueFormatters, SliderConfigs, 
                         DEFAULT_VALUE_FORMATTERS, StyleSheets)

class HEProcessingThread(QThread):
    """背景處理執行緒"""
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, processor: HEMaskProcessor, params: HEMaskingParams):
        super().__init__()
        self.processor = processor
        self.params = params
        
    def run(self):
        try:
            result = self.processor.process_mask(self.params, use_original=False)
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

class HEMaskGUI(QMainWindow):
    """HE 染色細胞遮罩 GUI 介面"""
    
    def __init__(self):
        super().__init__()
        self.processor = HEMaskProcessor()
        self.current_result = None
        self.processing_thread = None
        self.current_display_mode = "overlay_cyto"
        
        # 參數管理器
        self.param_manager = ParameterManager()
        
        # 防抖動計時器
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.request_processing)
        self.debounce_delay = 1
        
        # 顯示元件
        self.alpha_slider = None
        self.alpha_value_label = None
        self.display_mode_combo = None
        self.image_label = None
        self.info_label = None
        self.stats_label = None
        self.save_btn = None
        
        self.initUI()
        self.auto_load_he_image()
        
    def initUI(self):
        """初始化使用者介面"""
        self.setWindowTitle('HE 染色細胞遮罩工具 (細胞質/細胞核分離)')
        self.setGeometry(100, 50, 1800, 1000)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主要佈局 - 分割器
        main_splitter = QSplitter(Qt.Horizontal)
        
        # 左側控制面板
        control_panel = self.create_control_panel()
        control_panel.setMaximumWidth(400)
        
        # 右側圖像顯示
        image_panel = self.create_image_panel()
        
        main_splitter.addWidget(control_panel)
        main_splitter.addWidget(image_panel)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        
        main_layout = QHBoxLayout()
        main_layout.addWidget(main_splitter)
        central_widget.setLayout(main_layout)
        
        self.statusBar().showMessage('準備就緒')
        
    def create_control_panel(self):
        """建立左側控制面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 檔案操作組
        layout.addWidget(self.create_file_group())
        
        # 參數分頁
        layout.addWidget(self.create_parameter_tabs())
        
        # 顯示控制組
        layout.addWidget(self.create_display_group())
        
        # 統計資訊
        layout.addWidget(self.create_info_group())
        
        # 操作按鈕
        layout.addLayout(self.create_button_layout())
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_file_group(self):
        """建立檔案操作組"""
        file_group = QGroupBox("檔案操作")
        file_layout = QVBoxLayout()
        
        load_btn = QPushButton("載入影像...")
        load_btn.clicked.connect(self.load_custom_image)
        
        reload_btn = QPushButton("重新載入 HE")
        reload_btn.clicked.connect(self.auto_load_he_image)
        
        file_layout.addWidget(load_btn)
        file_layout.addWidget(reload_btn)
        file_group.setLayout(file_layout)
        
        return file_group
    
    def create_parameter_tabs(self):
        """建立參數分頁"""
        tab_widget = QTabWidget()
        
        # 細胞質參數分頁
        cyto_tab = QWidget()
        cyto_layout = QVBoxLayout()
        
        # 細胞質 HSV 組
        cyto_hsv_group, cyto_hsv_sliders = GroupBoxFactory.create_slider_group(
            "細胞質 (Eosin) HSV 閾值", 
            SliderConfigs.CYTOPLASM_HSV, 
            self.schedule_update
        )
        self.param_manager.register_slider_group("cyto_hsv", cyto_hsv_sliders)
        cyto_layout.addWidget(cyto_hsv_group)
        
        # 細胞質形態學組
        cyto_morph_group, cyto_morph_sliders = GroupBoxFactory.create_slider_group(
            "細胞質形態學參數", 
            SliderConfigs.CYTOPLASM_MORPH, 
            self.schedule_update,
            DEFAULT_VALUE_FORMATTERS
        )
        self.param_manager.register_slider_group("cyto_morph", cyto_morph_sliders)
        cyto_layout.addWidget(cyto_morph_group)
        
        cyto_layout.addStretch()
        cyto_tab.setLayout(cyto_layout)
        tab_widget.addTab(cyto_tab, "細胞質 (Eosin)")
        
        # 細胞核參數分頁
        nuclei_tab = QWidget()
        nuclei_layout = QVBoxLayout()
        
        # 細胞核 HSV 組
        nuclei_hsv_group, nuclei_hsv_sliders = GroupBoxFactory.create_slider_group(
            "細胞核 (Hematoxylin) HSV 閾值", 
            SliderConfigs.NUCLEI_HSV, 
            self.schedule_update
        )
        self.param_manager.register_slider_group("nuclei_hsv", nuclei_hsv_sliders)
        nuclei_layout.addWidget(nuclei_hsv_group)
        
        # 細胞核形態學組
        nuclei_morph_group, nuclei_morph_sliders = GroupBoxFactory.create_slider_group(
            "細胞核形態學參數", 
            SliderConfigs.NUCLEI_MORPH, 
            self.schedule_update,
            DEFAULT_VALUE_FORMATTERS
        )
        self.param_manager.register_slider_group("nuclei_morph", nuclei_morph_sliders)
        nuclei_layout.addWidget(nuclei_morph_group)
        
        nuclei_layout.addStretch()
        nuclei_tab.setLayout(nuclei_layout)
        tab_widget.addTab(nuclei_tab, "細胞核 (Hematoxylin)")
        
        # 形態學處理分頁
        morph_tab = QWidget()
        morph_layout = QVBoxLayout()
        
        # 平滑處理組
        smooth_group, smooth_sliders = GroupBoxFactory.create_slider_group(
            "平滑處理", 
            SliderConfigs.SMOOTHING, 
            self.schedule_update,
            DEFAULT_VALUE_FORMATTERS
        )
        self.param_manager.register_slider_group("smoothing", smooth_sliders)
        morph_layout.addWidget(smooth_group)
        
        morph_layout.addStretch()
        morph_tab.setLayout(morph_layout)
        tab_widget.addTab(morph_tab, "形態學處理")
        
        return tab_widget
    
    def create_display_group(self):
        """建立顯示控制組"""
        display_group = QGroupBox("顯示設定")
        layout = QVBoxLayout()
        
        # 透明度控制
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("遮罩透明度:"))
        
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(40)
        self.alpha_value_label = QLabel("40%")
        
        self.alpha_slider.valueChanged.connect(
            lambda v: self.alpha_value_label.setText(f"{v}%")
        )
        self.alpha_slider.valueChanged.connect(self.update_display_only)
        
        alpha_layout.addWidget(self.alpha_slider)
        alpha_layout.addWidget(self.alpha_value_label)
        layout.addLayout(alpha_layout)
        
        # 顯示模式選擇
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("顯示模式:"))
        
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems([
            "細胞質疊加", "細胞核疊加", "排他核疊加", "原始影像"
        ])
        self.display_mode_combo.currentTextChanged.connect(self.change_display_mode)
        
        mode_layout.addWidget(self.display_mode_combo)
        layout.addLayout(mode_layout)
        
        display_group.setLayout(layout)
        return display_group
    
    def create_info_group(self):
        """建立資訊顯示組"""
        info_group = QGroupBox("影像資訊")
        layout = QVBoxLayout()
        
        self.info_label = QLabel("等待載入影像...")
        self.info_label.setStyleSheet(StyleSheets.INFO_LABEL)
        layout.addWidget(self.info_label)
        
        self.stats_label = QLabel("處理統計：等待處理...")
        self.stats_label.setStyleSheet(StyleSheets.STATS_LABEL)
        layout.addWidget(self.stats_label)
        
        info_group.setLayout(layout)
        return info_group
    
    def create_button_layout(self):
        """建立按鈕佈局"""
        layout = QVBoxLayout()
        
        self.save_btn = QPushButton("儲存所有結果")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet(StyleSheets.SAVE_BUTTON)
        
        reset_btn = QPushButton("重置參數")
        reset_btn.clicked.connect(self.reset_parameters)
        
        save_params_btn = QPushButton("儲存參數...")
        save_params_btn.clicked.connect(self.save_parameters)
        
        load_params_btn = QPushButton("載入參數...")
        load_params_btn.clicked.connect(self.load_parameters)
        
        layout.addWidget(self.save_btn)
        layout.addWidget(reset_btn)
        layout.addWidget(save_params_btn)
        layout.addWidget(load_params_btn)
        
        return layout
        
    def create_image_panel(self):
        """建立右側圖像顯示面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        self.image_label = QLabel("尚未載入影像...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet(StyleSheets.IMAGE_LABEL)
        
        layout.addWidget(self.image_label)
        panel.setLayout(layout)
        return panel
    
    def get_current_params(self) -> HEMaskingParams:
        """取得目前的參數設定"""
        values = self.param_manager.get_all_values()
        
        # 確保 kernel 大小為奇數
        for key in ['cyto_kernel_size', 'nuclei_kernel_size', 'gaussian_kernel', 'median_kernel']:
            if key in values:
                kernel_value = values[key]
                values[key] = kernel_value if kernel_value % 2 == 1 else kernel_value + 1
        
        # 加入透明度
        if self.alpha_slider:
            values['alpha'] = self.alpha_slider.value()
        
        # 建立參數物件，使用預設值填充缺失的參數
        params = HEMaskingParams()
        
        # 直接設定所有存在的屬性
        for key, value in values.items():
            if hasattr(params, key):
                setattr(params, key, value)
        
        return params
    
    def schedule_update(self):
        """排程更新"""
        if self.processor.working_image is None:
            return
        self.update_timer.stop()
        self.update_timer.start(self.debounce_delay)
        
    def request_processing(self):
        """請求背景處理"""
        if self.processor.working_image is None:
            return
            
        if self.processing_thread and self.processing_thread.isRunning():
            return
            
        params = self.get_current_params()
        
        self.processing_thread = HEProcessingThread(self.processor, params)
        self.processing_thread.result_ready.connect(self.on_processing_complete)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        
        self.statusBar().showMessage("處理中...")
        self.processing_thread.start()
    
    @pyqtSlot(object)
    def on_processing_complete(self, result: HEProcessingResult):
        """處理完成回調"""
        try:
            self.current_result = result
            self.update_display(result)
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
        """僅更新顯示"""
        if self.current_result is not None:
            self.update_display(self.current_result)
    
    def change_display_mode(self, mode_text: str):
        """改變顯示模式"""
        mode_map = {
            "細胞質疊加": "overlay_cyto",
            "細胞核疊加": "overlay_nuclei", 
            "排他核疊加": "overlay_nuclei_exclusive",
            "原始影像": "original"
        }
        
        self.current_display_mode = mode_map.get(mode_text, "overlay_cyto")
        
        if self.current_result is not None:
            self.update_display(self.current_result)
        elif self.current_display_mode == "original" and self.processor.working_image is not None:
            self.display_image(self.processor.working_image)
    
    def update_display(self, result: HEProcessingResult):
        """更新顯示結果"""
        try:
            alpha = self.alpha_slider.value()
            
            if self.current_display_mode == "overlay_cyto":
                overlay = self.processor._create_overlay(
                    self.processor.working_image, result.mask_cyto_clean, 
                    alpha, color=(0, 255, 255)
                )
                self.display_image(overlay)
            elif self.current_display_mode == "overlay_nuclei":
                overlay = self.processor._create_overlay(
                    self.processor.working_image, result.mask_nuclei_clean, 
                    alpha, color=(255, 0, 0)
                )
                self.display_image(overlay)
            elif self.current_display_mode == "overlay_nuclei_exclusive":
                overlay = self.processor._create_overlay(
                    self.processor.working_image, result.mask_nuclei_exclusive, 
                    alpha, color=(255, 0, 0)
                )
                self.display_image(overlay)
            elif self.current_display_mode == "original":
                self.display_image(self.processor.working_image)
            
            self.update_stats_display(result)
                
        except Exception as e:
            QMessageBox.warning(self, "錯誤", f"更新顯示失敗: {str(e)}")

    def update_stats_display(self, result: HEProcessingResult):
        """更新統計資訊顯示"""
        try:
            # 更新影像資訊
            self.update_image_info_display()
            
            # 更新統計資訊
            stats_text = f"""處理統計:
• 細胞質覆蓋率: {result.cyto_area_percent:.1f}%
• 細胞核覆蓋率: {result.nuclei_area_percent:.1f}%
• 排他核覆蓋率: {result.nuclei_exclusive_percent:.1f}%
• 處理時間: {result.processing_time:.2f}秒"""
            
            self.stats_label.setText(stats_text)
            
        except Exception as e:
            self.stats_label.setText(f"統計資訊更新失敗: {str(e)}")
    
    def display_image(self, img):
        """顯示圖像"""
        try:
            pixmap = ImageDisplayHelper.numpy_to_qpixmap(img)
            if pixmap.isNull():
                return
            
            # 縮放適應標籤
            label_size = self.image_label.size()
            pixmap = ImageDisplayHelper.scale_pixmap_to_fit(
                pixmap, label_size.width() - 20, label_size.height() - 20
            )
            
            self.image_label.setPixmap(pixmap)
            self.image_label.update()
            
        except Exception as e:
            print(f"顯示影像錯誤: {e}")
    
    def auto_load_he_image(self):
        """自動載入 HE 影像"""
        try:
            # 使用絕對路徑
            picture_dir = Path(__file__).parent.parent.parent / "picture" / "tiff"
            if self.processor.load_he_from_directory(str(picture_dir)):
                info = self.processor.get_image_info()
                self.statusBar().showMessage(f"已載入: {info['filename']}")
                
                if self.processor.working_image is not None:
                    self.display_image(self.processor.working_image)
                
                self.image_label.setText("")
                self.schedule_update()
                
                # 立即更新影像資訊顯示
                self.update_image_info_display()
            else:
                QMessageBox.warning(self, "警告", "無法自動載入 HE 影像\n請使用「載入影像」按鈕手動選擇")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"載入影像時發生錯誤: {e}")
    
    def update_image_info_display(self):
        """更新影像資訊顯示"""
        try:
            info = self.processor.get_image_info()
            info_text = f"""影像資訊:
• 檔案: {info.get('filename', 'N/A')}
• 原始尺寸: {info.get('original_size', (0, 0))[0]}×{info.get('original_size', (0, 0))[1]}
• 工作尺寸: {info.get('working_size', (0, 0))[0]}×{info.get('working_size', (0, 0))[1]}
• 縮放比例: {info.get('working_scale', 0):.1%}"""
            
            self.info_label.setText(info_text)
        except Exception as e:
            self.info_label.setText(f"影像資訊更新失敗: {str(e)}")
    
    def load_custom_image(self):
        """載入自訂影像"""
        picture_dir = Path(__file__).parent.parent.parent / "picture" / "tiff"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇影像檔案", str(picture_dir),
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
                    
                    # 更新影像資訊顯示
                    self.update_image_info_display()
                else:
                    QMessageBox.critical(self, "錯誤", "載入影像失敗")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"載入影像失敗: {e}")
    
    def save_results(self):
        """儲存處理結果"""
        if self.current_result is None:
            QMessageBox.warning(self, "警告", "沒有可儲存的結果")
            return
        
        output_dir = QFileDialog.getExistingDirectory(
            self, "選擇輸出目錄", "output"
        )
        
        if output_dir:
            self.statusBar().showMessage("儲存中...")
            if self.processor.save_results(self.current_result, output_dir):
                QMessageBox.information(self, "儲存成功", f"結果已儲存至:\n{output_dir}")
                self.statusBar().showMessage("儲存完成")
            else:
                QMessageBox.critical(self, "儲存失敗", "無法儲存結果")
                self.statusBar().showMessage("儲存失敗")
    
    def save_parameters(self):
        """儲存參數設定"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "儲存參數檔案", "he_params.json",
            "JSON 檔案 (*.json);;所有檔案 (*)"
        )
        
        if file_path:
            try:
                params = self.get_current_params()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(params.to_dict(), f, indent=2, ensure_ascii=False)
                QMessageBox.information(self, "儲存成功", f"參數已儲存至:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "儲存失敗", f"無法儲存參數: {e}")
    
    def load_parameters(self):
        """載入參數設定"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "載入參數檔案", "",
            "JSON 檔案 (*.json);;所有檔案 (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 設定參數值
                self.param_manager.set_all_values(data)
                
                # 設定透明度
                if 'alpha' in data:
                    self.alpha_slider.setValue(data['alpha'])
                    self.alpha_value_label.setText(f"{data['alpha']}%")
                
                # 立即更新影像資訊顯示
                self.update_image_info_display()
                
                self.schedule_update()
                QMessageBox.information(self, "載入成功", f"參數已從以下檔案載入:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "載入失敗", f"無法載入參數: {e}")
    
    def reset_parameters(self):
        """重置參數"""
        default_params = HEMaskingParams()
        self.param_manager.reset_to_defaults(default_params.to_dict())
        
        self.alpha_slider.setValue(40)
        self.alpha_value_label.setText("40%")
        self.display_mode_combo.setCurrentText("細胞質疊加")
        
        self.schedule_update()
        self.statusBar().showMessage("參數已重置")
    
    def keyPressEvent(self, event):
        """鍵盤快捷鍵"""
        if event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            if self.save_btn.isEnabled():
                self.save_results()
        elif event.key() == Qt.Key_R and event.modifiers() == Qt.ControlModifier:
            self.reset_parameters()
        elif event.key() == Qt.Key_Q and event.modifiers() == Qt.ControlModifier:
            self.close()
        else:
            super().keyPressEvent(event)

def main():
    """主程式進入點"""
    app = QApplication(sys.argv)
    app.setApplicationName('HE 染色細胞遮罩工具')
    app.setStyle('Fusion')
    
    window = HEMaskGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()