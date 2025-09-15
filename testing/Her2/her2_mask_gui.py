#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
咖啡色區域提取工具 - GUI 介面
Coffee Color Region Extraction Tool - GUI Interface
"""

import sys
import os
import numpy as np
import cv2 as cv
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QSlider, QPushButton,
                            QGroupBox, QTextEdit, QMessageBox, QProgressBar,
                            QFileDialog, QSplitter, QCheckBox, QComboBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont

# 導入核心處理邏輯
from her2_mask_core import Her2MaskProcessor, Her2MaskingParams, Her2ProcessingResult

class Her2ProcessingThread(QThread):
    """背景處理執行緒"""
    result_ready = pyqtSignal(object)  # Her2ProcessingResult
    error_occurred = pyqtSignal(str)   # 錯誤訊息
    
    def __init__(self, processor: Her2MaskProcessor, params: Her2MaskingParams):
        super().__init__()
        self.processor = processor
        self.params = params
        
    def run(self):
        try:
            result = self.processor.process_mask(self.params, use_original=False)
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

class Her2MaskGUI(QMainWindow):
    """咖啡色區域提取 GUI 介面"""
    
    def __init__(self):
        super().__init__()
        self.processor = Her2MaskProcessor()
        self.current_result = None
        self.processing_thread = None
        
        # 防抖動計時器
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.request_processing)
        self.debounce_delay = 1000  # 1000ms 防抖動 (較長，避免頻繁處理)
        
        # UI 元件
        self.sliders = {}
        self.current_display_mode = "extract_membrane"  # 預設顯示膜提取模式
        
        self.initUI()
        self.auto_load_her2_image()
        
    def initUI(self):
        """初始化使用者介面"""
        self.setWindowTitle('咖啡色區域提取工具 (簡化版)')
        self.setGeometry(100, 50, 1400, 900)  # 縮小主窗口尺寸 (原1600x1000改為1400x900)
        
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
        
        self.reload_btn = QPushButton("重新載入 HER2")
        self.reload_btn.clicked.connect(self.auto_load_her2_image)
        
        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(self.reload_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # DAB 通道控制組
        dab_group = QGroupBox("DAB 通道閾值範圍 (咖啡色細胞膜)")
        dab_layout = QGridLayout()
        
        # DAB 下限
        dab_lower_label = QLabel("DAB 下限:")
        self.dab_lower_slider = QSlider(Qt.Horizontal)
        self.dab_lower_slider.setRange(0, 255)
        self.dab_lower_slider.setValue(0)      # 預設下限 0
        self.dab_lower_slider.setTracking(False)
        self.dab_lower_value_label = QLabel("0")
        
        # DAB 上限
        dab_upper_label = QLabel("DAB 上限:")
        self.dab_upper_slider = QSlider(Qt.Horizontal)
        self.dab_upper_slider.setRange(0, 255)
        self.dab_upper_slider.setValue(150)    # 預設上限 150 (增加閾值)
        self.dab_upper_slider.setTracking(False)
        self.dab_upper_value_label = QLabel("150")
        
        self.dab_lower_slider.valueChanged.connect(
            lambda v: self.dab_lower_value_label.setText(str(v))
        )
        self.dab_lower_slider.valueChanged.connect(self.schedule_update)
        
        self.dab_upper_slider.valueChanged.connect(
            lambda v: self.dab_upper_value_label.setText(str(v))
        )
        self.dab_upper_slider.valueChanged.connect(self.schedule_update)
        
        # DAB Otsu 選項
        self.dab_otsu_checkbox = QCheckBox("使用 Otsu 自動閾值")
        self.dab_otsu_checkbox.setChecked(False)  # 改為關閉，使用範圍閾值
        self.dab_otsu_checkbox.stateChanged.connect(self.on_dab_otsu_changed)
        
        # DAB 反轉選項
        self.dab_invert_checkbox = QCheckBox("反轉 DAB 通道")
        self.dab_invert_checkbox.setChecked(True)  # 預設開啟
        self.dab_invert_checkbox.stateChanged.connect(self.schedule_update)
        
        dab_layout.addWidget(dab_lower_label, 0, 0)
        dab_layout.addWidget(self.dab_lower_slider, 0, 1)
        dab_layout.addWidget(self.dab_lower_value_label, 0, 2)
        
        dab_layout.addWidget(dab_upper_label, 1, 0)
        dab_layout.addWidget(self.dab_upper_slider, 1, 1)
        dab_layout.addWidget(self.dab_upper_value_label, 1, 2)
        
        dab_layout.addWidget(self.dab_otsu_checkbox, 2, 0, 1, 3)
        dab_layout.addWidget(self.dab_invert_checkbox, 3, 0, 1, 3)
        
        dab_group.setLayout(dab_layout)
        layout.addWidget(dab_group)
        
        # Hematoxylin 通道控制組
        hema_group = QGroupBox("Hematoxylin 通道閾值範圍 (藍色細胞核)")
        hema_layout = QGridLayout()
        
        # Hematoxylin 下限
        hema_lower_label = QLabel("Hematoxylin 下限:")
        self.hema_lower_slider = QSlider(Qt.Horizontal)
        self.hema_lower_slider.setRange(0, 255)
        self.hema_lower_slider.setValue(0)      # 預設下限 0
        self.hema_lower_slider.setTracking(False)
        self.hema_lower_value_label = QLabel("0")
        
        # Hematoxylin 上限
        hema_upper_label = QLabel("Hematoxylin 上限:")
        self.hema_upper_slider = QSlider(Qt.Horizontal)
        self.hema_upper_slider.setRange(0, 255)
        self.hema_upper_slider.setValue(120)    # 預設上限 120
        self.hema_upper_slider.setTracking(False)
        self.hema_upper_value_label = QLabel("120")
        
        self.hema_lower_slider.valueChanged.connect(
            lambda v: self.hema_lower_value_label.setText(str(v))
        )
        self.hema_lower_slider.valueChanged.connect(self.schedule_update)
        
        self.hema_upper_slider.valueChanged.connect(
            lambda v: self.hema_upper_value_label.setText(str(v))
        )
        self.hema_upper_slider.valueChanged.connect(self.schedule_update)
        
        # Hematoxylin Otsu 選項
        self.hema_otsu_checkbox = QCheckBox("使用 Otsu 自動閾值")
        self.hema_otsu_checkbox.setChecked(False)  # 改為關閉，使用範圍閾值
        self.hema_otsu_checkbox.stateChanged.connect(self.on_hema_otsu_changed)
        
        hema_layout.addWidget(hema_lower_label, 0, 0)
        hema_layout.addWidget(self.hema_lower_slider, 0, 1)
        hema_layout.addWidget(self.hema_lower_value_label, 0, 2)
        
        hema_layout.addWidget(hema_upper_label, 1, 0)
        hema_layout.addWidget(self.hema_upper_slider, 1, 1)
        hema_layout.addWidget(self.hema_upper_value_label, 1, 2)
        
        hema_layout.addWidget(self.hema_otsu_checkbox, 2, 0, 1, 3)
        
        hema_group.setLayout(hema_layout)
        layout.addWidget(hema_group)
        
        # LAB 通道控制組
        lab_group = QGroupBox("LAB 通道閾值範圍")
        lab_layout = QGridLayout()
        
        # LAB A通道範圍 (綠-紅軸)
        lab_a_lower_label = QLabel("A通道下限 (綠-紅):")
        self.lab_a_lower_slider = QSlider(Qt.Horizontal)
        self.lab_a_lower_slider.setRange(0, 255)
        self.lab_a_lower_slider.setValue(0)
        self.lab_a_lower_slider.setTracking(False)
        self.lab_a_lower_value_label = QLabel("0")
        
        lab_a_upper_label = QLabel("A通道上限 (綠-紅):")
        self.lab_a_upper_slider = QSlider(Qt.Horizontal)
        self.lab_a_upper_slider.setRange(0, 255)
        self.lab_a_upper_slider.setValue(255)
        self.lab_a_upper_slider.setTracking(False)
        self.lab_a_upper_value_label = QLabel("255")
        
        # LAB B通道範圍 (藍-黃軸)
        lab_b_lower_label = QLabel("B通道下限 (藍-黃):")
        self.lab_b_lower_slider = QSlider(Qt.Horizontal)
        self.lab_b_lower_slider.setRange(0, 255)
        self.lab_b_lower_slider.setValue(0)
        self.lab_b_lower_slider.setTracking(False)
        self.lab_b_lower_value_label = QLabel("0")
        
        lab_b_upper_label = QLabel("B通道上限 (藍-黃):")
        self.lab_b_upper_slider = QSlider(Qt.Horizontal)
        self.lab_b_upper_slider.setRange(0, 255)
        self.lab_b_upper_slider.setValue(255)
        self.lab_b_upper_slider.setTracking(False)
        self.lab_b_upper_value_label = QLabel("255")
        
        # 連接事件
        self.lab_a_lower_slider.valueChanged.connect(
            lambda v: self.lab_a_lower_value_label.setText(str(v))
        )
        self.lab_a_lower_slider.valueChanged.connect(self.schedule_update)
        
        self.lab_a_upper_slider.valueChanged.connect(
            lambda v: self.lab_a_upper_value_label.setText(str(v))
        )
        self.lab_a_upper_slider.valueChanged.connect(self.schedule_update)
        
        self.lab_b_lower_slider.valueChanged.connect(
            lambda v: self.lab_b_lower_value_label.setText(str(v))
        )
        self.lab_b_lower_slider.valueChanged.connect(self.schedule_update)
        
        self.lab_b_upper_slider.valueChanged.connect(
            lambda v: self.lab_b_upper_value_label.setText(str(v))
        )
        self.lab_b_upper_slider.valueChanged.connect(self.schedule_update)
        
        # 布局
        lab_layout.addWidget(lab_a_lower_label, 0, 0)
        lab_layout.addWidget(self.lab_a_lower_slider, 0, 1)
        lab_layout.addWidget(self.lab_a_lower_value_label, 0, 2)
        
        lab_layout.addWidget(lab_a_upper_label, 1, 0)
        lab_layout.addWidget(self.lab_a_upper_slider, 1, 1)
        lab_layout.addWidget(self.lab_a_upper_value_label, 1, 2)
        
        lab_layout.addWidget(lab_b_lower_label, 2, 0)
        lab_layout.addWidget(self.lab_b_lower_slider, 2, 1)
        lab_layout.addWidget(self.lab_b_lower_value_label, 2, 2)
        
        lab_layout.addWidget(lab_b_upper_label, 3, 0)
        lab_layout.addWidget(self.lab_b_upper_slider, 3, 1)
        lab_layout.addWidget(self.lab_b_upper_value_label, 3, 2)
        
        lab_group.setLayout(lab_layout)
        layout.addWidget(lab_group)
        
        # 形態學處理控制組
        morph_group = QGroupBox("形態學處理")
        morph_layout = QGridLayout()
        
        morph_configs = [
            ('membrane_kernel_size', '膜 Kernel 大小', 1, 15, 3),
            ('membrane_open_iter', '膜開運算次數', 0, 5, 1),
            ('membrane_close_iter', '膜閉運算次數', 0, 5, 2),
            ('nuclei_kernel_size', '核 Kernel 大小', 1, 15, 3),
            ('nuclei_open_iter', '核開運算次數', 0, 5, 1),
            ('nuclei_close_iter', '核閉運算次數', 0, 5, 1),
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
            if 'kernel_size' in key:
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
        
        # 移除邊緣檢測控制組 - 簡化版本不需要
        
        # 移除核周圍 ROI 控制組 - 簡化版本不需要
        
        # 顯示控制組
        display_group = QGroupBox("顯示設定")
        display_layout = QGridLayout()
        
        # 顯示模式選擇
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("顯示模式:"))
        
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems([
            "膜提取 (原圖+透明背景)", "膜疊加 (紅色遮罩)", "原始影像", "純膜遮罩"
        ])
        self.display_mode_combo.currentTextChanged.connect(self.change_display_mode)
        
        mode_layout.addWidget(self.display_mode_combo)
        display_layout.addLayout(mode_layout, 0, 0, 1, 3)
        
        alpha_label = QLabel("遮罩透明度:")
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_value_label = QLabel("50%")
        
        self.alpha_slider.valueChanged.connect(
            lambda v: self.alpha_value_label.setText(f"{v}%")
        )
        self.alpha_slider.valueChanged.connect(self.update_display_only)
        
        display_layout.addWidget(alpha_label, 1, 0)
        display_layout.addWidget(self.alpha_slider, 1, 1)
        display_layout.addWidget(self.alpha_value_label, 1, 2)
        
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
        self.image_label.setMinimumSize(600, 450)  # 縮小預覽圖尺寸 (原800x600改為600x450)
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
    
    def get_current_params(self) -> Her2MaskingParams:
        """取得目前的參數設定"""
        params = Her2MaskingParams()
        
        # DAB 參數
        params.dab_lower = self.dab_lower_slider.value()
        params.dab_upper = self.dab_upper_slider.value()
        params.dab_use_otsu = self.dab_otsu_checkbox.isChecked()
        params.dab_invert = self.dab_invert_checkbox.isChecked()
        
        # Hematoxylin 參數
        params.hema_lower = self.hema_lower_slider.value()
        params.hema_upper = self.hema_upper_slider.value()
        params.hema_use_otsu = self.hema_otsu_checkbox.isChecked()
        
        # LAB 參數
        params.lab_a_lower = self.lab_a_lower_slider.value()
        params.lab_a_upper = self.lab_a_upper_slider.value()
        params.lab_b_lower = self.lab_b_lower_slider.value()
        params.lab_b_upper = self.lab_b_upper_slider.value()
        
        # 形態學參數
        for key in ['membrane_kernel_size', 'membrane_open_iter', 'membrane_close_iter',
                    'nuclei_kernel_size', 'nuclei_open_iter', 'nuclei_close_iter']:
            if key in self.sliders:
                value = self.sliders[key][0].value()
                if 'kernel_size' in key:
                    # 確保 kernel_size 為奇數
                    value = value if value % 2 == 1 else value + 1
                setattr(params, key, value)
        
        # 移除邊緣檢測參數 - 簡化版本不需要
        
        # 移除ROI參數 - 簡化版本不需要
        
        # 顯示參數
        params.alpha = self.alpha_slider.value()
        
        return params
    
    def on_dab_otsu_changed(self, state):
        """DAB Otsu 選項改變處理"""
        is_otsu = self.dab_otsu_checkbox.isChecked()
        # 當使用Otsu時，禁用範圍滑桿
        self.dab_lower_slider.setEnabled(not is_otsu)
        self.dab_lower_value_label.setEnabled(not is_otsu)
        self.dab_upper_slider.setEnabled(not is_otsu)
        self.dab_upper_value_label.setEnabled(not is_otsu)
        self.schedule_update()
    
    def on_hema_otsu_changed(self, state):
        """Hematoxylin Otsu 選項改變處理"""
        is_otsu = self.hema_otsu_checkbox.isChecked()
        # 當使用Otsu時，禁用範圍滑桿
        self.hema_lower_slider.setEnabled(not is_otsu)
        self.hema_lower_value_label.setEnabled(not is_otsu)
        self.hema_upper_slider.setEnabled(not is_otsu)
        self.hema_upper_value_label.setEnabled(not is_otsu)
        self.schedule_update()
    
    def schedule_update(self):
        """排程更新 (防抖動)"""
        # 如果沒有載入影像，不需要排程處理
        if self.processor.working_image is None:
            return
            
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
        print(f"GUI: 開始背景處理，DAB範圍=[{params.dab_lower}-{params.dab_upper}], LAB A=[{params.lab_a_lower}-{params.lab_a_upper}], B=[{params.lab_b_lower}-{params.lab_b_upper}]")
        
        # 建立背景處理執行緒
        self.processing_thread = Her2ProcessingThread(self.processor, params)
        self.processing_thread.result_ready.connect(self.on_processing_complete)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        
        self.statusBar().showMessage("處理中...")
        self.processing_thread.start()
    
    @pyqtSlot(object)
    def on_processing_complete(self, result: Her2ProcessingResult):
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
    
    def change_display_mode(self, mode_text: str):
        """改變顯示模式"""
        mode_map = {
            "膜提取 (原圖+透明背景)": "extract_membrane",
            "膜疊加 (紅色遮罩)": "overlay_membrane", 
            "原始影像": "original",
            "純膜遮罩": "mask_only"
        }
        
        self.current_display_mode = mode_map.get(mode_text, "extract_membrane")
        
        if self.current_result is not None:
            self.update_display(self.current_result)
        elif self.current_display_mode == "original" and self.processor.working_image is not None:
            self.display_image(self.processor.working_image)
    
    def update_display_only(self):
        """僅更新顯示 (不重新處理)"""
        if self.current_result is not None:
            self.update_display(self.current_result)
    
    def update_display(self, result: Her2ProcessingResult):
        """更新顯示結果"""
        try:
            alpha = self.alpha_slider.value()
            
            if self.current_display_mode == "extract_membrane":
                # 顯示膜提取結果 (透明背景)
                extract_img = result.extract_membrane
                # 轉換 BGRA 為顯示用的 BGR (白色背景)
                if extract_img.shape[2] == 4:
                    alpha_channel = extract_img[:, :, 3] / 255.0
                    alpha_3d = np.stack([alpha_channel] * 3, axis=2)
                    
                    # 創建白色背景
                    white_bg = np.ones_like(extract_img[:, :, :3]) * 255
                    
                    # 混合提取的膜與白色背景
                    display_img = (extract_img[:, :, :3] * alpha_3d + 
                                 white_bg * (1 - alpha_3d)).astype(np.uint8)
                else:
                    display_img = extract_img
                
                self.display_image(display_img)
                
            elif self.current_display_mode == "overlay_membrane":
                # 顯示紅色疊加遮罩
                overlay = self.processor._create_overlay(
                    self.processor.working_image, result.mask_membrane_clean, 
                    alpha, color=(0, 0, 255)  # 紅色膜遮罩
                )
                self.display_image(overlay)
                
            elif self.current_display_mode == "original":
                # 顯示原始影像
                self.display_image(self.processor.working_image)
                
            elif self.current_display_mode == "mask_only":
                # 顯示純膜遮罩 (黑白)
                mask_rgb = cv.cvtColor(result.mask_membrane_clean, cv.COLOR_GRAY2BGR)
                self.display_image(mask_rgb)
            
            self.update_stats_display(result)
                
        except Exception as e:
            QMessageBox.warning(self, "錯誤", f"更新顯示失敗: {str(e)}")
    
    def update_display_only(self):
        """僅更新顯示 (不重新處理)"""
        if self.current_result is not None:
            self.update_display(self.current_result)
    
    def update_stats_display(self, result: Her2ProcessingResult):
        """更新統計資訊顯示"""
        try:
            # 更新影像資訊
            self.update_image_info_display()
            
            # 更新統計資訊
            stats_text = f"""處理統計:
• 咖啡色覆蓋率: {result.membrane_coverage_percent:.2f}%
• DAB平均值: {result.dab_mean:.3f}
• DAB中位數: {result.dab_median:.3f}
• 處理時間: {result.processing_time:.2f}秒"""
            
            self.stats_label.setText(stats_text)
            
        except Exception as e:
            self.stats_label.setText(f"統計資訊更新失敗: {str(e)}")
    
    def display_image(self, img):
        """顯示圖像"""
        try:
            pixmap = self.numpy_to_qpixmap(img)
            if pixmap.isNull():
                return
            
            # 縮放適應標籤
            label_size = self.image_label.size()
            pixmap = self.scale_pixmap_to_fit(
                pixmap, label_size.width() - 20, label_size.height() - 20
            )
            
            self.image_label.setPixmap(pixmap)
            self.image_label.update()
            
        except Exception as e:
            print(f"顯示影像錯誤: {e}")
    
    def numpy_to_qpixmap(self, img: np.ndarray) -> QPixmap:
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
    
    def scale_pixmap_to_fit(self, pixmap: QPixmap, max_width: int, max_height: int) -> QPixmap:
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
    
    def auto_load_her2_image(self):
        """自動載入 HER2 影像"""
        try:
            # 使用絕對路徑
            picture_dir = Path(__file__).parent.parent.parent / "picture" / "tiff"
            if self.processor.load_her2_from_directory(str(picture_dir)):
                info = self.processor.get_image_info()
                self.statusBar().showMessage(f"已載入: {info['filename']}")
                
                if self.processor.working_image is not None:
                    self.display_image(self.processor.working_image)
                
                self.image_label.setText("")
                self.schedule_update()
                
                # 立即更新影像資訊顯示
                self.update_image_info_display()
            else:
                QMessageBox.warning(self, "警告", "無法自動載入 HER2 影像\n請使用「載入影像」按鈕手動選擇")
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
        
        # 固定輸出到 testing/Her2/output/
        output_dir = Path(__file__).parent / "output"
        
        self.statusBar().showMessage("儲存中...")
        if self.processor.save_results(self.current_result, str(output_dir)):
            QMessageBox.information(self, "儲存成功", f"結果已儲存至:\n{output_dir}")
            self.statusBar().showMessage("儲存完成")
        else:
            QMessageBox.critical(self, "儲存失敗", "無法儲存結果")
            self.statusBar().showMessage("儲存失敗")
    
    def reset_parameters(self):
        """重置參數"""
        default_params = Her2MaskingParams()
        
        # 重置 DAB 參數
        self.dab_lower_slider.setValue(default_params.dab_lower)
        self.dab_lower_value_label.setText(str(default_params.dab_lower))
        self.dab_upper_slider.setValue(default_params.dab_upper)
        self.dab_upper_value_label.setText(str(default_params.dab_upper))
        self.dab_otsu_checkbox.setChecked(default_params.dab_use_otsu)
        self.dab_invert_checkbox.setChecked(default_params.dab_invert)
        
        # 重置 LAB 參數
        self.lab_a_lower_slider.setValue(default_params.lab_a_lower)
        self.lab_a_lower_value_label.setText(str(default_params.lab_a_lower))
        self.lab_a_upper_slider.setValue(default_params.lab_a_upper)
        self.lab_a_upper_value_label.setText(str(default_params.lab_a_upper))
        self.lab_b_lower_slider.setValue(default_params.lab_b_lower)
        self.lab_b_lower_value_label.setText(str(default_params.lab_b_lower))
        self.lab_b_upper_slider.setValue(default_params.lab_b_upper)
        self.lab_b_upper_value_label.setText(str(default_params.lab_b_upper))
        
        # 重置 Hematoxylin 參數
        self.hema_lower_slider.setValue(default_params.hema_lower)
        self.hema_lower_value_label.setText(str(default_params.hema_lower))
        self.hema_upper_slider.setValue(default_params.hema_upper)
        self.hema_upper_value_label.setText(str(default_params.hema_upper))
        self.hema_otsu_checkbox.setChecked(default_params.hema_use_otsu)
        
        # 重置形態學參數
        morph_defaults = {
            'membrane_kernel_size': default_params.membrane_kernel_size,
            'membrane_open_iter': default_params.membrane_open_iter,
            'membrane_close_iter': default_params.membrane_close_iter,
            'nuclei_kernel_size': default_params.nuclei_kernel_size,
            'nuclei_open_iter': default_params.nuclei_open_iter,
            'nuclei_close_iter': default_params.nuclei_close_iter,
        }
        
        for key, default_val in morph_defaults.items():
            if key in self.sliders:
                self.sliders[key][0].setValue(default_val)
                self.sliders[key][1].setText(str(default_val))
        
        # 移除邊緣檢測參數重置 - 簡化版本不需要
        
        # 重置 ROI 參數
        # 移除ROI參數重置 - 簡化版本不需要
        
        # 重置顯示參數
        self.alpha_slider.setValue(default_params.alpha)
        self.alpha_value_label.setText(f"{default_params.alpha}%")
        
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
    app.setApplicationName('咖啡色區域提取工具')
    app.setStyle('Fusion')
    
    window = Her2MaskGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()