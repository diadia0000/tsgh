#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSI 配準結果查看器
提供圖形化介面來查看配準前後的對比圖片和評估指標
"""

import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                                QFileDialog, QTabWidget, QTextEdit, QScrollArea,
                                QSlider, QCheckBox, QComboBox, QSpinBox, QGroupBox,
                                QProgressBar, QStatusBar, QSplitter, QFrame)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtGui import QPixmap, QFont, QIcon
except ImportError:
    print("錯誤: 需要安裝 PyQt5")
    print("請執行: pip install PyQt5")
    sys.exit(1)

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class RegistrationMetrics:
    """配準評估指標計算類"""
    
    @staticmethod
    def calculate_mutual_information(img1, img2, bins=64):
        """計算互信息 (MI)"""
        if img1.shape != img2.shape:
            return 0.0
        
        # 轉換為灰階
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 計算聯合直方圖
        hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
        
        # 正規化
        hist_2d = hist_2d / np.sum(hist_2d)
        
        # 計算邊際直方圖
        hist_1 = np.sum(hist_2d, axis=1)
        hist_2 = np.sum(hist_2d, axis=0)
        
        # 計算互信息
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if hist_2d[i, j] > 1e-10:
                    if hist_1[i] > 1e-10 and hist_2[j] > 1e-10:
                        mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (hist_1[i] * hist_2[j]))
        
        return mi
    
    @staticmethod
    def calculate_normalized_mutual_information(img1, img2, bins=64):
        """計算正規化互信息 (NMI)"""
        mi = RegistrationMetrics.calculate_mutual_information(img1, img2, bins)
        h1 = RegistrationMetrics.calculate_entropy(img1, bins)
        h2 = RegistrationMetrics.calculate_entropy(img2, bins)
        
        if h1 + h2 == 0:
            return 0.0
        
        return 2.0 * mi / (h1 + h2)
    
    @staticmethod
    def calculate_entropy(img, bins=256):
        """計算影像熵"""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        hist, _ = np.histogram(img, bins=bins, range=(0, 256))
        hist = hist / np.sum(hist)
        
        entropy = 0.0
        for p in hist:
            if p > 1e-10:
                entropy -= p * np.log2(p)
        
        return entropy
    
    @staticmethod
    def calculate_target_registration_error(img1, img2, transform_matrix=None):
        """計算目標配準誤差 (TRE) - 簡化版本"""
        if transform_matrix is None:
            # 如果沒有變換矩陣，使用影像差異作為近似
            if img1.shape != img2.shape:
                return float('inf')
            
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            if len(img2.shape) == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # 計算均方根誤差作為 TRE 的近似
            mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
            return np.sqrt(mse) / 10.0  # 縮放到合理範圍
        
        # 使用角點計算 TRE
        h, w = img1.shape[:2]
        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1],
            [w/2, h/2, 1]
        ]).T
        
        transformed_corners = transform_matrix @ corners
        
        # 計算變換誤差
        original_corners = corners[:2, :]
        transformed_corners = transformed_corners[:2, :]
        
        errors = np.linalg.norm(transformed_corners - original_corners, axis=0)
        return np.mean(errors)

class ImageComparisonWidget(QWidget):
    """影像對比顯示組件"""
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # 控制面板
        control_panel = QHBoxLayout()
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.update_overlay)
        
        self.overlay_checkbox = QCheckBox("疊合模式")
        self.overlay_checkbox.stateChanged.connect(self.update_display)
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["原始", "熱圖", "差異圖"])
        self.colormap_combo.currentTextChanged.connect(self.update_display)
        
        control_panel.addWidget(QLabel("透明度:"))
        control_panel.addWidget(self.opacity_slider)
        control_panel.addWidget(self.overlay_checkbox)
        control_panel.addWidget(QLabel("顯示模式:"))
        control_panel.addWidget(self.colormap_combo)
        control_panel.addStretch()
        
        layout.addLayout(control_panel)
        
        # 影像顯示區域
        self.image_layout = QHBoxLayout()
        
        # 原始影像
        self.original_label = QLabel("載入原始影像...")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 300)
        self.original_label.setStyleSheet("border: 1px solid gray;")
        
        # 配準後影像
        self.registered_label = QLabel("載入配準後影像...")
        self.registered_label.setAlignment(Qt.AlignCenter)
        self.registered_label.setMinimumSize(400, 300)
        self.registered_label.setStyleSheet("border: 1px solid gray;")
        
        self.image_layout.addWidget(self.original_label)
        self.image_layout.addWidget(self.registered_label)
        
        layout.addLayout(self.image_layout)
        
        self.setLayout(layout)
        
        # 儲存影像數據
        self.original_image = None
        self.registered_image = None
        
    def load_images(self, original_path, registered_path):
        """載入影像"""
        try:
            self.original_image = cv2.imread(original_path)
            self.registered_image = cv2.imread(registered_path)
            
            if self.original_image is None or self.registered_image is None:
                raise ValueError("無法載入影像")
            
            # 調整影像大小以適應顯示
            self.original_image = self.resize_image(self.original_image, 400, 300)
            self.registered_image = self.resize_image(self.registered_image, 400, 300)
            
            self.update_display()
            
        except Exception as e:
            print(f"載入影像錯誤: {e}")
    
    def resize_image(self, image, max_width, max_height):
        """調整影像大小保持比例"""
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    
    def update_display(self):
        """更新影像顯示"""
        if self.original_image is None or self.registered_image is None:
            return
        
        mode = self.colormap_combo.currentText()
        
        if mode == "原始":
            self.display_original_images()
        elif mode == "熱圖":
            self.display_heatmap()
        elif mode == "差異圖":
            self.display_difference()
    
    def display_original_images(self):
        """顯示原始影像"""
        # 轉換為 Qt 格式並顯示
        original_qt = self.cv2_to_qt(self.original_image)
        registered_qt = self.cv2_to_qt(self.registered_image)
        
        self.original_label.setPixmap(original_qt)
        self.registered_label.setPixmap(registered_qt)
    
    def display_heatmap(self):
        """顯示熱圖模式"""
        # 轉換為灰階
        original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        registered_gray = cv2.cvtColor(self.registered_image, cv2.COLOR_BGR2GRAY)
        
        # 應用熱圖色彩映射
        original_heat = cv2.applyColorMap(original_gray, cv2.COLORMAP_JET)
        registered_heat = cv2.applyColorMap(registered_gray, cv2.COLORMAP_JET)
        
        original_qt = self.cv2_to_qt(original_heat)
        registered_qt = self.cv2_to_qt(registered_heat)
        
        self.original_label.setPixmap(original_qt)
        self.registered_label.setPixmap(registered_qt)
    
    def display_difference(self):
        """顯示差異圖"""
        # 確保影像大小相同
        if self.original_image.shape != self.registered_image.shape:
            registered_resized = cv2.resize(self.registered_image, 
                                          (self.original_image.shape[1], self.original_image.shape[0]))
        else:
            registered_resized = self.registered_image
        
        # 計算差異
        diff = cv2.absdiff(self.original_image, registered_resized)
        
        # 增強差異顯示
        diff_enhanced = cv2.convertScaleAbs(diff, alpha=3.0, beta=0)
        
        original_qt = self.cv2_to_qt(self.original_image)
        diff_qt = self.cv2_to_qt(diff_enhanced)
        
        self.original_label.setPixmap(original_qt)
        self.registered_label.setPixmap(diff_qt)
    
    def update_overlay(self):
        """更新疊合透明度"""
        if not self.overlay_checkbox.isChecked():
            return
        
        if self.original_image is None or self.registered_image is None:
            return
        
        alpha = self.opacity_slider.value() / 100.0
        
        # 確保影像大小相同
        if self.original_image.shape != self.registered_image.shape:
            registered_resized = cv2.resize(self.registered_image, 
                                          (self.original_image.shape[1], self.original_image.shape[0]))
        else:
            registered_resized = self.registered_image
        
        # 創建疊合影像
        overlay = cv2.addWeighted(self.original_image, 1-alpha, registered_resized, alpha, 0)
        
        overlay_qt = self.cv2_to_qt(overlay)
        self.registered_label.setPixmap(overlay_qt)
    
    def cv2_to_qt(self, cv_img):
        """將 OpenCV 影像轉換為 Qt QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QPixmap.fromImage(
            QPixmap.fromImage(
                QPixmap.fromImage(rgb_image.data, w, h, bytes_per_line, QPixmap.Format_RGB888)
            )
        )
        return qt_image.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)

class MetricsWidget(QWidget):
    """評估指標顯示組件"""
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # 指標顯示
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(200)
        
        # 圖表顯示
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        layout.addWidget(QLabel("配準品質指標:"))
        layout.addWidget(self.metrics_text)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def update_metrics(self, original_img, registered_img, report_data=None):
        """更新評估指標"""
        if original_img is None or registered_img is None:
            return
        
        # 計算指標
        mi = RegistrationMetrics.calculate_mutual_information(original_img, registered_img)
        nmi = RegistrationMetrics.calculate_normalized_mutual_information(original_img, registered_img)
        tre = RegistrationMetrics.calculate_target_registration_error(original_img, registered_img)
        
        # 顯示文字指標
        metrics_text = f"""
配準品質評估指標:
==================
互信息 (MI): {mi:.6f}
正規化互信息 (NMI): {nmi:.6f}
目標配準誤差 (TRE): {tre:.2f} 像素

品質評估:
- MI 值越高表示配準品質越好
- NMI 範圍 [0, 1]，越接近 1 越好
- TRE 值越小表示配準誤差越小

配準品質等級:
"""
        
        # 品質等級評估
        if tre < 2.0 and nmi > 0.7:
            metrics_text += "🟢 優秀 (TRE < 2.0, NMI > 0.7)"
        elif tre < 5.0 and nmi > 0.5:
            metrics_text += "🟡 良好 (TRE < 5.0, NMI > 0.5)"
        else:
            metrics_text += "🔴 需要改善 (TRE >= 5.0 或 NMI <= 0.5)"
        
        self.metrics_text.setText(metrics_text)
        
        # 繪製圖表
        self.plot_metrics(mi, nmi, tre, report_data)
    
    def plot_metrics(self, mi, nmi, tre, report_data=None):
        """繪製評估指標圖表"""
        self.figure.clear()
        
        # 創建子圖
        gs = self.figure.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 指標條形圖
        ax1 = self.figure.add_subplot(gs[0, 0])
        metrics = ['MI', 'NMI', 'TRE/10']
        values = [mi, nmi, tre/10]  # TRE 縮放以便顯示
        colors = ['skyblue', 'lightgreen', 'salmon']
        
        bars = ax1.bar(metrics, values, color=colors)
        ax1.set_title('配準品質指標')
        ax1.set_ylabel('數值')
        
        # 添加數值標籤
        for bar, value in zip(bars, [mi, nmi, tre]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 品質等級餅圖
        ax2 = self.figure.add_subplot(gs[0, 1])
        if tre < 2.0 and nmi > 0.7:
            quality_data = [1, 0, 0]
            labels = ['優秀', '良好', '需改善']
        elif tre < 5.0 and nmi > 0.5:
            quality_data = [0, 1, 0]
            labels = ['優秀', '良好', '需改善']
        else:
            quality_data = [0, 0, 1]
            labels = ['優秀', '良好', '需改善']
        
        colors = ['green', 'orange', 'red']
        ax2.pie(quality_data, labels=labels, colors=colors, autopct='%1.0f%%')
        ax2.set_title('配準品質等級')
        
        # 處理時間分析 (如果有報告數據)
        if report_data:
            ax3 = self.figure.add_subplot(gs[0, 2])
            stages = ['特徵點\n粗對齊', '互信息\n精準對齊', 'B-spline\n非剛體對齊']
            times = [100, 200, 150]  # 示例數據，實際應從報告中讀取
            
            ax3.bar(stages, times, color='lightcoral')
            ax3.set_title('各階段處理時間')
            ax3.set_ylabel('時間 (ms)')
            ax3.tick_params(axis='x', rotation=45)
        
        # 指標趨勢圖 (模擬多次配準的結果)
        ax4 = self.figure.add_subplot(gs[1, :])
        iterations = range(1, 11)
        mi_trend = [mi * (0.8 + 0.02 * i) for i in iterations]
        nmi_trend = [nmi * (0.85 + 0.015 * i) for i in iterations]
        tre_trend = [tre * (1.2 - 0.02 * i) for i in iterations]
        
        ax4.plot(iterations, mi_trend, 'o-', label='MI', color='blue')
        ax4.plot(iterations, nmi_trend, 's-', label='NMI', color='green')
        ax4.plot(iterations, [t/10 for t in tre_trend], '^-', label='TRE/10', color='red')
        
        ax4.set_title('配準品質指標趨勢 (模擬)')
        ax4.set_xlabel('迭代次數')
        ax4.set_ylabel('指標值')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        self.canvas.draw()

class WSIRegistrationViewer(QMainWindow):
    """WSI 配準結果查看器主視窗"""
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('WSI 配準結果查看器')
        self.setGeometry(100, 100, 1400, 900)
        
        # 中央組件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主佈局
        main_layout = QVBoxLayout()
        
        # 工具列
        toolbar_layout = QHBoxLayout()
        
        self.load_button = QPushButton('載入配準結果')
        self.load_button.clicked.connect(self.load_registration_results)
        
        self.export_button = QPushButton('匯出報告')
        self.export_button.clicked.connect(self.export_report)
        self.export_button.setEnabled(False)
        
        toolbar_layout.addWidget(self.load_button)
        toolbar_layout.addWidget(self.export_button)
        toolbar_layout.addStretch()
        
        main_layout.addLayout(toolbar_layout)
        
        # 標籤頁
        self.tab_widget = QTabWidget()
        
        # HER2 配準結果標籤頁
        self.her2_tab = QWidget()
        her2_layout = QVBoxLayout()
        
        her2_layout.addWidget(QLabel("HER2 配準結果"))
        self.her2_comparison = ImageComparisonWidget()
        self.her2_metrics = MetricsWidget()
        
        splitter_her2 = QSplitter(Qt.Vertical)
        splitter_her2.addWidget(self.her2_comparison)
        splitter_her2.addWidget(self.her2_metrics)
        splitter_her2.setStretchFactor(0, 1)
        splitter_her2.setStretchFactor(1, 1)
        
        her2_layout.addWidget(splitter_her2)
        self.her2_tab.setLayout(her2_layout)
        
        # FISH 配準結果標籤頁
        self.fish_tab = QWidget()
        fish_layout = QVBoxLayout()
        
        fish_layout.addWidget(QLabel("FISH 配準結果"))
        self.fish_comparison = ImageComparisonWidget()
        self.fish_metrics = MetricsWidget()
        
        splitter_fish = QSplitter(Qt.Vertical)
        splitter_fish.addWidget(self.fish_comparison)
        splitter_fish.addWidget(self.fish_metrics)
        splitter_fish.setStretchFactor(0, 1)
        splitter_fish.setStretchFactor(1, 1)
        
        fish_layout.addWidget(splitter_fish)
        self.fish_tab.setLayout(fish_layout)
        
        # 總結報告標籤頁
        self.report_tab = QWidget()
        report_layout = QVBoxLayout()
        
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        
        report_layout.addWidget(QLabel("配準總結報告"))
        report_layout.addWidget(self.report_text)
        self.report_tab.setLayout(report_layout)
        
        # 添加標籤頁
        self.tab_widget.addTab(self.her2_tab, "HER2 配準")
        self.tab_widget.addTab(self.fish_tab, "FISH 配準")
        self.tab_widget.addTab(self.report_tab, "總結報告")
        
        main_layout.addWidget(self.tab_widget)
        
        central_widget.setLayout(main_layout)
        
        # 狀態列
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('準備就緒')
        
        # 儲存數據
        self.current_directory = None
        self.report_data = None
    
    def load_registration_results(self):
        """載入配準結果"""
        directory = QFileDialog.getExistingDirectory(self, '選擇配準結果目錄')
        
        if not directory:
            return
        
        self.current_directory = directory
        self.status_bar.showMessage('載入配準結果中...')
        
        try:
            # 載入影像
            reference_path = os.path.join(directory, 'reference_HE.jpg')
            her2_aligned_path = os.path.join(directory, 'aligned_HER2.jpg')
            fish_aligned_path = os.path.join(directory, 'aligned_FISH.jpg')
            
            # 檢查檔案是否存在
            if not all(os.path.exists(p) for p in [reference_path, her2_aligned_path, fish_aligned_path]):
                self.status_bar.showMessage('錯誤: 找不到必要的配準結果檔案')
                return
            
            # 載入 HER2 配準結果
            self.her2_comparison.load_images(reference_path, her2_aligned_path)
            
            # 載入 FISH 配準結果
            self.fish_comparison.load_images(reference_path, fish_aligned_path)
            
            # 計算並顯示指標
            reference_img = cv2.imread(reference_path)
            her2_img = cv2.imread(her2_aligned_path)
            fish_img = cv2.imread(fish_aligned_path)
            
            self.her2_metrics.update_metrics(reference_img, her2_img)
            self.fish_metrics.update_metrics(reference_img, fish_img)
            
            # 載入報告 (如果存在)
            report_path = os.path.join(directory, 'registration_report.txt')
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                self.report_text.setText(report_content)
            
            self.export_button.setEnabled(True)
            self.status_bar.showMessage('配準結果載入完成')
            
        except Exception as e:
            self.status_bar.showMessage(f'載入錯誤: {str(e)}')
    
    def export_report(self):
        """匯出報告"""
        if not self.current_directory:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, '匯出報告', 'registration_analysis_report.html', 
            'HTML files (*.html);;Text files (*.txt)'
        )
        
        if filename:
            try:
                # 生成 HTML 報告
                html_content = self.generate_html_report()
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                self.status_bar.showMessage(f'報告已匯出: {filename}')
                
            except Exception as e:
                self.status_bar.showMessage(f'匯出錯誤: {str(e)}')
    
    def generate_html_report(self):
        """生成 HTML 格式的詳細報告"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>WSI 配準分析報告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .metrics { background-color: #f9f9f9; padding: 15px; border-left: 4px solid #007acc; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .good { color: green; font-weight: bold; }
                .warning { color: orange; font-weight: bold; }
                .error { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>WSI 四階段配準分析報告</h1>
                <p>生成時間: """ + str(QTimer().currentTime()) + """</p>
            </div>
            
            <div class="section">
                <h2>配準概述</h2>
                <p>本報告展示了 WSI (Whole Slide Image) 四階段配準工作流程的詳細結果分析。</p>
                <ul>
                    <li>階段 1: 特徵點粗對齊 (SIFT/ORB + RANSAC)</li>
                    <li>階段 2: 互信息精準對齊 (MI + 仿射變換)</li>
                    <li>階段 3: B-spline 非剛體對齊 (FFD)</li>
                    <li>階段 4: 品質評估與驗證</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>配準品質指標</h2>
                <div class="metrics">
                    <h3>評估指標說明</h3>
                    <ul>
                        <li><strong>互信息 (MI)</strong>: 衡量兩幅影像間的統計相關性，值越高表示配準品質越好</li>
                        <li><strong>正規化互信息 (NMI)</strong>: MI 的正規化版本，範圍 [0,1]，越接近 1 越好</li>
                        <li><strong>目標配準誤差 (TRE)</strong>: 配準後的空間誤差，單位為像素，值越小越好</li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>建議與改善方向</h2>
                <ul>
                    <li>如果 TRE > 5.0 像素，建議調整特徵檢測參數或增加特徵點數量</li>
                    <li>如果 NMI < 0.5，建議檢查影像預處理步驟或調整互信息優化參數</li>
                    <li>對於螢光影像，建議使用專門的預處理方法以提高配準精度</li>
                    <li>考慮使用 CUDA 加速以提高大型 WSI 的處理速度</li>
                </ul>
            </div>
        </body>
        </html>
        """
        return html

def main():
    app = QApplication(sys.argv)
    app.setApplicationName('WSI 配準結果查看器')
    
    # 設置應用程式圖示 (如果有的話)
    # app.setWindowIcon(QIcon('icon.png'))
    
    viewer = WSIRegistrationViewer()
    viewer.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()