#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics Panel Component
Reads and displays C++ registration evaluation results
"""

import json
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QTextEdit, QGroupBox, QProgressBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

class MetricsPanel(QWidget):
    """Panel for displaying registration evaluation metrics"""
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        """Initialize user interface"""
        layout = QVBoxLayout()
        
        # Metrics display group
        metrics_group = QGroupBox("評估指標 - Evaluation Metrics")
        metrics_layout = QVBoxLayout()
        
        # MI (Mutual Information)
        self.mi_layout = self.create_metric_row("互信息 (MI):", "0.000")
        metrics_layout.addLayout(self.mi_layout)
        
        # NMI (Normalized Mutual Information)
        self.nmi_layout = self.create_metric_row("正規化互信息 (NMI):", "0.000")
        metrics_layout.addLayout(self.nmi_layout)
        
        # TRE (Target Registration Error)
        self.tre_layout = self.create_metric_row("目標配準誤差 (TRE):", "0.00 px")
        metrics_layout.addLayout(self.tre_layout)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Quality assessment group
        quality_group = QGroupBox("品質評估 - Quality Assessment")
        quality_layout = QVBoxLayout()
        
        self.quality_label = QLabel("等待評估... - Waiting for evaluation...")
        self.quality_label.setAlignment(Qt.AlignCenter)
        self.quality_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                border: 2px solid #cccccc;
                border-radius: 5px;
                background-color: #f0f0f0;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        
        quality_layout.addWidget(self.quality_label)
        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)
        
        # Progress bars for visual representation
        progress_group = QGroupBox("指標視覺化 - Metrics Visualization")
        progress_layout = QVBoxLayout()
        
        # MI progress bar
        mi_progress_layout = QHBoxLayout()
        mi_progress_layout.addWidget(QLabel("MI:"))
        self.mi_progress = QProgressBar()
        self.mi_progress.setRange(0, 200)  # 0 to 2.0
        self.mi_progress.setValue(0)
        mi_progress_layout.addWidget(self.mi_progress)
        progress_layout.addLayout(mi_progress_layout)
        
        # NMI progress bar
        nmi_progress_layout = QHBoxLayout()
        nmi_progress_layout.addWidget(QLabel("NMI:"))
        self.nmi_progress = QProgressBar()
        self.nmi_progress.setRange(0, 100)  # 0 to 1.0
        self.nmi_progress.setValue(0)
        nmi_progress_layout.addWidget(self.nmi_progress)
        progress_layout.addLayout(nmi_progress_layout)
        
        # TRE progress bar (inverted - lower is better)
        tre_progress_layout = QHBoxLayout()
        tre_progress_layout.addWidget(QLabel("TRE:"))
        self.tre_progress = QProgressBar()
        self.tre_progress.setRange(0, 100)  # 0 to 10.0 px (inverted)
        self.tre_progress.setValue(0)
        tre_progress_layout.addWidget(self.tre_progress)
        progress_layout.addLayout(tre_progress_layout)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Detailed report
        report_group = QGroupBox("詳細報告 - Detailed Report")
        report_layout = QVBoxLayout()
        
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setMaximumHeight(200)
        self.report_text.setPlainText("等待載入配準結果... - Waiting for registration results...")
        
        report_layout.addWidget(self.report_text)
        report_group.setLayout(report_layout)
        layout.addWidget(report_group)
        
        self.setLayout(layout)
        
    def create_metric_row(self, label_text, initial_value):
        """Create a metric display row"""
        layout = QHBoxLayout()
        
        label = QLabel(label_text)
        label.setMinimumWidth(150)
        
        value_label = QLabel(initial_value)
        value_label.setAlignment(Qt.AlignRight)
        value_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        
        layout.addWidget(label)
        layout.addWidget(value_label)
        
        # Store reference to value label for updates
        if "MI" in label_text and "NMI" not in label_text:
            self.mi_value_label = value_label
        elif "NMI" in label_text:
            self.nmi_value_label = value_label
        elif "TRE" in label_text:
            self.tre_value_label = value_label
            
        return layout
        
    def update_metrics(self, reference_path, target_path, metrics_path=None):
        """Update metrics display"""
        try:
            # Try to load from JSON file first
            if metrics_path and Path(metrics_path).exists():
                self.load_metrics_from_json(metrics_path)
            else:
                # Calculate metrics from images
                self.calculate_metrics_from_images(reference_path, target_path)
                
        except Exception as e:
            self.report_text.setPlainText(f"載入指標時發生錯誤: {str(e)}")
            
    def load_metrics_from_json(self, json_path):
        """Load metrics from JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract metrics
            mi = data.get('mutual_information', 0.0)
            nmi = data.get('normalized_mutual_information', 0.0)
            tre = data.get('target_registration_error', 0.0)
            
            self.update_metric_displays(mi, nmi, tre)
            
            # Update detailed report
            report = self.generate_detailed_report(data)
            self.report_text.setPlainText(report)
            
        except Exception as e:
            self.report_text.setPlainText(f"JSON 載入錯誤: {str(e)}")
            
    def calculate_metrics_from_images(self, reference_path, target_path):
        """Calculate metrics from image files"""
        try:
            # Load images
            ref_img = cv2.imread(reference_path)
            target_img = cv2.imread(target_path)
            
            if ref_img is None or target_img is None:
                raise ValueError("無法載入影像檔案")
                
            # Calculate metrics
            mi = self.calculate_mutual_information(ref_img, target_img)
            nmi = self.calculate_normalized_mutual_information(ref_img, target_img)
            tre = self.calculate_target_registration_error(ref_img, target_img)
            
            self.update_metric_displays(mi, nmi, tre)
            
            # Generate simple report
            report = f"""
配準品質評估報告
================

基準影像: {Path(reference_path).name}
目標影像: {Path(target_path).name}

評估指標:
- 互信息 (MI): {mi:.6f}
- 正規化互信息 (NMI): {nmi:.6f}
- 目標配準誤差 (TRE): {tre:.2f} 像素

品質評估: {self.assess_quality(mi, nmi, tre)}

說明:
- MI 值越高表示影像間相關性越強
- NMI 範圍 [0,1]，越接近 1 越好
- TRE 值越小表示配準誤差越小
"""
            self.report_text.setPlainText(report)
            
        except Exception as e:
            self.report_text.setPlainText(f"計算指標錯誤: {str(e)}")
            
    def update_metric_displays(self, mi, nmi, tre):
        """Update metric display values"""
        # Update text labels
        self.mi_value_label.setText(f"{mi:.6f}")
        self.nmi_value_label.setText(f"{nmi:.6f}")
        self.tre_value_label.setText(f"{tre:.2f} px")
        
        # Update progress bars
        self.mi_progress.setValue(int(mi * 100))  # Scale to 0-200
        self.nmi_progress.setValue(int(nmi * 100))  # Scale to 0-100
        self.tre_progress.setValue(max(0, 100 - int(tre * 10)))  # Inverted scale
        
        # Update progress bar colors based on quality
        self.update_progress_bar_colors(mi, nmi, tre)
        
        # Update quality assessment
        quality = self.assess_quality(mi, nmi, tre)
        self.update_quality_display(quality)
        
    def update_progress_bar_colors(self, mi, nmi, tre):
        """Update progress bar colors based on quality thresholds"""
        # MI color (good > 1.5, normal > 0.8)
        if mi > 1.5:
            mi_color = "green"
        elif mi > 0.8:
            mi_color = "orange"
        else:
            mi_color = "red"
            
        # NMI color (good > 0.7, normal > 0.5)
        if nmi > 0.7:
            nmi_color = "green"
        elif nmi > 0.5:
            nmi_color = "orange"
        else:
            nmi_color = "red"
            
        # TRE color (good < 2.0, normal < 5.0)
        if tre < 2.0:
            tre_color = "green"
        elif tre < 5.0:
            tre_color = "orange"
        else:
            tre_color = "red"
            
        # Apply styles
        self.mi_progress.setStyleSheet(f"QProgressBar::chunk {{ background-color: {mi_color}; }}")
        self.nmi_progress.setStyleSheet(f"QProgressBar::chunk {{ background-color: {nmi_color}; }}")
        self.tre_progress.setStyleSheet(f"QProgressBar::chunk {{ background-color: {tre_color}; }}")
        
    def assess_quality(self, mi, nmi, tre):
        """Assess overall registration quality"""
        if mi > 1.5 and nmi > 0.7 and tre < 2.0:
            return "優秀 (Good)"
        elif mi > 0.8 and nmi > 0.5 and tre < 5.0:
            return "良好 (Normal)"
        else:
            return "需改善 (Bad)"
            
    def update_quality_display(self, quality):
        """Update quality assessment display"""
        self.quality_label.setText(quality)
        
        if "優秀" in quality:
            color = "#4CAF50"  # Green
        elif "良好" in quality:
            color = "#FF9800"  # Orange
        else:
            color = "#F44336"  # Red
            
        self.quality_label.setStyleSheet(f"""
            QLabel {{
                padding: 10px;
                border: 2px solid {color};
                border-radius: 5px;
                background-color: {color}20;
                color: {color};
                font-weight: bold;
                font-size: 14px;
            }}
        """)
        
    def calculate_mutual_information(self, img1, img2, bins=64):
        """Calculate mutual information between two images"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
        # Convert to grayscale
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
        # Calculate joint histogram
        hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
        
        # Normalize
        hist_2d = hist_2d / np.sum(hist_2d)
        
        # Calculate marginal histograms
        hist_1 = np.sum(hist_2d, axis=1)
        hist_2 = np.sum(hist_2d, axis=0)
        
        # Calculate mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if hist_2d[i, j] > 1e-10:
                    if hist_1[i] > 1e-10 and hist_2[j] > 1e-10:
                        mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (hist_1[i] * hist_2[j]))
                        
        return mi
        
    def calculate_normalized_mutual_information(self, img1, img2, bins=64):
        """Calculate normalized mutual information"""
        mi = self.calculate_mutual_information(img1, img2, bins)
        h1 = self.calculate_entropy(img1, bins)
        h2 = self.calculate_entropy(img2, bins)
        
        if h1 + h2 == 0:
            return 0.0
            
        return 2.0 * mi / (h1 + h2)
        
    def calculate_entropy(self, img, bins=256):
        """Calculate image entropy"""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        hist, _ = np.histogram(img, bins=bins, range=(0, 256))
        hist = hist / np.sum(hist)
        
        entropy = 0.0
        for p in hist:
            if p > 1e-10:
                entropy -= p * np.log2(p)
                
        return entropy
        
    def calculate_target_registration_error(self, img1, img2):
        """Calculate target registration error (simplified version)"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
        # Convert to grayscale
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
        # Calculate MSE as approximation of TRE
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        return np.sqrt(mse) / 10.0  # Scale to reasonable range
        
    def generate_detailed_report(self, data):
        """Generate detailed report from JSON data"""
        report = f"""
配準品質評估詳細報告
==================

處理時間: {data.get('processing_time', 'N/A')}
配準方法: {data.get('registration_method', 'Multi-stage Registration')}

階段 1 - 特徵點粗對齊:
- 檢測到的特徵點數: {data.get('feature_points', 'N/A')}
- 匹配點數: {data.get('matched_points', 'N/A')}
- RANSAC 內點數: {data.get('inlier_points', 'N/A')}

階段 2 - 互信息精準對齊:
- 迭代次數: {data.get('mi_iterations', 'N/A')}
- 收斂狀態: {data.get('mi_converged', 'N/A')}

階段 3 - B-spline 非剛體對齊:
- 網格大小: {data.get('bspline_grid_size', 'N/A')}
- 迭代次數: {data.get('bspline_iterations', 'N/A')}

評估指標:
- 互信息 (MI): {data.get('mutual_information', 0.0):.6f}
- 正規化互信息 (NMI): {data.get('normalized_mutual_information', 0.0):.6f}
- 目標配準誤差 (TRE): {data.get('target_registration_error', 0.0):.2f} 像素

品質等級: {self.assess_quality(
    data.get('mutual_information', 0.0),
    data.get('normalized_mutual_information', 0.0),
    data.get('target_registration_error', 0.0)
)}

建議:
{data.get('recommendations', '無特殊建議')}
"""
        return report