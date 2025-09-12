#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3: PyQt5 GUI Main Application
Cell Image Registration System GUI
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QSplitter, QListWidget, QLabel, 
                            QPushButton, QTextEdit, QGroupBox, QListWidgetItem)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon

from viewer import ImageViewer
from metrics import MetricsPanel

class CellRegistrationGUI(QMainWindow):
    """Cell Image Registration System Main GUI"""
    
    def __init__(self):
        super().__init__()
        self.output_dir = Path("picture/output")
        self.initUI()
        self.refresh_file_list()
        
    def initUI(self):
        """Initialize user interface"""
        self.setWindowTitle('細胞影像對位系統 - Cell Image Registration System')
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout - horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - File list
        left_panel = self.create_file_panel()
        left_panel.setMaximumWidth(300)
        
        # Center panel - Image display
        center_panel = self.create_image_panel()
        
        # Right panel - Results info
        right_panel = self.create_results_panel()
        right_panel.setMaximumWidth(350)
        
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(center_panel)
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setStretchFactor(0, 0)  # Fixed width
        main_splitter.setStretchFactor(1, 1)  # Expandable
        main_splitter.setStretchFactor(2, 0)  # Fixed width
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(main_splitter)
        central_widget.setLayout(main_layout)
        
        # Status bar
        self.statusBar().showMessage('準備就緒 - Ready')
        
    def create_file_panel(self):
        """Create left file list panel"""
        panel = QGroupBox("檔案清單 - File List")
        layout = QVBoxLayout()
        
        # Refresh button
        self.refresh_btn = QPushButton("重新整理 - Refresh")
        self.refresh_btn.clicked.connect(self.refresh_file_list)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        
        layout.addWidget(self.refresh_btn)
        layout.addWidget(self.file_list)
        panel.setLayout(layout)
        
        return panel
        
    def create_image_panel(self):
        """Create center image display panel"""
        panel = QGroupBox("影像顯示區 - Image Display")
        layout = QVBoxLayout()
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.compare_btn = QPushButton("比較模式 - Compare Mode")
        self.compare_btn.clicked.connect(self.toggle_compare_mode)
        
        self.triple_btn = QPushButton("三圖合成 - Triple Overlay")
        self.triple_btn.clicked.connect(self.show_triple_overlay)
        
        control_layout.addWidget(self.compare_btn)
        control_layout.addWidget(self.triple_btn)
        control_layout.addStretch()
        
        # Image viewer
        self.image_viewer = ImageViewer()
        
        layout.addLayout(control_layout)
        layout.addWidget(self.image_viewer)
        panel.setLayout(layout)
        
        return panel
        
    def create_results_panel(self):
        """Create right results info panel"""
        panel = QGroupBox("結果資訊 - Results Info")
        layout = QVBoxLayout()
        
        # Metrics panel
        self.metrics_panel = MetricsPanel()
        
        layout.addWidget(self.metrics_panel)
        panel.setLayout(layout)
        
        return panel
        
    def refresh_file_list(self):
        """Refresh file list from output directory"""
        self.file_list.clear()
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.statusBar().showMessage("輸出目錄不存在，已創建 - Output directory created")
            return
            
        # Expected output files
        expected_files = [
            ("aligned_HE.tiff", "HE 基準影像 - HE Reference"),
            ("aligned_Her2.tiff", "Her2 對齊影像 - Her2 Aligned"),
            ("aligned_DISH.tiff", "DISH 對齊影像 - DISH Aligned"),
            ("overlay_triple.tiff", "三圖疊合 - Triple Overlay")
        ]
        
        found_files = 0
        for filename, display_name in expected_files:
            file_path = self.output_dir / filename
            item = QListWidgetItem(display_name)
            
            if file_path.exists():
                item.setData(Qt.UserRole, str(file_path))
                item.setToolTip(f"檔案路徑: {file_path}")
                found_files += 1
            else:
                item.setData(Qt.UserRole, None)
                item.setToolTip("檔案不存在 - File not found")
                item.setForeground(Qt.gray)
                
            self.file_list.addItem(item)
            
        self.statusBar().showMessage(f"找到 {found_files}/4 個檔案 - Found {found_files}/4 files")
        
    def on_file_selected(self, item):
        """Handle file selection"""
        file_path = item.data(Qt.UserRole)
        
        if file_path is None:
            self.statusBar().showMessage("檔案不存在 - File not found")
            return
            
        # Load image in viewer
        self.image_viewer.load_image(file_path)
        
        # Update metrics if it's an aligned image
        if "aligned" in Path(file_path).name:
            self.update_metrics_for_file(file_path)
            
        self.statusBar().showMessage(f"已載入: {Path(file_path).name}")
        
    def update_metrics_for_file(self, file_path):
        """Update metrics panel for selected file"""
        file_path = Path(file_path)
        
        # Load reference HE image
        he_path = self.output_dir / "aligned_HE.tiff"
        if not he_path.exists():
            return
            
        # Load metrics from JSON if available
        metrics_path = self.output_dir / "registration_metrics.json"
        
        self.metrics_panel.update_metrics(str(he_path), str(file_path), str(metrics_path))
        
    def toggle_compare_mode(self):
        """Toggle comparison mode"""
        self.image_viewer.toggle_compare_mode()
        
        if self.image_viewer.compare_mode:
            self.compare_btn.setText("退出比較 - Exit Compare")
        else:
            self.compare_btn.setText("比較模式 - Compare Mode")
            
    def show_triple_overlay(self):
        """Show triple overlay image"""
        overlay_path = self.output_dir / "overlay_triple.tiff"
        
        if overlay_path.exists():
            self.image_viewer.load_image(str(overlay_path))
            self.statusBar().showMessage("顯示三圖疊合 - Showing triple overlay")
        else:
            self.statusBar().showMessage("三圖疊合檔案不存在 - Triple overlay file not found")

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName('Cell Image Registration System')
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = CellRegistrationGUI()
    window.show()
    
    # Auto-refresh timer (optional)
    timer = QTimer()
    timer.timeout.connect(window.refresh_file_list)
    timer.start(30000)  # Refresh every 30 seconds
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()