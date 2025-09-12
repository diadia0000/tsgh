#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Viewer Component
Handles image display with zoom, pan, rotation capabilities
"""

import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QSlider, QPushButton, QCheckBox, QComboBox)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QTransform, QPainter

class ImageViewer(QWidget):
    """Image viewer with basic operations"""
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.reference_image = None
        self.compare_mode = False
        self.zoom_factor = 1.0
        self.rotation_angle = 0
        self.pan_offset = QPoint(0, 0)
        self.initUI()
        
    def initUI(self):
        """Initialize user interface"""
        layout = QVBoxLayout()
        
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Image display area
        self.image_label = QLabel("請選擇影像檔案 - Please select an image file")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                border-radius: 5px;
                background-color: #f9f9f9;
                color: #666666;
                font-size: 14px;
            }
        """)
        
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        
    def create_control_panel(self):
        """Create image control panel"""
        panel = QWidget()
        layout = QHBoxLayout()
        
        # Zoom controls
        layout.addWidget(QLabel("縮放:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 500)  # 10% to 500%
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        layout.addWidget(self.zoom_slider)
        
        self.zoom_label = QLabel("100%")
        layout.addWidget(self.zoom_label)
        
        # Reset button
        reset_btn = QPushButton("重置 - Reset")
        reset_btn.clicked.connect(self.reset_view)
        layout.addWidget(reset_btn)
        
        # Rotation controls
        layout.addWidget(QLabel("旋轉:"))
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.valueChanged.connect(self.on_rotation_changed)
        layout.addWidget(self.rotation_slider)
        
        self.rotation_label = QLabel("0°")
        layout.addWidget(self.rotation_label)
        
        # Overlay controls
        self.overlay_checkbox = QCheckBox("疊合顯示")
        self.overlay_checkbox.stateChanged.connect(self.on_overlay_changed)
        layout.addWidget(self.overlay_checkbox)
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        self.opacity_slider.setEnabled(False)
        layout.addWidget(self.opacity_slider)
        
        layout.addStretch()
        panel.setLayout(layout)
        
        return panel
        
    def load_image(self, image_path):
        """Load image from file path"""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                self.image_label.setText(f"檔案不存在: {image_path.name}")
                return False
                
            # Load image using OpenCV
            self.current_image = cv2.imread(str(image_path))
            if self.current_image is None:
                self.image_label.setText(f"無法載入影像: {image_path.name}")
                return False
                
            # Convert BGR to RGB
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # Load reference image if this is not HE
            if "HE" not in image_path.name:
                he_path = image_path.parent / "aligned_HE.tiff"
                if he_path.exists():
                    self.reference_image = cv2.imread(str(he_path))
                    if self.reference_image is not None:
                        self.reference_image = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
            
            self.reset_view()
            self.update_display()
            return True
            
        except Exception as e:
            self.image_label.setText(f"載入錯誤: {str(e)}")
            return False
            
    def update_display(self):
        """Update image display"""
        if self.current_image is None:
            return
            
        # Get current image to display
        display_image = self.current_image.copy()
        
        # Apply overlay if enabled
        if (self.overlay_checkbox.isChecked() and 
            self.reference_image is not None and 
            self.current_image.shape == self.reference_image.shape):
            
            alpha = self.opacity_slider.value() / 100.0
            display_image = cv2.addWeighted(
                self.reference_image, 1 - alpha,
                self.current_image, alpha, 0
            )
        
        # Convert to QImage
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Apply transformations
        transform = QTransform()
        transform.scale(self.zoom_factor, self.zoom_factor)
        transform.rotate(self.rotation_angle)
        
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
        
        # Apply pan offset
        if not self.pan_offset.isNull():
            # Create a larger pixmap for panning
            larger_pixmap = QPixmap(pixmap.size() + QSize(200, 200))
            larger_pixmap.fill(Qt.white)
            
            painter = QPainter(larger_pixmap)
            painter.drawPixmap(self.pan_offset + QPoint(100, 100), pixmap)
            painter.end()
            
            pixmap = larger_pixmap
        
        # Scale to fit label if too large
        label_size = self.image_label.size()
        if pixmap.width() > label_size.width() or pixmap.height() > label_size.height():
            pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.image_label.setPixmap(pixmap)
        
    def on_zoom_changed(self, value):
        """Handle zoom slider change"""
        self.zoom_factor = value / 100.0
        self.zoom_label.setText(f"{value}%")
        self.update_display()
        
    def on_rotation_changed(self, value):
        """Handle rotation slider change"""
        self.rotation_angle = value
        self.rotation_label.setText(f"{value}°")
        self.update_display()
        
    def on_overlay_changed(self, state):
        """Handle overlay checkbox change"""
        self.opacity_slider.setEnabled(state == Qt.Checked)
        self.update_display()
        
    def on_opacity_changed(self, value):
        """Handle opacity slider change"""
        self.update_display()
        
    def reset_view(self):
        """Reset view to default"""
        self.zoom_factor = 1.0
        self.rotation_angle = 0
        self.pan_offset = QPoint(0, 0)
        
        self.zoom_slider.setValue(100)
        self.rotation_slider.setValue(0)
        
        self.update_display()
        
    def toggle_compare_mode(self):
        """Toggle comparison mode"""
        self.compare_mode = not self.compare_mode
        
        if self.compare_mode and self.reference_image is not None:
            # Show side-by-side comparison
            self.show_comparison()
        else:
            # Show single image
            self.update_display()
            
    def show_comparison(self):
        """Show side-by-side comparison"""
        if self.current_image is None or self.reference_image is None:
            return
            
        # Resize images to same height
        h1, w1 = self.current_image.shape[:2]
        h2, w2 = self.reference_image.shape[:2]
        
        target_height = min(h1, h2, 400)  # Limit height for display
        
        # Resize current image
        scale1 = target_height / h1
        new_w1 = int(w1 * scale1)
        img1_resized = cv2.resize(self.current_image, (new_w1, target_height))
        
        # Resize reference image
        scale2 = target_height / h2
        new_w2 = int(w2 * scale2)
        img2_resized = cv2.resize(self.reference_image, (new_w2, target_height))
        
        # Concatenate horizontally
        comparison_image = np.hstack([img2_resized, img1_resized])
        
        # Convert to QImage and display
        height, width, channel = comparison_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(comparison_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit label
        label_size = self.image_label.size()
        if pixmap.width() > label_size.width() or pixmap.height() > label_size.height():
            pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.image_label.setPixmap(pixmap)
        
    def mousePressEvent(self, event):
        """Handle mouse press for panning"""
        if event.button() == Qt.LeftButton:
            self.last_pan_point = event.pos()
            
    def mouseMoveEvent(self, event):
        """Handle mouse move for panning"""
        if hasattr(self, 'last_pan_point') and event.buttons() == Qt.LeftButton:
            delta = event.pos() - self.last_pan_point
            self.pan_offset += delta
            self.last_pan_point = event.pos()
            self.update_display()
            
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        delta = event.angleDelta().y()
        if delta > 0:
            # Zoom in
            new_value = min(self.zoom_slider.value() + 10, self.zoom_slider.maximum())
        else:
            # Zoom out
            new_value = max(self.zoom_slider.value() - 10, self.zoom_slider.minimum())
            
        self.zoom_slider.setValue(new_value)