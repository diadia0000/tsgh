#!/usr/bin/env python3
"""
CZI 影像對齊工具 (專用版本)

功能:
1. 專門處理 CZI 檔案格式
2. 使用金字塔最上層 (scale_factor=0.0625) 載入縮圖避免記憶體問題
3. 使用互訊息 (Mutual Information, MI) 作為圖像對齊品質評估指標
4. 提供視覺化對齊介面

記憶體管理策略:
- 只載入最低解析度層級 (約 6.25% 原始大小)
- 避免載入完整影像數據
"""

import sys
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import mutual_info_score
from aicspylibczi import CziFile
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QSlider, QLabel, QDoubleSpinBox, QGridLayout, QCheckBox, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage, QTransform
from PyQt6.QtCore import Qt
from skimage.transform import warp, AffineTransform
from skimage.color import rgb2gray
import tifffile

def read_czi_thumbnail(path, scale_factor=0.0625):
    """
    讀取 CZI 檔案縮圖

    Args:
        path: CZI 檔案路徑
        scale_factor: 縮放因子，預設使用最上層 (0.0625)

    Returns:
        numpy.ndarray: 縮圖影像數據 (BGR24 format)
    """
    print(f"Loading CZI thumbnail: {path}")
    print(f"Using scale factor: {scale_factor} (約 {scale_factor*100:.1f}% 解析度)")

    try:
        czi = CziFile(path)

        # 檢查是否為馬賽克影像
        if czi.is_mosaic():
            print("  - 檢測到馬賽克影像，讀取整體縮圖...")
            # 對馬賽克影像，使用邊界框讀取整體縮圖
            mosaic_bbox = czi.get_mosaic_bounding_box()
            img_data = czi.read_mosaic(
                region=(mosaic_bbox.x, mosaic_bbox.y, mosaic_bbox.w, mosaic_bbox.h),
                scale_factor=scale_factor,
                C=0  # 讀取第一個通道
            )
        else:
            print("  - 檢測到標準影像，讀取縮圖...")
            # 對標準影像，直接讀取縮放版本
            img_data, _ = czi.read_image(S=0, scale_factor=scale_factor)

        if img_data is None or img_data.size == 0:
            raise ValueError("無法讀取影像數據")

        # 處理影像維度
        img_data = img_data.squeeze()
        print(f"  - 原始數據形狀: {img_data.shape}, 數據類型: {img_data.dtype}")

        # 確保是 BGR24 格式 (高度, 寬度, 3) - 符合 CZI 原生格式
        if len(img_data.shape) == 2:
            # 灰階影像，轉換為 BGR24
            img_data = np.stack([img_data] * 3, axis=-1)
        elif len(img_data.shape) == 3:
            if img_data.shape[2] == 1:
                # 單通道，重複為 BGR24
                img_data = np.repeat(img_data, 3, axis=2)
            elif img_data.shape[2] > 3:
                # 多通道，只取前3個 (保持 BGR 順序)
                img_data = img_data[:, :, :3]
            elif img_data.shape[2] == 3:
                # 已經是3通道，CZI 格式通常是 BGR，保持原序
                pass
        else:
            raise ValueError(f"不支援的影像維度: {img_data.shape}")

        # 數據類型轉換
        if img_data.dtype == np.uint16:
            # 16位轉8位
            img_data = (img_data / 256).astype(np.uint8)
        elif img_data.dtype != np.uint8:
            # 其他類型正規化到 0-255
            img_min, img_max = img_data.min(), img_data.max()
            if img_max > img_min:
                img_data = ((img_data - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_data = np.zeros_like(img_data, dtype=np.uint8)

        print(f"  - 處理後形狀: {img_data.shape}, 數據類型: {img_data.dtype}")
        print(f"  - 成功載入 BGR24 縮圖: {img_data.shape[1]}x{img_data.shape[0]} pixels")

        return img_data

    except Exception as e:
        print(f"Error loading CZI file {path}: {e}")
        raise

def calculate_mutual_information(img1, img2):
    """
    計算兩個圖像之間的互訊息 (Mutual Information)

    Args:
        img1, img2: 兩個numpy陣列，形狀相同

    Returns:
        float: 互訊息值，數值越大表示兩個圖像越相關
    """
    try:
        # 轉為灰階
        if len(img1.shape) == 3:
            img1_gray = rgb2gray(img1)
        else:
            img1_gray = img1

        if len(img2.shape) == 3:
            img2_gray = rgb2gray(img2)
        else:
            img2_gray = img2

        # 確保數據範圍一致
        img1_norm = ((img1_gray - img1_gray.min()) / (img1_gray.max() - img1_gray.min()) * 255).astype(np.uint8)
        img2_norm = ((img2_gray - img2_gray.min()) / (img2_gray.max() - img2_gray.min()) * 255).astype(np.uint8)

        # 平坦化用於計算互訊息
        img1_flat = img1_norm.flatten()
        img2_flat = img2_norm.flatten()

        # 計算互訊息
        mi_score = mutual_info_score(img1_flat, img2_flat)
        return mi_score

    except Exception as e:
        print(f"計算互訊息時發生錯誤: {e}")
        return 0.0

class ImageControl(QWidget):
    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name
        self.filePath = None
        self.pixmapItem = None

        self.initUI()

    def initUI(self):
        layout = QGridLayout(self)
        self.setLayout(layout)

        self.loadButton = QPushButton(f"Load {self.name} (CZI only)")
        layout.addWidget(self.loadButton, 0, 0, 1, 2)

        layout.addWidget(QLabel("Opacity"), 1, 0)
        self.opacitySlider = QSlider(Qt.Orientation.Horizontal)
        self.opacitySlider.setRange(0, 100)
        self.opacitySlider.setValue(100)
        layout.addWidget(self.opacitySlider, 1, 1)

        layout.addWidget(QLabel("Scale"), 2, 0)
        self.scaleSpinBox = QDoubleSpinBox()
        self.scaleSpinBox.setRange(0.1, 5.0)
        self.scaleSpinBox.setSingleStep(0.05)
        self.scaleSpinBox.setValue(1.0)
        layout.addWidget(self.scaleSpinBox, 2, 1)

        layout.addWidget(QLabel("Rotation"), 3, 0)
        self.rotationSpinBox = QDoubleSpinBox()
        self.rotationSpinBox.setRange(-360, 360)
        self.rotationSpinBox.setSingleStep(1)
        self.rotationSpinBox.setValue(0)
        layout.addWidget(self.rotationSpinBox, 3, 1)

        layout.addWidget(QLabel("Translate X"), 4, 0)
        self.translateXSpinBox = QDoubleSpinBox()
        self.translateXSpinBox.setRange(-5000, 5000)
        self.translateXSpinBox.setSingleStep(10)
        self.translateXSpinBox.setValue(0)
        layout.addWidget(self.translateXSpinBox, 4, 1)

        layout.addWidget(QLabel("Translate Y"), 5, 0)
        self.translateYSpinBox = QDoubleSpinBox()
        self.translateYSpinBox.setRange(-5000, 5000)
        self.translateYSpinBox.setSingleStep(10)
        self.translateYSpinBox.setValue(0)
        layout.addWidget(self.translateYSpinBox, 5, 1)

        # 翻轉控制
        self.flipHorizontalCheckbox = QCheckBox("Flip Horizontal")
        self.flipVerticalCheckbox = QCheckBox("Flip Vertical")
        layout.addWidget(self.flipHorizontalCheckbox, 6, 0, 1, 2)
        layout.addWidget(self.flipVerticalCheckbox, 7, 0, 1, 2)

    def getTransform(self):
        return {
            'scale': self.scaleSpinBox.value(),
            'rotation': self.rotationSpinBox.value(),
            'translation_x': self.translateXSpinBox.value(),
            'translation_y': self.translateYSpinBox.value(),
            'flip_h': self.flipHorizontalCheckbox.isChecked(),
            'flip_v': self.flipVerticalCheckbox.isChecked()
        }

    def setTransform(self, transform_data):
        """設置變換參數"""
        self.scaleSpinBox.setValue(transform_data.get('scale', 1.0))
        self.rotationSpinBox.setValue(transform_data.get('rotation', 0.0))
        self.translateXSpinBox.setValue(transform_data.get('translation_x', 0.0))
        self.translateYSpinBox.setValue(transform_data.get('translation_y', 0.0))
        self.flipHorizontalCheckbox.setChecked(transform_data.get('flip_h', False))
        self.flipVerticalCheckbox.setChecked(transform_data.get('flip_v', False))

    def setEnabledControls(self, enabled):
        self.opacitySlider.setEnabled(enabled)
        self.scaleSpinBox.setEnabled(enabled)
        self.rotationSpinBox.setEnabled(enabled)
        self.translateXSpinBox.setEnabled(enabled)
        self.translateYSpinBox.setEnabled(enabled)
        self.flipHorizontalCheckbox.setEnabled(enabled)
        self.flipVerticalCheckbox.setEnabled(enabled)

class AlignerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CZI Image Aligner - MI Analysis")
        self.setGeometry(100, 100, 1400, 900)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        # 防止自動調整大小和縮放
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)

        self.imageControls = [
            ImageControl("Reference Image"),
            ImageControl("Target Image 1"),
            ImageControl("Target Image 2")
        ]
        
        # 參考影像不允許變換，只允許調整透明度
        self.imageControls[0].setEnabledControls(False)
        self.imageControls[0].opacitySlider.setEnabled(True)

        self.initUI()
        self.connectSignals()

    def initUI(self):
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        mainLayout = QHBoxLayout(centralWidget)

        # 控制面板
        controlPanel = QWidget()
        controlPanel.setMaximumWidth(350)
        controlLayout = QVBoxLayout(controlPanel)
        
        # 添加標題
        titleLabel = QLabel("CZI Image Aligner")
        titleLabel.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        controlLayout.addWidget(titleLabel)

        for ctrl in self.imageControls:
            controlLayout.addWidget(ctrl)
            controlLayout.addStretch(1)

        # 分析和操作按鈕
        self.miLabel = QLabel("MI (Mutual Information): N/A")
        self.miLabel.setStyleSheet("font-weight: bold; color: blue;")

        self.calculateButton = QPushButton("Calculate MI & Create Composite")
        self.calculateButton.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")

        self.saveTransformButton = QPushButton("Save Alignment Parameters")
        self.loadTransformButton = QPushButton("Load Alignment Parameters")
        self.saveCompositeButton = QPushButton("Save Composite Image")
        
        # 信息標籤
        infoLabel = QLabel("提示：只支援 CZI 檔案\n使用縮圖模式節省記憶體\n互訊息越高表示對齊越好")
        infoLabel.setStyleSheet("color: gray; font-size: 10px; margin: 10px;")

        actionLayout = QVBoxLayout()
        actionLayout.addWidget(self.calculateButton)
        actionLayout.addWidget(self.miLabel)
        actionLayout.addWidget(self.saveTransformButton)
        actionLayout.addWidget(self.loadTransformButton)
        actionLayout.addWidget(self.saveCompositeButton)
        actionLayout.addStretch(1)
        actionLayout.addWidget(infoLabel)

        controlLayout.addLayout(actionLayout)

        # 主要顯示區域
        mainLayout.addWidget(self.view, 3)
        mainLayout.addWidget(controlPanel, 1)

    def connectSignals(self):
        for i, ctrl in enumerate(self.imageControls):
            ctrl.loadButton.clicked.connect(lambda _, idx=i: self.loadImage(idx))
            ctrl.opacitySlider.valueChanged.connect(self.updateTransforms)
            ctrl.scaleSpinBox.valueChanged.connect(self.updateTransforms)
            ctrl.rotationSpinBox.valueChanged.connect(self.updateTransforms)
            ctrl.translateXSpinBox.valueChanged.connect(self.updateTransforms)
            ctrl.translateYSpinBox.valueChanged.connect(self.updateTransforms)
            ctrl.flipHorizontalCheckbox.stateChanged.connect(self.updateTransforms)
            ctrl.flipVerticalCheckbox.stateChanged.connect(self.updateTransforms)

        self.calculateButton.clicked.connect(self.calculateMiAndComposite)
        self.saveTransformButton.clicked.connect(self.saveTransforms)
        self.loadTransformButton.clicked.connect(self.loadTransforms)
        self.saveCompositeButton.clicked.connect(self.saveCompositeImage)

    def loadImage(self, index):
        filePath, _ = QFileDialog.getOpenFileName(
            self,
            "Open CZI Image",
            "",
            "CZI Files (*.czi)"
        )

        if filePath:
            try:
                ctrl = self.imageControls[index]
                ctrl.filePath = filePath

                print(f"Loading CZI image {index + 1}: {filePath}")

                # 使用專用的 CZI 縮圖讀取函數
                img_data = read_czi_thumbnail(filePath, scale_factor=0.0625)

                if img_data is None or img_data.size == 0:
                    raise ValueError("Failed to load CZI image")

                # 確保圖片數據格式正確 (BGR24 -> RGB for Qt display)
                img_data = np.ascontiguousarray(img_data, dtype=np.uint8)
                height, width, channel = img_data.shape

                # 將 BGR 轉換為 RGB 用於 Qt 顯示
                img_data_rgb = img_data[:, :, ::-1]  # BGR -> RGB
                img_data_rgb = np.ascontiguousarray(img_data_rgb)  # 確保連續的記憶體佈局
                bytes_per_line = 3 * width

                # 使用 tobytes() 將 numpy 數組轉換為 bytes 對象
                q_img = QImage(img_data_rgb.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)

                if ctrl.pixmapItem:
                    self.scene.removeItem(ctrl.pixmapItem)

                ctrl.pixmapItem = QGraphicsPixmapItem(pixmap)
                ctrl.pixmapItem.setOpacity(ctrl.opacitySlider.value() / 100.0)
                ctrl.pixmapItem.setZValue(index)

                self.scene.addItem(ctrl.pixmapItem)
                self.updateTransforms()

                # 設置初始視圖
                if index == 0:
                    self.view.setSceneRect(ctrl.pixmapItem.boundingRect())
                    self.view.fitInView(ctrl.pixmapItem, Qt.AspectRatioMode.KeepAspectRatio)
                    ctrl.setEnabledControls(True)

                print(f"Successfully loaded CZI image {index + 1}: {width}x{height}")

                # 顯示成功訊息
                QMessageBox.information(self, "Success",
                    f"成功載入 CZI 縮圖\n尺寸: {width}x{height}\n"
                    f"解析度: 約 6.25% (節省記憶體模式)")

            except Exception as e:
                print(f"Error loading CZI image {filePath}: {e}")
                QMessageBox.critical(self, "Error", f"載入 CZI 檔案失敗:\n{str(e)}")

                if hasattr(ctrl, 'pixmapItem') and ctrl.pixmapItem:
                    self.scene.removeItem(ctrl.pixmapItem)
                    ctrl.pixmapItem = None
                ctrl.filePath = None

    def updateTransforms(self):
        for ctrl in self.imageControls:
            if ctrl.pixmapItem:
                transform = QTransform()
                t = ctrl.getTransform()
                
                center = ctrl.pixmapItem.boundingRect().center()
                
                # 確定翻轉縮放
                scale_h = -1 if t['flip_h'] else 1
                scale_v = -1 if t['flip_v'] else 1

                transform.translate(center.x(), center.y())
                transform.rotate(t['rotation'])
                transform.scale(t['scale'] * scale_h, t['scale'] * scale_v)
                transform.translate(-center.x(), -center.y())
                transform.translate(t['translation_x'], t['translation_y'])

                ctrl.pixmapItem.setTransform(transform)
                ctrl.pixmapItem.setOpacity(ctrl.opacitySlider.value() / 100.0)

        self.view.viewport().update()

    def getWarpedImage(self, image_index):
        """獲取變換後的圖像"""
        ctrl = self.imageControls[image_index]
        ref_ctrl = self.imageControls[0]

        if not ctrl.filePath or not ref_ctrl.filePath:
            return None

        try:
            img = read_czi_thumbnail(ctrl.filePath, scale_factor=0.0625)
            ref_img = read_czi_thumbnail(ref_ctrl.filePath, scale_factor=0.0625)

            t = ctrl.getTransform()

            center_x = img.shape[1] / 2
            center_y = img.shape[0] / 2

            # 確定翻轉縮放
            scale_h = -1 if t['flip_h'] else 1
            scale_v = -1 if t['flip_v'] else 1

            # 建立變換矩陣
            tf_center = AffineTransform(translation=(-center_x, -center_y))
            tf_rotate = AffineTransform(rotation=np.deg2rad(t['rotation']))
            tf_scale_and_flip = AffineTransform(scale=(t['scale'] * scale_h, t['scale'] * scale_v))
            tf_translate = AffineTransform(translation=(t['translation_x'], t['translation_y']))
            tf_decenter = AffineTransform(translation=(center_x, center_y))

            transform_matrix = tf_center + tf_rotate + tf_scale_and_flip + tf_decenter + tf_translate

            warped_img = warp(img, transform_matrix.inverse, output_shape=ref_img.shape, preserve_range=True)
            return warped_img.astype(img.dtype)

        except Exception as e:
            print(f"獲取變換圖像時發生錯誤: {e}")
            return None

    def calculateMiAndComposite(self):
        """計算互訊息並生成複合影像"""
        ref_ctrl = self.imageControls[0]
        if not ref_ctrl.filePath:
            self.miLabel.setText("MI: 請先載入參考影像")
            QMessageBox.warning(self, "Warning", "請先載入參考影像 (CZI 檔案)")
            return

        try:
            print("開始計算互訊息...")
            ref_img_color = read_czi_thumbnail(ref_ctrl.filePath, scale_factor=0.0625)
            print(f"參考影像載入: shape={ref_img_color.shape}, dtype={ref_img_color.dtype}")

            # 確保參考圖片是3通道
            if ref_img_color.ndim == 2:
                ref_img_color = np.stack((ref_img_color,)*3, axis=-1)
            elif ref_img_color.ndim == 3 and ref_img_color.shape[2] == 4:
                ref_img_color = ref_img_color[:,:,:3]

            total_mi = 0
            num_images = 0
            mi_scores = []

            # 初始化複合影像
            self.composite_image = np.zeros_like(ref_img_color, dtype=np.float64)
            valid_images = 1  # 包含參考影像
            self.composite_image += ref_img_color.astype(np.float64)

            # 處理其他影像
            for i in range(1, len(self.imageControls)):
                ctrl = self.imageControls[i]
                if ctrl.filePath:
                    print(f"處理影像 {i+1}...")
                    warped_img_color = self.getWarpedImage(i)
                    if warped_img_color is not None:
                        print(f"變換影像 {i+1}: shape={warped_img_color.shape}, dtype={warped_img_color.dtype}")

                        # 確保變換後圖片是3通道
                        if warped_img_color.ndim == 2:
                            warped_img_color = np.stack((warped_img_color,)*3, axis=-1)
                        elif warped_img_color.ndim == 3 and warped_img_color.shape[2] == 4:
                            warped_img_color = warped_img_color[:,:,:3]

                        if warped_img_color.shape != ref_img_color.shape:
                            print(f"警告: 形狀不匹配，跳過影像 {i+1}. Ref: {ref_img_color.shape}, Warped: {warped_img_color.shape}")
                            continue

                        # 計算互訊息
                        mi_score = calculate_mutual_information(ref_img_color, warped_img_color)
                        print(f"影像 {i+1} 的互訊息: {mi_score:.6f}")

                        total_mi += mi_score
                        num_images += 1
                        mi_scores.append(mi_score)

                        # 添加到複合影像
                        self.composite_image += warped_img_color.astype(np.float64)
                        valid_images += 1
                    else:
                        print(f"無法獲取變換影像 {i+1}")

            # 計算平均互訊息
            if num_images > 0:
                avg_mi = total_mi / num_images
                self.miLabel.setText(f"平均 MI: {avg_mi:.6f} (樣本數: {num_images})")
                print(f"平均互訊息: {avg_mi:.6f}")

                # 顯示詳細統計
                if len(mi_scores) > 1:
                    std_mi = np.std(mi_scores)
                    print(f"互訊息標準差: {std_mi:.6f}")
            else:
                self.miLabel.setText("MI: 沒有有效的對比影像")
                print("沒有找到有效的對比影像")

            # 生成複合影像
            if valid_images > 0:
                self.composite_image = self.composite_image / valid_images
                self.composite_image = np.clip(self.composite_image, 0, 255).astype(np.uint8)
                print(f"複合影像生成成功，使用了 {valid_images} 張影像")

                QMessageBox.information(self, "Analysis Complete",
                    f"互訊息分析完成！\n"
                    f"平均 MI: {avg_mi:.6f}\n"
                    f"分析影像數: {num_images}\n"
                    f"複合影像已生成")
            else:
                print("沒有有效影像用於生成複合影像")

        except Exception as e:
            print(f"計算互訊息時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            self.miLabel.setText(f"MI: 計算錯誤 - {str(e)}")
            QMessageBox.critical(self, "Error", f"互訊息計算失敗:\n{str(e)}")

    def saveTransforms(self):
        """儲存對齊參數"""
        transforms = {}
        for i, ctrl in enumerate(self.imageControls):
            if ctrl.filePath:
                transforms[f'image_{i+1}'] = {
                    'filepath': ctrl.filePath,
                    'transform': ctrl.getTransform(),
                    'scale_factor_used': 0.0625,
                    'note': 'CZI thumbnail mode for memory efficiency'
                }
        
        if not transforms:
            QMessageBox.warning(self, "Warning", "沒有載入的影像可以儲存參數")
            return

        filePath, _ = QFileDialog.getSaveFileName(self, "Save Alignment Parameters", "", "JSON Files (*.json)")
        if filePath:
            try:
                with open(filePath, 'w', encoding='utf-8') as f:
                    json.dump(transforms, f, indent=4, ensure_ascii=False)
                print(f"對齊參數已儲存至: {filePath}")
                QMessageBox.information(self, "Success", f"對齊參數已成功儲存至:\n{filePath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"儲存參數失敗:\n{str(e)}")

    def saveCompositeImage(self):
        """儲存複合影像"""
        if not hasattr(self, 'composite_image'):
            QMessageBox.warning(self, "Warning", "請先計算互訊息並生成複合影像")
            return
            
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Composite Image", "", "TIFF Files (*.tiff);;PNG Files (*.png)")
        if filePath:
            try:
                # 添加元數據
                metadata = {
                    'ImageDescription': 'CZI Composite Image - Thumbnail Mode (6.25% resolution)',
                    'Software': 'CZI Image Aligner with MI Analysis',
                    'DateTime': str(np.datetime64('now'))
                }

                if filePath.lower().endswith('.tiff'):
                    tifffile.imwrite(filePath, self.composite_image, imagej=True, metadata=metadata)
                else:
                    # PNG 不支援複雜元數據，直接儲存
                    from PIL import Image
                    Image.fromarray(self.composite_image).save(filePath)

                print(f"複合影像已儲存至: {filePath}")
                QMessageBox.information(self, "Success", f"複合影像已成功儲存至:\n{filePath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"儲存複合影像失敗:\n{str(e)}")

    def loadTransforms(self):
        """載入對齊參數"""
        filePath, _ = QFileDialog.getOpenFileName(self, "Load Alignment Parameters", "", "JSON Files (*.json)")
        if filePath:
            try:
                print(f"正在載入對齊參數: {filePath}")

                with open(filePath, 'r', encoding='utf-8') as f:
                    transforms = json.load(f)

                print(f"載入的參數內容: {transforms}")

                # 驗證 JSON 格式
                if not isinstance(transforms, dict):
                    raise ValueError("無效的 JSON 格式：應為字典類型")

                loaded_count = 0
                skipped_count = 0
                error_details = []

                # 逐一載入每個影像的參數
                for img_key, img_data in transforms.items():
                    try:
                        # 解析影像編號
                        if img_key.startswith('image_'):
                            img_index = int(img_key.split('_')[1]) - 1  # 轉換為 0-based index

                            if 0 <= img_index < len(self.imageControls):
                                ctrl = self.imageControls[img_index]

                                # 檢查是否包含必要的資料
                                if 'transform' not in img_data:
                                    error_details.append(f"影像 {img_index + 1}: 缺少變換資料")
                                    skipped_count += 1
                                    continue

                                # 載入檔案路徑（如果存在）
                                if 'filepath' in img_data and img_data['filepath']:
                                    # 檢查檔案是否存在
                                    file_path = Path(img_data['filepath'])
                                    if file_path.exists():
                                        print(f"  - 嘗試自動載入檔案: {file_path}")
                                        try:
                                            # 設置檔案路徑但不立即載入（讓用戶手動載入）
                                            # 這樣可以避免路徑變更問題
                                            pass
                                        except Exception as e:
                                            print(f"  - 自動載入檔案失敗: {e}")
                                    else:
                                        print(f"  - 檔案不存在，需要手動載入: {file_path}")

                                # 載入變換參數
                                transform_data = img_data['transform']
                                ctrl.setTransform(transform_data)

                                print(f"  - ✓ 影像 {img_index + 1} 參數載入成功")
                                loaded_count += 1

                            else:
                                error_details.append(f"影像索引超出範圍: {img_index + 1}")
                                skipped_count += 1
                        else:
                            error_details.append(f"未知的鍵: {img_key}")
                            skipped_count += 1

                    except Exception as e:
                        error_details.append(f"{img_key}: {str(e)}")
                        skipped_count += 1

                # 更新視覺變換
                self.updateTransforms()

                # 顯示載入結果
                result_message = f"參數載入完成！\n\n"
                result_message += f"成功載入: {loaded_count} 個影像參數\n"

                if skipped_count > 0:
                    result_message += f"跳過: {skipped_count} 個項目\n"
                    if error_details:
                        result_message += f"\n詳細錯誤:\n" + "\n".join(error_details[:5])  # 只顯示前5個錯誤
                        if len(error_details) > 5:
                            result_message += f"\n... 還有 {len(error_details) - 5} 個錯誤"

                # 顯示縮放因子信息
                if transforms and list(transforms.values())[0].get('scale_factor_used'):
                    scale_factor = list(transforms.values())[0]['scale_factor_used']
                    result_message += f"\n\n原始縮放因子: {scale_factor}"
                    if scale_factor != 0.0625:
                        result_message += f"\n⚠️ 注意: 當前程序使用 0.0625，與載入的參數可能不同"

                result_message += f"\n\n提示: 如果檔案路徑已變更，請手動重新載入 CZI 檔案"

                if loaded_count > 0:
                    QMessageBox.information(self, "載入成功", result_message)
                    print("對齊參數載入完成")
                else:
                    QMessageBox.warning(self, "載入警告", result_message)

            except json.JSONDecodeError as e:
                error_msg = f"JSON 格式錯誤:\n{str(e)}\n\n請確認檔案格式正確"
                print(f"JSON 解析錯誤: {e}")
                QMessageBox.critical(self, "格式錯誤", error_msg)

            except Exception as e:
                error_msg = f"載入參數失敗:\n{str(e)}"
                print(f"載入參數失敗: {e}")
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "載入錯誤", error_msg)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = AlignerWindow()
    window.show()

    sys.exit(app.exec())
