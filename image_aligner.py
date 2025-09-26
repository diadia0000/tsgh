import sys
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QSlider, QLabel, QDoubleSpinBox, QGridLayout
)
from PyQt6.QtGui import QPixmap, QImage, QTransform
from PyQt6.QtCore import Qt
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import warp, AffineTransform, resize
from skimage.metrics import mean_squared_error
from skimage.color import rgb2gray

def read_thumbnail(path):
    """Reads image and creates 1/8 scale thumbnail using skimage."""
    print(f"Loading image: {path}")
    # 直接使用skimage載入所有圖片格式
    img = imread(path)

    # 轉換為RGB格式
    if img.ndim == 2:
        img = np.stack((img,)*3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:,:,:3]

    # 直接縮放到1/8大小
    h, w = img.shape[:2]
    new_h, new_w = h // 8, w // 8

    img = resize(img, (new_h, new_w), preserve_range=True, anti_aliasing=True)
    return img.astype(np.uint8)

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

        self.loadButton = QPushButton(f"Load {self.name}")
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
        self.rotationSpinBox.setRange(-180, 180)
        self.rotationSpinBox.setSingleStep(1)
        self.rotationSpinBox.setValue(0)
        layout.addWidget(self.rotationSpinBox, 3, 1)

        layout.addWidget(QLabel("Translate X"), 4, 0)
        self.translateXSpinBox = QDoubleSpinBox()
        self.translateXSpinBox.setRange(-1000, 1000)
        self.translateXSpinBox.setSingleStep(10)
        self.translateXSpinBox.setValue(0)
        layout.addWidget(self.translateXSpinBox, 4, 1)

        layout.addWidget(QLabel("Translate Y"), 5, 0)
        self.translateYSpinBox = QDoubleSpinBox()
        self.translateYSpinBox.setRange(-1000, 1000)
        self.translateYSpinBox.setSingleStep(10)
        self.translateYSpinBox.setValue(0)
        layout.addWidget(self.translateYSpinBox, 5, 1)

    def getTransform(self):
        return {
            'scale': self.scaleSpinBox.value(),
            'rotation': self.rotationSpinBox.value(),
            'translation_x': self.translateXSpinBox.value(),
            'translation_y': self.translateYSpinBox.value()
        }

    def setEnabledControls(self, enabled):
        self.opacitySlider.setEnabled(enabled)
        self.scaleSpinBox.setEnabled(enabled)
        self.rotationSpinBox.setEnabled(enabled)
        self.translateXSpinBox.setEnabled(enabled)
        self.translateYSpinBox.setEnabled(enabled)


class AlignerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Aligner")
        self.setGeometry(100, 100, 1200, 800)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        # Prevent automatic resizing and scaling
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)

        self.imageControls = [
            ImageControl("Image 1 (Reference)"),
            ImageControl("Image 2"),
            ImageControl("Image 3")
        ]
        
        self.imageControls[0].setEnabledControls(False)
        self.imageControls[0].opacitySlider.setEnabled(True)

        self.initUI()
        self.connectSignals()

    def initUI(self):
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        mainLayout = QHBoxLayout(centralWidget)

        controlPanel = QWidget()
        controlLayout = QVBoxLayout(controlPanel)
        
        for ctrl in self.imageControls:
            controlLayout.addWidget(ctrl)
            controlLayout.addStretch()

        self.mseLabel = QLabel("MSE: N/A")
        self.calculateButton = QPushButton("Calculate MSE & Composite")
        self.saveTransformButton = QPushButton("Save Transforms")
        self.saveCompositeButton = QPushButton("Save Composite Image")
        
        actionLayout = QVBoxLayout()
        actionLayout.addWidget(self.calculateButton)
        actionLayout.addWidget(self.mseLabel)
        actionLayout.addWidget(self.saveTransformButton)
        actionLayout.addWidget(self.saveCompositeButton)

        controlLayout.addLayout(actionLayout)

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

        self.calculateButton.clicked.connect(self.calculateMseAndComposite)
        self.saveTransformButton.clicked.connect(self.saveTransforms)
        self.saveCompositeButton.clicked.connect(self.saveCompositeImage)

    def loadImage(self, index):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp *.tiff *.czi)")
        if filePath:
            try:
                ctrl = self.imageControls[index]
                ctrl.filePath = filePath

                print(f"Loading image {index + 1}: {filePath}")

                img_data = read_thumbnail(filePath)

                if img_data is None or img_data.size == 0:
                    print(f"Failed to load image: {filePath}")
                    return

                # 確保圖片數據格式正確
                img_data = np.ascontiguousarray(img_data, dtype=np.uint8)
                height, width, channel = img_data.shape
                bytes_per_line = 3 * width

                q_img = QImage(img_data.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
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

                print(f"Successfully loaded image {index + 1}: {width}x{height}")

            except Exception as e:
                print(f"Error loading image {filePath}: {e}")
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
                
                transform.translate(center.x(), center.y())
                transform.rotate(t['rotation'])
                transform.scale(t['scale'], t['scale'])
                transform.translate(-center.x(), -center.y())
                
                transform.translate(t['translation_x'], t['translation_y'])

                ctrl.pixmapItem.setTransform(transform)
                ctrl.pixmapItem.setOpacity(ctrl.opacitySlider.value() / 100.0)
        self.view.viewport().update()


    def getWarpedImage(self, image_index):
        ctrl = self.imageControls[image_index]
        ref_ctrl = self.imageControls[0]

        if not ctrl.filePath or not ref_ctrl.filePath:
            return None

        img = read_thumbnail(ctrl.filePath)
        ref_img = read_thumbnail(ref_ctrl.filePath)

        t = ctrl.getTransform()
        
        center_x = img.shape[1] / 2
        center_y = img.shape[0] / 2

        tf_center = AffineTransform(translation=(-center_x, -center_y))
        tf_rotate = AffineTransform(rotation=np.deg2rad(t['rotation']))
        tf_scale = AffineTransform(scale=(t['scale'], t['scale']))
        tf_translate = AffineTransform(translation=(t['translation_x'], t['translation_y']))
        tf_decenter = AffineTransform(translation=(center_x, center_y))
        
        transform_matrix = tf_center + tf_rotate + tf_scale + tf_decenter + tf_translate

        warped_img = warp(img, transform_matrix.inverse, output_shape=ref_img.shape, preserve_range=True)
        return warped_img.astype(img.dtype)


    def calculateMseAndComposite(self):
        ref_ctrl = self.imageControls[0]
        if not ref_ctrl.filePath:
            self.mseLabel.setText("MSE: Load Reference Image")
            return

        try:
            print("Starting MSE calculation...")
            ref_img_color = read_thumbnail(ref_ctrl.filePath)
            print(f"Reference image loaded: shape={ref_img_color.shape}, dtype={ref_img_color.dtype}")

            # 確保參考圖片是3通道
            if ref_img_color.ndim == 2:
                ref_img_color = np.stack((ref_img_color,)*3, axis=-1)
            elif ref_img_color.ndim == 3 and ref_img_color.shape[2] == 4:
                ref_img_color = ref_img_color[:,:,:3]

            ref_img_gray = rgb2gray(ref_img_color)
            print(f"Reference gray image: shape={ref_img_gray.shape}")

            total_mse = 0
            num_images = 0

            self.composite_image = np.zeros_like(ref_img_color, dtype=np.float64)
            self.composite_image += ref_img_color.astype(np.float64) / len(self.imageControls)

            for i in range(1, len(self.imageControls)):
                ctrl = self.imageControls[i]
                if ctrl.filePath:
                    print(f"Processing image {i+1}...")
                    warped_img_color = self.getWarpedImage(i)
                    if warped_img_color is not None:
                        print(f"Warped image {i+1}: shape={warped_img_color.shape}, dtype={warped_img_color.dtype}")

                        # 確保warped圖片是3通道
                        if warped_img_color.ndim == 2:
                            warped_img_color = np.stack((warped_img_color,)*3, axis=-1)
                        elif warped_img_color.ndim == 3 and warped_img_color.shape[2] == 4:
                            warped_img_color = warped_img_color[:,:,:3]

                        warped_img_gray = rgb2gray(warped_img_color)

                        if warped_img_gray.shape != ref_img_gray.shape:
                            print(f"Warning: Shape mismatch for MSE. Ref: {ref_img_gray.shape}, Warped: {warped_img_gray.shape}. Skipping.")
                            continue

                        mse = mean_squared_error(ref_img_gray, warped_img_gray)
                        print(f"MSE for image {i+1}: {mse:.4f}")
                        total_mse += mse
                        num_images += 1

                        self.composite_image += warped_img_color.astype(np.float64) / len(self.imageControls)
                    else:
                        print(f"Failed to get warped image {i+1}")

            if num_images > 0:
                avg_mse = total_mse / num_images
                self.mseLabel.setText(f"Average MSE: {avg_mse:.4f}")
                print(f"Average MSE calculated: {avg_mse:.4f}")
            else:
                self.mseLabel.setText("MSE: No valid images to compare")
                print("No valid images found for MSE calculation")
                self.composite_image = ref_img_color.astype(np.float64)

            self.composite_image = np.clip(self.composite_image, 0, 255).astype(np.uint8)
            print("Composite image created successfully.")

        except Exception as e:
            print(f"Error in MSE calculation: {e}")
            import traceback
            traceback.print_exc()
            self.mseLabel.setText(f"MSE: Error - {str(e)}")


    def saveTransforms(self):
        transforms = {}
        for i, ctrl in enumerate(self.imageControls):
            if ctrl.filePath:
                transforms[f'image_{i+1}'] = {
                    'filepath': ctrl.filePath,
                    'transform': ctrl.getTransform()
                }
        
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Transforms", "", "JSON Files (*.json)")
        if filePath:
            with open(filePath, 'w') as f:
                json.dump(transforms, f, indent=4)
            print(f"Transforms saved to {filePath}")

    def saveCompositeImage(self):
        if not hasattr(self, 'composite_image'):
            print("Please calculate composite image first.")
            return
            
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Composite Image", "", "TIFF Files (*.tiff);;PNG Files (*.png)")
        if filePath:
            imsave(filePath, self.composite_image)
            print(f"Composite image saved to {filePath}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AlignerWindow()
    window.show()
    sys.exit(app.exec())
