"""
Manual Mask Annotation GUI for IHC-HER2 Images

PyQt6 paintbrush tool for marking tumor regions on histopathology tiles.
Generates binary masks compatible with the UNet++ training pipeline.

Usage:
    python cell_mask/unet_mask/manual_mask_gui.py

Controls:
    Left-drag           Paint tumor region
    Right-drag          Erase
    [ / ]               Decrease / increase brush size
    Ctrl+Z              Undo last stroke
    Ctrl+Shift+A        Clear all paint
    Left / Right        Previous / Next image
    Scroll              Zoom in / out
    Space+drag          Pan image
    Middle-drag         Pan image
    Ctrl+0              Fit image to view
    Ctrl+O              Import folder
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage
from skimage import io as skio

from lab_mask_generator import generate_mask
from config import config
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_BRUSH_SIZE = 30
MIN_BRUSH_SIZE = 2
MAX_BRUSH_SIZE = 200
BRUSH_STEP = 5
MAX_UNDO = 30
OVERLAY_GREEN = (0, 255, 0, 100)  # RGBA


# ──────────────────────────────────────────────────────────────
# Canvas — paintbrush annotation on QGraphicsView
# ──────────────────────────────────────────────────────────────
class AnnotationView(QGraphicsView):
    """Image viewer with paintbrush mask annotation, zoom, and pan."""

    mask_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setBackgroundBrush(QBrush(QColor(40, 40, 40)))
        self.setMouseTracking(True)

        # Image
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._image_size: Tuple[int, int] = (0, 0)  # (w, h)

        # Mask buffer — (h, w) uint8, 0 or 255
        self._mask: Optional[np.ndarray] = None
        self._overlay_item: Optional[QGraphicsPixmapItem] = None

        # Brush
        self._brush_size: int = DEFAULT_BRUSH_SIZE
        self._painting: bool = False
        self._erasing: bool = False
        self._last_pos: Optional[QPointF] = None

        # Brush cursor (scene circle)
        self._cursor_item: Optional[QGraphicsEllipseItem] = None

        # Pan
        self._panning: bool = False
        self._pan_start: Optional[QPointF] = None
        self._space_held: bool = False

        # Undo
        self._undo_stack: List[np.ndarray] = []

    # ── public API ──

    def load_image(self, path: Path) -> bool:
        try:
            pil_img = PILImage.open(str(path)).convert("RGB")
            w, h = pil_img.size
            self._image_size = (w, h)

            qimg = QImage(
                pil_img.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888
            )
            self._scene.clear()
            self._pixmap_item = self._scene.addPixmap(
                QPixmap.fromImage(qimg)
            )
            self._scene.setSceneRect(QRectF(0, 0, w, h))

            # Empty mask + overlay
            self._mask = np.zeros((h, w), dtype=np.uint8)
            transparent = QPixmap(w, h)
            transparent.fill(QColor(0, 0, 0, 0))
            self._overlay_item = self._scene.addPixmap(transparent)
            self._overlay_item.setZValue(10)
            self._update_overlay()

            # Brush cursor circle
            self._cursor_item = self._scene.addEllipse(
                0,
                0,
                self._brush_size,
                self._brush_size,
                QPen(QColor(255, 255, 255, 150), 1.5),
                QBrush(Qt.BrushStyle.NoBrush),
            )
            self._cursor_item.setZValue(100)

            self._undo_stack.clear()
            self._painting = False
            self._last_pos = None

            self.setCursor(Qt.CursorShape.BlankCursor)
            self.fit_in_view()
            return True
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return False

    def fit_in_view(self):
        if self._pixmap_item:
            self.fitInView(
                self._scene.sceneRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    def set_mask(self, mask: np.ndarray):
        """Load an existing mask for re-editing."""
        if self._mask is not None and mask.shape == self._mask.shape:
            self._mask = mask.copy()
            self._update_overlay()
            self.mask_updated.emit()

    def get_mask(self) -> Optional[np.ndarray]:
        return self._mask

    @property
    def brush_size(self) -> int:
        return self._brush_size

    @brush_size.setter
    def brush_size(self, val: int):
        self._brush_size = max(MIN_BRUSH_SIZE, min(MAX_BRUSH_SIZE, val))
        self._update_cursor_size()

    @property
    def has_paint(self) -> bool:
        return self._mask is not None and np.any(self._mask > 0)

    @property
    def paint_coverage(self) -> float:
        if self._mask is None:
            return 0.0
        return np.sum(self._mask > 0) / self._mask.size * 100

    def clear_all(self):
        if self._mask is not None:
            self._push_undo()
            self._mask[:] = 0
            self._update_overlay()
            self.mask_updated.emit()

    def undo(self):
        if self._undo_stack:
            self._mask = self._undo_stack.pop()
            self._update_overlay()
            self.mask_updated.emit()

    # ── mouse events ──

    def enterEvent(self, event):
        self.setFocus()
        self.setCursor(Qt.CursorShape.BlankCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._cursor_item:
            self._cursor_item.setVisible(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        # Pan: middle-click or Space+left-click
        if event.button() == Qt.MouseButton.MiddleButton or (
            self._space_held
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        if not self._in_bounds(scene_pos):
            super().mousePressEvent(event)
            return

        if event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            self._push_undo()
            self._painting = True
            self._erasing = event.button() == Qt.MouseButton.RightButton
            self._stroke_at(scene_pos)
            self._last_pos = scene_pos
            self._update_overlay()
            self.mask_updated.emit()

        event.accept()

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())

        # Move brush cursor
        if self._cursor_item:
            self._cursor_item.setVisible(True)
            r = self._brush_size / 2
            self._cursor_item.setRect(
                scene_pos.x() - r,
                scene_pos.y() - r,
                self._brush_size,
                self._brush_size,
            )

        # Pan
        if self._panning and self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
            return

        # Paint / erase
        if self._painting and self._last_pos is not None:
            self._stroke_line(self._last_pos, scene_pos)
            self._last_pos = scene_pos
            self._update_overlay()

        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton or (
            self._panning
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._panning = False
            self.setCursor(Qt.CursorShape.BlankCursor)
            event.accept()
            return

        if event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            self._painting = False
            self._last_pos = None
            self.mask_updated.emit()

        event.accept()

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
        else:
            self.scale(1.0 / factor, 1.0 / factor)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            if not self._painting:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            if not self._panning:
                self.setCursor(Qt.CursorShape.BlankCursor)
        else:
            super().keyReleaseEvent(event)

    # ── internal ──

    def _in_bounds(self, pos: QPointF) -> bool:
        w, h = self._image_size
        return 0 <= pos.x() <= w and 0 <= pos.y() <= h

    def _stroke_at(self, pos: QPointF):
        if self._mask is None:
            return
        color = 0 if self._erasing else 255
        cv2.circle(
            self._mask,
            (int(pos.x()), int(pos.y())),
            self._brush_size // 2,
            color,
            -1,
        )

    def _stroke_line(self, p1: QPointF, p2: QPointF):
        if self._mask is None:
            return
        color = 0 if self._erasing else 255
        cv2.line(
            self._mask,
            (int(p1.x()), int(p1.y())),
            (int(p2.x()), int(p2.y())),
            color,
            self._brush_size,
        )

    def _push_undo(self):
        if self._mask is not None:
            self._undo_stack.append(self._mask.copy())
            if len(self._undo_stack) > MAX_UNDO:
                self._undo_stack.pop(0)

    def _update_overlay(self):
        if self._mask is None or self._overlay_item is None:
            return
        h, w = self._mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        painted = self._mask > 0
        rgba[painted, 1] = OVERLAY_GREEN[1]  # G
        rgba[painted, 3] = OVERLAY_GREEN[3]  # A
        qimg = QImage(
            rgba.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888
        )
        self._overlay_item.setPixmap(QPixmap.fromImage(qimg))

    def _update_cursor_size(self):
        if self._cursor_item:
            rect = self._cursor_item.rect()
            cx = rect.x() + rect.width() / 2
            cy = rect.y() + rect.height() / 2
            r = self._brush_size / 2
            self._cursor_item.setRect(
                cx - r, cy - r, self._brush_size, self._brush_size
            )


# ──────────────────────────────────────────────────────────────
# Main Window
# ──────────────────────────────────────────────────────────────
class ManualMaskGUI(QMainWindow):
    """Paintbrush-based mask annotation tool."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manual Mask Annotation \u2014 IHC-HER2")
        self.resize(1400, 900)

        self._image_paths: List[Path] = []
        self._current_idx: int = -1
        self._input_dir: Optional[Path] = None
        self._output_dir: Optional[Path] = None

        self._setup_ui()
        self._setup_shortcuts()

    # ── UI ──

    def _setup_ui(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        act = QAction("Import Folder", self)
        act.triggered.connect(self._import_folder)
        toolbar.addAction(act)

        act = QAction("Output Folder", self)
        act.triggered.connect(self._set_output_folder)
        toolbar.addAction(act)

        toolbar.addSeparator()

        act = QAction("\u25c0 Prev", self)
        act.triggered.connect(self._prev_image)
        toolbar.addAction(act)

        act = QAction("Next \u25b6", self)
        act.triggered.connect(self._next_image)
        toolbar.addAction(act)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Brush: "))
        self._brush_spin = QSpinBox()
        self._brush_spin.setRange(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE)
        self._brush_spin.setValue(DEFAULT_BRUSH_SIZE)
        self._brush_spin.setSuffix(" px")
        self._brush_spin.valueChanged.connect(self._on_brush_changed)
        toolbar.addWidget(self._brush_spin)

        toolbar.addSeparator()

        act = QAction("Undo", self)
        act.triggered.connect(self._undo)
        toolbar.addAction(act)

        act = QAction("Clear All", self)
        act.triggered.connect(self._clear_all)
        toolbar.addAction(act)

        act = QAction("Auto Detect", self)
        act.triggered.connect(self._auto_detect)
        toolbar.addAction(act)

        act = QAction("Fit View", self)
        act.triggered.connect(lambda: self._canvas.fit_in_view())
        toolbar.addAction(act)

        # Splitter: file list | canvas
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        self._file_label = QLabel("Images (0)")
        left_layout.addWidget(self._file_label)
        self._file_list = QListWidget()
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        left_layout.addWidget(self._file_list)
        splitter.addWidget(left)

        self._canvas = AnnotationView()
        self._canvas.mask_updated.connect(self._update_status)
        splitter.addWidget(self._canvas)

        splitter.setSizes([250, 1150])
        self.setCentralWidget(splitter)

        self._status = QLabel(
            "Ready \u2014 Import a folder to begin"
        )
        self.statusBar().addWidget(self._status, 1)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self, self._import_folder)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Ctrl+Shift+A"), self, self._clear_all)
        QShortcut(
            QKeySequence("Ctrl+0"),
            self,
            lambda: self._canvas.fit_in_view(),
        )
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._prev_image)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._next_image)
        QShortcut(
            QKeySequence(Qt.Key.Key_BracketLeft),
            self,
            self._brush_smaller,
        )
        QShortcut(
            QKeySequence(Qt.Key.Key_BracketRight),
            self,
            self._brush_larger,
        )

    # ── Brush size ──

    def _on_brush_changed(self, val: int):
        self._canvas.brush_size = val

    def _brush_smaller(self):
        self._brush_spin.setValue(
            max(MIN_BRUSH_SIZE, self._brush_spin.value() - BRUSH_STEP)
        )

    def _brush_larger(self):
        self._brush_spin.setValue(
            min(MAX_BRUSH_SIZE, self._brush_spin.value() + BRUSH_STEP)
        )

    # ── Folder management ──

    def _import_folder(self):
        self._save_current()
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder"
        )
        if not folder:
            return

        self._input_dir = Path(folder)
        self._load_file_list()

        if self._output_dir is None:
            self._output_dir = self._input_dir / "masks"
            self._output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Default output folder: {self._output_dir}")

        if self._image_paths:
            self._navigate_to(0)

    def _set_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder"
        )
        if folder:
            self._output_dir = Path(folder)
            self._output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output folder: {self._output_dir}")
            self._refresh_icons()

    def _load_file_list(self):
        self._file_list.blockSignals(True)
        self._file_list.clear()

        self._image_paths = sorted(
            p
            for p in self._input_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        for path in self._image_paths:
            self._file_list.addItem(QListWidgetItem(path.name))

        self._file_label.setText(f"Images ({len(self._image_paths)})")
        self._file_list.blockSignals(False)
        self._refresh_icons()

    def _refresh_icons(self):
        if not self._output_dir:
            return
        for i, path in enumerate(self._image_paths):
            item = self._file_list.item(i)
            mask_exists = (
                self._output_dir / (path.stem + "_mask.png")
            ).exists()
            if mask_exists:
                item.setText("\u2713 " + path.name)
                item.setForeground(QColor(100, 200, 100))
            else:
                item.setText("  " + path.name)
                item.setForeground(QColor(200, 200, 200))

    # ── Navigation ──

    def _on_file_selected(self, row: int):
        if 0 <= row < len(self._image_paths):
            self._save_current()
            self._navigate_to(row)

    def _navigate_to(self, idx: int):
        if idx < 0 or idx >= len(self._image_paths):
            return

        self._current_idx = idx
        path = self._image_paths[idx]
        self._canvas.load_image(path)
        self._load_mask(path)

        self._file_list.blockSignals(True)
        self._file_list.setCurrentRow(idx)
        self._file_list.blockSignals(False)
        self._update_status()

    def _prev_image(self):
        if self._current_idx > 0:
            self._save_current()
            self._navigate_to(self._current_idx - 1)

    def _next_image(self):
        if self._current_idx < len(self._image_paths) - 1:
            self._save_current()
            self._navigate_to(self._current_idx + 1)

    # ── Actions ──

    def _undo(self):
        self._canvas.undo()

    def _clear_all(self):
        self._canvas.clear_all()

    def _auto_detect(self):
        """Run lab_mask_generator on the current image and load result as mask."""
        if self._current_idx < 0:
            return
        path = self._image_paths[self._current_idx]
        try:
            image = skio.imread(str(path))
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]

            stain_matrix = np.array(config.stain_matrix)
            mask, _ = generate_mask(
                image,
                stain_matrix=stain_matrix,
                dab_threshold=config.dab_threshold,
                fill_close_kernel=config.fill_close_kernel,
                fill_min_cell_area=config.fill_min_cell_area,
                fill_max_edge_hole_area=config.fill_max_edge_hole_area,
                fill_open_kernel=config.fill_open_kernel,
            )
            # generate_mask returns 0/255 uint8; set_mask expects the same
            self._canvas.set_mask(mask)
            logger.info(f"Auto detect applied: {path.name}")
        except Exception as e:
            logger.error(f"Auto detect failed: {path.name} | {e}")

    # ── Save / Load ──

    def _save_current(self):
        if self._current_idx < 0 or not self._output_dir:
            return
        mask = self._canvas.get_mask()
        if mask is None:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._image_paths[self._current_idx]
        mask_path = self._output_dir / (path.stem + "_mask.png")

        # Red-channel PNG — compatible with existing pipeline
        h, w = mask.shape
        red_mask = np.zeros((h, w, 3), dtype=np.uint8)
        red_mask[:, :, 0] = mask
        PILImage.fromarray(red_mask, "RGB").save(str(mask_path))

        logger.info(f"Saved: {mask_path.name}")
        self._refresh_icons()

    def _load_mask(self, path: Path):
        """Reload existing mask for re-editing."""
        if not self._output_dir:
            return
        mask_path = self._output_dir / (path.stem + "_mask.png")
        if mask_path.exists():
            try:
                arr = np.array(
                    PILImage.open(str(mask_path)).convert("RGB")
                )
                self._canvas.set_mask(arr[:, :, 0])
            except Exception as e:
                logger.warning(f"Failed to load mask: {e}")

    # ── Status ──

    def _update_status(self):
        parts = []
        if self._image_paths:
            parts.append(
                f"Image {self._current_idx + 1}/{len(self._image_paths)}"
            )
        parts.append(f"Brush: {self._canvas.brush_size}px")
        parts.append(f"Coverage: {self._canvas.paint_coverage:.1f}%")
        parts.append(
            "Left=paint  Right=erase  Space+drag=pan  []=brush size"
        )
        self._status.setText("  |  ".join(parts))

    def closeEvent(self, event):
        self._save_current()
        event.accept()


# ──────────────────────────────────────────────────────────────
DARK_STYLE = """
QMainWindow, QWidget { background-color: #2b2b2b; color: #ddd; }
QToolBar {
    background-color: #333; border: none; spacing: 6px; padding: 4px;
}
QToolBar QToolButton {
    background-color: #444; color: #ddd; border: 1px solid #555;
    border-radius: 4px; padding: 6px 12px; font-size: 13px;
}
QToolBar QToolButton:hover { background-color: #555; }
QToolBar QToolButton:pressed { background-color: #0078d4; }
QToolBar QLabel { color: #ccc; font-size: 13px; }
QSpinBox {
    background-color: #444; color: #ddd; border: 1px solid #555;
    border-radius: 4px; padding: 4px 8px; font-size: 13px; min-width: 80px;
}
QListWidget {
    background-color: #1e1e1e; border: 1px solid #444;
    color: #ddd; font-size: 13px;
}
QListWidget::item { padding: 4px; }
QListWidget::item:selected { background-color: #0078d4; }
QStatusBar { background-color: #333; color: #aaa; font-size: 12px; }
QLabel { color: #ccc; font-size: 13px; }
QSplitter::handle { background-color: #444; width: 3px; }
"""


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = ManualMaskGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
