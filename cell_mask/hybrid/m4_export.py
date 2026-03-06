"""
M4: CSV 與視覺化匯出模組

匯出內容:
  1. ``tile_report.csv`` — 每列一個細胞的定量紀錄。
  2. ``overlay_vis.png`` — 全 tile 細胞邊界 + dot 標註疊加圖。
  3. ``cells/cell_{id}.png`` — 逐細胞裁切標註影像。
"""

import csv
import logging
import math
from pathlib import Path
from typing import List

import cv2
import numpy as np
from scipy import ndimage

from m3_dot_quant import CellAnalysisResult

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 顏色常數 (BGR for OpenCV)
# ------------------------------------------------------------------
_COLOR_BOUNDARY = (0, 255, 0)     # 綠色：細胞邊界
_COLOR_BLACK_DOT = (255, 0, 0)    # 藍色：black dot 標記
_COLOR_RED_DOT = (0, 0, 255)      # 紅色：red dot 標記
_COLOR_TEXT = (255, 255, 255)      # 白色：文字標籤


# ------------------------------------------------------------------
# CSV 匯出
# ------------------------------------------------------------------

_CSV_HEADER = [
    "slide_id",
    "tile_id",
    "cell_id",
    "centroid_x",
    "centroid_y",
    "black_dot_count",
    "red_dot_count",
    "ratio",
    "is_border_cell",
    "model_version",
    "config_hash",
]


def export_tile_csv(
    results: List[CellAnalysisResult],
    output_path: Path,
    slide_id: str = "unknown",
    tile_id: str = "unknown",
    model_version: str = "v1.0.0",
    config_hash: str = "00000000",
) -> Path:
    """匯出單張 tile 的細胞量化結果至 CSV。

    若 ``results`` 為空，僅寫入表頭 (header-only CSV)。

    Args:
        results: 細胞量化結果列表。
        output_path: CSV 儲存路徑。
        slide_id: 玻片識別碼。
        tile_id: Tile 識別碼。
        model_version: 模型版本字串。
        config_hash: 配置雜湊。

    Returns:
        實際寫入的 CSV 路徑。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        for cell in results:
            writer.writerow(_format_csv_row(
                cell, slide_id, tile_id, model_version, config_hash
            ))

    logger.info(
        "CSV 匯出完成: %s (%d 列)", output_path.name, len(results)
    )
    return output_path


def _format_csv_row(
    cell: CellAnalysisResult,
    slide_id: str,
    tile_id: str,
    model_version: str,
    config_hash: str,
) -> list:
    """將單一 CellAnalysisResult 轉換為 CSV 列。"""
    ratio_str = _ratio_to_str(cell.ratio)
    return [
        slide_id,
        tile_id,
        cell.cell_id,
        f"{cell.centroid_x:.1f}",
        f"{cell.centroid_y:.1f}",
        cell.black_dot_count,
        cell.red_dot_count,
        ratio_str,
        False,  # is_border_cell — 邊界細胞已在 M2 移除
        model_version,
        config_hash,
    ]


def _ratio_to_str(ratio: float) -> str:
    """ratio → CSV 字串格式。"""
    if math.isinf(ratio):
        return "inf"
    if math.isnan(ratio):
        return "nan"
    return f"{ratio:.4f}"


# ------------------------------------------------------------------
# 全 tile overlay 視覺化
# ------------------------------------------------------------------

def export_overlay_visualization(
    masked_dish_image: np.ndarray,
    cell_instance_mask: np.ndarray,
    results: List[CellAnalysisResult],
    output_path: Path,
) -> Path:
    """匯出含有細胞邊界和 dot 標註的疊加圖。

    Args:
        masked_dish_image: shape ``(H, W, 3)`` RGB。
        cell_instance_mask: shape ``(H, W)`` 實例遮罩。
        results: 每個細胞的定量結果。
        output_path: 輸出 PNG 路徑。

    Returns:
        實際寫入的路徑。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = cv2.cvtColor(masked_dish_image.copy(), cv2.COLOR_RGB2BGR)
    _draw_cell_boundaries(canvas, cell_instance_mask)
    _draw_cell_labels(canvas, results)

    cv2.imwrite(str(output_path), canvas)
    logger.info("Overlay 匯出完成: %s", output_path.name)
    return output_path


def _draw_cell_boundaries(
    canvas: np.ndarray,
    cell_instance_mask: np.ndarray,
) -> None:
    """在 canvas 上繪製所有細胞的輪廓線。"""
    cell_ids = set(np.unique(cell_instance_mask)) - {0}
    for cid in cell_ids:
        binary = (cell_instance_mask == cid).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, _COLOR_BOUNDARY, 1)


def _draw_cell_labels(
    canvas: np.ndarray,
    results: List[CellAnalysisResult],
) -> None:
    """在每個細胞質心處繪製 B/R 計數標籤。"""
    for cell in results:
        label = f"B{cell.black_dot_count}/R{cell.red_dot_count}"
        position = (int(cell.centroid_x), int(cell.centroid_y))
        cv2.putText(
            canvas,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            _COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )


# ------------------------------------------------------------------
# 逐細胞裁切影像
# ------------------------------------------------------------------

def export_per_cell_images(
    masked_dish_image: np.ndarray,
    cell_instance_mask: np.ndarray,
    results: List[CellAnalysisResult],
    output_dir: Path,
    padding: int = 10,
) -> List[Path]:
    """裁切每個細胞的局部影像並儲存。

    Args:
        masked_dish_image: shape ``(H, W, 3)`` RGB。
        cell_instance_mask: shape ``(H, W)`` 實例遮罩。
        results: 細胞量化結果。
        output_dir: 輸出資料夾 (``cells/`` 子目錄)。
        padding: 裁切外擴像素數。

    Returns:
        儲存成功的檔案路徑列表。
    """
    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for cell in results:
        path = _export_single_cell(
            masked_dish_image,
            cell_instance_mask,
            cell,
            cells_dir,
            padding,
        )
        if path is not None:
            saved_paths.append(path)

    logger.info("匯出 %d 張 per-cell 影像至 %s", len(saved_paths), cells_dir)
    return saved_paths


def _export_single_cell(
    image: np.ndarray,
    mask: np.ndarray,
    cell: CellAnalysisResult,
    output_dir: Path,
    padding: int,
) -> Path:
    """裁切並儲存單一細胞影像。"""
    region = (mask == cell.cell_id)
    rows, cols = np.where(region)
    if rows.size == 0:
        return None

    h, w = image.shape[:2]
    r_min = max(0, rows.min() - padding)
    r_max = min(h, rows.max() + padding + 1)
    c_min = max(0, cols.min() - padding)
    c_max = min(w, cols.max() + padding + 1)

    crop = image[r_min:r_max, c_min:c_max].copy()
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

    cell_path = output_dir / f"cell_{cell.cell_id}.png"
    cv2.imwrite(str(cell_path), crop_bgr)
    return cell_path


# ------------------------------------------------------------------
# 統一匯出入口
# ------------------------------------------------------------------

def export_cell_dot_annotations(
    masked_dish_image: np.ndarray,
    cell_instance_mask: np.ndarray,
    results: List[CellAnalysisResult],
    output_dir: Path,
    slide_id: str = "unknown",
    tile_id: str = "unknown",
    model_version: str = "v1.0.0",
    config_hash: str = "00000000",
) -> None:
    """統一匯出 CSV + overlay PNG + per-cell PNG。

    Args:
        masked_dish_image: 遮罩後 DISH 影像。
        cell_instance_mask: 實例遮罩。
        results: 細胞量化結果列表。
        output_dir: 匯出根目錄。
        slide_id: 玻片識別碼。
        tile_id: Tile 識別碼。
        model_version: 模型版本。
        config_hash: 配置雜湊。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    export_tile_csv(
        results,
        output_dir / f"{tile_id}_report.csv",
        slide_id=slide_id,
        tile_id=tile_id,
        model_version=model_version,
        config_hash=config_hash,
    )

    export_overlay_visualization(
        masked_dish_image,
        cell_instance_mask,
        results,
        output_dir / f"{tile_id}_overlay.png",
    )

    export_per_cell_images(
        masked_dish_image,
        cell_instance_mask,
        results,
        output_dir,
    )
