"""
M4: CSV 與視覺化匯出模組

匯出內容:
  1. ``tile_report.csv`` — 每列一個細胞的分類紀錄。
  2. ``overlay_vis.png`` — 全 tile 細胞邊界 + 分類標註疊加圖。
  3. ``cells/cell_{id}.png`` — 逐細胞固定尺寸裁切影像。
"""

import csv
import logging
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

from cell_mask.hybrid.m3_cells_generator import CellAnalysisResult

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 顏色常數 (BGR for OpenCV)
# ------------------------------------------------------------------
_COLOR_BOUNDARY = (0, 255, 0)     # 綠色：細胞邊界
_COLOR_POSITIVE = (0, 0, 255)     # 紅色：HER2 陽性
_COLOR_NEGATIVE = (255, 0, 0)     # 藍色：HER2 陰性
_COLOR_TEXT = (255, 255, 255)     # 白色：文字標籤


# ------------------------------------------------------------------
# CSV 匯出
# ------------------------------------------------------------------

_CSV_HEADER = [
    "cell_id",
    "reddot",
    "blackdot",
    "ratio",
]


def _format_count(val: int, excluded: bool) -> str:
    return "NaN" if excluded else str(int(val))


def _format_ratio(ratio: float, excluded: bool) -> str:
    if excluded:
        return "NaN"
    if ratio == float("inf") or ratio == 0.0 or math.isnan(ratio):
        return "NaN"
    return f"{ratio:.4f}"


def export_tile_csv(
    results: List[CellAnalysisResult],
    output_path: Path,
    slide_id: str = "unknown",
    tile_id: str = "unknown",
    model_version: str = "v1.0.0",
    config_hash: str = "00000000",
) -> Path:
    """匯出單張 tile 的細胞分類結果至 CSV。

    Args:
        results: 細胞分類結果列表。
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
            excluded = bool(getattr(cell, "excluded", False))
            ratio = getattr(cell, "her2_cep17_ratio", 0.0)
            red = getattr(cell, "cep17_dot_count", 0)
            black = getattr(cell, "her2_dot_count", 0)
            writer.writerow([
                cell.cell_id,
                _format_count(red, excluded),
                _format_count(black, excluded),
                _format_ratio(ratio, excluded),
            ])

    logger.info(
        "CSV 匯出完成: %s (%d 列)", output_path.name, len(results)
    )
    return output_path


# ------------------------------------------------------------------
# 全 tile overlay 視覺化
# ------------------------------------------------------------------

def export_overlay_visualization(
    overlay_image: np.ndarray,
    cell_instance_mask: np.ndarray,
    results: List[CellAnalysisResult],
    output_path: Path,
) -> Path:
    """匯出含有細胞邊界和分類標註的疊加圖。

    Args:
        overlay_image: shape ``(H, W, 3)`` RGB。
        cell_instance_mask: shape ``(H, W)`` 實例遮罩。
        results: 每個細胞的分類結果。
        output_path: 輸出 PNG 路徑。

    Returns:
        實際寫入的路徑。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = cv2.cvtColor(overlay_image.copy(), cv2.COLOR_RGB2BGR)
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
    """在每個細胞質心處繪製 +/- 分類標籤。"""
    for cell in results:
        label = "+" if cell.is_her2_positive else "-"
        color = _COLOR_POSITIVE if cell.is_her2_positive else _COLOR_NEGATIVE
        position = (int(cell.centroid_x), int(cell.centroid_y))
        cv2.putText(
            canvas,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )


# ------------------------------------------------------------------
# 逐細胞固定尺寸裁切影像
# ------------------------------------------------------------------

def export_per_cell_images(
    source_image: np.ndarray,
    cell_instance_mask: np.ndarray,
    results: List[CellAnalysisResult],
    output_dir: Path,
    crop_size: int = 64,
) -> List[Path]:
    """以 Cellpose instance mask 形狀輸出每個細胞的固定尺寸影像。

    每顆細胞先以 instance mask 萃取原始形狀，細胞外背景填 255。
    之後放入固定 ``crop_size x crop_size`` 白底畫布；若原始細胞塊超過尺寸
    則以中心裁切保留中間區域。

    Args:
        source_image: shape ``(H, W, 3)`` RGB。
        cell_instance_mask: shape ``(H, W)`` 實例遮罩。
        results: 細胞分類結果。
        output_dir: 輸出資料夾 (``cells/`` 子目錄)。
        crop_size: 裁切尺寸 (正方形邊長, pixels)。

    Returns:
        儲存成功的檔案路徑列表。
    """
    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []

    for cell in results:
        region_mask = (cell_instance_mask == cell.cell_id)
        if not np.any(region_mask):
            continue

        cell_crop = _extract_mask_shaped_cell(
            source_image,
            region_mask,
        )
        fixed_crop = _fit_to_fixed_canvas(
            cell_crop,
            crop_size=crop_size,
            fill_value=255,
        )

        crop_bgr = cv2.cvtColor(fixed_crop, cv2.COLOR_RGB2BGR)
        cell_path = cells_dir / f"cell_{cell.cell_id}.png"
        cv2.imwrite(str(cell_path), crop_bgr)
        saved_paths.append(cell_path)

    logger.info("匯出 %d 張 per-cell 影像至 %s", len(saved_paths), cells_dir)
    return saved_paths


def _extract_mask_shaped_cell(
    image: np.ndarray,
    region_mask: np.ndarray,
) -> np.ndarray:
    """依 instance mask 形狀擷取單細胞區域，細胞外填 255。

    Args:
        image: shape ``(H, W, 3)``。
        region_mask: shape ``(H, W)``、bool。

    Returns:
        shape ``(h_cell, w_cell, 3)``。
    """
    ys, xs = np.where(region_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    src_patch = image[y0:y1, x0:x1]
    mask_patch = region_mask[y0:y1, x0:x1]

    cell_patch = np.full(src_patch.shape, 255, dtype=image.dtype)
    cell_patch[mask_patch] = src_patch[mask_patch]
    return cell_patch


def _fit_to_fixed_canvas(
    patch: np.ndarray,
    crop_size: int,
    fill_value: int = 255,
) -> np.ndarray:
    """將任意尺寸 patch 放入固定尺寸白底畫布。"""
    h, w = patch.shape[:2]
    canvas = np.full((crop_size, crop_size, 3), fill_value, dtype=patch.dtype)

    src_y0 = max((h - crop_size) // 2, 0)
    src_x0 = max((w - crop_size) // 2, 0)
    src_y1 = src_y0 + min(h, crop_size)
    src_x1 = src_x0 + min(w, crop_size)

    trimmed = patch[src_y0:src_y1, src_x0:src_x1]
    th, tw = trimmed.shape[:2]

    dst_y0 = (crop_size - th) // 2
    dst_x0 = (crop_size - tw) // 2
    canvas[dst_y0:dst_y0 + th, dst_x0:dst_x0 + tw] = trimmed
    return canvas


# ------------------------------------------------------------------
# 統一匯出入口
# ------------------------------------------------------------------

@dataclass
class DotStatsSummary:
    """有效雙色細胞統計摘要。

    可代表單一 tile 的統計，亦可由多個 tile 摘要合併為 slide-level。
    僅記錄 count；百分比於寫檔時計算，避免合併產生累積誤差。

    有效細胞定義：未排除（excluded=False）且 reddot >= 1 且 blackdot >= 1。
    """

    valid_cells: int = 0      # 雙色有效細胞總數
    ratio_lt2: int = 0        # her2/cep17 < 2
    ratio_gte2: int = 0       # her2/cep17 >= 2
    copy_lt4: int = 0         # blackdot < 4
    copy_4to5: int = 0        # 4 <= blackdot < 6
    copy_gte6: int = 0        # blackdot >= 6

    @classmethod
    def from_results(
        cls,
        results: List[CellAnalysisResult],
    ) -> "DotStatsSummary":
        valid = [
            r for r in results
            if not getattr(r, "excluded", False)
            and getattr(r, "cep17_dot_count", 0) >= 1
            and getattr(r, "her2_dot_count", 0) >= 1
        ]
        ratio_lt2 = sum(
            1 for r in valid
            if getattr(r, "her2_cep17_ratio", 0.0) < 2.0
        )
        copy_lt4 = sum(1 for r in valid if r.her2_dot_count < 4)
        copy_4to5 = sum(1 for r in valid if 4 <= r.her2_dot_count < 6)
        copy_gte6 = sum(1 for r in valid if r.her2_dot_count >= 6)
        return cls(
            valid_cells=len(valid),
            ratio_lt2=ratio_lt2,
            ratio_gte2=len(valid) - ratio_lt2,
            copy_lt4=copy_lt4,
            copy_4to5=copy_4to5,
            copy_gte6=copy_gte6,
        )

    def merge(self, other: "DotStatsSummary") -> "DotStatsSummary":
        """欄位逐項相加，回傳新 summary。"""
        return DotStatsSummary(**{
            f.name: getattr(self, f.name) + getattr(other, f.name)
            for f in fields(self)
        })

    @classmethod
    def aggregate(
        cls,
        summaries: Iterable["DotStatsSummary"],
    ) -> "DotStatsSummary":
        """多個 summary 合併為一（slide-level 接口）。"""
        total = cls()
        for s in summaries:
            total = total.merge(s)
        return total


def write_summary_csv(
    stats: DotStatsSummary,
    output_path: Path,
) -> Path:
    """Write DotStatsSummary to CSV; percentages are computed here."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = stats.valid_cells

    def pct(count: int) -> str:
        return f"{count / n * 100:.1f}%" if n > 0 else "N/A"

    rows = [
        ["Metric", "Count", "Percentage"],
        ["Valid tumor cells (both HER2 and CEP17 dots present)", n, "100%" if n > 0 else "N/A"],
        ["HER2/CEP17 ratio < 2", stats.ratio_lt2, pct(stats.ratio_lt2)],
        ["HER2/CEP17 ratio >= 2", stats.ratio_gte2, pct(stats.ratio_gte2)],
        ["HER2 copy number (black dots) < 4", stats.copy_lt4, pct(stats.copy_lt4)],
        ["HER2 copy number (black dots) 4-5", stats.copy_4to5, pct(stats.copy_4to5)],
        ["HER2 copy number (black dots) >= 6", stats.copy_gte6, pct(stats.copy_gte6)],
    ]

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)

    logger.info("Summary CSV exported: %s (%d valid cells)", output_path.name, n)
    return output_path


def export_summary_statistics(
    results: List[CellAnalysisResult],
    output_path: Path,
) -> Path:
    """Per-tile 便捷包裝：compute → write。"""
    stats = DotStatsSummary.from_results(results)
    return write_summary_csv(stats, output_path)


def export_cell_dot_annotations(
    overlay_image: np.ndarray,
    cell_instance_mask: np.ndarray,
    results: List[CellAnalysisResult],
    output_dir: Path,
    visualization_image: np.ndarray = None,
    slide_id: str = "unknown",
    tile_id: str = "unknown",
    model_version: str = "v1.0.0",
    config_hash: str = "00000000",
    crop_size: int = 64,
) -> None:
    """統一匯出 CSV + overlay PNG + per-cell PNG + 統計摘要。

    Args:
        overlay_image: IHC-DISH 疊合影像。
        cell_instance_mask: 實例遮罩。
        results: 細胞分類結果列表。
        output_dir: 匯出根目錄。
        visualization_image: 視覺化底圖；若為 None 則使用 overlay_image。
        slide_id: 玻片識別碼。
        tile_id: Tile 識別碼。
        model_version: 模型版本。
        config_hash: 配置雜湊。
        crop_size: 單細胞裁切尺寸 (pixels)。
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

    export_summary_statistics(
        results,
        output_dir / f"{tile_id}_summary.csv",
    )

    vis_image = (
        overlay_image if visualization_image is None
        else visualization_image
    )

    export_overlay_visualization(
        vis_image,
        cell_instance_mask,
        results,
        output_dir / f"{tile_id}_overlay.png",
    )

    export_per_cell_images(
        overlay_image,
        cell_instance_mask,
        results,
        output_dir,
        crop_size=crop_size,
    )
