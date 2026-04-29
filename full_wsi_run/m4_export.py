"""
M4: CSV 與視覺化匯出模組

匯出內容:
    1. summary CSV (有效雙色細胞統計)
    2. overlay PNG (細胞邊界 + +/- 標註)
"""

import csv
import logging
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
