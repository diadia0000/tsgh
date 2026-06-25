"""CSV 匯出與統計摘要。"""

import csv
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, List

from cell_mask.hybrid.hybrid_data_types import CellAnalysisResult

logger = logging.getLogger(__name__)

_CSV_HEADER = [
    "cell_id",
    "centroid_x",
    "centroid_y",
    "reddot",
    "blackdot",
    "score",
]


def _format_count(val: int, excluded: bool) -> str:
    return "NaN" if excluded else str(int(val))


def _format_score(score: float, excluded: bool) -> str:
    return "NaN" if excluded else f"{score:.4f}"


def export_tile_csv(
    results: List[CellAnalysisResult],
    output_path: Path,
    slide_id: str = "unknown",
    tile_id: str = "unknown",
    model_version: str = "v1.0.0",
    config_hash: str = "00000000",
) -> Path:
    """匯出單張 tile 的細胞分類結果至 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        for cell in results:
            excluded = bool(getattr(cell, "excluded", False))
            red = getattr(cell, "cep17_dot_count", 0)
            black = getattr(cell, "her2_dot_count", 0)
            score = getattr(cell, "score", 0.0)
            writer.writerow([
                cell.cell_id,
                f"{cell.centroid_x:.2f}",
                f"{cell.centroid_y:.2f}",
                _format_count(red, excluded),
                _format_count(black, excluded),
                _format_score(score, excluded),
            ])

    logger.info("CSV 匯出完成: %s (%d 列)", output_path.name, len(results))
    return output_path


@dataclass
class DotStatsSummary:
    """有效雙色細胞統計摘要。

    可代表單一 tile 的統計，亦可由多個 tile 摘要合併為 slide-level。
    僅記錄 count；百分比於寫檔時計算，避免合併產生累積誤差。

    有效細胞定義：未排除（excluded=False）且 reddot >= 2 且 blackdot >= 1。
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
            and getattr(r, "cep17_dot_count", 0) >= 2
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
