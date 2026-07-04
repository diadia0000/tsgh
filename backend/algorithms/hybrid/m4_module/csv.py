"""CSV 匯出與統計摘要。"""

import csv
import logging
import unicodedata
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, List

try:
    from ..hybrid_data_types import CellAnalysisResult
except ImportError:
    from hybrid_data_types import CellAnalysisResult

logger = logging.getLogger(__name__)

_CSV_HEADER = [
    "cell_id",
    "centroid_x",
    "centroid_y",
    "reddot",
    "blackdot",
    "score",
]

# ASCO/CAP 2013 dual-probe ISH 判讀閾值（guideline 固定值，非可調參數）。
_ASCO_RATIO_THRESHOLD = 2.0    # HER2/CEP17 ratio >= 2.0 → 陽性
_ASCO_COPY_POSITIVE = 6.0      # ratio<2 但平均 HER2 拷貝數 >= 6.0 → 陽性
_ASCO_COPY_EQUIVOCAL = 4.0     # ratio<2 且 4.0 <= 平均拷貝數 < 6.0 → equivocal


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
    total_her2: int = 0       # 有效細胞 HER2 黑點總和 (ΣHER2)，供 case ratio/平均拷貝數
    total_cep17: int = 0      # 有效細胞 CEP17 紅點總和 (ΣCEP17)，供 case ratio
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
            total_her2=sum(r.her2_dot_count for r in valid),
            total_cep17=sum(r.cep17_dot_count for r in valid),
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


def _asco_cap_2013_verdict(case_ratio: float, avg_her2_copies: float) -> str:
    """ASCO/CAP 2013 dual-probe ISH case-level 判讀。

    - ratio >= 2.0                       → Amplified (positive)
    - ratio < 2.0 且 平均拷貝數 >= 6.0    → Amplified (positive)
    - ratio < 2.0 且 4.0 <= 平均 < 6.0    → Equivocal
    - 其餘                                → Not amplified (negative)
    """
    if case_ratio >= _ASCO_RATIO_THRESHOLD:
        return "Amplified (positive)"
    if avg_her2_copies >= _ASCO_COPY_POSITIVE:
        return "Amplified (positive)"
    if avg_her2_copies >= _ASCO_COPY_EQUIVOCAL:
        return "Equivocal"
    return "Not amplified (negative)"


def _display_width(text: str) -> int:
    """字串在等寬字型下的顯示寬度；CJK 全形字 (W/F) 算 2 欄。"""
    return sum(
        2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        for ch in text
    )


def _pad(text: str, width: int) -> str:
    """右側補空白至指定顯示寬度（CJK 寬度感知），供純文字對齊。"""
    return text + " " * max(0, width - _display_width(text))


# 判讀結論的中文註解（讓醫師一眼看懂英文術語）。
_VERDICT_GLOSS = {
    "Amplified (positive)": "擴增 / 陽性",
    "Equivocal": "",
    "Not amplified (negative)": "未擴增 / 陰性",
    "Insufficient cells": "有效細胞不足，無法判讀",
}


def write_summary_csv(
    stats: DotStatsSummary,
    output_path: Path,
) -> Path:
    """把 DotStatsSummary 寫成對齊好讀的純文字報告（含 ASCO/CAP 2013 判讀）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = stats.valid_cells

    def pct(count: int) -> str:
        return f"{count / n * 100:.1f}%" if n > 0 else "N/A"

    # case-level 判讀：ratio = ΣHER2/ΣCEP17、平均拷貝數 = ΣHER2/有效細胞數。
    # 有效細胞需 cep17 >= 2，故 n > 0 時 total_cep17 必 > 0，無除零風險。
    if n > 0:
        case_ratio = stats.total_her2 / stats.total_cep17
        avg_her2_copies = stats.total_her2 / n
        verdict = _asco_cap_2013_verdict(case_ratio, avg_her2_copies)
        ratio_str = f"{case_ratio:.2f}"
        avg_str = f"{avg_her2_copies:.2f}"
    else:
        verdict = "Insufficient cells"
        ratio_str = "N/A"
        avg_str = "N/A"

    # ── 判讀區塊（label → 值，label 補齊到固定顯示寬度）──
    kv = [
        ("判讀結論", f"{verdict}（{_VERDICT_GLOSS.get(verdict, '')}）"),
        ("HER2/CEP17 比值", ratio_str),
        ("平均 HER2 拷貝數", avg_str),
        ("HER2 訊號總數", str(stats.total_her2)),
        ("CEP17 訊號總數", str(stats.total_cep17)),
        ("有效腫瘤細胞數", str(n)),
    ]
    kv_w = max(_display_width(k) for k, _ in kv) + 2

    # ── 分佈區塊（count 右對齊 + 百分比）──
    dist = [
        ("比值 < 2", stats.ratio_lt2),
        ("比值 >= 2", stats.ratio_gte2),
        ("HER2 拷貝數 < 4", stats.copy_lt4),
        ("HER2 拷貝數 4-5", stats.copy_4to5),
        ("HER2 拷貝數 >= 6", stats.copy_gte6),
    ]
    dist_w = max(_display_width(k) for k, _ in dist) + 2

    bar = "=" * 46
    sub = "-" * 46
    lines = [bar, "  ASCO/CAP 2013 判讀", bar]
    lines += [f"  {_pad(k, kv_w)}{v}" for k, v in kv]
    lines += ["", sub, "  細胞分佈（佔有效細胞比例）", sub]
    lines += [f"  {_pad(k, dist_w)}{c:>5}  ({pct(c)})" for k, c in dist]
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary 報告輸出: %s (%d valid cells)", output_path.name, n)
    return output_path


def export_summary_statistics(
    results: List[CellAnalysisResult],
    output_path: Path,
) -> Path:
    """Per-tile 便捷包裝：compute → write。"""
    stats = DotStatsSummary.from_results(results)
    return write_summary_csv(stats, output_path)
