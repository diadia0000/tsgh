"""CSV 匯出與統計摘要。"""

import csv
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List

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
    "is_her2_positive",
    "cell_type",
]

# ASCO/CAP 2013 dual-probe ISH 判讀閾值（guideline 固定值，非可調參數）。
_ASCO_RATIO_THRESHOLD = 2.0    # HER2/CEP17 ratio >= 2.0 → 陽性
_ASCO_COPY_POSITIVE = 6.0      # ratio<2 但平均 HER2 拷貝數 >= 6.0 → 陽性
_ASCO_COPY_EQUIVOCAL = 4.0     # ratio<2 且 4.0 <= 平均拷貝數 < 6.0 → equivocal


def _format_count(val: int, excluded: bool) -> str:
    return "NaN" if excluded else str(int(val))


def _format_score(score: float, excluded: bool) -> str:
    return "NaN" if excluded else f"{score:.4f}"


def _classify_cell_type(cell: CellAnalysisResult) -> str:
    """典型陽性／非典型陽性／陰性／蛋白陽性未擴增／排除 — 5 種互斥狀態。

    完全由既有欄位 (is_her2_positive, is_amplified, excluded) 推導，不需新增
    資料欄位；excluded 先判，被排除的細胞不進任何分型桶（與 CSV 打 NaN 一致）。
    """
    if cell.excluded:
        return "excluded"
    if cell.is_her2_positive:
        return "classic_positive" if cell.is_amplified else "protein_positive_not_amplified"
    return "non_classic_positive" if cell.is_amplified else "negative"


def export_tile_csv(
    results: List[CellAnalysisResult],
    output_path: Path,
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
                cell.is_her2_positive,
                _classify_cell_type(cell),
            ])

    logger.info("CSV 匯出完成: %s (%d 列)", output_path.name, len(results))
    return output_path


@dataclass
class DotStatsSummary:
    """有效細胞統計摘要。

    僅記錄 count；百分比於寫檔時計算，避免累積誤差。

    有效細胞定義：未排除（excluded=False）且 reddot >= 2。不要求 blackdot >= 1：
    有紅點但 0 黑點是切片平面沒切到該位點的真實低拷貝數細胞，丟掉它會同時拿走
    分子與分母中的一顆低值細胞，把比值與平均黑點數系統性往上拉（偏向判陽性）。

    決策 B (2026-08-17)：另需 is_her2_positive，避免新增的蛋白陰性族群稀釋既有
    ASCO/CAP 統計（該判讀原本的語意就是「蛋白陽性細胞」的 case-level 比值）。
    """

    valid_cells: int = 0      # 有效細胞總數
    total_her2: int = 0       # 有效細胞黑點總和 (Σ黑)，供 case 比值/平均黑點數
    total_cep17: int = 0      # 有效細胞紅點總和 (Σ紅)，供 case 比值
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
            if not r.excluded
            and r.is_her2_positive
            and r.cep17_dot_count >= 2
        ]
        ratio_lt2 = sum(1 for r in valid if r.her2_cep17_ratio < 2.0)
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


# 分型的「存在」門檻：低於這個佔比的族群視為偵測雜訊，不算存在。
# 論文自己沒給分型門檻，但引用了 ASCO/CAP 對異質性的定義（5-50% 的細胞），
# 取其下界 5%。全片幾萬顆細胞時，只看有無會讓三桶必定非零、永遠判成 Type 5。
_ITH_MIN_FRACTION = 0.05


@dataclass
class Her2IthSummary:
    """細胞異質性 Type 1-5 分型摘要（Hou et al. 2017, Breast Cancer Res Treat 166:447-457）。

    三類細胞由（蛋白陽性與否）×（基因擴增與否）決定；蛋白陽性但未擴增是真實的
    臨床組合（ASCO/CAP 2013 約 23% 的 3+ 病例），論文沒有對應的桶，故單獨計數
    並以百分比另行呈現，不摻進分型判定。
    """

    n_classic_positive: int = 0                 # 蛋白陽性 + 基因擴增
    n_non_classic_positive: int = 0             # 蛋白陰性 + 基因擴增
    n_negative: int = 0                         # 蛋白陰性 + 未擴增
    n_protein_positive_not_amplified: int = 0   # 蛋白陽性 + 未擴增（不進分型）
    n_excluded: int = 0                         # 已打 X，不進任何桶
    # n_excluded 的來源拆分：前者是染色/訊號品質，後兩者是 IHC↔DISH 對位品質。
    # 分開報是為了讓醫師知道該調染色還是該調對位（三者之和 == n_excluded）。
    n_excluded_low_cep17: int = 0               # 紅點 < 2，算不出 Score
    n_excluded_drop_out: int = 0                # 0 核、競爭落敗
    n_excluded_out_of_bounds: int = 0           # 0 核、壓在核心遮罩邊界

    @classmethod
    def from_results(cls, results: List[CellAnalysisResult]) -> "Her2IthSummary":
        s = cls()
        for r in results:
            t = _classify_cell_type(r)
            if t == "classic_positive":
                s.n_classic_positive += 1
            elif t == "non_classic_positive":
                s.n_non_classic_positive += 1
            elif t == "negative":
                s.n_negative += 1
            elif t == "protein_positive_not_amplified":
                s.n_protein_positive_not_amplified += 1
            else:
                s.n_excluded += 1
                reason = getattr(r, "exclusion_reason", "")
                if reason == "low_cep17":
                    s.n_excluded_low_cep17 += 1
                elif reason == "drop_out":
                    s.n_excluded_drop_out += 1
                elif reason == "out_of_bounds":
                    s.n_excluded_out_of_bounds += 1
        return s

    def type_verdict(self) -> str:
        """依「哪幾類佔比 >= _ITH_MIN_FRACTION」查表定型。

        分母只含論文定義的三桶；蛋白陽性未擴增不屬於任何 Type，不摻進分母，
        否則它一大就會把三桶全稀釋到門檻以下。

        論文 64 例皆為 IHC 3+ 選材，必有典型陽性，故沒有涵蓋「無典型陽性」的
        組合（全空 / 只有陰性 / 非典型+陰性）；那三種一律回 "Indeterminate"，
        不可默默落到某個論文有定義的 Type。
        """
        total = self.n_classic_positive + self.n_non_classic_positive + self.n_negative
        if total == 0:
            return "Indeterminate"

        has_classic = self.n_classic_positive / total >= _ITH_MIN_FRACTION
        has_non_classic = self.n_non_classic_positive / total >= _ITH_MIN_FRACTION
        has_negative = self.n_negative / total >= _ITH_MIN_FRACTION

        if has_classic and not has_non_classic and not has_negative:
            return "Type 1"
        elif not has_classic and has_non_classic and not has_negative:
            return "Type 2"
        elif has_classic and not has_non_classic and has_negative:
            return "Type 3"
        elif has_classic and has_non_classic and not has_negative:
            return "Type 4"
        elif has_classic and has_non_classic and has_negative:
            return "Type 5"
        else:
            return "Indeterminate"


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
    "Equivocal": "灰區 / 無法定論",
    "Not amplified (negative)": "未擴增 / 陰性",
    "Insufficient cells": "有效細胞不足，無法判讀",
}


def _her2ith_lines(ith: Her2IthSummary) -> List[str]:
    """細胞異質性分型區塊；分母只含蛋白陽性細胞（典型陽性 + 蛋白陽性未擴增）。"""
    verdict = ith.type_verdict()
    protein_positive_total = (
        ith.n_classic_positive + ith.n_protein_positive_not_amplified
    )
    if protein_positive_total > 0:
        pct_not_amp = (
            ith.n_protein_positive_not_amplified / protein_positive_total * 100
        )
        not_amp_str = (
            f"{pct_not_amp:.1f}% "
            f"({ith.n_protein_positive_not_amplified}/{protein_positive_total})"
        )
    else:
        not_amp_str = "N/A"

    # 三桶佔比就是 type_verdict() 的判定依據，一併印出來才看得出為什麼是這個 Type。
    typed_total = ith.n_classic_positive + ith.n_non_classic_positive + ith.n_negative

    def _n_pct(n: int) -> str:
        return str(n) if typed_total == 0 else f"{n}  ({n / typed_total * 100:.1f}%)"

    kv = [
        ("分型結果", verdict),
        ("典型陽性(咖啡色 + 黑/紅>=2)", _n_pct(ith.n_classic_positive)),
        ("非典型陽性(沒咖啡色 + 黑/紅>=2)", _n_pct(ith.n_non_classic_positive)),
        ("陰性(沒咖啡色 + 黑/紅<2)", _n_pct(ith.n_negative)),
        ("排除細胞", str(ith.n_excluded)),
        ("蛋白陽性但基因未擴增比例(咖啡色 + 黑/紅<2)", not_amp_str),
    ]
    kv_w = max(_display_width(k) for k, _ in kv) + 2

    bar = "=" * 46
    lines = [
        bar,
        f"  細胞異質性分型 Type 1-5（門檻 {_ITH_MIN_FRACTION:.0%}）",
        bar,
    ]
    lines += [f"  {_pad(k, kv_w)}{v}" for k, v in kv]
    lines.append("")
    return lines


def write_summary_csv(
    stats: DotStatsSummary,
    ith: Her2IthSummary,
    output_path: Path,
) -> Path:
    """把統計寫成對齊好讀的純文字報告（ASCO/CAP 2013 判讀 + 細胞異質性分型）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = stats.valid_cells

    def pct(count: int) -> str:
        return f"{count / n * 100:.1f}%" if n > 0 else "N/A"

    # case-level 判讀：比值 = Σ黑/Σ紅、平均黑點數 = Σ黑/有效細胞數。
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
        ("黑點/紅點 比值", ratio_str),
        ("平均黑點數", avg_str),
        ("黑點總數", str(stats.total_her2)),
        ("紅點總數", str(stats.total_cep17)),
        ("有效細胞數", str(n)),
    ]
    kv_w = max(_display_width(k) for k, _ in kv) + 2

    # ── 分佈區塊（count 右對齊 + 百分比）──
    dist = [
        ("比值 < 2", stats.ratio_lt2),
        ("比值 >= 2", stats.ratio_gte2),
        ("黑點數 < 4", stats.copy_lt4),
        ("黑點數 4-5", stats.copy_4to5),
        ("黑點數 >= 6", stats.copy_gte6),
    ]
    dist_w = max(_display_width(k) for k, _ in dist) + 2

    bar = "=" * 46
    sub = "-" * 46
    lines = [bar, "  ASCO/CAP 2013 判讀（全細胞統計）", bar]
    lines += [f"  {_pad(k, kv_w)}{v}" for k, v in kv]
    lines += ["", sub, "  細胞分佈（佔有效細胞比例）", sub]
    lines += [f"  {_pad(k, dist_w)}{c:>5}  ({pct(c)})" for k, c in dist]
    lines.append("")
    lines += _her2ith_lines(ith)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary 報告輸出: %s (%d valid cells)", output_path.name, n)
    return output_path


def export_summary_statistics(
    results: List[CellAnalysisResult],
    output_path: Path,
) -> Path:
    """Per-tile 便捷包裝：compute → write。"""
    stats = DotStatsSummary.from_results(results)
    ith = Her2IthSummary.from_results(results)
    return write_summary_csv(stats, ith, output_path)
