"""ASCO/CAP 2013 case-level 判讀（全細胞統計）。

覆蓋 `m4_module/csv.py` 的有效細胞篩選與三條判讀門檻——先前無測試覆蓋。
"""
from __future__ import annotations

from hybrid_data_types import CellAnalysisResult
from m4_module.csv import DotStatsSummary, Her2IthSummary
from m4_export import export_summary_statistics, export_tile_csv


def _cell(black: int, red: int, excluded: bool = False, cid: int = 1) -> CellAnalysisResult:
    return CellAnalysisResult(
        cell_id=cid,
        centroid_x=0.0,
        centroid_y=0.0,
        # 決策 B：ASCO/CAP 統計只吃蛋白陽性細胞，本檔測的就是那條路徑。
        is_her2_positive=True,
        her2_dot_count=black,
        cep17_dot_count=red,
        her2_cep17_ratio=(black / red) if red else 0.0,
        excluded=excluded,
    )


def _verdict(cells, tmp_path, name="summary.txt") -> str:
    out = export_summary_statistics(cells, tmp_path / name)
    return out.read_text(encoding="utf-8")


def test_zero_black_cells_count_toward_denominator(tmp_path):
    """0 黑點細胞必須計入——丟掉它們會把比值與平均黑點數往上拉成偽陽性。

    5 顆 6黑/2紅 + 5 顆 0黑/2紅：
      計入 → Σ黑30 / Σ紅20 = 1.50、平均 30/10 = 3.00 → 陰性
      丟掉 → Σ黑30 / Σ紅10 = 3.00、平均 30/5  = 6.00 → 陽性（舊行為）
    """
    cells = [_cell(6, 2, cid=i) for i in range(5)]
    cells += [_cell(0, 2, cid=5 + i) for i in range(5)]

    stats = DotStatsSummary.from_results(cells)
    assert stats.valid_cells == 10
    assert stats.total_her2 == 30
    assert stats.total_cep17 == 20

    text = _verdict(cells, tmp_path)
    assert "Not amplified (negative)" in text
    assert "1.50" in text and "3.00" in text


def test_verdict_thresholds(tmp_path):
    # 比值 >= 2.0 → 陽性（平均黑點數 5.0 < 6.0，單靠比值成立）
    assert "Amplified (positive)" in _verdict([_cell(5, 2)] * 10, tmp_path, "a.txt")
    # 比值 1.5 < 2.0 但平均黑點數 6.0 >= 6.0 → 陽性
    assert "Amplified (positive)" in _verdict([_cell(6, 4)] * 10, tmp_path, "b.txt")
    # 比值 1.0、平均 4.0 → 落在 4.0~6.0 灰區
    assert "Equivocal" in _verdict([_cell(4, 4)] * 10, tmp_path, "c.txt")
    # 比值 0.5、平均 2.0 → 陰性
    assert "Not amplified (negative)" in _verdict([_cell(2, 4)] * 10, tmp_path, "d.txt")
    # 無有效細胞 → 不可判讀，而非陰性
    assert "Insufficient cells" in _verdict([], tmp_path, "e.txt")


def test_invalid_cells_are_dropped():
    """紅點 < 2 或已被 M3 打 X 的細胞不進統計。"""
    cells = [
        _cell(4, 2, cid=1),              # 有效
        _cell(9, 1, cid=2),              # 紅點不足
        _cell(9, 0, cid=3),              # 無訊號
        _cell(9, 4, excluded=True, cid=4),  # M3 已排除
    ]
    stats = DotStatsSummary.from_results(cells)
    assert stats.valid_cells == 1
    assert stats.total_her2 == 4 and stats.total_cep17 == 2


def test_report_uses_dot_wording(tmp_path):
    """報告只用紅/黑點用語，不出現 HER2/CEP17 探針名稱。"""
    text = _verdict([_cell(4, 2)] * 3, tmp_path)
    for label in ("黑點/紅點 比值", "平均黑點數", "黑點總數", "紅點總數", "有效細胞數"):
        assert label in text
    assert "HER2" not in text and "CEP17" not in text


def _typed_cell(cell_id, is_positive, is_amplified, excluded=False) -> CellAnalysisResult:
    return CellAnalysisResult(
        cell_id=cell_id,
        centroid_x=0.0,
        centroid_y=0.0,
        is_her2_positive=is_positive,
        is_amplified=is_amplified,
        excluded=excluded,
    )


def test_type_verdict_covers_all_five_types_and_the_indeterminate_edge_case():
    """Type 1-5 查表 + 論文未涵蓋的「無典型陽性」邊界，全部 8 種組合都有著落。"""
    classic = _typed_cell(1, True, True)
    non_classic = _typed_cell(2, False, True)
    negative = _typed_cell(3, False, False)
    protein_pos_not_amp = _typed_cell(4, True, False)

    assert Her2IthSummary.from_results([classic]).type_verdict() == "Type 1"
    assert Her2IthSummary.from_results([non_classic]).type_verdict() == "Type 2"
    assert Her2IthSummary.from_results([classic, negative]).type_verdict() == "Type 3"
    assert Her2IthSummary.from_results([classic, non_classic]).type_verdict() == "Type 4"
    assert Her2IthSummary.from_results(
        [classic, non_classic, negative]).type_verdict() == "Type 5"
    # 邊界：沒有典型陽性 → 不可套用論文分型
    assert Her2IthSummary.from_results([negative]).type_verdict() == "Indeterminate"
    assert Her2IthSummary.from_results(
        [non_classic, negative]).type_verdict() == "Indeterminate"
    assert Her2IthSummary.from_results([]).type_verdict() == "Indeterminate"
    assert Her2IthSummary.from_results(
        [protein_pos_not_amp]).type_verdict() == "Indeterminate"
    # 被打 X 的細胞不進任何桶
    assert Her2IthSummary.from_results(
        [_typed_cell(5, True, True, excluded=True)]).type_verdict() == "Indeterminate"


def test_protein_positive_not_amplified_excluded_from_type1_5_but_counted_separately():
    cells = [_typed_cell(1, True, True), _typed_cell(2, True, False), _typed_cell(3, True, False)]
    ith = Her2IthSummary.from_results(cells)
    assert ith.n_classic_positive == 1 and ith.n_protein_positive_not_amplified == 2
    assert ith.type_verdict() == "Type 1"


def test_asco_cap_stat_excludes_the_new_off_population():
    """決策 B：蛋白陰性族群不得稀釋既有 ASCO/CAP 分母。"""
    off_population_cell = CellAnalysisResult(
        cell_id=1, centroid_x=0.0, centroid_y=0.0,
        is_her2_positive=False, her2_dot_count=4, cep17_dot_count=2,
        her2_cep17_ratio=2.0, excluded=False,
    )
    assert DotStatsSummary.from_results([off_population_cell]).valid_cells == 0
    # 同一顆若是蛋白陽性就該計入，證明擋掉的是 is_her2_positive 而非其他條件
    assert DotStatsSummary.from_results([_cell(4, 2)]).valid_cells == 1


def test_csv_exports_is_her2_positive_and_cell_type_columns(tmp_path):
    cells = [_typed_cell(1, True, True), _typed_cell(2, False, True)]
    text = export_tile_csv(cells, tmp_path / "report.csv").read_text(encoding="utf-8")
    header = text.splitlines()[0]
    assert "is_her2_positive" in header and "cell_type" in header
    assert "classic_positive" in text and "non_classic_positive" in text
