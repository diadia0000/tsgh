"""Shared data model types for the hybrid pipeline.

CellAnalysisResult, DetectedDot, CellDotResult are defined here so
that m3_module and m4_module both import from this neutral location
instead of m4 depending on m3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class DetectedDot:
    """單一偵測點（全 tile 座標）。"""

    y: float
    x: float
    radius: float
    dot_type: str          # "her2" | "cep17"
    cell_id: int           # 0 表示不在任何 Cellpose 細胞內
    area: int
    circularity: float
    solidity: float
    contrast: float        # 紅: mean_a_dot - mean_a_ring; 黑: mean_L_ring - mean_L_dot
    score: float           # 排序用（紅: mean_a; 黑: -mean_L）


@dataclass
class CellAnalysisResult:
    """單一細胞的分析結果。

    Attributes:
        cell_id: 細胞實例 ID (>0)。
        centroid_x: 細胞質心 X 座標 (pixels)。
        centroid_y: 細胞質心 Y 座標 (pixels)。
        is_her2_positive: 是否為 HER2 陽性。
        hematoxylin_ratio: 細胞區域中 Hematoxylin 陽性像素佔比。
    """

    cell_id: int
    centroid_x: float
    centroid_y: float
    is_her2_positive: bool
    hematoxylin_ratio: float
    # --- M3b DISH 點位偵測結果（預設值允許舊流程零變動沿用）---
    her2_dot_count: int = 0
    cep17_dot_count: int = 0
    her2_cep17_ratio: float = 0.0
    is_amplified: bool = False
    score: float = 0.0
    blue_region_count: int = 0
    excluded: bool = False


@dataclass
class CellDotResult:
    """單一細胞的點位計數結果。"""

    cell_id: int
    her2_dot_count: int = 0
    cep17_dot_count: int = 0
    her2_cep17_ratio: float = 0.0        # float("inf") 當 cep17_dot_count == 0
    is_amplified: bool = False
    score: float = 0.0                   # Score(r,b)=r/b（cep17≥2 且 ratio≥2 才>0，否則 0）
    blue_region_count: int = 0
    excluded: bool = False               # drop-out(0 核、競爭落敗) / cep17<2 → 排除、打 X
    exclude_reason: str = ""             # "" | "drop_out" | "out_of_bounds_nucleus" | "low_cep17"
    her2_dots: List[DetectedDot] = field(default_factory=list)
    cep17_dots: List[DetectedDot] = field(default_factory=list)
    # elastic matching 認領到的 DISH 核 ID（用於視覺化飄移箭頭與粉色輪廓）
    assigned_dish_ids: List[int] = field(default_factory=list)
