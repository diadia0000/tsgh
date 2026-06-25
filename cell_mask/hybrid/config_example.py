"""
IHC-DISH Overlay & Analysis Pipeline 配置範例

此檔案為團隊成員參考用範本。
請複製此檔案為 config.py，並依據自身環境與需求調整參數。

使用方式:
  cp config_example.py config.py
  # 編輯 config.py 中的參數
"""

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional



@dataclass
class Config:
    """IHC-DISH 分析 Pipeline 配置（範例）

    此處列出所有可調參數及其預設值，供團隊成員參考。
    實際使用時請修改 config.py。
    """

    # ========== 路徑設定 ==========
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.resolve()
    )

    # IHC (Her2) tile 目錄
    ihc_tile_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "tile" / "her2"
    )
    # DISH tile 目錄
    dish_tile_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "tile" / "dish"
    )
    # IHC 測試圖片目錄
    ihc_test_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "test_picture" / "her2"
    )
    # DISH 測試圖片目錄
    dish_test_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "test_picture" / "dish"
    )
    # 輸出根目錄
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "output"
    )
    # 合併結果目錄（批次 tile）
    merge_tile_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "tile" / "merge"
    )
    # 合併結果目錄（test_picture）
    merge_test_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "test_picture" / "mearge"
    )
    # 舊欄位保留，供舊版程式碼相容使用
    merge_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "merge"
    )

    # ========== 模型路徑 ==========
    # UNet++ 細胞膜分割模型
    unet_model_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "models" / "best_model_unet.pth"
    )
    # Cellpose 實例分割模型 (retrained on IHC-DISH blended overlay)
    cellpose_model_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "models" / "cellpose_ihc_dish_best"
    )
    # Cellpose DISH 細胞核偵測模型（用於多核細胞排除，取代 HED 閾值法）
    cellpose_dish_model_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "models" / "cellpose_dish_best"
    )

    # ========== UNet++ 參數 ==========
    unet_encoder_name: str = "efficientnet-b4"
    unet_num_classes: int = 2
    unet_image_size: Tuple[int, int] = (1024, 1024)

    # ========== Core Mask 後處理參數 ==========
    # 形態學閉合核大小，連接預測中微小的斷裂
    core_close_kernel: int = 7

    # ========== Overlay 參數 (M1) ==========
    mask_blur_sigma: float = 0.0
    background_fill_value: int = 255
    overlay_alpha: float = 0.5

    # ========== Cellpose 參數 (M2: IHC-DISH 細胞分割) ==========
    cellpose_diameter: Optional[float] = None
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0
    cellpose_gpu: bool = True
    clear_border_cells: bool = True

    # ========== Cellpose 參數 (M3b: DISH 細胞核偵測，多核排除用) ==========
    cellpose_dish_diameter: Optional[float] = None
    cellpose_dish_flow_threshold: float = 0.4
    cellpose_dish_cellprob_threshold: float = 0.0

    # ========== DISH 訊號點偵測參數 (M3b) ==========
    # 方案: LAB + H-morphology + 多準則閘控（see docs/dish_dot_detection_spec.md v0.2）
    #
    # --- 全域 / 背景 ---
    dot_background_l_threshold: float = 95.0
    dot_seed_dilate_radius: int = 3
    dot_cell_roi_dilate: int = 0

    # --- 點位歸屬 (cell assignment) ---
    dot_assignment_min_overlap_ratio: float = 0.20
    dot_assignment_max_distance: float = 3.0
    dot_assignment_boundary_margin: float = 0.5

    # --- 紅點 (CEP17) ---
    dot_red_h: float = 12.0
    dot_red_a_min: float = 25.0
    dot_red_min_area: int = 7
    dot_red_max_area: int = 400
    dot_red_min_circularity: float = 0.55
    dot_red_min_solidity: float = 0.65
    dot_red_ring_gap: int = 2
    dot_red_ring_width: int = 5
    dot_red_min_contrast: float = 10.0

    # --- 黑點 (HER2) ---
    dot_black_h: float = 12.0
    dot_black_l_max: float = 58.0
    dot_black_min_area: int = 3
    dot_black_max_area: int = 260
    dot_black_max_radius: float = 9.0
    dot_black_min_circularity: float = 0.40
    dot_black_min_solidity: float = 0.50
    dot_black_ring_gap: int = 2
    dot_black_ring_width: int = 5
    dot_black_min_contrast: float = 14.0
    dot_black_min_ring_l: float = 30.0
    dot_black_max_chroma: float = 24.0
    dot_black_max_median_chroma: float = 22.0
    dot_black_max_p90_chroma: float = 32.0
    dot_black_p20_l_max: float = 56.0
    dot_black_seed_dilate_radius: int = 1
    dot_black_very_dark_l_max: float = 40.0
    dot_black_very_dark_min_contrast: float = 10.0

    # --- 細胞放大（M3 配對前處理；見 m3_cells_generator.enlarge_cell_instances）---
    # 做彈性匹配前，先把每顆綠色細胞 mask 實際往背景膨脹，使其等效「面積」放大此倍數
    # （skimage expand_labels，Voronoi 式外擴、細胞間不重疊）。放大版只用於 M3 配對與
    # 點偵測，藉此蓋到更多 DISH 核、提高配對成功率；不影響醫師檢視的 overlay（仍畫原始
    # M2 綠框）。1.0=不放大。注意：配對之後仍會套用 dish_elastic_expand_factor 的 reach，
    # 兩者會疊加；若想讓物理膨脹成為唯一機制，可把 dish_elastic_expand_factor 設回 1.0。
    cell_enlarge_area_factor: float = 1.5

    # --- 彈性匹配（以細胞為中心；見 m3_elastic_matching.py）---
    # 以細胞為中心的搜尋範圍倍數（綠框「面積」放大倍數）：每顆 IHC 細胞把自身面積
    # 放大此倍數，等效圓半徑 reach = max(sqrt(factor * area / π), min_reach_px)。
    # 候選有二：(a) 與綠框像素重疊的 DISH 核（重疊優先、一律候選）；(b) 質心落在
    # reach 內的核。再以一對一、重疊優先 + 最近優先配對（先 lock，落敗者往後找）。
    dish_elastic_expand_factor: float = 1.5
    # reach 的「絕對下限 px」：用以橋接純飄移（核與綠框無重疊但很近）的情形。0=不設
    # 下限（僅靠面積公式 + 重疊候選）。欲讓無重疊的近鄰核也能配到，調大此值（例如 30）。
    dish_elastic_min_reach_px: float = 0.0
    # 是否排除 drop-out 細胞（核數 0、曾有候選核卻競爭落敗且無可用核）。
    # 預設 True：這類 competition loser 打 X、不計入分析。
    # 注意：核數 0 但「從頭就沒有候選」的細胞不算 drop-out，照常計入(0/0)。
    dish_elastic_exclude_zero: bool = True
    # DISH 核需「落在 UNet++ core_mask 內」的最小像素比例：1.0=完全包含才留（核只要
    # 有任一 pixel 跑出 mask 就整顆丟）；<1.0 容忍 mask 邊緣鋸齒（例如 0.95 容許 5% 出界）。
    dish_nucleus_core_min_inside_ratio: float = 1.0

    # --- 群聚合併 ---
    dot_merge_distance: float = 3.0
    dot_black_merge_distance: float = 1.4

    # --- HER2 擴增判定（Score）---
    # Score(r,b)=HER2/CEP17。CEP17 紅點數 < 此值且「有訊號」（非 0/0）→ 紅點不足、
    # 無法計算 Score，該細胞直接排除打 X（excluded, low_cep17）；完全無訊號的 0/0
    # 細胞不打 X、照常計入（顯示 0/0）。
    score_cep17_min_count: int = 2
    # Score ≥ 此值才視為擴增（is_amplified）；否則 Score 歸 0。
    dot_amplification_ratio: float = 2.0
    # 保留供舊資料相容；新 Score 邏輯不再用 HER2 絕對數獨立判定擴增。
    dot_her2_count_threshold: int = 6

    # ========== 單細胞裁切參數 (M4) ==========
    cell_crop_size: int = 256

    # ========== 執行參數 ==========
    num_workers: int = 4
    batch_size: int = 8
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ========== 影像規格 ==========
    default_tile_size: int = 1024
    supported_extensions: List[str] = field(
        default_factory=lambda: [".tiff", ".tif", ".png", ".jpg", ".jpeg"]
    )

    # ========== 視窗化分割 / 重疊去重（大 patch sliding-window）==========
    # M2 細胞與 M3b DISH 核分割皆以 default_tile_size「重疊」視窗逐塊跑 Cellpose 後去重。
    # 相鄰視窗的重疊寬度（px）：應 ≥ 最大細胞/核直徑，邊界細胞才會至少在一個視窗
    # 內完整出現，避免被接縫切成兩顆。stride = default_tile_size - window_overlap_px。
    window_overlap_px: int = 256
    # 重疊去重門檻：兩 instance 交集 / min(面積) ≥ 此值即視為同一顆（保留較大者）。
    window_dedup_iomin: float = 0.5
    # 在所有 *_overlay.png 上畫出 1k 視窗虛線格（視覺參考用；重疊視窗下非單一接縫線）。
    draw_window_grid: bool = True

    # ========== 追溯性欄位 ==========
    model_version: str = "v1.0.0"
    slide_id: str = "unknown"
