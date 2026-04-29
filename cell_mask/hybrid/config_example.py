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
    cellpose_dish_erode_radius: int = 3     # 核 mask 往內縮像素數，0=不縮

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

    # --- 多核排除 ---
    # 以 Cellpose DISH 模型輸出的細胞核 instance 與 IHC 細胞重疊計數判定多核。
    dot_blue_exclude_threshold: int = 2   # DISH 核重疊數 ≥ 此值 → 排除（多核細胞）

    # --- 群聚合併 ---
    dot_merge_distance: float = 3.0
    dot_black_merge_distance: float = 1.4

    # --- HER2 擴增判定 ---
    dot_amplification_ratio: float = 2.0
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

    # ========== 追溯性欄位 ==========
    model_version: str = "v1.0.0"
    slide_id: str = "unknown"
