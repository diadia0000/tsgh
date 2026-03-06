"""
IHC-DISH Overlay & Analysis Pipeline 配置範例

包含所有路徑設定、模型參數、色彩解卷積矩陣、偵測閾值等。
使用者應複製為 config.py 並依需要調整參數。
"""

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class Config:
    """IHC-DISH 分析 Pipeline 配置"""

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
    # 合併結果目錄
    merge_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "merge"
    )

    # ========== 模型路徑 ==========
    # UNet++ 細胞膜分割模型
    unet_model_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "models" / "best_model_unet.pth"
    )
    # Cellpose 實例分割模型
    cellpose_model_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "models" / "dish_cellpose"
    )

    # ========== UNet++ 參數 ==========
    unet_encoder_name: str = "efficientnet-b4"
    unet_num_classes: int = 2
    unet_image_size: Tuple[int, int] = (1024, 1024)

    # ========== Core Extraction 參數 ==========
    membrane_dilate_kernel: int = 7
    membrane_close_kernel: int = 20
    max_boundary_gap: int = 400

    # ========== Overlay 參數 (M1) ==========
    mask_blur_sigma: float = 0.0
    background_fill_value: int = 0

    # ========== Cellpose 參數 (M2) ==========
    cellpose_diameter: Optional[float] = None
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0
    cellpose_gpu: bool = True
    clear_border_cells: bool = True

    # ========== 色彩解卷積參數 (M3) ==========
    # OD 矩陣列向量: [Red-stain, Black-stain, Residual]
    # 每列代表一種染色劑在 R/G/B 通道的光學密度
    od_matrix: np.ndarray = field(
        default_factory=lambda: np.array([
            [0.18, 0.20, 0.08],   # Red stain OD
            [0.10, 0.21, 0.29],   # Black stain OD
            [0.01, 0.13, 0.01],   # Residual channel
        ], dtype=np.float64)
    )

    # ========== Blob 偵測參數 (M3) ==========
    # LoG (Laplacian of Gaussian) sigma 範圍
    log_min_sigma: float = 1.0
    log_max_sigma: float = 5.0
    log_num_sigma: int = 5
    log_threshold: float = 0.02
    # Blob 最小面積 (pixels) — 過小的偵測視為雜訊
    min_blob_area: int = 3
    # 叢集面積因子: 連通元件面積 > factor * avg_single_dot_area → 拆分為多顆
    cluster_area_factor: float = 2.5

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
