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

    # ========== Cellpose 參數 (M2) ==========
    cellpose_diameter: Optional[float] = None
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0
    cellpose_gpu: bool = True
    clear_border_cells: bool = True

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
