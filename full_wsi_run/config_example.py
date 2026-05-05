"""Full WSI Pipeline 配置範例

請複製為 config.py 後填入你的路徑與參數：
    cp config_example.py config.py

「全圖推論」採取 sliding window：
  - UNet++: 內建滑動視窗 (full_wsi_run.unet_inference)
  - Cellpose: 以 wsi_window_size/wsi_window_overlap 分塊後逐塊推論
  - 各 window 走完整 M1→M4，結果合併為 slide-level 輸出

對硬體的要求主要來自 Cellpose 單塊 inference 以及選擇性的 stitched instance mask
(uint32 BigTIFF, 約為 H*W*4 bytes)。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class FullWSIConfig:
    """Full WSI Pipeline 配置（sliding-window 版）。"""

    # ========== WSI 輸入路徑 ==========
    # 單張 IHC (Her2) 全圖 TIFF
    ihc_wsi_path: Path = field(
        default_factory=lambda: Path(
            "/data/tsgh/thriple_image_layer/output/HER2_processed.tiff"
        )
    )
    # 單張 DISH 全圖 TIFF
    dish_wsi_path: Path = field(
        default_factory=lambda: Path(
            "/data/tsgh/thriple_image_layer/output/DISH_processed.tiff"
        )
    )
    # 選填：合併影像 WSI (用於產出 merge_overlay)。None 則跳過。
    merge_wsi_path: Optional[Path] = None

    # ========== 模型路徑 ==========
    unet_model_path: Path = field(
        default_factory=lambda: Path(
            "/home/sec312/project/tsgh/full_wsi_run/models/best_model_unet.pth"
        )
    )
    cellpose_model_path: Path = field(
        default_factory=lambda: Path(
            "/home/sec312/project/tsgh/full_wsi_run/models/cellpose_ihc_dish_best"
        )
    )
    cellpose_dish_model_path: Path = field(
        default_factory=lambda: Path(
            "/home/sec312/project/tsgh/full_wsi_run/models/cellpose_dish_best"
        )
    )

    # ========== 輸出 ==========
    output_dir: Path = field(
        default_factory=lambda: Path(
            "/home/sec312/project/tsgh/full_wsi_run/output"
        )
    )
    # 主要輸出：slide-level stitched overlay RGB BigTIFF（醫師判讀用）。
    # 包含細胞邊界 + HER2/CEP17 點標註，由 render_overlay_image 逐 window 拼接。
    save_stitched_overlay: bool = True

    # ========== Sliding Window 參數 ==========
    # 每塊 window 大小 (pixels)。建議 1024 或 2048；過大 Cellpose 會 OOM。
    wsi_window_size: int = 1024
    # window 之間的 overlap (pixels)。
    # Pipeline 採 owned-box 機制：每個 cell 由「centroid 落在哪個 window 的
    # owned box」的 window 計入；overlap 區同時被兩個 window 看到，但只會
    # 被擁有方寫入，邊界細胞因此不再被丟棄。
    # 為了讓擁有方能完整看到邊界細胞（含完整 dot 計數），overlap 必須
    # >= 細胞直徑。IHC-DISH 典型 ~50-80 px，預設 128 px 已足夠。設為 0
    # 會 fallback 為「每個 window 各自看到部分邊界細胞並重複計算」，程式
    # 會發出 warning。
    wsi_window_overlap: int = 128
    # 只處理有足夠前景像素的 window (避免大面積白背景浪費時間)。
    # 以整個 IHC window 平均亮度作為篩選；> wsi_skip_white_threshold 視為空白。
    # 設為 None 關閉此篩選。
    wsi_skip_white_threshold: Optional[float] = 245.0

    # ========== Dot Detection 並行化 ==========
    # detect_all_dots ThreadPoolExecutor 最大執行緒數。
    # 0 = os.cpu_count()；1 = 關閉並行（serial，適合 debug / 重現）。
    dots_workers: int = 0

    # ========== I/O Prefetch (DataLoader) ==========
    # DataLoader sub-process 數量；每個 worker 持有自己的 openslide handle。
    # 0 = 同步讀（不 prefetch，僅供 debug）；2-4 一般夠用，HDD/慢存儲可調更高。
    wsi_io_workers: int = 4
    # DataLoader prefetch_factor：每個 worker 預先準備幾個 batch。
    # 整體 buffered batch 上限 = num_workers * prefetch_factor。
    wsi_io_prefetch_factor: int = 2

    # ========== Batch Sizes ==========
    # WSI window batch size (UNet++ + window-level processing)
    wsi_batch_size: int = 4
    # Cellpose predict_batch batch size
    cellpose_batch_size: int = 4

    # ========== BigTIFF 壓縮 (overlay) ==========
    stitched_overlay_pyramidal: bool = True
    stitched_overlay_jpeg_quality: int = 85
    stitched_overlay_tile_size: int = 256

    # ========== BigTIFF 壓縮 (core_mask) ==========
    stitched_core_pyramidal: bool = True
    stitched_core_jpeg_quality: int = 85
    stitched_core_tile_size: int = 256

    # ========== UNet++ 參數 ==========
    unet_encoder_name: str = "efficientnet-b4"
    unet_num_classes: int = 2
    unet_image_size: Tuple[int, int] = (1024, 1024)

    # ========== Core Mask 後處理 ==========
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

    # ========== Cellpose 參數 (M3b: DISH 核偵測) ==========
    cellpose_dish_diameter: Optional[float] = None
    cellpose_dish_flow_threshold: float = 0.4
    cellpose_dish_cellprob_threshold: float = 0.0
    cellpose_dish_erode_radius: int = 3

    # ========== DISH 訊號點偵測 (M3b) ==========
    dot_background_l_threshold: float = 95.0
    dot_seed_dilate_radius: int = 3
    dot_cell_roi_dilate: int = 0

    dot_assignment_min_overlap_ratio: float = 0.20
    dot_assignment_max_distance: float = 3.0
    dot_assignment_boundary_margin: float = 0.5

    dot_red_h: float = 12.0
    dot_red_a_min: float = 25.0
    dot_red_min_area: int = 7
    dot_red_max_area: int = 400
    dot_red_min_circularity: float = 0.55
    dot_red_min_solidity: float = 0.65
    dot_red_ring_gap: int = 2
    dot_red_ring_width: int = 5
    dot_red_min_contrast: float = 10.0

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

    dot_blue_min_signal: float = 0.05
    dot_blue_min_area: int = 40
    dot_blue_max_area: int = 2500
    dot_blue_expected_radius: int = 6
    dot_blue_close_radius: int = 1
    dot_blue_exclude_threshold: int = 2

    dot_merge_distance: float = 3.0
    dot_black_merge_distance: float = 1.4

    dot_amplification_ratio: float = 2.0
    dot_her2_count_threshold: int = 6

    # ========== 單細胞裁切 (M4) ==========
    cell_crop_size: int = 256

    # ========== 追溯性 ==========
    model_version: str = "v1.0.0-fullwsi"
    slide_id: str = "wsi_run"

    # ========== 影像規格 ==========
    supported_extensions: List[str] = field(
        default_factory=lambda: [".tiff", ".tif", ".png"]
    )


config = FullWSIConfig()
