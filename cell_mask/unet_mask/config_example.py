"""
HER2 兩階段陽性細胞分割 Pipeline 配置範例

Pipeline 架構:
  Stage 1: 背景分割 (閾值法) -> 組織 vs 背景
  Stage 2: UNet++ 膜分割 -> 棕色細胞膜 Mask
  Stage 3: Contour Fill -> 陽性細胞內部 Mask

針對 NVIDIA RTX 5090 32GB + Intel Ultra 265K + 64GB RAM 優化
"""
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass
class Config:
    """兩階段分割 Pipeline 配置範例"""

    # ========== 路徑設定 ==========
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.resolve()
    )
    # 訓練影像目錄
    train_image_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "tile/train/her2_chose"
    )
    # 偽標籤 mask 輸出目錄
    mask_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "output/mask"
    )
    # 模型保存路徑
    model_save_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "output/model"
    )
    # 訓練日誌路徑
    log_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "output/log"
    )

    # ========== Stage 1: Tumor Nest 聚集參數 ==========
    # 利用形態學將相鄰的膜連成的巨觀防護罩
    nest_close_kernel: int = 45
    nest_min_area: int = 1000
    nest_open_kernel: int = 15
    nest_expand_kernel: int = 75

    # ========== Stage 2: HSV 棕色膜偽標籤參數 ==========
    hsv_brown_lower: Tuple[int, int, int] = (8, 25, 50)
    hsv_brown_upper: Tuple[int, int, int] = (20, 255, 205)

    # 偽標籤生成路徑
    pseudo_label_input_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "tile/train/her2_chose"
    )
    pseudo_label_mask_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "output/mask"
    )
    pseudo_label_vis_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "output/vis"
    )
    # 是否儲存視覺化
    save_visualization: bool = True

    # ========== Stage 3: Contour Fill 參數 ==========
    membrane_dilate_kernel: int = 5
    min_cell_area: int = 50
    max_cell_area: int = 5000

    # ========== QuPath 染色矩陣 (Color Deconvolution) ==========
    stain_matrix: List[List[float]] = field(default_factory=lambda: [
        [0.651, 0.701, 0.290],
        [0.269, 0.568, 0.778],
        [0.633, -0.713, 0.302],
    ])

    # ========== 填充細胞區域參數 ==========
    dab_threshold: float = 0.03
    fill_close_kernel: int = 11
    fill_min_cell_area: int = 200
    fill_max_edge_hole_area: int = 5000

    # ========== 數據規格 ==========
    image_size: Tuple[int, int] = (1024, 1024)
    num_classes: int = 2
    class_names: List[str] = field(
        default_factory=lambda: ["Non-membrane", "Membrane"]
    )

    # ========== 數據集分割 ==========
    train_ratio: float = 0.85
    val_ratio: float = 0.15
    random_seed: int = 42

    # ========== 模型架構 ==========
    model_name: str = "unetplusplus"
    encoder_name: str = "efficientnet-b4"
    encoder_weights: str = "imagenet"
    aux_params: Optional[dict] = None

    # ========== 損失函數 ==========
    dice_weight: float = 0.5
    ce_weight: float = 0.5
    class_weights: List[float] = field(
        default_factory=lambda: [0.5, 2.0]
    )

    # ========== 優化器設定 ==========
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)

    # ========== 學習率調度器 ==========
    min_lr: float = 1e-6
    t_max: int = 100

    # ========== 訓練參數 ==========
    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 16
    pin_memory: bool = True
    gradient_accumulation_steps: int = 6

    # ========== 混合精度訓練 ==========
    use_amp: bool = True

    # ========== 模型保存 ==========
    save_top_k: int = 1
    monitor_metric: str = "val_miou"
    monitor_mode: str = "max"
    early_stopping_patience: int = 15

    # ========== 數據增強參數 ==========
    augmentation: dict = field(default_factory=lambda: {
        "random_rotate90": True,
        "horizontal_flip": True,
        "vertical_flip": True,
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.05,
        },
        "gaussian_blur": {
            "blur_limit": (3, 7),
            "p": 0.3,
        },
        "elastic_transform": {
            "alpha": 50,
            "sigma": 5,
            "p": 0.3,
        },
    })

    # ========== 視覺化參數 ==========
    vis_overlay_alpha: float = 0.5
    vis_membrane_color: Tuple[int, int, int] = (255, 0, 0)
    supported_extensions: List[str] = field(
        default_factory=lambda: [".tiff", ".tif", ".png", ".jpg", ".jpeg"]
    )

    # ========== 推論設定 ==========
    inference_input_dir: Path = field(
        default_factory=lambda: Path("/home/sec312/tsgh/cell_mask/unet_mask/tile/test")
    )
    inference_output_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "output/result"
    )
    inference_batch_size: int = 4

    # ========== HER2 陽性細胞抽取設定 ==========
    extraction_input_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "tile/test"
    )
    extraction_output_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "output/her2_positive_cells"
    )
    extraction_membrane_prob_threshold: float = 0.5
    extraction_positive_overlap_threshold: float = 0.05
    extraction_complete_boundary_threshold: float = 0.7
    extraction_dab_intensity_threshold: float = 0.15
    extraction_save_overlay: bool = True
    extraction_save_label_mask: bool = True
    extraction_export_individual_json: bool = True

    # ========== Phase 1: HSV + DAB 膜先驗 (Membrane Prior) ==========
    hsv_membrane_lower: Tuple[int, int, int] = (8, 25, 50)
    hsv_membrane_upper: Tuple[int, int, int] = (20, 255, 205)
    use_hsv_dab_fusion: bool = True
    membrane_prior_min_confidence: float = 0.1
    membrane_prior_save_soft_map: bool = True

    # ========== Phase 1: 細胞核偵測與過濾 ==========
    nucleus_min_area: int = 80
    nucleus_max_area: int = 5000
    nucleus_max_eccentricity: float = 0.85
    lymphocyte_max_area: int = 150
    stroma_min_eccentricity: float = 0.90

    # ========== Phase 1: 膜環分析 (Ring Analysis) ==========
    positive_ring_inner_radius: int = 3
    positive_ring_outer_radius: int = 15

    # ========== Phase 1: 陽性細胞判定閾值 ==========
    positive_membrane_prob_threshold: float = 0.35
    positive_dab_intensity_threshold: float = 0.15
    positive_ring_completeness_threshold: float = 0.35
    positive_cell_min_area: int = 300
    positive_cell_max_area: int = 30000
    positive_reject_uncertain: bool = True
    positive_save_debug_overlay: bool = True

    def __post_init__(self) -> None:
        """初始化後建立必要的目錄"""
        for dir_path in [
            self.mask_dir,
            self.model_save_dir,
            self.log_dir,
            self.inference_output_dir,
            self.extraction_output_dir,
        ]:
            if dir_path:
                dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path
                dir_path.mkdir(parents=True, exist_ok=True)

    @property
    def device(self) -> torch.device:
        """獲取計算設備"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def effective_batch_size(self) -> int:
        """有效 batch size (考慮梯度累積)"""
        return self.batch_size * self.gradient_accumulation_steps


# 建立全局配置實例
config = Config()
