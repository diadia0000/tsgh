"""
HER2 兩階段陽性細胞分割 Pipeline 配置檔案

Pipeline 架構:
  Stage 1: 背景分割 (閾值法) → 組織 vs 背景
  Stage 2: UNet++ 膜分割 → 棕色細胞膜 Mask
  Stage 3: Contour Fill → 陽性細胞內部 Mask

針對 NVIDIA RTX 5090 32GB + Intel Ultra 265K + 64GB RAM 優化
"""
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass
class Config:
    """兩階段分割 Pipeline 配置"""

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
        default_factory=lambda: Path(__file__).parent / "output/model/b4"
    )
    # ========== Stage 1: Tumor Nest 聚集參數 ==========
    # 利用形態學將相鄰的膜連成的巨觀防護罩
    nest_close_kernel: int = 50      # 閉合運算的 kernel size (越大連得越遠，可跨越細胞間隙)
    nest_min_area: int = 1500        # 組織塊的最小面積 (降低閾值以保留小群落陽性細胞)
    nest_open_kernel: int = 15       # 開運算去毛邊的 kernel size
    nest_expand_kernel: int = 75     # 膨脹運算的 kernel size (大幅增加以確保包覆邊緣細胞)

    # ========== Stage 2: HSV 棕色膜偽標籤參數 ==========
    # HSV 棕色範圍下界 (H, S, V)
    # 實測 brown pixel: H 5%ile=10, 95%ile=17, S 5%ile=27
    # 放寬以涵蓋所有棕色深淺變化
    hsv_brown_lower: Tuple[int, int, int] = (8, 25, 50)
    # HSV 棕色範圍上界
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
    # 膜膨脹核大小 (用於連通斷裂的膜)
    membrane_dilate_kernel: int = 7
    # 最小細胞面積 (低於此值視為雜訊，單位: pixels)
    min_cell_area: int = 500
    # 最大細胞面積 (超過此值視為非單一細胞，單位: pixels)
    max_cell_area: int = 5000

    # ========== QuPath 染色矩陣 (Color Deconvolution) ==========
    stain_matrix: List[List[float]] = field(default_factory=lambda: [
        [0.651, 0.701, 0.290],   # Hematoxylin
        [0.269, 0.568, 0.778],   # DAB
        [0.633, -0.713, 0.302],  # Residual
    ])

    # ========== 填充細胞區域參數 ==========
    # DAB 濃度固定閾值 (separate_stains 輸出的光學密度，通常 0~1.5)
    dab_threshold: float = 0.037
    fill_close_kernel: int = 12
    fill_min_cell_area: int = 100
    # 邊緣細胞最大面積 (碰邊界的背景區域小於此值視為邊緣細胞內部)
    fill_max_edge_hole_area: int = 10000
    # 形態學開運算核大小 (移除填充後殘留的小突起雜訊，0=不執行)
    fill_open_kernel: int = 5

    # ========== 數據規格 ==========
    image_size: Tuple[int, int] = (1024, 1024)
    # 二分類: 非膜=0, 棕色膜=1
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
    encoder_name: str = "timm-efficientnet-b4"
    encoder_weights: str = "imagenet"
    aux_params: Optional[dict] = None

    # ========== 損失函數 ==========
    dice_weight: float = 0.5
    focal_weight: float = 0.5
    # 類別權重 [非膜, 膜]
    class_weights: List[float] = field(
        default_factory=lambda: [0.5, 2.0]
    )

    # ========== 優化器設定 ==========
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)

    # ========== 學習率調度器 ==========
    min_lr: float = 1e-6
    t_max: int = 60

    # ========== 訓練參數 ==========
    epochs: int = 60
    batch_size: int = 7
    num_workers: int = 4
    pin_memory: bool = True
    gradient_accumulation_steps: int = 2

    # ========== 混合精度訓練 ==========
    use_amp: bool = True

    # ========== 模型保存 ==========
    save_top_k: int = 1
    monitor_metric: str = "val_miou"
    monitor_mode: str = "max"
    early_stopping_patience: int = 10

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
    # 推論輸入目錄 (單張影像或目錄)
    inference_input_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "tile/test"
    )
    inference_output_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "output/inference_results"
    )
    inference_batch_size: int = 4

    # ========== Phase 1: HSV + DAB 膜先驗 (Membrane Prior) ==========
    # HSV 棕色膜下界 (H, S, V)
    hsv_membrane_lower: Tuple[int, int, int] = (8, 25, 50)
    # HSV 棕色膜上界
    hsv_membrane_upper: Tuple[int, int, int] = (20, 255, 205)
    # 是否融合 HSV + DAB 作為膜先驗
    use_hsv_dab_fusion: bool = True
    # 膜先驗最低信賴度 (低於此值的先驗像素視為無效)
    membrane_prior_min_confidence: float = 0.1
    # 是否儲存 soft confidence map (float32 .npy)
    membrane_prior_save_soft_map: bool = True

    # ========== Phase 1: 細胞核偵測與過濾 ==========
    # 最小細胞核面積 (pixels)，低於此值視為碎片
    nucleus_min_area: int = 80
    # 最大細胞核面積 (pixels)，高於此值可能為融合核
    nucleus_max_area: int = 5000
    # 細胞核最大離心率 (eccentricity)，超過此值可能為間質細胞
    nucleus_max_eccentricity: float = 0.85
    # 淋巴球最大面積 (pixels)，小而圓的核視為淋巴球排除
    lymphocyte_max_area: int = 150
    # 間質細胞最低離心率 (高離心率 = 拉長形狀)
    stroma_min_eccentricity: float = 0.90

    # ========== Phase 1: 膜環分析 (Ring Analysis) ==========
    # 膜環內半徑 (從核心邊緣向外的偏移, pixels)
    positive_ring_inner_radius: int = 3
    # 膜環外半徑 (從核心邊緣向外的偏移, pixels)
    positive_ring_outer_radius: int = 15

    # ========== Phase 1: 陽性細胞判定閾值 ==========
    # 膜環上的平均膜機率必須超過此值
    positive_membrane_prob_threshold: float = 0.35
    # 膜環上的平均 DAB 強度必須超過此值
    positive_dab_intensity_threshold: float = 0.15
    # 膜環完整度 (圓周支持率，0-1) 必須超過此值
    positive_ring_completeness_threshold: float = 0.35
    # 陽性細胞最小面積 (填充後, pixels)
    positive_cell_min_area: int = 300
    # 陽性細胞最大面積 (填充後, pixels)
    positive_cell_max_area: int = 30000
    # 不確定的候選細胞是否直接拒絕 (True=高精度模式)
    positive_reject_uncertain: bool = True
    # 是否儲存 debug overlay (含接受/拒絕標記)
    positive_save_debug_overlay: bool = True

    def __post_init__(self) -> None:
        """初始化後建立必要的目錄

        只預建 mask / model 兩個目錄;inference_results 由 inference.py
        執行時自行建立,vis 由 lab_mask_generator 視需要建立。
        """
        for dir_path in [
            self.mask_dir,
            self.model_save_dir,
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
