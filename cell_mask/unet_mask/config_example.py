"""
HER2 語義分割訓練配置檔案

包含所有訓練超參數、路徑設定和硬體優化參數
針對 NVIDIA RTX 5090 32GB + Intel Ultra 265K + 64GB RAM 優化
"""
import os
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass
class Config:
    """訓練配置類"""
    
    # ========== 路徑設定 ==========
    # 基礎路徑
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.resolve())
    
    # 訓練資料夾路徑
    train_image_dir: Path = field(default_factory=lambda: Path(__file__).parent / "train/her2_chose")
    # 預處理後的 mask 路徑
    mask_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output/mask")
    # 模型保存路徑
    model_save_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output/model")
    # 訓練日誌路徑
    log_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output/log")
    
    # ========== 數據規格 ==========
    # 影像尺寸
    image_size: Tuple[int, int] = (1024, 1024)
    # 類別數量 (二分類: 非膜=0, 咖啡色膜=1)
    num_classes: int = 2
    # 類別名稱
    class_names: List[str] = field(default_factory=lambda: ["Non-membrane", "Membrane"])
    
    # ========== 數據集分割 ==========
    # 訓練集比例 80%
    train_ratio: float = 0.80
    # 驗證集比例 10%
    val_ratio: float = 0.10
    # 測試集比例 10%
    test_ratio: float = 0.10
    # 隨機種子
    random_seed: int = 42
    
    # ========== 模型架構 ==========
    # 使用 Unet++ 搭配 EfficientNet-B4 編碼器
    # EfficientNet 優點:
    # - 支援任意尺寸輸入
    # - SMP 原生支援，穩定性高
    # - 參數效率高，效能優秀
    model_name: str = "unetplusplus"
    encoder_name: str = "densenet121"
    # 預訓練權重
    encoder_weights: str = "imagenet"
    # 是否使用輔助分類器
    aux_params: Optional[dict] = None
    
    # ========== 損失函數 ==========
    # Dice Loss 權重
    dice_weight: float = 0.5
    # Cross-Entropy Loss 權重
    ce_weight: float = 0.5
    # 類別權重 (用於應對類別不平衡，二分類: [非膜, 膜])
    class_weights: List[float] = field(default_factory=lambda: [0.5, 2.0])
    
    # ========== 優化器設定 ==========
    # 學習率
    learning_rate: float = 1e-4
    # 權重衰減
    weight_decay: float = 1e-4
    # AdamW betas
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # ========== 學習率調度器 ==========
    # CosineAnnealingLR 最小學習率
    min_lr: float = 1e-6
    # T_max (通常設為 epochs)
    t_max: int = 100
    
    # ========== 訓練參數 ==========
    # 總訓練 epochs
    epochs: int = 100
    # Batch size (針對 32GB 顯存優化，1024 解析度下預計可設為 4-8)
    batch_size: int = 6
    # DataLoader num_workers (根據 CPU 核心數設定)
    num_workers: int = 12
    # 是否 pin_memory
    pin_memory: bool = True
    # 梯度累積步數 (用於模擬更大的 batch size)
    gradient_accumulation_steps: int = 4
    
    # ========== 混合精度訓練 ==========
    # 是否啟用 AMP
    use_amp: bool = True
    
    # ========== 模型保存 ==========
    # 保存最佳模型的數量
    save_top_k: int = 1
    # 監控指標
    monitor_metric: str = "val_miou"
    # 監控模式 (max 或 min)
    monitor_mode: str = "max"
    # 早停耐心值
    early_stopping_patience: int = 15
    
    # ========== 數據增強參數 ==========
    # 訓練時使用的數據增強
    augmentation: dict = field(default_factory=lambda: {
        "random_rotate90": True,
        "horizontal_flip": True,
        "vertical_flip": True,
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.05
        },
        "gaussian_blur": {
            "blur_limit": (3, 7),
            "p": 0.3
        },
        "elastic_transform": {
            "alpha": 50,
            "sigma": 5,
            "p": 0.3
        }
    })
    
    # ========== 預處理參數 (來自 her2_mask.py) ==========
    # 這些參數用於生成 pseudo mask
    min_dab_od: float = 0.13
    dab_dominance: float = 1.115
    closing_radius: int = 2
    min_total_od: float = 0.08
    min_interior_size: int = 50  # 最小內部區域大小
    min_hole_size: int = 50       # 最小空洞大小
    
    # ========== Mask 生成參數 (Nucleus-Based / K-Means) ==========
    # 輸入影像路徑 (檔案或目錄)
    kmeans_input_path: Path = field(default_factory=lambda: Path(__file__).parent / "train/her2_chose")
    # Mask 輸出目錄
    kmeans_mask_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output/kmeans_mask")
    # 視覺化輸出目錄
    kmeans_vis_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output/kmeans_vis")
    # 是否儲存視覺化結果
    kmeans_save_visualization: bool = True
    
    # --- 細胞核為中心方法參數 (預設使用) ---
    # 是否使用舊的 K-Means 方法 (False = 使用新方法)
    use_kmeans_method: bool = False
    # 是否使用 LAB 色彩空間方法 (True = 使用 LAB, False = 使用舊方法)
    use_lab_method: bool = True
    
    # --- LAB 色彩空間分析參數 ---
    # 最小亮度 (排除太暗區域)
    lab_l_min: float = 15.0
    # 最大亮度 (排除太亮區域/背景)
    lab_l_max: float = 85.0
    # 是否融合 DAB 通道 (HED) 進行雙重確認 (True=更精準, False=只用 LAB)
    use_dab_fusion: bool = True
    # 膜環寬度 (像素) - 從細胞核邊緣向外延伸的距離
    membrane_width: int = 5
    # 最小細胞核大小 (像素) - 小於此值的區域不視為細胞核
    min_nucleus_size: int = 50
    
    # --- 形態學處理參數 ---
    # 最小區域大小（移除小於此值的區域）
    min_region_size: int = 50
    # 最小空洞大小（填補小於此值的空洞）
    min_hole_size: int = 50
    
    # ========== 視覺化參數 (二分類) ==========
    # Overlay 透明度 (0.0 = 完全透明, 1.0 = 完全不透明)
    vis_overlay_alpha: float = 0.5
    # 細胞膜區域顏色 (RGB) - 二分類只需要膜的顏色
    vis_membrane_color: Tuple[int, int, int] = (255, 0, 0)      # 紅色
    # 支援的影像副檔名
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".tiff", ".tif", ".png", ".jpg", ".jpeg"
    ])
    
    # ========== Debug Visualizer 參數 ==========
    # 預設最大樣本數量
    debug_max_samples: int = 20
    # HTML 報告標題
    debug_report_title: str = "K-Means Mask Debug Report"
    # Cluster 視覺化高度
    debug_cluster_vis_height: int = 60
    
    def __post_init__(self) -> None:
        """初始化後建立必要的目錄"""
        for dir_path in [self.mask_dir, self.model_save_dir, self.log_dir]:
            if dir_path:
                # 確保路徑是 Path 對象
                if isinstance(dir_path, str):
                    dir_path = Path(dir_path)
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
