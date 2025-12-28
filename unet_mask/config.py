"""
HER2 語義分割訓練配置檔案

包含所有訓練超參數、路徑設定和硬體優化參數
針對 NVIDIA RTX 5090 32GB + Intel Ultra 265K + 64GB RAM 優化
"""
import os
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Config:
    """訓練配置類"""
    
    # ========== 路徑設定 ==========
    # 訓練資料夾路徑
    train_image_dir: str = "/home/sec312/tsgh/unet_mask/tile/train/her2/"
    # 預處理後的 mask 路徑
    mask_dir: str = "/home/sec312/tsgh/unet_mask/presudomask/her2/"
    # 模型保存路徑
    model_save_dir: str = "/home/sec312/tsgh/unet_mask/models/"
    # 訓練日誌路徑
    log_dir: str = "/home/sec312/tsgh/unet_mask/logs/"
    
    # ========== 數據規格 ==========
    # 影像尺寸
    image_size: Tuple[int, int] = (1024, 1024)
    # 類別數量 (背景=0, 細胞內部=1, 細胞膜=2)
    num_classes: int = 3
    # 類別名稱
    class_names: List[str] = field(default_factory=lambda: ["Background", "Interior", "Membrane"])
    
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
    encoder_name: str = "efficientnet-b4"
    # 預訓練權重
    encoder_weights: str = "imagenet"
    # 是否使用輔助分類器
    aux_params: Optional[dict] = None
    
    # ========== 損失函數 ==========
    # Dice Loss 權重
    dice_weight: float = 0.5
    # Cross-Entropy Loss 權重
    ce_weight: float = 0.5
    # 類別權重 (用於應對類別不平衡，細胞膜權重較高)
    class_weights: List[float] = field(default_factory=lambda: [0.5, 1.0, 3.0])
    
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
    batch_size: int = 8
    # DataLoader num_workers (根據 CPU 核心數設定)
    num_workers: int = 16
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
    
    def __post_init__(self):
        """初始化後建立必要的目錄"""
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
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
