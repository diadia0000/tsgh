"""
配準設定模組

此模組定義 WSIReg 配準流程的所有配置參數，
包括影像模態、配準路徑和 elastix 參數設定。
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModalityConfig:
    """單一影像模態的配置"""
    
    name: str
    filename: str
    resolution: float = 0.25  # 輸入解析度 µm/px (40X)
    output_resolution: Optional[float] = 0.5  # 輸出解析度 µm/px (20X = level 1)
    downsampling: int = 16  # 配準時降採樣倍率 (4 = 使用 1/4 解析度配準，節省記憶體)
    channel_names: list[str] = field(default_factory=lambda: ["default"])
    channel_colors: list[str] = field(default_factory=lambda: ["gray"])
    preprocessing: Optional[dict] = None


@dataclass
class RegistrationPathConfig:
    """配準路徑配置"""
    
    moving: str
    fixed: str
    stages: list[str] = field(default_factory=lambda: ["rigid", "affine", "nl"])


@dataclass
class ElastixParams:
    """Elastix 非剛性配準參數"""
    
    # B-spline 網格間距（越大越平滑）
    grid_spacing: float = 20.0
    
    # 多解析度層數
    num_resolutions: int = 4
    
    # 每層最大迭代次數
    max_iterations: int = 512
    
    # 正則化權重（彎曲能量懲罰，越大變形越平滑）
    bending_energy_weight: float = 5.0
    
    # CPU 多線程數（0 = 自動使用所有可用核心）
    num_threads: int = 0
    
    def to_elastix_dict(self) -> dict:
        """轉換為 elastix 參數字典格式"""
        params = {
            "FinalGridSpacingInPhysicalUnits": [str(self.grid_spacing)],
            "NumberOfResolutions": self.num_resolutions,
            "MaximumNumberOfIterations": self.max_iterations,
            "Metric": [
                "AdvancedMattesMutualInformation",
                "TransformBendingEnergyPenalty"
            ],
            "Metric0Weight": ["1.0"],
            "Metric1Weight": [str(self.bending_energy_weight)],
        }
        
        # 設定多線程（0 表示自動偵測）
        if self.num_threads > 0:
            params["NumberOfThreads"] = self.num_threads
        
        return params


@dataclass
class RegistrationConfig:
    """完整配準流程配置"""
    
    # 專案名稱
    project_name: str = "thriple_registration"
    
    # 輸入目錄
    input_dir: Path = Path("/home/sec312/tsgh/picture/czi/40X")
    
    # 輸出目錄
    output_dir: Path = Path("/home/sec312/tsgh/thriple_image_layer/output")
    
    # 參考模態名稱
    reference_modality: str = "HER2"
    
    # Elastix 參數
    elastix_params: ElastixParams = field(default_factory=ElastixParams)
    
    # 影像模態列表
    modalities: list[ModalityConfig] = field(default_factory=list)
    
    # 配準路徑列表
    registration_paths: list[RegistrationPathConfig] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化後設定預設模態和路徑"""
        if not self.modalities:
            self.modalities = self._default_modalities()
        if not self.registration_paths:
            self.registration_paths = self._default_paths()
    
    def _default_modalities(self) -> list[ModalityConfig]:
        """預設的影像模態配置"""
        return [
            ModalityConfig(
                name="HER2",
                filename="HER2_40X.czi",
                channel_names=["HER2"],
                channel_colors=["red"]
            ),
            ModalityConfig(
                name="DISH",
                filename="DISH_40X.czi",
                channel_names=["DISH"],
                channel_colors=["blue"]
            ),
            ModalityConfig(
                name="HE",
                filename="HE_40X.czi",
                channel_names=["HE"],
                channel_colors=["green"]
            ),
        ]
    
    def _default_paths(self) -> list[RegistrationPathConfig]:
        """預設的配準路徑配置"""
        return [
            RegistrationPathConfig(
                moving="DISH",
                fixed="HER2",
                stages=["rigid", "affine", "nl"]
            ),
            RegistrationPathConfig(
                moving="HE",
                fixed="HER2",
                stages=["rigid", "affine", "nl"]
            ),
        ]


def create_default_config() -> RegistrationConfig:
    """創建預設配置實例"""
    return RegistrationConfig()
