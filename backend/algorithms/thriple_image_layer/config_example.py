"""VALIS 配準流程配置模組"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from os import cpu_count

# === 全域預設值常數 ===
DEFAULT_SCALE_FACTOR: float = 0.5  # 縮放比例 (1.0 = 40X, 0.5 = 20X)


@dataclass
class ModalityConfig:
    """單一影像模態的配置
    
    Module 1 流程：CZI → TIFF (可選縮放)
    Module 2 流程：TIFF → 配準後 TIFF
    
    Attributes:
        name: 模態名稱 (e.g., "HER2", "DISH", "HE")
        filename: 處理後的 TIFF 檔名 (Module 1 輸出 / Module 2 輸入)
        czi_filename: CZI 原始檔名
        scale_factor: 縮放比例 (1.0 = 不縮放, 0.5 = 縮小一半)
    """
    
    name: str
    filename: str  # Module 1 輸出 / Module 2 輸入的 TIFF 檔名
    
    # === Module 1 專用 ===
    czi_filename: Optional[str] = None  # CZI 原始檔名
    scale_factor: float = DEFAULT_SCALE_FACTOR


@dataclass
class PreprocessConfig:
    """CZI 轉 TIFF 前處理參數配置"""

    strip_height: int = 2048
    num_processes: Optional[int] = cpu_count()


@dataclass
class ValisConfig:
    """VALIS 配準參數配置"""

    max_processed_image_dim_px: int = 1024
    max_non_rigid_registration_dim_px: int = 1024
    align_to_reference: bool = True
    reference_img_f: str = "HER2_processed.tiff"


@dataclass
class ROIConfig:
    """ROI 評估參數配置
    
    控制 Module 3 的 ROI 品質評估
    
    Attributes:
        roi_size: ROI 尺寸 (width, height) in pixels
    """
    
    roi_size: tuple = (2048, 2048)


@dataclass
class ThumbnailConfig:
    """縮圖生成參數配置
    
    控制 Module 4 的縮圖生成
    
    Attributes:
        level: 金字塔層級 (0=最高解析度，數字越大解析度越低)
        use_non_rigid: 是否對 DISH 使用非剛性變換
        laplacian_levels: 拉普拉斯金字塔融合層數
    """
    
    level: int = 0
    use_non_rigid: bool = True
    laplacian_levels: int = 4


@dataclass
class RegistrationConfig:
    """完整配準流程配置
    
    整合所有模組的配置參數
    
    Attributes:
        project_name: 專案名稱
        czi_input_dir: CZI 原始檔目錄 (Module 1 輸入)
        input_dir: TIFF 檔案目錄 (Module 1 輸出 / Module 2 輸入)
        output_dir: 配準結果輸出目錄
        reference_modality: 參考模態名稱
        modalities: 影像模態列表
        preprocess: CZI 前處理參數
        valis: VALIS 配準參數
        roi: ROI 評估參數
        thumbnail: 縮圖生成參數
    """
    
    # 專案名稱
    project_name: str = "thriple_registration"
    
    # CZI 原始檔目錄 (Module 1 輸入)
    czi_input_dir: Path = field(
        default_factory=lambda: Path("/data/czi/40X")
    )
    
    # Module 1 輸出目錄 / Module 2 輸入目錄 (處理後 TIFF)
    input_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "output"
    )
    
    # 輸出目錄 (配準結果)
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "output"
    )
    
    # 參考模態名稱
    reference_modality: str = "HER2"
    
    # 影像模態列表
    modalities: List[ModalityConfig] = field(default_factory=list)
    
    # 前處理參數
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

    # VALIS 配準參數
    valis: ValisConfig = field(default_factory=ValisConfig)
    
    # ROI 評估參數
    roi: ROIConfig = field(default_factory=ROIConfig)
    
    # 縮圖生成參數
    thumbnail: ThumbnailConfig = field(default_factory=ThumbnailConfig)

    def __post_init__(self) -> None:
        """初始化後設定預設模態"""
        if not self.modalities:
            self.modalities = self._default_modalities()
    
    def _default_modalities(self) -> List[ModalityConfig]:
        """預設的影像模態配置
        
        scale_factor=1.0 保持 40X 原始解析度
        
        Returns:
            List[ModalityConfig]: 預設模態配置列表
        """
        return [
            ModalityConfig(
                name="HER2",
                filename="HER2_processed.tiff",
                czi_filename="HER2_40X.czi",
                scale_factor=DEFAULT_SCALE_FACTOR,
            ),
            ModalityConfig(
                name="DISH",
                filename="DISH_processed.tiff",
                czi_filename="DISH_40X.czi",
                scale_factor=DEFAULT_SCALE_FACTOR,
            ),
            ModalityConfig(
                name="HE",
                filename="HE_processed.tiff",
                czi_filename="HE_40X.czi",
                scale_factor=DEFAULT_SCALE_FACTOR,
            ),
        ]
    
    def get_modality_by_name(self, name: str) -> Optional[ModalityConfig]:
        """根據名稱獲取模態配置
        
        Args:
            name: 模態名稱
            
        Returns:
            Optional[ModalityConfig]: 模態配置，找不到則返回 None
        """
        for m in self.modalities:
            if m.name == name:
                return m
        return None
    
    def get_reference_modality(self) -> Optional[ModalityConfig]:
        """獲取參考模態配置
        
        Returns:
            Optional[ModalityConfig]: 參考模態配置
        """
        return self.get_modality_by_name(self.reference_modality)
    
    @property
    def transform_params_dir(self) -> Path:
        """變換參數目錄
        
        Returns:
            Path: Transform_Params 目錄路徑
        """
        return self.output_dir / "Transform_Params"
    
    @property
    def pickle_path(self) -> Path:
        """Registrar pickle 檔案路徑
        
        Returns:
            Path: pickle 檔案完整路徑
        """
        return self.transform_params_dir / "data" / "Transform_Params_registrar.pickle"
    
    @property
    def temp_dir(self) -> Path:
        """暫存目錄
        
        Returns:
            Path: 暫存目錄路徑
        """
        return self.output_dir / "temp"


def create_default_config() -> RegistrationConfig:
    """創建預設配置實例
    
    Returns:
        RegistrationConfig: 預設配置實例
    """
    return RegistrationConfig()


# === 便利函數 ===

def get_modality_filenames(config: RegistrationConfig) -> dict:
    """獲取所有模態的檔名映射
    
    Args:
        config: 配置實例
        
    Returns:
        dict: {模態名稱: 檔名} 映射
    """
    return {m.name: m.filename for m in config.modalities}


def get_slide_key(filename: str) -> str:
    """從檔名獲取 VALIS slide 字典的 key
    
    VALIS 使用不含副檔名的檔名作為 key
    
    Args:
        filename: 檔名 (e.g., "HER2_processed.tiff")
        
    Returns:
        str: slide key (e.g., "HER2_processed")
    """
    return Path(filename).stem
