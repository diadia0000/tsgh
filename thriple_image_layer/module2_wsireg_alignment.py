"""
Module 2: WSIReg 影像配準模組

使用 WSIReg 執行多模態全玻片影像 (WSI) 配準，
取代原有的 VALIS 配準流程。

特點：
- 基於 elastix 的高品質配準
- 支援剛性、仿射和 B-spline 非剛性配準
- 可調整正則化參數控制變形平滑度
- 支援多線程 CPU 加速
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# 設定 ITK 多線程環境變數（在導入 ITK 之前設定）
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "0")  # 0 = 自動

from wsireg.wsireg2d import WsiReg2D

from config import (
    RegistrationConfig,
    ModalityConfig,
    RegistrationPathConfig,
    create_default_config,
)


def configure_multithreading(num_threads: int = 0) -> int:
    """配置 ITK 多線程
    
    Args:
        num_threads: 線程數，0 表示自動偵測
    
    Returns:
        實際使用的線程數
    """
    import itk
    
    if num_threads <= 0:
        # 自動偵測 CPU 核心數
        num_threads = os.cpu_count() or 1
    
    # 設定 ITK 全局線程數
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(num_threads)
    
    return num_threads


class WSIRegAligner:
    """WSIReg 影像配準器
    
    封裝 WSIReg 配準流程，提供簡潔的 API 接口。
    
    Attributes:
        config: 配準配置參數
        reg_graph: WSIReg 配準圖物件
    """
    
    def __init__(self, config: Optional[RegistrationConfig] = None):
        """初始化配準器
        
        Args:
            config: 配準配置，若為 None 則使用預設配置
        """
        self.config = config or create_default_config()
        self.reg_graph: Optional[WsiReg2D] = None
        
        self._validate_config()
    
    def _validate_config(self) -> None:
        """驗證配置有效性"""
        if not self.config.input_dir.exists():
            raise FileNotFoundError(
                f"輸入目錄不存在: {self.config.input_dir}"
            )
        
        for modality in self.config.modalities:
            filepath = self.config.input_dir / modality.filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"找不到影像檔案: {filepath}"
                )
    
    def _build_registration_graph(self) -> WsiReg2D:
        """建構 WSIReg 配準圖"""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        reg_graph = WsiReg2D(
            self.config.project_name,
            str(self.config.output_dir)
        )
        
        # 添加所有影像模態
        for modality in self.config.modalities:
            self._add_modality(reg_graph, modality)
        
        # 定義配準路徑
        for path in self.config.registration_paths:
            self._add_registration_path(reg_graph, path)
        
        return reg_graph
    
    def _add_modality(
        self,
        reg_graph: WsiReg2D,
        modality: ModalityConfig
    ) -> None:
        """添加單一影像模態到配準圖
        
        Args:
            reg_graph: WSIReg 配準圖
            modality: 模態配置
        """
        filepath = self.config.input_dir / modality.filename
        
        # 構建預處理參數（包含降採樣以節省記憶體）
        preprocessing = modality.preprocessing or {
            "image_type": "BF",  # Brightfield
            "as_uint8": True,
            "downsampling": modality.downsampling,  # 配準時降採樣
        }
        
        # 確保降採樣參數存在
        if "downsampling" not in preprocessing:
            preprocessing["downsampling"] = modality.downsampling
        
        reg_graph.add_modality(
            modality.name,
            image_fp=str(filepath),
            image_res=modality.resolution,
            channel_names=modality.channel_names,
            channel_colors=modality.channel_colors,
            preprocessing=preprocessing,
            output_res=modality.output_resolution,  # 輸出解析度 (20X = 0.5 µm/px)
        )
        
        output_info = f", 輸出: {modality.output_resolution} µm/px" if modality.output_resolution else ""
        ds_info = f", 降採樣: {modality.downsampling}x" if modality.downsampling > 1 else ""
        print(f"  ✓ 已添加模態: {modality.name} ({modality.filename}{output_info}{ds_info})")
    
    def _add_registration_path(
        self,
        reg_graph: WsiReg2D,
        path_config: RegistrationPathConfig
    ) -> None:
        """添加配準路徑
        
        Args:
            reg_graph: WSIReg 配準圖
            path_config: 路徑配置
        """
        # 轉換配準階段為 WSIReg 格式
        reg_params = self._build_reg_params(path_config.stages)
        
        reg_graph.add_reg_path(
            path_config.moving,
            path_config.fixed,
            reg_params=reg_params,
        )
        
        stages_str = " → ".join(path_config.stages)
        print(f"  ✓ 配準路徑: {path_config.moving} → {path_config.fixed} ({stages_str})")
    
    def _build_reg_params(self, stages: list[str]) -> list:
        """建構 WSIReg 配準參數列表
        
        Args:
            stages: 配準階段列表 ["rigid", "affine", "nl"]
        
        Returns:
            WSIReg 格式的配準參數
        """
        params = []
        
        for stage in stages:
            if stage in ("rigid", "affine"):
                params.append(stage)
            elif stage == "nl":
                # 非剛性配準使用自定義 elastix 參數
                elastix_dict = self.config.elastix_params.to_elastix_dict()
                params.append(("nl", elastix_dict))
            else:
                raise ValueError(f"未知的配準階段: {stage}")
        
        return params
    
    def run(self) -> WsiReg2D:
        """執行完整配準流程
        
        Returns:
            配準圖物件
        """
        print("=" * 60)
        print("WSIReg 影像配準")
        print("=" * 60)
        
        # Step 0: 配置多線程
        num_threads = configure_multithreading(self.config.elastix_params.num_threads)
        print(f"\n使用 {num_threads} 個 CPU 線程")
        
        # Step 1: 建構配準圖
        print("\n[1/4] 建構配準圖...")
        self.reg_graph = self._build_registration_graph()
        
        # Step 2: 執行配準
        print("\n[2/4] 執行影像配準...")
        print(f"  參考模態: {self.config.reference_modality}")
        print(f"  網格間距: {self.config.elastix_params.grid_spacing}")
        print(f"  正則化權重: {self.config.elastix_params.bending_energy_weight}")
        
        self.reg_graph.register_images()
        print("  ✓ 配準完成")
        
        # Step 3: 保存變換參數
        print("\n[3/4] 保存變換參數...")
        self.reg_graph.save_transformations()
        print(f"  ✓ 已保存至: {self.config.output_dir}")
        
        # Step 4: 輸出配準後影像
        print("\n[4/4] 輸出配準後影像...")
        self.reg_graph.transform_images(file_writer="ome.tiff")
        print("  ✓ 已輸出 OME-TIFF 格式")
        
        print("\n" + "=" * 60)
        print("配準流程完成")
        print("=" * 60)
        
        return self.reg_graph


def align_images(
    input_dir: Path,
    output_dir: Path,
    reference_name: str = "HER2",
    **kwargs
) -> WsiReg2D:
    """執行影像配準的便捷函數
    
    提供向後兼容的 API，類似原有 VALIS 的 align_images 函數。
    
    Args:
        input_dir: CZI 檔案目錄
        output_dir: 輸出目錄
        reference_name: 參考影像名稱
        **kwargs: 其他配置參數
    
    Returns:
        WSIReg 配準圖物件
    """
    config = create_default_config()
    config.input_dir = input_dir
    config.output_dir = output_dir
    config.reference_modality = reference_name
    
    # 更新 elastix 參數
    if "grid_spacing" in kwargs:
        config.elastix_params.grid_spacing = kwargs["grid_spacing"]
    if "bending_energy_weight" in kwargs:
        config.elastix_params.bending_energy_weight = kwargs["bending_energy_weight"]
    
    aligner = WSIRegAligner(config)
    return aligner.run()


if __name__ == "__main__":
    # 測試配準流程 (使用 config.py 的預設設定)
    config = create_default_config()
    
    print(f"輸入目錄: {config.input_dir}")
    print(f"輸出目錄: {config.output_dir}")
    
    aligner = WSIRegAligner(config)
    reg_graph = aligner.run()
    print(f"\n配準完成，結果儲存於: {config.output_dir}")
