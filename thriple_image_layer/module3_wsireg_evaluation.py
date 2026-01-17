"""
Module 3: ROI 品質評估模組

評估 WSIReg 配準結果的品質，
計算 NCC、MI 等指標並生成視覺化報告。
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from skimage import color
from sklearn.metrics import mutual_info_score


@dataclass
class EvaluationResult:
    """評估結果資料類別"""
    
    ncc_dish_her2: float
    ncc_he_her2: float
    mi_dish_her2: float
    mi_he_her2: float
    
    def to_dataframe(self) -> pd.DataFrame:
        """轉換為 DataFrame 格式"""
        return pd.DataFrame({
            "Comparison": ["DISH vs HER2", "HE vs HER2"],
            "NCC_Score": [self.ncc_dish_her2, self.ncc_he_her2],
            "MI_Score": [self.mi_dish_her2, self.mi_he_her2],
        })


class ROIEvaluator:
    """ROI 配準品質評估器
    
    從 WSIReg 輸出的 OME-TIFF 檔案中提取 ROI，
    計算配準品質指標。
    
    Attributes:
        output_dir: 配準輸出目錄
        roi_size: ROI 尺寸 (width, height)
    """
    
    DEFAULT_MODALITY_PATHS = {
        "HER2": "HER2_registered.ome.tiff",
        "DISH": "DISH_registered.ome.tiff",
        "HE": "HE_registered.ome.tiff",
    }
    
    def __init__(
        self,
        output_dir: Path,
        roi_size: tuple[int, int] = (2048, 2048)
    ):
        """初始化評估器
        
        Args:
            output_dir: WSIReg 輸出目錄
            roi_size: ROI 尺寸 (width, height)
        """
        self.output_dir = Path(output_dir)
        self.roi_size = roi_size
        
        # 儲存載入的影像
        self._images: dict[str, np.ndarray] = {}
        self._rois: dict[str, np.ndarray] = {}
    
    def _find_registered_images(self) -> dict[str, Path]:
        """尋找配準後的 OME-TIFF 檔案
        
        Returns:
            模態名稱到檔案路徑的對應
        """
        image_paths = {}
        
        # WSIReg 輸出目錄結構：output_dir/project_name/modality_registered.ome.tiff
        project_dir = self.output_dir / "thriple_registration"
        
        if not project_dir.exists():
            # 嘗試直接在 output_dir 尋找
            project_dir = self.output_dir
        
        for modality, default_name in self.DEFAULT_MODALITY_PATHS.items():
            # 嘗試找到對應檔案
            expected_path = project_dir / default_name
            
            if expected_path.exists():
                image_paths[modality] = expected_path
            else:
                # 嘗試其他命名模式
                pattern = f"*{modality}*.ome.tiff"
                matches = list(project_dir.glob(pattern))
                if matches:
                    image_paths[modality] = matches[0]
        
        return image_paths
    
    def _load_image(self, path: Path) -> np.ndarray:
        """載入 OME-TIFF 影像
        
        Args:
            path: 影像路徑
        
        Returns:
            NumPy 陣列格式的影像
        """
        # 使用 tifffile 讀取，支援大型 TIFF
        img = tifffile.imread(str(path))
        
        # 確保是 3 通道 RGB
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[0] in (3, 4):
            # CxHxW -> HxWxC
            img = np.moveaxis(img, 0, -1)
        
        if img.shape[-1] == 4:
            img = img[..., :3]  # 移除 alpha 通道
        
        return img
    
    def _extract_center_roi(self, image: np.ndarray) -> np.ndarray:
        """從影像中心提取 ROI
        
        Args:
            image: 完整影像
        
        Returns:
            中心 ROI
        """
        height, width = image.shape[:2]
        roi_w, roi_h = self.roi_size
        
        # 計算中心位置
        x_start = (width - roi_w) // 2
        y_start = (height - roi_h) // 2
        
        # 確保邊界合理
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(width, x_start + roi_w)
        y_end = min(height, y_start + roi_h)
        
        return image[y_start:y_end, x_start:x_end]
    
    def _calculate_ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """計算正規化互相關 (NCC)
        
        Args:
            img1: 第一張影像
            img2: 第二張影像
        
        Returns:
            NCC 值 (-1 到 1)
        """
        gray1 = color.rgb2gray(img1) if img1.ndim == 3 else img1
        gray2 = color.rgb2gray(img2) if img2.ndim == 3 else img2
        
        return float(np.corrcoef(gray1.ravel(), gray2.ravel())[0, 1])
    
    def _calculate_mi(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """計算互信息 (MI)
        
        Args:
            img1: 第一張影像
            img2: 第二張影像
        
        Returns:
            MI 值
        """
        gray1 = color.rgb2gray(img1) if img1.ndim == 3 else img1
        gray2 = color.rgb2gray(img2) if img2.ndim == 3 else img2
        
        # 轉為離散 bins
        bins1 = (gray1 * 255).astype(int).ravel()
        bins2 = (gray2 * 255).astype(int).ravel()
        
        return float(mutual_info_score(bins1, bins2))
    
    def _create_merged_visualization(self) -> np.ndarray:
        """創建三重疊合可視化影像 (R=HER2, G=HE, B=DISH)
        
        Returns:
            RGB 合併影像
        """
        her2_gray = color.rgb2gray(self._rois["HER2"])
        he_gray = color.rgb2gray(self._rois["HE"])
        dish_gray = color.rgb2gray(self._rois["DISH"])
        
        # 正規化到 0-255
        her2_u8 = (her2_gray * 255).astype(np.uint8)
        he_u8 = (he_gray * 255).astype(np.uint8)
        dish_u8 = (dish_gray * 255).astype(np.uint8)
        
        # R=HER2, G=HE, B=DISH
        return np.dstack([her2_u8, he_u8, dish_u8])
    
    def run(self) -> EvaluationResult:
        """執行完整評估流程
        
        Returns:
            評估結果
        """
        print("=" * 60)
        print("ROI 品質評估")
        print("=" * 60)
        
        # Step 1: 尋找配準後影像
        print("\n[1/4] 尋找配準後影像...")
        image_paths = self._find_registered_images()
        
        if len(image_paths) < 3:
            raise FileNotFoundError(
                f"找不到所有配準後影像。找到: {list(image_paths.keys())}"
            )
        
        for name, path in image_paths.items():
            print(f"  ✓ {name}: {path.name}")
        
        # Step 2: 載入影像並提取 ROI
        print(f"\n[2/4] 提取中心 ROI ({self.roi_size[0]}x{self.roi_size[1]})...")
        for name, path in image_paths.items():
            img = self._load_image(path)
            self._images[name] = img
            self._rois[name] = self._extract_center_roi(img)
            print(f"  ✓ {name}: {self._rois[name].shape}")
        
        # Step 3: 計算評估指標
        print("\n[3/4] 計算配準品質指標...")
        result = EvaluationResult(
            ncc_dish_her2=self._calculate_ncc(
                self._rois["DISH"], self._rois["HER2"]
            ),
            ncc_he_her2=self._calculate_ncc(
                self._rois["HE"], self._rois["HER2"]
            ),
            mi_dish_her2=self._calculate_mi(
                self._rois["DISH"], self._rois["HER2"]
            ),
            mi_he_her2=self._calculate_mi(
                self._rois["HE"], self._rois["HER2"]
            ),
        )
        
        print(f"  NCC (DISH-HER2): {result.ncc_dish_her2:.4f}")
        print(f"  NCC (HE-HER2):   {result.ncc_he_her2:.4f}")
        print(f"  MI (DISH-HER2):  {result.mi_dish_her2:.4f}")
        print(f"  MI (HE-HER2):    {result.mi_he_her2:.4f}")
        
        # Step 4: 保存結果
        print("\n[4/4] 保存評估結果...")
        
        # 保存合併影像
        merged = self._create_merged_visualization()
        merged_path = self.output_dir / "Merged_ROI.png"
        Image.fromarray(merged).save(merged_path)
        print(f"  ✓ 已保存: {merged_path.name}")
        
        # 保存指標 CSV
        df = result.to_dataframe()
        csv_path = self.output_dir / "Metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"  ✓ 已保存: {csv_path.name}")
        
        print("\n" + "=" * 60)
        print("評估完成")
        print("=" * 60)
        print(f"\n{df.to_string(index=False)}")
        
        return result


def evaluate_roi(
    output_dir: Path,
    roi_size: tuple[int, int] = (2048, 2048)
) -> EvaluationResult:
    """評估配準品質的便捷函數
    
    提供向後兼容的 API。
    
    Args:
        output_dir: 配準輸出目錄
        roi_size: ROI 尺寸
    
    Returns:
        評估結果
    """
    evaluator = ROIEvaluator(output_dir, roi_size)
    return evaluator.run()


if __name__ == "__main__":
    output_dir = Path("/home/sec312/tsgh/thriple_image_layer/output")
    evaluate_roi(output_dir)
