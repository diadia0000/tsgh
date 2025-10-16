"""Module 1: Preprocessing Pipeline"""
from pathlib import Path
from typing import Tuple, List
import numpy as np
from aicspylibczi import CziFile
import valis
from valis import preprocessing

def load_czi_with_scale(czi_path: Path, scale_factor: int = 32) -> np.ndarray:
    """使用 scale_factor 載入 CZI mosaic 影像"""
    czi = CziFile(czi_path)
    bbox = czi.get_mosaic_bounding_box()
    
    # 計算縮小後的尺寸
    scaled_width = bbox.w // scale_factor
    scaled_height = bbox.h // scale_factor
    
    # 讀取並縮放 mosaic
    img, _ = czi.read_mosaic(scale_factor=scale_factor, C=0)
    
    # 移除多餘維度並確保是 RGB
    img = np.squeeze(img)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    
    return img

def preprocess_images(
    czi_dir: Path,
    output_dir: Path,
    scale_factor: int = 32
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Module 1: 預處理三個 CZI 檔案
    
    Args:
        czi_dir: CZI 檔案目錄
        output_dir: 輸出目錄
        scale_factor: 縮放因子 (2的倍數)
    
    Returns:
        (images, masks): 處理後的影像列表和遮罩列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    czi_files = ["DISH_40X_2.czi", "HE_40X.czi", "HER2_40X.czi"]
    images = []
    masks = []
    
    preprocessor = preprocessing.ColorfulStandardizer()
    
    for czi_file in czi_files:
        print(f"處理 {czi_file}...")
        czi_path = czi_dir / czi_file
        
        # 載入縮小的影像
        img = load_czi_with_scale(czi_path, scale_factor)
        print(f"  載入影像尺寸: {img.shape}")
        
        # 正規化
        normalized = preprocessor.standardize_img(img)
        images.append(normalized)
        
        # 生成遮罩
        mask = valis.get_tissue_mask(img)
        masks.append(mask)
        
        print(f"  完成: {img.shape} -> {normalized.shape}")
    
    return images, masks

if __name__ == "__main__":
    czi_dir = Path(r"E:\Class\tsgh\picture\whole_size\40X")
    output_dir = Path(r"E:\Class\tsgh\thriple_image_layer\output")
    
    images, masks = preprocess_images(czi_dir, output_dir, scale_factor=32)
    print(f"\n預處理完成，共 {len(images)} 張影像")
