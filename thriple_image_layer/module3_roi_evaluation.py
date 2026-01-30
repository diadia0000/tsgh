"""Module 3: ROI Quality Check

提取 ROI 並評估對準品質
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from sklearn.metrics import mutual_info_score
from valis import registration, slide_io
import pyvips

from config import RegistrationConfig, create_default_config, get_slide_key


def evaluate_roi(
    config: RegistrationConfig,
) -> None:
    """
    Module 3: 提取 ROI 並評估對準品質
    
    Args:
        config: 配準流程配置
    """
    # 初始化 JVM
    try:
        slide_io.init_jvm()
    except Exception:
        pass
    
    output_dir = config.output_dir
    roi_size = config.roi.roi_size
    
    # 載入變換參數
    # 使用 VALIS 的 load_registrar 從 pickle 檔案載入
    pickle_path = config.pickle_path
    print(f"載入變換參數: {pickle_path}")
    registrar = registration.load_registrar(str(pickle_path))
    
    # 計算 ROI 中心位置
    aligned_shape = registrar.get_aligned_slide_shape(0)
    center_y = aligned_shape[0] // 2
    center_x = aligned_shape[1] // 2
    x_start = center_x - roi_size[0] // 2
    y_start = center_y - roi_size[1] // 2
    
    # 提取對齊後的 ROI (按名稱排序確保順序)
    print(f"提取對齊後的 ROI (中心: {center_x}, {center_y}, 尺寸: {roi_size})...")
    slide_names = sorted(registrar.slide_dict.keys())
    rois = {}
    
    for name in slide_names:
        slide_obj = registrar.slide_dict[name]
        roi = slide_obj.warp_img(
            img=slide_obj.image,
            non_rigid=True,
            crop=(x_start, y_start, roi_size[0], roi_size[1])
        )
        rois[name] = roi
        print(f"  {name}: {roi.shape if hasattr(roi, 'shape') else (roi.height, roi.width)}")
    
    # 從配置獲取模態的 slide key
    dish_key = get_slide_key(config.get_modality_by_name("DISH").filename)
    he_key = get_slide_key(config.get_modality_by_name("HE").filename)
    her2_key = get_slide_key(config.get_modality_by_name("HER2").filename)
    
    dish_roi = rois[dish_key]
    he_roi = rois[he_key]
    her2_roi = rois[her2_key]
    
    # 確保 ROI 是 numpy array 並處理格式
    def to_numpy_rgb(img: np.ndarray | pyvips.Image) -> np.ndarray:
        """將影像轉換為 RGB numpy array
        
        Args:
            img: 輸入影像 (numpy array 或 pyvips.Image)
            
        Returns:
            np.ndarray: RGB numpy array
        """
        if isinstance(img, pyvips.Image):
            from valis import warp_tools
            img = warp_tools.vips2numpy(img)
        if img.ndim == 2:
            return np.stack([img] * 3, axis=-1)
        # VALIS warp_img 輸出為 BGR，需轉換為 RGB
        if img.shape[2] >= 3:
            img = img[:, :, :3]  # 只取前3個通道
            img = img[:, :, ::-1]  # BGR → RGB
        return img
    
    dish_roi = to_numpy_rgb(dish_roi)
    he_roi = to_numpy_rgb(he_roi)
    her2_roi = to_numpy_rgb(her2_roi)
    
    # 生成疊合圖 (R=Her2, G=DISH, B=0)
    merged = np.dstack([her2_roi[:, :, 0], dish_roi[:, :, 0], np.zeros_like(her2_roi[:, :, 0])])
    merged_img = Image.fromarray(merged.astype(np.uint8))
    merged_path = output_dir / "Merged_ROI.png"
    merged_img.save(merged_path)
    print(f"已儲存: {merged_path} (R=Her2, G=DISH)")
    
    # 計算疊合指標
    dish_gray = color.rgb2gray(dish_roi)
    he_gray = color.rgb2gray(he_roi)
    her2_gray = color.rgb2gray(her2_roi)
    
    ncc_dish_her2 = np.corrcoef(her2_gray.ravel(), dish_gray.ravel())[0, 1]
    ncc_her2_he = np.corrcoef(he_gray.ravel(), her2_gray.ravel())[0, 1]
    
    # 將灰階值轉為離散bins計算MI
    dish_bins = (dish_gray * 255).astype(int)
    he_bins = (he_gray * 255).astype(int)
    her2_bins = (her2_gray * 255).astype(int)
    
    mi_dish_her2 = mutual_info_score(her2_bins.ravel(), dish_bins.ravel())
    mi_her2_he = mutual_info_score(he_bins.ravel(), her2_bins.ravel())
    
    # 輸出指標報告
    df = pd.DataFrame({
        'Comparison': ['DISH vs Her2', 'Her2 vs HE'],
        'NCC_Score': [ncc_dish_her2, ncc_her2_he],
        'MI_Score': [mi_dish_her2, mi_her2_he]
    })
    
    csv_path = output_dir / "Metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"已儲存: {csv_path}")
    print(f"\n評估結果:\n{df}")


if __name__ == "__main__":
    config = create_default_config()
    
    print("=" * 60)
    print("Module 3: ROI Quality Evaluation")
    print("=" * 60)
    print(f"ROI 尺寸: {config.roi.roi_size}")
    print()
    
    try:
        evaluate_roi(config)
    finally:
        # 清理 JVM
        try:
            slide_io.kill_jvm()
        except Exception:
            pass
