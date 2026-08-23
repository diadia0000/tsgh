"""Module 3: ROI Quality Check"""
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage import color
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from valis import registration, slide_io, warp_tools
import pyvips

try:
    from .config import RegistrationConfig, create_default_config, get_slide_key
except ImportError:
    from config import RegistrationConfig, create_default_config, get_slide_key


def _deformation_stats(bk_dxdy) -> tuple:
    """形變場的摺疊比例與最大位移。

    形變是 T(x) = x + d(x)，所以 Jacobian 是 I + J(d)。
    det <= 0 代表局部翻面（組織被扭到反過來），是 optical flow 的典型失敗模式，
    B-spline 應該接近 0。沒有這個數字就分不出「MI 變高」和「扭得更兇」。

    Returns:
        tuple: (摺疊像素占比 %, 最大位移 px)
    """
    if bk_dxdy is None:
        return 0.0, 0.0  # 參考影像不變形

    if isinstance(bk_dxdy, pyvips.Image):
        arr = warp_tools.vips2numpy(bk_dxdy)
        dx, dy = arr[..., 0], arr[..., 1]
    else:
        dx, dy = np.asarray(bk_dxdy[0]), np.asarray(bk_dxdy[1])

    dudy, dudx = np.gradient(dx)
    dvdy, dvdx = np.gradient(dy)
    det = (1 + dudx) * (1 + dvdy) - dudy * dvdx

    # 場在組織遮罩外被歸零，邊界的階躍會算出假摺疊，所以往內縮再統計
    inside = binary_erosion((dx != 0) | (dy != 0), np.ones((5, 5), dtype=bool))
    if not inside.any():
        return 0.0, 0.0

    max_disp = float(np.hypot(dx, dy)[inside].max())
    return 100.0 * float((det[inside] <= 0).mean()), max_disp


def evaluate_roi(config: RegistrationConfig) -> None:
    """
    Module 3: 提取 ROI 並評估對準品質

    Args:
        config: 配準流程配置
    """
    # 初始化 JVM
    try:
        slide_io.init_jvm()
    except:
        pass

    output_dir = config.output_dir
    roi_size = config.roi.roi_size

    # 載入變換參數
    registrar = registration.load_registrar(str(config.pickle_path))
    
    # 計算 ROI 中心位置
    aligned_shape = registrar.get_aligned_slide_shape(0)
    center_y = aligned_shape[0] // 2
    center_x = aligned_shape[1] // 2
    x_start = center_x - roi_size[0] // 2
    y_start = center_y - roi_size[1] // 2
    
    # 提取對齊後的 ROI (按名稱排序確保順序)
    print(f"提取對齊後的 ROI (中心: {center_x}, {center_y})...")
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
    
    # 根據 config 的模態檔名取得 VALIS slide key
    dish_roi = rois[get_slide_key(config.get_modality_by_name("DISH").filename)]
    he_roi = rois[get_slide_key(config.get_modality_by_name("HE").filename)]
    her2_roi = rois[get_slide_key(config.get_modality_by_name("HER2").filename)]
    
    # 確保 ROI 是 numpy array 並處理格式
    def to_numpy_rgb(img):
        if isinstance(img, pyvips.Image):
            from valis import warp_tools
            img = warp_tools.vips2numpy(img)
        if img.ndim == 2:
            return np.stack([img]*3, axis=-1)
        return img
    
    dish_roi = to_numpy_rgb(dish_roi)
    he_roi = to_numpy_rgb(he_roi)
    her2_roi = to_numpy_rgb(her2_roi)
    
    # 生成三重疊合圖 (R=Her2, G=HE, B=DISH)
    merged = np.dstack([her2_roi[:,:,0], he_roi[:,:,1], dish_roi[:,:,2]])
    merged_img = Image.fromarray(merged.astype(np.uint8))
    merged_img.save(output_dir / "Merged_ROI.png")
    print(f"已儲存: Merged_ROI.png (R=Her2, G=HE, B=DISH)")
    
    # 計算疊合指標
    grays = {
        'DISH': color.rgb2gray(dish_roi).ravel(),
        'HE': color.rgb2gray(he_roi).ravel(),
        'Her2': color.rgb2gray(her2_roi).ravel(),
    }
    # 將灰階值轉為離散bins計算MI
    bins = {k: (v * 255).astype(int) for k, v in grays.items()}

    # MI 未正規化 (nats)，上限隨影像熵浮動，不同 ROI/跑次之間不能直接比大小。
    # NMI 落在 [0, 1]，才是可以拿來比較的那個。
    # DISH vs Her2 不經過 HE，直接看兩張染色之間對得準不準
    df = pd.DataFrame([
        {
            'Comparison': f'{a} vs {b}',
            'NCC_Score': np.corrcoef(grays[a], grays[b])[0, 1],
            'MI_Score': mutual_info_score(bins[a], bins[b]),
            'NMI_Score': normalized_mutual_info_score(bins[a], bins[b]),
        }
        for a, b in [('DISH', 'HE'), ('Her2', 'HE'), ('DISH', 'Her2')]
    ])

    csv_path = output_dir / "Metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"已儲存: Metrics.csv")
    print(f"\n評估結果:\n{df}")

    # 形變品質：MI 高但摺疊率也高 = 扭曲換來的分數，不是對得更準
    deform_df = pd.DataFrame(
        [(name, *_deformation_stats(registrar.slide_dict[name].bk_dxdy))
         for name in slide_names],
        columns=['Slide', 'Fold_Percent', 'Max_Displacement_px'],
    )
    deform_df.to_csv(output_dir / "Deformation.csv", index=False)
    print(f"已儲存: Deformation.csv")
    print(f"\n形變場檢查:\n{deform_df}")

if __name__ == "__main__":
    try:
        evaluate_roi(create_default_config())
    finally:
        # 清理 JVM
        try:
            slide_io.kill_jvm()
        except:
            raise RuntimeError("無法關閉 Java 虛擬機，請確認 JVM 是否已啟動。")
