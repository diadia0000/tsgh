"""Module 3: ROI Quality Check

多區域取樣評估對準品質（中心 + 四角），指標包含 NCC、NMI、SSIM
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from skimage.metrics import structural_similarity as ssim
from valis import registration, slide_io
import pyvips

from config import RegistrationConfig, create_default_config, get_slide_key


def _to_numpy_rgb(img: np.ndarray | pyvips.Image) -> np.ndarray:
    """將影像轉換為 RGB numpy array

    Args:
        img: 輸入影像 (numpy array 或 pyvips.Image)

    Returns:
        np.ndarray: RGB uint8 numpy array
    """
    if isinstance(img, pyvips.Image):
        from valis import warp_tools
        img = warp_tools.vips2numpy(img)
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.shape[2] >= 3:
        img = img[:, :, :3]
        img = img[:, :, ::-1]  # BGR → RGB
    return img


def _advanced_nmi(img_a: np.ndarray, img_b: np.ndarray, bins: int = 64) -> tuple[float, float]:
    """計算正規化互信息 (NMI) 與聯合熵。

    使用 joint histogram 估計聯合機率分布，以 NMI = 2*MI/(H(A)+H(B)) 正規化，
    消除不同模態亮度分佈差異的干擾。

    Args:
        img_a: 灰階影像 A，值域 [0, 1]
        img_b: 灰階影像 B，值域 [0, 1]
        bins: 直方圖 bin 數

    Returns:
        tuple[float, float]: (NMI, joint_entropy)
    """
    hist_2d, _, _ = np.histogram2d(img_a.ravel(), img_b.ravel(), bins=bins, density=True)
    eps = np.finfo(float).eps
    hist_2d = hist_2d + eps
    hist_2d /= hist_2d.sum()

    p_a = hist_2d.sum(axis=1)
    p_b = hist_2d.sum(axis=0)

    h_a = -np.sum(p_a * np.log(p_a + eps))
    h_b = -np.sum(p_b * np.log(p_b + eps))
    h_ab = -np.sum(hist_2d * np.log(hist_2d + eps))

    mi = h_a + h_b - h_ab
    nmi = 2.0 * mi / (h_a + h_b + eps)
    return float(nmi), float(h_ab)


def _compute_pair_metrics(gray_a: np.ndarray, gray_b: np.ndarray) -> dict:
    """計算一組灰階影像的所有對準指標。

    Args:
        gray_a: 灰階影像 A，值域 [0, 1]
        gray_b: 灰階影像 B，值域 [0, 1]

    Returns:
        dict: 含 NCC, NMI, Joint_Entropy, SSIM 的字典
    """
    ncc = float(np.corrcoef(gray_a.ravel(), gray_b.ravel())[0, 1])
    nmi, joint_ent = _advanced_nmi(gray_a, gray_b)
    ssim_val = float(ssim(gray_a, gray_b, data_range=1.0))
    return {
        "NCC": ncc,
        "NMI": nmi,
        "Joint_Entropy": joint_ent,
        "SSIM": ssim_val,
    }


def _get_roi_positions(
    aligned_h: int, aligned_w: int, roi_w: int, roi_h: int, margin_ratio: float = 0.25
) -> dict[str, tuple[int, int]]:
    """計算 5 個 ROI 取樣位置（中心 + 四象限中點）。

    四象限中點位於中心與邊緣的中間位置，確保落在有效組織區域內，
    避免 VALIS 配準後邊緣的黑色填充。

    Args:
        aligned_h: 對齊後影像高度
        aligned_w: 對齊後影像寬度
        roi_w: ROI 寬度
        roi_h: ROI 高度
        margin_ratio: 四角距邊緣比例 (預設 0.25 = 25%)

    Returns:
        dict: {區域名稱: (x_start, y_start)}
    """
    aligned_h, aligned_w = int(aligned_h), int(aligned_w)
    roi_w, roi_h = int(roi_w), int(roi_h)
    cx, cy = aligned_w // 2, aligned_h // 2

    # 四象限中點：中心與邊緣（含 margin）的中間
    margin_x = int(aligned_w * margin_ratio)
    margin_y = int(aligned_h * margin_ratio)

    # 四個象限中點座標（取 ROI 左上角）
    q_left = (margin_x + cx) // 2 - roi_w // 2
    q_right = (cx + aligned_w - margin_x) // 2 - roi_w // 2
    q_top = (margin_y + cy) // 2 - roi_h // 2
    q_bottom = (cy + aligned_h - margin_y) // 2 - roi_h // 2

    return {
        "center": (cx - roi_w // 2, cy - roi_h // 2),
        "quad_top_left": (q_left, q_top),
        "quad_top_right": (q_right, q_top),
        "quad_bottom_left": (q_left, q_bottom),
        "quad_bottom_right": (q_right, q_bottom),
    }


def evaluate_roi(
    config: RegistrationConfig,
) -> None:
    """Module 3: 多區域取樣評估對準品質

    從對齊後影像的中心和四角各取一個 ROI，計算 NCC、NMI、SSIM 指標，
    輸出逐區域報告和整體摘要。

    Args:
        config: 配準流程配置
    """
    try:
        slide_io.init_jvm()
    except Exception:
        pass

    output_dir = config.output_dir
    roi_w, roi_h = config.roi.roi_size

    # 載入變換參數
    pickle_path = config.pickle_path
    print(f"載入變換參數: {pickle_path}")
    registrar = registration.load_registrar(str(pickle_path))

    # 使用 processed image 層級的座標系（reg_img_shape_rc）
    # slide_obj.image 是 processed image，warp_img 的 crop 座標基於此尺寸
    ref_slide = registrar.get_ref_slide()
    reg_shape_rc = ref_slide.reg_img_shape_rc  # (height, width)
    print(f"配準影像尺寸 (processed): {reg_shape_rc[1]}x{reg_shape_rc[0]}")

    # ROI 尺寸不能超過配準影像（轉 Python int，避免 numpy int 被 VALIS isinstance 檢查拒絕）
    roi_w = int(min(roi_w, reg_shape_rc[1] // 3))
    roi_h = int(min(roi_h, reg_shape_rc[0] // 3))

    positions = _get_roi_positions(reg_shape_rc[0], reg_shape_rc[1], roi_w, roi_h)
    print(f"ROI 尺寸: {roi_w}x{roi_h}，取樣 {len(positions)} 個區域")

    # 模態 key
    dish_key = get_slide_key(config.get_modality_by_name("DISH").filename)
    he_key = get_slide_key(config.get_modality_by_name("HE").filename)
    her2_key = get_slide_key(config.get_modality_by_name("HER2").filename)

    comparisons = [
        ("DISH_vs_HER2", her2_key, dish_key),
        ("HER2_vs_HE", he_key, her2_key),
    ]

    all_rows = []

    for region_name, (x_start, y_start) in positions.items():
        print(f"\n--- {region_name} (x={x_start}, y={y_start}) ---")

        # 邊界檢查：確保 crop 不超出對齊後影像範圍
        if (x_start < 0 or y_start < 0
                or x_start + roi_w > reg_shape_rc[1]
                or y_start + roi_h > reg_shape_rc[0]):
            print(f"  跳過：ROI 超出影像邊界")
            continue

        # 提取此區域的 ROI
        region_rois = {}
        skip_region = False
        for name in [dish_key, he_key, her2_key]:
            slide_obj = registrar.slide_dict[name]
            try:
                roi = slide_obj.warp_img(
                    img=slide_obj.image,
                    non_rigid=True,
                    crop=(x_start, y_start, roi_w, roi_h),
                )
            except Exception as e:
                print(f"  跳過：{name} warp 失敗 ({e})")
                skip_region = True
                break
            roi = _to_numpy_rgb(roi)
            roi = 255 - roi  # 反轉：組織亮、背景暗
            region_rois[name] = roi

        if skip_region:
            continue

        # 儲存中心區域的疊合圖
        if region_name == "center":
            her2_roi = region_rois[her2_key]
            dish_roi = region_rois[dish_key]
            merged = np.dstack([
                her2_roi[:, :, 0],
                dish_roi[:, :, 0],
                np.zeros_like(her2_roi[:, :, 0]),
            ])
            merged_img = Image.fromarray(merged.astype(np.uint8))
            merged_path = output_dir / "Merged_ROI_center.png"
            merged_img.save(merged_path)
            print(f"  已儲存疊合圖: {merged_path}")

        # 轉灰階 & 計算指標
        grays = {k: color.rgb2gray(v) for k, v in region_rois.items()}

        for comp_name, key_a, key_b in comparisons:
            metrics = _compute_pair_metrics(grays[key_a], grays[key_b])
            row = {"Region": region_name, "Comparison": comp_name, **metrics}
            all_rows.append(row)
            print(f"  {comp_name}: NCC={metrics['NCC']:.4f}  NMI={metrics['NMI']:.4f}  SSIM={metrics['SSIM']:.4f}")

    # 逐區域詳細報告
    df = pd.DataFrame(all_rows)
    csv_path = output_dir / "Metrics_by_region.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n已儲存逐區域報告: {csv_path}")

    # 整體摘要（每組比較的 mean ± std）
    summary_rows = []
    for comp_name in ["DISH_vs_HER2", "HER2_vs_HE"]:
        sub = df[df["Comparison"] == comp_name]
        if sub.empty:
            continue
        row = {"Comparison": comp_name}
        for metric in ["NCC", "NMI", "SSIM"]:
            mean_val = sub[metric].mean()
            std_val = sub[metric].std()
            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_std"] = std_val
        # 中心 vs 四象限差異（對齊均勻性指標）
        center_rows = sub[sub["Region"] == "center"]["NMI"]
        quad_rows = sub[sub["Region"] != "center"]["NMI"]
        if not center_rows.empty and not quad_rows.empty:
            row["NMI_center_quad_diff"] = float(center_rows.values[0] - quad_rows.mean())
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "Metrics_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"已儲存摘要報告: {summary_path}")
    print(f"\n=== 摘要 ===\n{df_summary.to_string(index=False)}")


if __name__ == "__main__":
    config = create_default_config()

    print("=" * 60)
    print("Module 3: ROI Quality Evaluation (Multi-Region)")
    print("=" * 60)
    print(f"ROI 尺寸: {config.roi.roi_size}")
    print()

    try:
        evaluate_roi(config)
    finally:
        try:
            slide_io.kill_jvm()
        except Exception:
            pass
