"""
IHC-HER2 細胞區域 Pseudo-Label 生成器

使用 QuPath 自訂染色矩陣進行顏色解捲積 (Color Deconvolution)，
分離出 DAB 濃度通道後，透過形態學填充產生「被膜包圍的整塊細胞區域」遮罩。

流程:
    1. QuPath 染色矩陣分解 → 取得 DAB 濃度
    2. 固定閾值二值化
    3. 形態學閉合 → 修補膜斷裂缺口
    4. 輪廓偵測 + 填充 → 將膜環內部填滿
    5. binary_fill_holes → 確保所有封閉區域被填滿
    6. 過濾雜訊小區域
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
from skimage import io, measure
from skimage.color import separate_stains, combine_stains

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# QuPath 自訂染色矩陣 (預設值，可由 config 覆蓋)
DEFAULT_STAIN_MATRIX = np.array([
    [0.651, 0.701, 0.290],   # Hematoxylin
    [0.269, 0.568, 0.778],   # DAB
    [0.633, -0.713, 0.302],  # Residual
])


def load_config():
    """
    載入配置檔案

    Returns:
        Config: 配置物件

    Raises:
        ImportError: 若 config.py 不存在，引導使用者複製範例檔
    """
    try:
        from config import config
        return config
    except ImportError:
        raise ImportError(
            "找不到 config.py！\n"
            "請複製 config_example.py 為 config.py 並設定參數:\n"
            "  cp config_example.py config.py"
        )


def separate_dab_qupath(
    image: np.ndarray,
    stain_matrix: np.ndarray = DEFAULT_STAIN_MATRIX,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    使用 QuPath 染色矩陣進行顏色解捲積，分離 DAB 濃度通道

    Args:
        image: RGB 影像 (H, W, 3)，uint8 或 float [0,1]
        stain_matrix: 染色矩陣 (3, 3)，每列為一種染劑的 RGB 吸收向量

    Returns:
        dab_concentration: DAB 濃度 (H, W)，值域 >=0，越高越深
        hematoxylin_concentration: Hematoxylin 濃度 (H, W)
        debug_info: 除錯資訊字典
    """
    # 確保影像為 uint8
    if image.dtype != np.uint8:
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_uint8 = image

    # 計算反矩陣
    custom_from_rgb = np.linalg.inv(stain_matrix)

    # 顏色解捲積 → 取得各染劑濃度
    stain_concentrations = separate_stains(image_uint8, custom_from_rgb)

    hematoxylin_conc = stain_concentrations[:, :, 0]
    dab_conc = stain_concentrations[:, :, 1]

    # 重建單一染劑的 RGB 影像 (用於視覺化)
    null = np.zeros_like(dab_conc)
    hematoxylin_rgb = combine_stains(
        np.stack((hematoxylin_conc, null, null), axis=-1),
        stain_matrix,
    )
    dab_rgb = combine_stains(
        np.stack((null, dab_conc, null), axis=-1),
        stain_matrix,
    )

    debug_info = {
        'hematoxylin_concentration': hematoxylin_conc,
        'dab_concentration': dab_conc,
        'hematoxylin_rgb': np.clip(hematoxylin_rgb, 0, 1),
        'dab_rgb': np.clip(dab_rgb, 0, 1),
    }

    return dab_conc, hematoxylin_conc, debug_info


def fill_enclosed_regions(
    dab_binary: np.ndarray,
    close_kernel_size: int = 11,
    min_cell_area: int = 500,
    max_edge_hole_area: int = 5000,
    open_kernel_size: int = 5,
) -> np.ndarray:
    """
    將二值化膜遮罩轉換為被膜包圍的填充細胞區域（含邊緣細胞）

    演算法：
    1. 形態學閉合 → 連接斷裂的膜片段
    2. 輪廓偵測 + 填充 → 將膜環的外部輪廓填滿（環內空洞變實心）
    3. 背景分析 → 分析背景連通區域，區分真正背景與被膜包圍的區域
       - 不碰邊界的背景 → 被膜完全包圍 → 填充
       - 碰邊界但面積小 → 邊緣細胞內部 → 填充
       - 碰邊界且面積大 → 真正背景 → 保留
    4. 形態學開運算 → 移除小突起與薄連接雜訊
    5. 面積過濾 → 移除過小的前景區域

    Args:
        dab_binary: 二值化後的膜遮罩 (H, W)，uint8 0/1
        close_kernel_size: 形態學閉合核大小（越大可跨越越寬的膜缺口）
        min_cell_area: 最小細胞面積 (像素)
        max_edge_hole_area: 邊緣細胞最大面積 (像素)，碰邊界的背景區域
                            小於此值視為邊緣細胞內部而非真正背景
        open_kernel_size: 形態學開運算核大小（移除小突起雜訊，0=不執行）

    Returns:
        filled_mask: 填充後的細胞區域遮罩 (H, W)，uint8 0/1
    """
    # 1. 形態學閉合 - 連接斷裂的膜片段
    if close_kernel_size > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
        )
        closed = cv2.morphologyEx(dab_binary, cv2.MORPH_CLOSE, close_kernel)
    else:
        closed = dab_binary.copy()

    # 2. 輪廓填充 - 找到膜的外部輪廓並填滿內部
    #    環狀膜的外輪廓被填滿後，原本環內的空洞就變成實心區域
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = closed.copy()
    for contour in contours:
        if cv2.contourArea(contour) >= min_cell_area:
            cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED)

    # 3. 背景連通區域分析 - 取代 binary_fill_holes，同時處理邊緣細胞
    #    原理：真正的背景是碰邊界且面積大的區域，
    #          其餘的「背景」要麼被膜完全包圍，要麼是邊緣細胞內部
    bg = (filled == 0).astype(np.uint8)
    bg_labeled, num_bg = measure.label(bg, connectivity=2, return_num=True)

    for i in range(1, num_bg + 1):
        region = (bg_labeled == i)
        area = np.sum(region)

        # 判斷是否觸及圖片邊界
        touches_border = (
            np.any(region[0, :]) or np.any(region[-1, :])
            or np.any(region[:, 0]) or np.any(region[:, -1])
        )

        if not touches_border:
            # 不碰邊界 → 被膜完全包圍的內部 → 填充
            filled[region] = 1
        elif area <= max_edge_hole_area:
            # 碰邊界但面積小 → 邊緣細胞內部 → 填充
            filled[region] = 1
        # else: 碰邊界且面積大 → 真正背景 → 保留為 0

    # 4. 形態學開運算 - 移除小突起與薄連接雜訊
    if open_kernel_size > 0:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size)
        )
        filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, open_kernel)

    # 5. 面積過濾 - 移除過小的前景區域（雜訊）
    labeled_regions, num_regions = measure.label(
        filled, connectivity=2, return_num=True
    )
    for i in range(1, num_regions + 1):
        region = (labeled_regions == i)
        area = np.sum(region)
        if area < min_cell_area:
            filled[region] = 0

    return filled


def generate_mask(
    image: np.ndarray,
    stain_matrix: np.ndarray = DEFAULT_STAIN_MATRIX,
    dab_threshold: float = 0.15,
    fill_close_kernel: int = 11,
    fill_min_cell_area: int = 200,
    fill_max_edge_hole_area: int = 5000,
    fill_open_kernel: int = 5,
) -> Tuple[np.ndarray, dict]:
    """
    從 IHC-HER2 影像生成填充細胞區域遮罩

    流程: RGB → QuPath 染色分解 → DAB 濃度 → 固定閾值二值化 → 填充

    Args:
        image: RGB 影像 (H, W, 3)
        stain_matrix: QuPath 染色矩陣 (3, 3)
        dab_threshold: DAB 濃度固定閾值 (separate_stains 輸出的光學密度，通常 0~1.5)
        fill_close_kernel: 形態學閉合核大小
        fill_min_cell_area: 最小細胞面積
        fill_max_edge_hole_area: 邊緣細胞最大面積
        fill_open_kernel: 形態學開運算核大小

    Returns:
        mask: 填充細胞區域遮罩 (H, W)，值域 0 或 255
        debug_info: 除錯資訊字典
    """
    # 1. QuPath 染色分解 → DAB 濃度
    dab_conc, _, stain_debug = separate_dab_qupath(image, stain_matrix)

    # 2. 固定閾值二值化 (不做 per-image 正規化，避免白底圖片雜訊被放大)
    dab_binary = (dab_conc > dab_threshold).astype(np.uint8)

    # 3. 填充被膜包圍的區域
    filled_region = fill_enclosed_regions(
        dab_binary=dab_binary,
        close_kernel_size=fill_close_kernel,
        min_cell_area=fill_min_cell_area,
        max_edge_hole_area=fill_max_edge_hole_area,
        open_kernel_size=fill_open_kernel,
    )

    mask_output = (filled_region * 255).astype(np.uint8)
    coverage = np.sum(filled_region > 0) / filled_region.size * 100

    debug_info = {
        **stain_debug,
        'dab_threshold': dab_threshold,
        'dab_binary': dab_binary,
        'filled_region': filled_region,
        'coverage_percent': coverage,
    }

    return mask_output, debug_info


def create_visualization(
    image: np.ndarray,
    mask: np.ndarray,
    debug_info: dict,
    membrane_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    建立六宮格視覺化圖像

    上排: Original | Hematoxylin | DAB
    下排: DAB Binary | Overlay | Filled Mask

    Args:
        image: 原始 RGB 影像 (H, W, 3)
        mask: 最終遮罩 (H, W)，0 或 255
        debug_info: 除錯資訊字典
        membrane_color: 覆蓋顏色 (R, G, B)

    Returns:
        vis: 視覺化圖像 (2H, 3W, 3)
    """
    h, w = image.shape[:2]

    # 確保影像為 uint8
    if image.dtype != np.uint8:
        img_display = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        img_display = image.copy()

    # Hematoxylin 重建視覺化
    hema_rgb = debug_info['hematoxylin_rgb']
    hema_vis = (np.clip(hema_rgb, 0, 1) * 255).astype(np.uint8)

    # DAB 重建視覺化
    dab_rgb = debug_info['dab_rgb']
    dab_vis = (np.clip(dab_rgb, 0, 1) * 255).astype(np.uint8)

    # DAB Binary 視覺化 (白色=膜, 黑色=背景)
    dab_binary = debug_info['dab_binary']
    dab_binary_vis = np.stack([dab_binary * 255] * 3, axis=-1).astype(np.uint8)

    # Overlay 視覺化 (半透明紅色覆蓋填充區域)
    overlay = img_display.copy().astype(np.float64)
    alpha_map = (mask > 0).astype(np.float64)
    blend_alpha = 0.4
    for c in range(3):
        overlay[:, :, c] = (
            overlay[:, :, c] * (1 - alpha_map * blend_alpha)
            + membrane_color[c] * alpha_map * blend_alpha
        )
    overlay = overlay.astype(np.uint8)

    # Filled Mask 視覺化 (紅底黑)
    mask_vis = np.zeros_like(img_display)
    mask_vis[:, :, 0] = mask

    # 組合六宮格 (2x3)
    top_row = np.hstack([img_display, hema_vis, dab_vis])
    bottom_row = np.hstack([dab_binary_vis, overlay, mask_vis])
    vis = np.vstack([top_row, bottom_row])

    # 添加標籤
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)
    thresh_val = debug_info.get('dab_threshold', 0)

    labels = [
        (10, 25, "Original"),
        (w + 10, 25, "Hematoxylin (QuPath)"),
        (2 * w + 10, 25, "DAB (QuPath)"),
        (10, h + 25, f"DAB Binary (thresh={thresh_val:.3f})"),
        (w + 10, h + 25, "Overlay (Filled Region)"),
        (2 * w + 10, h + 25, "Filled Cell Region"),
    ]

    for x, y, text in labels:
        cv2.putText(vis, text, (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(vis, text, (x, y), font, font_scale, color, thickness)

    return vis


def process_single_image(
    image_path: Path,
    output_mask_dir: Path,
    output_vis_dir: Optional[Path] = None,
    config=None,
) -> bool:
    """
    處理單張影像

    Args:
        image_path: 輸入影像路徑
        output_mask_dir: 遮罩輸出目錄
        output_vis_dir: 視覺化輸出目錄（可選）
        config: 配置物件

    Returns:
        success: 是否處理成功
    """
    if config is None:
        config = load_config()

    try:
        image = io.imread(str(image_path))
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        # 生成遮罩
        mask, debug_info = generate_mask(
            image,
            stain_matrix=np.array(config.stain_matrix),
            dab_threshold=config.dab_threshold,
            fill_close_kernel=config.fill_close_kernel,
            fill_min_cell_area=config.fill_min_cell_area,
            fill_max_edge_hole_area=config.fill_max_edge_hole_area,
            fill_open_kernel=config.fill_open_kernel,
        )

        # 儲存遮罩 (R 通道, 背景=黑色)
        output_mask_dir.mkdir(parents=True, exist_ok=True)
        mask_filename = image_path.stem + "_mask.png"
        mask_path = output_mask_dir / mask_filename

        h, w = mask.shape
        red_mask = np.zeros((h, w, 3), dtype=np.uint8)
        red_mask[:, :, 0] = mask
        io.imsave(str(mask_path), red_mask, check_contrast=False)

        # 儲存視覺化
        if output_vis_dir is not None:
            output_vis_dir.mkdir(parents=True, exist_ok=True)
            vis = create_visualization(
                image, mask, debug_info,
                membrane_color=config.vis_membrane_color,
            )
            vis_filename = image_path.stem + "_vis.png"
            vis_path = output_vis_dir / vis_filename
            io.imsave(str(vis_path), vis)

        logger.info(
            f"已處理: {image_path.name} | "
            f"覆蓋率: {debug_info['coverage_percent']:.2f}%"
        )
        return True

    except Exception as e:
        logger.error(f"處理失敗: {image_path.name} | 錯誤: {e}")
        return False


def process_directory(
    input_dir: Path,
    output_mask_dir: Path,
    output_vis_dir: Optional[Path] = None,
    extensions: Optional[List[str]] = None,
    config=None,
) -> Tuple[int, int]:
    """
    批次處理目錄中的所有影像

    Args:
        input_dir: 輸入目錄
        output_mask_dir: 遮罩輸出目錄
        output_vis_dir: 視覺化輸出目錄（可選）
        extensions: 支援的副檔名列表
        config: 配置物件

    Returns:
        (success_count, total_count): 成功/總共數量
    """
    if config is None:
        config = load_config()

    if extensions is None:
        extensions = config.supported_extensions

    # 收集所有影像
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_dir.glob(f"*{ext}"))
        image_paths.extend(input_dir.glob(f"*{ext.upper()}"))

    image_paths = sorted(set(image_paths))
    total_count = len(image_paths)

    if total_count == 0:
        logger.warning(f"目錄中沒有找到影像: {input_dir}")
        return 0, 0

    logger.info(f"找到 {total_count} 張影像，開始處理...")

    max_workers = config.num_workers
    logger.info(f"啟動多核心處理，使用核心數: {max_workers}")

    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_image, path, output_mask_dir, output_vis_dir, None
            ): path
            for path in image_paths
        }

        for i, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                if future.result():
                    success_count += 1
                logger.info(f"[{i}/{total_count}] 已完成: {path.name}")
            except Exception as e:
                logger.error(f"[{i}/{total_count}] 失敗: {path.name} | 錯誤: {e}")

    logger.info(f"處理完成: {success_count}/{total_count} 成功")
    return success_count, total_count


def main() -> None:
    """主程式入口"""
    config = load_config()

    input_path = config.train_image_dir
    output_mask_dir = config.mask_dir

    save_vis = config.save_visualization
    output_vis_dir = config.pseudo_label_vis_dir if save_vis else None

    logger.info("=" * 60)
    logger.info("IHC-HER2 填充細胞區域 Pseudo-Label 生成器")
    logger.info("=" * 60)
    logger.info(f"輸入路徑: {input_path}")
    logger.info(f"遮罩輸出: {output_mask_dir}")
    logger.info(f"視覺化輸出: {output_vis_dir}")
    logger.info("填充參數:")
    logger.info(f"  - dab_threshold: {config.dab_threshold}")
    logger.info(f"  - close_kernel: {config.fill_close_kernel}")
    logger.info(f"  - min_cell_area: {config.fill_min_cell_area}")
    logger.info("=" * 60)

    if input_path.is_file():
        process_single_image(input_path, output_mask_dir, output_vis_dir, config)
    elif input_path.is_dir():
        process_directory(input_path, output_mask_dir, output_vis_dir, config=config)
    else:
        logger.error(f"輸入路徑不存在: {input_path}")
        raise FileNotFoundError(f"輸入路徑不存在: {input_path}")


if __name__ == "__main__":
    main()
