"""
LAB 色彩空間膜分割生成器

使用 LAB 色彩空間分析進行細胞膜分割:
- LAB 色彩空間可精準分離棕色 (DAB) 染色
- L = 亮度, A = 綠(-)/紅(+), B = 藍(-)/黃(+)
- 棕色染色會呈現 A+ 和 B+ 的特徵
- 無需人工標註，可用於產生偽標籤 (Pseudo Labels)

原理:
    在 LAB 色彩空間中，棕色 (Brown) 具有:
    - 中等亮度 (L)
    - 正的 A 值 (偏紅)
    - 正的 B 值 (偏黃)
    
    透過分析 A 和 B 通道，可以有效分離 DAB 棕色染色區域。
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
from skimage import io
from skimage.color import rgb2lab, rgb2hed
from skimage.filters import frangi
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    disk,
)

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def extract_lab_channels(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    從 RGB 影像中提取 LAB 通道，並對 L 通道應用 CLAHE 強化
    
    LAB 色彩空間:
    - L: 亮度 (0-100)，應用 CLAHE 增強對比
    - A: 綠(-128) 到 紅(+127)，保持不變
    - B: 藍(-128) 到 黃(+127)，保持不變
    
    Args:
        image: RGB 影像，形狀為 (H, W, 3)，值域 0-255
        
    Returns:
        L, A, B: 三個通道，各為 (H, W)
    """
    # 確保影像為正確格式
    if image.dtype == np.uint8:
        image_float = image.astype(np.float64) / 255.0
    else:
        image_float = image.astype(np.float64)
    
    # RGB 轉 LAB
    lab = rgb2lab(image_float)
    
    L = lab[:, :, 0]  # 亮度 0-100
    A = lab[:, :, 1]  # 綠紅軸 -128 to 127
    B = lab[:, :, 2]  # 藍黃軸 -128 to 127
    
    # 對 L 通道應用 CLAHE 強化 (團隊測試參數)
    # 將 L 從 0-100 轉換為 0-255 以應用 CLAHE
    L_uint8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L_enhanced = clahe.apply(L_uint8)
    # 轉回 0-100 範圍
    L = L_enhanced.astype(np.float64) / 2.55
    
    return L, A, B


def detect_brown_membrane_lab(
    image: np.ndarray,
    l_min: float = 15.0,
    l_max: float = 85.0,
    use_dab_fusion: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    使用 LAB 色彩空間偵測棕色膜結構
    
    棕色 (Brown/DAB) 在 LAB 空間中的特徵:
    - A > 0: 偏紅色
    - B > 0: 偏黃色
    - 中等亮度 L
    
    直接輸出棕色分數作為遮罩，保留連續強度資訊。
    
    Args:
        image: RGB 影像 (H, W, 3)
        l_min: 最小亮度（排除太暗區域）
        l_max: 最大亮度（排除太亮區域/背景）
        
    Returns:
        brown_score_mask: 棕色分數遮罩 (H, W)，值域 0-1
        debug_info: 除錯資訊字典
    """
    # 提取 LAB 通道
    L, A, B = extract_lab_channels(image)
    
    # 計算棕色分數
    # 棕色特徵: A > 0 (紅) 且 B > 0 (黃)
    # 分數 = normalized(A) * normalized(B)，只考慮正值
    
    # 正規化 A 通道 (只取正值部分)
    A_positive = np.clip(A, 0, None)  # 只保留正值 (偏紅)
    A_norm = A_positive / 60.0  # 正規化，假設最大合理值約 60
    A_norm = np.clip(A_norm, 0, 1)
    
    # 正規化 B 通道 (只取正值部分)  
    B_positive = np.clip(B, 0, None)  # 只保留正值 (偏黃)
    B_norm = B_positive / 60.0  # 正規化
    B_norm = np.clip(B_norm, 0, 1)
    
    # 計算棕色分數 (A+ 和 B+ 的幾何平均) (LAB Score)
    lab_score = np.sqrt(A_norm * B_norm)
    
    brown_score = lab_score
    dab_debug = None
    
    if use_dab_fusion:
        # 計算 DAB 通道 (HED)
        hed = rgb2hed(image)
        dab = hed[:, :, 2] # DAB 通道
        
        # 正規化 DAB (0-1)
        # 1. 先從 Log Space 轉回線性 (如果是 HED 的原始輸出) 或直接 Clip
        dab_positive = np.clip(dab, 0, None)
        
        # 2. 轉換為 uint8 以進行 CLAHE
        dab_uint8 = cv2.normalize(dab_positive, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 3. 應用 CLAHE (局部對比增強)
        # clipLimit: 對比度限制 (越高對比越強，但也易放大雜訊)
        # tileGridSize: 網格大小 (越小細節越多)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        dab_enhanced = clahe.apply(dab_uint8)
        
        # 4. 轉回 0-1 float，並維持飽和閾值邏輯
        # 雖然 CLAHE 已經均衡化了，但設定飽和閾值仍有助於突出膜
        dab_norm = dab_enhanced.astype(np.float32) / 255.0
        
        # 融合 LAB 和 DAB (幾何平均)
        # Final = sqrt(LAB * DAB)
        brown_score = np.sqrt(lab_score * dab_norm)
        
        dab_debug = dab_norm

    # 亮度條件 - 只排除白色背景和太暗區域
    valid_lightness = (L >= l_min) & (L <= l_max)
    
    # 將無效區域設為 0
    brown_score_mask = brown_score.copy()
    brown_score_mask[~valid_lightness] = 0
    
    # 收集除錯資訊
    debug_info = {
        'L_channel': L,
        'A_channel': A,
        'B_channel': B,
        'lab_score': lab_score,
        'dab_channel': dab_debug,
        'brown_score': brown_score,
        'valid_lightness': valid_lightness,
    }
    
    return brown_score_mask, debug_info



def postprocess_mask(
    mask: np.ndarray,
    min_region_size: int = 50,
    min_hole_size: int = 50,
    closing_radius: int = 2,
) -> np.ndarray:
    """
    後處理遮罩：形態學清理
    
    Args:
        mask: 二值遮罩 (H, W)
        min_region_size: 最小區域大小（移除小於此值的區域）
        min_hole_size: 最小空洞大小（填補小於此值的空洞）
        closing_radius: 形態學閉合半徑
        
    Returns:
        cleaned_mask: 清理後的遮罩
    """
    cleaned = mask.copy()
    
    # 形態學閉合 - 連接斷裂的膜結構
    if closing_radius > 0:
        cleaned = binary_closing(cleaned, disk(closing_radius))
    
    # 移除小區域
    cleaned = remove_small_objects(cleaned, min_size=min_region_size)
    
    # 填補小空洞
    cleaned = remove_small_holes(cleaned, area_threshold=min_hole_size)
    
    return cleaned


def generate_lab_mask(
    image: np.ndarray,
    l_min: float = 15.0,
    l_max: float = 85.0,
    use_dab_fusion: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    使用 LAB 色彩空間生成膜分割遮罩
    
    直接輸出棕色分數作為灰階遮罩，保留完整的強度資訊。
    
    Args:
        image: RGB 影像 (H, W, 3)
        l_min: 最小亮度（排除太暗區域）
        l_max: 最大亮度（排除太亮區域/背景）
        use_dab_fusion: 是否融合 DAB 通道
        
    Returns:
        mask: 棕色分數遮罩 (H, W)，值域 0-255
        debug_info: 除錯資訊字典，包含中間結果
    """
    # LAB (+DAB) 偵測棕色
    brown_score_mask, lab_debug = detect_brown_membrane_lab(
        image,
        l_min=l_min,
        l_max=l_max,
        use_dab_fusion=use_dab_fusion,
    )
    
    # 轉換為 uint8 格式 (0-255) (膜=數值高, 背景=0)
    mask_output = (brown_score_mask * 255).astype(np.uint8)
    
    # 形態學閉合 (Closing)：連接斷裂的膜結構
    # 使用 5x5 核心，足以修補大部分的斷裂，且不會過度影響粗細
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    mask_output = cv2.morphologyEx(mask_output, cv2.MORPH_CLOSE, kernel)
    
    # 計算覆蓋率 (有顏色的像素佔比)
    coverage = np.sum(brown_score_mask > 0.1) / brown_score_mask.size * 100
    
    # 收集除錯資訊
    debug_info = {
        **lab_debug,
        'membrane_mask_raw': brown_score_mask,
        'membrane_mask_final': brown_score_mask,
        'coverage_percent': coverage,
    }
    
    return mask_output, debug_info


def create_visualization(
    image: np.ndarray,
    mask: np.ndarray,
    debug_info: dict,
    overlay_alpha: float = 1,
    membrane_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    建立視覺化圖像
    
    產生四宮格視覺化:
    - 左上: 原圖
    - 右上: A 通道 (綠-紅軸)
    - 左下: 棕色分數
    - 右下: Overlay 結果
    
    Args:
        image: 原始 RGB 影像
        mask: 最終遮罩
        debug_info: 除錯資訊
        overlay_alpha: 覆蓋透明度
        membrane_color: 膜顏色 (R, G, B)
        
    Returns:
        vis: 視覺化圖像 (2H, 2W, 3)
    """
    h, w = image.shape[:2]
    
    # 確保影像為 uint8
    if image.dtype != np.uint8:
        img_display = (image * 255).astype(np.uint8)
    else:
        img_display = image.copy()
    
    # DAB 通道視覺化 (HED 分離)
    hed = rgb2hed(image)
    dab = hed[:, :, 2]
    # 正規化 DAB 通道以利顯示 (0-255)
    dab_vis = cv2.normalize(dab, None, 0, 255, cv2.NORM_MINMAX)
    dab_vis = dab_vis.astype(np.uint8)
    dab_vis = cv2.applyColorMap(dab_vis, cv2.COLORMAP_BONE) # 使用骨骼色圖 (黑白帶藍) 或其他
    dab_vis = cv2.cvtColor(dab_vis, cv2.COLOR_BGR2RGB)

    # A 通道視覺化 (綠-紅軸, -128 to 127)
    A_channel = debug_info['A_channel']
    # 正規化到 0-255
    A_norm = ((A_channel + 128) / 256 * 255).astype(np.uint8)
    A_vis = cv2.applyColorMap(A_norm, cv2.COLORMAP_JET)
    A_vis = cv2.cvtColor(A_vis, cv2.COLOR_BGR2RGB)
    
    # 棕色分數視覺化
    brown_score = debug_info['brown_score']
    brown_vis = (brown_score * 255).astype(np.uint8)
    brown_vis = cv2.applyColorMap(brown_vis, cv2.COLORMAP_HOT)
    brown_vis = cv2.cvtColor(brown_vis, cv2.COLOR_BGR2RGB)
    
    # Overlay 視覺化 - 使用 Mask 的數值作為 Alpha
    overlay = img_display.copy().astype(np.float64)
    
    # 直接使用傳入的 mask (0-255) 作為 Alpha 權重
    # 因為 mask 已經經過了所有融合計算，它是最準確的結果
    # Overlay 視覺化 - 直接使用 Mask 進行二值化
    
    # 直接拿輸出的 mask (0-255) 來做二值化
    # 設定閾值 20 (約 8% 強度)，過濾掉肉眼幾乎看不見的極淡邊緣
    # 這樣出來的黑色框會比之前 (閾值 0.05) 細緻很多，且跟紅色遮罩形狀一致
    _, binary_mask = cv2.threshold(mask, 20, 1.0, cv2.THRESH_BINARY) 
    alpha_map = binary_mask
    
    for c in range(3):
        overlay[:, :, c] = (
            overlay[:, :, c] * (1 - alpha_map) +
            membrane_color[c] * alpha_map
        )
    overlay = overlay.astype(np.uint8)

    # 遮罩預覽 (紅底黑)
    mask_vis = np.zeros_like(img_display)
    mask_vis[:, :, 0] = mask
    
    # 組合六宮格 (2x3)
    # 上排: Original | DAB | A Channel
    top_row = np.hstack([img_display, dab_vis, A_vis])
    # 下排: Brown Score | Overlay | Mask Preview
    bottom_row = np.hstack([brown_vis, overlay, mask_vis])
    
    vis = np.vstack([top_row, bottom_row])
    
    # 添加標籤
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)
    
    labels = [
        (10, 25, "Original"),
        (w + 10, 25, "DAB Channel (HED)"),
        (2 * w + 10, 25, "A Channel (Green-Red)"),
        (10, h + 25, "Brown Score (LAB)"),
        (w + 10, h + 25, "Overlay (Black > 20)"),
        (2 * w + 10, h + 25, "Mask Output (Red)"),
    ]
    
    for x, y, text in labels:
        # 黑色陰影
        cv2.putText(vis, text, (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 1)
        # 白色文字
        cv2.putText(vis, text, (x, y), font, font_scale, color, thickness)
    
    return vis


def process_single_image(
    image_path: Path,
    output_mask_dir: Path,
    output_vis_dir: Optional[Path] = None,
    config = None,
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
        # 讀取影像
        image = io.imread(str(image_path))
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # 生成遮罩 (使用 LAB 色彩空間分析，直接輸出棕色分數)
        mask, debug_info = generate_lab_mask(
            image,
            l_min=config.lab_l_min,
            l_max=config.lab_l_max,
            use_dab_fusion=config.use_dab_fusion,
        )
        
        # 儲存遮罩 (膜=紅色, 背景=黑色)
        output_mask_dir.mkdir(parents=True, exist_ok=True)
        mask_filename = image_path.stem + "_mask.png"
        mask_path = output_mask_dir / mask_filename
        
        # 建立 RGB 遮罩
        h, w = mask.shape
        red_mask = np.zeros((h, w, 3), dtype=np.uint8)
        red_mask[:, :, 0] = mask  # 將數值填入 R 通道
        # G, B 通道保持 0 (黑色)
        
        io.imsave(str(mask_path), red_mask, check_contrast=False)
        
        # 儲存視覺化
        if output_vis_dir is not None:
            output_vis_dir.mkdir(parents=True, exist_ok=True)
            vis = create_visualization(
                image, mask, debug_info,
                overlay_alpha=config.vis_overlay_alpha,
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
    config = None,
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
    
    success_count = 0
    for i, image_path in enumerate(image_paths, 1):
        logger.info(f"[{i}/{total_count}] 處理中: {image_path.name}")
        if process_single_image(image_path, output_mask_dir, output_vis_dir, config):
            success_count += 1
    
    logger.info(f"處理完成: {success_count}/{total_count} 成功")
    return success_count, total_count


def main() -> None:
    """主程式入口"""
    config = load_config()
    
    input_path = config.kmeans_input_path
    output_mask_dir = config.kmeans_mask_dir
    output_vis_dir = config.kmeans_vis_dir if config.kmeans_save_visualization else None
    
    logger.info("=" * 60)
    logger.info("LAB 色彩空間膜分割生成器")
    logger.info("=" * 60)
    logger.info(f"輸入路徑: {input_path}")
    logger.info(f"遮罩輸出: {output_mask_dir}")
    logger.info(f"視覺化輸出: {output_vis_dir}")
    logger.info(f"LAB 參數:")
    logger.info(f"  - L range: [{config.lab_l_min}, {config.lab_l_max}]")
    logger.info("=" * 60)
    
    if input_path.is_file():
        # 單張影像
        process_single_image(input_path, output_mask_dir, output_vis_dir, config)
    elif input_path.is_dir():
        # 目錄批次處理
        process_directory(input_path, output_mask_dir, output_vis_dir, config=config)
    else:
        logger.error(f"輸入路徑不存在: {input_path}")
        raise FileNotFoundError(f"輸入路徑不存在: {input_path}")


if __name__ == "__main__":
    main()
