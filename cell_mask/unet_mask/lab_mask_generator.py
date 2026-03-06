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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    
    # 移除 CLAHE 處理，保留原始亮度數值避免強制拉黑
    return L, A, B


def detect_brown_membrane_lab(
    image: np.ndarray,
    l_min: float = 0.0,
    l_max: float = 80.0,
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
    
    # 移除全域 L 修正，防止淡色棕色或邊緣被過度放大而變得太粗
    A_norm = A_positive / 60.0
    A_norm = np.clip(A_norm, 0, 1)
    
    # 正規化 B 通道 (只取正值部分)  
    B_positive = np.clip(B, 0, None)  # 只保留正值 (偏黃)
    B_norm = B_positive / 60.0
    B_norm = np.clip(B_norm, 0, 1)
    
    # 計算棕色分數 (A+ 和 B+ 的幾何平均) (LAB Score)
    # [改良] 加上 Gamma 校正 (0.7~0.8) 來增幅微弱訊號，但強訊號依然最高為 1，厚度不變
    lab_score = np.sqrt(A_norm * B_norm)
    lab_score = np.power(lab_score, 0.75)
    
    brown_score = lab_score
    dab_debug = None
    
    if use_dab_fusion:
        # 計算 DAB 通道 (HED)
        hed = rgb2hed(image)
        dab = hed[:, :, 2] # DAB 通道
        
        # 正規化 DAB (0-1)
        # 1. 取得正值的 DAB
        dab_positive = np.clip(dab, 0, None)
        
        # 2. 自動判斷動態正規化 (取代強制的 MINMAX)
        # 避免在完全沒有膜的純白、細胞質區域放大極小雜訊 (這會造成滿屏紅色)
        v_max = dab_positive.max()
        if v_max > 0.15:
            # 圖塊中有明確的深棕色膜：使用最大最小值拉伸，強化對比
            dab_norm = dab_positive / v_max
        else:
            # 圖塊中沒有深色目標 (都是淺色雜訊)：保持微弱，不強制拉升到滿分為1
            dab_norm = np.clip(dab_positive / 0.15, 0, 1.0)
            
        # [改良] 對 DAB 訊號同樣加上 Gamma 校正增幅微弱訊號
        dab_norm = np.power(dab_norm, 0.75)
            
        # 融合 LAB 和 DAB
        # 恢復為幾何平均 (AND 邏輯)，這能有效過濾掉單獨在 DAB 或 LAB 的雜訊
        base_score = np.sqrt(lab_score * dab_norm)
        
        # 針對極深色的最終殺手鐧 (Black/Dark Brown Compensation):
        # 為了避免線條全面變粗（蓋到其他東西），我們僅針對極黑且具有「強烈 DAB 訊號」的隙縫做精準修補。
        # 限定條件：L < 45 (限縮在極黑範圍)、不偏藍 (B > -10) 防核、不偏綠 (A > -5)
        is_dark_membrane = (L < 45.0) & (B > -10.0) & (A > -5.0)
        
        # 建立一個與影像同大小的 dark_boost_score
        # 讓 L=45 分數為 0，L<=20 分數為 1.0
        dark_boost_score = np.zeros_like(base_score)
        compensation = np.clip((45.0 - L) / 25.0, 0.0, 1.0)
        
        # 【重要】：避免極黑訊號無腦擴張擴散，必須與 dab_norm 掛鉤！
        # 確保只有在 DAB 型態上也像是深色膜的縫隙才做補洞
        dark_boost_score[is_dark_membrane] = compensation[is_dark_membrane] * dab_norm[is_dark_membrane]
        
        brown_score = np.maximum(base_score, dark_boost_score)
        
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
    l_min: float = 0.0,
    l_max: float = 80.0,
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
    
    # [新增技術] 結構特化增強：使用 Frangi Filter (血管脊線濾波器) 強化微弱的膜狀線條
    # 由於細胞膜呈線狀/環形，雜訊多呈塊狀；Frangi 只會對「線狀結構」產生高頻響應。
    # 這樣即可針對「若隱若現但連成一線的微弱咖啡色」進行局部增強，而不會無腦擴張到非特定區域。
    try:
        # sigmas: 控制線條的粗細尺度 (1~2 足以捕捉細胞膜的寬度)
        ridge_score = frangi(brown_score_mask, sigmas=(1, 2), black_ridges=False)
        if ridge_score.max() > 0:
            ridge_score = ridge_score / ridge_score.max()
        
        # 放大 Frangi 權重以強制縫合極微弱邊緣 (由 0.6 提升至 1.0)
        # 用意是防呆，避免 Frangi 在全白背景上無中生有線條。
        is_faint_brown = brown_score_mask > 0.01  # 調低接觸門檻，抓取更邊緣微弱的訊號
        brown_score_enhanced = brown_score_mask.copy()
        brown_score_enhanced[is_faint_brown] = np.clip(
            brown_score_mask[is_faint_brown] + ridge_score[is_faint_brown] * 1.0,
            0, 1.0
        )
        brown_score_mask = brown_score_enhanced
    except Exception as e:
        logger.warning(f"Frangi 濾波器強化失敗: {e}")
    
    # 轉換為 uint8 格式 (0-255) (膜=數值高, 背景=0)
    mask_output = (brown_score_mask * 255).astype(np.uint8)
    
    # 形態學閉合 (Closing)：連接斷裂的膜結構
    # 將核心從 5x5 改為 3x3，避免線條變太粗而遮蓋其他細胞細節
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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
            l_min=getattr(config, 'lab_l_min', 0.0),
            l_max=getattr(config, 'lab_l_max', 80.0),
            use_dab_fusion=getattr(config, 'use_dab_fusion', True),
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
                overlay_alpha=getattr(config, 'vis_overlay_alpha', 0.5),
                membrane_color=getattr(config, 'vis_membrane_color', (255, 0, 0)),
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
        extensions = getattr(config, 'supported_extensions', [".tiff", ".tif", ".png", ".jpg", ".jpeg"])
    
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
    
    # 支援多核心處理
    max_workers = getattr(config, 'num_workers', max(1, multiprocessing.cpu_count() - 2))  # 預設保留 2 核心給系統
    logger.info(f"啟動多核心處理，使用核心數: {max_workers}")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 預先提交所有任務 (為了避免 Config 模組無法被 Pickle，傳遞 config=None 讓各個 Worker 獨立載入)
        futures = {
            executor.submit(process_single_image, path, output_mask_dir, output_vis_dir, None): path
            for path in image_paths
        }
        
        # 收集結果
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
    
    input_path = getattr(config, 'kmeans_input_path', getattr(config, 'pseudo_label_input_dir', getattr(config, 'train_image_dir', Path(__file__).parent / "tile/train/her2_chose")))
    output_mask_dir = getattr(config, 'kmeans_mask_dir', getattr(config, 'pseudo_label_mask_dir', getattr(config, 'mask_dir', Path(__file__).parent / "output/mask")))
    
    save_vis = getattr(config, 'kmeans_save_visualization', getattr(config, 'save_visualization', True))
    output_vis_dir = getattr(config, 'kmeans_vis_dir', getattr(config, 'pseudo_label_vis_dir', Path(__file__).parent / "output/vis")) if save_vis else None
    
    logger.info("=" * 60)
    logger.info("LAB 色彩空間膜分割生成器")
    logger.info("=" * 60)
    logger.info(f"輸入路徑: {input_path}")
    logger.info(f"遮罩輸出: {output_mask_dir}")
    logger.info(f"視覺化輸出: {output_vis_dir}")
    logger.info(f"LAB 參數:")
    l_min = getattr(config, 'lab_l_min', 0.0)
    l_max = getattr(config, 'lab_l_max', 80.0)
    logger.info(f"  - L range: [{l_min}, {l_max}]")
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
