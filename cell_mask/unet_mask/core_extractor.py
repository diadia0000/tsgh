"""
HER2 3+ 陽性細胞內部 (Core) 萃取模組

遵循架構設計：
1. 透過 UNet++ 獲取細胞膜 (Membrane) Mask
2. 利用形態學與拓樸學 (Topological Topology) 進行細胞膜破洞修補 (Dilation)
3. 將細胞膜圍繞的區域進行孔洞填充 (Fill Holes)，獲取「細胞膜 + 內部核心」混合區域
4. 邏輯相減：(細胞膜 + 內部核心) - 細胞膜 = 內部核心 (Core)

Author: TSGH AI Team
Date: 2026-02-28
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
from scipy import ndimage
from skimage import io, measure

from config import config

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_cell_cores(
    membrane_mask: np.ndarray,
    dilate_kernel_size: int = config.membrane_dilate_kernel,
    close_kernel_size: int = 20,
    max_boundary_gap: int = 400
) -> np.ndarray:
    """
    從細胞膜遮罩中萃取出被包圍的細胞內部 (Cell Cores)。
    新增：封閉邊界缺口，支援邊緣細胞的萃取。

    演算法步驟：
    1. 形態學閉合 (Closing)：以較大 Kernel 將膜上斷裂的中小型缺口縫合起來 (Bridge Gaps)。
    2. 膨脹 (Dilation)：將修補好的膜稍微加粗，確保絕對 watertight 防止填充外漏。
    3. 封閉邊界 (Close Edge Gaps)：尋找接觸影像邊界的短缺口，將其閉合以形成城牆。
    4. 填充 (Fill Holes)：把被膜包圍的內部孔洞填滿。
    5. 邏輯相減：將填充後的區域減去「閉合後的細胞膜」，得到內部的核心區域。
       (使用閉合後的膜而非原始膜，避免 Closing 橋接像素殘留為白線)

    Args:
        membrane_mask (np.ndarray): UNet++ 輸出的細胞膜二值化遮罩，shape (H, W)，值域 0 或 1。
        dilate_kernel_size (int): 膨脹核大小，用於最終確保水密性。預設由 config 提供。
        close_kernel_size (int): 閉合核大小，專門用來強力縫合破裂的膜，而不使其變粗。
        max_boundary_gap (int): 允許閉合的邊界最大缺口長度 (pixel)。預設 400 足以涵蓋一般邊緣細胞。

    Returns:
        np.ndarray: 乾淨的細胞內部核心二值化遮罩，shape (H, W)，值域 0 (背景) 或 1 (細胞內部)。
    """
    # 確保 mask 為 uint8 格式且只包含 0 和 1
    membrane_uint8 = (membrane_mask > 0).astype(np.uint8)

    # 1. 形態學閉合 (Closing)：縫合斷裂的缺口 (不增加膜的整體厚度)
    if close_kernel_size > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (close_kernel_size, close_kernel_size)
        )
        closed_membrane = cv2.morphologyEx(membrane_uint8, cv2.MORPH_CLOSE, close_kernel)
    else:
        closed_membrane = membrane_uint8.copy()

    # 2. 膨脹細胞膜 (修補極微小斷裂處並確保完全封閉)
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (dilate_kernel_size, dilate_kernel_size)
    )
    dilated_membrane = cv2.dilate(closed_membrane, dilate_kernel, iterations=1)

    # 2. 封閉邊緣缺口 (針對邊緣細胞)
    # 建立邊界遮罩 (只有影像四個外框邊為 1)
    boundary_mask = np.zeros_like(dilated_membrane)
    boundary_mask[0, :] = 1
    boundary_mask[-1, :] = 1
    boundary_mask[:, 0] = 1
    boundary_mask[:, -1] = 1

    # 找出邊界上「缺乏細胞膜」的缺口像素
    zero_boundary = np.logical_and(boundary_mask == 1, dilated_membrane == 0)

    # 標記邊界上獨立的連續缺口 (connectivity=2 確保四個角落可以連通)
    labeled_boundary, num = measure.label(zero_boundary, connectivity=2, return_num=True)
    
    # 遍歷每個缺口，若長度小於 max_boundary_gap，則將其設為細胞膜 (封閉它)
    for i in range(1, num + 1):
        gap_mask = (labeled_boundary == i)
        if np.sum(gap_mask) <= max_boundary_gap:
            dilated_membrane[gap_mask] = 1

    # 3. 拓樸孔洞填充 (取得 細胞膜 + 細胞內部)
    # binary_fill_holes 將完全被 1 包圍的 0 填成 1
    filled_mask = ndimage.binary_fill_holes(dilated_membrane).astype(np.uint8)

    # 4. 移除膨脹造成的外部光暈 (Halo)
    # 由於細胞膜曾向外膨脹，filled_mask 包含了外圍的一圈「膨脹假核心」。
    # 直接減去原始膜會殘留一圈白線。解法：尋找目前的外部背景，將背景向內膨脹相同的距離，
    # 就能將背景精準切齊回原始細胞膜外側邊界。
    exterior_background = (filled_mask == 0).astype(np.uint8)
    restored_exterior = cv2.dilate(exterior_background, dilate_kernel, iterations=1)
    
    # 真正的細胞實體區塊 (True Filled Core) ＝ 既不屬於外部背景，也不屬於原始細胞膜
    true_core_region = (restored_exterior == 0).astype(np.uint8)

    # 邏輯相減：真正的實體區塊 - 閉合後的細胞膜 = 純淨的內部
    # 注意：必須減去 closed_membrane (而非原始 membrane_uint8)，
    # 因為形態學閉合 (Closing) 會產生「橋接像素」來封閉膜上的缺口，
    # 這些橋接像素不在原始膜中，若只減去原始膜會導致殘留白線。
    core_mask = np.logical_and(true_core_region, np.logical_not(closed_membrane)).astype(np.uint8)

    return core_mask


def process_image_pipeline(
    image_path: Path,
    output_dir: Path,
    inferencer: 'UNetPPInference'  # type: ignore # 推論器物件
) -> Optional[Path]:
    """
    執行單張影像的完整流程： UNet++ 推論 -> 取出細胞核心 -> 儲存結果

    Args:
        image_path (Path): 原始影像路徑
        output_dir (Path): 結果儲存目錄
        inferencer: 已經初始化的 UNetPPInference 物件

    Returns:
        Optional[Path]: 儲存結果的圖片路徑。若處理失敗則回傳 None。
    """
    try:
        # 推論取得細胞膜 Mask
        # 此處呼叫 inferencer.predict_single，回傳 (H, W) array
        membrane_mask = inferencer.predict_single(image_path)
        
        # 萃取細胞內部
        core_mask = extract_cell_cores(
            membrane_mask=membrane_mask,
            dilate_kernel_size=config.membrane_dilate_kernel
        )
        
        # 準備疊加視覺化
        original_img = io.imread(str(image_path))
        if original_img.ndim == 2:
            original_img = np.stack([original_img]*3, axis=-1)
        elif original_img.shape[2] == 4:
            original_img = original_img[:, :, :3]

        # 建立純黑的背景
        overlay = np.zeros_like(original_img)
        
        # 標記細胞核心 (保留原圖像素)
        core_bool = (core_mask == 1)
        overlay[core_bool] = original_img[core_bool]

        # 儲存結果
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / f"{image_path.stem}_core_extracted.png"
        io.imsave(str(result_path), overlay)
        
        return result_path

    except Exception as e:
        logger.error(f"處裡影像 {image_path.name} 時發生錯誤: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # 簡單的測試執行模塊
    # 這裡我們需要匯入推理模塊來啟動
    try:
        from inference import UNetPPInference
    except ImportError:
        logger.error("無法匯入 UNetPPInference，請確認 inference.py 存在。")
        exit(1)

    # 取得模型與資料夾路徑
    model_path = config.model_save_dir / "best_model.pth"
    test_img_dir = config.inference_input_dir
    output_dir = config.base_dir / "output" / "core_extraction"

    logger.info("初始化 UNet++ 推論器...")
    num_classes = config.num_classes if hasattr(config, 'num_classes') else 2
    
    if not model_path.exists():
        logger.error(f"找不到模型權重: {model_path}")
        exit(1)

    inferencer = UNetPPInference(
        model_path=model_path,
        encoder_name=config.encoder_name,
        num_classes=num_classes,
        image_size=config.image_size,
    )

    # 讀取測試圖並處理
    supported_exts = config.supported_extensions if hasattr(config, 'supported_extensions') else [".tiff", ".tif", ".png", ".jpg", ".jpeg"]
    image_paths = []
    for ext in supported_exts:
        image_paths.extend(test_img_dir.glob(f"*{ext}"))
    if not image_paths:
        logger.warning(f"在 {test_img_dir} 找不到任何影像！")
    else:
        logger.info(f"找到 {len(image_paths)} 張測試影像，準備萃取細胞內部...")
        for img_path in image_paths:
            logger.info(f"正在處理: {img_path.name}")
            saved_path = process_image_pipeline(img_path, output_dir, inferencer)
            if saved_path:
                logger.info(f"成功儲存結果至: {saved_path}")
