"""
M1: IHC-DISH 遮罩疊加模組

將 IHC Her2+ 核心遮罩 (core mask) 套用至 DISH 影像，
保留 ROI 區域的原始像素，背景填充為指定值。

依賴:
  - UNet++ 膜分割推論器 (inference.UNetPPInference)
  - 核心萃取器 (inference.extract_cell_cores)
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Tile 座標解析
# ------------------------------------------------------------------
_TILE_COORD_PATTERN = re.compile(r"tile_x(\d+)_y(\d+)")


def parse_tile_coords(tile_name: str) -> Tuple[int, int]:
    """從 tile 檔名解析 (x, y) 座標。

    Args:
        tile_name: 檔名 (含或不含副檔名), 例如 ``tile_x1024_y2048.tiff``。

    Returns:
        (x, y) 整數座標 tuple。

    Raises:
        ValueError: 無法從檔名中解析座標。
    """
    match = _TILE_COORD_PATTERN.search(tile_name)
    if match is None:
        raise ValueError(
            f"無法從 '{tile_name}' 解析座標，"
            f"預期格式: tile_x{{int}}_y{{int}}"
        )
    return int(match.group(1)), int(match.group(2))


# ------------------------------------------------------------------
# IHC Core Mask 產生器
# ------------------------------------------------------------------

def generate_ihc_core_mask(
    ihc_tile_path: Path,
    unet_inferencer: object,
    dilate_kernel: int = 7,
    close_kernel: int = 20,
    max_boundary_gap: int = 400,
    sliding_window_overlap: int = 128,
) -> np.ndarray:
    """透過 UNet++ 推論 + 核心萃取生成 IHC Her2+ 核心遮罩。

    當影像大於模型訓練尺寸時，自動以滑動視窗推論。

    Args:
        ihc_tile_path: IHC tile 影像路徑。
        unet_inferencer: 已初始化的 ``UNetPPInference`` 物件。
        dilate_kernel: 膜膨脹核大小 (pixels)。
        close_kernel: 形態學閉合核大小 (pixels)。
        max_boundary_gap: 允許閉合的邊界最大缺口長度 (pixels)。
        sliding_window_overlap: 滑動視窗重疊像素。

    Returns:
        shape ``(H, W)``、值域 ``{0, 1}`` 的 ``uint8`` 核心遮罩。
    """
    from inference import extract_cell_cores  # noqa: WPS433

    membrane_mask: np.ndarray = unet_inferencer.predict_single(
        ihc_tile_path, overlap=sliding_window_overlap,
    )
    core_mask = extract_cell_cores(
        membrane_mask,
        dilate_kernel_size=dilate_kernel,
        close_kernel_size=close_kernel,
        max_boundary_gap=max_boundary_gap,
    )
    return core_mask.astype(np.uint8)


# ------------------------------------------------------------------
# Overlay 核心函式
# ------------------------------------------------------------------

def overlay_ihc_mask_on_dish(
    dish_image: np.ndarray,
    ihc_core_mask: np.ndarray,
    mask_blur_sigma: float = 0.0,
    background_fill_value: int = 0,
) -> np.ndarray:
    """將 IHC Her2+ 核心遮罩套用至 DISH 影像。

    非 ROI 像素填充為 ``background_fill_value``；
    ROI 像素保留 DISH 原始值。

    Args:
        dish_image: shape ``(H, W, 3)``、``uint8`` RGB 影像。
        ihc_core_mask: shape ``(H, W)``、``bool`` 或 ``{0, 1}`` 遮罩。
        mask_blur_sigma: 高斯模糊 σ，用於柔化遮罩邊緣。0 表示不模糊。
        background_fill_value: 非 ROI 區域的填充值 (0–255)。

    Returns:
        shape ``(H, W, 3)``、``uint8`` 的遮罩後 DISH 影像。

    Raises:
        ValueError: 空間維度不匹配時拋出。
    """
    _validate_overlay_inputs(dish_image, ihc_core_mask)

    mask_float = ihc_core_mask.astype(np.float32)
    if mask_blur_sigma > 0:
        mask_float = gaussian_filter(mask_float, sigma=mask_blur_sigma)
        mask_float = np.clip(mask_float, 0.0, 1.0)

    mask_3ch = mask_float[:, :, np.newaxis]

    background = np.full_like(dish_image, fill_value=background_fill_value)
    masked = (dish_image.astype(np.float32) * mask_3ch
              + background.astype(np.float32) * (1.0 - mask_3ch))

    return masked.astype(np.uint8)


def _validate_overlay_inputs(
    dish_image: np.ndarray,
    ihc_core_mask: np.ndarray,
) -> None:
    """檢查 overlay 輸入的 shape 與 dtype。

    Raises:
        ValueError: 維度或 shape 不符。
    """
    if dish_image.ndim != 3 or dish_image.shape[2] != 3:
        raise ValueError(
            f"dish_image 須為 (H, W, 3)，實際: {dish_image.shape}"
        )
    if ihc_core_mask.ndim != 2:
        raise ValueError(
            f"ihc_core_mask 須為 (H, W)，實際: {ihc_core_mask.shape}"
        )
    if dish_image.shape[:2] != ihc_core_mask.shape[:2]:
        raise ValueError(
            f"空間維度不匹配: dish={dish_image.shape[:2]}, "
            f"mask={ihc_core_mask.shape[:2]}"
        )


# ------------------------------------------------------------------
# 批次配對 Tile 載入
# ------------------------------------------------------------------

def find_paired_tiles(
    ihc_dir: Path,
    dish_dir: Path,
    extensions: Optional[list] = None,
) -> list:
    """搜尋座標相同的 IHC/DISH tile 配對。

    Args:
        ihc_dir: IHC tile 資料夾。
        dish_dir: DISH tile 資料夾。
        extensions: 允許的副檔名列表 (含 '.')。

    Returns:
        已排序的 ``[(ihc_path, dish_path), ...]`` 列表。
        座標不匹配的 tile 會以 warning 記錄並跳過。
    """
    if extensions is None:
        extensions = [".tiff", ".tif", ".png"]

    ihc_map = _build_coord_map(ihc_dir, extensions)
    dish_map = _build_coord_map(dish_dir, extensions)

    paired: list = []
    all_coords = set(ihc_map.keys()) | set(dish_map.keys())

    for coord in sorted(all_coords):
        if coord not in ihc_map:
            logger.warning("DISH tile 缺少對應 IHC: coord=%s", coord)
            continue
        if coord not in dish_map:
            logger.warning("IHC tile 缺少對應 DISH: coord=%s", coord)
            continue
        paired.append((ihc_map[coord], dish_map[coord]))

    logger.info("配對完成: %d 對 tile", len(paired))
    return paired


def _build_coord_map(
    tile_dir: Path,
    extensions: list,
) -> dict:
    """建立 {(x, y): Path} 座標索引。"""
    coord_map: dict = {}
    for ext in extensions:
        for filepath in tile_dir.glob(f"*{ext}"):
            try:
                coord = parse_tile_coords(filepath.stem)
                coord_map[coord] = filepath
            except ValueError:
                logger.debug("跳過無法解析座標的檔案: %s", filepath.name)
    return coord_map
