"""
M1: IHC-DISH 遮罩疊加模組

將 IHC Her2+ 核心遮罩 (core mask) 套用至 DISH 影像，
保留 ROI 區域的原始像素，背景填充為指定值。

依賴:
  - UNet++ 膜分割推論器 (unet_inference.UNetPPInference)
  - 膜預測後處理 (unet_inference.postprocess_membrane_mask)
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple, Union

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
    ihc_image: Union[np.ndarray, Path, str],
    unet_inferencer: object,
    close_kernel: int = 7,
) -> np.ndarray:
    """透過 UNet++ 推論 + 輕量後處理生成 IHC Her2+ 核心遮罩。

    模型已直接輸出 filled blob（整塊棕色細胞膜區域），
    後處理僅連接微小斷裂。

    當影像大於模型訓練尺寸時，自動以滑動視窗推論。

    Args:
        ihc_image: IHC tile 影像本身（RGB ``uint8 (H,W,3)`` ndarray）或其路徑——
            ``UNetPPInference.predict_single`` 兩者皆吃。管線路徑（``hybrid_pipeline``）
            一律傳「已讀好的 ndarray」，因為同一張圖 M1/M2 都要用，重讀一次是白花的 I/O。
        unet_inferencer: 已初始化的 ``UNetPPInference`` 物件。
        close_kernel: 形態學閉合核大小 (pixels)，連接微小斷裂。

    Returns:
        shape ``(H, W)``、值域 ``{0, 1}`` 的 ``uint8`` 核心遮罩。
    """
    try:
        from .unet_inference import postprocess_membrane_mask  # noqa: WPS433
    except ImportError:
        from unet_inference import postprocess_membrane_mask  # noqa: WPS433

    raw_mask: np.ndarray = unet_inferencer.predict_single(ihc_image)
    core_mask = postprocess_membrane_mask(
        raw_mask,
        close_kernel_size=close_kernel,
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


def apply_mask_to_ihc_image(
    ihc_image: np.ndarray,
    ihc_core_mask: np.ndarray,
    mask_blur_sigma: float = 0.0,
    background_fill_value: int = 0,
) -> np.ndarray:
    """將 IHC core mask 套用至 IHC 影像。

    非 ROI 像素填充為 ``background_fill_value``；
    ROI 像素保留 IHC 原始值。
    """
    _validate_overlay_inputs(ihc_image, ihc_core_mask)

    mask_float = ihc_core_mask.astype(np.float32)
    if mask_blur_sigma > 0:
        mask_float = gaussian_filter(mask_float, sigma=mask_blur_sigma)
        mask_float = np.clip(mask_float, 0.0, 1.0)

    mask_3ch = mask_float[:, :, np.newaxis]
    background = np.full_like(ihc_image, fill_value=background_fill_value)
    masked = (ihc_image.astype(np.float32) * mask_3ch
              + background.astype(np.float32) * (1.0 - mask_3ch))
    return masked.astype(np.uint8)


def fuse_masked_ihc_with_dish(
    dish_mask_overlay_image: np.ndarray,
    masked_ihc_image: np.ndarray,
    ihc_alpha: float = 0.5,
) -> np.ndarray:
    """將「DISH mask overlay」與「masked IHC」融合成 IHC-DISH 疊合圖。

    融合公式:
        dish_mask_overlay × (1 - alpha) + masked_ihc × alpha
    預設 alpha=0.5，即兩者各取 50%。

    Args:
        dish_mask_overlay_image: shape ``(H, W, 3)``、``uint8``，
            已套用 ``ihc_core_mask`` 的 DISH 影像。
        masked_ihc_image: shape ``(H, W, 3)``、``uint8``，經 core_mask 遮罩的 IHC。
        ihc_alpha: IHC 融合權重 (0.0–1.0)。預設 0.5 (各 50%)。

    Returns:
        shape ``(H, W, 3)``、``uint8`` 的 IHC-DISH 疊合圖。
    """
    if dish_mask_overlay_image.ndim != 3 or dish_mask_overlay_image.shape[2] != 3:
        raise ValueError(
            "dish_mask_overlay_image 須為 (H, W, 3)，實際: "
            f"{dish_mask_overlay_image.shape}"
        )
    if masked_ihc_image.ndim != 3 or masked_ihc_image.shape[2] != 3:
        raise ValueError(
            f"masked_ihc_image 須為 (H, W, 3)，實際: {masked_ihc_image.shape}"
        )
    if dish_mask_overlay_image.shape[:2] != masked_ihc_image.shape[:2]:
        raise ValueError(
            "空間維度不匹配: "
            f"dish={dish_mask_overlay_image.shape[:2]}, ihc={masked_ihc_image.shape[:2]}"
        )
    if ihc_alpha < 0.0 or ihc_alpha > 1.0:
        raise ValueError(f"ihc_alpha 需介於 0.0~1.0，實際: {ihc_alpha}")

    dish_float = dish_mask_overlay_image.astype(np.float32)
    ihc_float = masked_ihc_image.astype(np.float32)
    fused = dish_float * (1.0 - ihc_alpha) + ihc_float * ihc_alpha
    return np.clip(fused, 0, 255).astype(np.uint8)


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
    """搜尋 IHC/DISH tile 配對。

    Args:
        ihc_dir: IHC tile 資料夾。
        dish_dir: DISH tile 資料夾。
        extensions: 允許的副檔名列表 (含 '.')。

    Returns:
        已排序的 ``[(ihc_path, dish_path), ...]`` 列表。
        按檔名排序後依序配對。
    """
    if extensions is None:
        extensions = [".tiff", ".tif", ".png"]

    ihc_files = _collect_tile_files(ihc_dir, extensions)
    dish_files = _collect_tile_files(dish_dir, extensions)

    # 按檔名排序後依序配對
    paired: list = []
    min_count = min(len(ihc_files), len(dish_files))
    for i in range(min_count):
        paired.append((ihc_files[i], dish_files[i]))

    if len(ihc_files) != len(dish_files):
        logger.warning(
            "IHC (%d) 與 DISH (%d) 檔案數量不一致，僅配對 %d 對",
            len(ihc_files), len(dish_files), min_count
        )

    logger.info("配對完成: %d 對 tile", len(paired))
    return paired


def _collect_tile_files(
    tile_dir: Path,
    extensions: list,
) -> list:
    """收集資料夾中符合副檔名的檔案，按檔名排序。"""
    files: list = []
    for ext in extensions:
        files.extend(tile_dir.glob(f"*{ext}"))
    return sorted(files, key=lambda p: p.name)
