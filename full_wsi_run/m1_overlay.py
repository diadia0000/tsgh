"""
M1: IHC-DISH 遮罩疊加模組

將 IHC Her2+ 核心遮罩 (core mask) 套用至 DISH 影像，
保留 ROI 區域的原始像素，背景填充為指定值。
"""

import numpy as np
from scipy.ndimage import gaussian_filter


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
