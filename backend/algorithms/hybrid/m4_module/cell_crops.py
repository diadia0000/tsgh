"""逐細胞裁切匯出與統一匯出入口。"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import find_objects

try:
    from ..hybrid_data_types import CellAnalysisResult, CellDotResult, DetectedDot
except ImportError:
    from hybrid_data_types import CellAnalysisResult, CellDotResult, DetectedDot
from .csv import export_summary_statistics, export_tile_csv
from .overlay import (
    _COLOR_CEP17,
    _COLOR_DOT_CROSS,
    _COLOR_EXCLUDED,
    _COLOR_HER2,
    export_overlay_visualization,
)

logger = logging.getLogger(__name__)


def export_per_cell_images(
    source_image: np.ndarray,
    cell_instance_mask: np.ndarray,
    results: List[CellAnalysisResult],
    output_dir: Path,
    crop_size: int = 64,
    per_cell_dots: Optional[Dict[int, CellDotResult]] = None,
    dish_nucleus_mask: Optional[np.ndarray] = None,
) -> List[Path]:
    """輸出每個細胞的固定尺寸影像。

    被配對的細胞：region_mask = 匹配到的 DISH 核（粉色輪廓形狀），crop
    緊貼粉色邊緣，不與 IHC 本體聯集（drift 會讓聯集 bbox 暴增成「亂切」）。
    未配對的細胞：退回 IHC cell_instance_mask 形狀。細胞外背景填 255，
    之後放入固定 crop_size x crop_size 白底畫布。
    """
    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    cell_slices = find_objects(cell_instance_mask.astype(np.int32, copy=False))
    dish_slices = (
        find_objects(dish_nucleus_mask.astype(np.int32, copy=False))
        if dish_nucleus_mask is not None
        else []
    )

    for cell in results:
        cdr = per_cell_dots.get(cell.cell_id) if per_cell_dots is not None else None
        assigned_ids = (
            cdr.assigned_dish_ids
            if (cdr is not None and dish_nucleus_mask is not None)
            else []
        )

        region_local: Optional[np.ndarray] = None
        bbox: Optional[Tuple[int, int, int, int]] = None

        valid_dish_slices = []
        for did in assigned_ids:
            idx = int(did) - 1
            if 0 <= idx < len(dish_slices) and dish_slices[idx] is not None:
                valid_dish_slices.append((int(did), dish_slices[idx]))
        if valid_dish_slices and dish_nucleus_mask is not None:
            y0 = min(int(sl[0].start) for _, sl in valid_dish_slices)
            x0 = min(int(sl[1].start) for _, sl in valid_dish_slices)
            y1 = max(int(sl[0].stop) for _, sl in valid_dish_slices)
            x1 = max(int(sl[1].stop) for _, sl in valid_dish_slices)
            region_local = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            for did, sl in valid_dish_slices:
                yy0 = int(sl[0].start) - y0
                yy1 = int(sl[0].stop) - y0
                xx0 = int(sl[1].start) - x0
                xx1 = int(sl[1].stop) - x0
                region_local[yy0:yy1, xx0:xx1] |= (
                    dish_nucleus_mask[sl] == did
                )
            bbox = (y0, x0, y1, x1)

        if region_local is None or not region_local.any():
            idx = int(cell.cell_id) - 1
            if idx < 0 or idx >= len(cell_slices) or cell_slices[idx] is None:
                continue
            sl = cell_slices[idx]
            y0, x0 = int(sl[0].start), int(sl[1].start)
            y1, x1 = int(sl[0].stop), int(sl[1].stop)
            region_local = (cell_instance_mask[sl] == cell.cell_id)
            bbox = (y0, x0, y1, x1)

        if bbox is None or region_local is None or not region_local.any():
            continue

        y0, x0, y1, x1 = bbox
        cell_crop, (local_y0, local_x0, _, _) = _extract_mask_shaped_cell(
            source_image[y0:y1, x0:x1],
            region_local,
        )
        full_y0 = y0 + local_y0
        full_x0 = x0 + local_x0
        fixed_crop, (offset_y, offset_x), scale = _fit_to_fixed_canvas(
            cell_crop,
            crop_size=crop_size,
            fill_value=255,
        )

        crop_bgr = cv2.cvtColor(fixed_crop, cv2.COLOR_RGB2BGR)

        if per_cell_dots is not None:
            _annotate_cell_crop(
                crop_bgr=crop_bgr,
                cell=cell,
                cell_dot_result=per_cell_dots.get(cell.cell_id),
                bbox_y0=full_y0,
                bbox_x0=full_x0,
                canvas_offset_y=offset_y,
                canvas_offset_x=offset_x,
                canvas_scale=scale,
                crop_size=crop_size,
            )

        cell_path = cells_dir / f"cell_{cell.cell_id}.png"
        cv2.imwrite(str(cell_path), crop_bgr)
        saved_paths.append(cell_path)

    logger.info("匯出 %d 張 per-cell 影像至 %s", len(saved_paths), cells_dir)
    return saved_paths


def _annotate_cell_crop(
    crop_bgr: np.ndarray,
    cell: CellAnalysisResult,
    cell_dot_result: Optional[CellDotResult],
    bbox_y0: int,
    bbox_x0: int,
    canvas_offset_y: int,
    canvas_offset_x: int,
    canvas_scale: float,
    crop_size: int,
) -> None:
    """在單張細胞 crop 上畫 dot / AMP 標籤 / excluded X。

    座標換算: 全 tile 座標 → 細胞 bbox 座標 → (縮放) → 固定畫布座標
      canvas_y = (full_y - bbox_y0) * canvas_scale + canvas_offset_y
    """
    excluded = bool(getattr(cell, "excluded", False))

    if not excluded and cell_dot_result is not None:
        dots = cell_dot_result.her2_dots + cell_dot_result.cep17_dots
        for d in dots:
            local_y = (d.y - bbox_y0) * canvas_scale + canvas_offset_y
            local_x = (d.x - bbox_x0) * canvas_scale + canvas_offset_x
            if not (0 <= local_y < crop_size and 0 <= local_x < crop_size):
                continue
            color = _COLOR_HER2 if d.dot_type == "her2" else _COLOR_CEP17
            r = max(3, int(round(d.radius * canvas_scale + 1)))
            cv2.circle(
                crop_bgr, (int(local_x), int(local_y)), r,
                color, 1, cv2.LINE_AA,
            )
            cv2.drawMarker(
                crop_bgr, (int(local_x), int(local_y)),
                _COLOR_DOT_CROSS,
                markerType=cv2.MARKER_CROSS,
                markerSize=3, thickness=1,
            )
    elif excluded:
        center = crop_size // 2
        cv2.drawMarker(
            crop_bgr, (center, center), _COLOR_EXCLUDED,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=int(crop_size * 0.7),
            thickness=3, line_type=cv2.LINE_AA,
        )

    if excluded:
        n_blue = getattr(cell, "blue_region_count", 0)
        tag = f"NaN (blue={n_blue})"
        tag_color = _COLOR_EXCLUDED
    else:
        her2 = getattr(cell, "her2_dot_count", 0)
        cep17 = getattr(cell, "cep17_dot_count", 0)
        tag = f"H={her2} C={cep17}"
        if getattr(cell, "is_amplified", False):
            tag += " [AMP]"
            tag_color = (0, 140, 255)
        else:
            tag_color = (100, 100, 100)
    cv2.putText(
        crop_bgr, tag, (3, 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
        tag_color, 1, cv2.LINE_AA,
    )


def _extract_mask_shaped_cell(
    image: np.ndarray,
    region_mask: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """依 instance mask 形狀擷取單細胞區域，細胞外填 255。"""
    ys, xs = np.where(region_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    src_patch = image[y0:y1, x0:x1]
    mask_patch = region_mask[y0:y1, x0:x1]

    cell_patch = np.full(src_patch.shape, 255, dtype=image.dtype)
    cell_patch[mask_patch] = src_patch[mask_patch]
    return cell_patch, (y0, x0, y1, x1)


def _fit_to_fixed_canvas(
    patch: np.ndarray,
    crop_size: int,
    fill_value: int = 255,
) -> Tuple[np.ndarray, Tuple[int, int], float]:
    """將任意尺寸 patch 放入固定尺寸白底畫布。

    若 patch 任一邊超過 crop_size，等比例縮小以完整保留細胞（不裁切），
    縮放係數一併回傳供點座標換算。
    """
    h, w = patch.shape[:2]
    canvas = np.full((crop_size, crop_size, 3), fill_value, dtype=patch.dtype)

    scale = min(1.0, crop_size / max(h, w))
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    dst_y0 = (crop_size - h) // 2
    dst_x0 = (crop_size - w) // 2
    canvas[dst_y0:dst_y0 + h, dst_x0:dst_x0 + w] = patch
    return canvas, (dst_y0, dst_x0), scale


def export_cell_dot_annotations(
    overlay_image: np.ndarray,
    cell_instance_mask: np.ndarray,
    results: List[CellAnalysisResult],
    output_dir: Path,
    visualization_image: np.ndarray = None,
    slide_id: str = "unknown",
    tile_id: str = "unknown",
    model_version: str = "v1.0.0",
    config_hash: str = "00000000",
    crop_size: int = 64,
    all_dots: Optional[List[DetectedDot]] = None,
    per_cell_dots: Optional[Dict[int, CellDotResult]] = None,
    dish_nucleus_mask: Optional[np.ndarray] = None,
) -> None:
    """統一匯出 CSV + overlay PNG + per-cell PNG + 統計摘要。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    export_tile_csv(
        results,
        output_dir / f"{tile_id}_report.csv",
        slide_id=slide_id,
        tile_id=tile_id,
        model_version=model_version,
        config_hash=config_hash,
    )

    export_summary_statistics(
        results,
        output_dir / f"{tile_id}_summary.txt",
    )

    vis_image = (
        overlay_image if visualization_image is None
        else visualization_image
    )

    export_overlay_visualization(
        vis_image,
        cell_instance_mask,
        results,
        output_dir / f"{tile_id}_overlay.png",
        all_dots=all_dots,
        dish_nucleus_mask=dish_nucleus_mask,
        per_cell_dots=per_cell_dots,
    )

    export_per_cell_images(
        overlay_image,
        cell_instance_mask,
        results,
        output_dir,
        crop_size=crop_size,
        per_cell_dots=per_cell_dots,
        dish_nucleus_mask=dish_nucleus_mask,
    )
