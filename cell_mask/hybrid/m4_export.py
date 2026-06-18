"""
M4: CSV 與視覺化匯出模組

匯出內容:
  1. ``tile_report.csv`` — 每列一個細胞的分類紀錄。
  2. ``overlay_vis.png`` — 全 tile 細胞邊界 + 分類標註疊加圖。
  3. ``cells/cell_{id}.png`` — 逐細胞固定尺寸裁切影像。
"""

import csv
import logging
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import center_of_mass, find_objects

from cell_mask.hybrid.m3_cells_generator import CellAnalysisResult
from cell_mask.hybrid.m3_dot_detection import CellDotResult, DetectedDot

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 顏色常數 (BGR for OpenCV)
# ------------------------------------------------------------------
_COLOR_BOUNDARY = (0, 255, 0)     # 綠色：細胞邊界
_COLOR_HER2 = (0, 0, 0)           # 黑色：HER2 黑點
_COLOR_CEP17 = (0, 0, 220)        # 紅色：CEP17 紅點
_COLOR_AMP = (0, 255, 255)        # 黃色：擴增細胞標籤
_COLOR_NON_AMP = (255, 200, 0)    # 淺藍：非擴增細胞標籤
_COLOR_EXCLUDED = (0, 0, 180)     # 深紅：排除（多核）標記
_COLOR_DOT_CROSS = (255, 255, 255)  # 白色：dot 中心十字
_COLOR_DISH_BBOX = (0, 165, 255)    # 橘色：未被認領的 DISH 核輪廓
_COLOR_DISH_MATCHED = (147, 20, 255)  # 深粉色：被 elastic matching 認領的 DISH 核
_COLOR_DRIFT_ARROW = (139, 0, 0)    # 深藍：飄移箭頭 (IHC → 認領 DISH)
_COLOR_CELL_ID = (0, 255, 255)      # 黃色：細胞 ID 編號
_COLOR_WINDOW_GRID = (255, 255, 0)  # 青色：1k sliding-window 接縫虛線格


# ------------------------------------------------------------------
# CSV 匯出
# ------------------------------------------------------------------

_CSV_HEADER = [
    "cell_id",
    "reddot",
    "blackdot",
    "ratio",
]


def _format_count(val: int, excluded: bool) -> str:
    return "NaN" if excluded else str(int(val))


def _format_ratio(ratio: float, excluded: bool) -> str:
    if excluded:
        return "NaN"
    if ratio == float("inf") or ratio == 0.0 or math.isnan(ratio):
        return "NaN"
    return f"{ratio:.4f}"


def export_tile_csv(
    results: List[CellAnalysisResult],
    output_path: Path,
    slide_id: str = "unknown",
    tile_id: str = "unknown",
    model_version: str = "v1.0.0",
    config_hash: str = "00000000",
) -> Path:
    """匯出單張 tile 的細胞分類結果至 CSV。

    Args:
        results: 細胞分類結果列表。
        output_path: CSV 儲存路徑。
        slide_id: 玻片識別碼。
        tile_id: Tile 識別碼。
        model_version: 模型版本字串。
        config_hash: 配置雜湊。

    Returns:
        實際寫入的 CSV 路徑。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        for cell in results:
            excluded = bool(getattr(cell, "excluded", False))
            ratio = getattr(cell, "her2_cep17_ratio", 0.0)
            red = getattr(cell, "cep17_dot_count", 0)
            black = getattr(cell, "her2_dot_count", 0)
            writer.writerow([
                cell.cell_id,
                _format_count(red, excluded),
                _format_count(black, excluded),
                _format_ratio(ratio, excluded),
            ])

    logger.info(
        "CSV 匯出完成: %s (%d 列)", output_path.name, len(results)
    )
    return output_path


# ------------------------------------------------------------------
# 全 tile overlay 視覺化
# ------------------------------------------------------------------

def render_overlay_image(
    overlay_image: np.ndarray,
    cell_instance_mask: np.ndarray,
    results: List[CellAnalysisResult],
    all_dots: Optional[List[DetectedDot]] = None,
    dish_nucleus_mask: Optional[np.ndarray] = None,
    per_cell_dots: Optional[Dict[int, CellDotResult]] = None,
) -> np.ndarray:
    """細胞邊界 + 標籤 + 紅/黑點渲染成 RGB numpy array，不寫檔。"""
    canvas = cv2.cvtColor(overlay_image.copy(), cv2.COLOR_RGB2BGR)
    _draw_overlay_layers(
        canvas, cell_instance_mask, results,
        all_dots=all_dots,
        dish_nucleus_mask=dish_nucleus_mask,
        per_cell_dots=per_cell_dots,
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def export_overlay_visualization(
    overlay_image: np.ndarray,
    cell_instance_mask: np.ndarray,
    results: List[CellAnalysisResult],
    output_path: Path,
    all_dots: Optional[List[DetectedDot]] = None,
    dish_nucleus_mask: Optional[np.ndarray] = None,
    per_cell_dots: Optional[Dict[int, CellDotResult]] = None,
) -> Path:
    """匯出含有細胞邊界、AMP/排除標註與點位的疊加圖。

    Args:
        overlay_image: shape ``(H, W, 3)`` RGB。
        cell_instance_mask: shape ``(H, W)`` 實例遮罩。
        results: 每個細胞的分析結果。
        output_path: 輸出 PNG 路徑。
        all_dots: 所有偵測到的 HER2/CEP17 點；提供時會畫到圖上。
        dish_nucleus_mask: shape ``(H, W)`` DISH 細胞核 instance mask
            （Cellpose DISH 輸出）；提供時會沿每個核形狀畫輪廓。
        per_cell_dots: ``{cell_id: CellDotResult}``；與 ``dish_nucleus_mask``
            同時提供時，會把被 elastic matching 認領的 DISH 核改畫深粉色，
            並從每顆 IHC 細胞中心畫深藍箭頭到認領的 DISH 核中心。

    Returns:
        實際寫入的路徑。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = cv2.cvtColor(overlay_image.copy(), cv2.COLOR_RGB2BGR)
    # canvas[cell_instance_mask == 0] = 255
    _draw_overlay_layers(
        canvas, cell_instance_mask, results,
        all_dots=all_dots,
        dish_nucleus_mask=dish_nucleus_mask,
        per_cell_dots=per_cell_dots,
    )

    cv2.imwrite(str(output_path), canvas)
    logger.info("Overlay 匯出完成: %s", output_path.name)
    return output_path


def _draw_overlay_layers(
    canvas: np.ndarray,
    cell_instance_mask: np.ndarray,
    results: List[CellAnalysisResult],
    all_dots: Optional[List[DetectedDot]],
    dish_nucleus_mask: Optional[np.ndarray],
    per_cell_dots: Optional[Dict[int, CellDotResult]],
) -> None:
    """統一繪圖順序：dish 輪廓 → 細胞邊界 → 飄移箭頭 → 標籤 → dots。"""
    matched_ids: Optional[set] = None
    if dish_nucleus_mask is not None and per_cell_dots is not None:
        matched_ids = set()
        for cdr in per_cell_dots.values():
            # 粉色覆蓋所有被 elastic matching 認領的 DISH 核（一對一下每顆細胞
            # 至多 1 核；drop-out 細胞無認領核，故不會被塗粉色）。
            matched_ids.update(
                int(d) for d in getattr(cdr, "assigned_dish_ids", [])
            )

    if dish_nucleus_mask is not None:
        _draw_dish_nucleus_contours(canvas, dish_nucleus_mask, matched_ids)
    _draw_cell_boundaries(canvas, cell_instance_mask)
    if dish_nucleus_mask is not None and per_cell_dots is not None:
        _draw_drift_arrows(canvas, results, per_cell_dots, dish_nucleus_mask)
    _draw_cell_labels(canvas, results)
    if all_dots:
        _draw_dots(canvas, all_dots)


def export_dot_only_visualization(
    image: np.ndarray,
    all_dots: List[DetectedDot],
    output_path: Path,
) -> Path:
    """匯出純粹的 tile 層級紅/黑點 QA 圖（不含細胞邊界）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    _draw_dots(canvas, all_dots)
    cv2.imwrite(str(output_path), canvas)
    logger.info("Dot 視覺化匯出: %s (%d 點)", output_path.name, len(all_dots))
    return output_path


def _draw_cell_boundaries(
    canvas: np.ndarray,
    cell_instance_mask: np.ndarray,
) -> None:
    """在 canvas 上繪製所有細胞的輪廓線。"""
    mask_i32 = cell_instance_mask.astype(np.int32, copy=False)
    slices = find_objects(mask_i32)
    for label_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        y_sl, x_sl = sl
        local = (mask_i32[y_sl, x_sl] == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        cv2.drawContours(
            canvas, contours, -1, _COLOR_BOUNDARY, 1, cv2.LINE_8,
            offset=(int(x_sl.start), int(y_sl.start)),
        )


def _draw_dish_nucleus_contours(
    canvas: np.ndarray,
    dish_nucleus_mask: np.ndarray,
    matched_ids: Optional[set] = None,
) -> None:
    """為每個 DISH cellpose 細胞核 instance 沿實際形狀畫輪廓。

    使用 ``cv2.findContours`` 在每個 instance 的局部 bbox 內取外輪廓，
    再以全圖座標 offset 還原後 drawContours。逐 instance 處理可避免
    相鄰核合併成單一輪廓。

    若提供 ``matched_ids``，該集合內的 DISH 核 ID 用深粉色（已被 elastic
    matching 認領），其餘核維持橘色。
    """
    if dish_nucleus_mask.size == 0 or int(dish_nucleus_mask.max()) <= 0:
        return
    matched = matched_ids if matched_ids is not None else set()
    mask_i32 = dish_nucleus_mask.astype(np.int32, copy=False)
    slices = find_objects(mask_i32)
    for label_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        y_sl, x_sl = sl
        local = (mask_i32[y_sl, x_sl] == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        offset = (int(x_sl.start), int(y_sl.start))
        color = _COLOR_DISH_MATCHED if label_id in matched else _COLOR_DISH_BBOX
        cv2.drawContours(
            canvas, contours, -1, color, 1, cv2.LINE_8,
            offset=offset,
        )


def _draw_drift_arrows(
    canvas: np.ndarray,
    results: List[CellAnalysisResult],
    per_cell_dots: Dict[int, "CellDotResult"],
    dish_nucleus_mask: np.ndarray,
) -> None:
    """對每個非排除細胞畫飄移箭頭：IHC centroid → 認領的 DISH 核 centroid。

    一對一配對下每顆細胞至多認領 1 核，畫一支箭頭指向該核。
    被排除的細胞（excluded=True，drop-out）跳過，只保留斜十字標記。
    """
    if dish_nucleus_mask is None or per_cell_dots is None:
        return

    matched_ids: set = set()
    for cdr in per_cell_dots.values():
        if getattr(cdr, "excluded", False):
            continue
        matched_ids.update(int(d) for d in getattr(cdr, "assigned_dish_ids", []))
    if not matched_ids:
        return

    dish_centroids: Dict[int, Tuple[float, float]] = {}
    matched_list = sorted(matched_ids)
    centers = center_of_mass(
        np.ones(dish_nucleus_mask.shape, dtype=np.uint8),
        labels=dish_nucleus_mask,
        index=matched_list,
    )
    for did, (cy, cx) in zip(matched_list, centers):
        if not (math.isnan(cy) or math.isnan(cx)):
            dish_centroids[int(did)] = (float(cy), float(cx))

    for cell in results:
        cdr = per_cell_dots.get(cell.cell_id)
        if cdr is None or getattr(cdr, "excluded", False):
            continue
        assigned = getattr(cdr, "assigned_dish_ids", [])
        if not assigned:
            continue
        ihc_pt = (int(cell.centroid_x), int(cell.centroid_y))
        for did in assigned:
            ctr = dish_centroids.get(int(did))
            if ctr is None:
                continue
            dish_pt = (int(ctr[1]), int(ctr[0]))
            cv2.arrowedLine(
                canvas, ihc_pt, dish_pt, _COLOR_DRIFT_ARROW,
                1, cv2.LINE_8, tipLength=0.18,
            )


def _draw_cell_labels(
    canvas: np.ndarray,
    results: List[CellAnalysisResult],
) -> None:
    """在每個細胞質心處繪製編號與分類標籤。

    - 一律以黃色畫出 ``#cell_id`` 編號（中心點上方）
    - 排除細胞（多核）：另外加深紅斜十字
    - 有 dot 計數者：``H/C`` 顯示於中心點下方，AMP 用黃色、非 AMP 用淺藍
    """
    for cell in results:
        cx = int(cell.centroid_x)
        cy = int(cell.centroid_y)
        position = (cx, cy)

        cv2.putText(
            canvas, f"#{cell.cell_id}", (cx - 6, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            _COLOR_CELL_ID, 1, cv2.LINE_AA,
        )

        if getattr(cell, "excluded", False):
            cv2.drawMarker(
                canvas, position, _COLOR_EXCLUDED,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=10, thickness=2, line_type=cv2.LINE_AA,
            )
            continue

        her2 = int(getattr(cell, "her2_dot_count", 0))
        cep17 = int(getattr(cell, "cep17_dot_count", 0))
        if her2 == 0 and cep17 == 0:
            continue
        color = _COLOR_AMP if getattr(cell, "is_amplified", False) else _COLOR_NON_AMP
        cv2.putText(
            canvas, f"{her2}/{cep17}", (cx - 6, cy + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            color, 1, cv2.LINE_AA,
        )


def _draw_dots(
    canvas: np.ndarray,
    dots: Iterable[DetectedDot],
) -> None:
    """在 canvas 上繪製每個 dot：彩色圓 + 白色中心十字。"""
    for d in dots:
        color = _COLOR_HER2 if d.dot_type == "her2" else _COLOR_CEP17
        r = max(3, int(round(d.radius + 1)))
        cy, cx = int(round(d.y)), int(round(d.x))
        cv2.circle(canvas, (cx, cy), r, color, 1, cv2.LINE_AA)
        cv2.drawMarker(
            canvas, (cx, cy), _COLOR_DOT_CROSS,
            markerType=cv2.MARKER_CROSS,
            markerSize=3, thickness=1, line_type=cv2.LINE_AA,
        )


# ------------------------------------------------------------------
# 1k sliding-window 接縫虛線格
# ------------------------------------------------------------------

def draw_dashed_grid(
    canvas: np.ndarray,
    tile_size: int,
    color: Tuple[int, int, int] = _COLOR_WINDOW_GRID,
    thickness: int = 2,
    dash: int = 22,
    gap: int = 14,
) -> None:
    """在 ``canvas`` 上「就地」畫出 ``tile_size`` 間距的內部接縫虛線格。

    只畫影像內部的接縫線（不畫最外緣），讓醫師一眼看出 sliding-window 切在哪、
    驗證跨接縫細胞是否已正確縫合。座標慣例與 ``canvas`` 一致（BGR / RGB 皆可，
    僅 ``color`` 通道順序需相符）。
    """
    h, w = canvas.shape[:2]
    for x in range(tile_size, w, tile_size):
        _dashed_segment(canvas, (x, 0), (x, h), color, thickness, dash, gap)
    for y in range(tile_size, h, tile_size):
        _dashed_segment(canvas, (0, y), (w, y), color, thickness, dash, gap)


def _dashed_segment(
    canvas: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    dash: int,
    gap: int,
) -> None:
    """沿 pt1→pt2 以 (dash, gap) 節奏畫虛線（僅支援水平 / 垂直線）。"""
    (x1, y1), (x2, y2) = pt1, pt2
    step = dash + gap
    if x1 == x2:  # 垂直
        for y in range(y1, y2, step):
            cv2.line(canvas, (x1, y), (x1, min(y + dash, y2)), color, thickness)
    else:         # 水平
        for x in range(x1, x2, step):
            cv2.line(canvas, (x, y1), (min(x + dash, x2), y1), color, thickness)


def stamp_grid_on_overlays(
    output_dir: Path,
    tile_size: int,
    pattern: str = "*_overlay.png",
) -> int:
    """為 ``output_dir`` 中所有符合 ``pattern`` 的影像就地補上虛線格。

    放在所有匯出之後執行：以「檔名」為準（預設所有 ``*_overlay.png``）統一蓋格線，
    保證不漏任何一張疊圖，且不必把 grid 參數穿進每個匯出函式。

    Returns:
        實際蓋上格線的檔案數。
    """
    count = 0
    for path in sorted(output_dir.glob(pattern)):
        image = cv2.imread(str(path))  # BGR
        if image is None:
            continue
        draw_dashed_grid(image, tile_size)
        cv2.imwrite(str(path), image)
        count += 1
    if count:
        logger.info("已在 %d 張 %s 上補虛線格 (tile=%d)", count, pattern, tile_size)
    return count


# ------------------------------------------------------------------
# 逐細胞固定尺寸裁切影像
# ------------------------------------------------------------------

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
    之後放入固定 ``crop_size x crop_size`` 白底畫布。

    註：m3 偵測 ROI 已改為「配對到的 DISH 核區域」，與此 crop 的粉色核形狀
    一致——只有核內的紅黑點才計數，crop 與 CSV count 範圍相同，無核外漏算。

    若提供 ``per_cell_dots``，會在每張 crop 上：
      - 畫出該細胞範圍內的 HER2 黑點 / CEP17 紅點
      - 在角落標註 ``H=x C=y [AMP]``
      - 排除細胞（多核）改畫斜十字 X

    Args:
        source_image: shape ``(H, W, 3)`` RGB。
        cell_instance_mask: shape ``(H, W)`` IHC 實例遮罩（fallback 用）。
        results: 細胞分析結果。
        output_dir: 輸出資料夾 (``cells/`` 子目錄)。
        crop_size: 裁切尺寸 (正方形邊長, pixels)。
        per_cell_dots: ``{cell_id: CellDotResult}``；提供則畫點與標註。
        dish_nucleus_mask: shape ``(H, W)`` DISH 細胞核 instance mask；
            提供時優先以匹配的 DISH 核 region 做 crop。

    Returns:
        儲存成功的檔案路徑列表。
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

        # 被配對的細胞：只在匹配 DISH 核 bbox 的 union 內建立局部 mask，
        # 避免每顆細胞都配置 4096x4096 全圖 boolean mask。
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

        # 未配對（無粉色）或粉色核遺失 → 退回 IHC 細胞實例形狀。
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
      ``canvas_y = (full_y - bbox_y0) * canvas_scale + canvas_offset_y``
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
    """依 instance mask 形狀擷取單細胞區域，細胞外填 255。

    Returns:
        ``(cell_patch, (y0, x0, y1, x1))``，其中 bbox 座標為全圖座標。
    """
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

    若 patch 任一邊超過 ``crop_size``，等比例縮小以完整保留細胞（不裁切），
    縮放係數一併回傳供點座標換算。

    Returns:
        ``(canvas, (offset_y, offset_x), scale)``；座標換算為
        ``canvas_coord = patch_coord * scale + offset``。
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


# ------------------------------------------------------------------
# 統一匯出入口
# ------------------------------------------------------------------

@dataclass
class DotStatsSummary:
    """有效雙色細胞統計摘要。

    可代表單一 tile 的統計，亦可由多個 tile 摘要合併為 slide-level。
    僅記錄 count；百分比於寫檔時計算，避免合併產生累積誤差。

    有效細胞定義：未排除（excluded=False）且 reddot >= 1 且 blackdot >= 1。
    """

    valid_cells: int = 0      # 雙色有效細胞總數
    ratio_lt2: int = 0        # her2/cep17 < 2
    ratio_gte2: int = 0       # her2/cep17 >= 2
    copy_lt4: int = 0         # blackdot < 4
    copy_4to5: int = 0        # 4 <= blackdot < 6
    copy_gte6: int = 0        # blackdot >= 6

    @classmethod
    def from_results(
        cls,
        results: List[CellAnalysisResult],
    ) -> "DotStatsSummary":
        valid = [
            r for r in results
            if not getattr(r, "excluded", False)
            and getattr(r, "cep17_dot_count", 0) >= 1
            and getattr(r, "her2_dot_count", 0) >= 1
        ]
        ratio_lt2 = sum(
            1 for r in valid
            if getattr(r, "her2_cep17_ratio", 0.0) < 2.0
        )
        copy_lt4 = sum(1 for r in valid if r.her2_dot_count < 4)
        copy_4to5 = sum(1 for r in valid if 4 <= r.her2_dot_count < 6)
        copy_gte6 = sum(1 for r in valid if r.her2_dot_count >= 6)
        return cls(
            valid_cells=len(valid),
            ratio_lt2=ratio_lt2,
            ratio_gte2=len(valid) - ratio_lt2,
            copy_lt4=copy_lt4,
            copy_4to5=copy_4to5,
            copy_gte6=copy_gte6,
        )

    def merge(self, other: "DotStatsSummary") -> "DotStatsSummary":
        """欄位逐項相加，回傳新 summary。"""
        return DotStatsSummary(**{
            f.name: getattr(self, f.name) + getattr(other, f.name)
            for f in fields(self)
        })

    @classmethod
    def aggregate(
        cls,
        summaries: Iterable["DotStatsSummary"],
    ) -> "DotStatsSummary":
        """多個 summary 合併為一（slide-level 接口）。"""
        total = cls()
        for s in summaries:
            total = total.merge(s)
        return total


def write_summary_csv(
    stats: DotStatsSummary,
    output_path: Path,
) -> Path:
    """Write DotStatsSummary to CSV; percentages are computed here."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = stats.valid_cells

    def pct(count: int) -> str:
        return f"{count / n * 100:.1f}%" if n > 0 else "N/A"

    rows = [
        ["Metric", "Count", "Percentage"],
        ["Valid tumor cells (both HER2 and CEP17 dots present)", n, "100%" if n > 0 else "N/A"],
        ["HER2/CEP17 ratio < 2", stats.ratio_lt2, pct(stats.ratio_lt2)],
        ["HER2/CEP17 ratio >= 2", stats.ratio_gte2, pct(stats.ratio_gte2)],
        ["HER2 copy number (black dots) < 4", stats.copy_lt4, pct(stats.copy_lt4)],
        ["HER2 copy number (black dots) 4-5", stats.copy_4to5, pct(stats.copy_4to5)],
        ["HER2 copy number (black dots) >= 6", stats.copy_gte6, pct(stats.copy_gte6)],
    ]

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)

    logger.info("Summary CSV exported: %s (%d valid cells)", output_path.name, n)
    return output_path


def export_summary_statistics(
    results: List[CellAnalysisResult],
    output_path: Path,
) -> Path:
    """Per-tile 便捷包裝：compute → write。"""
    stats = DotStatsSummary.from_results(results)
    return write_summary_csv(stats, output_path)


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
    """統一匯出 CSV + overlay PNG + per-cell PNG + 統計摘要。

    Args:
        overlay_image: IHC-DISH 疊合影像（per-cell 裁切的來源）。
        cell_instance_mask: 實例遮罩。
        results: 細胞分析結果列表。
        output_dir: 匯出根目錄。
        visualization_image: 視覺化底圖；若為 None 則使用 overlay_image。
        slide_id: 玻片識別碼。
        tile_id: Tile 識別碼。
        model_version: 模型版本。
        config_hash: 配置雜湊。
        crop_size: 單細胞裁切尺寸 (pixels)。
        all_dots: 所有偵測點；提供時 overlay PNG 會畫出點。
        per_cell_dots: ``{cell_id: CellDotResult}``；提供時 per-cell PNG 會畫點與 AMP 標籤。
        dish_nucleus_mask: shape ``(H, W)`` DISH 細胞核 instance mask；
            提供時主 overlay PNG 會沿每個核的形狀畫橘色輪廓。
    """
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
        output_dir / f"{tile_id}_summary.csv",
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
