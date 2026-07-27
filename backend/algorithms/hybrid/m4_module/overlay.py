"""全 tile overlay 視覺化與接縫虛線。"""

import math
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import center_of_mass, find_objects

try:
    from ..hybrid_data_types import CellAnalysisResult, CellDotResult, DetectedDot
except ImportError:
    from hybrid_data_types import CellAnalysisResult, CellDotResult, DetectedDot

# ------------------------------------------------------------------
# 顏色常數 (BGR for OpenCV)
# ------------------------------------------------------------------
_COLOR_BOUNDARY = (0, 255, 0)
_COLOR_HER2 = (0, 0, 0)
_COLOR_CEP17 = (0, 0, 220)
_COLOR_AMP = (0, 255, 255)
_COLOR_NON_AMP = (255, 200, 0)
_COLOR_EXCLUDED = (0, 0, 180)
_COLOR_DOT_CROSS = (255, 255, 255)
_COLOR_DISH_BBOX = (0, 165, 255)
_COLOR_DISH_MATCHED = (147, 20, 255)
_COLOR_DRIFT_ARROW = (139, 0, 0)
_COLOR_CELL_ID = (0, 255, 255)


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
    """為每個 DISH cellpose 細胞核 instance 沿實際形狀畫輪廓。"""
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
    """對每個「配對到核」的細胞畫飄移箭頭：IHC centroid → 認領的 DISH 核 centroid。

    只要細胞配到核（assigned_dish_ids 非空）就畫，含因紅點<2 被排除（low_cep17）
    者；drop_out / 出界核細胞本就 0 核、無 assigned_dish_ids，不受影響。
    """
    if dish_nucleus_mask is None or per_cell_dots is None:
        return

    matched_ids: set = set()
    for cdr in per_cell_dots.values():
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
        if cdr is None:
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
    """在每個細胞質心處繪製編號與分類標籤。"""
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
# tile 接縫虛線
# ------------------------------------------------------------------

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


# tile 核心裁切已由 render_overlay_image 轉回 RGB，故此接縫色為 RGB（藍），
# 與檔案上方 BGR 顏色常數不同色序。
_COLOR_TILE_SEAM = (0, 0, 255)


def draw_tile_seam_edges(
    crop: np.ndarray,
    *,
    right: bool,
    bottom: bool,
    color: Tuple[int, int, int] = _COLOR_TILE_SEAM,
    thickness: int = 2,
    dash: int = 22,
    gap: int = 14,
) -> None:
    """在 tile 核心裁切（RGB）的內部接縫邊就地畫藍色虛線。

    只畫右 / 下邊（接縫的左 / 上側 owner）：拼回 overlay_slide 後，這些邊恰落在相鄰
    tile 的接縫（``geometry.cuts_x`` / ``cuts_y``）上，成為 tile 邊界的視覺參考。
    ``right`` / ``bottom`` 由 caller 依 ``edge_flags`` 判定——碰觸真實 slide 邊的邊不畫；
    左 / 上邊交由相鄰 tile 的右 / 下邊負責，避免同一接縫重畫。
    """
    h, w = crop.shape[:2]
    if right:
        _dashed_segment(crop, (w - 1, 0), (w - 1, h), color, thickness, dash, gap)
    if bottom:
        _dashed_segment(crop, (0, h - 1), (w, h - 1), color, thickness, dash, gap)
