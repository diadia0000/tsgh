"""獨立測試 m3_dot_detection 模組。

使用 test_picture/tile_x88064_y88064 中已 pre-computed 的：
    - tile_x88064_y88064_dish_mask_overlay.png  (M1 輸出)
    - tile_x88064_y88064_m2_cell_instance_mask.tiff  (M2 輸出)

直接執行 M3b 偵測流程，產出：
    test_dot_result/
      ├── tile_x88064_y88064_dish_dot_vis.png          (tile 層級 QA 圖)
      ├── tile_x88064_y88064_report.csv                (細胞計數表)
      ├── tile_x88064_y88064_dots_raw.csv              (所有偵測點明細)
      ├── tile_x88064_y88064_overlay.png               (細胞邊界 + 點位)
      └── cells/cell_{id}.png                          (逐細胞裁切 + 點位標記)
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage import io

_HYBRID_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _HYBRID_DIR.parent.parent
for _path in (str(_PROJECT_ROOT), str(_HYBRID_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import config  # noqa: E402
from cell_mask.hybrid.m3_cells_generator import (  # noqa: E402
    build_all_positive_results,
)
from cell_mask.hybrid.m4_export import (  # noqa: E402
    DotStatsSummary,
    _format_count,
    _format_ratio,
    write_summary_csv,
)
from m2_segmentation import CellposeSegmenter  # noqa: E402
from m3_dot_detection import (  # noqa: E402
    CellDotResult,
    DetectedDot,
    detect_all_dots,
    merge_dot_results_to_cell_analysis,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 常數
# ------------------------------------------------------------------
TILE_ID = "tile_x88064_y88064"
INPUT_DIR = _HYBRID_DIR / "test_picture" / TILE_ID
DISH_RAW_PATH = _HYBRID_DIR / "test_picture" / "dish" / f"{TILE_ID}.tiff"
OUTPUT_DIR = _HYBRID_DIR / "test_dot_result"

_COLOR_CELL_BOUNDARY = (0, 255, 0)      # 綠色
_COLOR_HER2 = (0, 0, 0)                 # 黑點 (BGR)
_COLOR_CEP17 = (0, 0, 220)              # 紅點 (BGR)
_COLOR_AMP = (0, 255, 255)              # 黃色 — 擴增細胞
_COLOR_NON_AMP = (255, 200, 0)          # 淺藍 — 非擴增
_COLOR_EXCLUDED = (0, 0, 180)           # 深紅 — 排除（多核）


# ------------------------------------------------------------------
# 輸入載入
# ------------------------------------------------------------------

def _load_inputs() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """讀取 dish_mask_overlay、instance_mask、raw DISH 圖。"""
    dish_path = INPUT_DIR / f"{TILE_ID}_dish_mask_overlay.png"
    mask_path = INPUT_DIR / f"{TILE_ID}_m2_cell_instance_mask.tiff"
    if not dish_path.exists():
        raise FileNotFoundError(f"找不到 DISH overlay: {dish_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"找不到 instance mask: {mask_path}")

    dish = io.imread(str(dish_path))
    if dish.ndim == 2:
        dish = np.stack([dish] * 3, axis=-1)
    elif dish.shape[2] == 4:
        dish = dish[:, :, :3]
    dish = dish.astype(np.uint8)

    mask = io.imread(str(mask_path))
    mask = mask.astype(np.int32)

    # Raw DISH 圖（供 DISH Cellpose 核偵測使用）
    if DISH_RAW_PATH.exists():
        dish_raw = io.imread(str(DISH_RAW_PATH))
        if dish_raw.ndim == 2:
            dish_raw = np.stack([dish_raw] * 3, axis=-1)
        elif dish_raw.shape[2] == 4:
            dish_raw = dish_raw[:, :, :3]
        dish_raw = dish_raw.astype(np.uint8)
        logger.info("raw DISH shape=%s", dish_raw.shape)
    else:
        logger.warning("找不到 raw DISH 圖 %s，退回使用 dish_mask_overlay", DISH_RAW_PATH)
        dish_raw = dish

    logger.info("dish_mask_overlay shape=%s dtype=%s", dish.shape, dish.dtype)
    logger.info("instance_mask shape=%s n_cells=%d", mask.shape, int(mask.max()))
    return dish, mask, dish_raw


# ------------------------------------------------------------------
# 輸出: CSV
# ------------------------------------------------------------------

def _export_report_csv(
    cell_results,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "cell_id",
        "reddot",
        "blackdot",
        "ratio",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for c in cell_results:
            excluded = bool(getattr(c, "excluded", False))
            writer.writerow([
                c.cell_id,
                _format_count(c.cep17_dot_count, excluded),
                _format_count(c.her2_dot_count, excluded),
                _format_ratio(c.her2_cep17_ratio, excluded),
            ])
    logger.info("CSV 匯出: %s (%d 列)", output_path.name, len(cell_results))


def _export_raw_dots_csv(
    all_dots: List[DetectedDot],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "dot_type", "cell_id", "y", "x", "radius",
        "area", "circularity", "solidity", "contrast", "score",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for d in all_dots:
            writer.writerow([
                d.dot_type, d.cell_id,
                f"{d.y:.2f}", f"{d.x:.2f}", f"{d.radius:.2f}",
                d.area,
                f"{d.circularity:.3f}",
                f"{d.solidity:.3f}",
                f"{d.contrast:.2f}",
                f"{d.score:.2f}",
            ])
    logger.info("Raw dots CSV 匯出: %s (%d 列)",
                output_path.name, len(all_dots))


# ------------------------------------------------------------------
# 輸出: Tile 層級視覺化
# ------------------------------------------------------------------

def _draw_dot(canvas_bgr: np.ndarray, d: DetectedDot,
              color: Tuple[int, int, int]) -> None:
    r = max(3, int(round(d.radius + 1)))
    cy, cx = int(round(d.y)), int(round(d.x))
    cv2.circle(canvas_bgr, (cx, cy), r, color, 1, cv2.LINE_AA)
    cv2.drawMarker(canvas_bgr, (cx, cy), (255, 255, 255),
                   markerType=cv2.MARKER_CROSS,
                   markerSize=3, thickness=1, line_type=cv2.LINE_AA)


def _export_tile_dot_visualization(
    dish_overlay: np.ndarray,
    all_dots: List[DetectedDot],
    output_path: Path,
) -> None:
    canvas = cv2.cvtColor(dish_overlay, cv2.COLOR_RGB2BGR).copy()
    for d in all_dots:
        color = _COLOR_HER2 if d.dot_type == "her2" else _COLOR_CEP17
        _draw_dot(canvas, d, color)
    cv2.imwrite(str(output_path), canvas)
    logger.info("Dot 視覺化: %s", output_path.name)


def _export_cell_overlay(
    dish_overlay: np.ndarray,
    instance_mask: np.ndarray,
    cell_results,
    all_dots: List[DetectedDot],
    output_path: Path,
) -> None:
    """細胞邊界 + 擴增標記 + 點位一起畫。"""
    canvas = cv2.cvtColor(dish_overlay, cv2.COLOR_RGB2BGR).copy()

    # 細胞邊界
    cids = set(np.unique(instance_mask)) - {0}
    for cid in cids:
        binary = (instance_mask == cid).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, _COLOR_CELL_BOUNDARY, 1)

    # 擴增 / 非擴增標記；excluded 改畫 X
    for c in cell_results:
        pos = (int(c.centroid_x), int(c.centroid_y))
        if getattr(c, "excluded", False):
            cv2.drawMarker(canvas, pos, _COLOR_EXCLUDED,
                           markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=10, thickness=2, line_type=cv2.LINE_AA)
            continue
        color = _COLOR_AMP if c.is_amplified else _COLOR_NON_AMP
        label = f"{c.her2_dot_count}/{c.cep17_dot_count}"
        cv2.putText(canvas, label, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    color, 1, cv2.LINE_AA)

    # 點位
    for d in all_dots:
        color = _COLOR_HER2 if d.dot_type == "her2" else _COLOR_CEP17
        _draw_dot(canvas, d, color)

    cv2.imwrite(str(output_path), canvas)
    logger.info("細胞-點位合成圖: %s", output_path.name)


# ------------------------------------------------------------------
# 輸出: 逐細胞裁切圖
# ------------------------------------------------------------------

def _extract_cell_patch(
    source: np.ndarray,
    region_mask: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """擷取細胞 bbox，背景填 255。回傳 (patch, (y0, x0, y1, x1))"""
    ys, xs = np.where(region_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    src_patch = source[y0:y1, x0:x1]
    mask_patch = region_mask[y0:y1, x0:x1]
    cell_patch = np.full(src_patch.shape, 255, dtype=source.dtype)
    cell_patch[mask_patch] = src_patch[mask_patch]
    return cell_patch, (y0, x0, y1, x1)


def _fit_to_canvas(
    patch: np.ndarray,
    crop_size: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """將 patch 置入固定尺寸白底畫布；回傳 (canvas, (dst_y0, dst_x0))。"""
    h, w = patch.shape[:2]
    canvas = np.full((crop_size, crop_size, 3), 255, dtype=patch.dtype)
    src_y0 = max((h - crop_size) // 2, 0)
    src_x0 = max((w - crop_size) // 2, 0)
    src_y1 = src_y0 + min(h, crop_size)
    src_x1 = src_x0 + min(w, crop_size)
    trimmed = patch[src_y0:src_y1, src_x0:src_x1]
    th, tw = trimmed.shape[:2]
    dst_y0 = (crop_size - th) // 2
    dst_x0 = (crop_size - tw) // 2
    canvas[dst_y0:dst_y0 + th, dst_x0:dst_x0 + tw] = trimmed
    return canvas, (dst_y0 - src_y0, dst_x0 - src_x0)


def _export_per_cell_crops(
    dish_overlay: np.ndarray,
    instance_mask: np.ndarray,
    cell_results,
    per_cell_dots: Dict[int, CellDotResult],
    output_dir: Path,
    crop_size: int = 128,
) -> None:
    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    for c in cell_results:
        region_mask = (instance_mask == c.cell_id)
        if not np.any(region_mask):
            continue

        patch, (y0, x0, _, _) = _extract_cell_patch(dish_overlay, region_mask)
        canvas, (canvas_offset_y, canvas_offset_x) = _fit_to_canvas(
            patch, crop_size=crop_size)
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        cdr = per_cell_dots.get(c.cell_id)
        dots: List[DetectedDot] = []
        if cdr is not None:
            dots = cdr.her2_dots + cdr.cep17_dots

        excluded = bool(getattr(c, "excluded", False))
        if not excluded:
            for d in dots:
                # 全 tile 座標 → 細胞 bbox 座標 → canvas 座標
                local_y = d.y - y0 + canvas_offset_y
                local_x = d.x - x0 + canvas_offset_x
                if not (0 <= local_y < crop_size and 0 <= local_x < crop_size):
                    continue
                color = _COLOR_HER2 if d.dot_type == "her2" else _COLOR_CEP17
                r = max(3, int(round(d.radius + 1)))
                cv2.circle(canvas_bgr, (int(local_x), int(local_y)), r,
                           color, 1, cv2.LINE_AA)
                cv2.drawMarker(canvas_bgr, (int(local_x), int(local_y)),
                               (255, 255, 255),
                               markerType=cv2.MARKER_CROSS,
                               markerSize=3, thickness=1)
        else:
            # 多核排除：畫大 X
            cx = crop_size // 2
            cy = crop_size // 2
            cv2.drawMarker(canvas_bgr, (cx, cy), _COLOR_EXCLUDED,
                           markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=int(crop_size * 0.7),
                           thickness=3, line_type=cv2.LINE_AA)

        # 角落浮水印：計數 / excluded / 擴增
        if excluded:
            n_blue = getattr(c, "blue_region_count", 0)
            tag = f"NaN (blue={n_blue})"
            tag_color = _COLOR_EXCLUDED
        else:
            tag = f"H={c.her2_dot_count} C={c.cep17_dot_count}"
            if c.is_amplified:
                tag += " [AMP]"
            tag_color = (0, 140, 255) if c.is_amplified else (100, 100, 100)
        cv2.putText(canvas_bgr, tag, (3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    tag_color, 1, cv2.LINE_AA)

        cv2.imwrite(str(cells_dir / f"cell_{c.cell_id}.png"), canvas_bgr)

    logger.info("Per-cell 裁切: %d 張輸出至 %s", len(cell_results), cells_dir)


# ------------------------------------------------------------------
# 摘要統計
# ------------------------------------------------------------------

def _summarize(cell_results, all_dots: List[DetectedDot]) -> None:
    n_cells = len(cell_results)
    her2_total = sum(1 for d in all_dots if d.dot_type == "her2")
    cep17_total = sum(1 for d in all_dots if d.dot_type == "cep17")
    excluded_cells = [c for c in cell_results if getattr(c, "excluded", False)]
    valid_cells = [c for c in cell_results if not getattr(c, "excluded", False)]
    amp_cells = [c for c in valid_cells if c.is_amplified]
    non_empty = [c for c in valid_cells
                 if c.her2_dot_count > 0 or c.cep17_dot_count > 0]
    if non_empty:
        avg_h = np.mean([c.her2_dot_count for c in non_empty])
        avg_c = np.mean([c.cep17_dot_count for c in non_empty])
    else:
        avg_h = avg_c = 0.0

    logger.info("---------- 摘要 ----------")
    logger.info("細胞總數: %d", n_cells)
    logger.info("排除細胞數 (多核): %d", len(excluded_cells))
    logger.info("HER2 黑點總數: %d", her2_total)
    logger.info("CEP17 紅點總數: %d", cep17_total)
    logger.info("有點位的細胞數 (未排除): %d (%.1f%%)",
                len(non_empty), 100 * len(non_empty) / max(n_cells, 1))
    logger.info("平均 HER2 / 紅點細胞: %.2f", avg_h)
    logger.info("平均 CEP17 / 紅點細胞: %.2f", avg_c)
    logger.info("擴增細胞數: %d", len(amp_cells))
    logger.info("--------------------------")


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dish, mask, dish_raw = _load_inputs()

    # DISH Cellpose 細胞核偵測（多核排除用）
    dish_segmenter = CellposeSegmenter(
        model_path=config.cellpose_dish_model_path,
        diameter=config.cellpose_dish_diameter,
        flow_threshold=config.cellpose_dish_flow_threshold,
        cellprob_threshold=config.cellpose_dish_cellprob_threshold,
        gpu=config.cellpose_gpu,
    )
    t_nuc = time.perf_counter()
    dish_nucleus_mask = dish_segmenter.predict(dish_raw)
    logger.info(
        "DISH 核偵測完成: %d 個核, 耗時 %.3f 秒",
        int(dish_nucleus_mask.max()), time.perf_counter() - t_nuc,
    )

    t0 = time.perf_counter()
    all_dots, per_cell_dots = detect_all_dots(dish, mask, config, dish_nucleus_mask=dish_nucleus_mask)
    detect_secs = time.perf_counter() - t0
    logger.info("detect_all_dots 耗時: %.3f 秒", detect_secs)

    cell_results = build_all_positive_results(mask)
    cell_results = merge_dot_results_to_cell_analysis(
        cell_results, per_cell_dots)

    _summarize(cell_results, all_dots)

    _export_report_csv(cell_results, OUTPUT_DIR / f"{TILE_ID}_report.csv")
    _export_raw_dots_csv(all_dots, OUTPUT_DIR / f"{TILE_ID}_dots_raw.csv")
    _export_tile_dot_visualization(
        dish, all_dots,
        OUTPUT_DIR / f"{TILE_ID}_dish_dot_vis.png",
    )
    _export_cell_overlay(
        dish, mask, cell_results, all_dots,
        OUTPUT_DIR / f"{TILE_ID}_overlay.png",
    )
    _export_per_cell_crops(
        dish, mask, cell_results, per_cell_dots,
        OUTPUT_DIR, crop_size=128,
    )

    # 匯出分類統計摘要
    stats = DotStatsSummary.from_results(cell_results)
    write_summary_csv(stats, OUTPUT_DIR / f"{TILE_ID}_summary.csv")
    logger.info(
        "分類摘要 — 有效雙色細胞: %d | ratio<2: %d (%.1f%%) | ratio>=2: %d (%.1f%%) "
        "| copy<4: %d | copy[4,6): %d | copy>=6: %d",
        stats.valid_cells,
        stats.ratio_lt2, 100 * stats.ratio_lt2 / max(stats.valid_cells, 1),
        stats.ratio_gte2, 100 * stats.ratio_gte2 / max(stats.valid_cells, 1),
        stats.copy_lt4, stats.copy_4to5, stats.copy_gte6,
    )

    logger.info("全部輸出已寫入: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
