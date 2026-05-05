"""Full WSI end-to-end pipeline (sliding-window over whole slide image)。

思路：
  - UNet++ / Cellpose / dot detection 全部走 sliding window
  - 每個 window 重用 cell_mask.hybrid 的 M1→M4 模組
  - 每個 window 的結果 (CellAnalysisResult) 將 centroid 從 window-local
    座標轉為 WSI 全圖座標，再合併為 slide-level CSV / summary
  - 可選擇性輸出 stitched core_mask / instance_mask (BigTIFF)

架構：
    主迴圈同步跑 GPU forward + post 處理；I/O 由 ``torch.utils.data.DataLoader``
    多個 worker process 平行 prefetch（每個 worker 自己持有 openslide handle）。
    post 處理（dot detection / CSV / stitch）在主執行緒同步處理
    （CPU 工作 << GPU forward，不會卡）。

用法:
    cp config_example.py config.py
    # 編輯 config.py 路徑
    python full_wsi_pipeline.py

    # 自訂 override
    python full_wsi_pipeline.py --window 2048 --overlap 128
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader, Dataset

# ---- path wiring so we can reuse cell_mask.hybrid modules -----------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path("/data/tsgh")
_HYBRID_DIR = _PROJECT_ROOT / "cell_mask" / "hybrid"
for _p in (str(_PROJECT_ROOT), str(_HYBRID_DIR), str(_THIS_DIR)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from config import config  # noqa: E402
from m1_overlay import (  # noqa: E402
    apply_mask_to_ihc_image,
    fuse_masked_ihc_with_dish,
    overlay_ihc_mask_on_dish,
)
from m2_segmentation import CellposeSegmenter  # noqa: E402
from cell_mask.hybrid.m3_cells_generator import (  # noqa: E402
    CellAnalysisResult,
    build_all_positive_results,
)
from m3_dot_detection import detect_all_dots, merge_dot_results_to_cell_analysis  # noqa: E402
from m4_export import DotStatsSummary, write_summary_csv, render_overlay_image  # noqa: E402
from m5_tiffwriter import BigTiffWriter  # noqa: E402
from unet_inference import UNetPPInference, postprocess_membrane_mask  # noqa: E402

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("full_wsi")


# ---------------------------------------------------------------------------
# WSI reader (used in main process only, for dimension lookup)
# ---------------------------------------------------------------------------

class WSIReader:
    """Lazy tile reader for a WSI via openslide (主程序使用，DataLoader workers 自有 handle)."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"WSI not found: {path}")
        self.path = path
        self._slide = openslide.OpenSlide(str(path))
        self.width, self.height = self._slide.dimensions
        logger.info(
            "WSI loaded: %s shape=(%d,%d,3)",
            path.name, self.height, self.width,
        )

    def read(self, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
        y1c = min(y1, self.height)
        x1c = min(x1, self.width)
        region = self._slide.read_region((x0, y0), 0, (x1c - x0, y1c - y0))
        return np.array(region)[:, :, :3]

    def close(self) -> None:
        self._slide.close()


# ---------------------------------------------------------------------------
# DataLoader-backed window dataset
# ---------------------------------------------------------------------------

class _WSIWindowDataset(Dataset):
    """從兩張對齊 WSI 讀取單一 window 的 IHC + DISH patch。

    每個 DataLoader worker process 自行 lazy-init 一份 openslide handle
    （openslide handle 不可跨 process 共享），由 ``persistent_workers`` 持有
    整輪迭代以避免重複開啟。空白 window（IHC 平均亮度 > skip_white_threshold）
    回傳 ``is_blank=True`` 但仍占據 batch index，由 collate 後在主迴圈解析。
    """

    def __init__(
        self,
        windows: List[Tuple[int, int, int, int]],
        ihc_path: Path,
        dish_path: Path,
        skip_white_threshold: Optional[float],
    ) -> None:
        self.windows = windows
        self._ihc_path = str(ihc_path)
        self._dish_path = str(dish_path)
        self._skip_white = skip_white_threshold
        # openslide handles: lazy in worker (do not pickle)
        self._ihc: Optional[openslide.OpenSlide] = None
        self._dish: Optional[openslide.OpenSlide] = None

    def _ensure_open(self) -> None:
        if self._ihc is None:
            self._ihc = openslide.OpenSlide(self._ihc_path)
            self._dish = openslide.OpenSlide(self._dish_path)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._ensure_open()
        y0, x0, y1, x1 = self.windows[idx]
        ihc_region = self._ihc.read_region(
            (x0, y0), 0, (x1 - x0, y1 - y0),
        )
        ihc_patch = np.array(ihc_region)[:, :, :3]
        if (self._skip_white is not None
                and float(ihc_patch.mean()) > self._skip_white):
            return {
                "window": (y0, x0, y1, x1),
                "is_blank": True,
                "ihc": None,
                "dish": None,
            }
        dish_region = self._dish.read_region(
            (x0, y0), 0, (x1 - x0, y1 - y0),
        )
        dish_patch = np.array(dish_region)[:, :, :3]
        return {
            "window": (y0, x0, y1, x1),
            "is_blank": False,
            "ihc": ihc_patch,
            "dish": dish_patch,
        }


def _identity_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """保留 list-of-dict 結構；patch 大小可變不能用預設 collate。"""
    return batch


def _split_batch(
    batch: List[Dict[str, Any]],
) -> Tuple[
    List[Tuple[int, int, int, int]],
    List[np.ndarray],
    List[np.ndarray],
    int,
]:
    """將 DataLoader 一個 batch 拆成 kept_windows / IHC / DISH list 與空白計數。"""
    kept_windows: List[Tuple[int, int, int, int]] = []
    kept_ihc: List[np.ndarray] = []
    kept_dish: List[np.ndarray] = []
    n_blank = 0
    for item in batch:
        if item["is_blank"]:
            n_blank += 1
            continue
        kept_windows.append(item["window"])
        kept_ihc.append(item["ihc"])
        kept_dish.append(item["dish"])
    return kept_windows, kept_ihc, kept_dish, n_blank


# ---------------------------------------------------------------------------
# Window grid
# ---------------------------------------------------------------------------

def generate_windows(
    height: int,
    width: int,
    window_size: int,
    overlap: int,
) -> List[Tuple[int, int, int, int]]:
    """Produce (y0, x0, y1, x1) list. Edge windows are clipped, not padded."""
    if overlap < 0 or overlap >= window_size:
        raise ValueError(f"overlap {overlap} must satisfy 0 <= overlap < window_size")
    stride = window_size - overlap
    windows: List[Tuple[int, int, int, int]] = []
    y = 0
    while y < height:
        x = 0
        while x < width:
            y1 = min(y + window_size, height)
            x1 = min(x + window_size, width)
            windows.append((y, x, y1, x1))
            if x + window_size >= width:
                break
            x += stride
        if y + window_size >= height:
            break
        y += stride
    return windows


def compute_owned_box(
    window: Tuple[int, int, int, int],
    slide_h: int,
    slide_w: int,
    overlap: int,
) -> Tuple[int, int, int, int]:
    """這個 window 對 cell 擁有權的 WSI 座標框 ``[yt, yb) x [xt, xb)``。

    內部邊界各佔 ``overlap/2``、slide 邊界保留全部範圍。``overlap=0`` 時
    退化為 window 全範圍（此情況邊界細胞會在相鄰 window 各自看到部分而被
    重複計算，建議將 overlap 設為 >= 細胞直徑）。
    """
    y0, x0, y1, x1 = window
    half = overlap // 2
    other = overlap - half
    yt = y0 if y0 == 0 else y0 + half
    xt = x0 if x0 == 0 else x0 + half
    yb = y1 if y1 >= slide_h else y1 - other
    xb = x1 if x1 >= slide_w else x1 - other
    return yt, xt, yb, xb


def process_window_batch(
    ihc_patches: List[np.ndarray],
    dish_patches: List[np.ndarray],
    unet: UNetPPInference,
    cellpose: CellposeSegmenter,
    dish_cellpose: CellposeSegmenter,
    cellpose_batch_size: int,
) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
    """批次執行 M1→M2 + dish cellpose。每個輸出 tuple 對應同 index 的輸入。

    回傳 (core_mask, instance_mask, dish_overlay, dish_nuc_mask, ihc_dish_blend)。
    dish_overlay / ihc_dish_blend / dish_nuc_mask 在 core 為空時為 None。
    dot detection 不在此函式中執行，由 caller 在 GPU forward 後同步處理。
    """
    n = len(ihc_patches)
    if n == 0:
        return []

    raws = unet.predict_batch_arrays(ihc_patches)
    core_masks: List[np.ndarray] = [
        postprocess_membrane_mask(
            r, close_kernel_size=config.core_close_kernel,
        ).astype(np.uint8)
        for r in raws
    ]

    m2_inputs: List[np.ndarray] = []
    dish_overlays: List[np.ndarray] = []
    m2_indices: List[int] = []
    for i in range(n):
        if core_masks[i].sum() == 0:
            continue
        masked_ihc = apply_mask_to_ihc_image(
            ihc_patches[i], core_masks[i],
            mask_blur_sigma=config.mask_blur_sigma,
            background_fill_value=config.background_fill_value,
        )
        dish_overlay = overlay_ihc_mask_on_dish(
            dish_patches[i], core_masks[i],
            mask_blur_sigma=config.mask_blur_sigma,
            background_fill_value=config.background_fill_value,
        )
        m2_input = fuse_masked_ihc_with_dish(
            dish_overlay, masked_ihc,
            ihc_alpha=config.overlay_alpha,
        )
        m2_inputs.append(m2_input)
        dish_overlays.append(dish_overlay)
        m2_indices.append(i)

    if m2_inputs:
        m2_masks = cellpose.predict_batch(
            m2_inputs, batch_size=cellpose_batch_size,
        )
    else:
        m2_masks = []

    instance_masks: List[np.ndarray] = [
        np.zeros(core_masks[i].shape, dtype=np.int32) for i in range(n)
    ]
    for sub_i, full_i in enumerate(m2_indices):
        instance_masks[full_i] = m2_masks[sub_i].astype(np.int32)

    dish_overlay_by_full = {
        full_i: dish_overlays[pos] for pos, full_i in enumerate(m2_indices)
    }
    ihc_dish_by_full = {
        full_i: m2_inputs[pos] for pos, full_i in enumerate(m2_indices)
    }
    dish_indices: List[int] = [
        full_i for full_i in m2_indices if instance_masks[full_i].any()
    ]
    dish_inputs = [dish_patches[i] for i in dish_indices]
    if dish_inputs:
        dish_nuc_masks = dish_cellpose.predict_batch(
            dish_inputs, batch_size=cellpose_batch_size,
        )
    else:
        dish_nuc_masks = []
    dish_nuc_by_full = {
        full_i: dish_nuc_masks[pos] for pos, full_i in enumerate(dish_indices)
    }

    return [
        (
            core_masks[i],
            instance_masks[i],
            dish_overlay_by_full.get(i),
            dish_nuc_by_full.get(i),
            ihc_dish_by_full.get(i),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Slide-level CSV writers
# ---------------------------------------------------------------------------

_GLOBAL_CSV_HEADER = [
    "global_cell_id",
    "window_y0", "window_x0",
    "local_cell_id",
    "centroid_x_wsi", "centroid_y_wsi",
    "reddot", "blackdot", "ratio",
    "excluded",
]


def _format_count(val: int, excluded: bool) -> str:
    return "NaN" if excluded else str(int(val))


def _format_ratio(ratio: float, excluded: bool) -> str:
    if excluded:
        return "NaN"
    if ratio == float("inf") or ratio == 0.0 or math.isnan(ratio):
        return "NaN"
    return f"{ratio:.4f}"


# ---------------------------------------------------------------------------
# Dot detection per batch (parallelised via ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _run_dot_detection(
    kept_windows: List[Tuple[int, int, int, int]],
    batch_outputs: List[
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
    ],
    dots_executor: Optional[ThreadPoolExecutor],
) -> Tuple[List[List[CellAnalysisResult]], List[List]]:
    """為每個 window 跑 build_all_positive_results + detect_all_dots。

    回傳 ``(results_per_window, all_dots_per_window)``。
    ``all_dots_per_window[i]`` 是該 window 偵測到的 DetectedDot list（供 overlay 繪圖）。
    """
    n_win = len(kept_windows)
    results_per_window: List[List[CellAnalysisResult]] = [[] for _ in range(n_win)]
    all_dots_per_window: List[List] = [[] for _ in range(n_win)]
    dish_indices: List[int] = []

    for i, (y0, x0, _y1, _x1) in enumerate(kept_windows):
        _, instance_mask, dish_overlay, dish_nuc_mask, _ = batch_outputs[i]
        if not instance_mask.any():
            continue
        results = build_all_positive_results(instance_mask)
        if not results:
            continue
        if dish_overlay is None or dish_nuc_mask is None:
            raise RuntimeError(
                f"w_y{y0}_x{x0}: missing dish overlay/nucleus mask"
            )
        results_per_window[i] = results
        dish_indices.append(i)

    if not dish_indices:
        return results_per_window, all_dots_per_window

    if dots_executor is None or len(dish_indices) <= 1:
        for i in dish_indices:
            (_, instance_mask, dish_overlay, dish_nuc_mask, _) = batch_outputs[i]
            all_dots, per_cell_dots = detect_all_dots(
                dish_overlay, instance_mask, config,
                dish_nucleus_mask=dish_nuc_mask,
            )
            all_dots_per_window[i] = all_dots
            results_per_window[i] = merge_dot_results_to_cell_analysis(
                results_per_window[i], per_cell_dots,
            )
    else:
        futures: dict = {}
        for i in dish_indices:
            (_, instance_mask, dish_overlay, dish_nuc_mask, _) = batch_outputs[i]
            fut = dots_executor.submit(
                detect_all_dots, dish_overlay, instance_mask, config,
                dish_nucleus_mask=dish_nuc_mask,
            )
            futures[fut] = i
        for fut in as_completed(futures):
            idx = futures[fut]
            all_dots, per_cell_dots = fut.result()
            all_dots_per_window[idx] = all_dots
            results_per_window[idx] = merge_dot_results_to_cell_analysis(
                results_per_window[idx], per_cell_dots,
            )
    return results_per_window, all_dots_per_window


def _log_progress(idx: int, total: int, skipped: int) -> None:
    logger.info("[%d/%d] skipped=%d", idx, total, skipped)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    window_size: Optional[int] = None,
    overlap: Optional[int] = None,
    limit: Optional[int] = None,
) -> None:
    win = window_size or config.wsi_window_size
    ov = overlap if overlap is not None else config.wsi_window_overlap

    output_dir: Path = config.output_dir / config.slide_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("slide_id=%s", config.slide_id)
    logger.info("window=%d overlap=%d", win, ov)
    if ov == 0:
        logger.warning(
            "wsi_window_overlap=0：被切到 window 邊界的細胞會在兩個 window 各看到一部分、"
            "centroid 各落在自家 owned box 中而被重複計算。建議 overlap >= 細胞直徑。"
        )

    # ---- open WSIs (主程序：取得尺寸 + 後續無關 DataLoader 的需求) ----
    ihc_reader = WSIReader(config.ihc_wsi_path)
    dish_reader = WSIReader(config.dish_wsi_path)
    if (ihc_reader.height, ihc_reader.width) != (dish_reader.height, dish_reader.width):
        raise ValueError(
            f"IHC / DISH 尺寸不一致: IHC={ihc_reader.height}x{ihc_reader.width} "
            f"vs DISH={dish_reader.height}x{dish_reader.width}。請確認兩張 WSI 已對齊。"
        )
    H, W = ihc_reader.height, ihc_reader.width
    # 主程序的 reader 只用於尺寸檢查；DataLoader workers 各自開啟 handle
    ihc_reader.close()
    dish_reader.close()

    windows = generate_windows(H, W, win, ov)
    if limit:
        windows = windows[:limit]
    total = len(windows)
    logger.info("generated %d windows over %dx%d", total, H, W)

    # ---- stitched output files ----
    overlay_writer: Optional[BigTiffWriter] = None
    if config.save_stitched_overlay:
        overlay_writer = BigTiffWriter(
            output_dir / f"{config.slide_id}_overlay.tiff",
            H, W, np.uint8,
            bands=3,
            pyramidal=config.stitched_overlay_pyramidal,
            jpeg_quality=config.stitched_overlay_jpeg_quality,
            tile_size=config.stitched_overlay_tile_size,
        )

    # ---- init models once ----
    logger.info("initialising models…")
    unet = UNetPPInference(
        model_path=config.unet_model_path,
        encoder_name=config.unet_encoder_name,
        num_classes=config.unet_num_classes,
        image_size=config.unet_image_size,
        device=None,
    )
    if win > unet.image_size[0] or win > unet.image_size[1]:
        raise ValueError(
            f"wsi_window_size={win} exceeds unet_image_size={unet.image_size}; "
            "sliding-window inference disabled"
        )
    cellpose = CellposeSegmenter(
        model_path=config.cellpose_model_path,
        diameter=config.cellpose_diameter,
        flow_threshold=config.cellpose_flow_threshold,
        cellprob_threshold=config.cellpose_cellprob_threshold,
        gpu=config.cellpose_gpu,
    )
    dish_cellpose = CellposeSegmenter(
        model_path=config.cellpose_dish_model_path,
        diameter=config.cellpose_dish_diameter,
        flow_threshold=config.cellpose_dish_flow_threshold,
        cellprob_threshold=config.cellpose_dish_cellprob_threshold,
        gpu=config.cellpose_gpu,
    )

    # ---- Open slide-level CSV streams ----
    report_path = output_dir / f"{config.slide_id}_report.csv"
    report_fh = report_path.open("w", newline="", encoding="utf-8")
    report_writer = csv.writer(report_fh)
    report_writer.writerow(_GLOBAL_CSV_HEADER)

    # ---- Iterate windows in batches ----
    batch_size = max(1, int(config.wsi_batch_size))
    logger.info(
        "batched inference: wsi_batch_size=%d cellpose_batch_size=%d",
        batch_size, config.cellpose_batch_size,
    )

    # ---- Dot detection thread pool (shared across all batches) ----
    n_dot_workers = config.dots_workers or (os.cpu_count() or 4)
    if n_dot_workers > 1:
        dots_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=n_dot_workers, thread_name_prefix="dots",
        )
        logger.info("dot detection thread pool: max_workers=%d", n_dot_workers)
    else:
        dots_executor = None
        logger.info("dot detection: serial mode (dots_workers=1)")

    # ---- DataLoader for I/O prefetch ----
    dataset = _WSIWindowDataset(
        windows,
        config.ihc_wsi_path,
        config.dish_wsi_path,
        config.wsi_skip_white_threshold,
    )
    n_io_workers = max(0, int(getattr(config, "wsi_io_workers", 4)))
    prefetch_factor = int(getattr(config, "wsi_io_prefetch_factor", 2))
    loader_kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_io_workers,
        collate_fn=_identity_collate,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )
    if n_io_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    logger.info(
        "DataLoader: num_workers=%d prefetch_factor=%d batch_size=%d",
        n_io_workers, prefetch_factor if n_io_workers > 0 else 0, batch_size,
    )

    # ---- Main-loop running counters ----
    global_cell_counter = 0
    edge_dropped = 0
    skipped_blank = 0
    processed_so_far = 0
    summary_chunks: List[DotStatsSummary] = []

    try:
        for batch in loader:
            kept_windows, kept_ihc, kept_dish, n_blank = _split_batch(batch)

            # ---- 跳過空白 window 的進度更新 ----
            skipped_blank += n_blank
            for _ in range(n_blank):
                processed_so_far += 1
                if (processed_so_far % 50 == 0
                        or processed_so_far == total):
                    _log_progress(processed_so_far, total, skipped_blank)

            if not kept_windows:
                continue

            # ---- GPU forward (block 直到完成) ----
            batch_outputs = process_window_batch(
                kept_ihc, kept_dish, unet, cellpose, dish_cellpose,
                cellpose_batch_size=config.cellpose_batch_size,
            )

            # ---- Dot detection (CPU，可平行) ----
            results_per_window, all_dots_per_window = _run_dot_detection(
                kept_windows, batch_outputs, dots_executor,
            )

            # ---- Owned-box / CSV / stitch / summary ----
            for idx, (
                (y0, x0, y1, x1),
                dish_patch,
                (core_mask, instance_mask, _dish_overlay, _dish_nuc_mask, _),
            ) in enumerate(zip(kept_windows, kept_dish, batch_outputs)):
                tile_id = f"w_y{y0}_x{x0}"
                processed_so_far += 1

                results = results_per_window[idx]

                owned_yt, owned_xt, owned_yb, owned_xb = compute_owned_box(
                    (y0, x0, y1, x1), H, W, ov,
                )
                owned_results: List[CellAnalysisResult] = []
                owned_cell_ids: set = set()
                for r in results:
                    wsi_cx = r.centroid_x + x0
                    wsi_cy = r.centroid_y + y0
                    if ((owned_yt <= wsi_cy < owned_yb)
                            and (owned_xt <= wsi_cx < owned_xb)):
                        owned_results.append(r)
                        owned_cell_ids.add(int(r.cell_id))
                edge_dropped += len(results) - len(owned_results)

                if overlay_writer is not None:
                    base = _dish_overlay if _dish_overlay is not None else dish_patch
                    overlay_rgb = render_overlay_image(
                        base, instance_mask, results,
                        all_dots=all_dots_per_window[idx] or None,
                    )
                    overlay_writer.write(y0, x0, overlay_rgb)

                for r in owned_results:
                    global_cell_counter += 1
                    wsi_cx = r.centroid_x + x0
                    wsi_cy = r.centroid_y + y0
                    report_writer.writerow([
                        global_cell_counter,
                        y0, x0,
                        r.cell_id,
                        f"{wsi_cx:.2f}", f"{wsi_cy:.2f}",
                        _format_count(r.cep17_dot_count, r.excluded),
                        _format_count(r.her2_dot_count, r.excluded),
                        _format_ratio(r.her2_cep17_ratio, r.excluded),
                        int(r.excluded),
                    ])

                summary_chunks.append(DotStatsSummary.from_results(owned_results))

                if (processed_so_far % 10 == 0
                        or processed_so_far == total):
                    _log_progress(processed_so_far, total, skipped_blank)
    finally:
        # DataLoader workers 在迭代結束時自動回收；明確 del 可加速釋放
        del loader
        if dots_executor is not None:
            dots_executor.shutdown(wait=True)

    report_fh.close()

    # ---- slide-level summary CSV ----
    slide_summary = DotStatsSummary.aggregate(summary_chunks)
    summary_path = output_dir / f"{config.slide_id}_summary.csv"
    write_summary_csv(slide_summary, summary_path)

    # ---- close writers ----
    if overlay_writer is not None:
        overlay_writer.close()

    logger.info(
        "完成。處理 %d windows (跳過空白 %d)，共 %d cells"
        "（另有 %d 顆邊界細胞由相鄰 window 計入）。",
        total - skipped_blank, skipped_blank, global_cell_counter, edge_dropped,
    )
    logger.info("輸出: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full WSI sliding-window pipeline")
    p.add_argument("--window", type=int, default=None, help="window size (px)")
    p.add_argument("--overlap", type=int, default=None, help="overlap (px)")
    p.add_argument("--limit", type=int, default=None,
                   help="只處理前 N 個 window (smoke test)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(
        window_size=args.window,
        overlap=args.overlap,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
