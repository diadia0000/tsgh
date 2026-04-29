"""Full WSI end-to-end pipeline (sliding-window over whole slide image)。

思路：
  - UNet++ / Cellpose / dot detection 全部走 sliding window
  - 每個 window 重用 cell_mask.hybrid 的 M1→M4 模組
  - 每個 window 的結果 (CellAnalysisResult) 將 centroid 從 window-local
    座標轉為 WSI 全圖座標，再合併為 slide-level CSV / summary
  - 可選擇性輸出 stitched core_mask / instance_mask (BigTIFF)

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
import queue
import sys
import threading
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import openslide

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
from m4_export import DotStatsSummary, write_summary_csv  # noqa: E402
from m5_tiffwriter import BigTiffWriter  # noqa: E402
from unet_inference import UNetPPInference, postprocess_membrane_mask  # noqa: E402

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("full_wsi")


# ---------------------------------------------------------------------------
# WSI reader
# ---------------------------------------------------------------------------

class WSIReader:
    """Lazy tile reader for a WSI via openslide."""

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
) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    """批次執行 M1→M2 + dish cellpose。每個輸出 tuple 對應同 index 的輸入。

    回傳 (core_mask, instance_mask, dish_overlay, dish_nuc_mask)。dot detection
    改由 post thread 執行，讓 GPU pipeline 不中斷。
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
        )
        for i in range(n)
    ]


class _PipelineState:
    """Stream pipeline 共用可變狀態。所有寫入都在 post thread；
    main GPU thread 與 io thread 不寫，無需鎖。"""

    def __init__(self) -> None:
        self.global_cell_counter: int = 0
        self.instance_offset: int = 0
        self.edge_dropped: int = 0
        self.skipped_blank: int = 0
        self.processed_so_far: int = 0
        self.slide_summary: Optional["DotStatsSummary"] = None


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

    # ---- open WSIs ----
    ihc_reader = WSIReader(config.ihc_wsi_path)
    dish_reader = WSIReader(config.dish_wsi_path)
    if (ihc_reader.height, ihc_reader.width) != (dish_reader.height, dish_reader.width):
        raise ValueError(
            f"IHC / DISH 尺寸不一致: IHC={ihc_reader.height}x{ihc_reader.width} "
            f"vs DISH={dish_reader.height}x{dish_reader.width}。請確認兩張 WSI 已對齊。"
        )
    H, W = ihc_reader.height, ihc_reader.width
    windows = generate_windows(H, W, win, ov)
    if limit:
        windows = windows[:limit]
    total = len(windows)
    logger.info("generated %d windows over %dx%d", total, H, W)

    # ---- stitched output files ----
    core_writer: Optional[BigTiffWriter] = None
    inst_writer: Optional[BigTiffWriter] = None
    if config.save_stitched_core_mask:
        core_writer = BigTiffWriter(
            output_dir / f"{config.slide_id}_core_mask.tiff",
            H, W, np.uint8,
            pyramidal=config.stitched_core_pyramidal,
            jpeg_quality=config.stitched_core_jpeg_quality,
            tile_size=config.stitched_core_tile_size,
        )
    if config.save_stitched_instance_mask:
        inst_writer = BigTiffWriter(
            output_dir / f"{config.slide_id}_instance_mask.tiff",
            H, W, np.uint32,
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
    # detect_all_dots 在 scipy/skimage 釋放 GIL 期間可由 threads 平行；
    # pool 一次建立、整輪 run() 重用，避免每個 batch 都重建 thread。
    n_dot_workers = config.dots_workers or (os.cpu_count() or 4)
    if n_dot_workers > 1:
        dots_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=n_dot_workers,
            thread_name_prefix="dots",
        )
        logger.info("dot detection thread pool: max_workers=%d", n_dot_workers)
    else:
        dots_executor = None
        logger.info("dot detection: serial mode (dots_workers=1)")

    state = _PipelineState()  # 共用可變狀態（只由 post thread 寫入）
    state.slide_summary = DotStatsSummary()
    _thread_errors: List[BaseException] = []

    # ---- Stream pipeline: I/O thread → GPU main → Post thread ----
    # 每個 batch 約 350 MB (IHC+DISH+masks)；queue 容量 = pipeline_queue_size。
    qsize = max(1, int(config.pipeline_queue_size))
    io_q: "queue.Queue" = queue.Queue(maxsize=qsize)
    post_q: "queue.Queue" = queue.Queue(maxsize=qsize)
    SENTINEL = object()
    logger.info(
        "stream pipeline: queue_size=%d (I/O thread → GPU main → Post thread)",
        qsize,
    )

    def io_worker() -> None:
        try:
            for batch_start in range(0, total, batch_size):
                batch_windows = windows[batch_start: batch_start + batch_size]
                kept_windows: List[Tuple[int, int, int, int]] = []
                kept_ihc: List[np.ndarray] = []
                kept_dish: List[np.ndarray] = []
                n_blank = 0
                for (y0, x0, y1, x1) in batch_windows:
                    ihc_patch = ihc_reader.read(y0, x0, y1, x1)
                    if (config.wsi_skip_white_threshold is not None
                            and float(ihc_patch.mean())
                            > config.wsi_skip_white_threshold):
                        n_blank += 1
                        continue
                    dish_patch = dish_reader.read(y0, x0, y1, x1)
                    kept_windows.append((y0, x0, y1, x1))
                    kept_ihc.append(ihc_patch)
                    kept_dish.append(dish_patch)
                io_q.put((kept_windows, kept_ihc, kept_dish, n_blank))
        except Exception as exc:
            _thread_errors.append(exc)
            raise
        finally:
            io_q.put(SENTINEL)

    def post_worker() -> None:
        """Owned-box / CSV / stitch / summary。所有共用狀態只在這裡寫。"""
        try:
            while True:
                item = post_q.get()
                if item is SENTINEL:
                    break
                kept_windows, kept_ihc, kept_dish, batch_outputs, n_blank = item
                state.skipped_blank += n_blank
                for _ in range(n_blank):
                    state.processed_so_far += 1
                    if (state.processed_so_far % 50 == 0
                            or state.processed_so_far == total):
                        _log_progress(
                            state.processed_so_far, total, state.skipped_blank,
                        )

                results_per_window: List[List[CellAnalysisResult]] = [
                    [] for _ in range(len(kept_windows))
                ]
                dish_indices: List[int] = []

                for i, (y0, x0, _y1, _x1) in enumerate(kept_windows):
                    _, instance_mask, dish_overlay, dish_nuc_mask = (
                        batch_outputs[i]
                    )
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

                if dish_indices:
                    if dots_executor is None or len(dish_indices) <= 1:
                        for i in dish_indices:
                            (_, instance_mask, dish_overlay,
                             dish_nuc_mask) = batch_outputs[i]
                            _, per_cell_dots = detect_all_dots(
                                dish_overlay,
                                instance_mask,
                                config,
                                dish_nucleus_mask=dish_nuc_mask,
                            )
                            results_per_window[i] = (
                                merge_dot_results_to_cell_analysis(
                                    results_per_window[i],
                                    per_cell_dots,
                                )
                            )
                    else:
                        futures: dict = {}
                        for i in dish_indices:
                            (_, instance_mask, dish_overlay,
                             dish_nuc_mask) = batch_outputs[i]
                            fut = dots_executor.submit(
                                detect_all_dots,
                                dish_overlay,
                                instance_mask,
                                config,
                                dish_nucleus_mask=dish_nuc_mask,
                            )
                            futures[fut] = i

                        for fut in as_completed(futures):
                            idx = futures[fut]
                            _, per_cell_dots = fut.result()
                            results_per_window[idx] = (
                                merge_dot_results_to_cell_analysis(
                                    results_per_window[idx],
                                    per_cell_dots,
                                )
                            )
                for idx, ((y0, x0, y1, x1), _ihc_patch, dish_patch, (
                    core_mask, instance_mask, _dish_overlay, _dish_nuc_mask,
                )) in enumerate(zip(
                    kept_windows, kept_ihc, kept_dish, batch_outputs,
                )):
                    tile_id = f"w_y{y0}_x{x0}"
                    state.processed_so_far += 1

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
                    state.edge_dropped += len(results) - len(owned_results)

                    if core_writer is not None:
                        core_writer.write(
                            y0, x0, (core_mask * 255).astype(np.uint8),
                        )
                    if (inst_writer is not None
                            and instance_mask.any()
                            and owned_cell_ids):
                        max_label = int(instance_mask.max())
                        owned_ids = np.fromiter(
                            (
                                cid for cid in owned_cell_ids
                                if 0 < cid <= max_label
                            ),
                            dtype=np.int32,
                        )
                        if owned_ids.size > 0:
                            keep_lut = np.zeros(max_label + 1, dtype=bool)
                            keep_lut[owned_ids] = True
                            keep = keep_lut[instance_mask]
                            filtered = np.where(
                                keep, instance_mask, 0,
                            ).astype(np.uint32)
                            if filtered.any():
                                filtered[filtered > 0] += state.instance_offset
                                inst_writer.write(y0, x0, filtered)

                    for r in owned_results:
                        state.global_cell_counter += 1
                        wsi_cx = r.centroid_x + x0
                        wsi_cy = r.centroid_y + y0
                        report_writer.writerow([
                            state.global_cell_counter,
                            y0, x0,
                            r.cell_id,
                            f"{wsi_cx:.2f}", f"{wsi_cy:.2f}",
                            _format_count(r.cep17_dot_count, r.excluded),
                            _format_count(r.her2_dot_count, r.excluded),
                            _format_ratio(r.her2_cep17_ratio, r.excluded),
                            int(r.excluded),
                        ])

                    state.slide_summary = state.slide_summary.merge(
                        DotStatsSummary.from_results(owned_results),
                    )
                    if instance_mask.any():
                        state.instance_offset += int(instance_mask.max())

                    if config.save_per_window_artifacts and owned_results:
                        win_dir = output_dir / "windows" / tile_id
                        win_dir.mkdir(parents=True, exist_ok=True)
                        from m4_export import export_overlay_visualization  # noqa: WPS433
                        export_overlay_visualization(
                            dish_patch, instance_mask, owned_results,
                            win_dir / f"{tile_id}_overlay.png",
                        )

                    if (state.processed_so_far % 10 == 0
                            or state.processed_so_far == total):
                        _log_progress(
                            state.processed_so_far, total, state.skipped_blank,
                        )
        except Exception as exc:
            _thread_errors.append(exc)
            raise

    io_thread = threading.Thread(target=io_worker, name="io_worker", daemon=True)
    post_thread = threading.Thread(target=post_worker, name="post_worker", daemon=True)
    io_thread.start()
    post_thread.start()

    # ---- Main GPU loop: 只負責餵 GPU ----
    try:
        while True:
            item = io_q.get()
            if item is SENTINEL:
                break
            kept_windows, kept_ihc, kept_dish, n_blank = item
            if not kept_windows:
                post_q.put(([], [], [], [], n_blank))
                continue
            batch_outputs = process_window_batch(
                kept_ihc, kept_dish, unet, cellpose, dish_cellpose,
                cellpose_batch_size=config.cellpose_batch_size,
            )
            post_q.put(
                (kept_windows, kept_ihc, kept_dish, batch_outputs, n_blank),
            )
    finally:
        post_q.put(SENTINEL)
        io_thread.join()
        post_thread.join()
        if _thread_errors:
            raise _thread_errors[0]

    global_cell_counter = state.global_cell_counter
    edge_dropped = state.edge_dropped
    skipped_blank = state.skipped_blank
    slide_summary = state.slide_summary

    report_fh.close()

    if dots_executor is not None:
        dots_executor.shutdown(wait=True)

    # ---- slide-level summary CSV ----
    summary_path = output_dir / f"{config.slide_id}_summary.csv"
    write_summary_csv(slide_summary, summary_path)

    # ---- close writers ----
    if core_writer is not None:
        core_writer.close()
    if inst_writer is not None:
        inst_writer.close()
    ihc_reader.close()
    dish_reader.close()

    logger.info(
        "完成。處理 %d windows (跳過空白 %d)，共 %d cells"
        "（另有 %d 顆邊界細胞由相鄰 window 計入）。",
        total - skipped_blank, skipped_blank, global_cell_counter, edge_dropped,
    )
    logger.info("輸出: %s", output_dir)


def _log_progress(idx: int, total: int, skipped: int) -> None:
    logger.info("[%d/%d] skipped=%d", idx, total, skipped)


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
