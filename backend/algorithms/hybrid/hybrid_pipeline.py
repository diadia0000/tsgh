"""
IHC-DISH Overlay & Analysis Pipeline — 主入口

串接 M1 (Overlay) → M2 (Segmentation) → M3 (Cell Results) → M4 (Export)。
支援單 tile 處理與批次掃描。

流程:
  - M1: IHC → UNet++ mask → mask on IHC & DISH → 50/50 alpha blend
  - M2: Cellpose 分割 IHC-DISH 疊合影像 → cell instance mask
  - M3: 將 M2 cell mask 套用至 dish_mask_overlay → 逐細胞結果
  - M4: CSV + overlay 視覺化

落地產物只有三樣：``report.csv`` / ``summary.txt`` / ``overlay_slide.tiff``。
逐塊的遮罩、masked IHC、per-cell 裁切等中繼陣列一律留在記憶體，不寫硬碟。

Usage:
    # 單一 ROI/WSI 影像對：先預切成重疊 tile 檔（暫存於 output/_precut_scratch），
    # 再走逐塊分析 + slide 級縫合流程。
    python hybrid_pipeline.py --ihc roi_ihc.tiff --dish roi_dish.tiff

    # 使用內建 test_picture ROI 範例（走完整 precut+分析流程）
    python hybrid_pipeline.py --test
"""

import argparse
import gc
import logging
import os
import pickle
import resource
import shutil
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyvips
from torch import sparse,cuda
from skimage import io

# Cellpose 在 get_masks_torch 內呼叫 torch.sparse_coo_tensor；PyTorch 要求顯式表態
# sparse invariant 檢查的開關，否則會持續發出 UserWarning。維持預設 (關閉檢查) 以保留效能。
sparse.check_sparse_tensor_invariants.disable()

# 將專案根目錄與 hybrid 目錄加入 sys.path，確保直接執行腳本時可解析套件匯入
_HYBRID_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _HYBRID_DIR.parent.parent
for _path in (str(_PROJECT_ROOT), str(_HYBRID_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from .config import config, compute_config_hash
    from .m1_overlay import (
        apply_mask_to_ihc_image,
        find_paired_tiles,
        fuse_masked_ihc_with_dish,
        generate_ihc_core_mask,
        overlay_ihc_mask_on_dish,
        parse_tile_coords,
    )
    from .m2_segmentation import (
        CellposeSegmenter,
        segment_windowed,
    )
    from .m3_cell_detection import (
        CellAnalysisResult,
        build_all_positive_results,
        detect_all_dots,
        enlarge_cell_instances,
        merge_dot_results_to_cell_analysis,
    )
    from .m4_export import (
        draw_tile_seam_edges,
        export_summary_statistics,
        export_tile_csv,
        render_overlay_image,
    )
    from .m0_stitch import (
        ChunkResult,
        TileGeometry,
        clear_slide_edge_cells,
        compute_tile_geometry,
        core_crop_bounds,
        filter_and_absolutize,
    )
    from .m0_reader import PrecutStream
except ImportError:
    from config import config, compute_config_hash
    from m1_overlay import (
        apply_mask_to_ihc_image,
        find_paired_tiles,
        fuse_masked_ihc_with_dish,
        generate_ihc_core_mask,
        overlay_ihc_mask_on_dish,
        parse_tile_coords,
    )
    from m2_segmentation import (
        CellposeSegmenter,
        segment_windowed,
    )
    from m3_cell_detection import (
        CellAnalysisResult,
        build_all_positive_results,
        detect_all_dots,
        enlarge_cell_instances,
        merge_dot_results_to_cell_analysis,
    )
    from m4_export import (
        draw_tile_seam_edges,
        export_summary_statistics,
        export_tile_csv,
        render_overlay_image,
    )
    from m0_stitch import (
        ChunkResult,
        TileGeometry,
        clear_slide_edge_cells,
        compute_tile_geometry,
        core_crop_bounds,
        filter_and_absolutize,
    )
    from m0_reader import PrecutStream

# ------------------------------------------------------------------
# Logging 設定
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 逐塊標註 overlay 的暫存夾。這不是產物而是縫合用的**串流緩衝**：pyvips 以
# ``access="sequential"`` 惰性讀這些檔再 ``tiffsave``，整片才不必同時在記憶體裡
# （純記憶體拼接就是當初被砍掉的全片 canvas 失效模式）。``_stitch_overlay_slide``
# 寫完 overlay_slide.tiff 後整夾刪除。
_STITCH_SCRATCH = "_stitch_scratch"


@dataclass
class M1StageArtifacts:
    """M1 輸出影像與遮罩，供後續 M2/M3/M4 共用。"""

    core_mask: np.ndarray
    dish_mask_overlay: np.ndarray
    m2_input_overlay: np.ndarray


@dataclass
class _ChunkGpuState:
    """單塊『三個 GPU 前向已跑完、CPU 後段（detect_all_dots）尚未跑』的中繼狀態。

    這是把單塊分析從中間切開的交接資料：M1 UNet / M2 Cellpose / M3b DISH Cellpose
    三個前向都已在主執行緒 / 單一 CUDA context 跑完，剩下的 build_all_positive_results /
    enlarge_cell_instances / detect_all_dots + merge 是純 CPU（joblib threads），可丟到
    背景執行緒與『下一塊的 GPU 前向』重疊。座標為分塊局部。
    """

    abs_x: int
    abs_y: int
    core_mask: np.ndarray                    # detect_all_dots 用來濾掉核心區外的 DISH 核
    dish_mask_overlay: np.ndarray
    instance_mask: np.ndarray
    dish_nucleus_mask: np.ndarray            # segment_windowed 產出，尚未經 detect_all_dots 過濾


@dataclass
class _TileGpuResult:
    """單一 precut tile 的 GPU 前段結果，交給背景執行緒跑 CPU 後段 + 落地寫檔。"""

    tile_name: str
    abs_x: int
    abs_y: int
    crop: Tuple[int, int, int, int]          # (lx0, lx1, ly0, ly1) 核心裁切界
    start_time: float
    chunk: Optional[_ChunkGpuState]          # None = 背景塊（核心遮罩全空）


# ------------------------------------------------------------------
# 模型初始化
# ------------------------------------------------------------------

def _init_unet_inferencer():
    """延遲初始化 UNet++ 推論器。"""
    try:
        from .unet_inference import UNetPPInference  # noqa: WPS433
    except ImportError:
        from unet_inference import UNetPPInference  # noqa: WPS433

    return UNetPPInference(
        model_path=config.unet_model_path,
        encoder_name=config.unet_encoder_name,
        num_classes=config.unet_num_classes,
        image_size=config.unet_image_size,
        batch_size=config.batch_size,
        device=None,
    )


def _init_cellpose_segmenter() -> CellposeSegmenter:
    """延遲初始化 Cellpose 分割器（M2 IHC-DISH 細胞分割）。"""
    return CellposeSegmenter(
        model_path=config.cellpose_model_path,
        diameter=config.cellpose_diameter,
        flow_threshold=config.cellpose_flow_threshold,
        cellprob_threshold=config.cellpose_cellprob_threshold,
        batch_size=config.cellpose_batch_size,
        gpu=config.cellpose_gpu,
    )


def _init_dish_cellpose_segmenter() -> CellposeSegmenter:
    """延遲初始化 DISH 細胞核偵測 Cellpose 分割器（M3b 多核排除用）。"""
    return CellposeSegmenter(
        model_path=config.cellpose_dish_model_path,
        diameter=config.cellpose_dish_diameter,
        flow_threshold=config.cellpose_dish_flow_threshold,
        cellprob_threshold=config.cellpose_dish_cellprob_threshold,
        batch_size=config.cellpose_batch_size,
        gpu=config.cellpose_gpu,
    )


@contextmanager
def _frozen_gc_generation():
    """把當下已存在的受追蹤物件移進 GC 永久世代，離開時還原。

    三個 GPU 模型（UNet++ / 兩個 Cellpose）在整批期間都活著且可達，generational GC
    每次 full collect 都要重新掃描這整張物件圖一遍。``gc.freeze()`` 把它們移進
    「收集器不再掃描」的永久世代，**不改變 collect 的呼叫頻率**，只砍掉每次的掃描量。

    實測（docs/hybrid-pipeline/16-gc-collect-frequency-result.md，441 tile，各 n=3）：
    每次 ``gc.collect()`` 83.2 ms → 1.2 ms，整批 36.71 s → 0.52 s；扣掉 GPU 前段
    （B1，與本改動無關但有 ±3% 抖動）後的主執行緒殘量 109.7 s → 72.7 s，即
    **−37.0 s（±1.5 s）**，端到端 571.5 s → 530.4 s（約 1.07x）。因為每塊仍照常
    collect，記憶體有界性不受影響（peak RSS 3.88 → 3.93 GB，鋸齒形狀維持）。

    離開時**務必** ``unfreeze``：``run_batch`` 會在長駐的 API server 行程中被反覆
    呼叫（``backend/api/hybrid.py``），只 freeze 不 unfreeze 會讓每次呼叫都把當下
    所有受追蹤物件永久凍結、之後再也不回收 → 跨請求的無界記憶體成長，正是本專案
    memory-bounded 不變量要擋的失效模式。此處無其他呼叫端會 freeze，故 unfreeze
    （會解凍全部）不會誤傷別人凍結的物件。
    """
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()


# ------------------------------------------------------------------
# 單 tile 處理
# ------------------------------------------------------------------

def _process_precut_tile_gpu(
    ihc_tile_path: Path,
    dish_tile_path: Path,
    abs_x: int,
    abs_y: int,
    geometry: TileGeometry,
    unet_inferencer: object,
    cellpose_segmenter: CellposeSegmenter,
    dish_cellpose_segmenter: CellposeSegmenter,
    output_dir: Path,
) -> Optional[_TileGpuResult]:
    """單塊的 **GPU 前段**：讀檔 → M1→M2→M3b 三個前向，回傳交接狀態。

    只做需 GPU / 需序列於單一 CUDA context 的工作；``detect_all_dots`` + 落地寫檔等純 CPU
    後段留給 ``_process_precut_tile_cpu`` 在背景執行緒跑。回傳：
      - ``None``：讀檔 / 維度等真實錯誤（呼叫端據此 fail-fast）。
      - ``chunk is None`` 的 ``_TileGpuResult``：背景塊（核心遮罩全空）。
      - 完整 ``_TileGpuResult``：待跑 CPU 後段。
    """
    tile_name = f"tile_x{abs_x}_y{abs_y}"
    start_time = time.perf_counter()

    try:
        ihc = _read_rgb(ihc_tile_path)
        dish = _read_rgb(dish_tile_path)
    except Exception as exc:
        logger.error("Tile %s 讀取失敗: %s", tile_name, exc)
        return None

    if ihc.shape[:2] != dish.shape[:2]:
        logger.error(
            "Tile %s IHC/DISH 尺寸不一致: %s vs %s",
            tile_name, ihc.shape[:2], dish.shape[:2],
        )
        return None

    lx0, lx1, ly0, ly1 = core_crop_bounds(
        geometry, abs_x, abs_y, config.default_tile_size
    )

    try:
        chunk = _process_one_chunk_gpu(
            ihc, dish, abs_x, abs_y,
            geometry.edge_flags(abs_x, abs_y),
            unet_inferencer, cellpose_segmenter, dish_cellpose_segmenter,
        )
    except ValueError as exc:
        logger.error("Tile %s 維度錯誤: %s", tile_name, exc)
        return None
    except Exception as exc:
        logger.error("Tile %s 處理失敗: %s", tile_name, exc, exc_info=True)
        return None

    return _TileGpuResult(
        tile_name=tile_name,
        abs_x=abs_x,
        abs_y=abs_y,
        crop=(lx0, lx1, ly0, ly1),
        start_time=start_time,
        chunk=chunk,
    )


def _process_precut_tile_cpu(
    tg: _TileGpuResult,
    geometry: TileGeometry,
    output_dir: Path,
    merge_dir: Optional[Path] = None,
) -> List[CellAnalysisResult]:
    """單塊的 **CPU 後段**：detect_all_dots + merge → 核心去重 → 寫縫合用 overlay。

    完全不碰 torch / CUDA，設計為可在背景執行緒執行，與主執行緒『下一塊的 GPU 前向』重疊。
    回傳核心區擁有、質心已絕對化（``cell_id`` 仍為分塊局部）的細胞清單。
    """
    tile_name = tg.tile_name
    lx0, lx1, ly0, ly1 = tg.crop

    # tile 內部接縫（拼回 overlay_slide 後的相鄰 tile 邊界）畫藍色虛線參考：只在
    # 「非真實 slide 邊」的右 / 下邊畫，碰真實 slide 邊者不畫（該邊不是接縫）。
    # config.draw_window_grid 關閉時 → (False, False) → 不畫。
    _ct, clear_bottom, _cl, clear_right = geometry.edge_flags(tg.abs_x, tg.abs_y)
    seam_edges = (
        (not clear_right, not clear_bottom)
        if config.draw_window_grid else (False, False)
    )

    if tg.chunk is None:
        # 背景塊（核心遮罩全空）：仍寫空白 placeholder，讓縫合每格恰有一檔。
        _write_blank_tile(
            output_dir, tile_name, (ly1 - ly0, lx1 - lx0), seam_edges=seam_edges,
        )
        logger.info("Tile %s: 核心遮罩全空 → 空白 placeholder", tile_name)
        return []

    cr = _finish_chunk_cpu(tg.chunk)

    owned = filter_and_absolutize(cr, geometry, tg.abs_x, tg.abs_y)

    # 標註 overlay（醫師 / slide 級 QuPath），以 dish_mask_overlay 為底畫全塊、核心裁切。
    # 以全塊 results/instance_mask/all_dots/dish_nucleus_mask/per_cell_dots 繪製後裁核心：
    # 核心區彼此無重疊、無縫隙，故每個標註像素在拼回的整片中恰出現一次。
    annotated = render_overlay_image(
        cr.dish_mask_overlay, cr.instance_mask, cr.results,
        all_dots=cr.all_dots,
        dish_nucleus_mask=cr.dish_nucleus_mask,
        per_cell_dots=cr.per_cell_dots,
    )
    # ascontiguousarray：crop 為非連續 view，cv2 畫線需連續緩衝。
    overlay_crop = np.ascontiguousarray(annotated[ly0:ly1, lx0:lx1])
    draw_tile_seam_edges(
        overlay_crop, right=seam_edges[0], bottom=seam_edges[1]
    )
    _save_tile_array(
        output_dir / _STITCH_SCRATCH / f"{tile_name}.tiff",
        overlay_crop,
    )

    # 可選 merge overlay（呼叫端給了 merge_dir 才產）。
    if merge_dir is not None:
        _export_chunk_merge_overlay(
            merge_dir, output_dir, tile_name, cr, (lx0, lx1, ly0, ly1)
        )

    elapsed = time.perf_counter() - tg.start_time
    pos_count = sum(1 for r in owned if r.is_her2_positive)
    logger.info(
        "Tile %s 完成: 核心擁有 %d 細胞 (%d 陽性), %.2f 秒",
        tile_name, len(owned), pos_count, elapsed,
    )
    return owned


def _save_tile_array(path: Path, array: np.ndarray) -> None:
    """建立父目錄後把單塊陣列落地（RGB/mask PNG 或 int32 label TIFF 皆用此路徑）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(str(path), array, check_contrast=False)


def _write_blank_tile(
    output_dir: Path,
    tile_name: str,
    crop_hw: tuple,
    seam_edges: tuple = (False, False),
) -> None:
    """核心遮罩全空的背景塊：寫一張空白 overlay placeholder，維持縫合每格一檔。

    ``seam_edges`` = ``(right, bottom)``：空白塊的 overlay 仍畫 tile 接縫虛線，讓拼回
    overlay_slide 後的接縫格線在空白區域維持連續、不留缺口。
    """
    ch, cw = crop_hw
    blank_overlay = np.full(
        (ch, cw, 3), config.background_fill_value, dtype=np.uint8
    )
    draw_tile_seam_edges(
        blank_overlay, right=seam_edges[0], bottom=seam_edges[1]
    )
    _save_tile_array(
        output_dir / _STITCH_SCRATCH / f"{tile_name}.tiff",
        blank_overlay,
    )


def _export_chunk_merge_overlay(
    merge_dir: Path,
    output_dir: Path,
    tile_name: str,
    cr: ChunkResult,
    crop: tuple,
) -> None:
    """若 ``merge_dir`` 有同名（同座標）merge tile 且同尺寸，畫邊界後核心裁切落地。

    假設 merge tile 已與 IHC/DISH 以相同 ``tile_x{x}_y{y}`` 命名、相同格線預切；
    尺寸不符則跳過（best-effort，不阻斷主流程）。
    """
    lx0, lx1, ly0, ly1 = crop
    merge_path = _find_merge_tile(merge_dir, tile_name)
    if merge_path is None:
        return
    merge_image = _read_rgb(merge_path)
    if merge_image.shape[:2] != cr.instance_mask.shape[:2]:
        logger.warning(
            "Tile %s: merge 影像尺寸 %s 與 mask %s 不匹配，跳過 merge overlay",
            tile_name, merge_image.shape[:2], cr.instance_mask.shape[:2],
        )
        return
    annotated = render_overlay_image(
        merge_image, cr.instance_mask, cr.results, all_dots=cr.all_dots
    )
    _save_tile_array(
        output_dir / "merge_overlay" / f"{tile_name}.tiff",
        annotated[ly0:ly1, lx0:lx1],
    )
    logger.info("Merge overlay 匯出完成: %s", tile_name)


def _process_one_chunk_gpu(
    ihc: np.ndarray,
    dish: np.ndarray,
    abs_x: int,
    abs_y: int,
    edge_flags: tuple,
    unet_inferencer: object,
    cellpose_segmenter: CellposeSegmenter,
    dish_cellpose_segmenter: CellposeSegmenter,
) -> Optional[_ChunkGpuState]:
    """單塊的 GPU 前段：M1→M2→M3b 三個前向，核心遮罩全空回 None。

    只跑到 M3b 的 ``segment_windowed`` 為止；``build_all_positive_results`` /
    ``enlarge_cell_instances`` / ``detect_all_dots`` + merge 這整段純 CPU 後段都留給
    ``_finish_chunk_cpu``。前兩者原本夾在 M2 與 M3b 兩個前向之間跑在**主執行緒**上，
    以 ``torch.cuda.Event`` 實測每塊會撐開一段 GPU 完全閒置的空窗（medium 錨點 84.5
    ms/塊，其中這兩支佔 8.4 s / 121 塊）；而主執行緒這條 arm 才是關鍵路徑（BG/MAIN
    ≈ 0.74，背景 arm 尚有餘裕）。兩者皆為純 NumPy/skimage、不碰 torch/CUDA，且其產物
    只有 ``_finish_chunk_cpu`` 會讀，故移到背景執行緒不改變任何計算結果。

    ``edge_flags`` = ``(clear_top, clear_bottom, clear_left, clear_right)``（由
    ``TileGeometry.edge_flags`` 提供）：M2 不在分塊內部接縫清邊（``remove_border=False``），
    改在 M3 之前只清「碰到真實 slide 外緣」的細胞；跨塊重複偵測由縫合層的質心
    core-ownership 去重。單塊時四邊皆真實 slide 邊 → 等同現行 M2 清邊行為。
    """
    core_mask = generate_ihc_core_mask(
        ihc,
        unet_inferencer,
        close_kernel=config.core_close_kernel,
    )
    if core_mask.sum() == 0:
        return None

    m1 = _run_m1_overlay_stage(ihc, dish, core_mask)

    instance_mask = segment_windowed(
        m1.m2_input_overlay,
        cellpose_segmenter,
        tile_size=config.default_tile_size,
        overlap=config.window_overlap_px,
        dedup_iomin=config.window_dedup_iomin,
    )
    if config.clear_border_cells:
        clear_top, clear_bottom, clear_left, clear_right = edge_flags
        instance_mask = clear_slide_edge_cells(
            instance_mask,
            clear_top=clear_top,
            clear_bottom=clear_bottom,
            clear_left=clear_left,
            clear_right=clear_right,
        )

    dish_nucleus_mask = segment_windowed(
        dish,
        dish_cellpose_segmenter,
        tile_size=config.default_tile_size,
        overlap=config.window_overlap_px,
        dedup_iomin=config.window_dedup_iomin,
    )

    return _ChunkGpuState(
        abs_x=abs_x,
        abs_y=abs_y,
        core_mask=core_mask,
        dish_mask_overlay=m1.dish_mask_overlay,
        instance_mask=instance_mask,
        dish_nucleus_mask=dish_nucleus_mask,
    )


def _finish_chunk_cpu(gs: _ChunkGpuState) -> ChunkResult:
    """單塊的 CPU 後段：M3 前處理 + ``detect_all_dots`` + merge → 完整 ``ChunkResult``。

    純 CPU、無 torch，設計為在背景執行緒執行。計算與拆分前完全相同（同樣的輸入、同樣的
    先後順序），只是 ``build_all_positive_results`` / ``enlarge_cell_instances`` 這兩步
    改由本執行緒算，而不再佔用主執行緒的 GPU arm——見 ``_process_one_chunk_gpu`` docstring。
    """
    results_pre = build_all_positive_results(gs.instance_mask)
    # M3 配對前處理：把綠色細胞 mask 實際放大（面積 ×cell_enlarge_area_factor），讓細胞
    # 蓋到更多 DISH 核以提高配對成功率。放大版僅供配對 / 點偵測；原始 instance_mask 仍
    # 用於 M4 視覺化與裁切，醫師看到的綠框維持不變。
    matching_mask = enlarge_cell_instances(gs.instance_mask, config)

    all_dots, per_cell_dots, dish_nucleus_mask = detect_all_dots(
        gs.dish_mask_overlay,
        matching_mask,
        config,
        dish_nucleus_mask=gs.dish_nucleus_mask,
        core_mask=gs.core_mask,
        n_jobs=config.dot_detect_n_jobs,
    )
    results = merge_dot_results_to_cell_analysis(results_pre, per_cell_dots)

    return ChunkResult(
        abs_x=gs.abs_x,
        abs_y=gs.abs_y,
        instance_mask=gs.instance_mask,
        dish_nucleus_mask=dish_nucleus_mask,
        dish_mask_overlay=gs.dish_mask_overlay,
        results=results,
        all_dots=all_dots,
        per_cell_dots=per_cell_dots,
    )


def _run_m1_overlay_stage(
    ihc_image: np.ndarray,
    dish_image: np.ndarray,
    core_mask: np.ndarray,
) -> M1StageArtifacts:
    """執行 M1 overlay：masked IHC/DISH 與 M2 輸入影像。"""
    masked_ihc = apply_mask_to_ihc_image(
        ihc_image,
        core_mask,
        mask_blur_sigma=config.mask_blur_sigma,
        background_fill_value=config.background_fill_value,
    )
    dish_mask_overlay = overlay_ihc_mask_on_dish(
        dish_image,
        core_mask,
        mask_blur_sigma=config.mask_blur_sigma,
        background_fill_value=config.background_fill_value,
    )
    overlay_image = fuse_masked_ihc_with_dish(
        dish_mask_overlay,
        masked_ihc,
        ihc_alpha=config.overlay_alpha,
    )

    return M1StageArtifacts(
        core_mask=core_mask,
        dish_mask_overlay=dish_mask_overlay,
        m2_input_overlay=overlay_image,
    )


def _find_merge_tile(merge_dir: Optional[Path], tile_id: str) -> Optional[Path]:
    """嘗試在 merge 目錄中找到對應的合併影像。"""
    if merge_dir is None or not merge_dir.exists():
        return None
    for ext in config.supported_extensions:
        candidate = merge_dir / f"{tile_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _read_rgb(src: Path) -> np.ndarray:
    """讀取影像並確保為 RGB uint8。"""
    image = io.imread(str(src))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    return image.astype(np.uint8)


# ------------------------------------------------------------------
# 跨 tile 多行程（Candidate D，doc 20 §2）
# ------------------------------------------------------------------
# 單行程下所有槓桿的下限是 `BG + outside`（doc 18 §6.2：large 錨點 389.3 s，即 1.23x），
# 因為背景 CPU 臂遲早會變成關鍵路徑。多行程不受該下限約束——**每個行程自帶自己的 BG
# 執行緒**，兩條臂一起平行。代價是每行程要自己重載三個模型（自己的 CUDA context）。
#
# 為何是 `spawn` 而非 `fork`：CUDA context 不是 fork-safe，forked child 會繼承一個壞掉
# 的 context，第一次 CUDA 呼叫就掛（本目錄 CLAUDE.md 明載的限制）。`spawn` 讓每個 child
# 乾淨地重新 import + 重新初始化，代價是每行程付一次 init/VRAM。
#
# 正確性不變量（doc 20 §1）在此的落實方式：
#   1. 全域 cell 重編號仍**只在父行程做一次**：worker 只回傳 (abs_x, abs_y, owned)，
#      與單行程迴圈內部產生的 tuple 完全相同，父行程照原本的 (abs_y, abs_x, cell_id)
#      排序後重編號——那段程式碼一個字都沒改。
#   2. fail-fast 是整批而非單 worker：任一 worker 出錯即回報，父行程**先終止所有兄弟
#      行程**再 raise。讓兄弟跑完會產出「有未記載破洞的玻片」，正是 fail-fast 要擋的。
#   3. 每塊恰由一個 worker 處理：動態工作佇列，每塊只被 put 一次、被一個 worker get 到，
#      故逐塊輸出檔天然無競爭。
#   4. `gc.freeze()` 契約：worker 是每次 `run_batch` 現生現死（非常駐池），故
#      freeze/unfreeze 的每次呼叫語意自動成立，不需要新的設計面。

_MP_POISON = None                            # 工作佇列的結束哨兵

# worker 內量測掛勾（doc 26 Tier 1.1 / DISCOVERED #40）。
#
# `scripts/perf_measure.py` 是 monkeypatch 式的：它在**父行程**的 hybrid_pipeline 命名空間
# 上包一層計時器。spawn 出來的 worker 重新 import 這個模組，拿到的是乾淨的原函式，所以
# `workers>1` 下所有 per-stage 的桶（B1/B2/B3/...）都是空的——整批只剩端到端 wall 可讀，
# 「為什麼慢」完全不可測（doc 21 §10 follow-up #1）。
#
# 修法刻意不讓管線去 import 量測腳本（那會把測量工具變成執行期相依）。改成：worker 啟動時
# 讀 `HYBRID_MP_WORKER_PROBE` 環境變數，格式 `module.path:callable`；有設才 import 並呼叫，
# 該 callable 需回傳一個「無參數、回傳 dict」的收集器。worker 收工前把收集到的 dict 從既有的
# result queue 送回父行程。沒設環境變數時這段是一個 `os.environ.get` 的成本，其餘為零。
_MP_PROBE_ENV = "HYBRID_MP_WORKER_PROBE"

# 上一次 `run_batch(workers>1)` 收到的 worker 端 TIMINGS，供量測腳本在跑完後取用。
# 每個元素是 `{"worker": <name>, "timings": {...}}`。
LAST_MP_WORKER_TIMINGS: List[dict] = []


def _install_worker_probe():
    """依 ``HYBRID_MP_WORKER_PROBE`` 安裝 worker 端量測掛勾，回傳收集器或 ``None``。

    刻意吞掉所有例外並只記 warning：量測掛勾壞掉不該讓一批真實的分析工作失敗。
    """
    spec = os.environ.get(_MP_PROBE_ENV)
    if not spec:
        return None
    try:
        import importlib
        mod_name, _, func_name = spec.partition(":")
        return getattr(importlib.import_module(mod_name), func_name)()
    except Exception as exc:                 # noqa: BLE001 — 見 docstring
        logger.warning("worker 量測掛勾 %r 安裝失敗: %r（本 worker 不回報 timings）",
                       spec, exc)
        return None


def _mp_tile_worker(
    task_q,
    result_q,
    geometry: TileGeometry,
    output_dir: Path,
    merge_dir: Optional[Path],
    parent_cfg_hash: str,
) -> None:
    """Worker 行程進入點：載一次模型，然後把工作佇列抽乾。

    迴圈結構刻意與 ``run_batch`` 的單行程迴圈**逐行對應**（深度 1 的 GPU/CPU 重疊：
    背景執行緒跑前一塊的 CPU 後段，主執行緒跑下一塊的 GPU 前段），因為那個重疊本身
    就值 doc 18 §2 量到的 −8.0%。少了它，每個 worker 都會退化成 round-3 之前的序列
    版本，多行程賺到的會被這裡賠掉。

    以 ``result_q`` 回報四種訊息：
      - ``("ready", None)``：模型已載完（供父行程量測 init 是否平行）。
      - ``("ok", (abs_x, abs_y, owned))``：一塊完成。
      - ``("timings", {...})``：本 worker 的 per-stage 計時（僅在量測掛勾啟用時）。
      - ``("error", message)``：真實錯誤 → 父行程終止全部 worker 後 raise。
    """
    collect_timings = _install_worker_probe()   # 必須早於 _init_*，否則量不到 init
    try:
        cfg_hash = compute_config_hash(config)
        if cfg_hash != parent_cfg_hash:
            # spawn 的 child 是重新 import config 的，不會繼承父行程對 config 單例的
            # 執行期修改（例如 perf_measure.py 的 --cellpose-batch-size）。靜靜地用
            # 不同設定跑會產出無法追溯的結果，故明確擋掉。
            result_q.put(("error", (
                f"worker config_hash {cfg_hash} != parent {parent_cfg_hash}；"
                f"父行程對 config 的執行期修改不會傳到 spawn 的 worker。"
            )))
            return

        unet = _init_unet_inferencer()
        cellpose = _init_cellpose_segmenter()
        dish_cellpose = _init_dish_cellpose_segmenter()
        result_q.put(("ready", None))

        pending: Optional[Tuple[int, int, Future]] = None
        with _frozen_gc_generation(), \
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="tile-cpu") as pool:
            while True:
                task = task_q.get()
                if task is _MP_POISON:
                    break
                ihc_path, dish_path, ax, ay = task

                tg = _process_precut_tile_gpu(
                    ihc_path, dish_path, ax, ay, geometry,
                    unet, cellpose, dish_cellpose, output_dir,
                )
                if tg is None:
                    result_q.put(("error", (
                        f"process_precut_tile 於 tile_x{ax}_y{ay} 失敗"
                        f"（讀檔 / 維度不符，見 worker 日誌）。"
                    )))
                    return

                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                gc.collect()

                # 先收再提交：同時最多只有兩塊在飛，記憶體有界。加深這條管線（depth 2）
                # 已實測並否決 — 見 doc 21 §6（單行程 +2.8%，W=3 下持平）。
                if pending is not None:
                    p_ax, p_ay, fut = pending
                    result_q.put(("ok", (p_ax, p_ay, fut.result())))
                pending = (ax, ay, pool.submit(
                    _process_precut_tile_cpu, tg, geometry, output_dir, merge_dir,
                ))

            if pending is not None:
                p_ax, p_ay, fut = pending
                result_q.put(("ok", (p_ax, p_ay, fut.result())))

        if collect_timings is not None:
            # 收工後才送：父行程的收件迴圈在收滿 total 塊後才會去撈這些，先送也只是排隊。
            import multiprocessing
            result_q.put(("timings", {
                "worker": multiprocessing.current_process().name,
                "timings": collect_timings(),
            }))
    except Exception as exc:                 # noqa: BLE001 — 任何例外都必須傳回父行程
        logger.error("Worker 例外: %s", exc, exc_info=True)
        result_q.put(("error", f"worker 例外: {exc!r}"))


def _run_tiles_multiprocess(
    tiles,
    total: int,
    geometry: TileGeometry,
    output_dir: Path,
    merge_dir: Optional[Path],
    workers: int,
    cfg_hash: str,
    on_tile=None,
) -> List[Tuple[int, int, List[CellAnalysisResult]]]:
    """把 tile 分派給 ``workers`` 個 spawn 行程，回收每塊的 ``(abs_x, abs_y, owned)``。

    採**動態工作佇列**而非靜態輪流分配：每塊成本差異很大（背景塊走快速路徑、組織密集塊
    要跑滿三個前向），且事前不知道，靜態分配會讓某個 worker 卡在密集區時其他人空等。

    ``tiles`` 是迭代器（可能是邊切邊產出的 ``PrecutStream``），故以一條 feeder 執行緒
    餵佇列，主執行緒同時收結果——否則兩邊會互等而死鎖。

    ``on_tile(abs_x, abs_y, owned)`` 若給定，會在**每塊回來的當下**於父行程呼叫（斷點
    續跑就掛在這裡）。回傳值仍是完整清單，與不給時相同。
    """
    import multiprocessing as mp
    import queue as _queue
    import threading

    # CUDA allocator 設定必須在 worker 的第一次 CUDA 配置之前決定。父行程在這條路徑上
    # 完全不碰 CUDA，spawn 的 child 又會繼承 environ，所以「spawn 之前寫 os.environ」
    # 是唯一能真正生效、且不需要呼叫端記得先 export 的位置。
    # 預設空字串 = 完全不碰環境 = 今日行為；`workers≥6` 的 24.76 GiB 配置器氣球
    # （19 #7b / DISCOVERED #2）就是靠這個旋鈕掃 `expandable_segments:True` 的。
    alloc_conf = getattr(config, "cuda_alloc_conf", "")
    if alloc_conf and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
    logger.info("worker CUDA allocator 設定: PYTORCH_CUDA_ALLOC_CONF=%r",
                os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))

    ctx = mp.get_context("spawn")            # 絕不用 fork（見本節開頭）
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    LAST_MP_WORKER_TIMINGS.clear()

    procs = [
        ctx.Process(
            target=_mp_tile_worker,
            args=(task_q, result_q, geometry, output_dir, merge_dir, cfg_hash),
            name=f"tile-worker-{i}",
            daemon=False,
        )
        for i in range(workers)
    ]

    def _kill_all() -> None:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=10)

    for p in procs:
        p.start()

    stop_feeding = threading.Event()

    def _feed() -> None:
        try:
            for ihc_path, dish_path, (ax, ay) in tiles:
                # 中止後必須真的停下來：feeder 是 daemon thread，run_batch 又會在長駐的
                # API server 行程裡被反覆呼叫，若 fail-fast 之後它還在拉 PrecutStream，
                # 就會在整批已放棄的情況下繼續把整張玻片切到磁碟。
                if stop_feeding.is_set():
                    break
                task_q.put((ihc_path, dish_path, ax, ay))
        except Exception as exc:             # noqa: BLE001 — 例如串流切檔失敗
            result_q.put(("error", f"tile 供給失敗: {exc!r}"))
        finally:
            for _ in procs:
                task_q.put(_MP_POISON)

    feeder = threading.Thread(target=_feed, name="tile-feeder", daemon=True)
    feeder.start()

    collected: List[Tuple[int, int, List[CellAnalysisResult]]] = []
    ready = 0
    t_start = time.perf_counter()
    try:
        while len(collected) < total:
            try:
                kind, payload = result_q.get(timeout=5)
            except _queue.Empty:
                # worker 猝死（OOM / segfault）不會送任何訊息；不檢查就會永遠卡住。
                if any(p.exitcode not in (None, 0) for p in procs):
                    raise RuntimeError(
                        f"worker 行程異常結束（exit codes "
                        f"{[p.exitcode for p in procs]}）；整批 fail-fast 中止。"
                    )
                if all(p.exitcode is not None for p in procs):
                    # 全部乾淨退出卻還沒收滿：代表有塊被吞了，不能當成功回傳。
                    raise RuntimeError(
                        f"所有 worker 已結束，但只收到 {len(collected)}/{total} 塊；"
                        f"整批 fail-fast 中止以免產出有未記載破洞的玻片。"
                    )
                continue

            if kind == "error":
                raise RuntimeError(f"{payload}；整批 fail-fast 中止。")
            if kind == "ready":
                ready += 1
                if ready == workers:
                    logger.info(
                        "%d 個 worker 模型載入完成，耗時 %.2f 秒（平行）",
                        workers, time.perf_counter() - t_start,
                    )
                continue
            if kind == "timings":
                LAST_MP_WORKER_TIMINGS.append(payload)
                continue
            collected.append(payload)
            if on_tile is not None:
                on_tile(*payload)
            logger.info("[%d/%d] 已回收 tile_x%d_y%d",
                        len(collected), total, payload[0], payload[1])
    except BaseException:
        # 任一塊失敗 → 先停止供料、再終止所有兄弟行程，然後才往上拋。放兄弟跑完會產出
        # 「有未記載破洞」的玻片，正是單行程 fail-fast 設計要擋的失效模式（doc 20 §1 item 2）。
        stop_feeding.set()
        _kill_all()
        raise

    # worker 的 timings 是收工前最後才送的，所以最後一塊被收下時通常還沒到。必須在
    # join 之前撈完：worker 送完就退出，而 join 會等到佇列被排空才回來。
    if os.environ.get(_MP_PROBE_ENV):
        deadline = time.perf_counter() + 30
        while len(LAST_MP_WORKER_TIMINGS) < workers and time.perf_counter() < deadline:
            try:
                kind, payload = result_q.get(timeout=1)
            except _queue.Empty:
                if all(p.exitcode is not None for p in procs):
                    break
                continue
            if kind == "timings":
                LAST_MP_WORKER_TIMINGS.append(payload)
        if len(LAST_MP_WORKER_TIMINGS) < workers:
            logger.warning("只收到 %d/%d 個 worker 的 timings。",
                           len(LAST_MP_WORKER_TIMINGS), workers)

    for p in procs:
        p.join(timeout=60)
    _kill_all()                              # 任何沒乾淨退出的殘留一律收掉
    return collected


# ------------------------------------------------------------------
# 斷點續跑（partial resume）
# ------------------------------------------------------------------
# `run_batch` 是 fail-fast 的：任一塊真實失敗即中止整批。對「各自獨立的影像」那是錯的，
# 對「同一張玻片的碎片」那是對的——留下有未記載破洞的玻片比大聲崩潰更糟（doc 20 §1
# item 2）。但 fail-fast 有一個它沒回答的問題：**已經算完的 27,000 塊怎麼辦**。全片
# 27,565 塊時，第 25,000 塊 OOM 會賠掉整整數小時的運算，而那些塊的結果本來完全可用。
# 這正是 round 5b/6 建議 unattended 跑 `workers=4` 而非更快的 `workers=6` 的真正原因
# （doc 21 §10 follow-up #7 / 19 #1c）。
#
# 這裡補的是最小可用版本：每塊完成時把它的 `owned` 結果（唯一無法從磁碟重建的東西——
# 其餘每塊產物本來就已落地）pickle 到 `output_dir/_resume/`，重跑時載回來並跳過該塊。
# 語意上**不放寬 fail-fast**：失敗照樣中止整批，只是重跑時不必從頭算。
#
# 設計取捨：
#   - 預設關閉（`checkpoint=False`）。API 的單塊請求開了只是白付 I/O，而且會在輸出目錄
#     留下 scratch；這是給長時間無人看顧的整片跑用的旋鈕，由呼叫端明講。
#   - 每塊一檔而非單一 append log：多行程下每塊恰由一個 worker 產生、由**父行程**寫出，
#     一檔一塊天然免鎖；寫入走 tmp+rename，故中途被 kill 不會留下半個檔。
#   - 存 config_hash：設定變了，先前那些塊就不再可信。不符時大聲警告並整批重算
#     （而不是靜靜混用兩種設定的結果，那是無法追溯的產物）。

_CKPT_DIRNAME = "_resume"


def _checkpoint_load(
    ckpt_dir: Path, cfg_hash: str, positions: Iterable[Tuple[int, int]],
) -> Dict[Tuple[int, int], List[CellAnalysisResult]]:
    """載入先前跑到一半留下的每塊結果；沒有 / 設定不符 / 壞檔一律當作沒有。"""
    if not ckpt_dir.is_dir():
        return {}

    hash_file = ckpt_dir / "config_hash.txt"
    prev = hash_file.read_text().strip() if hash_file.exists() else ""
    if prev != cfg_hash:
        logger.warning(
            "斷點目錄 %s 的 config_hash 為 %r，與本次的 %r 不符——先前的塊是以不同設定"
            "算出的，不可混用。本次將整批重算（舊檔會被逐一覆蓋，不刪除）。",
            ckpt_dir, prev, cfg_hash,
        )
        return {}

    wanted = set(positions)
    done: Dict[Tuple[int, int], List[CellAnalysisResult]] = {}
    for path in ckpt_dir.glob("tile_x*_y*.pkl"):
        try:
            ax, ay = parse_tile_coords(path.name)
        except ValueError:
            continue
        if (ax, ay) not in wanted:
            continue                         # 上次跑的是別的格線，不是這批的塊
        try:
            with open(path, "rb") as fh:
                done[(ax, ay)] = pickle.load(fh)
        except Exception as exc:             # noqa: BLE001 — 壞檔就當沒算過，重算即可
            logger.warning("斷點檔 %s 讀取失敗（%r），該塊將重算。", path, exc)
    if done:
        logger.info("斷點續跑：%s 已有 %d 塊完成，本批將跳過它們。", ckpt_dir, len(done))
    return done


def _checkpoint_init(ckpt_dir: Path, cfg_hash: str) -> None:
    """建立斷點目錄並寫下本次的 config_hash。"""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config_hash.txt").write_text(cfg_hash)


def _checkpoint_save(
    ckpt_dir: Path, abs_x: int, abs_y: int, owned: List[CellAnalysisResult],
) -> None:
    """把一塊的結果原子性地寫進斷點目錄（tmp+rename，故中途被 kill 不留半個檔）。"""
    dst = ckpt_dir / f"tile_x{abs_x}_y{abs_y}.pkl"
    tmp = dst.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(owned, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, dst)


def _skip_completed(tiles, done: Dict[Tuple[int, int], List[CellAnalysisResult]]):
    """過濾掉斷點已記錄完成的塊。

    串流模式下仍會讓 ``PrecutStream`` 把該塊切出來（切檔是冪等的、且格線由它供給），
    只是不再送去分析——省下的是三個 GPU 前向與 CPU 後段，那才是成本所在。
    """
    for ihc_path, dish_path, (ax, ay) in tiles:
        if (ax, ay) not in done:
            yield ihc_path, dish_path, (ax, ay)


# ------------------------------------------------------------------
# 批次處理
# ------------------------------------------------------------------

def run_batch(
    ihc_dir: Path,
    dish_dir: Path,
    output_dir: Path,
    merge_dir: Optional[Path] = None,
    tile_stream: Optional[object] = None,
    workers: int = 4,
    checkpoint: bool = False,
) -> dict:
    """批次處理『已預切』tile 目錄：逐塊分析 → 全域合併細胞表 → slide 級 overlay 縫合。

    ``ihc_dir`` / ``dish_dir`` 為 ``precut_paired_tiles`` 產出的、以
    ``tile_x{int}_y{int}`` 命名的重疊 tile 檔目錄（兩路同名同格線）。

    整批的所有 tile 是**同一張玻片**被切開的碎片，必須全數成功、每格恰一檔，
    slide 級輸出才可信。因此本函式採 **fail-fast**：任何一塊發生真實錯誤
    （``process_precut_tile`` 回傳 ``None``：讀檔 / 維度不符）即 raise 中止整批，
    而非「記錄後續跑」——這是有意偏離舊 ``run_batch`` 行為（舊行為適用於各自獨立、
    互不相關的影像；現在每塊是單一玻片的一部分，靜默略過會產出「有未記載破洞」的
    玻片，比大聲崩潰更糟）。

    Args:
        ihc_dir: 已預切 IHC tile 目錄。
        dish_dir: 已預切 DISH tile 目錄。
        output_dir: 輸出根目錄。
        merge_dir: 合併影像目錄 (可選)，用於產出 merge overlay。
        tile_stream: 可選的 ``m0_reader.PrecutStream``。給定時**改由它供給 tile**
            （``ihc_dir`` / ``dish_dir`` 不再被讀取），預切與本分析迴圈重疊執行，省掉
            「整批切完才開工」的序列等待；不給則維持原本掃目錄的行為。處理順序改變不
            影響輸出（全域重編號依 ``(abs_y, abs_x, cell_id)`` 排序、縫合按座標讀檔）。
        workers: 跨 tile 平行的**行程**數。``1``（預設）= 今日的單行程雙臂路徑，
            一行為變化都沒有；``>1`` 走 ``_run_tiles_multiprocess``，每個 worker 自帶
            一份模型與 CUDA context。預設必須維持 1：API 的單塊請求不該為了平行度去付
            N 份模型初始化成本（doc 20 §1 item 7）。
        checkpoint: 是否啟用斷點續跑。``True`` 時每塊完成即把結果落地到
            ``output_dir/_resume/``，且**本次開始前**會先載入該目錄中屬於這批格線、
            且 config_hash 相符的塊並跳過它們。給無人看顧的整片跑用（27,565 塊時，
            第 25,000 塊失敗不該賠掉前面數小時）；預設 ``False``，API 的單塊請求開了
            只是白付 I/O。**不放寬 fail-fast**：任一塊失敗照樣中止整批。

    Returns:
        ``{"success": int, "skipped": int}`` 統計。批次內任一塊真實失敗即
        raise 中止整批（見上），故統計裡不設 ``failed`` 計數。
    """
    run_id = uuid.uuid4().hex[:8]
    cfg_hash = compute_config_hash(config)
    output_dir = Path(output_dir)
    logger.info(
        "批次處理開始 — run_id=%s, config_hash=%s", run_id, cfg_hash
    )

    if tile_stream is None:
        paired_tiles = find_paired_tiles(
            ihc_dir, dish_dir, config.supported_extensions
        )

        if not paired_tiles:
            logger.warning("未找到任何配對 tile")
            return {"success": 0, "skipped": 0}

        # 由檔名解析每塊 (abs_x, abs_y)；IHC/DISH 同名同座標，防禦性地兩路都解析並比對。
        positions: List[Tuple[int, int]] = []
        for ihc_path, dish_path in paired_tiles:
            ax, ay = parse_tile_coords(dish_path.name)
            ihc_xy = parse_tile_coords(ihc_path.name)
            if ihc_xy != (ax, ay):
                raise ValueError(
                    f"IHC/DISH tile 座標不一致: {ihc_path.name} {ihc_xy} "
                    f"vs {dish_path.name} {(ax, ay)}"
                )
            positions.append((ax, ay))
        tiles = ((i, d, p) for (i, d), p in zip(paired_tiles, positions))
    else:
        # 串流模式：格線先到（讀檔頭即可算出），tile 檔隨切隨到。IHC/DISH 兩路檔名由
        # 同一個 pos 生成，座標一致是建構保證，不需再比對。
        positions = list(tile_stream.positions)
        if not positions:
            logger.warning("未找到任何配對 tile")
            return {"success": 0, "skipped": 0}
        tiles = iter(tile_stream)

    # 一次算出縫合幾何；格線不完整（缺格 / 重複 / 對不上）會 raise ValueError，
    # 屬預期的 fail-fast 驗證，不吞。
    geometry = compute_tile_geometry(
        positions, config.default_tile_size, config.window_overlap_px
    )

    stats = {"success": 0, "skipped": 0}
    total = len(positions)
    # 每塊回傳 (abs_x, abs_y, owned_results)，供迴圈後全域排序 / 重編號。
    per_tile_owned: List[Tuple[int, int, List[CellAnalysisResult]]] = []

    # 斷點續跑：先把已完成的塊補進 per_tile_owned，再把它們從待跑的 tiles 濾掉。
    # 兩條路徑（單行程 / 多行程）共用同一份載入與存檔，跟 _finish_batch 一樣只有一個實作。
    ckpt_dir = output_dir / _CKPT_DIRNAME
    done: Dict[Tuple[int, int], List[CellAnalysisResult]] = {}
    if checkpoint:
        done = _checkpoint_load(ckpt_dir, cfg_hash, positions)
        _checkpoint_init(ckpt_dir, cfg_hash)
        if done:
            tiles = _skip_completed(tiles, done)
            for (ax, ay), owned in done.items():
                per_tile_owned.append((ax, ay, owned))
                stats["skipped" if len(owned) == 0 else "success"] += 1

    def _record(abs_x: int, abs_y: int, owned: List[CellAnalysisResult]) -> None:
        """累計統計，並在啟用斷點時把這塊落地。父行程唯一的收斂點。"""
        per_tile_owned.append((abs_x, abs_y, owned))
        stats["skipped" if len(owned) == 0 else "success"] += 1
        if checkpoint:
            _checkpoint_save(ckpt_dir, abs_x, abs_y, owned)

    remaining = total - len(done)

    if remaining == 0:
        # 斷點裡已經有全部的塊：直接去做全域合併 + 縫合。不短路的話，兩條路徑都會為了
        # 零塊工作先把三個模型載起來（多行程還是每個 worker 各一份），純浪費。這條路徑
        # 也正好是「只想重跑最後的縫合」時要走的。
        logger.info("斷點已涵蓋全部 %d 塊，跳過分析，直接進行全域合併與縫合。", total)
        return _finish_batch(
            run_id, per_tile_owned, stats, total, output_dir, geometry,
        )

    if workers > 1:
        # 多行程路徑：模型只在 worker 內載入，父行程完全不碰 CUDA（不然會白付一份
        # context + 權重）。回傳的 tuple 與單行程迴圈產生的完全同型，故下方的全域
        # 合併 / 重編號 / 縫合完全共用，一行都不必改。
        _run_tiles_multiprocess(
            tiles, remaining, geometry, output_dir, merge_dir, workers, cfg_hash,
            on_tile=_record,
        )
        return _finish_batch(
            run_id, per_tile_owned, stats, total, output_dir, geometry,
        )

    unet = _init_unet_inferencer()
    cellpose = _init_cellpose_segmenter()
    dish_cellpose = _init_dish_cellpose_segmenter()

    def _collect(entry: Tuple[int, int, Future]) -> None:
        """收一塊已提交的 CPU 後段結果並累計統計；其真實錯誤在此 raise → fail-fast。"""
        e_ax, e_ay, fut = entry
        _record(e_ax, e_ay, fut.result())

    # 兩段式管線（單一背景執行緒，管線深度 1）：三個 GPU 模型仍只在主行程 / 單一 CUDA
    # context 載入一次並重用，且**所有 GPU 前向仍只在主執行緒序列執行**（背景執行緒完全
    # 不碰 torch），故不觸犯跨行程 fork-under-CUDA 限制。差別只在把每塊的 CPU 後段
    # （detect_all_dots + 落地寫檔，原本讓 GPU 整段閒置）丟到背景執行緒，與『下一塊的 GPU
    # 前向』重疊執行，藉此填補 GPU 閒置。fail-fast 與 empty_cache/gc 清理語意皆保留（見下）。
    pending: Optional[Tuple[int, int, Future]] = None
    with _frozen_gc_generation(), \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="tile-cpu") as pool:
        for idx, (ihc_path, dish_path, (ax, ay)) in enumerate(tiles, start=1):
            logger.info(
                "[%d/%d] 處理 tile: %s", idx, remaining, dish_path.stem
            )
            tg = _process_precut_tile_gpu(
                ihc_path, dish_path, ax, ay, geometry,
                unet, cellpose, dish_cellpose, output_dir,
            )
            if tg is None:
                # 真實錯誤：此塊是單一玻片之一，任一塊失敗即整片不可信 → fail-fast 中止整批。
                logger.error(
                    "Tile %s 發生真實錯誤（讀檔 / 維度不符），且該格未寫任何檔——"
                    "此為單一玻片之一塊，整批中止以免產出有未記載破洞的玻片。",
                    dish_path.stem,
                )
                raise RuntimeError(
                    f"process_precut_tile 於 {dish_path.stem} 失敗（見上方日誌）；"
                    f"整批 fail-fast 中止。"
                )

            # tile 間釋放 GPU allocator cache 並跑一次 Python GC（主執行緒持有 CUDA），
            # 防止長批次記憶體單調成長。此時背景執行緒可能仍在跑前一塊的 CPU 後段——它只
            # 持有 numpy 陣列（該塊 GPU 前向早已在主執行緒跑完），故此清理不影響其正確性。
            # 兩者都維持「每塊一次」。改成每 N 塊掃一次（N=4/8/16）已實測並否決：在
            # gc.freeze() 之後整批 gc 只剩 0.52 s，最多再省 0.36 s（<0.1% wall，遠在雜訊內），
            # 卻讓 peak RSS 變成所有配置中最高的一個。零收益的層一律砍掉。見 doc 16 §4.2。
            try:
                if cuda.is_available():
                    cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()

            # 收前一塊的 CPU 後段（剛與本塊 GPU 前向重疊執行）；其真實錯誤在此 raise → fail-fast。
            # 先收再提交本塊：同時最多只有兩塊在飛（本塊 GPU 狀態 + 前一塊 CPU 後段），記憶體有界。
            if pending is not None:
                _collect(pending)
            pending = (ax, ay, pool.submit(
                _process_precut_tile_cpu, tg, geometry, output_dir, merge_dir,
            ))

        # 收最後一塊的 CPU 後段。
        if pending is not None:
            _collect(pending)

    return _finish_batch(
        run_id, per_tile_owned, stats, total, output_dir, geometry,
    )


def _finish_batch(
    run_id: str,
    per_tile_owned: List[Tuple[int, int, List[CellAnalysisResult]]],
    stats: dict,
    total: int,
    output_dir: Path,
    geometry: TileGeometry,
) -> dict:
    """全域合併 → 重編號 → 表格輸出 → slide 級縫合。

    單行程與多行程兩條路徑**共用這一段**，這是 doc 20 §1 item 1 的落實方式：全域 cell
    重編號只有這一個實作、只跑一次、只在父行程跑。worker 絕不重編號、也絕不合併部分排序
    ——它們只回傳與單行程迴圈同型的 ``(abs_x, abs_y, owned)``，順序無關（排序鍵是幾何
    座標而非到達順序）。
    """
    # 全域合併：攤平所有塊的 owned 結果，依正典幾何序 (abs_y, abs_x, cell_id) 排序後
    # 重新編號成 1..N。這是唯一發生全域 cell 編號的地方；質心等其他欄位保留。
    flat = [
        (ay, ax, r.cell_id, r)
        for ax, ay, results in per_tile_owned
        for r in results
    ]
    flat.sort(key=lambda t: (t[0], t[1], t[2]))
    renumbered = [
        replace(r, cell_id=i) for i, (_ay, _ax, _cid, r) in enumerate(flat, start=1)
    ]

    # 表格輸出（空清單由 export_* 自身妥善處理，不特判）。
    export_tile_csv(renumbered, output_dir / "report.csv")
    export_summary_statistics(renumbered, output_dir / "summary.txt")

    # slide 級 overlay：把每格核心裁切的暫存 tile 惰性拼回整片，拼完刪暫存夾。
    _stitch_overlay_slide(output_dir, geometry)

    _log_batch_summary(run_id, stats, total)
    return stats


def _ensure_nofile_limit(needed: int) -> None:
    """確保本行程的 open-file 軟上限足以讓縫合同時開著 ``needed`` 個 tile。

    ``_join_overlay_tiles`` 是**惰性**的：整張玻片的 27,565 個 overlay tile 會全部
    同時以 pyvips image 開著，一直到最後 ``tiffsave`` 拉資料時才真的逐一讀。本機的
    軟上限是 1,048,576，所以一路沒事；但 Linux 常見預設是 **1,024**，在那種機器上
    整片分析（數小時）跑完之後才會在最後一步炸掉——這是這條管線最貴的失敗方式，
    也正是 doc 25 §5.2 記下但沒補的缺口。

    故在開任何檔之前先檢查：軟上限不夠就自己往上提（soft→hard 一律允許，不需
    特權），硬上限也不夠才大聲失敗並講清楚要調多少。
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY or soft >= needed:
        return
    if hard == resource.RLIM_INFINITY or hard >= needed:
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))
        logger.info(
            "RLIMIT_NOFILE 軟上限 %d 不足，已自行提高為 %d（縫合需同時開啟 %d 個 tile）",
            soft, needed, needed,
        )
        return
    raise RuntimeError(
        f"overlay 縫合需同時開啟約 {needed} 個檔案，但本行程的 RLIMIT_NOFILE "
        f"硬上限只有 {hard}（軟上限 {soft}）。請提高硬上限後重跑最後的縫合步驟"
        f"（例如 `ulimit -Hn {needed}`，或於 systemd unit 設 LimitNOFILE={needed}）"
        f"——整片的逐塊 overlay 已在 {_STITCH_SCRATCH}/ 落地，不需重跑。"
    )


def _join_overlay_tiles(output_dir: Path, geometry: TileGeometry) -> pyvips.Image:
    """把 ``_stitch_scratch/`` 內每格核心裁切的 tile 惰性拼成一張全片影像（尚未編碼）。

    每格 overlay 的尺寸依所在欄 / 列而異（邊界格較小），但**同欄同寬、同列同高**
    （由 ``core_crop_bounds`` 的建構保證）。``pyvips.Image.arrayjoin()`` 只做「等格montage」
    （以最大寬高為格、其餘留白），無法正確處理非均勻的每欄 / 每列尺寸（已用合成測試證實
    它會多出留白、尺寸錯誤）。故改為手動：每列內由左至右水平 join（同列高已相等），
    再把各列由上至下垂直 join（各列總寬已相等）——可逐像素還原原始版面。

    與 ``_stitch_overlay_slide`` 拆開，是為了讓 ``scripts/stitch_probe.py`` 能在**同一張
    已拼好的影像**上獨立消融 ``tiffsave`` 的各個參數（doc 26 Tier 1.1）：join 與 encode
    的成本必須分開讀，否則量到的差異分不清是哪一段造成的。
    """
    overlay_dir = output_dir / _STITCH_SCRATCH
    xs = sorted(geometry.col_of)  # abs_x，欄序（升冪即欄索引序）
    ys = sorted(geometry.row_of)  # abs_y，列序

    _ensure_nofile_limit(len(xs) * len(ys) + 256)

    row_images: List[pyvips.Image] = []
    for ay in ys:
        row_tiles: List[pyvips.Image] = []
        for ax in xs:
            path = overlay_dir / f"tile_x{ax}_y{ay}.tiff"
            if not path.exists():
                raise FileNotFoundError(
                    f"{_STITCH_SCRATCH} 缺少 tile: {path}——每格應恰有一檔。"
                )
            row_tiles.append(
                pyvips.Image.new_from_file(str(path), access="sequential")
            )
        row = row_tiles[0]
        for tile in row_tiles[1:]:
            row = row.join(tile, "horizontal", expand=True)
        row_images.append(row)

    slide = row_images[0]
    for row in row_images[1:]:
        slide = slide.join(row, "vertical", expand=True)
    return slide


def _stitch_overlay_slide(output_dir: Path, geometry: TileGeometry) -> None:
    """把 ``_stitch_scratch/`` 內的 tile 拼成一張全片 pyramid TIFF 並寫出，寫完刪暫存夾。

    壓縮採 **lzw（無失真）**：這是帶細胞邊界線 / 標籤文字 / 紅黑點的標註影像，JPEG
    的區塊假影會糊掉細線與小點，醫師判讀不宜；lzw 保真且仍可壓。

    ``tiffsave`` 的其餘參數（tile 大小、pyramid 深度、predictor）已在真實 16.22 GP
    規模上逐一消融過，沒有一個換得到值得留下的時間——見
    ``docs/hybrid-pipeline/27-remaining-work-implementation.md`` §3 與
    ``scripts/stitch_probe.py --ablate``。故此處維持 pyvips 預設，不加旋鈕：零貢獻的
    層一律不留（playbook step 4）。

    ``tiffsave`` 是同步的——回傳時惰性管線已把每格讀完寫出，故其後即可安全刪除暫存夾。
    刪除**不放 finally**：拼合失敗時保留暫存夾，才能重跑縫合而不必重算整批。
    """
    overlay_dir = output_dir / _STITCH_SCRATCH
    slide = _join_overlay_tiles(output_dir, geometry)
    slide.tiffsave(
        str(output_dir / "overlay_slide.tiff"),
        tile=True,
        pyramid=True,
        compression="lzw",
        bigtiff=True,
    )
    logger.info(
        "overlay_slide.tiff 縫合完成: %d×%d px", slide.width, slide.height
    )
    if overlay_dir.exists():
        shutil.rmtree(overlay_dir)


def _log_batch_summary(
    run_id: str,
    stats: dict,
    total: int,
) -> None:
    """輸出批次處理摘要。"""
    logger.info(
        "批次完成 — run_id=%s | 總計=%d | 成功=%d | 跳過=%d",
        run_id,
        total,
        stats["success"],
        stats["skipped"],
    )


# ------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """建構命令列參數解析器。"""
    parser = argparse.ArgumentParser(
        description="IHC-DISH Overlay & Analysis Pipeline",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="跑內建 test_picture ROI 範例（走完整 precut+分析流程）",
    )
    parser.add_argument(
        "--ihc",
        type=str,
        default=None,
        help="單 tile 模式: IHC tile 檔名或路徑",
    )
    parser.add_argument(
        "--dish",
        type=str,
        default=None,
        help="單 tile 模式: DISH tile 檔名或路徑",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="輸出目錄 (預設: config.output_dir)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="跨 tile 平行的行程數（預設 1 = 單行程）。>1 尚未通過整片驗證，"
             "見 docs/hybrid-pipeline/19-open-backlog.md #7",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="啟用斷點續跑：每塊完成即落地到 output/_resume/，重跑時跳過已完成的塊。"
             "給無人看顧的整片跑用",
    )
    return parser


def main() -> None:
    """CLI 主入口。"""
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else config.output_dir

    if args.test:
        _run_single_tile_cli(
            str(config.ihc_test_path), str(config.dish_test_path), output_dir,
            workers=args.workers, checkpoint=args.resume,
        )
    elif args.ihc and args.dish:
        _run_single_tile_cli(args.ihc, args.dish, output_dir,
                             workers=args.workers, checkpoint=args.resume)
    else:
        parser.print_help()


def _run_single_tile_cli(
    ihc_arg: str,
    dish_arg: str,
    output_dir: Path,
    workers: int = 1,
    checkpoint: bool = False,
) -> None:
    """CLI 單一 ROI/WSI 影像對模式：預切成重疊 tile 檔並與 ``run_batch`` 重疊執行。

    內部分塊已移除，故以 ``PrecutStream`` 把整對影像切到
    ``output_dir/_precut_scratch/{ihc,dish}``（保留供檢查，不自動清理）。格線只讀檔頭
    即可算出，故先交出幾何、tile 檔隨切隨送進 ``run_batch``，切塊不再是分析開始前的一段
    序列等待（large 錨點實測省下約 75% 的預切時間）。``run_batch`` 照常做逐塊分析 +
    全域合併 + slide 級縫合。
    """
    ihc_path = _resolve_tile_path(ihc_arg, config.ihc_tile_dir)
    dish_path = _resolve_tile_path(dish_arg, config.dish_tile_dir)

    output_dir = Path(output_dir)
    scratch = output_dir / "_precut_scratch"
    ihc_out = scratch / "ihc"
    dish_out = scratch / "dish"
    logger.info("單圖模式：預切 %s / %s → %s（與分析迴圈重疊）", ihc_path.name,
                dish_path.name, scratch)
    stream = PrecutStream(
        ihc_path,
        dish_path,
        ihc_out,
        dish_out,
        tile_size=config.default_tile_size,
        overlap=config.window_overlap_px,
    )

    run_batch(ihc_out, dish_out, output_dir, tile_stream=stream,
              workers=workers, checkpoint=checkpoint)


def _resolve_tile_path(arg: str, default_dir: Path) -> Path:
    """將 CLI 引數解析為完整路徑。"""
    candidate = Path(arg)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    resolved = default_dir / arg
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"找不到 tile: {arg} (搜尋: {resolved})")


if __name__ == "__main__":
    main()
