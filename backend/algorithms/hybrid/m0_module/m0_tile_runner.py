"""
M0 逐塊執行層：單一 precut tile 的 GPU 前段 / CPU 後段。

模型初始化（UNet++ / 兩個 Cellpose）、單塊的 M1→M2→M3b 前向與其後的純 CPU 尾段，
以及逐塊 overlay 的落地。由 ``m0_multiprocess`` 的 worker 與 ``hybrid_pipeline``
的單行程迴圈共用。
"""
import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from torch import sparse
from skimage import io

# Cellpose 在 get_masks_torch 內呼叫 torch.sparse_coo_tensor；PyTorch 要求顯式表態
# sparse invariant 檢查的開關，否則會持續發出 UserWarning。維持預設 (關閉檢查) 以保留效能。
sparse.check_sparse_tensor_invariants.disable()

try:
    from ..config import config
    from ..m1_overlay import (
        apply_mask_to_ihc_image,
        fuse_masked_ihc_with_dish,
        generate_ihc_core_mask,
        overlay_ihc_mask_on_dish,
    )
    from ..m2_segmentation import (
        CellposeSegmenter,
        segment_windowed,
    )
    from ..m3_cell_detection import (
        CellAnalysisResult,
        build_all_positive_results,
        detect_all_dots,
        enlarge_cell_instances,
        merge_dot_results_to_cell_analysis,
    )
    from ..m4_export import (
        draw_tile_seam_edges,
        render_overlay_image,
    )
except ImportError:
    from config import config
    from m1_overlay import (
        apply_mask_to_ihc_image,
        fuse_masked_ihc_with_dish,
        generate_ihc_core_mask,
        overlay_ihc_mask_on_dish,
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
        render_overlay_image,
    )

from .m0_stitch import (
    _STITCH_SCRATCH,
    ChunkResult,
    TileGeometry,
    clear_slide_edge_cells,
    core_crop_bounds,
    filter_and_absolutize,
)

logger = logging.getLogger(__name__)


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
        from ..unet_inference import UNetPPInference  # noqa: WPS433
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
