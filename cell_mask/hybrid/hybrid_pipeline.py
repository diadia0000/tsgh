"""
IHC-DISH Overlay & Analysis Pipeline — 主入口

串接 M1 (Overlay) → M2 (Segmentation) → M3 (Cell Results) → M4 (Export)。
支援單 tile 處理與批次掃描。

流程:
  - M1: IHC → UNet++ mask → mask on IHC & DISH → 50/50 alpha blend
  - M2: Cellpose 分割 IHC-DISH 疊合影像 → cell instance mask
  - M3: 將 M2 cell mask 套用至 dish_mask_overlay → 逐細胞結果
  - M4: CSV + overlay 視覺化 + 固定 256×256 逐細胞裁切

Usage:
    # 單 tile
    python hybrid_pipeline.py --ihc tile_x1024_y2048.tiff --dish tile_x1024_y2048.tiff

    # 批次 (目錄掃描)
    python hybrid_pipeline.py --batch

    # 使用 test_picture 資料夾
    python hybrid_pipeline.py --batch --test
"""

import argparse
import logging
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from torch import sparse
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
    segment_masked_dish,
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
    export_cell_dot_annotations,
    export_overlay_visualization,
    stamp_grid_on_overlays,
)
from m0_reader import iter_paired_chunks, read_size
from m0_stitch import ChunkResult, clear_slide_edge_cells, stitch_chunks

# ------------------------------------------------------------------
# Logging 設定
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class M1StageArtifacts:
    """M1 輸出影像與遮罩，供後續 M2/M3/M4 共用。"""

    core_mask: np.ndarray
    masked_ihc: np.ndarray
    dish_mask_overlay: np.ndarray
    overlay_image: np.ndarray
    m2_input_overlay: np.ndarray


@dataclass
class M3StageArtifacts:
    """M3 點位偵測與逐細胞分析輸出。"""

    results: List[CellAnalysisResult]
    all_dots: list
    per_cell_dots: dict
    dish_nucleus_mask: np.ndarray


# ------------------------------------------------------------------
# 模型初始化
# ------------------------------------------------------------------

def _init_unet_inferencer():
    """延遲初始化 UNet++ 推論器。"""
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
        batch_size=getattr(config, "cellpose_batch_size", 16),
        gpu=config.cellpose_gpu,
    )


def _init_dish_cellpose_segmenter() -> CellposeSegmenter:
    """延遲初始化 DISH 細胞核偵測 Cellpose 分割器（M3b 多核排除用）。"""
    return CellposeSegmenter(
        model_path=config.cellpose_dish_model_path,
        diameter=config.cellpose_dish_diameter,
        flow_threshold=config.cellpose_dish_flow_threshold,
        cellprob_threshold=config.cellpose_dish_cellprob_threshold,
        batch_size=getattr(config, "cellpose_batch_size", 16),
        gpu=config.cellpose_gpu,
    )


# ------------------------------------------------------------------
# 單 tile 處理
# ------------------------------------------------------------------

def process_single_tile(
    ihc_tile_path: Path,
    dish_tile_path: Path,
    unet_inferencer: object,
    cellpose_segmenter: CellposeSegmenter,
    dish_cellpose_segmenter: CellposeSegmenter,
    output_dir: Path,
    cfg_hash: str,
    merge_dir: Optional[Path] = None,
    tile_id: Optional[str] = None,
) -> Optional[List[CellAnalysisResult]]:
    """處理單一配對 tile 的完整流水線。

    Args:
        ihc_tile_path: IHC tile 影像路徑。
        dish_tile_path: DISH tile 影像路徑。
        unet_inferencer: 已初始化的 UNetPPInference。
        cellpose_segmenter: 已初始化的 CellposeSegmenter（M2 IHC-DISH 細胞分割）。
        dish_cellpose_segmenter: 已初始化的 CellposeSegmenter（M3b DISH 核偵測）。
        output_dir: 輸出目錄。
        cfg_hash: 配置雜湊。
        merge_dir: 合併影像目錄 (可選)。
        tile_id: 輸出子目錄/檔名前綴。None 時取 ``dish_tile_path.stem``。

    Returns:
        分類結果列表，失敗時回傳 None。
    """
    if tile_id is None:
        tile_id = dish_tile_path.stem
    tile_output = output_dir / tile_id
    start_time = time.perf_counter()

    tile_size = config.default_tile_size
    overlap = config.window_overlap_px

    # M0：以 pyvips 逐塊讀取（不整載），對每塊跑 M1–M3，最後縫成 slide-level 輸出。
    # 尺寸/配對驗證由 iter_paired_chunks 負責（IHC/DISH 同尺寸、邊長 ≥ tile_size）。
    try:
        full_h, full_w = read_size(dish_tile_path)
    except Exception as exc:
        logger.error("Tile %s 讀取尺寸失敗: %s", tile_id, exc)
        return None

    try:
        chunk_results: List[ChunkResult] = []
        for chunk in iter_paired_chunks(ihc_tile_path, dish_tile_path, tile_size, overlap):
            cr = _process_one_chunk(
                chunk, full_h, full_w,
                unet_inferencer, cellpose_segmenter, dish_cellpose_segmenter,
            )
            if cr is not None:
                chunk_results.append(cr)

        if not chunk_results:
            logger.warning("Tile %s: 全部分塊核心遮罩皆空 — 僅匯出空 CSV", tile_id)
            _export_empty_tile(
                tile_output, tile_id, cfg_hash,
                np.zeros((full_h, full_w), dtype=np.uint8),
            )
            return []

        stitched = stitch_chunks(
            chunk_results, full_h, full_w, overlap,
            background_fill_value=config.background_fill_value,
        )

        m1_artifacts = M1StageArtifacts(
            core_mask=stitched.core_mask,
            masked_ihc=stitched.masked_ihc,
            dish_mask_overlay=stitched.dish_mask_overlay,
            overlay_image=stitched.overlay_image,
            m2_input_overlay=stitched.overlay_image,
        )
        _write_m1_artifacts(tile_output, tile_id, m1_artifacts)

        io.imsave(
            str(tile_output / f"{tile_id}_m2_cell_instance_binary.png"),
            ((stitched.instance_mask > 0).astype(np.uint8) * 255),
            check_contrast=False,
        )

        m3_artifacts = M3StageArtifacts(
            results=stitched.results,
            all_dots=stitched.all_dots,
            per_cell_dots=stitched.per_cell_dots,
            dish_nucleus_mask=stitched.dish_nucleus_mask,
        )

        _export_tile_outputs(
            tile_output,
            tile_id,
            cfg_hash,
            m1_artifacts,
            stitched.instance_mask,
            m3_artifacts,
            merge_dir,
        )

        elapsed = time.perf_counter() - start_time
        pos_count = sum(1 for r in m3_artifacts.results if r.is_her2_positive)
        logger.info(
            "Tile %s 處理完成: %d 細胞 (%d 陽性), %d 分塊, %.2f 秒",
            tile_id,
            len(m3_artifacts.results),
            pos_count,
            len(chunk_results),
            elapsed,
        )
        return m3_artifacts.results

    except ValueError as exc:
        logger.error("Tile %s 維度錯誤: %s", tile_id, exc)
        return None
    except Exception as exc:
        logger.error("Tile %s 處理失敗: %s", tile_id, exc, exc_info=True)
        return None


def _process_one_chunk(
    chunk,
    full_h: int,
    full_w: int,
    unet_inferencer: object,
    cellpose_segmenter: CellposeSegmenter,
    dish_cellpose_segmenter: CellposeSegmenter,
) -> Optional[ChunkResult]:
    """單一分塊 M1→M2→M3；回傳帶絕對座標的 ``ChunkResult``，核心遮罩全空則 None。

    與整圖路徑的差異僅在「清邊」：M2 不在分塊內部接縫清邊（``remove_border=False``），
    改在 M3 之前只清「碰到真實 slide 外緣」的細胞；跨塊重複偵測由縫合層的質心
    core-ownership 去重。單塊時四邊皆真實 slide 邊 → 等同現行 M2 清邊行為。
    """
    th, tw = chunk.ihc.shape[:2]
    core_mask = generate_ihc_core_mask(  # pyright: ignore[reportArgumentType]
        chunk.ihc,
        unet_inferencer,
        close_kernel=config.core_close_kernel,
    )
    if core_mask.sum() == 0:
        return None

    m1 = _run_m1_overlay_stage(chunk.ihc, chunk.dish, core_mask)

    instance_mask = segment_masked_dish(
        m1.m2_input_overlay,
        cellpose_segmenter,
        remove_border=False,
        tile_size=config.default_tile_size,
        overlap=config.window_overlap_px,
        dedup_iomin=config.window_dedup_iomin,
    )
    if config.clear_border_cells:
        instance_mask = clear_slide_edge_cells(
            instance_mask,
            clear_top=(chunk.abs_y == 0),
            clear_bottom=(chunk.abs_y + th >= full_h),
            clear_left=(chunk.abs_x == 0),
            clear_right=(chunk.abs_x + tw >= full_w),
        )

    m3 = _run_m3_analysis_stage(
        chunk.dish,
        m1.dish_mask_overlay,
        instance_mask,
        m1.core_mask,
        dish_cellpose_segmenter,
    )

    return ChunkResult(
        abs_x=chunk.abs_x,
        abs_y=chunk.abs_y,
        instance_mask=instance_mask,
        dish_nucleus_mask=m3.dish_nucleus_mask,
        core_mask=core_mask,
        masked_ihc=m1.masked_ihc,
        dish_mask_overlay=m1.dish_mask_overlay,
        overlay_image=m1.overlay_image,
        results=m3.results,
        all_dots=m3.all_dots,
        per_cell_dots=m3.per_cell_dots,
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
        masked_ihc=masked_ihc,
        dish_mask_overlay=dish_mask_overlay,
        overlay_image=overlay_image,
        m2_input_overlay=overlay_image,
    )


def _write_m1_artifacts(
    tile_output: Path,
    tile_id: str,
    artifacts: M1StageArtifacts,
) -> None:
    """落地 M1 中間結果，保留既有檔名以維持相容。"""
    tile_output.mkdir(parents=True, exist_ok=True)
    io.imsave(
        str(tile_output / f"{tile_id}_ihc_core_mask.png"),
        (artifacts.core_mask * 255).astype(np.uint8),
        check_contrast=False,
    )
    io.imsave(
        str(tile_output / f"{tile_id}_masked_ihc.png"),
        artifacts.masked_ihc,
        check_contrast=False,
    )
    io.imsave(
        str(tile_output / f"{tile_id}_ihc_tumor.png"),
        artifacts.masked_ihc,
        check_contrast=False,
    )
    io.imsave(
        str(tile_output / f"{tile_id}_dish_mask_overlay.png"),
        artifacts.dish_mask_overlay,
        check_contrast=False,
    )
    io.imsave(
        str(tile_output / f"{tile_id}_dish_tumor.png"),
        artifacts.dish_mask_overlay,
        check_contrast=False,
    )
    io.imsave(
        str(tile_output / f"{tile_id}_ihc_dish_overlay_raw.png"),
        artifacts.overlay_image,
        check_contrast=False,
    )
    io.imsave(
        str(tile_output / f"{tile_id}_m2_input_overlay.png"),
        artifacts.m2_input_overlay,
        check_contrast=False,
    )


def _run_m3_analysis_stage(
    dish_image: np.ndarray,
    dish_mask_overlay: np.ndarray,
    instance_mask: np.ndarray,
    core_mask: np.ndarray,
    dish_cellpose_segmenter: CellposeSegmenter,
) -> M3StageArtifacts:
    """執行 M3：逐細胞結果、DISH 核偵測、紅黑點偵測與結果合併。"""
    results = build_all_positive_results(instance_mask)
    # M3 配對前處理：把綠色細胞 mask 實際放大（面積 ×cell_enlarge_area_factor），讓細胞
    # 蓋到更多 DISH 核以提高配對成功率。放大版僅供配對 / 點偵測；原始 instance_mask 仍
    # 用於 M4 視覺化與裁切，醫師看到的綠框維持不變。
    matching_mask = enlarge_cell_instances(instance_mask, config)
    dish_nucleus_mask = segment_windowed(
        dish_image,
        dish_cellpose_segmenter,
        tile_size=config.default_tile_size,
        overlap=config.window_overlap_px,
        dedup_iomin=config.window_dedup_iomin,
    )
    all_dots, per_cell_dots, dish_nucleus_mask = detect_all_dots(
        dish_mask_overlay,
        matching_mask,
        config,
        dish_nucleus_mask=dish_nucleus_mask,
        core_mask=core_mask,
    )
    results = merge_dot_results_to_cell_analysis(results, per_cell_dots)
    return M3StageArtifacts(
        results=results,
        all_dots=all_dots,
        per_cell_dots=per_cell_dots,
        dish_nucleus_mask=dish_nucleus_mask,
    )


def _export_empty_tile(
    tile_output: Path,
    tile_id: str,
    cfg_hash: str,
    core_mask: np.ndarray,
) -> None:
    """核心遮罩全空時，只輸出空的 M4 報表與空底圖。"""
    export_cell_dot_annotations(
        np.zeros((core_mask.shape[0], core_mask.shape[1], 3), dtype=np.uint8),
        np.zeros(core_mask.shape, dtype=np.int32),
        [],
        tile_output,
        slide_id=config.slide_id,
        tile_id=tile_id,
        model_version=config.model_version,
        config_hash=cfg_hash,
        crop_size=config.cell_crop_size,
    )


def _export_tile_outputs(
    tile_output: Path,
    tile_id: str,
    cfg_hash: str,
    m1_artifacts: M1StageArtifacts,
    instance_mask: np.ndarray,
    m3_artifacts: M3StageArtifacts,
    merge_dir: Optional[Path],
) -> None:
    """執行 M4 輸出、醫師檢視圖、merge overlay 與 sliding-window 格線。"""
    export_cell_dot_annotations(
        m1_artifacts.dish_mask_overlay,
        instance_mask,
        m3_artifacts.results,
        tile_output,
        visualization_image=m1_artifacts.dish_mask_overlay,
        slide_id=config.slide_id,
        tile_id=tile_id,
        model_version=config.model_version,
        config_hash=cfg_hash,
        crop_size=config.cell_crop_size,
        all_dots=m3_artifacts.all_dots,
        per_cell_dots=m3_artifacts.per_cell_dots,
        dish_nucleus_mask=m3_artifacts.dish_nucleus_mask,
    )

    export_overlay_visualization(
        m1_artifacts.overlay_image,
        instance_mask,
        m3_artifacts.results,
        tile_output / f"{tile_id}_ihc_dish_overlay.png",
        all_dots=m3_artifacts.all_dots,
    )

    _export_merge_overlay(
        merge_dir,
        tile_id,
        tile_output,
        instance_mask,
        m3_artifacts,
    )

    if config.draw_window_grid:
        stamp_grid_on_overlays(tile_output, tile_size=config.default_tile_size)


def _export_merge_overlay(
    merge_dir: Optional[Path],
    tile_id: str,
    tile_output: Path,
    instance_mask: np.ndarray,
    m3_artifacts: M3StageArtifacts,
) -> None:
    """若存在對應 merge 影像，將 cellpose 邊界繪製在原始合併影像上。"""
    merge_tile_path = _find_merge_tile(merge_dir, tile_id)
    if merge_tile_path is None:
        return

    merge_image = _read_rgb(merge_tile_path)
    if merge_image.shape[:2] == instance_mask.shape[:2]:
        export_overlay_visualization(
            merge_image,
            instance_mask,
            m3_artifacts.results,
            tile_output / f"{tile_id}_merge_overlay.png",
            all_dots=m3_artifacts.all_dots,
        )
        logger.info("Merge overlay 匯出完成: %s", tile_id)
        return

    logger.warning(
        "Tile %s: merge 影像尺寸 %s 與 mask 尺寸 %s 不匹配，跳過 merge overlay",
        tile_id, merge_image.shape[:2], instance_mask.shape[:2],
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
# 批次處理
# ------------------------------------------------------------------

def run_batch(
    ihc_dir: Path,
    dish_dir: Path,
    output_dir: Path,
    merge_dir: Optional[Path] = None,
) -> dict:
    """批次掃描目錄並處理所有配對 tile。

    Args:
        ihc_dir: IHC tile 目錄。
        dish_dir: DISH tile 目錄。
        output_dir: 輸出根目錄。
        merge_dir: 合併影像目錄 (可選)，用於產出 merge overlay。

    Returns:
        ``{"success": int, "failed": int, "skipped": int}`` 統計。
    """
    run_id = uuid.uuid4().hex[:8]
    cfg_hash = compute_config_hash(config)
    logger.info(
        "批次處理開始 — run_id=%s, config_hash=%s", run_id, cfg_hash
    )

    paired_tiles = find_paired_tiles(
        ihc_dir, dish_dir, config.supported_extensions
    )

    if not paired_tiles:
        logger.warning("未找到任何配對 tile")
        return {"success": 0, "failed": 0, "skipped": 0}

    unet = _init_unet_inferencer()
    cellpose = _init_cellpose_segmenter()
    dish_cellpose = _init_dish_cellpose_segmenter()

    stats = {"success": 0, "failed": 0, "skipped": 0}
    total = len(paired_tiles)

    for idx, (ihc_path, dish_path) in enumerate(paired_tiles, start=1):
        logger.info(
            "[%d/%d] 處理 tile: %s", idx, total, dish_path.stem
        )
        try:
            result = process_single_tile(
                ihc_path, dish_path, unet, cellpose, dish_cellpose, output_dir,
                cfg_hash, merge_dir=merge_dir,
            )
        except Exception as exc:
            # patch 規格不符（ValueError）或前置讀檔失敗：記錄後跳過，批次續跑。
            logger.error("Tile %s 前置/規格檢查失敗，跳過: %s", dish_path.stem, exc)
            stats["failed"] += 1
            continue
        if result is None:
            stats["failed"] += 1
        elif len(result) == 0:
            stats["skipped"] += 1
        else:
            stats["success"] += 1

    # TODO(slide-level): 若需整體玻片統計，於此聚合 per-tile summary。
    # 作法：讓 process_single_tile() 額外回傳 DotStatsSummary，收集到 list 後：
    #   from cell_mask.hybrid.m4_export import DotStatsSummary, write_summary_csv
    #   slide_stats = DotStatsSummary.aggregate(per_tile_summaries)
    #   write_summary_csv(slide_stats, output_dir / f"{run_id}_slide_summary.csv")

    _log_batch_summary(run_id, stats, total)
    return stats


def _log_batch_summary(
    run_id: str,
    stats: dict,
    total: int,
) -> None:
    """輸出批次處理摘要。"""
    logger.info(
        "批次完成 — run_id=%s | 總計=%d | 成功=%d | 跳過=%d | 失敗=%d",
        run_id,
        total,
        stats["success"],
        stats["skipped"],
        stats["failed"],
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
        "--batch",
        action="store_true",
        help="啟用批次掃描模式",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="使用 test_picture 目錄 (搭配 --batch)",
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
    return parser


def main() -> None:
    """CLI 主入口。"""
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else config.output_dir

    if args.batch:
        ihc_dir = config.ihc_test_dir if args.test else config.ihc_tile_dir
        dish_dir = config.dish_test_dir if args.test else config.dish_tile_dir
        merge_dir = config.merge_test_dir if args.test else config.merge_tile_dir
        run_batch(ihc_dir, dish_dir, output_dir, merge_dir=merge_dir)
    elif args.ihc and args.dish:
        _run_single_tile_cli(args.ihc, args.dish, output_dir)
    else:
        parser.print_help()


def _run_single_tile_cli(
    ihc_arg: str,
    dish_arg: str,
    output_dir: Path,
) -> None:
    """CLI 單 tile 模式入口。"""
    ihc_path = _resolve_tile_path(ihc_arg, config.ihc_tile_dir)
    dish_path = _resolve_tile_path(dish_arg, config.dish_tile_dir)

    cfg_hash = compute_config_hash(config)
    unet = _init_unet_inferencer()
    cellpose = _init_cellpose_segmenter()
    dish_cellpose = _init_dish_cellpose_segmenter()

    process_single_tile(
        ihc_path, dish_path, unet, cellpose, dish_cellpose, output_dir, cfg_hash,
        merge_dir=config.merge_tile_dir,
    )


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
