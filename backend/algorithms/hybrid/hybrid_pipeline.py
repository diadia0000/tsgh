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
    # 單一 ROI/WSI 影像對：先預切成重疊 tile 檔（暫存於 output/_precut_scratch），
    # 再走與 --batch 相同的逐塊分析 + slide 級縫合流程。
    python hybrid_pipeline.py --ihc roi_ihc.tiff --dish roi_dish.tiff

    # 批次：ihc_dir / dish_dir 為「已預切」的 tile 檔目錄（tile_x{int}_y{int}）。
    python hybrid_pipeline.py --batch

    # 使用 test_picture 資料夾（同為已預切 tile 目錄）
    python hybrid_pipeline.py --batch --test
"""

import argparse
import gc
import logging
import sys
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyvips
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
        segment_masked_dish,
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
        export_per_cell_images,
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
    from .m0_reader import precut_paired_tiles
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
        export_per_cell_images,
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
    from m0_reader import precut_paired_tiles

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

def process_precut_tile(
    ihc_tile_path: Path,
    dish_tile_path: Path,
    abs_x: int,
    abs_y: int,
    geometry: TileGeometry,
    unet_inferencer: object,
    cellpose_segmenter: CellposeSegmenter,
    dish_cellpose_segmenter: CellposeSegmenter,
    output_dir: Path,
    merge_dir: Optional[Path] = None,
) -> Optional[List[CellAnalysisResult]]:
    """處理單一『已預切』tile：M1→M2→M3，逐塊落地產物，回傳核心區擁有的細胞清單。

    落地產物（皆以 ``tile_x{abs_x}_y{abs_y}`` 命名，分別置於 ``output_dir`` 下各子夾）：
      - 5 張非 overlay 陣列（原樣、局部 ID、未過濾、未重編號）:
        ``core_mask/`` ``masked_ihc/`` ``dish_mask_overlay/``（PNG）、
        ``instance_mask/`` ``dish_nucleus_mask/``（int32 TIFF，保留 label）。
      - ``overlay_annotated/``: 以 ``overlay_image`` 為底繪細胞邊界/標籤/dots，核心裁切後
        存 TIFF，供後續 pyvips arrayjoin 拼成 slide 級 QuPath 影像（另一支任務負責）。
      - ``cell_crops/tile_x{x}_y{y}/cells/``: 核心擁有細胞的固定尺寸裁切。
      - ``merge_overlay/``（可選）: 若 ``merge_dir`` 有同名 tile，畫邊界後核心裁切存 TIFF。

    表格 CSV / summary 不在此產出——回傳的 ``owned``（質心已絕對化、``cell_id`` 仍為
    分塊局部）由 batch driver 收集後全域合併去重再寫出。

    Returns:
        - 正常（含 0 細胞）→ 該塊核心區擁有、質心絕對化的 ``CellAnalysisResult`` 清單。
        - 核心遮罩全空的背景塊 → 仍寫空白 placeholder（維持 arrayjoin 每格一檔），回傳 ``[]``。
        - 讀檔 / 維度等真實錯誤 → ``None``。
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

    th, tw = ihc.shape[:2]
    lx0, lx1, ly0, ly1 = core_crop_bounds(
        geometry, abs_x, abs_y, config.default_tile_size
    )

    try:
        cr = _process_one_chunk(
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

    if cr is None:
        # 背景塊（核心遮罩全空）：仍寫空白 placeholder，讓 arrayjoin 每格恰有一檔。
        _write_blank_tile(output_dir, tile_name, th, tw, (ly1 - ly0, lx1 - lx0))
        logger.info("Tile %s: 核心遮罩全空 → 空白 placeholder", tile_name)
        return []

    owned = filter_and_absolutize(cr, geometry, abs_x, abs_y)
    owned_ids = {r.cell_id for r in owned}
    local_owned = [r for r in cr.results if r.cell_id in owned_ids]

    # (a) 5 張非 overlay 陣列：原樣、全塊局部、未過濾、未重編號。
    _save_tile_array(
        output_dir / "core_mask" / f"{tile_name}.png",
        (cr.core_mask * 255).astype(np.uint8),
    )
    _save_tile_array(output_dir / "masked_ihc" / f"{tile_name}.png", cr.masked_ihc)
    _save_tile_array(
        output_dir / "dish_mask_overlay" / f"{tile_name}.png", cr.dish_mask_overlay
    )
    _save_tile_array(
        output_dir / "instance_mask" / f"{tile_name}.tiff",
        cr.instance_mask.astype(np.int32, copy=False),
    )
    _save_tile_array(
        output_dir / "dish_nucleus_mask" / f"{tile_name}.tiff",
        cr.dish_nucleus_mask.astype(np.int32, copy=False),
    )

    # (b) 標註 overlay（醫師 / slide 級 QuPath），以 overlay_image 為底畫全塊、核心裁切。
    #     以全塊 results/instance_mask/all_dots 繪製後裁核心：核心區彼此無重疊、無縫隙，
    #     故每個標註像素在拼回的整片中恰出現一次。
    annotated = render_overlay_image(
        cr.overlay_image, cr.instance_mask, cr.results, all_dots=cr.all_dots
    )
    _save_tile_array(
        output_dir / "overlay_annotated" / f"{tile_name}.tiff",
        annotated[ly0:ly1, lx0:lx1],
    )

    # (c) per-cell crop：核心擁有子集（局部座標）；per-chunk 子夾避免 cell_id 撞名。
    export_per_cell_images(
        cr.dish_mask_overlay,
        cr.instance_mask,
        local_owned,
        output_dir / "cell_crops" / tile_name,
        crop_size=config.cell_crop_size,
        per_cell_dots=cr.per_cell_dots,
        dish_nucleus_mask=cr.dish_nucleus_mask,
    )

    # (d) 可選 merge overlay。
    if merge_dir is not None:
        _export_chunk_merge_overlay(
            merge_dir, output_dir, tile_name, cr, (lx0, lx1, ly0, ly1)
        )

    elapsed = time.perf_counter() - start_time
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
    th: int,
    tw: int,
    crop_hw: tuple,
) -> None:
    """核心遮罩全空的背景塊：為每項輸出寫一張空白 placeholder，維持每格一檔。"""
    fill = config.background_fill_value
    ch, cw = crop_hw
    _save_tile_array(
        output_dir / "core_mask" / f"{tile_name}.png",
        np.zeros((th, tw), dtype=np.uint8),
    )
    _save_tile_array(
        output_dir / "masked_ihc" / f"{tile_name}.png",
        np.full((th, tw, 3), fill, dtype=np.uint8),
    )
    _save_tile_array(
        output_dir / "dish_mask_overlay" / f"{tile_name}.png",
        np.full((th, tw, 3), fill, dtype=np.uint8),
    )
    _save_tile_array(
        output_dir / "instance_mask" / f"{tile_name}.tiff",
        np.zeros((th, tw), dtype=np.int32),
    )
    _save_tile_array(
        output_dir / "dish_nucleus_mask" / f"{tile_name}.tiff",
        np.zeros((th, tw), dtype=np.int32),
    )
    _save_tile_array(
        output_dir / "overlay_annotated" / f"{tile_name}.tiff",
        np.full((ch, cw, 3), fill, dtype=np.uint8),
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


def _process_one_chunk(
    ihc: np.ndarray,
    dish: np.ndarray,
    abs_x: int,
    abs_y: int,
    edge_flags: tuple,
    unet_inferencer: object,
    cellpose_segmenter: CellposeSegmenter,
    dish_cellpose_segmenter: CellposeSegmenter,
) -> Optional[ChunkResult]:
    """單一分塊 M1→M2→M3；回傳帶絕對座標左上角的 ``ChunkResult``，核心遮罩全空則 None。

    ``edge_flags`` = ``(clear_top, clear_bottom, clear_left, clear_right)``（由
    ``TileGeometry.edge_flags`` 提供）：M2 不在分塊內部接縫清邊（``remove_border=False``），
    改在 M3 之前只清「碰到真實 slide 外緣」的細胞；跨塊重複偵測由縫合層的質心
    core-ownership 去重。單塊時四邊皆真實 slide 邊 → 等同現行 M2 清邊行為。
    """
    core_mask = generate_ihc_core_mask(  # pyright: ignore[reportArgumentType]
        ihc,
        unet_inferencer,
        close_kernel=config.core_close_kernel,
    )
    if core_mask.sum() == 0:
        return None

    m1 = _run_m1_overlay_stage(ihc, dish, core_mask)

    instance_mask = segment_masked_dish(
        m1.m2_input_overlay,
        cellpose_segmenter,
        remove_border=False,
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

    m3 = _run_m3_analysis_stage(
        dish,
        m1.dish_mask_overlay,
        instance_mask,
        m1.core_mask,
        dish_cellpose_segmenter,
    )

    return ChunkResult(
        abs_x=abs_x,
        abs_y=abs_y,
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

    Returns:
        ``{"success": int, "failed": int, "skipped": int}`` 統計
        （``failed`` 於 fail-fast 下恆為 0——真正失敗會 raise）。
    """
    run_id = uuid.uuid4().hex[:8]
    cfg_hash = compute_config_hash(config)
    output_dir = Path(output_dir)
    logger.info(
        "批次處理開始 — run_id=%s, config_hash=%s", run_id, cfg_hash
    )

    paired_tiles = find_paired_tiles(
        ihc_dir, dish_dir, config.supported_extensions
    )

    if not paired_tiles:
        logger.warning("未找到任何配對 tile")
        return {"success": 0, "failed": 0, "skipped": 0}

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

    # 一次算出縫合幾何；格線不完整（缺格 / 重複 / 對不上）會 raise ValueError，
    # 屬預期的 fail-fast 驗證，不吞。
    geometry = compute_tile_geometry(
        positions, config.default_tile_size, config.window_overlap_px
    )

    unet = _init_unet_inferencer()
    cellpose = _init_cellpose_segmenter()
    dish_cellpose = _init_dish_cellpose_segmenter()

    stats = {"success": 0, "failed": 0, "skipped": 0}
    total = len(paired_tiles)
    # 每塊回傳 (abs_x, abs_y, owned_results)，供迴圈後全域排序 / 重編號。
    per_tile_owned: List[Tuple[int, int, List[CellAnalysisResult]]] = []

    # 序列（非平行）迴圈：三個 GPU 模型只在主行程 / CUDA context 載入一次並重用，
    # 真正的加速要靠跨 tile 批次化 GPU 推論——那是另案追蹤、刻意延後的優化，不在本輪
    # （本輪只求新架構的正確性，不碰吞吐）。tile 間保留 empty_cache + gc.collect 清理。
    for idx, ((ihc_path, dish_path), (ax, ay)) in enumerate(
        zip(paired_tiles, positions), start=1
    ):
        logger.info(
            "[%d/%d] 處理 tile: %s", idx, total, dish_path.stem
        )
        result = process_precut_tile(
            ihc_path, dish_path, ax, ay, geometry,
            unet, cellpose, dish_cellpose, output_dir, merge_dir=merge_dir,
        )
        if result is None:
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
        per_tile_owned.append((ax, ay, result))
        if len(result) == 0:
            stats["skipped"] += 1
        else:
            stats["success"] += 1

        # Release GPU allocator cache and run Python GC between tiles to
        # prevent monotonic memory growth across a long batch.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()

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
    export_tile_csv(
        renumbered,
        output_dir / "report.csv",
        slide_id=config.slide_id,
        tile_id=output_dir.name,
        model_version=config.model_version,
        config_hash=cfg_hash,
    )
    export_summary_statistics(renumbered, output_dir / "summary.txt")

    # slide 級 overlay：把每格核心裁切的 overlay_annotated tile 惰性拼回整片。
    _stitch_overlay_slide(output_dir, geometry)

    _log_batch_summary(run_id, stats, total)
    return stats


def _stitch_overlay_slide(output_dir: Path, geometry: TileGeometry) -> None:
    """把 ``overlay_annotated/`` 內每格核心裁切的 tile 惰性拼成一張全片 pyramid TIFF。

    每格 overlay 的尺寸依所在欄 / 列而異（邊界格較小），但**同欄同寬、同列同高**
    （由 ``core_crop_bounds`` 的建構保證）。``pyvips.Image.arrayjoin()`` 只做「等格montage」
    （以最大寬高為格、其餘留白），無法正確處理非均勻的每欄 / 每列尺寸（已用合成測試證實
    它會多出留白、尺寸錯誤）。故改為手動：每列內由左至右水平 join（同列高已相等），
    再把各列由上至下垂直 join（各列總寬已相等）——可逐像素還原原始版面。

    壓縮採 **lzw（無失真）**：這是帶細胞邊界線 / 標籤文字 / 紅黑點的標註影像，JPEG
    的區塊假影會糊掉細線與小點，醫師判讀不宜；lzw 保真且仍可壓。
    """
    overlay_dir = output_dir / "overlay_annotated"
    xs = sorted(geometry.col_of)  # abs_x，欄序（升冪即欄索引序）
    ys = sorted(geometry.row_of)  # abs_y，列序

    row_images: List[pyvips.Image] = []
    for ay in ys:
        row_tiles: List[pyvips.Image] = []
        for ax in xs:
            path = overlay_dir / f"tile_x{ax}_y{ay}.tiff"
            if not path.exists():
                raise FileNotFoundError(
                    f"overlay_annotated 缺少 tile: {path}——每格應恰有一檔。"
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

    slide.tiffsave(
        str(output_dir / "overlay_slide.tiff"),
        tile=True,
        pyramid=True,
        tile_width=256,
        tile_height=256,
        compression="lzw",
        bigtiff=True,
    )
    logger.info(
        "overlay_slide.tiff 縫合完成: %d×%d px", slide.width, slide.height
    )


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
    """CLI 單一 ROI/WSI 影像對模式：先預切成重疊 tile 檔，再走 ``run_batch`` 流程。

    內部分塊已移除，故先用 ``precut_paired_tiles`` 把整對影像切到
    ``output_dir/_precut_scratch/{ihc,dish}``（保留供檢查，不自動清理），再交給
    ``run_batch`` 做逐塊分析 + 全域合併 + slide 級縫合。
    """
    ihc_path = _resolve_tile_path(ihc_arg, config.ihc_tile_dir)
    dish_path = _resolve_tile_path(dish_arg, config.dish_tile_dir)

    output_dir = Path(output_dir)
    scratch = output_dir / "_precut_scratch"
    ihc_out = scratch / "ihc"
    dish_out = scratch / "dish"
    logger.info("單圖模式：預切 %s / %s → %s", ihc_path.name, dish_path.name, scratch)
    precut_paired_tiles(
        ihc_path,
        dish_path,
        ihc_out,
        dish_out,
        tile_size=config.default_tile_size,
        overlap=config.window_overlap_px,
    )

    run_batch(ihc_out, dish_out, output_dir)


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
