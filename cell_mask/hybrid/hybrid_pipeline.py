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
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from skimage import io

# Cellpose 在 get_masks_torch 內呼叫 torch.sparse_coo_tensor；PyTorch 要求顯式表態
# sparse invariant 檢查的開關，否則會持續發出 UserWarning。維持預設 (關閉檢查) 以保留效能。
torch.sparse.check_sparse_tensor_invariants.disable()

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
from m2_segmentation import CellposeSegmenter, segment_masked_dish
from cell_mask.hybrid.m3_cells_generator import (
    CellAnalysisResult,
    build_all_positive_results,
)
from m3_dot_detection import detect_all_dots, merge_dot_results_to_cell_analysis
from m4_export import export_cell_dot_annotations, export_overlay_visualization

# ------------------------------------------------------------------
# Logging 設定
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
        device=None,
    )


def _init_cellpose_segmenter() -> CellposeSegmenter:
    """延遲初始化 Cellpose 分割器（M2 IHC-DISH 細胞分割）。"""
    return CellposeSegmenter(
        model_path=config.cellpose_model_path,
        diameter=config.cellpose_diameter,
        flow_threshold=config.cellpose_flow_threshold,
        cellprob_threshold=config.cellpose_cellprob_threshold,
        gpu=config.cellpose_gpu,
    )


def _init_dish_cellpose_segmenter() -> CellposeSegmenter:
    """延遲初始化 DISH 細胞核偵測 Cellpose 分割器（M3b 多核排除用）。"""
    return CellposeSegmenter(
        model_path=config.cellpose_dish_model_path,
        diameter=config.cellpose_dish_diameter,
        flow_threshold=config.cellpose_dish_flow_threshold,
        cellprob_threshold=config.cellpose_dish_cellprob_threshold,
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

    Returns:
        分類結果列表，失敗時回傳 None。
    """
    tile_id = dish_tile_path.stem
    tile_output = output_dir / tile_id
    start_time = time.perf_counter()

    try:
        # ---- M1: 產生核心遮罩 + IHC-DISH 50/50 疊合 ----
        core_mask = generate_ihc_core_mask(
            ihc_tile_path,
            unet_inferencer,
            close_kernel=config.core_close_kernel,
        )

        if core_mask.sum() == 0:
            logger.warning(
                "Tile %s: 核心遮罩全空 — 僅匯出空 CSV", tile_id
            )
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
            return []

        dish_image = _read_rgb(dish_tile_path)
        ihc_image = _read_rgb(ihc_tile_path)

        # M1 Step 1: IHC core mask → masked IHC
        masked_ihc = apply_mask_to_ihc_image(
            ihc_image,
            core_mask,
            mask_blur_sigma=config.mask_blur_sigma,
            background_fill_value=config.background_fill_value,
        )

        # M1 Step 2: IHC core mask → masked DISH
        dish_mask_overlay = overlay_ihc_mask_on_dish(
            dish_image,
            core_mask,
            mask_blur_sigma=config.mask_blur_sigma,
            background_fill_value=config.background_fill_value,
        )

        # M1 Step 3: 50/50 alpha blend → IHC-DISH 疊合影像
        overlay_image = fuse_masked_ihc_with_dish(
            dish_mask_overlay,
            masked_ihc,
            ihc_alpha=config.overlay_alpha,
        )

        # 明確定義 M2 Cellpose 的輸入影像
        m2_input_overlay = overlay_image

        # ---- M1 中間結果落地 ----
        tile_output.mkdir(parents=True, exist_ok=True)
        io.imsave(
            str(tile_output / f"{tile_id}_ihc_core_mask.png"),
            (core_mask * 255).astype(np.uint8),
            check_contrast=False,
        )
        io.imsave(
            str(tile_output / f"{tile_id}_masked_ihc.png"),
            masked_ihc,
            check_contrast=False,
        )
        io.imsave(
            str(tile_output / f"{tile_id}_ihc_tumor.png"),
            masked_ihc,
            check_contrast=False,
        )
        io.imsave(
            str(tile_output / f"{tile_id}_dish_mask_overlay.png"),
            dish_mask_overlay,
            check_contrast=False,
        )
        io.imsave(
            str(tile_output / f"{tile_id}_dish_tumor.png"),
            dish_mask_overlay,
            check_contrast=False,
        )
        io.imsave(
            str(tile_output / f"{tile_id}_ihc_dish_overlay_raw.png"),
            overlay_image,
            check_contrast=False,
        )
        io.imsave(
            str(tile_output / f"{tile_id}_m2_input_overlay.png"),
            m2_input_overlay,
            check_contrast=False,
        )

        # ---- M2: Cellpose 分割 IHC-DISH 疊合影像 ----
        instance_mask = segment_masked_dish(
            m2_input_overlay,
            cellpose_segmenter,
            remove_border=config.clear_border_cells,
        )

        io.imsave(
            str(tile_output / f"{tile_id}_m2_cell_instance_binary.png"),
            ((instance_mask > 0).astype(np.uint8) * 255),
            check_contrast=False,
        )

        # ---- M3: 逐細胞分析 ----
        # M3a: 所有細胞標記為陽性（centroid + cell_id）
        results = build_all_positive_results(instance_mask)

        # M3b: DISH 細胞核偵測（用純 DISH 圖） + 紅/黑點偵測 + 多核排除
        dish_nucleus_mask = dish_cellpose_segmenter.predict(dish_image)
        # 用 IHC core_mask 過濾掉主要落在白色背景區的 DISH 核 instance，
        # 避免橘色輪廓 / 多核計數誤觸 mask 之外的細胞。
        dish_nucleus_mask = _filter_dish_nucleus_by_core_mask(
            dish_nucleus_mask, core_mask
        )
        all_dots, per_cell_dots = detect_all_dots(
            dish_mask_overlay,
            instance_mask,
            config,
            dish_nucleus_mask=dish_nucleus_mask,
        )
        results = merge_dot_results_to_cell_analysis(results, per_cell_dots)

        # ---- M4: 匯出 (單細胞來源為 dish_mask_overlay) ----
        export_cell_dot_annotations(
            dish_mask_overlay,
            instance_mask,
            results,
            tile_output,
            visualization_image=dish_mask_overlay,
            slide_id=config.slide_id,
            tile_id=tile_id,
            model_version=config.model_version,
            config_hash=cfg_hash,
            crop_size=config.cell_crop_size,
            all_dots=all_dots,
            per_cell_dots=per_cell_dots,
            dish_nucleus_mask=dish_nucleus_mask,
        )

        # 醫師檢視圖: IHC-DISH 疊合底圖 + 細胞邊界/AMP 標記 + 點位
        export_overlay_visualization(
            overlay_image,
            instance_mask,
            results,
            tile_output / f"{tile_id}_ihc_dish_overlay.png",
            all_dots=all_dots,
        )

        # ---- Merge overlay: 將 cellpose 細胞邊界繪製在原始合併影像上 ----
        merge_tile_path = _find_merge_tile(merge_dir, tile_id)
        if merge_tile_path is not None:
            merge_image = _read_rgb(merge_tile_path)
            if merge_image.shape[:2] == instance_mask.shape[:2]:
                export_overlay_visualization(
                    merge_image,
                    instance_mask,
                    results,
                    tile_output / f"{tile_id}_merge_overlay.png",
                    all_dots=all_dots,
                )
                logger.info("Merge overlay 匯出完成: %s", tile_id)
            else:
                logger.warning(
                    "Tile %s: merge 影像尺寸 %s 與 mask 尺寸 %s 不匹配，跳過 merge overlay",
                    tile_id, merge_image.shape[:2], instance_mask.shape[:2],
                )

        elapsed = time.perf_counter() - start_time
        pos_count = sum(1 for r in results if r.is_her2_positive)
        logger.info(
            "Tile %s 處理完成: %d 細胞 (%d 陽性), %.2f 秒",
            tile_id,
            len(results),
            pos_count,
            elapsed,
        )
        return results

    except ValueError as exc:
        logger.error("Tile %s 維度錯誤: %s", tile_id, exc)
        return None
    except Exception as exc:
        logger.error("Tile %s 處理失敗: %s", tile_id, exc, exc_info=True)
        return None


def _find_merge_tile(merge_dir: Optional[Path], tile_id: str) -> Optional[Path]:
    """嘗試在 merge 目錄中找到對應的合併影像。"""
    if merge_dir is None or not merge_dir.exists():
        return None
    for ext in config.supported_extensions:
        candidate = merge_dir / f"{tile_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _filter_dish_nucleus_by_core_mask(
    dish_nucleus_mask: np.ndarray,
    core_mask: np.ndarray,
    keep_threshold: float = 0.5,
) -> np.ndarray:
    """移除主要落在 IHC core_mask 之外的 DISH 核 instance。

    對每個 nucleus label 計算「在 core_mask 內的像素比例」，
    比例 < ``keep_threshold`` 的整個 instance 設為 0（移出 mask）。
    其餘 instance 像素保持不變、label 不重編。

    Why: dish_cellpose 在原始 DISH 圖推論，會在 IHC core_mask 之外的
    白色背景區也偵測出細胞核，導致橘色輪廓出現在 mask 外的白底區，
    也讓多核排除誤把背景核算進來。
    """
    if dish_nucleus_mask.size == 0:
        return dish_nucleus_mask
    mask_i32 = dish_nucleus_mask.astype(np.int32, copy=False)
    max_id = int(mask_i32.max())
    if max_id <= 0:
        return mask_i32
    core_bool = core_mask.astype(bool, copy=False)
    flat = mask_i32.ravel()
    total = np.bincount(flat, minlength=max_id + 1)
    inside = np.bincount(
        flat,
        weights=core_bool.ravel().astype(np.float32),
        minlength=max_id + 1,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total > 0, inside / total, 0.0)
    drop = ratio < keep_threshold
    drop[0] = False
    if drop.any():
        remap = np.arange(max_id + 1, dtype=np.int32)
        remap[drop] = 0
        mask_i32 = remap[mask_i32]
    return mask_i32


def _read_rgb(path: Path) -> np.ndarray:
    """讀取影像並確保為 RGB uint8。"""
    image = io.imread(str(path))
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
        result = process_single_tile(
            ihc_path, dish_path, unet, cellpose, dish_cellpose, output_dir, cfg_hash,
            merge_dir=merge_dir,
        )
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
