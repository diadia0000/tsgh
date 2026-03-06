"""
IHC-DISH Overlay & Analysis Pipeline — 主入口

串接 M1 (Overlay) → M2 (Segmentation) → M3 (Dot Quantification) → M4 (Export)。
支援單 tile 處理與批次掃描。

Usage:
    # 單 tile
    python hybrid_mask.py --ihc tile_x1024_y2048.tiff --dish tile_x1024_y2048.tiff

    # 批次 (目錄掃描)
    python hybrid_mask.py --batch

    # 使用 test_picture 資料夾
    python hybrid_mask.py --batch --test
"""

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
from skimage import io

# 將 hybrid 目錄加入 sys.path 以便 config import
_HYBRID_DIR = Path(__file__).resolve().parent
if str(_HYBRID_DIR) not in sys.path:
    sys.path.insert(0, str(_HYBRID_DIR))

from config import config, compute_config_hash
from m1_overlay import (
    find_paired_tiles,
    generate_ihc_core_mask,
    overlay_ihc_mask_on_dish,
    parse_tile_coords,
)
from m2_segmentation import CellposeSegmenter, segment_masked_dish
from m3_dot_quant import CellAnalysisResult, quantify_dish_signals
from m4_export import export_cell_dot_annotations

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
    _unet_dir = Path(__file__).resolve().parent.parent / "unet_mask"
    if str(_unet_dir) not in sys.path:
        sys.path.insert(0, str(_unet_dir))

    from inference import UNetPPInference  # noqa: WPS433

    return UNetPPInference(
        model_path=config.unet_model_path,
        encoder_name=config.unet_encoder_name,
        num_classes=config.unet_num_classes,
        image_size=config.unet_image_size,
        device=None,
    )


def _init_cellpose_segmenter() -> CellposeSegmenter:
    """延遲初始化 Cellpose 分割器。"""
    return CellposeSegmenter(
        model_path=config.cellpose_model_path,
        diameter=config.cellpose_diameter,
        flow_threshold=config.cellpose_flow_threshold,
        cellprob_threshold=config.cellpose_cellprob_threshold,
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
    output_dir: Path,
    cfg_hash: str,
) -> Optional[List[CellAnalysisResult]]:
    """處理單一配對 tile 的完整流水線。

    Args:
        ihc_tile_path: IHC tile 影像路徑。
        dish_tile_path: DISH tile 影像路徑。
        unet_inferencer: 已初始化的 UNetPPInference。
        cellpose_segmenter: 已初始化的 CellposeSegmenter。
        output_dir: 輸出目錄。
        cfg_hash: 配置雜湊。

    Returns:
        量化結果列表，失敗時回傳 None。
    """
    tile_id = dish_tile_path.stem
    tile_output = output_dir / tile_id
    start_time = time.perf_counter()

    try:
        # ---- M1: 產生核心遮罩並疊加至 DISH ----
        core_mask = generate_ihc_core_mask(
            ihc_tile_path,
            unet_inferencer,
            dilate_kernel=config.membrane_dilate_kernel,
            close_kernel=config.membrane_close_kernel,
            max_boundary_gap=config.max_boundary_gap,
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
            )
            return []

        dish_image = _read_rgb(dish_tile_path)
        masked_dish = overlay_ihc_mask_on_dish(
            dish_image,
            core_mask,
            mask_blur_sigma=config.mask_blur_sigma,
            background_fill_value=config.background_fill_value,
        )

        # ---- M2: Cellpose 分割 ----
        instance_mask = segment_masked_dish(
            masked_dish,
            cellpose_segmenter,
            remove_border=config.clear_border_cells,
        )

        # ---- M3: Dot 定量 ----
        results = quantify_dish_signals(
            masked_dish,
            instance_mask,
            od_matrix=config.od_matrix,
            min_sigma=config.log_min_sigma,
            max_sigma=config.log_max_sigma,
            num_sigma=config.log_num_sigma,
            log_threshold=config.log_threshold,
            min_blob_area=config.min_blob_area,
            cluster_area_factor=config.cluster_area_factor,
        )

        # ---- M4: 匯出 ----
        export_cell_dot_annotations(
            masked_dish,
            instance_mask,
            results,
            tile_output,
            slide_id=config.slide_id,
            tile_id=tile_id,
            model_version=config.model_version,
            config_hash=cfg_hash,
        )

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Tile %s 處理完成: %d 細胞, %.2f 秒",
            tile_id,
            len(results),
            elapsed,
        )
        return results

    except ValueError as exc:
        logger.error("Tile %s 維度錯誤: %s", tile_id, exc)
        return None
    except Exception as exc:
        logger.error("Tile %s 處理失敗: %s", tile_id, exc, exc_info=True)
        return None


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
) -> dict:
    """批次掃描目錄並處理所有配對 tile。

    Args:
        ihc_dir: IHC tile 目錄。
        dish_dir: DISH tile 目錄。
        output_dir: 輸出根目錄。

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

    stats = {"success": 0, "failed": 0, "skipped": 0}
    total = len(paired_tiles)

    for idx, (ihc_path, dish_path) in enumerate(paired_tiles, start=1):
        logger.info(
            "[%d/%d] 處理 tile: %s", idx, total, dish_path.stem
        )
        result = process_single_tile(
            ihc_path, dish_path, unet, cellpose, output_dir, cfg_hash
        )
        if result is None:
            stats["failed"] += 1
        elif len(result) == 0:
            stats["skipped"] += 1
        else:
            stats["success"] += 1

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
        run_batch(ihc_dir, dish_dir, output_dir)
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

    process_single_tile(
        ihc_path, dish_path, unet, cellpose, output_dir, cfg_hash
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
