"""
IHC-DISH Overlay & Analysis Pipeline — 主入口

串接 M1 (Overlay) → M2 (Segmentation) → M3 (Cell Results) → M4 (Export)。
支援單 tile 處理與批次掃描。

流程:
  - M0: 先把整片玻片切成重疊 tile
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
import sys
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch import cuda

# 將專案根目錄與 hybrid 目錄加入 sys.path，確保直接執行腳本時可解析套件匯入
_HYBRID_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _HYBRID_DIR.parent.parent
for _path in (str(_PROJECT_ROOT), str(_HYBRID_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from .config import config, compute_config_hash
    from .m1_overlay import (
        find_paired_tiles,
        parse_tile_coords,
    )
    from .m3_cell_detection import CellAnalysisResult
    from .m4_export import (
        export_summary_statistics,
        export_tile_csv,
    )
    from .m0_slide import (
        _CKPT_DIRNAME,
        _checkpoint_init,
        _checkpoint_load,
        _checkpoint_save,
        _frozen_gc_generation,
        _init_cellpose_segmenter,
        _init_dish_cellpose_segmenter,
        _init_unet_inferencer,
        _process_precut_tile_cpu,
        _process_precut_tile_gpu,
        prefetch_tile_reads,
        _run_tiles_multiprocess,
        _skip_completed,
        _stitch_overlay_slide,
        LAST_MP_WORKER_TIMINGS,
        PrecutStream,
        TileGeometry,
        compute_tile_geometry,
        dedup_cross_tile_duplicates,
    )
except ImportError:
    from config import config, compute_config_hash
    from m1_overlay import (
        find_paired_tiles,
        parse_tile_coords,
    )
    from m3_cell_detection import CellAnalysisResult
    from m4_export import (
        export_summary_statistics,
        export_tile_csv,
    )
    from m0_slide import (
        _CKPT_DIRNAME,
        _checkpoint_init,
        _checkpoint_load,
        _checkpoint_save,
        _frozen_gc_generation,
        _init_cellpose_segmenter,
        _init_dish_cellpose_segmenter,
        _init_unet_inferencer,
        _process_precut_tile_cpu,
        _process_precut_tile_gpu,
        prefetch_tile_reads,
        _run_tiles_multiprocess,
        _skip_completed,
        _stitch_overlay_slide,
        LAST_MP_WORKER_TIMINGS,
        PrecutStream,
        TileGeometry,
        compute_tile_geometry,
        dedup_cross_tile_duplicates,
    )

# ------------------------------------------------------------------
# Logging 設定
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    """批次處理已預切 tile 目錄：逐塊分析 → 全域合併細胞表 → slide 級 overlay 縫合。

    ``ihc_dir`` / ``dish_dir`` 為 ``precut_paired_tiles`` 產出的、以
    ``tile_x{int}_y{int}`` 命名的重疊 tile 檔目錄（兩路同名同格線）。

    Args:
        ihc_dir: 已預切 IHC tile 目錄。
        dish_dir: 已預切 DISH tile 目錄。
        output_dir: 輸出根目錄。
        merge_dir: 合併影像目錄（可選），用於產出 merge overlay。
        tile_stream: 可選的 ``m0_reader.PrecutStream``。給定時改由它供給 tile，
            ``ihc_dir`` / ``dish_dir`` 不再被讀取；不給則掃描目錄。全域重編號依
            ``(abs_y, abs_x, cell_id)`` 排序、縫合按座標讀檔，故供 tile 的順序不影響輸出。
        workers: 跨 tile 平行的行程數。預設 ``4``；``1`` 走單行程路徑；``>1`` 走
            ``_run_tiles_multiprocess``，每個 worker 自帶一份模型與 CUDA context。
            單塊 API 請求（`backend/api/hybrid.py`）應明確傳入 ``workers=1``。
        checkpoint: 是否啟用斷點續跑。``True`` 時每塊完成即落地到
            ``output_dir/_resume/``；執行前會先載入該目錄中屬於這批格線、
            config_hash 相符的塊並跳過。預設 ``False``。不放寬 fail-fast：
            任一塊失敗仍中止整批。

    Returns:
        ``{"success": int, "skipped": int}`` 統計。任一塊真實失敗即
        raise 中止整批，故統計裡不設 ``failed`` 計數。
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
    # 只分析 ROI 時格線從 ROI 起點算起（tile 座標仍是全片絕對值），把原點交給驗證器，
    # 這樣「第一格不見了」在兩種模式下都還是抓得到。
    region = getattr(tile_stream, "region", None) if tile_stream is not None else None
    origin = (region[0], region[1]) if region else (0, 0)
    geometry = compute_tile_geometry(
        positions, config.default_tile_size, config.window_overlap_px, origin=origin
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
        owned = fut.result()
        _record(e_ax, e_ay, owned)
        # 把這塊新增的結果計入再凍結節奏。只掛在單行程路徑上：多行程時 _record 由父行程
        # 在 _frozen_gc_generation 之外呼叫，在那裡 freeze 就成了沒有配對 unfreeze 的洩漏
        # （doc 15 擋的正是這個），而父行程本來就不逐塊 collect，也沒有這個成本可省。
        refreezer.note(len(owned))

    # 兩段式管線（單一背景執行緒，管線深度 1）：三個 GPU 模型仍只在主行程 / 單一 CUDA
    # context 載入一次並重用，且**所有 GPU 前向仍只在主執行緒序列執行**（背景執行緒完全
    # 不碰 torch），故不觸犯跨行程 fork-under-CUDA 限制。差別只在把每塊的 CPU 後段
    # （detect_all_dots + 落地寫檔，原本讓 GPU 整段閒置）丟到背景執行緒，與『下一塊的 GPU
    # 前向』重疊執行，藉此填補 GPU 閒置。fail-fast 與 empty_cache/gc 清理語意皆保留（見下）。
    pending: Optional[Tuple[int, int, Future]] = None
    with _frozen_gc_generation() as refreezer, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="tile-cpu") as pool, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="tile-read") as rpool:
        # 讀檔提前一塊（doc 30 Option L）：本塊的 GPU 前向跑的同時，下一塊的兩個檔已在
        # 背景解碼。用**獨立**的單執行緒池，不跟上面 tile-cpu 那條（跑 detect_all_dots
        # 的背景臂）互搶——那是兩份不相干的工作，共用一條會把讀取排在 CPU 後段之後。
        for idx, ((ihc_path, dish_path, (ax, ay)), read_fut) in enumerate(
                prefetch_tile_reads(tiles, rpool), start=1):
            logger.info(
                "[%d/%d] 處理 tile: %s", idx, remaining, dish_path.stem
            )
            try:
                preread = read_fut.result()
            except Exception as exc:
                # 預讀的失敗必須在「消費該塊的這一刻」浮出來，語意與過去同步讀檔失敗
                # 完全一致（記錄後往下走 tg is None 那條 fail-fast），不能靜默丟掉。
                logger.error("Tile tile_x%d_y%d 讀取失敗: %s", ax, ay, exc)
                preread = None
            tg = None if preread is None else _process_precut_tile_gpu(
                ihc_path, dish_path, ax, ay, geometry,
                unet, cellpose, dish_cellpose, output_dir, preread=preread,
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
            # 每次 collect 的**掃描量**則由下方 refreezer 定期收斂（doc 28 Option H）——
            # 不然整批累積的結果會讓這一行從 1.2 ms 一路爬到 80.5 ms。
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
    # Ghost-row 安全網：兩塊各自獨立偵測到同一顆物理細胞、質心差幾個像素時，
    # core-ownership 去重會兩邊都留下（見 m0_stitch 模組 docstring）。
    ghost_dedup_px = float(getattr(config, "ghost_dedup_distance_px", 6.0))
    per_tile_owned = dedup_cross_tile_duplicates(per_tile_owned, ghost_dedup_px)

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
        default=4,
        help="跨 tile 平行的行程數（預設 4 = round 8 全片驗證通過、round 12 重測仍通過"
             "的生產設定，見 docs/hybrid-pipeline/39-round-12-multiprocess-scaling-"
             "ceiling-implementation.md）。",
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
