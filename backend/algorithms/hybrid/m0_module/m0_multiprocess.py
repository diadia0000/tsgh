"""
M0 跨 tile 多行程層：以 spawn worker 把逐塊分析攤到多個行程。

只負責分派與回收 ``(abs_x, abs_y, owned)``；全域 cell 重編號與縫合仍完全留在父行程
（``hybrid_pipeline._finish_batch``）。
"""
import gc
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from ..output.tile_x98304_y20480.config import config, compute_config_hash
    from ..m3_cell_detection import CellAnalysisResult
except ImportError:
    from backend.algorithms.hybrid.output.tile_x98304_y20480.config import config, compute_config_hash
    from m3_cell_detection import CellAnalysisResult

from .m0_stitch import TileGeometry
from .m0_tile_runner import (
    _frozen_gc_generation,
    _init_cellpose_segmenter,
    _init_dish_cellpose_segmenter,
    _init_unet_inferencer,
    _process_precut_tile_cpu,
    _process_precut_tile_gpu,
    _read_tile_pair,
    prefetch_tile_reads,
)

logger = logging.getLogger(__name__)


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

# Option L 的 worker 端消融開關（doc 34 step 2）。設為 "1" 時 worker 退回逐塊同步讀檔，
# 也就是 round 9 之前的形狀，作為量測的控制組。
#
# 為何是環境變數而不是參數或 monkeypatch：`perf_measure.py --no-prefetch` 是在**父行程**
# 的命名空間上替換 `prefetch_tile_reads`，而 spawn 出來的 worker 會重新 import 這個模組、
# 拿到乾淨的原函式——那個旋鈕碰不到 worker（doc 25 §2.2 對 `_save_tile_array` 記過同一件
# 事）。環境變數會被 spawn 的 child 繼承，是上面 `_MP_PROBE_ENV` 已經在用的同一條路。
# 沒設時的成本是每批一次 `os.environ.get`。
_MP_NO_PREFETCH_ENV = "HYBRID_MP_NO_PREFETCH"

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


def _drain_task_queue(task_q):
    """把 worker 的工作佇列包成 ``(ihc, dish, (abs_x, abs_y))`` 的產生器。

    ``prefetch_tile_reads`` 要的是一個可迭代的 tile 來源，而 worker 這邊的來源是一條
    跨行程佇列加一個毒丸哨兵。包成產生器是唯一需要的轉接，佇列語意一字不改：每塊仍只
    被一個 worker ``get`` 到一次，毒丸仍然結束這個 worker。
    """
    while True:
        task = task_q.get()
        if task is _MP_POISON:
            return
        ihc_path, dish_path, ax, ay = task
        yield ihc_path, dish_path, (ax, ay)


def _inline_tile_reads(tiles, pool, depth: int = 1):
    """``prefetch_tile_reads`` 的控制組：在消費該塊的當下、於呼叫端執行緒同步讀。

    產生器契約與 ``prefetch_tile_reads`` 完全相同（含把讀檔例外裝進 future，好讓
    ``.result()`` 在同一個位置重拋），故兩臂之間**只差「位元組在哪一刻被解碼」**這一件事，
    迴圈本體一行都不必分岔（playbook 反模式 #7：一次只變一件事）。
    """
    for item in tiles:
        fut: Future = Future()
        try:
            fut.set_result(_read_tile_pair(item[0], item[1]))
        except Exception as exc:             # noqa: BLE001 — 與預讀臂同樣延到消費點重拋
            fut.set_exception(exc)
        yield item, fut


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
    背景執行緒跑前一塊的 CPU 後段，主執行緒跑下一塊的 GPU 前段；再加上 doc 34 step 2
    起的第三條臂——下一塊的讀檔），因為那個重疊本身就值 doc 18 §2 量到的 −8.0%。少了
    它，每個 worker 都會退化成 round-3 之前的序列版本，多行程賺到的會被這裡賠掉。

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

        # 讀檔提前一塊（doc 30 Option L，doc 34 step 2 把它從單行程路徑搬到這裡）。
        # `workers=4` 才是實際出貨的組態，而 round 9 只把 Option L 接上單行程路徑，
        # 等於量到的 −1.75% 全部落在生產不跑的那條路上（doc 33 follow-up 3）。
        #
        # 用**獨立**的單執行緒池，理由與 `run_batch` 那邊一字不差：上面的 tile-cpu 是跑
        # `detect_all_dots` 的背景臂，共用一條會把讀取排在前一塊的 CPU 後段之後，正好
        # 序列化掉這個改動存在的理由。每個 worker 各自一條，四個行程就是四條，彼此不共享。
        read_tiles = (_inline_tile_reads
                      if os.environ.get(_MP_NO_PREFETCH_ENV) == "1"
                      else prefetch_tile_reads)
        pending: Optional[Tuple[int, int, Future]] = None
        with _frozen_gc_generation(), \
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="tile-cpu") as pool, \
                ThreadPoolExecutor(max_workers=1,
                                   thread_name_prefix="tile-read") as rpool:
            # 預讀會讓本 worker 比今日多從佇列拿一塊在手上。動態派工下這只是把「取工作」
            # 提前一塊，每塊仍恰好被一個 worker 處理一次；四個 worker 合計多握 4 塊，
            # 對 27,565 塊的負載平衡影響可忽略。
            for (ihc_path, dish_path, (ax, ay)), read_fut in read_tiles(
                    _drain_task_queue(task_q), rpool):
                try:
                    preread = read_fut.result()
                except Exception as exc:     # noqa: BLE001 — 見下方註解
                    # 預讀的失敗必須在「消費該塊的這一刻」浮出來，且必須算在**它自己那
                    # 一塊**頭上，語意與過去同步讀檔失敗完全一致（記錄後往下走 tg is None
                    # 那條 fail-fast），不能靜默丟掉、也不能記到別塊身上。
                    logger.error("Tile tile_x%d_y%d 讀取失敗: %s", ax, ay, exc)
                    preread = None

                tg = None if preread is None else _process_precut_tile_gpu(
                    ihc_path, dish_path, ax, ay, geometry,
                    unet, cellpose, dish_cellpose, output_dir, preread=preread,
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
        # 被 terminate 的 worker 不會再讀 task_q，佇列裡尚未寫出的工作於是永遠卡在它自己的
        # feeder 執行緒的 pipe write 上；而 multiprocessing 的 atexit（`util._exit_function`
        # → `Queue._finalize_join`）會去 join 那條執行緒 → **正確的 fail-fast 變成一個永遠
        # 不回來的行程**。doc 35 §6.2 記到 49 分鐘後人工砍掉；本輪用
        # `scripts/exit_latency_probe.py` 重現並抓到堆疊，證實就是這兩條互等。
        # `cancel_join_thread()` 的語義正是「行程結束時直接丟掉沒 flush 完的資料」——這批
        # 已經放棄了，那些工作本來就不該再送到任何地方，所以丟掉是正確而非將就。
        task_q.cancel_join_thread()
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
