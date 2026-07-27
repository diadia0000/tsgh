"""
M0 斷點續跑層：每塊完成即把 ``owned`` 結果落地，重跑時載回並跳過該塊。

純檔案 I/O，不碰模型也不碰 GPU；單行程與多行程兩條路徑共用同一份存 / 載實作。
"""
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    from ..m1_overlay import parse_tile_coords
    from ..m3_cell_detection import CellAnalysisResult
except ImportError:
    from m1_overlay import parse_tile_coords
    from m3_cell_detection import CellAnalysisResult

logger = logging.getLogger(__name__)


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
