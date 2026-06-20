"""
M2: Cellpose 實例分割模組

在 IHC-DISH 50/50 alpha blending 疊合影像上執行 Cellpose 推論，
產出 ``cell_instance_mask`` (背景=0, 細胞ID=1..N)。

模型已在醫師標註的 IHC-DISH 疊合影像上重新訓練。
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import find_objects
from skimage.segmentation import clear_border

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Cellpose 推論器
# ------------------------------------------------------------------

class CellposeSegmenter:
    """封裝 Cellpose 模型載入與推論。

    Attributes:
        model: Cellpose 模型物件。
        diameter: 預估細胞直徑 (pixels); None 時自動估計。
        flow_threshold: flow error 閾值。
        cellprob_threshold: 細胞機率閾值。
    """

    def __init__(
        self,
        model_path: Path,
        diameter: Optional[float] = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        batch_size: int = 16,
        gpu: bool = True,
    ) -> None:
        """初始化 Cellpose 推論器。

        Args:
            model_path: 預訓練 Cellpose 模型路徑。
            diameter: 細胞直徑 (pixels), None 則自動估計。
            flow_threshold: flow 誤差閾值。
            cellprob_threshold: 細胞機率閾值。
            gpu: 是否使用 GPU。
        """
        from cellpose.models import CellposeModel  # noqa: WPS433

        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.batch_size = max(1, int(batch_size))

        self.model = CellposeModel(
            gpu=gpu,
            pretrained_model=str(model_path),
        )
        logger.info(
            "Cellpose 模型載入完成: %s (diameter=%s)",
            model_path.name,
            diameter,
        )

    def predict(self, image: np.ndarray) -> np.ndarray:
        """執行單張影像的實例分割。

        Args:
            image: shape ``(H, W, 3)``、``uint8`` RGB 影像。

        Returns:
            shape ``(H, W)``、``int32`` 實例遮罩。
            背景=0, 細胞ID=1..N。
        """
        masks, _, _ = self.model.eval(
            image,
            batch_size=self.batch_size,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
        )
        return masks.astype(np.int32)

# ------------------------------------------------------------------
# 視窗化分割 + 重疊去重（大 patch sliding-window）
# ------------------------------------------------------------------

def segment_windowed(
    image: np.ndarray,
    segmenter: CellposeSegmenter,
    tile_size: int = 1024,
    overlap: int = 256,
    dedup_iomin: float = 0.5,
) -> np.ndarray:
    """以「重疊」視窗逐塊跑 Cellpose，再用 IoMin 去重得到完整 instance mask。

    產線情境：醫師手切的大張 patch 不整載，逐 ``tile_size`` 視窗推論。無重疊視窗
    會把邊界細胞切成兩半、在接縫被算成兩顆。本函式改用**重疊**視窗——只要
    ``overlap`` ≥ 最大細胞/核直徑，每顆邊界細胞至少在一個視窗內完整出現——再對
    全域所有 instance 去重：

        1. 視窗以 ``stride = tile_size - overlap`` 滑動（最後一格貼齊邊界以覆蓋全圖），
           逐視窗 ``segmenter.predict``，把每個 instance 連同其 bbox / 局部遮罩
           換算成全圖座標收集起來。
        2. 依面積由大到小貪婪去重：若一個 instance 與「已接受」的某 instance
           交集 / min(兩者面積) ≥ ``dedup_iomin``，視為同一顆（重疊視窗的重複偵測），
           丟棄較小者、保留較完整者。空間雜湊只比對 bbox 相鄰者，避免 O(n²)。
        3. 把接受的 instance 依序 paint 成 1..N（只填空白像素，較完整者優先佔位）。

    單一 ``tile_size`` 大小（或更小）的影像 → 僅一個視窗、無重複，行為等同直接
    ``segmenter.predict`` 後重編號。

    Args:
        image: shape ``(H, W, 3)``、``uint8`` RGB 影像。
        segmenter: 已初始化的 ``CellposeSegmenter``。
        tile_size: 視窗邊長（pixels）。
        overlap: 相鄰視窗的重疊寬度（pixels）；應 ≥ 最大細胞/核直徑。
        dedup_iomin: 去重門檻；交集 / min(面積) ≥ 此值即視為同一顆。

    Returns:
        shape ``(H, W)``、``int32`` 全域實例遮罩（背景=0，細胞=1..N，已去重）。
    """
    h, w = image.shape[:2]

    instances: List[Tuple[int, int, int, np.ndarray]] = []  # (area, y0, x0, local_bool)
    for y0, x0, y1, x1 in _overlap_window_coords(h, w, tile_size, overlap):
        local = segmenter.predict(image[y0:y1, x0:x1])
        local_max = int(local.max())
        if local_max <= 0:
            continue
        slices = find_objects(local)
        for lid in range(1, local_max + 1):
            sl = slices[lid - 1] if lid - 1 < len(slices) else None
            if sl is None:
                continue
            patch = local[sl] == lid
            area = int(patch.sum())
            if area == 0:
                continue
            instances.append((area, y0 + sl[0].start, x0 + sl[1].start, patch))

    canvas = np.zeros((h, w), dtype=np.int32)
    if not instances:
        return canvas

    accepted = _dedup_instances(instances, dedup_iomin)
    for new_id, (_area, iy, ix, patch) in enumerate(accepted, start=1):
        ph, pw = patch.shape
        sub = canvas[iy:iy + ph, ix:ix + pw]
        sub[(sub == 0) & patch] = new_id

    n_raw, n_kept = len(instances), len(accepted)
    if n_kept != n_raw:
        logger.info(
            "重疊去重: %d 個視窗 instance → %d 顆（去除重複 %d），overlap=%d, iomin=%.2f",
            n_raw, n_kept, n_raw - n_kept, overlap, dedup_iomin,
        )
    return canvas


def _overlap_window_coords(
    img_h: int,
    img_w: int,
    tile: int,
    overlap: int,
) -> List[Tuple[int, int, int, int]]:
    """重疊視窗的 (y0, x0, y1, x1) 座標；最後一格貼齊邊界以覆蓋全圖。"""
    stride = max(1, tile - max(0, overlap))

    def _starts(length: int) -> List[int]:
        if length <= tile:
            return [0]
        s = list(range(0, length - tile + 1, stride))
        if s[-1] != length - tile:
            s.append(length - tile)
        return s

    ys, xs = _starts(img_h), _starts(img_w)
    return [
        (y, x, min(y + tile, img_h), min(x + tile, img_w))
        for y in ys
        for x in xs
    ]


def _dedup_instances(
    instances: List[Tuple[int, int, int, np.ndarray]],
    iomin_threshold: float,
    grid: int = 256,
) -> List[Tuple[int, int, int, np.ndarray]]:
    """面積由大到小貪婪去重：交集/min(面積) ≥ 門檻者視為同一顆，保留較大者。

    用粗格空間雜湊把比對限制在 bbox 相鄰的 instance，避免全域 O(n²)。重複偵測來自
    重疊視窗、其 bbox 必相交，故必落在相同/相鄰格子內被比到。
    """
    order = sorted(instances, key=lambda t: t[0], reverse=True)
    buckets: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    accepted: List[Tuple[int, int, int, np.ndarray]] = []

    for area, iy, ix, patch in order:
        ph, pw = patch.shape
        iy1, ix1 = iy + ph, ix + pw
        gy0, gy1 = iy // grid, (iy1 - 1) // grid
        gx0, gx1 = ix // grid, (ix1 - 1) // grid

        is_dup = False
        seen_ids: set = set()
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                for rec in buckets.get((gy, gx), ()):  # noqa: WPS526
                    if id(rec) in seen_ids:
                        continue
                    seen_ids.add(id(rec))
                    oy0 = max(iy, rec["y0"]); ox0 = max(ix, rec["x0"])
                    oy1 = min(iy1, rec["y1"]); ox1 = min(ix1, rec["x1"])
                    if oy0 >= oy1 or ox0 >= ox1:
                        continue
                    inter = int(np.logical_and(
                        patch[oy0 - iy:oy1 - iy, ox0 - ix:ox1 - ix],
                        rec["patch"][oy0 - rec["y0"]:oy1 - rec["y0"],
                                     ox0 - rec["x0"]:ox1 - rec["x0"]],
                    ).sum())
                    if inter and inter / min(area, rec["area"]) >= iomin_threshold:
                        is_dup = True
                        break
                if is_dup:
                    break
            if is_dup:
                break

        if is_dup:
            continue
        accepted.append((area, iy, ix, patch))
        rec = {"area": area, "y0": iy, "x0": ix, "y1": iy1, "x1": ix1, "patch": patch}
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                buckets[(gy, gx)].append(rec)

    return accepted


# ------------------------------------------------------------------
# 分割入口
# ------------------------------------------------------------------

def segment_masked_dish(
    masked_overlay_image: np.ndarray,
    segmenter: CellposeSegmenter,
    remove_border: bool = True,
    tile_size: int = 1024,
    overlap: int = 256,
    dedup_iomin: float = 0.5,
) -> np.ndarray:
    """在 IHC-DISH 疊合影像上以重疊視窗 Cellpose 分割 + 去重。

    傳入的應為經 ``fuse_masked_ihc_with_dish`` 產生的
    IHC-DISH 50/50 alpha blending 疊合影像，
    非 ROI 區域已填充為背景值。

    分割以 ``tile_size`` 重疊視窗逐塊推論後去重（見 ``segment_windowed``）；
    ``remove_border`` 在去重完成的全域 mask 上清邊，因此只會移除碰觸 patch
    「真實外緣」的細胞，跨視窗邊界的細胞已在重疊視窗內完整偵測、不受影響。

    Args:
        masked_overlay_image: shape ``(H, W, 3)``、``uint8``。
            IHC-DISH 疊合影像。
        segmenter: 已初始化的 ``CellposeSegmenter``。
        remove_border: 是否移除碰觸 patch 外緣的細胞。
        tile_size: 視窗邊長（pixels）。
        overlap: 相鄰視窗重疊寬度（pixels）。
        dedup_iomin: 重疊去重門檻（交集/min(面積)）。

    Returns:
        shape ``(H, W)``、``int32`` 實例遮罩 (背景=0)。
    """
    instance_mask = segment_windowed(
        masked_overlay_image,
        segmenter,
        tile_size=tile_size,
        overlap=overlap,
        dedup_iomin=dedup_iomin,
    )

    if remove_border:
        instance_mask = _remove_border_cells(instance_mask)

    num_cells = len(np.unique(instance_mask)) - 1  # 扣除背景 0
    logger.info("Cellpose 分割完成: %d 個有效細胞", num_cells)
    return instance_mask


def _remove_border_cells(instance_mask: np.ndarray) -> np.ndarray:
    """移除碰觸影像邊界的細胞，並重新編號。

    Args:
        instance_mask: ``int32`` 實例遮罩。

    Returns:
        移除邊界細胞後的 ``int32`` 實例遮罩（ID 連續化）。
    """
    before_ids = set(np.unique(instance_mask)) - {0}

    cleaned = clear_border(instance_mask)
    cleaned = cleaned.astype(np.int32)

    after_ids = set(np.unique(cleaned)) - {0}
    removed_count = len(before_ids) - len(after_ids)
    if removed_count > 0:
        logger.info("移除 %d 個邊界細胞", removed_count)

    return _relabel_sequential(cleaned)


def _relabel_sequential(mask: np.ndarray) -> np.ndarray:
    """將非連續 ID 重新標記為 1..N 連續整數。"""
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]

    relabeled = np.zeros_like(mask, dtype=np.int32)
    for new_id, old_id in enumerate(sorted(unique_ids), start=1):
        relabeled[mask == old_id] = new_id

    return relabeled
