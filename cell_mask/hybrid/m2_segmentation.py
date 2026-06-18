"""
M2: Cellpose 實例分割模組

在 IHC-DISH 50/50 alpha blending 疊合影像上執行 Cellpose 推論，
產出 ``cell_instance_mask`` (背景=0, 細胞ID=1..N)。

模型已在醫師標註的 IHC-DISH 疊合影像上重新訓練。
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import DisjointSet
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
# 視窗化分割 + 接縫縫合（大 patch sliding-window）
# ------------------------------------------------------------------

def segment_windowed(
    image: np.ndarray,
    segmenter: CellposeSegmenter,
    tile_size: int = 1024,
    min_seam_contact_px: int = 1,
) -> np.ndarray:
    """以無重疊 1k 視窗逐塊跑 Cellpose，再把跨接縫被切開的細胞縫合回完整 instance。

    產線情境：醫師手切的大張 patch 不整載，逐 ``tile_size`` 視窗推論。視窗邊界會
    把細胞 / 核切成數塊，本函式：
        1. 逐視窗 ``segmenter.predict``，每視窗 label 加上累積偏移避免撞號後寫入
           全域 label canvas。
        2. 對每條內部接縫檢查「上下左右」相鄰且雙方皆非零的 label pair，接觸像素數
           ≥ ``min_seam_contact_px`` 才以 disjoint-set 聯集成同一顆（抑制雜訊誤併）。
        3. 依聯集根 relabel 為 1..N 連續整數。

    單一 ``tile_size`` 大小（或更小）的影像 → 僅一個視窗、無接縫，行為與直接
    ``segmenter.predict`` 相同（向後相容）。

    Args:
        image: shape ``(H, W, 3)``、``uint8`` RGB 影像。
        segmenter: 已初始化的 ``CellposeSegmenter``。
        tile_size: 視窗邊長（pixels）。
        min_seam_contact_px: 一對跨接縫 label 的接觸像素數門檻，≥ 此值才聯集。

    Returns:
        shape ``(H, W)``、``int32`` 全域實例遮罩（背景=0，細胞=1..N，接縫已縫合）。
    """
    h, w = image.shape[:2]
    canvas = np.zeros((h, w), dtype=np.int32)

    offset = 0
    for y0, x0, y1, x1 in _window_coords(h, w, tile_size):
        local = segmenter.predict(image[y0:y1, x0:x1])
        local_max = int(local.max())
        if local_max <= 0:
            continue
        canvas[y0:y1, x0:x1] = np.where(local > 0, local + offset, 0)
        offset += local_max

    if offset == 0:
        return canvas

    ds = DisjointSet(range(1, offset + 1))
    for x_s in range(tile_size, w, tile_size):           # 垂直接縫
        _union_seam(ds, canvas[:, x_s - 1], canvas[:, x_s], min_seam_contact_px)
    for y_s in range(tile_size, h, tile_size):           # 水平接縫
        _union_seam(ds, canvas[y_s - 1, :], canvas[y_s, :], min_seam_contact_px)

    stitched = _relabel_by_subsets(canvas, ds, offset)
    n_before, n_after = offset, int(stitched.max())
    if n_after != n_before:
        logger.info(
            "接縫縫合: %d 個視窗碎塊 → %d 顆完整 instance（合併 %d）",
            n_before, n_after, n_before - n_after,
        )
    return stitched


def _window_coords(
    img_h: int,
    img_w: int,
    tile: int,
) -> List[Tuple[int, int, int, int]]:
    """無重疊視窗的 (y0, x0, y1, x1) 座標；邊緣不足一格者切較小 patch。"""
    coords: List[Tuple[int, int, int, int]] = []
    for y0 in range(0, img_h, tile):
        for x0 in range(0, img_w, tile):
            coords.append((y0, x0, min(y0 + tile, img_h), min(x0 + tile, img_w)))
    return coords


def _union_seam(
    ds: DisjointSet,
    side_a: np.ndarray,
    side_b: np.ndarray,
    min_contact: int,
) -> None:
    """聯集一條接縫上「跨縫相鄰且雙方非零」的 label pair（接觸數 ≥ 門檻才併）。"""
    valid = (side_a > 0) & (side_b > 0)
    if not valid.any():
        return
    pairs = np.stack([side_a[valid], side_b[valid]], axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    for (la, lb), c in zip(uniq, counts):
        if c >= min_contact:
            ds.merge(int(la), int(lb))


def _relabel_by_subsets(
    canvas: np.ndarray,
    ds: DisjointSet,
    max_id: int,
) -> np.ndarray:
    """以 disjoint-set 子集為單位，把全域 canvas relabel 成 1..N 連續整數。

    僅針對 canvas 中實際出現的 label 指派新號（同一子集共用同號），確保輸出
    連續、且不受 cellpose 偶發非連續 label 影響。
    """
    present = np.unique(canvas)
    present = present[present != 0]
    lut = np.zeros(max_id + 1, dtype=np.int32)
    rep_to_new: dict = {}
    for old_id in present:
        rep = ds[int(old_id)]
        new_id = rep_to_new.setdefault(rep, len(rep_to_new) + 1)
        lut[int(old_id)] = new_id
    return lut[canvas]


# ------------------------------------------------------------------
# 分割入口
# ------------------------------------------------------------------

def segment_masked_dish(
    masked_overlay_image: np.ndarray,
    segmenter: CellposeSegmenter,
    remove_border: bool = True,
    tile_size: int = 1024,
    min_seam_contact_px: int = 1,
) -> np.ndarray:
    """在 IHC-DISH 疊合影像上以視窗化 Cellpose 分割 + 接縫縫合。

    傳入的應為經 ``fuse_masked_ihc_with_dish`` 產生的
    IHC-DISH 50/50 alpha blending 疊合影像，
    非 ROI 區域已填充為背景值。

    分割以 ``tile_size`` 視窗逐塊推論後縫合（見 ``segment_windowed``）；
    ``remove_border`` 在縫合完成的全域 mask 上清邊，因此只會移除碰觸 patch
    「真實外緣」的細胞，內部接縫細胞已先縫合不受影響。

    Args:
        masked_overlay_image: shape ``(H, W, 3)``、``uint8``。
            IHC-DISH 疊合影像。
        segmenter: 已初始化的 ``CellposeSegmenter``。
        remove_border: 是否移除碰觸 patch 外緣的細胞。
        tile_size: 視窗邊長（pixels）。
        min_seam_contact_px: 接縫聯集的接觸像素門檻。

    Returns:
        shape ``(H, W)``、``int32`` 實例遮罩 (背景=0)。
    """
    instance_mask = segment_windowed(
        masked_overlay_image,
        segmenter,
        tile_size=tile_size,
        min_seam_contact_px=min_seam_contact_px,
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
