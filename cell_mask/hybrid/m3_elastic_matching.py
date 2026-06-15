"""M3b DISH 細胞核彈性匹配：把 DISH 核 instance 認領給 IHC 細胞。

從 m3_dot_detection.py 拆出。greedy exclusive centroid 分配，每顆 DISH 核最多被
一顆 IHC 細胞認領；matched DISH 核數用於排除多核細胞並擴張偵測 ROI。
詳見 docs/sdd-elastic-dish-matching.md。
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import center_of_mass, find_objects
from skimage.morphology import binary_dilation, disk

logger = logging.getLogger(__name__)


def elastic_dish_nucleus_matching(
    dish_nucleus_mask: np.ndarray,
    strict_instance_mask: np.ndarray,
    cfg: object,
) -> Dict[int, List[int]]:
    """彈性匹配：IHC region 等向膨脹 → 候選收集 → centroid greedy 分配。

    三步驟流程（詳見 docs/sdd-elastic-dish-matching.md）：
        Step 1: 對每個 IHC 細胞 region 計算等向 dilation radius，使面積放大至
                ``dish_elastic_expand_factor`` 倍（近似圓形推導：
                r = (sqrt(A*f) - sqrt(A)) / sqrt(π)）。
        Step 2: 找出膨脹後 region 與 DISH 核的重疊 pixel，收集 candidate DISH ID。
        Step 3: 對 candidate pair 依 centroid 歐氏距離排序，greedy exclusive 分配
                ——每個 DISH 核最多被一個 IHC 細胞認領，IHC 可認領多顆（多核）。
                超過 ``dish_elastic_max_dist_px`` 的 pair 直接排除。

    Returns:
        ``{ihc_cell_id: [assigned_dish_id, ...]}``。
    """
    expand_factor = float(getattr(cfg, "dish_elastic_expand_factor", 1.5))
    max_dist_px = float(getattr(cfg, "dish_elastic_max_dist_px", 50.0))

    ihc_ids: List[int] = [int(v) for v in np.unique(strict_instance_mask) if v != 0]
    dish_ids: List[int] = [int(v) for v in np.unique(dish_nucleus_mask) if v != 0]
    result: Dict[int, List[int]] = {cid: [] for cid in ihc_ids}

    if not ihc_ids or not dish_ids:
        return result

    ihc_centroids: Dict[int, Tuple[float, float]] = {}
    cell_candidates: Dict[int, List[int]] = {cid: [] for cid in ihc_ids}

    # 用 find_objects 一次取得每顆 label 的 bbox，把所有 per-cell 運算侷限到
    # 自己的小視窗，避免在整張 strip 大小的陣列上反覆掃描 / dilation。
    H, W = strict_instance_mask.shape
    ihc_slices = find_objects(strict_instance_mask)
    for cid in ihc_ids:
        sl = ihc_slices[cid - 1] if cid - 1 < len(ihc_slices) else None
        if sl is None:
            continue
        local = (strict_instance_mask[sl] == cid)
        cy, cx = center_of_mass(local)
        ihc_centroids[cid] = (float(cy) + sl[0].start, float(cx) + sl[1].start)

        area = int(local.sum())
        if area == 0:
            continue
        r = max(1, round(
            (math.sqrt(area * expand_factor) - math.sqrt(area)) / math.sqrt(math.pi)
        ))
        # dilation 至多外擴 r：取 bbox 外加 r padding 的視窗，侷部 dilate 後
        # 結果與全圖 dilation 完全相同，但成本只與單顆細胞大小相關。
        r0 = max(0, sl[0].start - r)
        r1 = min(H, sl[0].stop + r)
        c0 = max(0, sl[1].start - r)
        c1 = min(W, sl[1].stop + r)
        win = (strict_instance_mask[r0:r1, c0:c1] == cid)
        expanded = binary_dilation(win, disk(r))
        overlap_vals = dish_nucleus_mask[r0:r1, c0:c1][expanded]
        cell_candidates[cid] = [int(v) for v in np.unique(overlap_vals) if v != 0]

    dish_centroids: Dict[int, Tuple[float, float]] = {}
    dish_slices = find_objects(dish_nucleus_mask)
    for did in dish_ids:
        sl = dish_slices[did - 1] if did - 1 < len(dish_slices) else None
        if sl is None:
            continue
        local = (dish_nucleus_mask[sl] == did)
        cy, cx = center_of_mass(local)
        dish_centroids[did] = (float(cy) + sl[0].start, float(cx) + sl[1].start)

    pairs: List[Tuple[float, int, int]] = []
    for cid, candidates in cell_candidates.items():
        iy, ix = ihc_centroids[cid]
        for did in candidates:
            dy, dx = dish_centroids[did]
            dist = math.hypot(iy - dy, ix - dx)
            if dist <= max_dist_px:
                pairs.append((dist, cid, did))
    pairs.sort(key=lambda x: x[0])

    assigned_dish_ids: set = set()
    for _dist, cid, did in pairs:
        if did in assigned_dish_ids:
            continue
        result[cid].append(did)
        assigned_dish_ids.add(did)

    exclude_thr = int(getattr(cfg, "dot_blue_exclude_threshold", 2))
    n_matched_any = sum(1 for v in result.values() if len(v) > 0)
    n_multi = sum(1 for v in result.values() if len(v) >= exclude_thr)
    logger.info(
        "elastic_dish_nucleus_matching: IHC=%d, DISH=%d, 有匹配=%d, "
        "多核候選(>=%d)=%d, expand_factor=%.1f, max_dist=%.1fpx",
        len(ihc_ids), len(dish_ids), n_matched_any, exclude_thr, n_multi,
        expand_factor, max_dist_px,
    )
    return result
