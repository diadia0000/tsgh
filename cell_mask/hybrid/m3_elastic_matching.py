"""M3b DISH 細胞核彈性匹配：以「IHC 細胞為中心」把最近的 DISH 核認領給細胞。

**以細胞為中心 + 1.5x 膨脹**：每顆 IHC 細胞（綠框）的搜尋範圍 = 把自身面積放大
``dish_elastic_expand_factor`` 倍後的等效圓半徑 ``reach = sqrt(factor*area/π)``；
凡質心落在此半徑內的 DISH 核（橘框）皆為候選。配對採「一對一、最近優先、lock」：
把所有 ``(細胞, 核)`` 候選對依歐式距離（細胞質心↔核質心）升冪排序，最近的先配並
各自鎖定；落敗的細胞自動往下找仍可用的核。每顆細胞最多認領 1 顆核。

最終每顆 IHC 的狀態：認到 1 核=valid；0 核但曾有候選卻競爭落敗、已無可用核
=drop-out（排除、打 X）；0 核且從頭就沒有候選=忽略配對、照常計入（顯示 0/0）。
一對一配對下每顆細胞至多 1 核，故不再有「多核排除」。

詳見 docs/sdd-elastic-dish-matching.md（註：該文件描述舊「以核為中心」版本）。
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.ndimage import center_of_mass, find_objects
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def _centroids_and_areas(
    mask: np.ndarray, ids: List[int]
) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, int]]:
    """用各自 bbox 局部算每個 label 的質心（全圖座標）與面積，避免掃整張 strip。"""
    centroids: Dict[int, Tuple[float, float]] = {}
    areas: Dict[int, int] = {}
    slices = find_objects(mask)
    for lid in ids:
        sl = slices[lid - 1] if lid - 1 < len(slices) else None
        if sl is None:
            continue
        local = mask[sl] == lid
        area = int(local.sum())
        if area == 0:
            continue
        cy, cx = center_of_mass(local)
        centroids[lid] = (float(cy) + sl[0].start, float(cx) + sl[1].start)
        areas[lid] = area
    return centroids, areas


def elastic_dish_nucleus_matching(
    dish_nucleus_mask: np.ndarray,
    strict_instance_mask: np.ndarray,
    cfg: object,
) -> Tuple[Dict[int, List[int]], Set[int]]:
    """以細胞為中心、一對一的最近 DISH 核認領。

    流程：
        Step 1: 算每顆 IHC 細胞與 DISH 核的質心；細胞另算面積以決定搜尋半徑。
        Step 2: 細胞 c 的搜尋半徑 ``reach = sqrt(factor * area_c / π)``——把面積
                放大 ``factor`` 倍後的等效圓半徑。質心落在 reach 內的核為 c 的候選。
        Step 3: 蒐集所有 ``(細胞, 核)`` 候選對，依細胞質心↔核質心歐式距離升冪
                排序，貪婪一對一配對（最近先配並 lock，落敗者自動往下找可用核）。

    Args:
        dish_nucleus_mask: ``(H, W)`` int，0=背景，1..M=DISH 細胞核 ID。
        strict_instance_mask: ``(H, W)`` int，0=背景，1..N=IHC 細胞 ID。
        cfg: 具 ``dish_elastic_expand_factor``（綠框面積放大倍數）的配置。

    Returns:
        ``({ihc_cell_id: [assigned_dish_id] 或 []}, drop_out_ids)``，其中
        ``drop_out_ids`` 為「曾有候選核卻競爭落敗、最終 0 核」的 IHC ID 集合
        （competition loser，下游打 X）；從頭無候選的 0 核細胞不在此集合。
    """
    factor = float(getattr(cfg, "dish_elastic_expand_factor", 1.5))

    ihc_ids: List[int] = [int(v) for v in np.unique(strict_instance_mask) if v != 0]
    dish_ids: List[int] = [int(v) for v in np.unique(dish_nucleus_mask) if v != 0]
    result: Dict[int, List[int]] = {cid: [] for cid in ihc_ids}

    if not ihc_ids or not dish_ids:
        return result, set()

    cell_centroids, cell_areas = _centroids_and_areas(
        strict_instance_mask.astype(np.int32, copy=False), ihc_ids
    )
    nuc_centroids, _ = _centroids_and_areas(
        dish_nucleus_mask.astype(np.int32, copy=False), dish_ids
    )

    valid_dish = [d for d in dish_ids if d in nuc_centroids]
    valid_cells = [c for c in ihc_ids if c in cell_centroids]
    if not valid_dish or not valid_cells:
        return result, set()

    nuc_pts = np.array([nuc_centroids[d] for d in valid_dish], dtype=np.float64)
    cell_pts = np.array([cell_centroids[c] for c in valid_cells], dtype=np.float64)
    reaches = np.array(
        [math.sqrt(factor * cell_areas[c] / math.pi) for c in valid_cells],
        dtype=np.float64,
    )

    # 每顆細胞在自身 reach 半徑內的候選核（質心↔質心距離）。
    tree = cKDTree(nuc_pts)
    neighbor_lists = tree.query_ball_point(cell_pts, reaches)

    candidates: List[Tuple[float, int, int]] = []  # (dist, cell_id, dish_id)
    had_candidate: Set[int] = set()
    for ci, nbrs in enumerate(neighbor_lists):
        if not nbrs:
            continue
        cid = valid_cells[ci]
        cy, cx = cell_pts[ci]
        had_candidate.add(cid)
        for nj in nbrs:
            ny, nx = nuc_pts[nj]
            candidates.append((math.hypot(cy - ny, cx - nx), cid, valid_dish[nj]))

    # 最近優先 + lock：全域升冪貪婪，落敗細胞自動往後找仍可用的核。
    candidates.sort(key=lambda t: t[0])
    assigned_cell: Set[int] = set()
    assigned_dish: Set[int] = set()
    for _d, cid, did in candidates:
        if cid in assigned_cell or did in assigned_dish:
            continue
        result[cid].append(did)
        assigned_cell.add(cid)
        assigned_dish.add(did)

    # 0 核細胞再分：曾有候選卻競爭落敗=drop-out（打 X）；從無候選=照常計入(0/0)。
    drop_out_ids: Set[int] = {c for c in had_candidate if c not in assigned_cell}

    n_nocand = sum(1 for c in ihc_ids if not result[c] and c not in had_candidate)
    logger.info(
        "elastic_dish_nucleus_matching(cell-centric): IHC=%d, DISH=%d, "
        "matched(1核)=%d, drop-out(競爭落敗)=%d, 無候選(計入)=%d, factor=%.2f",
        len(ihc_ids), len(dish_ids), len(assigned_cell),
        len(drop_out_ids), n_nocand, factor,
    )
    return result, drop_out_ids
