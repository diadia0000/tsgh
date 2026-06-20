"""M3b DISH 細胞核彈性匹配：以「IHC 細胞為中心」把最近的 DISH 核認領給細胞。

**以細胞為中心 + 重疊優先 + reach 候選**：每顆 IHC 細胞（綠框）的 DISH 核候選有兩
來源——(1)**像素重疊**：凡與綠框有任一像素重疊的 DISH 核（橘框）皆為候選；(2)
**reach 半徑**：把自身面積放大 ``dish_elastic_expand_factor`` 倍後的等效圓半徑
``reach = max(sqrt(factor*area/π), dish_elastic_min_reach_px)``，質心落在此半徑內
的 DISH 核亦為候選（``min_reach_px`` 為絕對下限，用以橋接純飄移、無重疊的情況）。
配對採「一對一、重疊優先、最近優先、lock」：先把**重疊候選對**依細胞質心↔核質心
距離升冪配對並鎖定，再處理**僅 reach 候選對**（同樣距離升冪）；落敗的細胞自動往下
找仍可用的核。每顆細胞最多認領 1 顆核。物理上壓在綠框上的橘核因此一定優先被認領，
直接解決「附近有橘核卻配不到」。

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


def _overlap_pairs(
    strict_instance_mask: np.ndarray,
    dish_nucleus_mask: np.ndarray,
) -> Set[Tuple[int, int]]:
    """回傳所有「IHC 細胞 ↔ DISH 核」有像素重疊的 ``(cell_id, dish_id)`` 配對。

    單次掃描兩張 label mask 同時非零的像素，把 ``(cell, dish)`` 編碼成單一整數後
    取 unique，即得所有重疊配對（不含背景 0）。
    """
    both = (strict_instance_mask > 0) & (dish_nucleus_mask > 0)
    if not both.any():
        return set()
    cells = strict_instance_mask[both].astype(np.int64)
    nuclei = dish_nucleus_mask[both].astype(np.int64)
    base = int(dish_nucleus_mask.max()) + 1
    keys = np.unique(cells * base + nuclei)
    return {(int(k // base), int(k % base)) for k in keys}


def elastic_dish_nucleus_matching(
    dish_nucleus_mask: np.ndarray,
    strict_instance_mask: np.ndarray,
    cfg: object,
) -> Tuple[Dict[int, List[int]], Set[int]]:
    """以細胞為中心、一對一的 DISH 核認領（重疊優先 + reach 候選）。

    流程：
        Step 1: 算每顆 IHC 細胞與 DISH 核的質心；細胞另算面積以決定搜尋半徑。
        Step 2: 候選來源有二——
                (a) **像素重疊**：與細胞有任一像素重疊的核（一律候選、優先配）。
                (b) **reach 半徑**：``reach = max(sqrt(factor*area_c/π), min_reach_px)``，
                    質心落在 reach 內的核為候選。
        Step 3: 蒐集所有 ``(細胞, 核)`` 候選對，先配「重疊對」（依質心距離升冪）、
                再配「僅 reach 對」（同樣升冪），貪婪一對一（先配並 lock，落敗者
                自動往下找可用核）。

    Args:
        dish_nucleus_mask: ``(H, W)`` int，0=背景，1..M=DISH 細胞核 ID。
        strict_instance_mask: ``(H, W)`` int，0=背景，1..N=IHC 細胞 ID。
        cfg: 具 ``dish_elastic_expand_factor``（綠框面積放大倍數）與
            ``dish_elastic_min_reach_px``（reach 絕對下限 px）的配置。

    Returns:
        ``({ihc_cell_id: [assigned_dish_id] 或 []}, drop_out_ids)``，其中
        ``drop_out_ids`` 為「曾有候選核卻競爭落敗、最終 0 核」的 IHC ID 集合
        （competition loser，下游打 X）；從頭無候選的 0 核細胞不在此集合。
    """
    factor = float(cfg.dish_elastic_expand_factor)
    min_reach = float(cfg.dish_elastic_min_reach_px)

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
        [max(math.sqrt(factor * cell_areas[c] / math.pi), min_reach) for c in valid_cells],
        dtype=np.float64,
    )

    # 候選來源 (a)：與細胞像素重疊的核——一律候選、優先配。
    overlap_set = _overlap_pairs(strict_instance_mask, dish_nucleus_mask)

    # 候選來源 (b)：質心落在細胞 reach 半徑內的核。
    tree = cKDTree(nuc_pts)
    neighbor_lists = tree.query_ball_point(cell_pts, reaches)

    cell_idx = {c: i for i, c in enumerate(valid_cells)}
    nuc_idx = {d: j for j, d in enumerate(valid_dish)}

    # (is_reach_only, dist, cell_id, dish_id)：排序鍵讓重疊對(0)永遠先於僅 reach 對(1)。
    candidates: List[Tuple[int, float, int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    had_candidate: Set[int] = set()

    for cid, did in overlap_set:
        ci = cell_idx.get(cid)
        nj = nuc_idx.get(did)
        if ci is None or nj is None:
            continue
        cy, cx = cell_pts[ci]
        ny, nx = nuc_pts[nj]
        candidates.append((0, math.hypot(cy - ny, cx - nx), cid, did))
        seen.add((cid, did))
        had_candidate.add(cid)

    for ci, nbrs in enumerate(neighbor_lists):
        if not nbrs:
            continue
        cid = valid_cells[ci]
        cy, cx = cell_pts[ci]
        for nj in nbrs:
            did = valid_dish[nj]
            if (cid, did) in seen:
                continue
            ny, nx = nuc_pts[nj]
            candidates.append((1, math.hypot(cy - ny, cx - nx), cid, did))
            had_candidate.add(cid)

    # 重疊優先 → 最近優先 + lock：升冪貪婪，落敗細胞自動往後找仍可用的核。
    candidates.sort(key=lambda t: (t[0], t[1]))
    assigned_cell: Set[int] = set()
    assigned_dish: Set[int] = set()
    n_overlap_assigned = 0
    for is_reach_only, _d, cid, did in candidates:
        if cid in assigned_cell or did in assigned_dish:
            continue
        result[cid].append(did)
        assigned_cell.add(cid)
        assigned_dish.add(did)
        if is_reach_only == 0:
            n_overlap_assigned += 1

    # 0 核細胞再分：曾有候選卻競爭落敗=drop-out（打 X）；從無候選=照常計入(0/0)。
    drop_out_ids: Set[int] = {c for c in had_candidate if c not in assigned_cell}

    n_nocand = sum(1 for c in ihc_ids if not result[c] and c not in had_candidate)
    logger.info(
        "elastic_dish_nucleus_matching(overlap-first): IHC=%d, DISH=%d, "
        "matched(1核)=%d (其中重疊=%d), drop-out(競爭落敗)=%d, 無候選(計入)=%d, "
        "factor=%.2f, min_reach=%.1f",
        len(ihc_ids), len(dish_ids), len(assigned_cell), n_overlap_assigned,
        len(drop_out_ids), n_nocand, factor, min_reach,
    )
    return result, drop_out_ids
