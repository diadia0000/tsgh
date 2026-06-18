"""M3b DISH 細胞核彈性匹配：把 DISH 核 instance 認領給 IHC 細胞。

從 m3_dot_detection.py 拆出。以「競爭消解的最佳指派」取代舊 greedy：可行
(IHC, DISH) pair 切成連通分量，每分量解 1:1 最佳配對（最近者優先、競爭落敗的
IHC 自動改配次近的核），再把落單的核補配出去。最終每顆 IHC 的 matched 核數：
1=valid；>=2=多核（排除）；0 再細分——曾有可行候選卻在競爭中落敗且無備援核者
=drop-out（排除、打 X），從頭就沒有可行候選者=忽略配對、照常計入分析。
詳見 docs/sdd-elastic-dish-matching.md 與 docs/elastic_matching_v2_explainer.html。
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.ndimage import center_of_mass, find_objects
from scipy.optimize import linear_sum_assignment
from skimage.morphology import binary_dilation, disk

logger = logging.getLogger(__name__)


def elastic_dish_nucleus_matching(
    dish_nucleus_mask: np.ndarray,
    strict_instance_mask: np.ndarray,
    cfg: object,
) -> Tuple[Dict[int, List[int]], Set[int]]:
    """彈性匹配：IHC region 等向膨脹 → 候選收集 → 競爭消解最佳指派。

    流程（詳見 docs/sdd-elastic-dish-matching.md）：
        Step 1: 對每個 IHC 細胞 region 計算等向 dilation radius，使面積放大至
                ``dish_elastic_expand_factor`` 倍（近似圓形推導：
                r = (sqrt(A*f) - sqrt(A)) / sqrt(π)）。
        Step 2: 找出膨脹後 region 與 DISH 核的重疊 pixel，收集 candidate DISH ID。
        Step 3: 競爭消解最佳指派（取代舊 greedy）。可行 pair 依連通分量切分，
                每分量以 ``linear_sum_assignment`` 解 1:1 最佳配對（最大配對數 →
                最小總距離）；最近者優先，競爭落敗的 IHC 自動改配次近的核。
        Step 4: phase-1 後仍無人認領的 DISH 核補配給最近的可行 IHC；經最大配對
                保證，該核必落在已配對細胞的 territory，故使其成為多核。超過
                ``dish_elastic_max_dist_px`` 的 pair 全程不納入。

    最終每顆 IHC 的核數：1=valid；>=2=多核（排除）；0 再分兩種——曾有可行候選卻
    競爭落敗且無備援核=drop-out（排除）；從頭就沒有可行候選=忽略配對、照常計入。

    Returns:
        ``({ihc_cell_id: [assigned_dish_id, ...]}, drop_out_ids)``，其中
        ``drop_out_ids`` 為「核數 0 但曾有可行候選」的 IHC ID 集合（competition
        loser，下游打 X）；核數 0 但從無可行候選的 IHC 不在此集合（照常計入）。
    """
    expand_factor = float(getattr(cfg, "dish_elastic_expand_factor", 1.5))
    max_dist_px = float(getattr(cfg, "dish_elastic_max_dist_px", 50.0))

    ihc_ids: List[int] = [int(v) for v in np.unique(strict_instance_mask) if v != 0]
    dish_ids: List[int] = [int(v) for v in np.unique(dish_nucleus_mask) if v != 0]
    result: Dict[int, List[int]] = {cid: [] for cid in ihc_ids}

    if not ihc_ids or not dish_ids:
        return result, set()

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

    # 可行 pair = 候選且 centroid 距離 <= max_dist。had_feasible 記錄「至少有一個
    # 可行候選」的 IHC，用以區分競爭落敗的 drop-out 與「從頭就沒有候選」。
    feasible: List[Tuple[float, int, int]] = []
    had_feasible: Set[int] = set()
    for cid, candidates in cell_candidates.items():
        iy, ix = ihc_centroids[cid]
        for did in candidates:
            dy, dx = dish_centroids[did]
            dist = math.hypot(iy - dy, ix - dx)
            if dist <= max_dist_px:
                feasible.append((dist, cid, did))
                had_feasible.add(cid)

    # 把可行 pair 切成連通分量：互相競爭的鄰近 IHC/DISH 才落在同一分量，
    # 各分量規模小可獨立求全域最佳解（分量間無邊，per-分量最佳 = 全域最佳），
    # 避免在整張 strip 的 IHC×DISH 上做 O(n^3) dense 指派。
    parent: Dict[Tuple[str, int], Tuple[str, int]] = {}

    def _find(x: Tuple[str, int]) -> Tuple[str, int]:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def _union(a: Tuple[str, int], b: Tuple[str, int]) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for _dist, cid, did in feasible:
        _union(("I", cid), ("D", did))

    comp_edges: Dict[Tuple[str, int], List[Tuple[float, int, int]]] = {}
    for dist, cid, did in feasible:
        comp_edges.setdefault(_find(("I", cid)), []).append((dist, cid, did))

    # Phase 1：每分量做 1:1 最佳指派（最大配對數，再最小化總距離）。非可行 pair
    # 給極大成本、事後濾除；linear_sum_assignment 會優先用可行邊，故等同
    # 「最大配對數 + 最小距離」——每顆 IHC 先拿到最該屬於它的那一顆核。
    big = max_dist_px * 1e6 + 1.0
    assigned_dish_ids: set = set()
    for edges in comp_edges.values():
        comp_ihcs = sorted({cid for _d, cid, _did in edges})
        comp_dishes = sorted({did for _d, _cid, did in edges})
        i_idx = {cid: k for k, cid in enumerate(comp_ihcs)}
        d_idx = {did: k for k, did in enumerate(comp_dishes)}
        cost = np.full((len(comp_ihcs), len(comp_dishes)), big, dtype=float)
        for dist, cid, did in edges:
            cost[i_idx[cid], d_idx[did]] = dist
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] <= max_dist_px:
                result[comp_ihcs[r]].append(comp_dishes[c])
                assigned_dish_ids.add(comp_dishes[c])

    # Phase 2：phase-1 後仍無人認領的 DISH 核 → 補配給最近的可行 IHC。
    # 經最大配對保證：落單核的可行鄰居必為「已配對」細胞，補上即代表該細胞
    # territory 內不只一顆核 → 多核（下游 blue_region_count>=threshold 打 X）。
    leftover: Dict[int, Tuple[float, int]] = {}
    for dist, cid, did in feasible:
        if did in assigned_dish_ids:
            continue
        prev = leftover.get(did)
        if prev is None or dist < prev[0]:
            leftover[did] = (dist, cid)
    for did, (_dist, cid) in leftover.items():
        result[cid].append(did)
        assigned_dish_ids.add(did)

    # 核數 0 再細分：曾有可行候選=競爭落敗的 drop-out（打 X）；從無可行候選=照常計入。
    drop_out_ids: Set[int] = {
        cid for cid in ihc_ids if not result[cid] and cid in had_feasible
    }

    exclude_thr = int(getattr(cfg, "dot_blue_exclude_threshold", 2))
    n_valid = sum(1 for v in result.values() if len(v) == 1)
    n_multi = sum(1 for v in result.values() if len(v) >= exclude_thr)
    n_dropout = len(drop_out_ids)
    n_nocand = sum(
        1 for cid, v in result.items() if not v and cid not in had_feasible
    )
    logger.info(
        "elastic_dish_nucleus_matching: IHC=%d, DISH=%d, valid(1核)=%d, "
        "多核(>=%d)=%d, drop-out(競爭落敗)=%d, 無候選(計入)=%d, "
        "expand_factor=%.1f, max_dist=%.1fpx",
        len(ihc_ids), len(dish_ids), n_valid, exclude_thr, n_multi,
        n_dropout, n_nocand, expand_factor, max_dist_px,
    )
    return result, drop_out_ids
