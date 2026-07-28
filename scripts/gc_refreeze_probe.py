"""Synthetic accumulation harness for docs/hybrid-pipeline/28-gc-collect-round2-plan.md.

Doc 16 shipped `gc.freeze()` (Option C) and measured per-call `gc.collect()` at
1.2 ms on a 441-tile crop. Doc 27 §6.4 then measured the same call at **80.5 ms**
on the real 27,565-tile slide (2,218.4 s = 16.1% of wall): `run_batch` appends
356,255 `CellAnalysisResult` dataclasses to `per_tile_owned` *after* the one
`gc.freeze()` ran, so every one of them is tracked and rescanned by every
subsequent collection. Doc 28 Option H proposes calling `gc.freeze()` again,
periodically, from inside the existing `_frozen_gc_generation()` scope.

This harness runs doc 28 §5 Exp 0/1/2 **without a GPU, a model, or slide data**,
so the cadence can be designed in seconds instead of behind a 3.8-hour run:

  Exp 0  the climb curve -- per-call gc.collect() cost vs accumulated object
         count, no re-freeze. Reproduces doc 27's regression qualitatively.
  Exp 1  gc.freeze()'s **own** repeated cost at cadence 1/10/100/1000 tiles.
         This is doc 28 §3's one unverified assumption: if freeze's cost scales
         with the *cumulative* tracked count rather than the count *new since
         the last freeze*, Option H reproduces the problem it is solving.
  Exp 2  cadence sweep -- total gc.collect()+gc.freeze() cost as a function of
         the re-freeze interval, keyed by tile count and by accumulated cell
         count, at full-slide-equivalent object counts.

What the harness deliberately does NOT model: the transient per-tile numpy /
torch garbage the real loop creates. That garbage is short-lived and roughly
constant per tile, so it shifts every curve here by the same additive offset and
cannot change the *shape* the cadence is designed against. Absolute ms figures
here are therefore not comparable to doc 27's; the shape and the crossover are.

Usage:
  python scripts/gc_refreeze_probe.py [--out FILE.json] [--tiles N] [--quick]
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HYBRID = REPO / "backend" / "algorithms" / "hybrid"
sys.path.insert(0, str(HYBRID))
sys.path.insert(0, str(REPO))

from hybrid_data_types import CellAnalysisResult  # noqa: E402

# Real full-slide shape (doc 27 §6.4): 27,565 tiles, 55.8% background (0 cells),
# 356,255 cells total -> 29.24 cells per tissue tile.
FULL_TILES = 27_565
BG_SHARE = 0.558
CELLS_PER_TISSUE_TILE = 29.24

# The resident object graph `gc.freeze()` was originally aimed at: three model
# graphs, weights, closures. Fabricated so Exp 0's *pre-freeze* baseline is a
# realistic control rather than an empty process.
RESIDENT_OBJECTS = 300_000


def _resident_graph(n: int) -> list:
    """A tracked, long-lived object graph standing in for the three GPU models."""
    return [{"w": [0.0] * 4, "i": i} for i in range(n // 2)]


def _make_owned(k: int, base: int) -> list:
    """One tile's `owned` list, of the exact dataclass run_batch accumulates."""
    return [
        CellAnalysisResult(
            cell_id=base + i,
            centroid_x=float(i),
            centroid_y=float(base),
            is_her2_positive=bool(i & 1),
        )
        for i in range(k)
    ]


def _cells_for_tile(idx: int, bg_share: float, cells: float) -> int:
    """Deterministic tissue/background alternation matching the slide's ratio."""
    # Spread background tiles evenly rather than clumping them at one end, so the
    # accumulation curve grows smoothly the way the real interleaved slide does.
    if ((idx * (1.0 - bg_share)) % 1.0) >= (1.0 - bg_share):
        return 0
    return int(cells)


def simulate(
    n_tiles: int,
    refreeze_every_tiles: int | None = None,
    refreeze_every_cells: int | None = None,
    initial_freeze: bool = True,
    resident: int = RESIDENT_OBJECTS,
    sample_every: int = 1,
) -> dict:
    """Drive run_batch's accumulate + per-tile `gc.collect()` structure.

    Mirrors the real loop's ordering: one `gc.collect()` per tile, then the
    `_record` append. `refreeze_every_*` adds Option H's extra `gc.freeze()`.
    Returns per-call collect timings, freeze timings, and the curve samples.
    """
    models = _resident_graph(resident)
    per_tile_owned: list = []

    collect_ms: list[float] = []
    freeze_ms: list[float] = []
    curve: list[tuple[int, float]] = []          # (accumulated cells, ms)
    accumulated = 0
    since_freeze_cells = 0
    n_freezes = 0

    gc.collect()
    if initial_freeze:
        gc.freeze()
    try:
        for idx in range(n_tiles):
            k = _cells_for_tile(idx, BG_SHARE, CELLS_PER_TISSUE_TILE)

            t0 = time.perf_counter()
            gc.collect()
            dt = (time.perf_counter() - t0) * 1000.0
            collect_ms.append(dt)
            if idx % sample_every == 0:
                curve.append((accumulated, round(dt, 4)))

            owned = _make_owned(k, accumulated)
            per_tile_owned.append((idx, idx, owned))
            accumulated += k
            since_freeze_cells += k

            due = (
                (refreeze_every_tiles is not None
                 and (idx + 1) % refreeze_every_tiles == 0)
                or (refreeze_every_cells is not None
                    and since_freeze_cells >= refreeze_every_cells)
            )
            if due:
                t0 = time.perf_counter()
                gc.freeze()
                freeze_ms.append((time.perf_counter() - t0) * 1000.0)
                n_freezes += 1
                since_freeze_cells = 0
    finally:
        gc.unfreeze()

    # Keep the accumulation alive until after unfreeze, exactly as run_batch does
    # (the list is consumed by _finish_batch), then drop it.
    n_live = len(per_tile_owned)
    del per_tile_owned, models
    gc.collect()

    tail = collect_ms[-max(1, len(collect_ms) // 20):]
    return {
        "n_tiles": n_tiles,
        "n_tiles_recorded": n_live,
        "accumulated_cells": accumulated,
        "n_freezes": n_freezes,
        "collect_total_s": round(sum(collect_ms) / 1000.0, 4),
        "collect_first_ms": round(collect_ms[0], 4),
        "collect_last5pct_mean_ms": round(statistics.fmean(tail), 4),
        "collect_mean_ms": round(statistics.fmean(collect_ms), 4),
        "freeze_total_s": round(sum(freeze_ms) / 1000.0, 6),
        "freeze_mean_ms": round(statistics.fmean(freeze_ms), 6) if freeze_ms else 0.0,
        "freeze_max_ms": round(max(freeze_ms), 6) if freeze_ms else 0.0,
        "combined_total_s": round((sum(collect_ms) + sum(freeze_ms)) / 1000.0, 4),
        "curve": curve,
    }


# ------------------------------------------------------------------
# Exp 0 -- the climb
# ------------------------------------------------------------------
def exp0(tiles: int, control_tiles: int) -> dict:
    """Per-call collect cost vs accumulated objects, with and without the
    original one-shot freeze. The no-freeze arm is doc 28's control group.

    Collect cost is O(accumulated), so a whole run is O(n^2) and the never-frozen
    control is the most expensive arm by far. It is capped at `control_tiles`
    because it exists to establish a shape, not a full-slide number -- doc 16
    already measured the never-frozen cost on real data.
    """
    frozen = simulate(tiles, initial_freeze=True, sample_every=max(1, tiles // 60))
    unfrozen = simulate(control_tiles, initial_freeze=False,
                        sample_every=max(1, control_tiles // 60))
    climb = (frozen["collect_last5pct_mean_ms"] / frozen["collect_first_ms"]
             if frozen["collect_first_ms"] else 0.0)
    return {
        "frozen_once": frozen,
        "never_frozen_control": unfrozen,
        "climb_factor_first_to_last": round(climb, 1),
    }


# ------------------------------------------------------------------
# Exp 1 -- freeze's own cost
# ------------------------------------------------------------------
def exp1(tiles: int) -> dict:
    """Does gc.freeze() cost scale with the cumulative frozen count, or only
    with what is newly tracked since the last freeze?

    Cadence 1 freezes ~29 new objects against a permanent generation that grows
    to hundreds of thousands; cadence 1000 freezes ~29,000 new ones. If freeze
    is O(cumulative), per-call cost rises with cadence *and* with run length; if
    it is O(new) it tracks the batch size; if it is O(1) neither moves it.
    """
    runs = {}
    for every in (1, 10, 100, 1000):
        r = simulate(tiles, refreeze_every_tiles=every, sample_every=tiles)
        r.pop("curve", None)
        runs[f"every_{every}_tiles"] = r
    means = [runs[k]["freeze_mean_ms"] for k in runs]
    spread = (max(means) / min(means)) if min(means) > 0 else float("inf")
    return {
        "runs": runs,
        "freeze_mean_ms_by_cadence": {k: runs[k]["freeze_mean_ms"] for k in runs},
        "max_over_min_freeze_cost": round(spread, 3) if spread != float("inf") else None,
        "note": "cadence 1 freezes ~29 objects/call, cadence 1000 ~29,000; a flat "
                "per-call cost across the four is evidence freeze is O(1) in both "
                "the cumulative and the new-object count",
    }


# ------------------------------------------------------------------
# Exp 2 -- cadence sweep
# ------------------------------------------------------------------
def exp2(tiles: int) -> dict:
    """Total collect+freeze cost vs re-freeze interval, tile- and cell-keyed."""
    by_tiles, by_cells = {}, {}
    for every in (1, 5, 25, 100, 500, 2000, None):
        r = simulate(tiles, refreeze_every_tiles=every, sample_every=tiles)
        r.pop("curve", None)
        by_tiles["none" if every is None else str(every)] = r
    for every in (100, 500, 2_000, 10_000, 50_000):
        r = simulate(tiles, refreeze_every_cells=every, sample_every=tiles)
        r.pop("curve", None)
        by_cells[str(every)] = r
    return {"by_tile_count": by_tiles, "by_cell_count": by_cells}


def _fmt_row(label: str, r: dict) -> str:
    return (f"  {label:>18}  collect {r['collect_total_s']:8.3f} s   "
            f"freeze {r['freeze_total_s']:7.4f} s ({r['n_freezes']:5d}x)   "
            f"combined {r['combined_total_s']:8.3f} s   "
            f"per-call first {r['collect_first_ms']:7.3f} -> "
            f"last5% {r['collect_last5pct_mean_ms']:7.3f} ms")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=FULL_TILES,
                    help=f"tiles to simulate (default {FULL_TILES} = the real slide)")
    ap.add_argument("--quick", action="store_true",
                    help="5,000 tiles instead of the full slide, for a fast check")
    ap.add_argument("--control-tiles", type=int, default=5_000,
                    help="cap on Exp 0's never-frozen control arm (O(n^2), and it "
                         "only has to establish a shape)")
    ap.add_argument("--out", default=None, help="write the full JSON here")
    args = ap.parse_args()

    tiles = 5_000 if args.quick else args.tiles
    control_tiles = min(args.control_tiles, tiles)
    print(f"gc re-freeze probe -- {tiles} tiles, {BG_SHARE:.1%} background, "
          f"{CELLS_PER_TISSUE_TILE} cells/tissue tile, python {sys.version.split()[0]}")

    t0 = time.perf_counter()
    print("\n=== Exp 0: the climb (doc 27's regression, reproduced) ===")
    e0 = exp0(tiles, control_tiles)
    print(_fmt_row("frozen once", e0["frozen_once"]))
    print(_fmt_row(f"never frozen ({control_tiles}t)", e0["never_frozen_control"]))
    print(f"  climb over the batch: {e0['climb_factor_first_to_last']}x")

    print("\n=== Exp 1: gc.freeze()'s own repeated cost ===")
    e1 = exp1(tiles)
    for k, v in e1["freeze_mean_ms_by_cadence"].items():
        print(f"  {k:>18}  freeze mean {v:.6f} ms/call   "
              f"max {e1['runs'][k]['freeze_max_ms']:.6f} ms")
    print(f"  max/min across cadences: {e1['max_over_min_freeze_cost']}")

    print("\n=== Exp 2: cadence sweep ===")
    e2 = exp2(tiles)
    print("  -- keyed by tile count --")
    for k, v in e2["by_tile_count"].items():
        print(_fmt_row(f"every {k}", v))
    print("  -- keyed by accumulated cell count --")
    for k, v in e2["by_cell_count"].items():
        print(_fmt_row(f"every {k} cells", v))

    result = {
        "tiles": tiles,
        "python": sys.version,
        "bg_share": BG_SHARE,
        "cells_per_tissue_tile": CELLS_PER_TISSUE_TILE,
        "resident_objects": RESIDENT_OBJECTS,
        "exp0_climb": e0,
        "exp1_freeze_cost": e1,
        "exp2_cadence_sweep": e2,
        "probe_wall_s": round(time.perf_counter() - t0, 1),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")
    print(f"probe wall: {result['probe_wall_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
