"""Re-weight a measured run to the real slide's tissue/background composition.

doc 24 §0.4 built its full-WSI arm model by splitting each bucket by the population it
actually runs on (all tiles vs tissue tiles only) and re-multiplying by the real slide's
populations -- but with the crop's own rates and an *estimated* background share. This
script does the same arithmetic from a real measurement:

  * bucket rates come from a `perf_measure.py --mp-workers 1` run (the only mode where
    the per-function buckets are populated at all),
  * the populations come from that run's own tile counts, so "per tissue tile" and
    "per tile" are never mixed up,
  * the slide populations come from `core_mask_map.py` (measured) or --background-share.

Output: the per-population rate table, the projected MAIN/BG arm totals at full-WSI
scale, and the resulting `max(MAIN, BG) + outside` wall estimate at workers=1.

Usage:
  .venv/bin/python scripts/wsi_projection.py --timings <label>_timings.json \
      --slide-tiles 27565 --background-share 0.5525
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Which population each bucket runs on, read off hybrid_pipeline.py:
#   ALL     -- every tile (the UNet++ core-mask forward, the tile read)
#   TISSUE  -- only tiles whose core mask is non-empty
#   BG_TILE -- only tiles whose core mask is empty
ALL, TISSUE, BG_TILE = "all", "tissue", "background"
POP = {
    "B1_unet_coremask": ALL,
    "B2r_tile_read": ALL,
    "B4_gc_collect": ALL,
    "B4_empty_cache": ALL,
    "B1_m3b_cellpose": TISSUE,
    "B1_m2_cellpose": TISSUE,
    "BM1_apply_mask": TISSUE,
    "BM1_overlay_dish": TISSUE,
    "BM1_fuse": TISSUE,
    "Bs_clear_edge": TISSUE,
    "B3_build_results": TISSUE,
    "B3_enlarge_cells": TISSUE,
    "B3_detect_dots": TISSUE,
    "B3_merge_dots": TISSUE,
    "Bs_filter_absolutize": TISSUE,
    "B2_png_encode": TISSUE,
    "B2_tiff_encode": TISSUE,
    "B2_render_overlay": TISSUE,
    "B2_percell_crops": TISSUE,
    "F_write_blank_tile": BG_TILE,
}
# Arm membership (same source of truth as scripts/arm_report.py).
MAIN_BUCKETS = {
    "B1_unet_coremask", "B1_m3b_cellpose", "B1_m2_cellpose", "B2r_tile_read",
    "BM1_apply_mask", "BM1_overlay_dish", "BM1_fuse", "Bs_clear_edge",
    "B3_build_results", "B3_enlarge_cells", "B4_gc_collect", "B4_empty_cache",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timings", required=True)
    ap.add_argument("--slide-tiles", type=int, default=27565)
    ap.add_argument("--background-share", type=float, required=True)
    ap.add_argument("--stitch-s", type=float, default=None,
                    help="measured full-scale Phase D cost (scripts/stitch_probe.py); "
                         "defaults to scaling the run's own D_stitch_overlay by pixels")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = json.load(open(args.timings))
    T = d["timings"]
    n_tiles = d["n_tiles"]
    n_bg = (T.get("F_write_blank_tile") or {}).get("n", 0)
    n_tissue = n_tiles - n_bg
    if n_tissue <= 0:
        raise SystemExit("run contains no tissue tiles")

    slide_bg = round(args.slide_tiles * args.background_share)
    slide_tissue = args.slide_tiles - slide_bg
    counts = {ALL: (n_tiles, args.slide_tiles),
              TISSUE: (n_tissue, slide_tissue),
              BG_TILE: (n_bg, slide_bg)}

    rows = []
    for bucket, pop in POP.items():
        rec = T.get(bucket)
        if not rec or rec["t"] <= 0:
            continue
        run_n, slide_n = counts[pop]
        rate = rec["t"] / run_n
        rows.append({
            "bucket": bucket,
            "arm": "MAIN" if bucket in MAIN_BUCKETS else "BG",
            "population": pop,
            "run_s": round(rec["t"], 3),
            "rate_s": round(rate, 5),
            "wsi_s": round(rate * slide_n, 1),
        })
    rows.sort(key=lambda r: -r["wsi_s"])

    main_s = sum(r["wsi_s"] for r in rows if r["arm"] == "MAIN")
    bg_s = sum(r["wsi_s"] for r in rows if r["arm"] == "BG")

    run_px_ratio = args.slide_tiles / n_tiles
    stitch = (args.stitch_s if args.stitch_s is not None
              else (T.get("D_stitch_overlay") or {}).get("t", 0.0) * run_px_ratio)
    init = sum((T.get(k) or {}).get("t", 0.0)
               for k in ("init_unet", "init_cellpose_m2", "init_cellpose_m3b"))
    outside = stitch + init
    wall = max(main_s, bg_s) + outside

    print(f"run {d['label']}: {n_tiles} tiles = {n_tissue} tissue + {n_bg} background "
          f"({n_bg / n_tiles:.1%} background)")
    print(f"slide: {args.slide_tiles} tiles = {slide_tissue} tissue + {slide_bg} "
          f"background ({args.background_share:.1%} background)\n")
    print(f"{'bucket':24s} {'arm':5s} {'pop':11s} {'run s':>8s} {'rate s':>9s} "
          f"{'WSI s':>9s} {'WSI min':>8s}")
    for r in rows:
        print(f"{r['bucket']:24s} {r['arm']:5s} {r['population']:11s} "
              f"{r['run_s']:8.2f} {r['rate_s']:9.5f} {r['wsi_s']:9.1f} "
              f"{r['wsi_s'] / 60:8.1f}")
    print(f"\nMAIN  {main_s / 60:8.1f} min\nBG    {bg_s / 60:8.1f} min  "
          f"(BG/MAIN = {bg_s / main_s:.3f})")
    print(f"outside {outside / 60:6.1f} min  (stitch {stitch / 60:.1f} + init "
          f"{init / 60:.2f})")
    print(f"wall = max(MAIN,BG) + outside = {wall / 3600:.2f} h  at workers=1")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "source": d["label"],
            "run_population": {"tiles": n_tiles, "tissue": n_tissue, "background": n_bg},
            "slide_population": {"tiles": args.slide_tiles, "tissue": slide_tissue,
                                 "background": slide_bg,
                                 "background_share": args.background_share},
            "rows": rows,
            "main_s": round(main_s, 1), "bg_s": round(bg_s, 1),
            "outside_s": round(outside, 1), "wall_s": round(wall, 1),
            "wall_h": round(wall / 3600, 3),
        }, indent=2))


if __name__ == "__main__":
    main()
