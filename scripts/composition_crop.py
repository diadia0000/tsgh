"""Build a *composition-matched* crop: a rectangular tile grid whose tissue /
background mix matches the real slide's, instead of a crop's.

Doc 24 §0.1/§4 item 1: every crop measured in rounds 1-6 is ~86% tissue, while the
real slide is ~61% tissue / ~39% background. Costs that scale with *background*-tile
count (doc 24 Candidate F) are therefore under-represented ~2.8x in every anchor
this project has. This script picks the crop window whose stride-768 grid hits the
real ratio, so one measurement run answers the composition question directly instead
of extrapolating a blended per-tile rate.

Method: read a pyramid level of the aligned IHC slide (never the full 10 GB level 0),
reduce it to one mean-brightness value per stride-768 grid cell, label a cell
"background" when that mean is >= the background level, then slide a GxG window over
the map (via an integral image) and take the window whose background share is closest
to the target. The crop is then cut at full resolution with the same pyvips path
scripts/crop_roi.py uses, so the analysis path stays bit-identical to a real WSI crop.

The brightness rule is a *proxy* for what the pipeline actually does (a tile is
background iff UNet++'s core mask comes back empty); the realized ratio is reported
by run_batch as skipped/total and must be checked against the prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvips


def cell_means(path: Path, level: int, stride: int) -> np.ndarray:
    """Mean brightness per stride-px grid cell, computed on a pyramid level."""
    full = pyvips.Image.new_from_file(str(path))
    img = pyvips.Image.new_from_file(str(path), subifd=level) if level >= 0 else full
    scale = full.width / img.width
    cell = stride / scale
    # Pyramid levels round each halving, so the scale is only approximately a power
    # of two (141818/8863 = 16.0011). Rounding the cell size costs ~0.007% per cell,
    # i.e. under a pixel of drift across the whole map -- irrelevant to a brightness
    # proxy, but refuse anything larger so a wrong --level can't pass silently.
    if abs(cell - round(cell)) / cell > 0.01:
        raise ValueError(
            f"stride {stride} at level {level} (scale {scale:g}) is {cell:g} px "
            f"per cell -- too far from an integer; pick another level."
        )
    cell = int(round(cell))

    grey = img.colourspace("b-w")
    # shrink = box filter = exact per-cell mean; the trailing partial cell is dropped
    # by vips, which matches the grid's last row/col being snapped rather than padded.
    small = grey.shrink(cell, cell)
    arr = np.ndarray(
        buffer=small.write_to_memory(),
        dtype=np.uint8,
        shape=(small.height, small.width),
    )
    return arr.astype(np.float32)


def best_window(bg: np.ndarray, grid: int, target: float,
                max_row: int, max_col: int):
    """GxG window whose background share is closest to `target` (integral image).

    `max_row`/`max_col` bound the top-left cell so the full-resolution crop stays
    inside *both* slides (IHC and DISH differ by ~160 px in this dataset).
    """
    ii = np.zeros((bg.shape[0] + 1, bg.shape[1] + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(bg.astype(np.int64), axis=0), axis=1)
    h, w = bg.shape
    if h < grid or w < grid:
        raise ValueError(f"map {h}x{w} smaller than requested {grid}x{grid} grid")
    sums = (
        ii[grid:, grid:] - ii[:-grid, grid:] - ii[grid:, :-grid] + ii[:-grid, :-grid]
    )
    shares = sums / float(grid * grid)
    shares = shares[:max_row + 1, :max_col + 1]
    if shares.size == 0:
        raise ValueError("no window fits inside both slides at this grid size")
    err = np.abs(shares - target)
    r, c = np.unravel_index(np.argmin(err), err.shape)
    return int(r), int(c), float(shares[r, c])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ihc", required=True)
    ap.add_argument("--dish", required=True)
    ap.add_argument("--grid", type=int, default=24, help="GxG tiles in the crop")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--level", type=int, default=3, help="pyramid subifd to analyse")
    ap.add_argument("--bg-level", type=float, default=188.0,
                    help="cell mean >= this is background; the default is the "
                         "threshold scripts/tissue_calibrate.py found best "
                         "reproduces the pipeline's own empty-core-mask rule")
    ap.add_argument("--target-bg", type=float, default=0.5525,
                    help="background share to match; the default is the measured "
                         "slide-wide share (tissue_calibrate.py, n=800)")
    ap.add_argument("--map", default=None,
                    help="core_mask_map.py .npz; when given, cells are labelled by "
                         "the pipeline's own empty-core-mask rule and the brightness "
                         "proxy is not used at all")
    ap.add_argument("--out-ihc", default=None)
    ap.add_argument("--out-dish", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    stride = args.tile - args.overlap
    if args.map:
        core_px = np.load(args.map)["core_px"]
        bg = core_px == 0
        label_rule = "empty core mask (measured)"
    else:
        means = cell_means(Path(args.ihc), args.level, stride)
        bg = means >= args.bg_level
        label_rule = f"cell mean >= {args.bg_level} (brightness proxy)"
    slide_bg = float(bg.mean())

    size = (args.grid - 1) * stride + args.tile   # exactly `grid` starts per axis
    ihc_img = pyvips.Image.new_from_file(args.ihc)
    dish_img = pyvips.Image.new_from_file(args.dish)
    limit_w = min(ihc_img.width, dish_img.width)
    limit_h = min(ihc_img.height, dish_img.height)
    max_col = (limit_w - size) // stride
    max_row = (limit_h - size) // stride

    r, c, share = best_window(bg, args.grid, args.target_bg, max_row, max_col)
    x0, y0 = c * stride, r * stride

    report = {
        "map_shape": list(bg.shape),
        "label_rule": label_rule,
        "slide_background_share": round(slide_bg, 4),
        "slide_tissue_share": round(1 - slide_bg, 4),
        "grid": args.grid,
        "n_tiles": args.grid ** 2,
        "crop": {"x": x0, "y": y0, "size": size},
        "window_background_share": round(share, 4),
        "window_tissue_share": round(1 - share, 4),
        "target_bg": args.target_bg,
    }
    print(json.dumps(report, indent=2))

    if args.out_ihc and args.out_dish:
        for src, dst in ((args.ihc, args.out_ihc), (args.dish, args.out_dish)):
            img = pyvips.Image.new_from_file(src, access="random")
            w = min(size, img.width - x0)
            h = min(size, img.height - y0)
            if (w, h) != (size, size):
                raise ValueError(
                    f"{src}: window {size} at ({x0},{y0}) exceeds {img.width}x{img.height}"
                )
            crop = img.crop(x0, y0, w, h)
            if crop.bands == 4:
                crop = crop.extract_band(0, n=3)
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            crop.write_to_file(dst, compression="deflate")
            print(f"wrote {dst} ({w}x{h})")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
