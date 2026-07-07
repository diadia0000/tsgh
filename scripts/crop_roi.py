"""Crop an aligned square ROI from the warped IHC/DISH WSI pair (pyvips, lossless).

Used to build medium/large regime-validation inputs without loading the full
20GB+ WSI into RAM. Output tiles are written as deflate TIFF (same codec the
precut stage uses), keeping the analysis path bit-identical to a real WSI crop.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyvips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ihc", required=True)
    ap.add_argument("--dish", required=True)
    ap.add_argument("--x", type=int, required=True)
    ap.add_argument("--y", type=int, required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--out-ihc", required=True)
    ap.add_argument("--out-dish", required=True)
    args = ap.parse_args()

    ihc = pyvips.Image.new_from_file(args.ihc, access="random")
    dish = pyvips.Image.new_from_file(args.dish, access="random")
    print(f"IHC full: {ihc.width}x{ihc.height} bands={ihc.bands}")
    print(f"DISH full: {dish.width}x{dish.height} bands={dish.bands}")

    w = min(args.size, ihc.width - args.x, dish.width - args.x)
    h = min(args.size, ihc.height - args.y, dish.height - args.y)
    print(f"cropping {w}x{h} at ({args.x},{args.y})")

    ic = ihc.crop(args.x, args.y, w, h)
    dc = dish.crop(args.x, args.y, w, h)
    if ic.bands == 4:
        ic = ic.extract_band(0, n=3)
    if dc.bands == 4:
        dc = dc.extract_band(0, n=3)

    Path(args.out_ihc).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_dish).parent.mkdir(parents=True, exist_ok=True)
    ic.write_to_file(args.out_ihc, compression="deflate")
    dc.write_to_file(args.out_dish, compression="deflate")
    print(f"wrote {args.out_ihc} + {args.out_dish}")


if __name__ == "__main__":
    main()
