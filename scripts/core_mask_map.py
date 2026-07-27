"""Label every grid cell of the slide tissue/background by the pipeline's own rule.

doc 23 §1 estimated the slide is ~39% background from a brightness thumbnail; doc 24's
whole composition argument (§0.1-§0.4) is built on that number. But the pipeline's rule
is not brightness -- a tile is background iff `generate_ihc_core_mask` comes back empty
(`_process_one_chunk_gpu`, hybrid_pipeline.py:576), which is what decides whether the
two Cellpose forwards and the entire BG arm run. This script runs that exact rule over
the **whole** grid and saves the map, so the composition is measured rather than
estimated, and so a composition-matched crop can be selected exactly instead of via a
brightness proxy.

Only the M1 forward runs (no cellpose, no BG arm). Tile cutting is prefetched on a
thread pool so the GPU is not waiting on pyvips.

Usage:
.venv/bin/python scripts/core_mask_map.py --ihc <slide> --out map.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyvips

HYBRID = Path(__file__).resolve().parent.parent / "backend" / "algorithms" / "hybrid"
sys.path.insert(0, str(HYBRID))
sys.path.insert(0, str(HYBRID.parent.parent.parent))

import hybrid_pipeline as HP  # noqa: E402
from config import config  # noqa: E402
from m0_reader import chunk_offsets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ihc", required=True)
    ap.add_argument("--prefetch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="stop after N tiles (debug)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tile = config.default_tile_size
    overlap = config.window_overlap_px
    stride = tile - overlap

    img = pyvips.Image.new_from_file(args.ihc, access="random")
    if img.bands == 4:
        img = img.extract_band(0, n=3)
    positions = chunk_offsets(img.height, img.width, tile, overlap)
    if args.limit:
        positions = positions[: args.limit]
    xs = sorted({x for x, _ in positions})
    ys = sorted({y for _, y in positions})
    col_of = {x: i for i, x in enumerate(xs)}
    row_of = {y: i for i, y in enumerate(ys)}
    print(f"grid {len(xs)}x{len(ys)} = {len(positions)} tiles")

    def cut(pos):
        x, y = pos
        w = min(tile, img.width - x)
        h = min(tile, img.height - y)
        crop = img.crop(x, y, w, h)
        if (w, h) != (tile, tile):
            crop = crop.gravity("north-west", tile, tile, extend="white")
        return pos, np.ndarray(buffer=crop.write_to_memory(), dtype=np.uint8,
                            shape=(tile, tile, 3))

    unet = HP._init_unet_inferencer()
    core_px = np.zeros((len(ys), len(xs)), dtype=np.int64)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.prefetch) as pool:
        for k, (pos, arr) in enumerate(pool.map(cut, positions), start=1):
            mask = HP.generate_ihc_core_mask(
                arr, unet, close_kernel=config.core_close_kernel)
            x, y = pos
            core_px[row_of[y], col_of[x]] = int(mask.sum())
            if k % 500 == 0:
                el = time.perf_counter() - t0
                print(f"  {k}/{len(positions)}  {el:.0f}s  "
                    f"eta {el / k * (len(positions) - k):.0f}s", flush=True)

    bg = core_px == 0
    summary = {
        "slide": [img.width, img.height],
        "grid": [len(xs), len(ys)],
        "n_tiles": int(bg.size),
        "n_background": int(bg.sum()),
        "background_share": round(float(bg.mean()), 4),
        "tissue_share": round(float(1 - bg.mean()), 4),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    print(json.dumps(summary, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, core_px=core_px, xs=np.array(xs), ys=np.array(ys))
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
