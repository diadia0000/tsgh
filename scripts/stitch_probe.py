"""Candidate A (doc 24 §2.1/§4 item 2): size `_stitch_overlay_slide` at real scale.

Phase D is the one fully serial block left in `run_batch`: a manual pyvips row/column
join feeding a single `tiffsave(lzw, tile, pyramid)`. It costs 5.05 s at the 441-tile
crop (0.46 gigapixel) and doc 24 extrapolates ~3 min at the real slide's 16.2
gigapixels -- an extrapolation nobody has checked. It also runs in the parent process
only, so its share of wall *grows* as `workers>1` shrinks everything else.

This probe measures the real function at a chosen slide size without running any model
inference. It rebuilds the exact tile grid `chunk_offsets` would produce for that slide,
computes each tile's core-crop size with the pipeline's own `core_crop_bounds` (so edge
columns/rows keep their real, different sizes), and materialises the `overlay_annotated`
directory from a pool of REAL overlay tiles: one encoded template per (width, height,
pool member), every other position a hard link. Content therefore compresses like real
output, and building a 27,565-tile input costs seconds instead of an hour.

`--ablate` adds doc 26 Tier 1.1 (19 #1b / DISCOVERED #5): the cheap, dependency-free
`tiffsave` knobs nobody had tried. It builds the input **once**, then re-runs only the
encode across a list of single-knob variants, so each number differs from the baseline
in exactly one parameter (playbook step 4: every layer justifies itself by ablation,
and changing two things at once means you cannot attribute the result). The join is
timed separately from the encode via `_join_overlay_tiles`, because a knob can only
move the encode half.

Usage:
  .venv/bin/python scripts/stitch_probe.py --overlay-src <dir of real overlay tiles> \
      --slide-w 141818 --slide-h 114366 --out out.json

  # single-knob ablation of the tiffsave parameters, at a chosen scale
  .venv/bin/python scripts/stitch_probe.py --overlay-src <dir> --ablate \
      --slide-w 70909 --slide-h 57183 --out ablate_4gp.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import shutil
import sys
import time
from pathlib import Path

# pyvips logs one INFO line per temp-file open; the ablation loop makes that
# hundreds of thousands of lines, which is both noise and measurable I/O.
logging.getLogger("pyvips").setLevel(logging.WARNING)

import pyvips  # noqa: E402

HYBRID = Path(__file__).resolve().parent.parent / "backend" / "algorithms" / "hybrid"
sys.path.insert(0, str(HYBRID))
sys.path.insert(0, str(HYBRID.parent.parent.parent))

import hybrid_pipeline as HP  # noqa: E402
from config import config  # noqa: E402
from m0_reader import chunk_offsets  # noqa: E402
from m0_stitch import compute_tile_geometry, core_crop_bounds  # noqa: E402


def build_inputs(out_dir: Path, positions, geometry, tile: int,
                 pool: list[pyvips.Image], blank_pool: list[pyvips.Image],
                 annotated_share: float) -> dict:
    """Materialise overlay_annotated/ for every position; one encode per size class.

    `annotated_share` sets how many positions get an annotated tile rather than a blank
    placeholder. It is the dominant variable in this measurement -- blank tiles are
    constant fill and compress almost for free, annotated ones do not -- so it has to be
    set to the real slide's composition rather than inherited from whatever crop the
    source tiles came from.
    """
    overlay = out_dir / "overlay_annotated"
    overlay.mkdir(parents=True, exist_ok=True)

    templates: dict[tuple, Path] = {}
    n_encoded = 0
    n_annotated = 0
    t0 = time.perf_counter()
    for i, (ax, ay) in enumerate(positions):
        lx0, lx1, ly0, ly1 = core_crop_bounds(geometry, ax, ay, tile)
        w, h = lx1 - lx0, ly1 - ly0
        # exact share regardless of grid size: advance a counter, not a modulus
        use_annotated = int((i + 1) * annotated_share) > int(i * annotated_share)
        src_pool = pool if use_annotated else blank_pool
        n_annotated += use_annotated
        key = (w, h, use_annotated, i % len(src_pool))
        dst = overlay / f"tile_x{ax}_y{ay}.tiff"
        src = templates.get(key)
        if src is None:
            img = src_pool[i % len(src_pool)]
            # tile the source out to at least (w,h), then crop: keeps real texture
            reps_x = -(-w // img.width)
            reps_y = -(-h // img.height)
            big = img.replicate(reps_x, reps_y) if (reps_x > 1 or reps_y > 1) else img
            # `_save_tile_array` goes through skimage.io.imsave, which writes an
            # uncompressed TIFF -- match that, or the stitch would be reading inputs
            # that decode differently from the real ones.
            big.crop(0, 0, w, h).write_to_file(str(dst), compression="none")
            templates[key] = dst
            n_encoded += 1
        else:
            os.link(src, dst)
    return {
        "n_positions": len(positions),
        "n_annotated": n_annotated,
        "annotated_share": round(n_annotated / len(positions), 4),
        "n_encoded_templates": n_encoded,
        "build_s": round(time.perf_counter() - t0, 2),
        "input_bytes": sum(f.stat().st_size for f in templates.values()),
    }


# The shipped call, and one single-knob deviation from it per row. Keep this list
# single-knob: two changes at once cannot be attributed (playbook anti-pattern #7).
BASELINE = dict(tile=True, pyramid=True, compression="lzw", bigtiff=True)

ABLATIONS: list[tuple[str, dict, str]] = [
    ("baseline", {}, "the shipped call: 128x128 tiles, full pyramid, LZW, BigTIFF"),
    ("tile_256", dict(tile_width=256, tile_height=256), "fewer, larger tiles"),
    ("tile_512", dict(tile_width=512, tile_height=512), "fewer, larger tiles"),
    ("tile_1024", dict(tile_width=1024, tile_height=1024),
     "one tile per source tile core"),
    ("predictor_horizontal", dict(predictor="horizontal"),
     "LZW with horizontal differencing (usually the big LZW win on photos)"),
    ("predictor_none", dict(predictor="none"), "LZW with no differencing"),
    ("depth_onetile", dict(depth="onetile"),
     "stop the pyramid once a level fits one tile (fewer levels than the default)"),
    ("shrink_nearest", dict(region_shrink="nearest"),
     "cheapest downsample kernel for the pyramid levels"),
    ("subifd", dict(subifd=True), "pyramid levels as sub-IFDs rather than pages"),
    ("deflate", dict(compression="deflate"), "still lossless, different codec"),
    ("zstd_1", dict(compression="zstd", level=1), "still lossless, different codec"),
    # Bounds, not candidates -- both change the artifact, so they are vetoed on
    # correctness/usability regardless of speed. They exist to size the ceiling.
    ("BOUND_no_pyramid", dict(pyramid=False),
     "VETOED (QuPath needs the pyramid) -- measures what pyramid generation costs"),
    ("BOUND_no_compression", dict(compression="none"),
     "VETOED (output size) -- measures what compression costs"),
]


def run_ablation(out_dir, geometry, gp: float, only: list[str] | None) -> list[dict]:
    """Time join once, then each single-knob encode variant on that same image.

    The joined image is lazy, so it must be rebuilt per variant (a pyvips pipeline is
    consumed by the sink); the join is re-timed each round and reported separately so
    the encode delta is not contaminated by it.
    """
    rows: list[dict] = []
    dst = out_dir / "overlay_slide.tiff"
    for name, override, note in ABLATIONS:
        if only and name not in only:
            continue
        kwargs = dict(BASELINE, **override)

        t0 = time.perf_counter()
        slide = HP._join_overlay_tiles(out_dir, geometry)
        join_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        try:
            slide.tiffsave(str(dst), **kwargs)
        except pyvips.Error as exc:
            rows.append({"name": name, "note": note, "error": str(exc).strip()[:200]})
            print(f"  {name:22} FAILED: {str(exc).strip()[:90]}")
            continue
        encode_s = time.perf_counter() - t1
        size = dst.stat().st_size
        dst.unlink()

        rows.append({
            "name": name, "note": note, "override": override,
            "join_s": round(join_s, 3),
            "encode_s": round(encode_s, 3),
            "total_s": round(join_s + encode_s, 3),
            "s_per_gigapixel": round((join_s + encode_s) / gp, 3),
            "output_bytes": size,
        })
        print(f"  {name:22} join {join_s:7.2f}s  encode {encode_s:7.2f}s  "
              f"total {join_s + encode_s:7.2f}s  {size / 1e9:6.2f} GB")

    base = next((r for r in rows if r["name"] == "baseline" and "error" not in r), None)
    if base:
        for r in rows:
            if "error" in r:
                continue
            r["speedup_vs_baseline"] = round(base["total_s"] / r["total_s"], 4)
            r["size_vs_baseline"] = round(r["output_bytes"] / base["output_bytes"], 4)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay-src", required=True,
                    help="directory of real overlay_annotated tiles to sample from")
    ap.add_argument("--slide-w", type=int, default=141818)
    ap.add_argument("--slide-h", type=int, default=114366)
    ap.add_argument("--pool", type=int, default=24,
                    help="distinct source tiles per class (annotated / blank)")
    ap.add_argument("--annotated-share", type=float, default=0.4475,
                    help="fraction of positions that get an annotated tile; default is "
                         "the slide's measured tissue share (core_mask_map.py)")
    ap.add_argument("--root", default=None, help="scratch root for the synthetic input")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ablate", action="store_true",
                    help="single-knob ablation of the tiffsave parameters (doc 26 Tier 1.1) "
                         "instead of one baseline timing")
    ap.add_argument("--only", default=None,
                    help="comma-separated ablation names to run (default: all)")
    args = ap.parse_args()

    tile = config.default_tile_size
    overlap = config.window_overlap_px
    positions = chunk_offsets(args.slide_h, args.slide_w, tile, overlap)
    geometry = compute_tile_geometry(positions, tile, overlap)
    cols = len(geometry.col_of)
    rows = len(geometry.row_of)

    srcs = sorted(Path(args.overlay_src).glob("*.tiff"))
    if not srcs:
        raise SystemExit(f"no overlay tiles in {args.overlay_src}")
    # Split the source tiles into annotated vs blank placeholders. A blank tile is the
    # constant background fill plus thin seam dashes, so its mean sits just under 255
    # (measured: 254.45 vs 190-197 for annotated tiles); its deviation is NOT near zero,
    # because the seam dashes alone move it to ~11.8.
    # Load into memory: each pool member is read many times (once per size class), and
    # a "sequential" pyvips image can only be consumed once, in order.
    annotated, blank = [], []
    for p in srcs:
        img = pyvips.Image.new_from_file(str(p), access="random").copy_memory()
        (blank if img.avg() > 250.0 else annotated).append(img)
        if len(annotated) >= args.pool and len(blank) >= args.pool:
            break
    pool, blank_pool = annotated[: args.pool], blank[: args.pool]
    print(f"source pool: {len(pool)} annotated, {len(blank_pool)} blank "
          f"(of {len(srcs)} tiles in {args.overlay_src})")
    if not pool or not blank_pool:
        raise SystemExit("need both annotated and blank source tiles")

    root = Path(args.root) if args.root else Path(config.output_dir) / "_stitch_probe"
    out_dir = root / f"{args.slide_w}x{args.slide_h}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"grid {cols}x{rows} = {len(positions)} tiles for a "
          f"{args.slide_w}x{args.slide_h} slide "
          f"({args.slide_w * args.slide_h / 1e9:.2f} gigapixels)")
    build = build_inputs(out_dir, positions, geometry, tile, pool, blank_pool,
                         args.annotated_share)
    print(f"  built input in {build['build_s']}s "
          f"({build['n_encoded_templates']} encoded, rest hard-linked)")

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    gp = args.slide_w * args.slide_h / 1e9
    result = {
        "slide": [args.slide_w, args.slide_h],
        "gigapixels": round(gp, 3),
        "grid": [cols, rows],
        "n_tiles": len(positions),
        "build": build,
        "rlimit_nofile": [soft, hard],
    }

    if args.ablate:
        only = args.only.split(",") if args.only else None
        print(f"\nablating tiffsave knobs at {gp:.2f} GP "
              f"({len(positions)} tiles, {build['annotated_share']:.2%} annotated):")
        result["ablations"] = run_ablation(out_dir, geometry, gp, only)
    else:
        t0 = time.perf_counter()
        HP._stitch_overlay_slide(out_dir, geometry)
        stitch_s = time.perf_counter() - t0
        result["stitch_s"] = round(stitch_s, 3)
        result["s_per_gigapixel"] = round(stitch_s / gp, 3)
        result["output_bytes"] = (out_dir / "overlay_slide.tiff").stat().st_size

    print(json.dumps(result, indent=2))

    if not args.keep:
        shutil.rmtree(out_dir)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
