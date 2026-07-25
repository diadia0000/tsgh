"""B1 microbenchmark: does stacking N tiles into one Cellpose `eval` call pay off?

Executes step 1/2 of docs/hybrid-pipeline/22-next-optimization-cycle-plan.md §3 (B1).
**Standalone — imports the pipeline but does not modify it and never calls `run_batch`.**

Background (traced from `cellpose==4.2.1.1` source, verified before writing this):
  - `CellposeSegmenter.predict` (m2_segmentation.py:81) calls `model.eval(image_2d, ...)`,
    i.e. always `Lz=1`. At 1024px / `bsize=384` that is `ntiles = 4*4 = 16` patches
    (`ny = ceil(1.2 * 1024/384) = 4`), which already equals `cellpose_batch_size=16` —
    one `_forward` call, nothing left for `batch_size` to group.
  - `models.py:231` loops one-at-a-time when `x` is a **list**, so a list buys nothing.
  - But a genuine stacked `(Lz,H,W,C)` ndarray reaches `core.run_net`, which packs
    `nimgs = batch_size // ntiles` *images'* patches into one `IMGa` buffer (`core.py:207-224`).

So the question this script answers is per-model and empirical: with the same real tile
inputs, does `eval(stack_of_G_tiles, batch_size=G*ntiles)` cost less **per tile** than G
separate `eval(one_tile)` calls? It also splits the cost into the batchable part
(`_run_net`, measured with `compute_masks=False`) and the non-batchable per-slice part
(`_compute_masks`/dynamics), because only the former can benefit.

Usage:
  .venv/bin/python scripts/cellpose_batch_probe.py --tiles 8 --groups 1,2,4,8 \
      --out /path/to/out.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.getLogger("pyvips").setLevel(logging.WARNING)

HYBRID = Path(__file__).resolve().parent.parent / "backend" / "algorithms" / "hybrid"
sys.path.insert(0, str(HYBRID))
sys.path.insert(0, str(HYBRID.parent.parent.parent))

import hybrid_pipeline as HP  # noqa: E402
import torch  # noqa: E402
from config import config  # noqa: E402

ROI = HYBRID / "test_picture" / "_roi_crops"


def _load_tiles(ihc_path: Path, dish_path: Path, n_tiles: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cut the first `n_tiles` tissue-bearing tiles off the crop, on the pipeline's own grid."""
    import pyvips

    ihc_img = pyvips.Image.new_from_file(str(ihc_path), access="random")
    dish_img = pyvips.Image.new_from_file(str(dish_path), access="random")
    ts = config.default_tile_size
    stride = ts - config.window_overlap_px

    pairs = []
    for y in range(0, ihc_img.height - ts + 1, stride):
        for x in range(0, ihc_img.width - ts + 1, stride):
            ihc = np.ndarray(
                buffer=ihc_img.crop(x, y, ts, ts).write_to_memory(),
                dtype=np.uint8, shape=[ts, ts, ihc_img.bands],
            )[:, :, :3]
            # skip near-empty tiles: they short-circuit in the real pipeline anyway
            if ihc.mean() > 235:
                continue
            dish = np.ndarray(
                buffer=dish_img.crop(x, y, ts, ts).write_to_memory(),
                dtype=np.uint8, shape=[ts, ts, dish_img.bands],
            )[:, :, :3]
            pairs.append((np.ascontiguousarray(ihc), np.ascontiguousarray(dish)))
            if len(pairs) >= n_tiles:
                return pairs
    return pairs


def _build_inputs(pairs):
    """Produce the *real* per-model inputs: M2 gets the M1 overlay, M3b gets the raw DISH tile."""
    unet = HP._init_unet_inferencer()
    m2_inputs, m3b_inputs = [], []
    for ihc, dish in pairs:
        core_mask = HP.generate_ihc_core_mask(ihc, unet, close_kernel=config.core_close_kernel)
        if core_mask.sum() == 0:
            continue
        m1 = HP._run_m1_overlay_stage(ihc, dish, core_mask)
        m2_inputs.append(m1.m2_input_overlay)
        m3b_inputs.append(dish)
    del unet
    torch.cuda.empty_cache()
    return m2_inputs, m3b_inputs


def _eval(model, x, batch_size, compute_masks=True):
    return model.eval(
        x,
        batch_size=batch_size,
        diameter=None,
        flow_threshold=config.cellpose_flow_threshold,
        cellprob_threshold=config.cellpose_cellprob_threshold,
        compute_masks=compute_masks,
    )


def _timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def _ntiles_per_image(h, w, bsize=384, tile_overlap=0.1):
    """Mirror of core.run_net's patch-count arithmetic, for reporting only."""
    ny = 1 if h <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * h / bsize))
    nx = 1 if w <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * w / bsize))
    return ny * nx


def probe_model(name, segmenter, inputs, groups, reps, out):
    model = segmenter.model
    h, w = inputs[0].shape[:2]
    ntiles = _ntiles_per_image(h, w)
    print(f"\n=== {name}: {len(inputs)} tiles, {h}x{w}, patches/image={ntiles} ===")

    # warm up (cuDNN autotune, allocator, bfloat16 kernels)
    for _ in range(2):
        _eval(model, inputs[0], config.cellpose_batch_size)
    torch.cuda.synchronize()

    ref_masks = None
    for g in groups:
        n = (len(inputs) // g) * g          # drop the ragged tail so every config sees the same work
        if n == 0:
            continue
        for masks_flag in (True, False):
            per_rep = []
            for _ in range(reps):
                torch.cuda.reset_peak_memory_stats()
                total = 0.0
                collected = []
                for i in range(0, n, g):
                    chunk = inputs[i:i + g]
                    x = chunk[0] if g == 1 else np.stack(chunk, axis=0)
                    dt, res = _timed(
                        lambda: _eval(model, x, config.cellpose_batch_size * g, masks_flag)
                    )
                    total += dt
                    if masks_flag:
                        collected.append(res[0])
                per_rep.append(total)
                peak = torch.cuda.max_memory_allocated() / 1e6
            best = min(per_rep)
            key = "full" if masks_flag else "net_only"
            rec = out.setdefault(name, {}).setdefault(str(g), {})
            rec[key] = {
                "n_tiles": n,
                "total_s": round(best, 4),
                "per_tile_s": round(best / n, 4),
                "reps_s": [round(v, 4) for v in per_rep],
                "peak_alloc_mb": round(peak, 1),
                "batch_size_arg": config.cellpose_batch_size * g,
            }
            print(f"  G={g:2d} {key:8s}: {best/n*1000:8.1f} ms/tile  "
                  f"(total {best:6.2f}s, peak_alloc {peak:7.1f} MB)")
            if masks_flag:
                flat = []
                for m in collected:
                    m = np.asarray(m)
                    flat.extend(list(m) if m.ndim == 3 else [m])
                counts = [int(np.asarray(m).max()) for m in flat]
                rec["cells"] = counts
                if ref_masks is None:
                    ref_masks = flat
                else:
                    diffs = [
                        float((np.asarray(a) != np.asarray(b)).mean())
                        for a, b in zip(ref_masks, flat)
                    ]
                    rec["px_diff_vs_G1"] = [round(d, 6) for d in diffs]
                    print(f"       cells {counts} vs G=1 {out[name]['1']['cells']}, "
                          f"max px diff {max(diffs):.4%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ihc", default=str(ROI / "large_ihc.tiff"))
    ap.add_argument("--dish", default=str(ROI / "large_dish.tiff"))
    ap.add_argument("--tiles", type=int, default=8)
    ap.add_argument("--groups", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    groups = [int(g) for g in args.groups.split(",")]
    print(f"loading {args.tiles} tiles from {args.ihc}")
    pairs = _load_tiles(Path(args.ihc), Path(args.dish), args.tiles)
    print(f"got {len(pairs)} tissue tiles; building real M1/M2 inputs")
    m2_inputs, m3b_inputs = _build_inputs(pairs)
    print(f"M2 inputs: {len(m2_inputs)}, M3b inputs: {len(m3b_inputs)}")

    out = {
        "device": torch.cuda.get_device_name(0),
        "cellpose_batch_size": config.cellpose_batch_size,
        "n_input_tiles": len(m2_inputs),
        "reps": args.reps,
    }
    probe_model("m2_cellpose", HP._init_cellpose_segmenter(), m2_inputs, groups, args.reps, out)
    torch.cuda.empty_cache()
    probe_model("m3b_dish_cellpose", HP._init_dish_cellpose_segmenter(), m3b_inputs, groups,
                args.reps, out)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
