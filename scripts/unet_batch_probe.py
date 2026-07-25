"""B1b microbenchmark: does stacking N tiles into one UNet++ forward pay off?

Companion to `cellpose_batch_probe.py`, for docs/hybrid-pipeline/22-next-optimization-cycle-plan.md
§3 (B1b). **Standalone — does not modify the pipeline.**

At the pipeline's 1024px tile size the tile equals `unet_image_size`, so
`UNetPPInference.predict_single` takes the `_predict_direct` branch (one image, batch of 1);
`_predict_sliding_window`'s `self.batch_size` grouping is never reached. This measures the
only batching axis actually available at this tile size: stacking G separate *tiles* into a
single `(G,3,1024,1024)` forward, versus G separate forwards.

Amdahl context (do not read a win here as a wall-clock win): the UNet++ forward is
~13.68 s / 441 tiles ≈ 0.4% of the round-3 anchor wall.

Usage:
  .venv/bin/python scripts/unet_batch_probe.py --tiles 8 --groups 1,2,4,8 --out out.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cellpose_batch_probe import ROI, _load_tiles, _timed  # noqa: E402


def _forward_group(inf, images):
    """One forward over G stacked tiles, using the inferencer's own pre/post-processing."""
    tensors, sizes = [], []
    for img in images:
        t, sz = inf.preprocess(img)
        tensors.append(t)
        sizes.append(sz)
    batch = torch.cat(tensors, dim=0).to(inf.device)
    out = inf.model(batch)
    masks = []
    for i, (h, w) in enumerate(sizes):
        pred = out[i:i + 1].argmax(dim=1).squeeze(0)
        masks.append(pred.cpu().numpy().astype(np.uint8)[:h, :w])
    return masks


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
    pairs = _load_tiles(Path(args.ihc), Path(args.dish), args.tiles)
    images = [ihc for ihc, _ in pairs]
    print(f"{len(images)} tiles, tile={images[0].shape}, unet_image_size={config.unet_image_size}, "
          f"config.batch_size={config.batch_size}")

    inf = HP._init_unet_inferencer()
    out = {"device": torch.cuda.get_device_name(0), "n_input_tiles": len(images),
           "reps": args.reps, "unet_batch_size_cfg": config.batch_size}

    with torch.inference_mode():
        for _ in range(3):                       # warm up cuDNN autotune
            _forward_group(inf, images[:1])
        torch.cuda.synchronize()

        ref = None
        for g in groups:
            n = (len(images) // g) * g
            if n == 0:
                continue
            per_rep, collected = [], []
            for r in range(args.reps):
                torch.cuda.reset_peak_memory_stats()
                total, collected = 0.0, []
                for i in range(0, n, g):
                    dt, masks = _timed(lambda: _forward_group(inf, images[i:i + g]))
                    total += dt
                    collected.extend(masks)
                per_rep.append(total)
                peak = torch.cuda.max_memory_allocated() / 1e6
            best = min(per_rep)
            rec = {"n_tiles": n, "total_s": round(best, 4), "per_tile_s": round(best / n, 4),
                   "reps_s": [round(v, 4) for v in per_rep], "peak_alloc_mb": round(peak, 1)}
            if ref is None:
                ref = collected
            else:
                rec["px_diff_vs_G1"] = round(
                    float(max((a != b).mean() for a, b in zip(ref, collected))), 8)
            out[str(g)] = rec
            print(f"  G={g:2d}: {best/n*1000:7.2f} ms/tile (total {best:5.2f}s, "
                  f"peak_alloc {peak:7.1f} MB"
                  + (f", max px diff vs G=1 {rec['px_diff_vs_G1']:.6%}" if ref is not collected else "")
                  + ")")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
