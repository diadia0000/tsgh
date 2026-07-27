"""Calibrate the brightness proxy for "background tile" against the real rule.

`scripts/composition_crop.py` labels a grid cell background from its mean brightness;
the pipeline labels a tile background when UNet++'s core mask comes back empty
(`_process_one_chunk_gpu`, hybrid_pipeline.py:576). Those are not the same rule, and
doc 24's whole composition argument rests on the second one. This script samples grid
cells from the real slide, cuts each tile exactly as `PrecutStream` does, runs the
real M1 forward, and reports:

  * the sampled true background share (the number doc 23 §1 estimated at ~39%),
  * the brightness threshold that best reproduces the true labels,
  * the confusion of the proxy at that threshold.

M1 only -- no cellpose, no BG arm -- so a few hundred tiles cost a couple of minutes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyvips

HYBRID = Path(__file__).resolve().parent.parent / "backend" / "algorithms" / "hybrid"
sys.path.insert(0, str(HYBRID))
sys.path.insert(0, str(HYBRID.parent.parent.parent))

import hybrid_pipeline as HP  # noqa: E402
from composition_crop import cell_means  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ihc", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stride = args.tile - args.overlap
    means = cell_means(Path(args.ihc), args.level, stride)
    rows, cols = means.shape

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(rows * cols, size=min(args.n, rows * cols), replace=False)

    img = pyvips.Image.new_from_file(args.ihc, access="random")
    if img.bands == 4:
        img = img.extract_band(0, n=3)
    unet = HP._init_unet_inferencer()

    samples = []
    t0 = time.perf_counter()
    for k, flat in enumerate(idx, start=1):
        r, c = divmod(int(flat), cols)
        x, y = c * stride, r * stride
        w = min(args.tile, img.width - x)
        h = min(args.tile, img.height - y)
        crop = img.crop(x, y, w, h)
        if (w, h) != (args.tile, args.tile):
            crop = crop.gravity("north-west", args.tile, args.tile, extend="white")
        tile = np.ndarray(
            buffer=crop.write_to_memory(), dtype=np.uint8,
            shape=(args.tile, args.tile, 3),
        )
        core = HP.generate_ihc_core_mask(
            tile, unet, close_kernel=HP.config.core_close_kernel,
        )
        samples.append({
            "x": x, "y": y,
            "cell_mean": float(means[r, c]),
            "core_px": int(core.sum()),
            "background": bool(core.sum() == 0),
        })
        if k % 25 == 0:
            print(f"  {k}/{len(idx)}  ({time.perf_counter() - t0:.1f}s)", flush=True)

    bright = np.array([s["cell_mean"] for s in samples])
    bg = np.array([s["background"] for s in samples])

    # threshold sweep: proxy says background when cell_mean >= t
    best = None
    for t in np.arange(150, 226, 1.0):
        pred = bright >= t
        acc = float((pred == bg).mean())
        if best is None or acc > best["accuracy"]:
            best = {
                "threshold": float(t),
                "accuracy": round(acc, 4),
                "predicted_bg_share": round(float(pred.mean()), 4),
                "false_bg": int((pred & ~bg).sum()),
                "false_tissue": int((~pred & bg).sum()),
            }

    n = len(samples)
    share = float(bg.mean())
    # binomial 95% CI, normal approximation -- enough to say whether 0.39 is plausible
    half = 1.96 * (share * (1 - share) / n) ** 0.5
    report = {
        "n_sampled": n,
        "true_background_share": round(share, 4),
        "ci95": [round(max(0.0, share - half), 4), round(min(1.0, share + half), 4)],
        "true_tissue_share": round(1 - share, 4),
        "best_brightness_threshold": best,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"report": report, "samples": samples}, indent=2))


if __name__ == "__main__":
    main()
