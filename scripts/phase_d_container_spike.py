"""Phase 1 spike for docs/hybrid-pipeline/29-phase-d-gpu-port-plan.md §2.3.

Doc 29 asks one question before any pipeline code is written: **is the 19.2x
nvImageCodec encode number (doc 25 §8.2) reachable end-to-end, once the container
assembly nobody had scoped is included?** §2.2 named the two things that had to be
checked first, and this script exists because both now have answers:

  1. **Does a maintained Python TIFF writer accept externally pre-compressed tile
     bytes?**  YES. `tifffile.TiffWriter.write` accepts `data` as `Iterator[bytes]`
     whose members must match the declared `compression`/`predictor`. Verified here
     (`--verify-passthrough`): LZW-compressed tiles handed straight to tifffile
     round-trip bit-identical through a tiled BigTIFF. So doc 29's feared fallback --
     hand-writing IFD offset/bytecount tables -- is NOT needed.

  2. **Can nvImageCodec produce those bytes?**  NO. Its TIFF encoder refuses tiling
     outright (`[WARNING][nvtiff_cuda_encoder] Tiling is not supported with TIFF
     encoder`, encode returns None), and its strip layout is chosen internally to
     hit roughly 8 KB per strip -- measured RowsPerStrip 10 / 5 / 2 / 1 at 256 /
     512 / 1024 / 4096 px wide, i.e. 4,096 separate LZW streams for one 4096^2
     buffer. A TIFF *tile* must be exactly one LZW stream, and independent LZW
     streams cannot be concatenated into one (each ends in an EOI code, so a
     decoder stops at the first). There is therefore no path from nvImageCodec's
     output to a tiled container's tile bytes.

That kills the headline GPU-codec route, and leaves §2.2's own conservative
fallback as the thing worth measuring: **let a CPU library do the compression and
container writing, and ask whether it beats pyvips at all** -- plus whether moving
only the pyramid generation (doc 29 §1.2's ~22% share) to the GPU adds anything.

Candidates, all producing LZW + tiled + pyramidal + BigTIFF (§2.1's hard constraint):

  A  pyvips_tiffsave     what `_stitch_overlay_slide` calls today -- the baseline
  B  tifffile_cpu        imagecodecs LZW per tile (threaded) -> tifffile
                         pre-compressed passthrough; pyramid downsampled on CPU
  C  tifffile_gpu_pyr    same container path, pyramid levels downsampled on GPU
                         (torch), which is the only piece a GPU can still touch

Judged on end-to-end wall for the WHOLE operation (§2.3 step 4: doc 25's number was
encode-only and must not be quoted as an end-to-end figure), against §1.3's corrected
ceiling of ~1.05-1.09x. Correctness is a veto: **every pyramid level** must be
pixel-identical, not just level 0 -- checking only level 0 is what let a predictor
bug through to QuPath, which crashed on it (doc 32 section 5.1).

Usage:
  .venv/bin/python scripts/phase_d_container_spike.py --side 16384 --out spike.json
  .venv/bin/python scripts/phase_d_container_spike.py --verify-passthrough
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logging.getLogger("pyvips").setLevel(logging.WARNING)

import numpy as np
import pyvips
import tifffile
import imagecodecs

# Both of these must match what `tiffsave(tile=True, pyramid=True)` produces, or the
# comparison is between two different outputs and the speedup is partly just "did less
# work". pyvips defaults to 128x128 tiles and halves until the level fits one tile.
TILE = 128
MIN_LEVEL = 128
# pyvips's tiffsave default when no resolution is given: 1 pixel/mm == 25.4 pixels/inch.
# The candidates must declare the same thing or QuPath scales them differently.
PYVIPS_DPI = 25.4


def overlay_like(side: int, seed: int = 0) -> np.ndarray:
    """Overlay-slide-like content: mostly flat white with sparse annotation.

    Same construction as `scripts/gpu_codec_spike.py::_sample`, so LZW compresses
    roughly the way it does on real annotated tiles rather than on noise (which
    would make every codec look equally bad and rank them wrongly).
    """
    rng = np.random.default_rng(seed)
    a = np.full((side, side, 3), 255, dtype=np.uint8)
    n = side * side // 400
    ys = rng.integers(0, side, size=n)
    xs = rng.integers(0, side, size=n)
    a[ys, xs] = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    a[::64, :, :] = 0
    a[:, ::64, :] = 0
    return a


def n_levels(side: int) -> int:
    lv, s = 0, side
    while s // 2 >= MIN_LEVEL:
        s //= 2
        lv += 1
    return lv


def check_shape_matches_baseline(a: Path, b: Path) -> dict:
    """Assert the candidate's container shape equals the baseline's.

    A candidate that writes fewer pyramid levels, or larger tiles, has done less work
    and its speedup is not comparable. Checked rather than assumed.
    """
    with tifffile.TiffFile(str(a)) as fa, tifffile.TiffFile(str(b)) as fb:
        return {
            "baseline_pages": len(fa.pages),
            "candidate_pages": len(fb.pages),
            "pages_match": len(fa.pages) == len(fb.pages),
            "baseline_tile": (fa.pages[0].tilewidth, fa.pages[0].tilelength),
            "candidate_tile": (fb.pages[0].tilewidth, fb.pages[0].tilelength),
            "tile_match": (fa.pages[0].tilewidth == fb.pages[0].tilewidth
                           and fa.pages[0].tilelength == fb.pages[0].tilelength),
        }


# ------------------------------------------------------------------
# A -- the baseline: exactly what _stitch_overlay_slide calls
# ------------------------------------------------------------------
def cand_pyvips(arr: np.ndarray, dst: Path) -> dict:
    img = pyvips.Image.new_from_memory(
        arr.tobytes(), arr.shape[1], arr.shape[0], 3, "uchar")
    t0 = time.perf_counter()
    img.tiffsave(str(dst), tile=True, pyramid=True, compression="lzw", bigtiff=True)
    return {"wall_s": round(time.perf_counter() - t0, 3)}


# ------------------------------------------------------------------
# B / C -- pre-compressed tile bytes into tifffile's container
# ------------------------------------------------------------------
def _tile_bytes(level: np.ndarray, workers: int) -> list[bytes]:
    """LZW-compress every tile of one pyramid level, horizontal predictor applied.

    Predictor 2 is applied here because tifffile is told `predictor=True` and only
    *declares* the tag for pre-compressed input -- it does not transform the bytes.
    Getting this wrong produces a file that opens and decodes to garbage, which is
    exactly the silent-corruption failure mode the correctness veto exists for.
    """
    h, w = level.shape[:2]
    tiles = []
    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            t = level[y:y + TILE, x:x + TILE]
            if t.shape[0] != TILE or t.shape[1] != TILE:      # edge tile: pad
                pad = np.zeros((TILE, TILE, 3), dtype=level.dtype)
                pad[:t.shape[0], :t.shape[1]] = t
                t = pad
            tiles.append(np.ascontiguousarray(t))

    def enc(t: np.ndarray) -> bytes:
        # TIFF predictor 2 is: out[0] = in[0]; out[k] = in[k] - in[k-1], per channel,
        # in modulo-256 arithmetic. The *first* column must keep its absolute value --
        # prepending the first column instead of zero makes out[0] == 0, which silently
        # drops each row's base value and shifts the whole row on decode. That bug shipped
        # in the first version of this spike and QuPath is what caught it; see doc 32 §5.
        zero = np.zeros((t.shape[0], 1, t.shape[2]), dtype=t.dtype)
        d = np.diff(t, axis=1, prepend=zero)
        return imagecodecs.lzw_encode(np.ascontiguousarray(d).tobytes())

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(enc, tiles))
    return [enc(t) for t in tiles]


def _pyramid_cpu(arr: np.ndarray, levels: int) -> list[np.ndarray]:
    out = [arr]
    cur = arr
    for _ in range(levels):
        # 2x2 box shrink, matching pyvips's default pyramid reduction closely enough
        # for a viewer aid; only level 0 carries the correctness veto.
        h, w = (cur.shape[0] // 2) * 2, (cur.shape[1] // 2) * 2
        cur = (cur[:h:2, :w:2].astype(np.uint16) + cur[1:h:2, :w:2] +
               cur[:h:2, 1:w:2] + cur[1:h:2, 1:w:2] + 2) // 4
        cur = cur.astype(np.uint8)
        out.append(cur)
    return out


def _pyramid_gpu(arr: np.ndarray, levels: int) -> list[np.ndarray]:
    import torch
    out = [arr]
    t = torch.from_numpy(arr).cuda().permute(2, 0, 1).unsqueeze(0).float()
    for _ in range(levels):
        t = torch.nn.functional.avg_pool2d(t, 2)
        out.append(t.round().clamp(0, 255).to(torch.uint8)
                   .squeeze(0).permute(1, 2, 0).cpu().numpy())
    torch.cuda.synchronize()
    return out


def cand_tifffile(arr: np.ndarray, dst: Path, gpu_pyramid: bool,
                  workers: int) -> dict:
    lv = n_levels(arr.shape[0])
    t0 = time.perf_counter()
    levels = _pyramid_gpu(arr, lv) if gpu_pyramid else _pyramid_cpu(arr, lv)
    t_pyr = time.perf_counter() - t0

    t1 = time.perf_counter()
    encoded = [_tile_bytes(l, workers) for l in levels]
    t_enc = time.perf_counter() - t1

    t2 = time.perf_counter()
    with tifffile.TiffWriter(str(dst), bigtiff=True) as tif:
        for i, (l, segs) in enumerate(zip(levels, encoded)):
            tif.write(
                iter(segs), shape=l.shape, dtype=l.dtype, tile=(TILE, TILE),
                compression="lzw", predictor=True, photometric="rgb",
                subfiletype=1 if i else 0,
                # Match the baseline's resolution tags. pyvips defaults to 1 px/mm
                # (25.4 px/inch); tifffile defaults to "no unit", which makes QuPath
                # compute a different physical pixel size and display the image at a
                # different scale than the file this is meant to replace.
                resolution=(PYVIPS_DPI, PYVIPS_DPI), resolutionunit="inch",
            )
    t_ctr = time.perf_counter() - t2

    return {
        "wall_s": round(t_pyr + t_enc + t_ctr, 3),
        "pyramid_s": round(t_pyr, 3),
        "encode_s": round(t_enc, 3),
        "container_s": round(t_ctr, 3),
        "levels": lv + 1,
        "_levels": levels,          # popped before serialisation; used by verify()
    }


# ------------------------------------------------------------------
def verify(dst: Path, arr: np.ndarray, levels: list | None = None) -> dict:
    """Correctness veto (§2.3 step 5).

    **Every** pyramid level is checked against what was handed to the writer, not just
    level 0. The first version of this spike checked level 0 only, and level 0 happened
    to be immune to the predictor bug it had (the synthetic image's black grid lines fall
    on every tile boundary, so every tile's first column was 0 either way). The reduced
    levels were corrupt in the shipped file and QuPath crashed on one of them. A
    per-level check is what makes that a caught bug instead of a shipped one.
    """
    out: dict = {}
    with tifffile.TiffFile(str(dst)) as tf:
        p = tf.pages[0]
        got = p.asarray()[:arr.shape[0], :arr.shape[1]]
        out["level0_pixel_identical"] = bool(np.array_equal(arr, got))
        if levels is not None:
            bad = []
            for i, want in enumerate(levels):
                have = tf.pages[i].asarray()[:want.shape[0], :want.shape[1]]
                if not np.array_equal(want, have):
                    bad.append({"level": i,
                                "max_abs_delta": int(np.abs(
                                    have.astype(int) - want.astype(int)).max())})
            out["all_levels_pixel_identical"] = not bad
            out["corrupt_levels"] = bad
        out.update({
            "bigtiff": bool(tf.is_bigtiff),
            "tiled": bool(p.is_tiled),
            "compression": str(p.compression),
            "n_pages": len(tf.pages),
            "resolution": str(p.tags['XResolution'].value if 'XResolution' in p.tags
                              else None),
            "bytes": dst.stat().st_size,
        })
    return out


def verify_passthrough() -> int:
    """§2.2 question 1, standalone: does tifffile really take pre-compressed tiles?"""
    a = overlay_like(1024)
    segs = _tile_bytes(a, workers=1)
    dst = Path("/tmp/_passthrough_check.tif")
    tifffile.imwrite(str(dst), iter(segs), shape=a.shape, dtype=a.dtype,
                     tile=(TILE, TILE), compression="lzw", predictor=True,
                     photometric="rgb", bigtiff=True)
    ok = np.array_equal(a, tifffile.imread(str(dst)))
    print(f"tifffile pre-compressed tile passthrough lossless: {ok}")
    dst.unlink(missing_ok=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=16384,
                    help="square slab edge in px (16384 = 0.27 GP)")
    ap.add_argument("--workers", type=int, default=8,
                    help="threads for the CPU LZW encode in candidates B/C")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--outdir", default="/tmp/_phase_d_spike")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify-passthrough", action="store_true")
    args = ap.parse_args()

    if args.verify_passthrough:
        return verify_passthrough()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"building {args.side}x{args.side} overlay-like slab "
          f"({args.side ** 2 / 1e9:.3f} GP) ...")
    arr = overlay_like(args.side)

    cands = {
        "A_pyvips_tiffsave": lambda d: cand_pyvips(arr, d),
        "B_tifffile_cpu": lambda d: cand_tifffile(arr, d, False, args.workers),
        "C_tifffile_gpu_pyramid": lambda d: cand_tifffile(arr, d, True, args.workers),
    }

    results = {}
    for name, fn in cands.items():
        dst = outdir / f"{name}.tif"
        best = None
        for _ in range(args.reps):
            dst.unlink(missing_ok=True)
            try:
                r = fn(dst)
            except Exception as exc:                      # noqa: BLE001
                r = {"error": repr(exc)}
                break
            if best is None or r["wall_s"] < best["wall_s"]:
                best = r
        r = best or r
        if "error" not in r:
            r.update(verify(dst, arr, r.pop("_levels", None)))
            r["mp_per_s"] = round(args.side ** 2 / 1e6 / r["wall_s"], 1)
        results[name] = r
        print(f"  {name:<26} {json.dumps(r)}")

    base = results.get("A_pyvips_tiffsave", {}).get("wall_s")
    for name, r in results.items():
        if base and "wall_s" in r:
            r["speedup_vs_pyvips"] = round(base / r["wall_s"], 4)
        if name != "A_pyvips_tiffsave" and "error" not in r:
            r["shape_vs_baseline"] = check_shape_matches_baseline(
                outdir / "A_pyvips_tiffsave.tif", outdir / f"{name}.tif")

    payload = {
        "side": args.side,
        "gigapixels": round(args.side ** 2 / 1e9, 4),
        "tile": TILE,
        "workers": args.workers,
        "results": results,
        "nvimgcodec_finding": (
            "TIFF encoder refuses tiling ('Tiling is not supported with TIFF "
            "encoder'); strip layout is internal (~8 KB/strip: RowsPerStrip 10/5/2/1 "
            "at 256/512/1024/4096 px). Tiled pre-compressed passthrough is therefore "
            "impossible from nvImageCodec output."
        ),
    }
    print("\n" + json.dumps({n: {k: v for k, v in r.items()
                                 if k in ("wall_s", "speedup_vs_pyvips", "bytes",
                                          "level0_pixel_identical")}
                             for n, r in results.items()}, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
