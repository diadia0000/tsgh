"""Convert aligned pipeline TIFFs into OpenSlide-friendly viewer copies.

The alignment pipeline writes pyramidal TIFFs with `subifd=True` (VALIS needs
that to re-read the pyramid). OpenSlide -- which the tile server uses -- does not
descend into SubIFDs, so it reports those files as `level_count=1` and the viewer
reads the full-resolution level to fill an overview tile, which exhausts RAM and
crashes the backend. See docs/UI/09-viewer-tiff-subifd.md for the full story.

This script produces the required *extra* copy: a `subifd=False` (pyvips default)
+ tiled + pyramidal TIFF that OpenSlide reads as a real pyramid. It does NOT touch
the source file -- the subifd=True original stays for VALIS.

Usage (run inside the tsgh conda env, which has pyvips):
    python scripts/make_viewer_copy.py INPUT.tiff [INPUT2.tiff ...] --out-dir DIR

Each INPUT is written to  <out-dir>/<input-stem>_viewer.tiff .
Point the backend at the output dir with  TSGH_SLIDES_DIR=<out-dir> .
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyvips


def make_viewer_copy(src: Path, dst: Path, quality: int, tile: int) -> None:
    """Write a subifd=False + tiled pyramid copy of `src` at `dst`."""
    image = pyvips.Image.new_from_file(str(src), access="sequential")
    # subifd defaults to False here -> pyramid levels go on the main IFD chain,
    # which is exactly what OpenSlide enumerates.
    image.tiffsave(
        str(dst),
        compression="jpeg",
        Q=quality,
        tile=True,
        tile_width=tile,
        tile_height=tile,
        pyramid=True,
        bigtiff=True,
    )


def _verify(dst: Path) -> str:
    """Report OpenSlide's level_count so the caller can confirm the pyramid is
    visible (should be > 1). Returns a human-readable status line."""
    try:
        import openslide
    except ImportError:
        return "  (openslide not importable; skipped level_count check)"
    try:
        slide = openslide.OpenSlide(str(dst))
        n = slide.level_count
        slide.close()
    except Exception as exc:  # noqa: BLE001 -- just for the status line
        return f"  level_count check failed: {exc}"
    flag = "OK" if n > 1 else "WARNING: still 1 -- OpenSlide sees no pyramid"
    return f"  OpenSlide level_count = {n}  [{flag}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="aligned .tiff/.ome.tiff source(s)")
    parser.add_argument("--out-dir", required=True, type=Path, help="where to write *_viewer.tiff")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (default 85)")
    parser.add_argument("--tile", type=int, default=256, help="tile size (default 256)")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rc = 0
    for src in args.inputs:
        if not src.is_file():
            print(f"SKIP  {src}  (not a file)", file=sys.stderr)
            rc = 1
            continue
        dst = args.out_dir / f"{src.stem}_viewer.tiff"
        print(f"[convert] {src.name} -> {dst.name}")
        t0 = time.perf_counter()
        make_viewer_copy(src, dst, args.quality, args.tile)
        secs = time.perf_counter() - t0
        size_gb = dst.stat().st_size / 1024**3
        print(f"  done in {secs/60:.2f} min, {size_gb:.2f} GB")
        print(_verify(dst))

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
