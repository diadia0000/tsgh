"""Framework-free DeepZoom tile source over OpenSlide-readable slides
(pyramidal TIFF / SVS / ...).

backend/api/tiles.py is the thin HTTP wrapper; this module does the actual
image reading + tiling and must NOT import any web framework (guardrail 1,
docs/UI/04-guardrails-red-lines.md). It also only speaks DeepZoom
level/col/row -- viewport<->pixel conversion never happens here (guardrail 4;
that lives in api/roi.py).

Slides are addressed by `slide_id` (guardrail 2: the frontend never sees a
filesystem path). A slide_id resolves to `<SLIDES_DIR>/<slide_id>.<ext>`;
set TSGH_SLIDES_DIR to point elsewhere (defaults to `<repo>/slides`).
"""
from __future__ import annotations

import io
import os
import threading
from pathlib import Path

from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator

# OpenSeadragon-friendly defaults; overlap=1 avoids seams between tiles.
TILE_SIZE = 256
OVERLAP = 1
JPEG_QUALITY = 80

SLIDES_DIR = Path(
    os.environ.get("TSGH_SLIDES_DIR", Path(__file__).resolve().parents[2] / "slides")
)
_SUPPORTED = (".tiff", ".tif", ".svs", ".ndpi", ".mrxs", ".vms", ".scn", ".bif")

_cache: dict[str, DeepZoomGenerator] = {}
_lock = threading.Lock()


class SlideNotFound(Exception):
    """slide_id did not resolve to a readable slide file under SLIDES_DIR."""


def _resolve(slide_id: str) -> Path:
    # Reject anything that could escape SLIDES_DIR before touching the disk.
    if not slide_id or "/" in slide_id or "\\" in slide_id or ".." in slide_id:
        raise SlideNotFound(slide_id)
    for ext in _SUPPORTED:
        path = SLIDES_DIR / f"{slide_id}{ext}"
        if path.is_file():
            return path
    raise SlideNotFound(slide_id)


def resolve(slide_id: str) -> Path:
    """Absolute filesystem path for a slide_id under SLIDES_DIR, or raise
    SlideNotFound. Public so other endpoints (e.g. hybrid) can take slide_ids
    instead of filesystem paths (guardrail 2) via this single resolver."""
    return _resolve(slide_id)


def _get_dz(slide_id: str) -> DeepZoomGenerator:
    with _lock:
        dz = _cache.get(slide_id)
        if dz is None:
            slide = OpenSlide(str(_resolve(slide_id)))
            dz = DeepZoomGenerator(
                slide, tile_size=TILE_SIZE, overlap=OVERLAP, limit_bounds=True
            )
            _cache[slide_id] = dz
        return dz


def invalidate(slide_id: str) -> None:
    """Drop a cached DeepZoomGenerator so the next request re-reads the file
    from disk -- needed when a slide_id's underlying file is replaced in place
    (e.g. a pipeline re-run overwriting a fixed result filename)."""
    with _lock:
        _cache.pop(slide_id, None)


def list_slides() -> list[str]:
    """Every readable slide under SLIDES_DIR, addressed by slide_id (guardrail 2:
    the frontend picks from ids, never filesystem paths). Sorted for a stable
    picker order."""
    if not SLIDES_DIR.is_dir():
        return []
    ids = {
        p.stem
        for p in SLIDES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED
    }
    return sorted(ids)


def get_dzi(slide_id: str) -> str:
    """The .dzi XML descriptor OpenSeadragon fetches first to bootstrap."""
    return _get_dz(slide_id).get_dzi("jpeg")


def get_tile(slide_id: str, level: int, col: int, row: int) -> bytes:
    """One DeepZoom tile as JPEG bytes. Raises ValueError for out-of-range
    level/address (openslide's own contract)."""
    tile = _get_dz(slide_id).get_tile(level, (col, row))
    buf = io.BytesIO()
    tile.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()
