"""M5 — BigTIFF slide-level writer (pyvips backend).

Supports uint8 (JPEG) and uint32 (LZW) buffers, single-band or 3-band.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pyvips

logger = logging.getLogger("full_wsi")

_VIPS_FORMAT: dict[np.dtype, str] = {
    np.dtype("uint8"): "uchar",
    np.dtype("uint32"): "uint",
}


class BigTiffWriter:
    """Slide-level tiled BigTIFF writer.

    Tiles are accumulated in a numpy array. On close(), the buffer is
    written via pyvips as a tiled (pyramidal) BigTIFF.

    Args:
        path:          Output path.
        height:        Slide height in pixels.
        width:         Slide width in pixels.
        dtype:         Buffer dtype — np.uint8 (JPEG) or np.uint32 (LZW).
        bands:         Number of channels. 1 for masks, 3 for RGB overlay.
        pyramidal:     Write sub-IFD pyramid levels.
        jpeg_quality:  JPEG Q (only used when dtype=uint8).
        tile_size:     Tile size in pixels.

    Peak RAM = height × width × bands × itemsize bytes.
    """

    def __init__(
        self,
        path: Path,
        height: int,
        width: int,
        dtype: type = np.uint8,
        *,
        bands: int = 1,
        pyramidal: bool = True,
        jpeg_quality: int = 85,
        tile_size: int = 256,
    ) -> None:
        self.path = path
        self.height = height
        self.width = width
        self._dtype = np.dtype(dtype)
        self._bands = bands
        self._pyramidal = pyramidal
        self._jpeg_quality = jpeg_quality
        self._tile_size = tile_size

        shape = (height, width, bands) if bands > 1 else (height, width)
        needed_gib = height * width * bands * self._dtype.itemsize / (1024 ** 3)
        logger.info(
            "BigTiffWriter %s: %.2f GiB buffer (%s x%d bands, pyramidal=%s)",
            path.name, needed_gib, self._dtype, bands, pyramidal,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: Optional[np.ndarray] = np.zeros(shape, dtype=self._dtype)

    def write(self, y0: int, x0: int, patch: np.ndarray) -> None:
        h, w = patch.shape[:2]
        y1 = min(y0 + h, self.height)
        x1 = min(x0 + w, self.width)
        self._buffer[y0:y1, x0:x1] = patch[:y1 - y0, :x1 - x0]

    def close(self) -> None:
        buf = self._buffer
        # pyvips new_from_memory requires explicit band dimension
        if buf.ndim == 2:
            buf = buf[:, :, np.newaxis]
        h, w, b = buf.shape

        vips_fmt = _VIPS_FORMAT.get(self._dtype, "uchar")
        # Pass a memoryview (zero-copy) instead of tobytes() to avoid doubling peak RAM.
        img = pyvips.Image.new_from_memory(np.ascontiguousarray(buf).data, w, h, b, vips_fmt)

        use_jpeg = self._dtype == np.dtype("uint8")
        save_kwargs: dict = dict(
            tile=True,
            tile_width=self._tile_size,
            tile_height=self._tile_size,
            pyramid=self._pyramidal,
            compression="jpeg" if use_jpeg else "lzw",
            bigtiff=True
        )
        if use_jpeg:
            save_kwargs["Q"] = self._jpeg_quality

        img.tiffsave(str(self.path), **save_kwargs)
        del self._buffer
        self._buffer = None
        size_mb = self.path.stat().st_size / (1024 ** 2)
        logger.info(
            "%s written: %.1f MB (pyramidal=%s)",
            self.path.name, size_mb, self._pyramidal,
        )
