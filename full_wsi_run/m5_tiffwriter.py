"""M6 — BigTIFF slide-level mask writer (pyvips backend)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pyvips

logger = logging.getLogger("full_wsi")

_NUMPY_TO_VIPS = {
    np.dtype("uint8"): "uchar",
    np.dtype("uint32"): "uint",
}


class BigTiffWriter:
    """Slide-level mask writer backed by pyvips.

    Patches are accumulated in a raw numpy memmap; ``close()`` converts it to
    a tiled (optionally JPEG-pyramid) BigTIFF via ``pyvips.rawload → tiffsave``,
    which streams the data without loading the whole slide into RAM.

    Both pyramidal and non-pyramidal outputs share the same code path —
    the only difference is whether JPEG compression and ``pyramid=True`` are
    passed to ``tiffsave``.
    """

    def __init__(
        self,
        path: Path,
        height: int,
        width: int,
        dtype: np.dtype,
        *,
        pyramidal: bool = False,
        jpeg_quality: int = 85,
        tile_size: int = 256,
    ) -> None:
        dtype = np.dtype(dtype)
        if dtype not in _NUMPY_TO_VIPS:
            raise ValueError(
                f"Unsupported dtype {dtype}; supported: {list(_NUMPY_TO_VIPS)}"
            )
        if pyramidal and dtype != np.dtype(np.uint8):
            raise ValueError(
                f"pyramidal JPEG requires uint8, got {dtype}; "
                "pass pyramidal=False for non-uint8 masks"
            )

        self.path = path
        self.height = height
        self.width = width
        self.dtype = dtype
        self._pyramidal = pyramidal
        self._jpeg_quality = jpeg_quality
        self._tile_size = tile_size
        self._vips_fmt = _NUMPY_TO_VIPS[dtype]

        needed_gib = height * width * dtype.itemsize / (1024 ** 3)
        logger.info(
            "BigTiffWriter %s: %.2f GiB raw buffer (dtype=%s, pyramidal=%s)",
            path.name, needed_gib, dtype, pyramidal,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        self._raw_path = path.with_name(path.name + ".raw")
        self._mm = np.memmap(
            str(self._raw_path), dtype=self.dtype, mode="w+",
            shape=(height, width),
        )

    def write(self, y0: int, x0: int, patch: np.ndarray) -> None:
        h, w = patch.shape[:2]
        y1 = min(y0 + h, self.height)
        x1 = min(x0 + w, self.width)
        self._mm[y0:y1, x0:x1] = patch[:y1 - y0, :x1 - x0].astype(self.dtype)

    def close(self) -> None:
        self._mm.flush()
        del self._mm

        t0 = time.perf_counter()
        img = pyvips.Image.rawload(
            str(self._raw_path),
            self.width, self.height, 1,
            format=self._vips_fmt,
        )
        kwargs: dict = dict(
            tile=True,
            tile_width=self._tile_size,
            tile_height=self._tile_size,
            pyramid=self._pyramidal,
            compression="jpeg" if self._pyramidal else "none",
            bigtiff=True,
            subifd=self._pyramidal,
        )
        if self._pyramidal:
            kwargs["Q"] = self._jpeg_quality

        img.tiffsave(str(self.path), **kwargs)
        self._raw_path.unlink()

        size_mb = self.path.stat().st_size / (1024 ** 2)
        logger.info(
            "%s written: %.1f MB in %.1fs (pyramidal=%s)",
            self.path.name, size_mb, time.perf_counter() - t0, self._pyramidal,
        )
