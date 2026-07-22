"""Guard backend/__init__.py's import-order fix.

If pyvips' libopenslide 3.4.1 wins over openslide_bin's 4.0.0, every openslide
call returns -1 *without* setting openslide_get_error, so OpenSlide() sails
past its own error check and the tile endpoints 500 on
`ValueError: Array length must be >= 0, not -1`. Silent enough to be
reintroduced by an innocent import reshuffle, so pin it here.
"""
import tempfile
import unittest
from pathlib import Path

import backend  # noqa: F401  -- must stay first, that IS the thing under test
import pyvips
from openslide import OpenSlide


class OpenSlideCoexistsWithPyvipsTest(unittest.TestCase):
    def test_openslide_reads_a_vips_written_pyramid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            slide = Path(tmp) / "tiny.tiff"
            pyvips.Image.black(2048, 2048, bands=3).tiffsave(
                str(slide), tile=True, tile_width=256, tile_height=256, pyramid=True
            )
            with OpenSlide(str(slide)) as osr:
                # -1 here == the two-libopenslide clash is back
                self.assertGreater(osr.level_count, 1)
                self.assertEqual(osr.dimensions, (2048, 2048))


if __name__ == "__main__":
    unittest.main()
