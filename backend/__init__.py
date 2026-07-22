"""Load openslide before anything drags in pyvips.

Two libopenslide live on this machine: the `openslide_bin` wheel's 4.0.0
(libopenslide.so.1, what openslide-python binds to) and the system 3.4.1
(libopenslide.so.0, what libvips dlopens for its openslide loader). Whichever
lands in the process first wins symbol resolution for both. pyvips first ->
every openslide call returns -1 *without* setting openslide_get_error, so
OpenSlide() sails past its own error check and dies later on a nonsense value
(`ValueError: Array length must be >= 0, not -1` out of read_icc_profile).
Importing openslide here makes 4.0.0 win; libvips is happy with it either way.
"""
import openslide  # noqa: F401
