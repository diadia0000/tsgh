"""M3 backward-compatible re-export — 統一入口取代四個散落的 shim。

Prefer importing from ``cell_mask.hybrid.m3_module``.
"""

from cell_mask.hybrid.m3_module.m3_cells_generator import *  # noqa: F403
from cell_mask.hybrid.m3_module.m3_dot_detection import *  # noqa: F403
from cell_mask.hybrid.m3_module.m3_dot_kernels import *  # noqa: F403
from cell_mask.hybrid.m3_module.m3_elastic_matching import *  # noqa: F403
