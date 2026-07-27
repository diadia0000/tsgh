"""Make `backend/algorithms/hybrid/` importable the same way the pipeline does.

`hybrid_pipeline.py` and its siblings import each other by bare module name
(`from m0_stitch import ...`) with a `try: from .x import ...` fallback, so the
package directory itself has to be on `sys.path` — importing them as
`backend.algorithms.hybrid.*` is not the supported path.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HYBRID = REPO_ROOT / "backend" / "algorithms" / "hybrid"

for _p in (str(REPO_ROOT), str(HYBRID)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
