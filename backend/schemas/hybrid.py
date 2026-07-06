"""Pydantic request models for the hybrid (IHC-DISH cell segmentation) pipeline API.

Only this module is allowed to know about Pydantic; algorithms/ stays framework-free.
Generic job types (JobAccepted/JobStatus) live in backend/schemas/common.py.
"""
from typing import Optional

from pydantic import BaseModel


class HybridTileIn(BaseModel):
    ihc_path: str
    dish_path: str
    output_dir: Optional[str] = None
    merge_dir: Optional[str] = None
