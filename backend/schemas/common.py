"""Request/response models shared across all pipeline APIs (alignment, and
future hybrid). Kept separate from any one pipeline's schema file so adding a
second pipeline doesn't require reaching into another pipeline's module.
"""
from typing import Optional

from pydantic import BaseModel


class JobAccepted(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    result_path: Optional[str] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None
