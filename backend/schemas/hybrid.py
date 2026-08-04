"""Pydantic request models for the hybrid (IHC-DISH cell segmentation) pipeline API.

Only this module is allowed to know about Pydantic; algorithms/ stays framework-free.
Generic job types (JobAccepted/JobStatus) live in backend/schemas/common.py.
"""
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class HybridResult(BaseModel):
    """What a finished hybrid run left behind, as the UI needs to show it.

    A run writes exactly three files (hybrid_pipeline.py:349-353): report.csv,
    summary.txt and overlay_slide.tiff. summary.txt is small and is rendered as
    the report card, so it is inlined here; the CSV is a download and the TIFF is
    addressed by slide_id, never by path (guardrail 2)."""

    summary: Optional[str] = None       # summary.txt verbatim; None until a run finishes
    has_report: bool = False            # report.csv exists -> download link is live
    overlay_slide_id: Optional[str] = None   # registered slide_id for the viewer


class HybridActive(BaseModel):
    """The analysis still executing server-side, if any.

    A run outlives the browser tab that started it: the job id lives only in the
    page's memory, so a reload -- or a crash mid-run -- would otherwise leave a
    multi-hour job with nothing watching it, and no way back to its result except
    running it again. The server is the one place that knows."""

    job_id: Optional[str] = None


class HybridTileIn(BaseModel):
    # slide_ids (not filesystem paths): the frontend addresses slides by id and
    # the endpoint resolves them under SLIDES_DIR (guardrail 2). output_dir /
    # merge_dir stay optional server-side overrides, not exposed in the UI.
    ihc_slide_id: str
    dish_slide_id: str
    output_dir: Optional[str] = None
    merge_dir: Optional[str] = None

    # Analysis region in slide pixels. All four together or none: a whole slide
    # is ~27,500 tiles / hours of GPU, so the UI normally sends an ROI. Bounds
    # are validated against the actual slide in PrecutStream, which is the only
    # place that knows its dimensions.
    roi_x: Optional[int] = Field(default=None, ge=0)
    roi_y: Optional[int] = Field(default=None, ge=0)
    roi_w: Optional[int] = Field(default=None, gt=0)
    roi_h: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _roi_all_or_nothing(self) -> "HybridTileIn":
        roi = (self.roi_x, self.roi_y, self.roi_w, self.roi_h)
        if any(v is not None for v in roi) and any(v is None for v in roi):
            raise ValueError("roi_x / roi_y / roi_w / roi_h 必須四個一起給，或都不給")
        return self

    def region(self) -> Optional[tuple]:
        """(x, y, w, h) for PrecutStream, or None for the whole slide."""
        if self.roi_x is None:
            return None
        return (self.roi_x, self.roi_y, self.roi_w, self.roi_h)
