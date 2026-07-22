"""Pydantic request models for the image-alignment (thriple_image_layer /
VALIS registration) pipeline API.

These mirror the dataclasses in backend/algorithms/thriple_image_layer/config.py.
Only this module is allowed to know about Pydantic; algorithms/ stays framework-free.
Generic job types (JobAccepted/JobStatus) live in backend/schemas/common.py since
they're shared with any future pipeline (e.g. hybrid), not alignment-specific.
"""
import re
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, field_validator

from backend.algorithms.thriple_image_layer.config import (
    ModalityConfig,
    PreprocessConfig,
    RegistrationConfig,
    ROIConfig,
    ThumbnailConfig,
    ValisConfig,
)

# Root for per-upload isolated run directories. Each upload gets its own
# {STORAGE_DIR}/{run_id}/ subtree so concurrent users never share czi_input /
# output paths (fixes the shared-storage race).
STORAGE_DIR = Path("/home/sec312/project/storge_tsgh")

# A run_id is a user-chosen folder name. Restricting it to letters/digits/-/_
# is what stops a client-supplied value escaping STORAGE_DIR (no "/", no ".."),
# and requiring a leading alphanumeric keeps internal directories like
# _tus_incoming out of the run listing. A UUID still matches, so runs created
# before naming existed keep working.
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def is_run_id(name: str) -> bool:
    return _RUN_ID_RE.fullmatch(name) is not None


def run_base(run_id: str) -> Path:
    """Filesystem base for a run. Validates the name so a client-supplied value
    can never escape STORAGE_DIR via path traversal."""
    if not is_run_id(run_id):
        raise ValueError(f"invalid run_id: {run_id!r}")
    return STORAGE_DIR / run_id


class RunSummary(BaseModel):
    """What a run directory already contains, so the UI can resume it instead of
    re-running the whole pipeline."""

    run_id: str
    done: List[str]  # step names whose artifacts are already on disk, in pipeline order
    job_id: Optional[str] = None  # a step still executing server-side, if any


class ModalityConfigIn(BaseModel):
    name: str
    filename: str
    czi_filename: Optional[str] = None
    scale_factor: Optional[float] = None


class PreprocessConfigIn(BaseModel):
    strip_height: Optional[int] = None
    num_processes: Optional[int] = None


class ValisConfigIn(BaseModel):
    max_processed_image_dim_px: Optional[int] = None
    max_non_rigid_registration_dim_px: Optional[int] = None
    align_to_reference: Optional[bool] = None
    reference_img_f: Optional[str] = None


class ROIConfigIn(BaseModel):
    roi_size: Optional[List[int]] = None


class ThumbnailConfigIn(BaseModel):
    level: Optional[int] = None
    use_non_rigid: Optional[bool] = None
    laplacian_levels: Optional[int] = None


class AlignmentConfigIn(BaseModel):
    """Request body accepted by every /api/alignment/* endpoint.

    All fields are optional; anything omitted falls back to RegistrationConfig's
    own dataclass defaults (see config.py) so a caller only needs to send the
    fields that differ from the CLI default.
    """

    run_id: Optional[str] = None
    project_name: Optional[str] = None
    czi_input_dir: Optional[str] = None
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None
    reference_modality: Optional[str] = None
    modalities: Optional[List[ModalityConfigIn]] = None
    preprocess: Optional[PreprocessConfigIn] = None
    valis: Optional[ValisConfigIn] = None
    roi: Optional[ROIConfigIn] = None
    thumbnail: Optional[ThumbnailConfigIn] = None

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not is_run_id(v):  # 422 -> blocks path traversal
            raise ValueError("run_id must be 1-64 letters, digits, '-' or '_', starting alphanumeric")
        return v

    def to_registration_config(self) -> RegistrationConfig:
        """Translate the request body into the algorithms-layer RegistrationConfig.

        Only fields explicitly provided override RegistrationConfig's dataclass
        defaults; this is the one place JSON meets the algorithm's own types.
        """
        kwargs = {}
        if self.run_id is not None:
            # Run-isolated paths derived from run_id: czi_input holds the upload,
            # output holds every step's intermediate + final artifacts.
            base = run_base(self.run_id)
            kwargs["czi_input_dir"] = base / "czi_input"
            kwargs["input_dir"] = base / "output"
            kwargs["output_dir"] = base / "output"
        if self.project_name is not None:
            kwargs["project_name"] = self.project_name
        if self.czi_input_dir is not None:
            kwargs["czi_input_dir"] = Path(self.czi_input_dir)
        if self.input_dir is not None:
            kwargs["input_dir"] = Path(self.input_dir)
        if self.output_dir is not None:
            kwargs["output_dir"] = Path(self.output_dir)
        if self.reference_modality is not None:
            kwargs["reference_modality"] = self.reference_modality
        if self.modalities is not None:
            kwargs["modalities"] = [
                ModalityConfig(**m.model_dump(exclude_none=True)) for m in self.modalities
            ]
        if self.preprocess is not None:
            kwargs["preprocess"] = PreprocessConfig(**self.preprocess.model_dump(exclude_none=True))
        if self.valis is not None:
            kwargs["valis"] = ValisConfig(**self.valis.model_dump(exclude_none=True))
        if self.roi is not None:
            roi_kwargs = self.roi.model_dump(exclude_none=True)
            if "roi_size" in roi_kwargs:
                roi_kwargs["roi_size"] = tuple(roi_kwargs["roi_size"])
            kwargs["roi"] = ROIConfig(**roi_kwargs)
        if self.thumbnail is not None:
            kwargs["thumbnail"] = ThumbnailConfig(**self.thumbnail.model_dump(exclude_none=True))
        return RegistrationConfig(**kwargs)
