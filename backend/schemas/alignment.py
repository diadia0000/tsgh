"""Pydantic request models for the image-alignment (thriple_image_layer /
VALIS registration) pipeline API.

These mirror the dataclasses in backend/algorithms/thriple_image_layer/config.py.
Only this module is allowed to know about Pydantic; algorithms/ stays framework-free.
Generic job types (JobAccepted/JobStatus) live in backend/schemas/common.py since
they're shared with any future pipeline (e.g. hybrid), not alignment-specific.
"""
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from backend.algorithms.thriple_image_layer.config import (
    ModalityConfig,
    PreprocessConfig,
    RegistrationConfig,
    ROIConfig,
    ThumbnailConfig,
    TileConfig,
    ValisConfig,
)


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


class TileConfigIn(BaseModel):
    tile_width: Optional[int] = None
    tile_height: Optional[int] = None
    workers: Optional[int] = None
    compression: Optional[str] = None


class AlignmentConfigIn(BaseModel):
    """Request body accepted by every /api/alignment/* endpoint.

    All fields are optional; anything omitted falls back to RegistrationConfig's
    own dataclass defaults (see config.py) so a caller only needs to send the
    fields that differ from the CLI default.
    """

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
    tile: Optional[TileConfigIn] = None

    def to_registration_config(self) -> RegistrationConfig:
        """Translate the request body into the algorithms-layer RegistrationConfig.

        Only fields explicitly provided override RegistrationConfig's dataclass
        defaults; this is the one place JSON meets the algorithm's own types.
        """
        kwargs = {}
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
        if self.tile is not None:
            kwargs["tile"] = TileConfig(**self.tile.model_dump(exclude_none=True))
        return RegistrationConfig(**kwargs)
