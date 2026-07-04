"""Thin FastAPI wrapper over the hybrid (IHC-DISH cell segmentation) pipeline.
Mounted at /api/hybrid/* -- the image-alignment pipeline has its own
/api/alignment/* module, not this one.

Guardrail 3 (docs/UI/04-guardrails-red-lines.md): endpoints only validate
params, call the existing algorithms/ functions, and return a file path +
JSON metadata. No numpy, no image processing, no business logic here.
"""
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks

from backend.algorithms.hybrid.config import config
from backend.algorithms.hybrid.hybrid_pipeline import (
    compute_config_hash,
    process_single_tile,
    run_batch,
)
from backend.api.jobs import submit_job
from backend.schemas.common import JobAccepted
from backend.schemas.hybrid import HybridBatchIn, HybridTileIn

router = APIRouter(prefix="/api/hybrid")

_unet = None
_cellpose = None
_dish_cellpose = None


def _get_models():
    global _unet, _cellpose, _dish_cellpose
    if _unet is None:
        from backend.algorithms.hybrid.hybrid_pipeline import (
            _init_cellpose_segmenter,
            _init_dish_cellpose_segmenter,
            _init_unet_inferencer,
        )
        _unet = _init_unet_inferencer()
        _cellpose = _init_cellpose_segmenter()
        _dish_cellpose = _init_dish_cellpose_segmenter()
    return _unet, _cellpose, _dish_cellpose


@router.post("/batch", response_model=JobAccepted)
def run_hybrid_batch(body: HybridBatchIn, background_tasks: BackgroundTasks) -> JobAccepted:
    ihc_dir = Path(body.ihc_dir)
    dish_dir = Path(body.dish_dir)
    output_dir = Path(body.output_dir) if body.output_dir else config.output_dir
    merge_dir = Path(body.merge_dir) if body.merge_dir else None

    def _run():
        stats = run_batch(
            ihc_dir=ihc_dir, dish_dir=dish_dir, output_dir=output_dir, merge_dir=merge_dir
        )
        return str(output_dir), stats

    return JobAccepted(job_id=submit_job(background_tasks, _run))


@router.post("/tile", response_model=JobAccepted)
def run_hybrid_tile(body: HybridTileIn, background_tasks: BackgroundTasks) -> JobAccepted:
    ihc_path = Path(body.ihc_path)
    dish_path = Path(body.dish_path)
    output_dir = Path(body.output_dir) if body.output_dir else config.output_dir
    merge_dir = Path(body.merge_dir) if body.merge_dir else None

    def _run():
        unet, cellpose, dish_cellpose = _get_models()
        result = process_single_tile(
            ihc_path, dish_path, unet, cellpose, dish_cellpose, output_dir,
            compute_config_hash(config), merge_dir=merge_dir,
        )
        return str(output_dir / dish_path.stem), {
            "cell_count": len(result) if result is not None else 0
        }

    return JobAccepted(job_id=submit_job(background_tasks, _run))
