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
from backend.algorithms.hybrid.hybrid_pipeline import run_batch
from backend.algorithms.hybrid.m0_reader import precut_paired_tiles
from backend.api.jobs import submit_job
from backend.schemas.common import JobAccepted
from backend.schemas.hybrid import HybridTileIn

router = APIRouter(prefix="/api/hybrid")


@router.post("/tile", response_model=JobAccepted)
def run_hybrid_tile(body: HybridTileIn, background_tasks: BackgroundTasks) -> JobAccepted:
    ihc_path = Path(body.ihc_path)
    dish_path = Path(body.dish_path)
    output_dir = Path(body.output_dir) if body.output_dir else config.output_dir
    merge_dir = Path(body.merge_dir) if body.merge_dir else None

    def _run():
        scratch = output_dir / "_precut_scratch"
        ihc_out = scratch / "ihc"
        dish_out = scratch / "dish"
        precut_paired_tiles(
            ihc_path, dish_path, ihc_out, dish_out,
            tile_size=config.default_tile_size,
            overlap=config.window_overlap_px,
        )
        stats = run_batch(ihc_out, dish_out, output_dir, merge_dir=merge_dir)
        return str(output_dir), stats

    return JobAccepted(job_id=submit_job(background_tasks, _run))
