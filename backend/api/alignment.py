"""Thin FastAPI wrapper over the image-alignment (thriple_image_layer / VALIS
registration) pipeline. Mounted at /api/alignment/* -- a future hybrid
(cell segmentation) pipeline gets its own /api/hybrid/* module, not this one.

Guardrail 3 (docs/UI/04-guardrails-red-lines.md): endpoints only validate
params, call the existing algorithms/ functions, and return a file path +
JSON metadata. No numpy, no image processing, no business logic here.
"""
from fastapi import APIRouter, BackgroundTasks

from backend.algorithms.thriple_image_layer.module1_preprocess import CziPreprocessor
from backend.algorithms.thriple_image_layer.module2_alignment import align_images
from backend.algorithms.thriple_image_layer.module3_roi_evaluation import evaluate_roi
from backend.algorithms.thriple_image_layer.module4_thumbnail import generate_thumbnail
from backend.algorithms.thriple_image_layer.module5_tile_generator import generate_triple_tiles
from backend.api.jobs import submit_job
from backend.schemas.alignment import AlignmentConfigIn
from backend.schemas.common import JobAccepted

router = APIRouter(prefix="/api/alignment")


@router.post("/preprocess", response_model=JobAccepted)
def run_preprocess(body: AlignmentConfigIn, background_tasks: BackgroundTasks) -> JobAccepted:
    config = body.to_registration_config()

    def _run():
        CziPreprocessor(config).run()
        return str(config.input_dir), {"modalities": [m.name for m in config.modalities]}

    return JobAccepted(job_id=submit_job(background_tasks, _run))


@router.post("/align", response_model=JobAccepted)
def run_align(body: AlignmentConfigIn, background_tasks: BackgroundTasks) -> JobAccepted:
    config = body.to_registration_config()

    def _run():
        align_images(config)  # returns a non-serializable VALIS registrar; not forwarded (guardrail 7)
        return str(config.pickle_path), {"output_dir": str(config.output_dir)}

    return JobAccepted(job_id=submit_job(background_tasks, _run))


@router.post("/roi-eval", response_model=JobAccepted)
def run_roi_eval(body: AlignmentConfigIn, background_tasks: BackgroundTasks) -> JobAccepted:
    config = body.to_registration_config()

    def _run():
        evaluate_roi(config)
        return str(config.output_dir / "Metrics.csv"), {
            "overlay": str(config.output_dir / "Merged_ROI.png")
        }

    return JobAccepted(job_id=submit_job(background_tasks, _run))


@router.post("/thumbnail", response_model=JobAccepted)
def run_thumbnail(body: AlignmentConfigIn, background_tasks: BackgroundTasks) -> JobAccepted:
    config = body.to_registration_config()

    def _run():
        generate_thumbnail(config)
        level = config.thumbnail.level
        return str(config.output_dir / f"Merged_Aligned_lv{level}.tiff"), {"level": level}

    return JobAccepted(job_id=submit_job(background_tasks, _run))


@router.post("/tiles", response_model=JobAccepted)
def run_tiles(
    body: AlignmentConfigIn, background_tasks: BackgroundTasks, level: int = 0
) -> JobAccepted:
    config = body.to_registration_config()

    def _run():
        generate_triple_tiles(config, level=level)
        tile_dir = config.output_dir / f"tiles_lv{level}-{config.tile.tile_width}"
        return str(tile_dir), {"level": level}

    return JobAccepted(job_id=submit_job(background_tasks, _run))
