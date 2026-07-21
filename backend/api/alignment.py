"""Thin FastAPI wrapper over the image-alignment (thriple_image_layer / VALIS
registration) pipeline. Mounted at /api/alignment/* -- a future hybrid
(cell segmentation) pipeline gets its own /api/hybrid/* module, not this one.

Guardrail 3 (docs/UI/04-guardrails-red-lines.md): endpoints only validate
params, call the existing algorithms/ functions, and return a file path +
JSON metadata. No numpy, no image processing, no business logic here.

Resumable CZI upload is delegated to the `tuspyserver` tus implementation
(mounted as `tus_router`); this module only maps a finished upload into the
run's czi_input/ directory the pipeline reads from.
"""
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks
from tuspyserver import create_tus_router

from backend.algorithms.thriple_image_layer.module1_preprocess import CziPreprocessor
from backend.algorithms.thriple_image_layer.module2_alignment import align_images
from backend.algorithms.thriple_image_layer.module3_roi_evaluation import evaluate_roi
from backend.algorithms.thriple_image_layer.module4_thumbnail import generate_thumbnail
from backend.api.jobs import submit_job
from backend.io import pyramid
from backend.io.pyramid import SLIDES_DIR
from backend.schemas.alignment import STORAGE_DIR, AlignmentConfigIn, run_base
from backend.schemas.common import JobAccepted

router = APIRouter(prefix="/api/alignment")

_MODALITY_DESTINATIONS = {
    "her2": "HER2_40X.czi",
    "dish": "DISH_40X.czi",
    "he": "HE_40X.czi",
}

# Resumable uploads land here first; _place_uploaded_czi moves each finished
# file into {run_id}/czi_input/ once tus reports it complete.
_TUS_INCOMING = STORAGE_DIR / "_tus_incoming"
_TUS_INCOMING.mkdir(parents=True, exist_ok=True)


def _place_uploaded_czi(file_path: str, metadata: dict) -> None:
    """tus on_upload_complete hook, called synchronously as the final PATCH
    returns. Moves the assembled upload to
    {STORAGE_DIR}/{run_id}/czi_input/{HER2_40X,DISH_40X,HE_40X}.czi so the
    pipeline finds it. run_base() rejects a non-UUID run_id (path traversal);
    an unknown modality raises KeyError -- both are our own frontend's bugs, not
    recoverable states, so they surface as 500 rather than a silent fallback."""
    dest = run_base(metadata["run_id"]) / "czi_input" / _MODALITY_DESTINATIONS[metadata["modality"]]
    dest.parent.mkdir(parents=True, exist_ok=True)
    Path(file_path).replace(dest)  # atomic rename within STORAGE_DIR


# ponytail: dev works because the Vite proxy forwards the browser Host
# (changeOrigin:false) so tus's absolute Location stays reachable. Behind a
# real prod proxy, forward X-Forwarded-Proto / X-Forwarded-Host instead.
tus_router = create_tus_router(
    prefix="api/alignment/tus",
    files_dir=str(_TUS_INCOMING),
    on_upload_complete=_place_uploaded_czi,
)


def _publish_aligned_result(result_tiff: Path) -> dict:
    """Expose the generated TIFF to the existing slide tile service."""
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    slide_link = SLIDES_DIR / "aligned_result.tiff"
    if slide_link.exists() or slide_link.is_symlink():
        slide_link.unlink()
    try:
        slide_link.symlink_to(result_tiff)
    except OSError:
        shutil.copyfile(result_tiff, slide_link)
    pyramid.invalidate("aligned_result")
    return {"slide_id": "aligned_result"}


def _publish_aligned_layers(config) -> dict:
    """Expose the per-modality warped TIFFs module4 leaves in temp/ as individual
    slides, so the overlay viewer can blend HER2/DISH with live per-layer alpha.
    They're already OpenSlide-readable pyramids (module4_thumbnail.py:48-52), so a
    symlink into SLIDES_DIR is enough. HE is never warped (doctors don't need it
    yet), so no HE layer is published — the frontend shows an empty HE slider."""
    level = config.thumbnail.level
    layers = {
        "aligned_her2": config.temp_dir / f"her2_warped_lv{level}.tiff",
        "aligned_dish": config.temp_dir / f"dish_warped_lv{level}.tiff",
    }
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    for slide_id, src in layers.items():
        link = SLIDES_DIR / f"{slide_id}.tiff"
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            link.symlink_to(src)
        except OSError:
            shutil.copyfile(src, link)
        pyramid.invalidate(slide_id)
    return {"layer_slide_ids": list(layers)}


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
        result_tiff = config.output_dir / f"Merged_Aligned_lv{level}.tiff"
        return str(result_tiff), {
            "level": level,
            **_publish_aligned_result(result_tiff),
            **_publish_aligned_layers(config),
        }

    return JobAccepted(job_id=submit_job(background_tasks, _run))


@router.post("/full", response_model=JobAccepted)
def run_full(body: AlignmentConfigIn, background_tasks: BackgroundTasks) -> JobAccepted:
    config = body.to_registration_config()

    def _run():
        # Single-active-job system: wipe any stale output from a previous run of
        # THIS run_id before starting, so a re-run never mixes in old artifacts.
        output_dir = config.output_dir
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        CziPreprocessor(config).run()
        align_images(config)  # returns a non-serializable VALIS registrar; not forwarded (guardrail 7)
        evaluate_roi(config)
        generate_thumbnail(config)

        result_tiff = config.output_dir / f"Merged_Aligned_lv{config.thumbnail.level}.tiff"

        return str(result_tiff), {
            **_publish_aligned_result(result_tiff),
            **_publish_aligned_layers(config),
        }

    return JobAccepted(job_id=submit_job(background_tasks, _run))
