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
import os
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException
from tuspyserver import create_tus_router

from backend.algorithms.thriple_image_layer.module1_preprocess import CziPreprocessor
from backend.algorithms.thriple_image_layer.module2_alignment import align_images
from backend.algorithms.thriple_image_layer.module3_roi_evaluation import evaluate_roi
from backend.algorithms.thriple_image_layer.module4_thumbnail import generate_thumbnail
from backend.api.jobs import active_job, submit_job
from backend.io import pyramid
from backend.io.pyramid import SLIDES_DIR
from backend.schemas.alignment import (
    STORAGE_DIR,
    AlignmentConfigIn,
    RunSummary,
    is_run_id,
    run_base,
)
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


# Dev shortcut: CZIs that already sit on the server, so a developer doesn't wait
# out a multi-GB upload over Tailscale just to exercise the pipeline. Set
# TSGH_DEV_CZI_DIR to point at them; defaults to `<repo>/dev_data/czi/40X`.
_DEV_CZI_DIR = Path(
    os.environ.get(
        "TSGH_DEV_CZI_DIR", Path(__file__).resolve().parents[2] / "dev_data" / "czi" / "40X"
    )
)


@router.post("/dev-run")
def create_dev_run(body: AlignmentConfigIn) -> dict:
    """Fill the named run's czi_input/ with symlinks to the server-local CZIs,
    i.e. the same state a finished upload leaves behind. Everything downstream
    (paths, output isolation, slide publishing) is unchanged."""
    if body.run_id is None:
        raise HTTPException(422, "run_id required")
    missing = [n for n in _MODALITY_DESTINATIONS.values() if not (_DEV_CZI_DIR / n).is_file()]
    if missing:
        raise HTTPException(404, f"{_DEV_CZI_DIR} is missing {missing}")
    czi_input = run_base(body.run_id) / "czi_input"
    czi_input.mkdir(parents=True, exist_ok=True)
    for name in _MODALITY_DESTINATIONS.values():
        link = czi_input / name
        if link.exists() or link.is_symlink():  # a reused name must not collide
            link.unlink()
        link.symlink_to(_DEV_CZI_DIR / name)
    return {"run_id": body.run_id}


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


def _merged_tiff(config) -> Path:
    return config.output_dir / f"Merged_Aligned_lv{config.thumbnail.level}.tiff"


# Every step leaves exactly one artifact behind, so the run directory IS the
# progress record -- no job database needed. The in-memory registry in jobs.py
# dies with the process; these files don't, which is what lets a reloaded
# browser (or a restarted backend) pick up mid-pipeline instead of starting over.
# Declaration order is pipeline order.
_STEP_ARTIFACT = {
    "preprocess": lambda c: all((c.input_dir / m.filename).is_file() for m in c.modalities),
    "align": lambda c: c.pickle_path.is_file(),
    "roi-eval": lambda c: (c.output_dir / "Metrics.csv").is_file(),
    "thumbnail": lambda c: _merged_tiff(c).is_file(),
}


@router.get("/runs", response_model=List[RunSummary])
def list_runs() -> List[RunSummary]:
    """Every run folder and how far it got, newest first. The frontend adopts the
    first one on load and offers the rest as a picker, so a user comes back to
    their work instead of re-running the pipeline from scratch."""
    by_mtime = []
    for d in STORAGE_DIR.iterdir():
        if not d.is_dir() or not is_run_id(d.name):  # skips _tus_incoming
            continue
        # Steps write into output/, so that is what tracks progress; the run
        # dir's own mtime stops moving once czi_input/ and output/ exist.
        out = d / "output"
        by_mtime.append((max(d.stat().st_mtime, out.stat().st_mtime if out.is_dir() else 0), d.name))

    summaries = []
    for _, run_id in sorted(by_mtime, reverse=True):
        config = AlignmentConfigIn(run_id=run_id).to_registration_config()
        summaries.append(
            RunSummary(
                run_id=run_id,
                done=[step for step, has in _STEP_ARTIFACT.items() if has(config)],
                job_id=active_job(run_id),
            )
        )
    return summaries


@router.post("/publish")
def publish_run(body: AlignmentConfigIn) -> dict:
    """Re-point the viewer at an already-finished run's TIFFs. SLIDES_DIR holds
    one global `aligned_result`, overwritten by whichever run ran thumbnail last,
    so a resumed run has to reclaim it -- by symlink only, no recomputation."""
    config = body.to_registration_config()
    result_tiff = _merged_tiff(config)
    if not result_tiff.is_file():
        raise HTTPException(404, "this run has no thumbnail result to publish")
    return {**_publish_aligned_result(result_tiff), **_publish_aligned_layers(config)}


@router.post("/preprocess", response_model=JobAccepted)
def run_preprocess(body: AlignmentConfigIn, background_tasks: BackgroundTasks) -> JobAccepted:
    config = body.to_registration_config()

    def _run():
        CziPreprocessor(config).run()
        return str(config.input_dir), {"modalities": [m.name for m in config.modalities]}

    return JobAccepted(job_id=submit_job(background_tasks, _run, key=body.run_id))


@router.post("/align", response_model=JobAccepted)
def run_align(body: AlignmentConfigIn, background_tasks: BackgroundTasks) -> JobAccepted:
    config = body.to_registration_config()

    def _run():
        align_images(config)  # returns a non-serializable VALIS registrar; not forwarded (guardrail 7)
        return str(config.pickle_path), {"output_dir": str(config.output_dir)}

    return JobAccepted(job_id=submit_job(background_tasks, _run, key=body.run_id))


@router.post("/roi-eval", response_model=JobAccepted)
def run_roi_eval(body: AlignmentConfigIn, background_tasks: BackgroundTasks) -> JobAccepted:
    config = body.to_registration_config()

    def _run():
        evaluate_roi(config)
        return str(config.output_dir / "Metrics.csv"), {
            "overlay": str(config.output_dir / "Merged_ROI.png")
        }

    return JobAccepted(job_id=submit_job(background_tasks, _run, key=body.run_id))


@router.post("/thumbnail", response_model=JobAccepted)
def run_thumbnail(body: AlignmentConfigIn, background_tasks: BackgroundTasks) -> JobAccepted:
    config = body.to_registration_config()

    def _run():
        generate_thumbnail(config)
        result_tiff = _merged_tiff(config)
        return str(result_tiff), {
            "level": config.thumbnail.level,
            **_publish_aligned_result(result_tiff),
            **_publish_aligned_layers(config),
        }

    return JobAccepted(job_id=submit_job(background_tasks, _run, key=body.run_id))
