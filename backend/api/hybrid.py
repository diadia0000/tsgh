"""Thin FastAPI wrapper over the hybrid (IHC-DISH cell segmentation) pipeline.
Mounted at /api/hybrid/* -- the image-alignment pipeline has its own
/api/alignment/* module, not this one.

Guardrail 3 (docs/UI/04-guardrails-red-lines.md): endpoints only validate
params, call the existing algorithms/ functions, and return a file path +
JSON metadata. No numpy, no image processing, no business logic here.
"""
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from backend.algorithms.hybrid.config import config
from backend.algorithms.hybrid.hybrid_pipeline import run_batch
from backend.algorithms.hybrid.m0_slide import PrecutStream
from backend.api import hybrid_progress
from backend.api.jobs import active_job, current_job_id, set_progress, submit_job
from backend.io import pyramid
from backend.schemas.common import JobAccepted
from backend.schemas.hybrid import HybridActive, HybridResult, HybridTileIn

router = APIRouter(prefix="/api/hybrid")

# slide_id the annotated overlay is served under, mirroring alignment's
# `aligned_result`: one global id owned by the most recent run.
OVERLAY_SLIDE_ID = "hybrid_overlay"

# Every analysis writes the same output_dir, so two at once would interleave
# their artifacts. One key for the whole endpoint (alignment keys per run_id
# because it has one directory per run; this pipeline has only the default one).
_JOB_KEY = "hybrid"

# Republish the pipeline's per-tile log line as job progress (see
# hybrid_progress). Installed at import, so it is attached before any request.
hybrid_progress.install()


@router.get("/result", response_model=HybridResult)
def hybrid_result() -> HybridResult:
    """Read back the artifacts of the most recent run in the default output dir.

    The UI never sends output_dir (it is a server-side override), so results
    always land in config.output_dir -- that is the only place to look. Returns
    empty fields rather than 404 when nothing has run yet: "no results" is a
    normal state for the panel, not an error."""
    out = config.output_dir
    summary_file = out / "summary.txt"
    overlay = out / "overlay_slide.tiff"

    overlay_slide_id = None
    if overlay.is_file():
        # Same trick as alignment's publish: hand the tile server a path outside
        # SLIDES_DIR so the viewer can open it by id (guardrail 2).
        pyramid.register(OVERLAY_SLIDE_ID, overlay)
        overlay_slide_id = OVERLAY_SLIDE_ID

    return HybridResult(
        summary=summary_file.read_text(encoding="utf-8") if summary_file.is_file() else None,
        has_report=(out / "report.csv").is_file(),
        overlay_slide_id=overlay_slide_id,
    )


@router.get("/report.csv")
def hybrid_report_csv() -> FileResponse:
    """The per-cell table, as a download. A file body, not JSON -- it is the one
    artifact the pathologist takes away to another tool."""
    report = config.output_dir / "report.csv"
    if not report.is_file():
        raise HTTPException(status_code=404, detail="no report.csv -- run the analysis first")
    return FileResponse(report, media_type="text/csv", filename="report.csv")


@router.get("/summary.txt")
def hybrid_summary_txt() -> FileResponse:
    """The ASCO/CAP summary as a download, alongside /result which inlines the
    same file for the on-screen report card. A doctor keeping the interpretation
    for a case needs a file, not a panel they have to keep open."""
    summary = config.output_dir / "summary.txt"
    if not summary.is_file():
        raise HTTPException(status_code=404, detail="no summary.txt -- run the analysis first")
    return FileResponse(summary, media_type="text/plain; charset=utf-8", filename="summary.txt")


@router.post("/tile", response_model=JobAccepted)
def run_hybrid_tile(body: HybridTileIn, background_tasks: BackgroundTasks) -> JobAccepted:
    # Resolve slide_ids to paths here (guardrail 2); 404 up front on a bad id
    # rather than failing later inside the background job.
    try:
        ihc_path = pyramid.resolve(body.ihc_slide_id)
        dish_path = pyramid.resolve(body.dish_slide_id)
    except pyramid.SlideNotFound as e:
        raise HTTPException(status_code=404, detail=f"slide not found: {e}")
    output_dir = Path(body.output_dir) if body.output_dir else config.output_dir
    merge_dir = Path(body.merge_dir) if body.merge_dir else None

    def _run():
        scratch = output_dir / "_precut_scratch"
        ihc_out = scratch / "ihc"
        dish_out = scratch / "dish"
        # 預切與分析迴圈重疊：格線先算出，tile 檔隨切隨送進 run_batch。
        stream = PrecutStream(
            ihc_path, dish_path, ihc_out, dish_out,
            tile_size=config.default_tile_size,
            overlap=config.window_overlap_px,
            region=body.region(),   # None = whole slide
        )
        # The grid is known as soon as the stream exists (it is derived from the
        # file headers, no pixels decoded), so the panel gets a denominator
        # immediately rather than an empty bar until the first tile finishes.
        job_id = current_job_id()
        if job_id is not None:
            set_progress(job_id, hybrid_progress.PHASE_ANALYZE, 0,
                         len(stream.positions), hybrid_progress.UNIT_TILE)
        stats = run_batch(ihc_out, dish_out, output_dir, merge_dir=merge_dir,
                          tile_stream=stream, workers=1)
        return str(output_dir), stats

    # Keyed: a second submission while one is still running returns the running
    # job's id instead of starting a rival run over the same output_dir.
    return JobAccepted(job_id=submit_job(background_tasks, _run, key=_JOB_KEY))


@router.get("/active", response_model=HybridActive)
def hybrid_active() -> HybridActive:
    """The unfinished analysis job, for a page that lost track of it.

    job_id=None means nothing is running -- a normal state, not a 404. What
    finished already is read back through /result, which reads the artifacts on
    disk and so survives a backend restart too."""
    return HybridActive(job_id=active_job(_JOB_KEY))
