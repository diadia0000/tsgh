# hybrid — IHC-DISH Overlay & Analysis Pipeline

Per-tile **M0→M1→M2→M3→M4**: cut the ROI/WSI pair into overlapping 1024px tiles on disk,
analyze each independently, merge cell tables globally, lazily stitch the overlay into one
QuPath-openable pyramid TIFF. Models init once per batch.

## Running (`hybrid_pipeline.py`)

```bash
python hybrid_pipeline.py --test [--output DIR]                             # bundled test pair
python hybrid_pipeline.py --ihc a.tiff --dish b.tiff --workers 4 --resume   # full slide
```

`_run_single_tile_cli()` starts a `PrecutStream` writing into `output_dir/_precut_scratch/`
(never auto-deleted) *while* `run_batch()` analyzes; `backend/api/hybrid.py`'s
`/api/hybrid/tile` shares this path (optional ROI, `workers=1`). Tiles pair by filename
`tile_x{int}_y{int}`. `--workers` default **4**.

## Configuration

`config.py` is **gitignored**: `cp config_example.py config.py`, then edit the model paths,
tile dirs, `output_dir`. `compute_config_hash()` guards spawn workers and resume.

## Architecture

- **M0 `m0_module/`** — slide layer; outside code imports the `m0_slide.py` facade only (else:
  import cycles). `m0_reader` streams aligned tiles; `m0_stitch` derives geometry from tile
  names and dedups cells by **centroid core-ownership**; `m0_tile_runner` overlaps GPU
  (M1→M2) with CPU (M3+writes); `m0_multiprocess` (spawn); `m0_checkpoint`.
- **M1 `m1_overlay.py`** — UNet++ core mask → 50/50 IHC/DISH blend; empty mask → empty CSV.
- **M2 `m2_segmentation.py`** — Cellpose instance mask, `remove_border=False` (seams are M0's).
- **M3 `m3_module/`** — cell centroids, elastic DISH-nucleus matching (≤1 per cell), HER2
  (black) / CEP17 (red) dots on LAB patches. Score = HER2/CEP17; drop-out, boundary
  contamination, or `cep17 < score_cep17_min_count` excludes the cell (X), except 0/0.
- **M4 `m4_export.py`** — facade over `m4_module/{csv,overlay}.py`; pure library, owns no file.
- `unet_inference.py` (UNet++/EfficientNet-B4), `hybrid_data_types.py` (dot/cell dataclasses).

## Don't break these

- A run leaves exactly 3 files in `output_dir/`: `report.csv`, `summary.txt`,
  `overlay_slide.tiff`. No per-tile intermediates.
- `_stitch_scratch/` streams one band at a time — in-memory stitching resurrects the ≈400GB
  OOM. Its `rmtree` is deliberately not in a `finally`, so a failed stitch can be re-run.
- Two stitch backends (`config.stitch_backend`, default `"tifffile"`); keep `"pyvips"` as
  fallback. It sits in `config._HASH_EXCLUDE` so swapping encoders won't kill a resume.
- No `pyvips.arrayjoin()` — mis-pads at slide edges; use `Image.join(..., expand=True)`.
- Cross-tile parallelism must be **spawn**, never fork (fork breaks the CUDA context).
- Global `cell_id` 1..N assigned only in `_finish_batch()`, sorted `(abs_y, abs_x, cell_id)`.
- **fail-fast at any worker count** — one bad tile aborts the batch; a silent skip ships a hole.
- Resume pickles each tile's `owned` to `_resume/*.pkl`; hash mismatch drops it loudly. Full
  coverage short-circuits to `_finish_batch()` — also how you re-run just the stitch.
- Images are RGB `uint8 (H,W,3)` between modules, BGR only for OpenCV drawing. Core mask
  `uint8{0,1} (H,W)`; instance mask `int32 (H,W)`, background 0, cells 1..N.
