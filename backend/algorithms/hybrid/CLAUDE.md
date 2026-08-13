# hybrid — IHC-DISH Overlay & Analysis Pipeline

Per-tile **M0→M1→M2→M3→M4**: precut the ROI/WSI pair into overlapping 1024px tiles on
disk, analyze each independently (fuse IHC/Her2 + DISH → segment cells → detect
HER2/CEP17 dots → judge amplification), merge cell tables globally, lazily stitch the
overlay tiles into one QuPath-openable pyramid TIFF. Models init once per batch.

## Running (entry: `hybrid_pipeline.py`)

```bash
python hybrid_pipeline.py --ihc a.tiff --dish b.tiff                        # tile, ROI or full WSI
python hybrid_pipeline.py --test [--output DIR]                             # bundled test_picture pair
python hybrid_pipeline.py --ihc a.tiff --dish b.tiff --workers 4 --resume   # unattended full slide
```

`_run_single_tile_cli()` builds a `PrecutStream` that cuts into
`output_dir/_precut_scratch/{ihc,dish}` *while* `run_batch()` analyzes (grid comes from
the file header); that scratch dir is never auto-deleted. `--test` and
`backend/api/hybrid.py`'s `/api/hybrid/tile` take the same stream+batch path; the API
also accepts an ROI (`roi_x/y/w/h`, four or none — `backend/schemas/hybrid.py`) and
passes `workers=1` (a single tile shouldn't pay N model inits). Tiles pair by filename
`tile_x{int}_y{int}`. `--workers N` → `run_batch(workers=N)` (default **4**);
`--resume` → `checkpoint=True`.

## Configuration

`config.py` is **gitignored**: `cp config_example.py config.py`, then edit paths. Key
fields: `unet_model_path`, `cellpose_model_path` (M2), `cellpose_dish_model_path` (M3b),
tile dirs, `output_dir`. `compute_config_hash()` guards spawn workers (must match parent)
and resume; not written to the CSV.

## Architecture

**M0 `m0_module/`** — slide layer. Inside the package files import each other directly;
everything outside imports the `m0_slide.py` facade only (otherwise: cycles).

- `m0_reader.py` — `PrecutStream`: pyvips `access="random"`, writes aligned
  `tile_x{abs_x}_y{abs_y}.tiff` on the `tile_size`/`window_overlap_px` grid, short edges
  white-filled, yielding each pair as it lands.
- `m0_stitch.py` — `compute_tile_geometry()` derives cut lines + slide-edge flags from
  parsed tile positions alone (no WSI read-back), raising on gaps/dupes.
  `clear_slide_edge_cells()` clears cells only on true slide edges.
  `filter_and_absolutize()` dedups by **centroid core-ownership** (a cell counts in the
  tile whose core — the strip inside overlap/2 — holds its centroid; no IoMin pass),
  absolutizes centroids by `+(abs_x, abs_y)`, does *not* renumber `cell_id`.
  `_stitch_overlay_slide()` joins `_stitch_scratch/`.
- `m0_tile_runner.py` — `_process_precut_tile_gpu()` (M1→M2) / `_process_precut_tile_cpu()`
  (M3 + writes) run two-stage: each tile's CPU tail overlaps the next tile's GPU forward,
  reads prefetched one tile ahead. Also model init and `_frozen_gc_generation()`.
- `m0_multiprocess.py` — `_run_tiles_multiprocess()`, **spawn**-based.
- `m0_checkpoint.py` — `_checkpoint_{load,init,save}`, `_skip_completed`.

**M1 `m1_overlay.py`** — UNet++ core mask → applied to IHC & DISH → 50/50 alpha blend
(`overlay_alpha`) is M2's input; empty core mask short-circuits to an empty CSV.

**M2 `m2_segmentation.py`** — `CellposeSegmenter` → instance mask, `remove_border=False`;
interior seams are M0's job.

**M3 `m3_module/`** (`m3_cell_detection.py` is a re-export shim) —
`build_all_positive_results()` (centroid per cell); `elastic_dish_nucleus_matching()`
reach = `sqrt(factor×area/π)`, nearest-first with locking so a cell claims ≤1 nucleus;
`detect_all_dots()` finds HER2 (black) / CEP17 (red) on a local LAB patch. Score =
HER2/CEP17: drop-out, boundary contamination, or `cep17 < score_cep17_min_count`
(default 2) → excluded (X), except 0/0 which counts normally; otherwise
`score = ratio if ratio ≥ dot_amplification_ratio else 0`, `is_amplified = score > 0`.

**M4 `m4_export.py`** — facade over `m4_module/{csv,overlay}.py`, the only import callers
use. Pure library: `render_overlay_image` to arrays, `export_tile_csv` /
`export_summary_statistics`; owns no slide-level image file.

**`unet_inference.py`** — `UNetPPInference` (EfficientNet-B4), sliding-window.
**`hybrid_data_types.py`** — `DetectedDot`, `CellAnalysisResult`, `CellDotResult`.

## Don't break these

- A run leaves exactly 3 files in `output_dir/`: `report.csv`, `summary.txt`,
  `overlay_slide.tiff`. **No per-tile intermediates** — masks die with the chunk.
  `merge_overlay/` only when the caller passes `merge_dir`.
- `_stitch_scratch/` is the one disk round-trip: a streaming buffer read lazily one band
  at a time. Nothing larger than one band of one level is ever resident — in-memory
  stitching resurrects the ≈400GB full-canvas OOM. Do not "simplify" into a whole-image
  load. Its `rmtree` is deliberately **not** in a `finally`, so a failed stitch can be
  re-run without recomputing the batch (`backend/tests/test_stitch_scratch_cleanup.py`).
- **Two stitch backends**, `config.stitch_backend`. Default **`"tifffile"`**:
  band-streamed read on a background thread, CPU box pyramid, per-tile LZW + Predictor 2
  (Phase D 1.91x, peak RSS 45.6 → 17.0 GB vs pyvips). `"pyvips"` is kept as fallback and
  measurement control — do not delete. Same picture, different bytes; tests assert every
  pyramid level is pixel-identical. `stitch_backend` is in `config._HASH_EXCLUDE` on
  purpose — swapping encoders must not invalidate a multi-hour resume checkpoint.
- No `pyvips.arrayjoin()` — it assumes a uniform grid and silently mis-pads at slide
  edges; use manual row-then-column `Image.join(..., expand=True)` (both backends do).
- Cross-tile parallelism must be **spawn**, never fork (forked child inherits a broken
  CUDA context). ~2.8GB VRAM + ~3.1s init per worker; `workers=1→4` measures **2.05x**
  end-to-end, peak RSS 13.66 GB on the registered 35,700-tile slide. (Older 27,565-tile
  numbers were measured on an unregistered canvas — see `docs/hybrid-pipeline/44-*.md`;
  scaling and stitch details in `docs/hybrid-pipeline/{39,41,46}-*.md`.)
- Global `cell_id` 1..N is assigned in exactly one place: `_finish_batch()`, in the
  parent, sorting `(abs_y, abs_x, cell_id)`. Workers return only `(abs_x, abs_y, owned)`.
- **fail-fast at any worker count** — one bad tile aborts the batch (multiprocess kills
  siblings first); a silent skip ships a slide with an undocumented hole.
- Resume pickles each tile's `owned` to `_resume/*.pkl` (tmp+`os.replace`, one file per
  tile, no lock); config-hash mismatch drops the checkpoint loudly. Full coverage
  short-circuits to `_finish_batch()` without loading any model — also how you re-run
  just the stitch. Does **not** relax fail-fast (`tests/test_run_batch_resume.py`).

## Invariants

Images between modules are RGB `uint8 (H,W,3)` (enforced by `_read_rgb()`), BGR only when
drawing with OpenCV. core mask `uint8{0,1} (H,W)`; instance mask `int32 (H,W)`:
background 0, cells 1..N.
