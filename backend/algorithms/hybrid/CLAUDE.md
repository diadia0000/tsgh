# hybrid — IHC-DISH Overlay & Analysis Pipeline

Per-tile **M0→M1→M2→M3→M4**: precut the ROI/WSI pair into overlapping 1024px tiles
on disk, analyze each independently (fuse IHC/Her2 + DISH → segment cells → detect
HER2/CEP17 dots → judge amplification), merge the per-tile cell tables globally,
lazily stitch the overlay tiles into one QuPath-openable pyramid TIFF.
Models are initialized once and reused for the whole batch.

## Running (entry: `hybrid_pipeline.py`)

```bash
python hybrid_pipeline.py --ihc a.tiff --dish b.tiff        # tile, ROI or full WSI — any size
python hybrid_pipeline.py --test [--output DIR]             # bundled test_picture pair, same path
python hybrid_pipeline.py --ihc a.tiff --dish b.tiff --workers 4 --resume   # unattended full slide
```

`_run_single_tile_cli()` builds a `PrecutStream` that cuts into
`output_dir/_precut_scratch/{ihc,dish}` *while* `run_batch()` analyzes — the grid
comes from the file header, so nothing waits for the full cut; that scratch dir is
kept for inspection, never auto-deleted. `--test` and `backend/api/hybrid.py`'s
`/api/hybrid/tile` take this same stream+batch path; the API also accepts an ROI
(`roi_x/y/w/h`, four or none — `backend/schemas/hybrid.py`). Tiles pair by filename
`tile_x{int}_y{int}`.

`--workers N` → `run_batch(workers=N)`; `--resume` → `checkpoint=True`. `run_batch`'s
signature default is `workers=4`, but every real caller passes 1 — `>1` is not
cleared for production until full-WSI validation lands.

## Configuration

`config.py` is **gitignored**: `cp config_example.py config.py`, then edit paths.
Key fields: `unet_model_path`, `cellpose_model_path` (M2), `cellpose_dish_model_path`
(M3b), tile dirs, `output_dir`. `compute_config_hash()` is not written to the CSV; it
guards spawn workers (hash must match the parent) and resume (won't reuse tiles cut
under a different config).

## Architecture

**M0 `m0_module/`** — slide layer. Inside the package files import each other directly;
everything outside imports the `m0_slide.py` facade only (otherwise: cycles).

- `m0_reader.py` — `PrecutStream`: pyvips `access="random"`, writes aligned
  `tile_x{abs_x}_y{abs_y}.tiff` on the `tile_size`/`window_overlap_px` grid, short
  edges white-filled, yielding each pair as it lands.
- `m0_stitch.py` — `compute_tile_geometry()` derives cut lines + real-slide-edge flags
  from the parsed tile positions alone (no WSI read-back), raising on gaps/dupes.
  `clear_slide_edge_cells()` clears cells only on true slide edges.
  `filter_and_absolutize()` dedups across tiles by **centroid core-ownership** — a cell
  counts in the tile whose core (the strip inside overlap/2) holds its centroid, no
  IoMin pass — absolutizes centroids by `+(abs_x, abs_y)`, and does *not* renumber
  `cell_id`. `_stitch_overlay_slide()` joins `_stitch_scratch/`.
- `m0_tile_runner.py` — `_process_precut_tile_gpu()` (M1→M2) /
  `_process_precut_tile_cpu()` (M3 + writes) run two-stage: each tile's CPU tail
  overlaps the next tile's GPU forward, with reads prefetched one tile ahead. Also
  model init and `_frozen_gc_generation()`.
- `m0_multiprocess.py` — `_run_tiles_multiprocess()`, **spawn**-based.
- `m0_checkpoint.py` — `_checkpoint_{load,init,save}`, `_skip_completed`.

**M1 `m1_overlay.py`** — UNet++ core mask → applied to IHC & DISH → 50/50 alpha blend
(`overlay_alpha`) is M2's input; an empty core mask short-circuits to an empty CSV.

**M2 `m2_segmentation.py`** — `CellposeSegmenter` → cell instance mask. Called with
`remove_border=False`; interior seams are M0's job, not M2's.

**M3 `m3_module/`** (`m3_cell_detection.py` is a re-export shim) —
`build_all_positive_results()` (centroid per cell); `elastic_dish_nucleus_matching()`
reach = `sqrt(factor×area/π)`, nearest-first with locking so a cell claims ≤1 nucleus;
`detect_all_dots()` finds HER2 (black) / CEP17 (red) on a local LAB patch.
Score = HER2/CEP17: drop-out, boundary contamination, or `cep17 <
score_cep17_min_count` (default 2) → excluded (X), except 0/0 which counts normally;
otherwise `score = ratio if ratio ≥ dot_amplification_ratio else 0`,
`is_amplified = score > 0`.

**M4 `m4_export.py`** — facade over `m4_module/{csv,overlay}.py`, the only import
callers use. Pure library: renders to arrays (`render_overlay_image`) and writes the
two global tables (`export_tile_csv`, `export_summary_statistics`); owns no slide-level
image file.

**`unet_inference.py`** — `UNetPPInference` (EfficientNet-B4), sliding-window on large
images. **`hybrid_data_types.py`** — `DetectedDot`, `CellAnalysisResult`, `CellDotResult`.

## Don't break these

- A run leaves exactly 3 files in `output_dir/`: `report.csv`, `summary.txt`,
  `overlay_slide.tiff`. **No per-tile intermediates** — masks stay in memory and die
  with the chunk. `merge_overlay/` only when the caller passes `merge_dir`.
- `_stitch_scratch/` is the one disk round-trip: a streaming buffer pyvips reads lazily
  (`access="sequential"`). Stitching from in-memory tiles resurrects the ≈400GB
  full-canvas OOM that killed the old `StitchAccumulator`.
- Its `rmtree` is deliberately **not** in a `finally` — a failed stitch keeps the tiles
  so it can be re-run without recomputing the batch
  (`backend/tests/test_stitch_scratch_cleanup.py`).
- No `pyvips.arrayjoin()` — it assumes a uniform grid and silently mis-pads at slide
  edges; use the manual row-then-column `Image.join(..., expand=True)`.
- Cross-tile parallelism must be **spawn**, never fork (a forked child inherits a broken
  CUDA context). ~2.8GB VRAM + ~3.1s init per worker; measured 3.09x at N=3.
- Global `cell_id` 1..N is assigned in exactly one place: `_finish_batch()`, in the
  parent, sorting `(abs_y, abs_x, cell_id)`. Workers only return `(abs_x, abs_y, owned)`.
- **fail-fast at any worker count** — one bad tile aborts the batch (multiprocess kills
  its siblings first), because a silent skip ships a slide with an undocumented hole.
- Resume pickles each tile's `owned` to `_resume/*.pkl` (tmp+`os.replace`, one file per
  tile so no lock is needed); a config-hash mismatch drops the whole checkpoint loudly.
  Full coverage short-circuits to `_finish_batch()` without loading any model — also how
  you re-run just the stitch. It does **not** relax fail-fast; it only makes retry cheap
  (`tests/test_run_batch_resume.py`).

## Invariants

Images between modules are RGB `uint8 (H,W,3)` (enforced by `_read_rgb()`), converted to
BGR only when drawing with OpenCV. core mask is `uint8{0,1} (H,W)`; instance mask is
`int32 (H,W)`: background 0, cells 1..N.
