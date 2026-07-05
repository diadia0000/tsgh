# cell_mask/hybrid — IHC-DISH Overlay & Analysis Pipeline

Per-tile chain of **M0→M1→M2→M3→M4**: precut the ROI/WSI pair into overlapping
1024px tile files on disk, analyze each tile independently, fuse IHC(Her2)
with DISH, segment cells, detect HER2/CEP17 signal dots, judge amplification,
then merge per-tile cell tables globally and lazily stitch the annotated
overlay tiles back into one slide-level QuPath-openable pyramid TIFF.
Models are initialized once before the batch loop and reused.

## Running (entry: `hybrid_pipeline.py`)

```bash
python hybrid_pipeline.py --ihc roi_ihc.tiff --dish roi_dish.tiff   # single ROI/WSI pair: precut then batch
python hybrid_pipeline.py --batch [--test] [--output DIR]           # batch: dirs are already-precut tile folders
```

Tile pairing is by filename coordinate parsing `tile_x{int}_y{int}`.
Key imports in `hybrid_pipeline.py`: all local-style — `m0_reader`, `m0_stitch`, `m1_overlay`, `m2_segmentation`, `m3_cell_detection`, `m4_export`.
`--ihc`/`--dish` accepts a single tile, an arbitrary ROI, or a WSI of any size — `_run_single_tile_cli()` calls
`precut_paired_tiles()` (M0 reader) to cut it into `default_tile_size` tile files under `output_dir/_precut_scratch/`
first, then runs the same `run_batch()` path as `--batch`, so memory stays bounded on both read and analysis sides.
`backend/api/hybrid.py`'s `/api/hybrid/tile` and `/api/hybrid/batch` endpoints mirror this same split.

## Configuration

`config.py` is **gitignored**; run `cp config_example.py config.py` first, then edit paths/params.
Key fields: `unet_model_path`, `cellpose_model_path` (M2), `cellpose_dish_model_path` (M3b),
tile dirs, `output_dir`, `slide_id`/`model_version`. `compute_config_hash()` is written into every CSV for traceability.

## Architecture

- **M0 `m0_reader.py` + `m0_stitch.py`** — precut-to-folder + per-chunk analysis, so a ROI/WSI far larger than
  one tile never needs a full in-memory canvas (the old full-slide `StitchAccumulator` peaked at ≈400GB and was
  deleted entirely; there is no in-pipeline chunked *read* step anymore either — cutting happens once, upfront).
  - `m0_reader.py` — `precut_paired_tiles()` opens IHC/DISH with `pyvips.Image.new_from_file(access="random")`
    and writes aligned `tile_x{abs_x}_y{abs_y}.tiff` files to disk on the same grid as
    `m2_segmentation._overlap_window_coords` (`tile_size`/`window_overlap_px` from config); short edges are
    white-filled like `module5_tile_generator._crop_tile`. Runs a thread pool (`workers=`) since it's pure I/O.
  - `m0_stitch.py` — `compute_tile_geometry()` derives a `TileGeometry` (cut lines + which tiles touch a real
    slide edge) purely from the set of `(abs_x, abs_y)` positions parsed from tile filenames — no read-back of
    the original WSI's true dimensions needed — and raises `ValueError` if the grid has gaps/dupes (fail-fast:
    the analysis stage has no other way to catch a partially-completed precut job). `hybrid_pipeline.process_precut_tile()`
    runs M1→M2→M3 per tile via `_process_one_chunk()` with `remove_border=False` (no interior-seam clearing);
    `clear_slide_edge_cells()` (gated by `geometry.edge_flags()`) only clears cells touching a *real* slide edge
    before M3. `filter_and_absolutize()` then dedups cross-tile duplicates by **centroid core-ownership**: each
    tile's core region is the strip inside `overlap/2` of its neighbors, and a cell counts only in the tile
    whose core contains its centroid — no IoMin pass needed across tiles. It absolutizes each kept cell's
    centroid by `+(abs_x, abs_y)` but deliberately does **not** renumber `cell_id` (still tile-local); the batch
    driver (`run_batch()`) flattens all tiles' kept cells, sorts by `(abs_y, abs_x, cell_id)`, and renumbers
    1..N exactly once — the only place global cell IDs are assigned. Single-tile input degenerates to final
    ID == local ID, matching the pre-refactor single-image path (GPU inference itself is non-deterministic, so
    cross-run comparisons are judged against a noise floor, not exact equality).
  - Per-tile artifacts land in per-array-type folders under `output_dir/` (`core_mask/`, `masked_ihc/`,
    `dish_nucleus_mask/`, `dish_mask_overlay/`, `instance_mask/`, `cell_crops/tile_x{x}_y{y}/`) and stay at
    1024px — only `report.csv` + `summary.txt` (global, via `export_tile_csv`/`export_summary_statistics` on
    the renumbered cell list) and the **annotated** slide-level overlay are assembled globally. The overlay is
    built per-tile as `overlay_annotated/tile_x{x}_y{y}.tiff` (core-cropped via `core_crop_bounds()`, drawn with
    `render_overlay_image()` — cell boundaries + HER2/CEP17 dot markers) then `_stitch_overlay_slide()` joins
    them into `overlay_slide.tiff`, a pyramidal (`tile=True, pyramid=True`) TIFF QuPath can open directly.
    `pyvips.Image.arrayjoin()` cannot be used here — it assumes a uniform per-cell grid size and silently
    mis-pads when row/column tile sizes differ (as they do at slide edges); the fix is a manual row-then-column
    `Image.join(..., expand=True)`. `run_batch()` is intentionally sequential (not parallelized) across tiles —
    the 3 GPU models are loaded once in the main process and share one CUDA context, so cross-tile process
    parallelism is unsafe (fork-under-CUDA); it also runs **fail-fast**, raising immediately if any tile errors,
    since all tiles are pieces of one slide and a silent skip would produce a slide with an undocumented hole.
- **M1 `m1_overlay.py`** — UNet++ produces the IHC core mask → applied to IHC & DISH →
  50/50 alpha blend (`overlay_alpha`) becomes the M2 input; an empty core mask short-circuits to an empty CSV.
- **M2 `m2_segmentation.py`** — `CellposeSegmenter` segments the fused image → cell instance mask.
  Border clearing now happens at the M0 stitch layer (`clear_slide_edge_cells`), not here — within a chunk,
  `segment_masked_dish` is called with `remove_border=False` so interior seam edges are left for M0 to dedup.
- **M3 `m3_module/`** — package; `hybrid_pipeline.py` imports from `cell_mask.hybrid.m3_module`.
  - `m3_cells_generator.py` — `CellAnalysisResult`, `build_all_positive_results()` (centroid per cell).
  - `m3_elastic_matching.py` — `elastic_dish_nucleus_matching()`; reach = `sqrt(factor×area/π)`;
    nearest-first with locking so each cell claims at most one nucleus.
  - `m3_dot_detection.py` — `CellDotResult`, `detect_all_dots()`, `merge_dot_results_to_cell_analysis()`;
    HER2 (black) / CEP17 (red) on local LAB patch; drop-out / boundary-contamination / `cep17 < score_cep17_min_count` (except 0/0) → excluded (X); 0/0 no-signal cases still counted normally.
  - `m3_dot_kernels.py` — `DetectedDot`; pixel-level dot detection, ring statistics, merge core.
  - Score(r,b)=HER2/CEP17: `cep17 < score_cep17_min_count` (default 2) and not 0/0 → excluded with X (0/0 still counted normally); otherwise `score = ratio if ratio ≥ dot_amplification_ratio else 0`; `is_amplified = score > 0`.
  - `m3_cell_detection.py` — backward-compat shim; re-exports all symbols from `m3_module/` in one file.
- **M4 `m4_export.py` + `m4_module/`** — facade re-export over three sub-modules:
  - `m4_module/csv.py` — `DotStatsSummary`, `export_tile_csv`, `export_summary_statistics`, `write_summary_csv`.
  - `m4_module/overlay.py` — `render_overlay_image`, `export_overlay_visualization`, `export_dot_only_visualization`, `stamp_grid_on_overlays`.
  - `m4_module/cell_crops.py` — `export_per_cell_images`, `export_cell_dot_annotations` (unified entry point: CSV + overlay + per-cell crops).
  - `m4_export.py` is the stable public API; callers import only from it.
- **`unet_inference.py`** — `UNetPPInference` (EfficientNet-B4); large images use sliding-window inference.
- **`heatmap_visualizer.py`** — Standalone validation tool (**does not import pipeline**); reads per-tile `*_report.csv` →
  full-slide coordinates → three heatmaps (DISH orange / Her2 green / Geneity pink). `--output-dir DIR --n N [--background overlay]`.

## Invariants

- Images passed between modules are RGB `uint8 (H,W,3)` (enforced by `_read_rgb()`); converted to BGR only when drawing with OpenCV.
- core mask is `uint8{0,1} (H,W)`; instance mask is `int32 (H,W)`: background 0, cells 1..N.