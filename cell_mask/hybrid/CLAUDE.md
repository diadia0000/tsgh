# cell_mask/hybrid — IHC-DISH Overlay & Analysis Pipeline

Per-tile chain of **M1→M2→M3→M4**: fuse IHC(Her2) with DISH, segment cells,
detect HER2/CEP17 signal dots, and judge amplification. Models are initialized
once before the batch loop and reused.

## Running (entry: `hybrid_pipeline.py`)
```bash
python hybrid_pipeline.py --ihc tile_x1024_y2048.tiff --dish tile_x1024_y2048.tiff  # single tile
python hybrid_pipeline.py --batch [--test] [--output DIR]                            # batch scan dirs
```
Tile pairing is by filename coordinate parsing `tile_x{int}_y{int}`.
Key imports in `hybrid_pipeline.py`: all local-style — `m1_overlay`, `m2_segmentation`, `m3_cell_detection`, `m4_export`.

## Configuration
`config.py` is **gitignored**; run `cp config_example.py config.py` first, then edit paths/params.
Key fields: `unet_model_path`, `cellpose_model_path` (M2), `cellpose_dish_model_path` (M3b),
tile dirs, `output_dir`, `slide_id`/`model_version`. `compute_config_hash()` is written into every CSV for traceability.

## Architecture
- **M1 `m1_overlay.py`** — UNet++ produces the IHC core mask → applied to IHC & DISH →
  50/50 alpha blend (`overlay_alpha`) becomes the M2 input; an empty core mask short-circuits to an empty CSV.
- **M2 `m2_segmentation.py`** — `CellposeSegmenter` segments the fused image → cell instance mask;
  `clear_border_cells` drops seam-touching cells, then labels are renumbered.
- **M3 `m3_module/`** — package; `hybrid_pipeline.py` imports from `cell_mask.hybrid.m3_module`.
  - `m3_cells_generator.py` — `CellAnalysisResult`, `build_all_positive_results()` (centroid per cell).
  - `m3_elastic_matching.py` — `elastic_dish_nucleus_matching()`; reach = `sqrt(factor×area/π)`;
    nearest-first with locking so each cell claims at most one nucleus.
  - `m3_dot_detection.py` — `CellDotResult`, `detect_all_dots()`, `merge_dot_results_to_cell_analysis()`;
    HER2 (black) / CEP17 (red) on local LAB patch; drop-out / boundary-contamination → excluded (X); no candidate → 0/0.
  - `m3_dot_kernels.py` — `DetectedDot`; pixel-level dot detection, ring statistics, merge core.
  - Amplification: HER2/CEP17 ≥ `dot_amplification_ratio` **or** HER2 ≥ `dot_her2_count_threshold`.
  - `m3_cell_detection.py` — backward-compat shim; re-exports all symbols from `m3_module/` in one file.
- **M4 `m4_export.py` + `m4_module/`** — facade re-export over three sub-modules:
  - `m4_module/csv.py` — `DotStatsSummary`, `export_tile_csv`, `export_summary_statistics`, `write_summary_csv`.
  - `m4_module/overlay.py` — `render_overlay_image`, `export_overlay_visualization`, `export_dot_only_visualization`, `stamp_grid_on_overlays`.
  - `m4_module/cell_crops.py` — `export_per_cell_images`, `export_cell_dot_annotations` (unified entry point: CSV + overlay + per-cell crops).
  - `m4_export.py` is the stable public API; callers import only from it.
- **`unet_inference.py`** — `UNetPPInference` (EfficientNet-B4); large images use sliding-window inference.

## Invariants
- Images passed between modules are RGB `uint8 (H,W,3)` (enforced by `_read_rgb()`); converted to BGR only when drawing with OpenCV.
- core mask is `uint8{0,1} (H,W)`; instance mask is `int32 (H,W)`: background 0, cells 1..N.

## Misc
- Design docs in `docs/`: `dish_dot_detection_spec.md`, `sdd-elastic-dish-matching.md`.
