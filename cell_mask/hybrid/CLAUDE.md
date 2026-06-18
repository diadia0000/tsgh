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

## Configuration
`config.py` is **gitignored**; run `cp config_example.py config.py` first, then edit paths/params.
Key fields: `unet_model_path`, `cellpose_model_path` (M2), `cellpose_dish_model_path` (M3b),
tile dirs, `output_dir`, `slide_id`/`model_version`. `compute_config_hash()` is written into every CSV for traceability.

## Architecture
- **M1 `m1_overlay.py`** — UNet++ produces the IHC core mask → applied to IHC & DISH →
  50/50 alpha blend (`overlay_alpha`) becomes the M2 input; an empty core mask short-circuits to an empty CSV.
- **M2 `m2_segmentation.py`** — `CellposeSegmenter` segments the fused image → cell instance mask;
  `clear_border_cells` drops seam-touching cells, then labels are renumbered.
- **M3 cell analysis**
  - `m3_cells_generator.py` — `build_all_positive_results()` computes a centroid per cell.
  - `m3_elastic_matching.py` — cell-centric one-to-one matching: each IHC cell's search radius is its area scaled by `dish_elastic_expand_factor` (1.5×, `reach=sqrt(factor*area/π)`); every (cell, nucleus) candidate pair is matched nearest-first with locking, so each cell claims at most one nucleus.
  - `m3_dot_detection.py` — per-cell red-dot (CEP17) / black-dot (HER2) detection on a local LAB patch;
    cell with no claimed nucleus but a lost candidate → drop-out exclusion (X); no candidate at all → counted as 0/0. No multi-nucleus exclusion under one-to-one matching.
  - `m3_dot_kernels.py` — pixel-level red/black dot detection, ring statistics, and merge core.
  - Amplification: HER2/CEP17 ≥ `dot_amplification_ratio` or HER2 ≥ `dot_her2_count_threshold`.
- **M4 `m4_export.py`** — `export_cell_dot_annotations()` writes the report CSV, overlay PNG, and
  `cells/cell_{id}.png` per-cell crops (`cell_crop_size`); `export_overlay_visualization()` produces the clinician-review overlay.
- **`unet_inference.py`** — `UNetPPInference` (EfficientNet-B4); large images use sliding-window inference.

## Invariants
- Images passed between modules are RGB `uint8 (H,W,3)` (enforced by `_read_rgb()`); converted to BGR only when drawing with OpenCV.
- core mask is `uint8{0,1} (H,W)`; instance mask is `int32 (H,W)`: background 0, cells 1..N.

## Misc
- Design docs in `docs/`: `dish_dot_detection_spec.md`, `sdd-elastic-dish-matching.md`.
