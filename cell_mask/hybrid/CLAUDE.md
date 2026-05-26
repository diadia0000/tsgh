# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Pipeline

```bash
# Single tile pair
python hybrid_pipeline.py --ihc tile_x1024_y2048.tiff --dish tile_x1024_y2048.tiff

# Batch mode (scans tile/ directories defined in config)
python hybrid_pipeline.py --batch

# Batch mode using test_picture/ directories
python hybrid_pipeline.py --batch --test

# Override output directory
python hybrid_pipeline.py --batch --output /path/to/output
```

## Configuration

`config.py` is **gitignored**. Create it from the example before running:

```bash
cp config_example.py config.py
# Edit model paths, tile directories, and Cellpose parameters
```

Critical fields in `Config`:
- `unet_model_path`: path to `best_model_unet.pth`
- `cellpose_model_path`: path to `cellpose_ihc_dish_best` (retrained on IHC-DISH blend)
- `ihc_tile_dir` / `dish_tile_dir`: input tile directories (default: `tile/her2`, `tile/dish`)
- `output_dir`: root output directory (default: `output/`)
- `slide_id` / `model_version`: written into CSV for traceability

`compute_config_hash(config)` produces a short hash included in every CSV for reproducibility.

## Architecture

Pipeline runs **M1 → M2 → M3 → M4** sequentially per tile pair. Models are initialized once before the batch loop and reused.

### M1 — `m1_overlay.py`
Generates the IHC-DISH fused image fed to Cellpose:
1. `generate_ihc_core_mask()`: runs UNet++ on the IHC tile → binary core mask (morphological close applied)
2. `apply_mask_to_ihc_image()`: masks IHC image, background filled to `background_fill_value`
3. `overlay_ihc_mask_on_dish()`: same mask applied to DISH image
4. `fuse_masked_ihc_with_dish()`: 50/50 alpha blend of masked DISH + masked IHC

Tile filename pairing uses coordinate parsing (`tile_x{int}_y{int}` pattern). `find_paired_tiles()` builds coord→path maps for both modalities and returns sorted matched pairs.

### M2 — `m2_segmentation.py`
`CellposeSegmenter` wraps `CellposeModel` (pretrained on IHC-DISH blend). `segment_masked_dish()` runs inference on the M1 fused image and optionally removes border-touching cells via `skimage.segmentation.clear_border`. Cell IDs are re-labeled sequentially after border removal.

### M3 — `m3_cells_generator.py`
`build_all_positive_results()` iterates all cell IDs in the instance mask and computes centroid via `scipy.ndimage.center_of_mass`. All cells are currently marked HER2-positive (`is_her2_positive=True`, `hematoxylin_ratio=1.0`). Returns a list of `CellAnalysisResult` dataclasses.

### M4 — `m4_export.py`
`export_cell_dot_annotations()` is the unified export entry point:
- `{tile_id}_report.csv`: per-cell table (cell_id, centroid_x/y, is_her2_positive, hematoxylin_ratio)
- `{tile_id}_overlay.png`: full-tile visualization with green cell boundaries and +/- labels
- `cells/cell_{id}.png`: 256×256 white-canvas crops centered on each cell (from `dish_mask_overlay`)

`export_overlay_visualization()` is also called separately to produce `{tile_id}_ihc_dish_overlay.png` (IHC-DISH blend as background) and optionally `{tile_id}_merge_overlay.png` (merge tile as background).

### `unet_inference.py` — UNet++ Engine
`UNetPPInference` loads a `segmentation_models_pytorch.UnetPlusPlus` model (EfficientNet-B4 encoder by default, configurable). Inference uses `torch.inference_mode()` and `cudnn.benchmark`. For images larger than `image_size` (minimum 1024×1024), sliding window inference is used with non-overlapping patches.

`postprocess_membrane_mask()` applies morphological close and removes connected components below `min_area=550` pixels.

## Output Structure Per Tile

Each tile writes to `output/{tile_id}/`:

```
{tile_id}_ihc_core_mask.png          # UNet++ binary mask
{tile_id}_masked_ihc.png             # IHC masked to ROI
{tile_id}_dish_mask_overlay.png      # DISH masked to ROI
{tile_id}_ihc_dish_overlay_raw.png   # 50/50 blend (M2 input)
{tile_id}_m2_input_overlay.png       # same as above (explicit M2 input copy)
{tile_id}_m2_cell_instance_binary.png
{tile_id}_report.csv
{tile_id}_overlay.png                # dish_mask_overlay + cell annotations
{tile_id}_ihc_dish_overlay.png       # IHC-DISH blend + cell annotations
{tile_id}_merge_overlay.png          # merge tile + cell annotations (if merge_dir provided)
cells/cell_{id}.png                  # per-cell crops
```

## Key Invariants

- All images passed between modules are RGB `uint8` `(H, W, 3)`. The `_read_rgb()` helper in `hybrid_pipeline.py` enforces this.
- Core mask is `uint8 {0, 1}` shape `(H, W)`.
- Instance mask is `int32` shape `(H, W)`, background=0, cells=1..N.
- Empty core mask (all zeros) causes early exit: only an empty CSV is written, no Cellpose inference runs.
- OpenCV functions receive BGR images; the pipeline stores images as RGB and converts at draw time (`cv2.cvtColor(..., cv2.COLOR_RGB2BGR)`).
