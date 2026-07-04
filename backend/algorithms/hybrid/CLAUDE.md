# cell_mask/hybrid — IHC-DISH Overlay & Analysis Pipeline

Per-tile chain of **M0→M1→M2→M3→M4**: read the pair in bounded-memory chunks,
fuse IHC(Her2) with DISH, segment cells, detect HER2/CEP17 signal dots, judge
amplification, then stitch chunk results back into one slide-level output.
Models are initialized once before the batch loop and reused.

## Running (entry: `hybrid_pipeline.py`)

```bash
python hybrid_pipeline.py --ihc tile_x1024_y2048.tiff --dish tile_x1024_y2048.tiff  # single tile
python hybrid_pipeline.py --batch [--test] [--output DIR]                            # batch scan dirs
```

Tile pairing is by filename coordinate parsing `tile_x{int}_y{int}`.
Key imports in `hybrid_pipeline.py`: all local-style — `m0_reader`, `m0_stitch`, `m1_overlay`, `m2_segmentation`, `m3_cell_detection`, `m4_export`.
Input images may be a single tile, an arbitrary ROI, or a WSI — `process_single_tile()` reads/processes/stitches
in `default_tile_size` chunks (M0) regardless of the input's actual size, so memory stays bounded.

## Configuration

`config.py` is **gitignored**; run `cp config_example.py config.py` first, then edit paths/params.
Key fields: `unet_model_path`, `cellpose_model_path` (M2), `cellpose_dish_model_path` (M3b),
tile dirs, `output_dir`, `slide_id`/`model_version`. `compute_config_hash()` is written into every CSV for traceability.

## Architecture

- **M0 `m0_reader.py` + `m0_stitch.py`** — chunked read/stitch wrapper around M1–M3 so a ROI/WSI far
  larger than one tile doesn't need a full in-memory load (a 20k² ROI peaked at ≈31GB before this).
  - `m0_reader.py` — `iter_paired_chunks()` opens IHC/DISH with `pyvips.Image.new_from_file(access="random")`
    and yields aligned `Chunk(ihc, dish, abs_x, abs_y)` on the same grid as `m2_segmentation._overlap_window_coords`
    (`tile_size`/`window_overlap_px` from config); short edges are white-filled like `module5_tile_generator._crop_tile`.
    Input ≤ `tile_size` degenerates to a single chunk = whole-file read (regression baseline: pyvips decode is
    bit-identical to `skimage.io.imread` for JPEG-TIFF, so this is the seam against the pre-M0 code path).
  - `m0_stitch.py` — `_process_one_chunk()` in `hybrid_pipeline.py` runs M1→M2→M3 per chunk with
    `remove_border=False` (no interior-seam clearing); `clear_slide_edge_cells()` only clears cells touching a
    *real* slide edge (`abs_x/abs_y` at 0 or the full extent) before M3. `StitchAccumulator` then dedups
    cross-chunk duplicates by **centroid core-ownership**: each chunk's core region is the strip inside
    `overlap/2` of its neighbors, and a cell counts only in the chunk whose core contains its centroid — no
    IoMin pass needed across chunks. It also does the slide-level relabeling (1..N cells, 1..M DISH nuclei,
    `assigned_dish_ids` rewritten to match) and absolutizes every centroid/dot coordinate by `+(abs_x, abs_y)`,
    then paints M1 artifacts (`core_mask`/`masked_ihc`/`dish_mask_overlay`/`overlay_image`) into the full-size
    canvas one core-region at a time. Each `ChunkResult` is freed (GC-able) right after `acc.add()`; the batch
    loop also runs `torch.cuda.empty_cache()` + `gc.collect()` between tiles to prevent monotonic growth.
    Single-chunk input has an unseamed full core region → global ID == local ID, bit-identical to the pre-M0
    single-image path (GPU inference itself is non-deterministic, so cross-run comparisons are judged against a
    noise floor, not exact equality).
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