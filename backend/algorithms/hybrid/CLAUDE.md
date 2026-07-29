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
python hybrid_pipeline.py --test [--output DIR]                     # smoke-test: bundled test_picture ROI pair, same precut+batch path
python hybrid_pipeline.py --ihc a.tiff --dish b.tiff --workers 4 --resume   # unattended full-slide run
```

`--workers N` (default 4 — round 8's full-WSI validation cleared it for production, round 12
re-confirmed on current code) is `run_batch(workers=N)`; `--resume` is `run_batch(checkpoint=True)`.
Both are documented under M0 below.

Tile pairing is by filename coordinate parsing `tile_x{int}_y{int}`.
Key imports in `hybrid_pipeline.py`: all local-style — `m0_reader`, `m0_stitch`, `m1_overlay`, `m2_segmentation`, `m3_cell_detection`, `m4_export`.
`--ihc`/`--dish` accepts a single tile, an arbitrary ROI, or a WSI of any size — `_run_single_tile_cli()` calls
`precut_paired_tiles()` (M0 reader) to cut it into `default_tile_size` tile files under `output_dir/_precut_scratch/`
first, then runs the `run_batch()` analysis path, so memory stays bounded on both read and analysis sides.
`--test` routes a bundled `test_picture` smoke-test ROI pair through this exact same precut+`run_batch()` path (not a
separate pretiled-dir path). `backend/api/hybrid.py`'s `/api/hybrid/tile` endpoint mirrors this precut+batch flow.

## Configuration

`config.py` is **gitignored**; run `cp config_example.py config.py` first, then edit paths/params.
Key fields: `unet_model_path`, `cellpose_model_path` (M2), `cellpose_dish_model_path` (M3b),
tile dirs, `output_dir`. `compute_config_hash()` is no longer written into the CSV — it now guards two
things instead: spawn workers verify their hash matches the parent's, and the resume checkpoint refuses
to reuse tiles computed under a different config.

## Architecture

- **M0 `m0_reader.py` + `m0_stitch.py`** — precut-to-folder + per-chunk analysis, so a ROI/WSI far larger than
  one tile never needs a full in-memory canvas (the old full-slide `StitchAccumulator` peaked at ≈400GB and was
  deleted entirely; there is no in-pipeline chunked *read* step anymore either — cutting happens once, upfront).
  - `m0_reader.py` — `precut_paired_tiles()` opens IHC/DISH with `pyvips.Image.new_from_file(access="random")`
    and writes aligned `tile_x{abs_x}_y{abs_y}.tiff` files to disk on the same grid as
    `m2_segmentation._overlap_window_coords` (`tile_size`/`window_overlap_px` from config); short edges are
    white-filled. Runs a thread pool (`workers=`) since it's pure I/O.
  - `m0_stitch.py` — `compute_tile_geometry()` derives a `TileGeometry` (cut lines + which tiles touch a real
    slide edge) purely from the set of `(abs_x, abs_y)` positions parsed from tile filenames — no read-back of
    the original WSI's true dimensions needed — and raises `ValueError` if the grid has gaps/dupes (fail-fast:
    the analysis stage has no other way to catch a partially-completed precut job). Per tile,
    `hybrid_pipeline._process_precut_tile_gpu()` runs M1→M2 (via `_process_one_chunk_gpu()`) and
    `_process_precut_tile_cpu()` finishes M3 (via `_finish_chunk_cpu()`), with `remove_border=False`
    (no interior-seam clearing);
    `clear_slide_edge_cells()` (gated by `geometry.edge_flags()`) only clears cells touching a *real* slide edge
    before M3. `filter_and_absolutize()` then dedups cross-tile duplicates by **centroid core-ownership**: each
    tile's core region is the strip inside `overlap/2` of its neighbors, and a cell counts only in the tile
    whose core contains its centroid — no IoMin pass needed across tiles. It absolutizes each kept cell's
    centroid by `+(abs_x, abs_y)` but deliberately does **not** renumber `cell_id` (still tile-local); the batch
    driver (`run_batch()`) flattens all tiles' kept cells, sorts by `(abs_y, abs_x, cell_id)`, and renumbers
    1..N exactly once — the only place global cell IDs are assigned. Single-tile input degenerates to final
    ID == local ID, matching the pre-refactor single-image path (GPU inference itself is non-deterministic, so
    cross-run comparisons are judged against a noise floor, not exact equality).
  - A run leaves exactly three files in `output_dir/`: `report.csv` + `summary.txt` (global, via
    `export_tile_csv`/`export_summary_statistics` on the renumbered cell list) and `overlay_slide.tiff`.
    **Per-tile intermediates are never written** — core mask, `dish_mask_overlay`, instance mask and DISH
    nucleus mask all stay in memory inside `_process_precut_tile_cpu()` and die with the chunk. The masked-IHC
    array and the per-cell crop export were deleted outright, not merely kept in memory. The one exception is the annotated overlay, which must round-trip through disk: it is written
    per-tile as `_stitch_scratch/tile_x{x}_y{y}.tiff` (core-cropped via `core_crop_bounds()`, drawn with
    `render_overlay_image()` on `dish_mask_overlay` — DISH nucleus contours + cell boundaries + drift arrows
    + labels + HER2/CEP17 dot markers), then `_stitch_overlay_slide()` joins
    them into `overlay_slide.tiff`, a pyramidal (`tile=True, pyramid=True`) TIFF QuPath can open directly,
    and `shutil.rmtree`s the scratch dir. That dir is a **streaming buffer, not an artifact**: pyvips reads it
    lazily (`access="sequential"`) so the full slide never materializes in RAM — building it from in-memory
    tile buffers instead would resurrect the ≈400GB full-canvas failure mode. The rmtree is deliberately *not*
    in a `finally`, so a failed stitch keeps the tiles and can be re-run without recomputing the batch
    (`backend/tests/test_stitch_scratch_cleanup.py` covers the round-trip + cleanup).
    `merge_overlay/` is still written per-tile when the caller passes `merge_dir` — an explicitly requested
    output, not an intermediate.
    `pyvips.Image.arrayjoin()` cannot be used here — it assumes a uniform per-cell grid size and silently
    mis-pads when row/column tile sizes differ (as they do at slide edges); the fix is a manual row-then-column
    `Image.join(..., expand=True)`. `run_batch()` defaults to `workers=1`, which is sequential across tiles —
    the 3 GPU models are loaded once in the main process and share one CUDA context, and **`fork`-based**
    cross-tile parallelism would be unsafe (fork-under-CUDA: a forked child inherits a broken context).
    `run_batch(workers=N)` does cross-tile multiprocessing safely by using **`spawn`** instead, so each worker
    re-imports and re-initializes its own models/context (VRAM ~2.8 GB and ~3.1 s init per worker); measured
    **3.09x at N=3** — see `docs/hybrid-pipeline/21-cross-tile-multiprocessing-implementation.md`. Workers only
    ever return `(abs_x, abs_y, owned)`; the global cell renumbering still happens exactly once, in the parent.
    **`workers=4` is the default** — round 8's full-WSI validation cleared it for production and round 12
    re-confirmed it on current code (2.216x → 1.745x after later optimizations shrank the denominator faster
    than the numerator; see `docs/hybrid-pipeline/39-round-12-multiprocess-scaling-ceiling-implementation.md`).
    `backend/api/hybrid.py`'s single-tile endpoint explicitly overrides this to `workers=1`, since a one-tile
    request shouldn't pay N workers' model-init cost for parallelism it can't use. `run_batch()` also runs
    **fail-fast** at any worker count, raising
    immediately if any tile errors — and under multiprocessing the parent terminates every sibling worker
    before re-raising — since all tiles are pieces of one slide and a silent skip would produce a slide with
    an undocumented hole.
  - **Partial resume (`run_batch(checkpoint=True)`, CLI `--resume`)** — fail-fast is right about not shipping
    a holed slide, but it says nothing about the 27,000 tiles already computed when tile 25,000 OOMs, which
    at 27,565 tiles is hours of GPU work. With the checkpoint on, each tile's `owned` list — the *only*
    per-tile product not reconstructible from disk — is pickled to `output_dir/_resume/tile_x{x}_y{y}.pkl`
    the moment it lands in the parent (single-process and multiprocess share one write path via
    `_run_tiles_multiprocess(on_tile=...)`, so the parent is still the only writer). One file per tile rather
    than an append log means no lock is needed under multiprocessing; each write is tmp+`os.replace`, so a
    kill mid-write leaves no half file. On restart `_checkpoint_load()` takes back only the tiles that belong
    to *this* grid and whose `config_hash.txt` matches, and `_skip_completed()` filters them out of the
    analysis stream — precut still cuts them (it is idempotent and supplies the grid), but the three GPU
    forwards and the CPU tail are skipped, which is where the cost is. A config-hash mismatch discards the
    whole checkpoint loudly instead of silently blending results from two configs into one CSV.
    This **does not relax fail-fast** — a failing tile still aborts the batch; it only makes the retry cheap.
    If the checkpoint already covers every tile, `run_batch()` short-circuits straight to `_finish_batch()`
    without initializing any model (that is also how you re-run *just* the stitch — it works because a failed
    stitch leaves `_stitch_scratch/` in place). Default is **off**: the API's single-tile request would pay
    the I/O for nothing and leave scratch in the output dir. `tests/test_run_batch_resume.py` covers the
    store, the skip filter, and the staleness guards.
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
- **M4 `m4_export.py` + `m4_module/`** — facade re-export over two sub-modules. M4 is a **pure library**:
  it renders to arrays and writes the two global tables, and owns no slide-level image file.
  - `m4_module/csv.py` — `export_tile_csv`, `export_summary_statistics` (+ internal `DotStatsSummary`,
    `write_summary_csv`).
  - `m4_module/overlay.py` — `render_overlay_image` (returns RGB, writes nothing), `draw_tile_seam_edges`.
  - `m4_export.py` is the stable public API; callers import only from it.
- **`unet_inference.py`** — `UNetPPInference` (EfficientNet-B4); large images use sliding-window inference.
- **`heatmap_visualizer.py`** — Standalone validation tool (**does not import pipeline**); reads per-tile `*_report.csv` →
  full-slide coordinates → three heatmaps (DISH orange / Her2 green / Geneity pink). `--output-dir DIR --n N [--background overlay]`.

## Invariants

- Images passed between modules are RGB `uint8 (H,W,3)` (enforced by `_read_rgb()`); converted to BGR only when drawing with OpenCV.
- core mask is `uint8{0,1} (H,W)`; instance mask is `int32 (H,W)`: background 0, cells 1..N.