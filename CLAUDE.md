# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TSGH is a medical imaging analysis platform for Whole Slide Image (WSI) analysis, focused on:
- **Triple-modality image registration** (HER2, DISH, H&E staining) using VALIS
- **Cell segmentation and quantification** using UNet++ and Cellpose
- **IHC-DISH overlay analysis** for cancer biomarker (HER2/DISH) detection

## Development Environment

### Docker (primary)
```bash
docker compose build
docker compose up -d
docker compose exec tsgh bash
```
Base image: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` with Python 3.11 and Java 17.

### Dependencies
Install from `requirements.txt` (200+ packages). Key: PyTorch 2.10+cu130, Cellpose 4.0.8, segmentation_models_pytorch, SimpleITK, pyvips, OpenSlide, aicspylibczi.

## Running the Pipelines

### VALIS Registration Pipeline
```bash
cd thriple_image_layer
python run_full_pipeline.py              # Run all modules
python run_full_pipeline.py --preprocess # Run M1 (CZI→TIFF) first
python run_full_pipeline.py --tiles      # Run M5 (tile generation)
```

Modules run in sequence: M1 (preprocess) → M2 (VALIS alignment) → M3 (ROI quality metrics) → M4 (thumbnail) → M5 (tile generation).

### IHC-DISH Hybrid Pipeline
```bash
cd cell_mask/hybrid
python hybrid_pipeline.py --ihc FILE --dish FILE  # Single pair
python hybrid_pipeline.py --batch                 # Process tile directories
python hybrid_pipeline.py --test                  # Use test_picture dirs
```

Modules: M1 (UNet++ overlay) → M2 (Cellpose segmentation) → M3 (per-cell analysis) → M4 (CSV/visualization export).

### UNet++ Training
```bash
cd cell_mask/unet_mask
python train_unetpp.py
```

### Cellpose Training
```bash
cd cell_mask/dish_mask
python train_cellpose.py
```

### Utility Scripts
```bash
python scripts/cuda_test.py         # Verify CUDA/GPU setup
python scripts/check_tiff_size.py   # Inspect TIFF metadata
```

## Architecture

### Two Independent Sub-Pipelines

**1. `thriple_image_layer/`** — Registration pipeline
- `run_full_pipeline.py`: Orchestrator, accepts `--preprocess` and `--tiles` flags
- `module1_preprocess.py`: CZI → BigTIFF using `aicspylibczi` + `pyvips`, multiprocessing (bypasses GIL)
- `module2_alignment.py`: VALIS alignment — LightGlue/Disk feature matching (10,000 features), non-rigid SimpleElastix B-spline warping; reference modality is H&E
- `module3_roi_evaluation.py`: Quality metrics (NCC, NMI, SSIM) at center + 4 corners
- `module4_thumbnail.py`: Laplacian pyramid fusion (4 levels) → merged TIFF
- `module5_tile_generator.py`: OpenSlide streaming → 1024×1024 tiles with ThreadPoolExecutor (16 workers)

**2. `cell_mask/hybrid/`** — Cell analysis pipeline
- `hybrid_pipeline.py`: Orchestrator for M1–M4
- `m1_overlay.py`: UNet++ membrane segmentation → DISH mask overlay (50/50 alpha blend)
- `m2_segmentation.py`: Cellpose instance segmentation on masked DISH image
- `m3_cells_generator.py`: Per-cell mask extraction, HED color deconvolution, statistics
- `m4_export.py`: CSV output, dot annotation visualization, 256×256 cell crops
- `unet_inference.py`: Shared UNet++ inference engine (EfficientNet-B6 encoder, 1024×1024 input)

**Supporting sub-projects:**
- `cell_mask/unet_mask/`: UNet++ training with LAB-space pseudo-label generation
- `cell_mask/dish_mask/`: Cellpose training on DISH modality

### Configuration System

**Critical**: `config.py` files are **gitignored**. Each environment needs its own `config.py` created from `config_example.py`.

Configs use Python dataclasses with field factories. Key config files:
- `thriple_image_layer/config.py`: `RegistrationConfig` (master), contains `ValisConfig`, `ROIConfig`, `ThumbnailConfig`, `TileConfig`, and per-modality `ModalityConfig`
- `cell_mask/hybrid/config.py`: Model paths, Cellpose params, UNet++ params, I/O dirs, batch settings
- `cell_mask/unet_mask/config.py`: Training hyperparameters, augmentation settings

Default paths in configs assume `/home/sec312/project/tsgh/` — adjust for other environments.

### Data Flow
```
CZI files (40X) → [M1] BigTIFF → [M2] VALIS registration → Transform params
                                                           → [M3] Quality metrics
                                                           → [M4] Merged thumbnail
                                                           → [M5] Aligned tiles (HER2/DISH/Merge)
                                                                    ↓
                                              [Hybrid M1] UNet++ membrane mask
                                              [Hybrid M2] Cellpose cell instances
                                              [Hybrid M3] Per-cell HED analysis
                                              [Hybrid M4] CSV + visualizations
```

### Key Technical Patterns
- **Multiprocessing in M1**: `multiprocessing.Pool` for CZI conversion (GIL bypass for pyvips)
- **Streaming in M5**: OpenSlide + ThreadPoolExecutor avoids loading full WSI into memory
- **Model lazy loading**: Models initialized on first use via cached properties
- **Batch vs. single mode**: Hybrid pipeline scans tile directories or processes a single pair

### Gitignored Directories
`output/`, `tile/`, `models/`, `picture/`, `config.py` — these must be present locally but are not in the repo. Models must be downloaded separately.
