# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical image analysis platform for HER2+ breast cancer diagnosis. Processes histopathology slides through three pipelines: UNet++ membrane segmentation (IHC), Cellpose instance segmentation (DISH), and a hybrid IHC-DISH single-cell analysis pipeline. Images are registered using VALIS and tiled to 1024×1024 patches.

## Environment

- Python 3.11, PyTorch 2.10.0 with CUDA 13.0 (RTX 5090 32GB)
- Docker deployment: Ubuntu 24.04 + CUDA 12.8.1 + pyvips
- Dependencies: `pip install -r requirements.txt`
- CUDA validation: `python scripts/cuda_test.py`

## Running Pipelines

```bash
# UNet++ pseudo-label generation
python cell_mask/unet_mask/lab_mask_generator.py

# UNet++ training
python cell_mask/unet_mask/train_unetpp.py

# UNet++ inference (single or batch)
python cell_mask/unet_mask/inference.py

# Grad-CAM visualization
python cell_mask/unet_mask/watch_unet.py

# Cellpose training (DISH)
python cell_mask/dish_mask/train_cellpose.py

# Hybrid IHC-DISH pipeline (single tile)
python cell_mask/hybrid/hybrid_pipeline.py --ihc tile.tiff --dish tile.tiff

# Hybrid IHC-DISH pipeline (batch)
python cell_mask/hybrid/hybrid_pipeline.py --batch --test

# Full VALIS registration pipeline
python thriple_image_layer/run_full_pipeline.py
```

## Architecture

### Three Pipelines

**UNet++ Membrane Segmentation** (`cell_mask/unet_mask/`): 3-stage process — pseudo-label generation via LAB/HED color deconvolution → UNet++ (EfficientNet-B6 encoder) training with focal loss → sliding-window inference with morphological post-processing. Prioritizes precision over recall.

**Cellpose Instance Segmentation** (`cell_mask/dish_mask/`): Fine-tunes Cellpose on DISH images with color augmentation for cell-level instance masks.

**Hybrid IHC-DISH Analysis** (`cell_mask/hybrid/`): 4-module pipeline chained M1→M2→M3→M4:
- M1 (`m1_overlay.py`): Overlays IHC HER2+ core mask onto DISH coordinate space
- M2 (`m2_segmentation.py`): Cellpose segmentation on masked DISH regions
- M3 (`m3_dot_quant.py`): Black/red dot quantification via HSV + color deconvolution
- M4 (`m4_export.py`): CSV export and visualization
- Orchestrated by `hybrid_pipeline.py` with `unet_inference.py` as UNet++ wrapper

**VALIS Registration** (`thriple_image_layer/`): CZI→BigTIFF conversion (module1) → non-rigid registration (module2) → ROI quality assessment (module3) → thumbnail generation (module4) → 1024×1024 tile extraction (module5). Module1 is heavily optimized with multiprocessing and pyvips memory mapping.

### Configuration Pattern

All components use dataclass-based `config.py` files with `config_example.py` templates. Paths use `pathlib.Path` exclusively. Environment variables can override config values.

## Code Conventions

- Type hints on all functions; Google-style docstrings with Args/Returns/Raises
- Use `logging` module — no `print()` in batch processing code
- Color space awareness: LAB for pseudo-labels, HSV for DISH dots, HED for stain deconvolution, watch for RGB↔BGR in CZI→TIFF paths
- Morphological post-processing: closing to connect membrane fragments, opening to remove noise, binary fill for cell cores

## Documentation

- [cell_mask/unet_mask/docs/architecture.md](cell_mask/unet_mask/docs/architecture.md) — System architecture overview
- [cell_mask/unet_mask/docs/her2_positive_cell_pipeline_blueprint.md](cell_mask/unet_mask/docs/her2_positive_cell_pipeline_blueprint.md) — Design blueprint (precision-over-recall philosophy)
- [cell_mask/hybrid/docs/implementation_plan.md](cell_mask/hybrid/docs/implementation_plan.md) — Hybrid pipeline SDD with data contracts
- [cell_mask/hybrid/docs/dish_dot_counting_research.md](cell_mask/hybrid/docs/dish_dot_counting_research.md) — DISH dot counting methodology
