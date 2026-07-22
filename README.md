# TSGH - Whole Slide Image Analysis Pipeline

A deep learning pipeline for automated HER2/CEP17 amplification analysis in histopathology. Processes IHC (HER2) and DISH tile images through a sliding-window M1→M4 chain to detect cells, count signal dots, and classify amplification status.

**Language:** Traditional Chinese / English  
**Python Version:** 3.11  
**GPU:** NVIDIA CUDA (recommended)

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Hybrid Pipeline (Main)](#hybrid-pipeline-main)
  - [Triple Image Layer Pipeline](#triple-image-layer-pipeline)
  - [Standalone Scripts](#standalone-scripts)
- [Pipeline Architecture](#pipeline-architecture)
- [Configuration](#configuration)
- [Output Formats](#output-formats)
- [Hardware Requirements](#hardware-requirements)
- [Docker Support](#docker-support)
- [Development](#development)

---

## Features

**Key Capabilities:**

- **HER2/CEP17 Amplification Analysis**: Automated per-cell dot counting for FISH/DISH signal quantification
- **Cell Detection & Segmentation**: Deep learning models for automated cell identification:
  - UNet++ (EfficientNet-B4) for IHC tissue/core mask generation
  - Cellpose for cell instance segmentation on fused IHC-DISH images
  - Cellpose for DISH nucleus detection (multi-nucleus cell exclusion)
- **Sliding Window Architecture**: Processes large tiles without loading entire image into RAM; per-window overlap deduplication prevents boundary double-counting
- **Elastic IHC-DISH Matching**: Nearest-first nucleus locking to align cells across stain channels
- **Amplification Criteria**: HER2/CEP17 ≥ 2.0 **or** HER2 ≥ 6 dots per cell
- **Flexible Output**: Per-cell CSV reports, overlay visualizations, and 256×256 per-cell crops
- **GPU Acceleration**: CUDA support for high-performance inference
- **Containerized Deployment**: Docker support with GPU runtime
- **Image Preprocessing**: VALIS-based CZI→BigTIFF registration and tile generation

---

## Project Structure

```
tsgh/
├── cell_mask/                       # Cell detection & analysis modules
│   ├── unet_mask/                   # UNet++ model training & inference
│   │   ├── train_unetpp.py          # Training script
│   │   ├── inference.py             # Standalone inference
│   │   ├── lab_mask_generator.py    # LAB-based mask generation
│   │   ├── manual_mask_gui.py       # GUI for manual annotation
│   │   ├── watch_unet.py            # Watch & inference on new files
│   │   ├── config_example.py
│   │   └── docs/                    # UNet++ architecture & blueprint docs
│   │
│   ├── dish_mask/                   # DISH Cellpose model training & prediction
│   │   ├── train_cellpose.py
│   │   └── generate_predictions.py
│   │
│   └── hybrid/                      # Main IHC-DISH analysis pipeline
│       ├── hybrid_pipeline.py       # Entry point (M1→M2→M3→M4)
│       ├── m1_overlay.py            # Module 1: UNet++ core mask → IHC-DISH fusion
│       ├── m2_segmentation.py       # Module 2: Cellpose cell segmentation
│       ├── m3_cell_detection.py     # Module 3: shim re-exporting m3_module
│       ├── m3_module/               # Module 3 sub-package
│       │   ├── m3_cells_generator.py    # Cell centroid extraction
│       │   ├── m3_elastic_matching.py   # IHC-DISH nucleus matching
│       │   ├── m3_dot_detection.py      # HER2/CEP17 dot detection
│       │   └── m3_dot_kernels.py        # Pixel-level dot kernels
│       ├── m4_export.py             # Module 4: facade for CSV + overlays + crops
│       ├── m4_module/
│       │   ├── csv.py               # CSV export & summary statistics
│       │   ├── overlay.py           # Overlay visualization rendering
│       │   └── cell_crops.py        # Per-cell 256×256 crop export
│       ├── hybrid_data_types.py     # Shared dataclasses
│       ├── unet_inference.py        # UNetPPInference (sliding-window)
│       └── config_example.py        # Configuration template
│
├── backend/algorithms/thriple_image_layer/  # VALIS-based preprocessing pipeline
│   ├── run_full_pipeline.py         # Orchestration entry point
│   ├── module1_preprocess.py        # CZI → BigTIFF conversion
│   ├── module2_alignment.py         # VALIS image alignment
│   ├── module3_roi_evaluation.py    # ROI quality evaluation
│   ├── module4_thumbnail.py         # Thumbnail generation
│   ├── config_example.py
│   └── artifacts/                   # Intermediate alignment artifacts
│
├── scripts/                         # Utility scripts
│   ├── check_tiff_size.py           # Check TIFF file dimensions
│   ├── cuda_test.py                 # Test CUDA availability
│   ├── tile_generator.py            # Generate image tiles
│   └── tiff to png.py               # Convert TIFF to PNG
│
├── docs/                            # Project documentation
│   ├── elastic_matching_v3_explainer.html
│   ├── sliding-window-seam-stitch.html
│   └── next-phase-ui-architecture.md
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## Quick Start

### Using Docker (Recommended)

```bash
# Build Docker image
docker-compose build

# Run pipeline in container
docker-compose run --rm tsgh \
  python cell_mask/hybrid/hybrid_pipeline.py --batch
```

### Local Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd tsgh

# 2. Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure and run
cd cell_mask/hybrid
cp config_example.py config.py
# Edit config.py with your model and tile paths
python hybrid_pipeline.py --batch
```

---

## Installation

### System Requirements

- **OS**: Ubuntu 20.04+, macOS, or Windows (with WSL2)
- **Python**: 3.11
- **GPU**: NVIDIA GPU with CUDA Compute Capability 7.0+ (recommended)
- **RAM**: 16+ GB (32+ GB recommended for large WSI)
- **Storage**: SSD with 100+ GB free space (for intermediate results)

### Dependencies

**Key Libraries:**
- PyTorch with CUDA support
- OpenSlide 4.0+ (for WSI reading)
- Cellpose (cell segmentation)
- Segmentation-models-pytorch with timm (UNet++/EfficientNet-B4)
- VALIS (image alignment and registration)
- scikit-image, scipy, pandas, OpenCV (image processing & analysis)
- pyvips, tifffile, imagecodecs (large image I/O)

**Full dependency list:** See `pyproject.toml` and `requirements.txt`

### Installation Steps

#### Option 1: Using UV (Fast Package Manager)

```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install dependencies
uv sync
```

#### Option 2: Using pip

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install --pre torch torchvision \
  --index-url https://download.pytorch.org/whl/nightly/cu128

# Install other dependencies
pip install -r requirements.txt
```

#### Option 3: Using Docker

```bash
docker build -t tsgh:latest .
```

---

## Usage

### Hybrid Pipeline (Main)

The hybrid pipeline processes pre-tiled IHC (HER2) and DISH images through the M1→M4 analysis chain.

#### Setup

```bash
cd cell_mask/hybrid
cp config_example.py config.py
# Edit config.py — set model paths, tile directories, output_dir, slide_id
```

#### Basic Usage

```bash
# Single tile pair
python hybrid_pipeline.py \
  --ihc tile_x1024_y2048.tiff \
  --dish tile_x1024_y2048.tiff

# Batch mode (scans ihc_tile_dir and dish_tile_dir from config.py)
python hybrid_pipeline.py --batch

# Batch mode using test_picture directories
python hybrid_pipeline.py --batch --test

# Custom output directory
python hybrid_pipeline.py --batch --output /path/to/output
```

Tile pairing is by filename coordinate: `tile_x{int}_y{int}`.

#### Key Configuration Parameters

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    # Tile input directories
    ihc_tile_dir: Path = Path("tile/her2")
    dish_tile_dir: Path = Path("tile/dish")
    output_dir: Path = Path("output")

    # Model paths
    unet_model_path: Path = Path("models/best_model_unet.pth")
    cellpose_model_path: Path = Path("models/cellpose_ihc_dish_best")
    cellpose_dish_model_path: Path = Path("models/cellpose_dish_best")

    # UNet++ parameters
    unet_encoder_name: str = "efficientnet-b4"
    unet_image_size: tuple = (1024, 1024)

    # Cellpose parameters (M2: cell segmentation)
    cellpose_diameter: float = None       # auto-detect
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0

    # Sliding-window deduplication (M2 & M3b)
    window_overlap_px: int = 256          # overlap between adjacent windows
    window_dedup_iomin: float = 0.5       # IoMin threshold for deduplication

    # Amplification criteria (M3)
    dot_amplification_ratio: float = 2.0  # HER2/CEP17 ≥ 2.0
    dot_her2_count_threshold: int = 6     # or HER2 ≥ 6

    # Export (M4)
    cell_crop_size: int = 256             # per-cell crop dimensions
    slide_id: str = "unknown"
    model_version: str = "v1.0.0"
```

**Copy `config_example.py` → `config.py` for the full parameter list.**

### Triple Image Layer Pipeline

VALIS-based pipeline for CZI → BigTIFF → aligned tiles preprocessing.

```bash
cd backend/algorithms/thriple_image_layer
cp config_example.py config.py

# Full pipeline (alignment + ROI evaluation + thumbnail)
python run_full_pipeline.py

# With CZI preprocessing
python run_full_pipeline.py --preprocess

# Or run individual modules
python module1_preprocess.py   # CZI → BigTIFF
python module2_alignment.py    # VALIS registration
python module3_roi_evaluation.py
python module4_thumbnail.py
```

### Standalone Scripts

```bash
# Check TIFF file dimensions
python scripts/check_tiff_size.py --file image.tiff

# Test CUDA availability and info
python scripts/cuda_test.py

# Generate image tiles
python scripts/tile_generator.py --input image.tiff --output tiles/

# Convert TIFF to PNG
python "scripts/tiff to png.py" --input image.tiff --output image.png
```

---

## Pipeline Architecture

### Hybrid Analysis Pipeline (M1→M4)

```
┌──────────────────────────────────────────────────────────┐
│ Input: IHC (HER2) + DISH tile image pairs               │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ M1: Overlay & Core Mask (m1_overlay.py)                 │
│ - UNet++ (EfficientNet-B4) → IHC core mask              │
│ - Apply mask to IHC and DISH channels                   │
│ - 50/50 alpha blend → fused IHC-DISH overlay            │
│ - Empty core mask → short-circuit to empty CSV          │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ M2: Cell Segmentation (m2_segmentation.py)              │
│ - Cellpose on fused overlay → cell instance mask        │
│ - Sliding-window with overlap deduplication             │
│ - Clear border cells, renumber labels                   │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ M3: Cell Analysis & Dot Detection (m3_module/)          │
│ - Extract cell centroids and areas                      │
│ - Elastic IHC-DISH nucleus matching (nearest-first)     │
│ - Detect HER2 (black) and CEP17 (red) signal dots      │
│   via LAB color space + H-morphology on per-cell patch  │
│ - Classify: amplified if HER2/CEP17 ≥ 2.0 or HER2 ≥ 6 │
│ - Exclude: boundary-contaminated or no-candidate cells  │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ M4: Export (m4_export.py)                               │
│ - Per-tile CSV with per-cell dot counts & status        │
│ - Summary CSV with slide-level statistics               │
│ - Overlay visualizations (annotated PNG)                │
│ - 256×256 per-cell crops with dot annotations           │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ Output: CSV reports, overlay PNGs, per-cell crops       │
└──────────────────────────────────────────────────────────┘
```

### Triple Image Layer Pipeline (Preprocessing)

```
CZI files → Module 1 (BigTIFF conversion)
          → Module 2 (VALIS alignment)
          → Module 3 (ROI evaluation)
          → Module 4 (thumbnail)
          → Aligned BigTIFF  →  Hybrid Pipeline (M0 precuts the IHC + DISH tile pairs)
```

---

## Configuration

The hybrid pipeline uses a Python dataclass `Config` in `config.py`. Copy `config_example.py` and edit:

- **Paths**: `ihc_tile_dir`, `dish_tile_dir`, `output_dir`, model paths
- **UNet++**: `unet_encoder_name`, `unet_image_size`, `core_close_kernel`
- **Cellpose (M2)**: `cellpose_diameter`, `cellpose_flow_threshold`, `cellpose_cellprob_threshold`
- **Cellpose DISH (M3b)**: `cellpose_dish_diameter`, `cellpose_dish_flow_threshold`
- **Dot detection (M3)**: LAB thresholds, ring statistics, `dot_amplification_ratio`, `dot_her2_count_threshold`
- **Sliding window**: `window_overlap_px`, `window_dedup_iomin`
- **Tracing**: `slide_id`, `model_version` (written into every output CSV via `compute_config_hash()`)

---

## Output Formats

### CSV Outputs

**`{slide_id}_tile_{x}_{y}.csv`** — Per-cell per-tile analysis:
- `global_cell_id`: Unique cell identifier
- `centroid_y`, `centroid_x`: Cell center coordinates (tile space)
- `area`: Cell area in pixels
- `her2_dots` (black): HER2 signal dot count
- `cep17_dots` (red): CEP17 reference dot count
- `ratio`: HER2 / CEP17
- `status`: Amplified / Normal / Excluded (X) / boundary-contaminated

**`{slide_id}_summary.csv`** — Slide-level statistics:
- `total_cells`: Total cells analyzed
- `amplified`: Cells classified as amplified
- `her2_dot_distribution`, `cep17_dot_distribution`

### Visual Outputs

**`*_overlay.png`** — Fused IHC-DISH with cell outlines, dot markers, and amplification labels  
**`*_dot_only.png`** — Dot-only visualization  
**`cells/{cell_id}.png`** — 256×256 per-cell crops with dot annotations

---

## Hardware Requirements

### Minimum Configuration

- CPU: 4 cores
- RAM: 16 GB
- GPU: NVIDIA with 8 GB VRAM
- Storage: 100 GB SSD

### Recommended Configuration

- CPU: 8+ cores
- RAM: 32 GB
- GPU: NVIDIA A100 / RTX 4090 with 24 GB VRAM
- Storage: 1 TB NVMe SSD

### Performance Estimates

Per 1024×1024 tile (with GPU):

| Module | Time |
|--------|------|
| M1: UNet++ core mask | 0.1–0.3 s |
| M2: Cellpose segmentation | 1–3 s |
| M3: Dot detection | 0.1–0.5 s |
| M4: Export | 0.05–0.2 s |
| **Total per tile** | **~1.5–4 s** |

---

## Docker Support

### Building Docker Image

```bash
# Build with default settings
docker build -t tsgh:latest .

# Or using docker-compose
docker-compose build
```

### Running with Docker

```bash
# Interactive shell
docker run --rm --gpus all -it -v /data:/data tsgh:latest bash

# Run hybrid pipeline
docker run --rm --gpus all \
  -v /path/to/tiles:/tiles \
  -v /path/to/output:/output \
  tsgh:latest \
  python cell_mask/hybrid/hybrid_pipeline.py --batch

# Using docker-compose
docker-compose run --rm tsgh \
  python cell_mask/hybrid/hybrid_pipeline.py --batch
```

---

## Development

### Setting Up Development Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytest black flake8
pip install -e .
```

### Code Style

```bash
black cell_mask/ backend/algorithms/thriple_image_layer/ scripts/
flake8 cell_mask/ backend/algorithms/thriple_image_layer/ scripts/
```

### Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test
3. Commit: `git commit -m "feat: description"`
4. Push: `git push origin feature/your-feature`
5. Create pull request

---

## Troubleshooting

### CUDA/GPU Issues

```bash
# Check CUDA availability
python scripts/cuda_test.py

# Check GPU memory
nvidia-smi

# Run with CPU only (slow)
export CUDA_VISIBLE_DEVICES=""
python cell_mask/hybrid/hybrid_pipeline.py --batch
```

### Memory Issues

```bash
# Reduce sliding window overlap in config.py
window_overlap_px = 128  # default 256

# Reduce batch size if OOM during UNet++ inference
# (controlled via unet_image_size in config.py)
```

### OpenSlide Issues

```bash
# Install OpenSlide (Ubuntu)
sudo apt-get install libopenslide0 libopenslide-dev

# Install OpenSlide (macOS)
brew install openslide

# Verify installation
python -c "import openslide; print(openslide.__version__)"
```

### Common Errors

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `window_overlap_px` or `unet_image_size` in config |
| `OpenSlide not found` | Install system dependencies via apt/brew |
| `Model not found` | Check `unet_model_path`, `cellpose_model_path` in config.py |
| `No paired tiles found` | Verify tile filenames match `tile_x{int}_y{int}` convention |
| `config not found` | Run `cp config_example.py config.py` in `cell_mask/hybrid/` |

---

## Acknowledgments

This project builds upon:
- [Cellpose: A generalist algorithm for cellular segmentation](https://www.cellpose.org/)
- [Segmentation Models PyTorch (UNet++)](https://github.com/qubvel/segmentation_models.pytorch)
- [VALIS: Registration Framework for Whole Slide Images](https://github.com/MannLabs/valis)
- [OpenSlide for WSI Reading](https://openslide.org/)

---

**Last Updated:** June 2026  
**Version:** 0.1.0
