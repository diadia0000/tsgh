# TSGH - Whole Slide Image Analysis Pipeline

A comprehensive deep learning-based pipeline for automated analysis of whole slide images (WSI) in histopathology. This project processes large pathology images using sliding-window techniques to detect cells, segment cellular regions, and extract quantitative analysis for research and clinical applications.

**Language:** Traditional Chinese / English  
**Python Version:** 3.11  
**GPU:** NVIDIA CUDA 13.0 (recommended)

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Full WSI Pipeline](#full-wsi-pipeline)
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

✨ **Key Capabilities:**

- **Whole Slide Image Processing**: Efficiently handles large WSI files (>10 GB) using lazy loading
- **Cell Detection & Segmentation**: Deep learning models for automated cell identification:
  - UNet++ for tissue segmentation
  - Cellpose for cell instance segmentation
  - Multi-dot detection for marker analysis
- **Sliding Window Architecture**: Process large images without loading entire dataset into RAM
- **GPU Acceleration**: CUDA 13.0 support for high-performance inference
- **Flexible Output**: CSV reports, stitched masks, and statistical summaries
- **Containerized Deployment**: Docker support with GPU runtime
- **Modular Design**: Reusable components for custom analysis workflows

---

## Project Structure

```
tsgh/
├── full_wsi_run/                    # Full WSI end-to-end pipeline
│   ├── full_wsi_pipeline.py         # Main entry point (M1→M5)
│   ├── m1_overlay.py                # Module 1: Image overlay & preprocessing
│   ├── m2_segmentation.py           # Module 2: Cell segmentation (Cellpose)
│   ├── m3_cells_generator.py        # Module 3: Cell analysis & filtering
│   ├── m3_dot_detection.py          # Module 3b: Multi-dot detection
│   ├── m4_export.py                 # Module 4: Result export & statistics
│   ├── m5_tiffwriter.py             # Module 5: Write stitched BigTIFF masks
│   ├── unet_inference.py            # UNet++ inference & postprocessing
│   ├── config_example.py            # Configuration template
│   └── README.md                    # Detailed pipeline documentation
│
├── thriple_image_layer/             # Triple image layer processing
│   ├── module1_preprocess.py        # Preprocessing module
│   ├── module2_alignment.py         # Image alignment
│   ├── module3_roi_evaluation.py    # ROI evaluation
│   ├── module4_thumbnail.py         # Thumbnail generation
│   ├── module5_tile_generator.py    # Tile generation
│   ├── run_full_pipeline.py         # Pipeline orchestration
│   ├── config_example.py            # Configuration template
│   └── artifacts/                   # Documentation and analysis results
│
├── cell_mask/                       # Cell detection & mask modules
│   ├── unet_mask/                   # UNet++ mask generation
│   ├── dish_mask/                   # Dish-based mask detection
│   └── hybrid/                      # Hybrid approach combining multiple methods
│       ├── m1_overlay.py
│       ├── m2_segmentation.py
│       └── m3_cells_generator.py
│
├── scripts/                         # Utility scripts
│   ├── check_tiff_size.py          # Check TIFF file dimensions
│   ├── cuda_test.py                # Test CUDA availability
│   ├── tiff_preview_server.py      # Web server for TIFF preview
│   └── tile_generator.py           # Generate image tiles
│
├── Dockerfile                       # Docker container definition
├── docker-compose.yml               # Docker Compose configuration
├── pyproject.toml                   # Project metadata and dependencies
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Quick Start

### Using Docker (Recommended)

```bash
# Build Docker image
docker-compose build

# Run pipeline in container
docker-compose run --rm tsgh python full_wsi_run/full_wsi_pipeline.py

# Or with custom parameters
docker-compose run --rm tsgh python full_wsi_run/full_wsi_pipeline.py \
  --window 2048 --overlap 128 --limit 20
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

# 4. Run pipeline
python full_wsi_run/full_wsi_pipeline.py
```

---

## Installation

### System Requirements

- **OS**: Ubuntu 20.04+, macOS, or Windows (with WSL2)
- **Python**: 3.11
- **GPU**: NVIDIA GPU with CUDA Compute Capability 7.0+ (recommended)
- **CUDA**: 13.0 or later
- **RAM**: 16+ GB (32+ GB recommended for large WSI)
- **Storage**: SSD with 100+ GB free space (for intermediate results)

### Dependencies

**Key Libraries:**
- PyTorch 2.10+ with CUDA 13.0
- OpenSlide 4.0+ (for WSI reading)
- Cellpose (cell segmentation)
- Segmentation-models-pytorch (UNet++)
- VALIS (image alignment)
- scikit-image, scipy, pandas (image processing & analysis)

**Full dependency list:** See `requirements.txt`

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

### Full WSI Pipeline

The full WSI pipeline processes entire whole slide images using a sliding-window approach.

#### Basic Usage

```bash
cd full_wsi_run

# Copy and configure
cp config_example.py config.py

# Edit config.py with:
# - ihc_wsi_path: Path to IHC (immunohistochemistry) image
# - dish_wsi_path: Path to DISH (dual in-situ hybridization) image
# - slide_id: Unique identifier for the slide
# - output_dir: Directory for results
# - Model paths (UNet++, Cellpose)

# Run pipeline
python full_wsi_pipeline.py
```

#### Advanced Usage

```bash
# Test run with limited windows (20 windows)
python full_wsi_pipeline.py --window 1024 --overlap 128 --limit 20

# Custom window size and overlap
python full_wsi_pipeline.py --window 2048 --overlap 256

# Full production run
python full_wsi_pipeline.py --save-stitched-core-mask --save-stitched-instance-mask
```

#### Configuration Parameters

Edit `config.py` to customize:

```python
config = {
    # Input paths
    "ihc_wsi_path": "/path/to/ihc_slide.tiff",
    "dish_wsi_path": "/path/to/dish_slide.tiff",
    
    # Output
    "slide_id": "SLIDE_001",
    "output_dir": "/path/to/output",
    
    # Pipeline parameters
    "window_size": 1024,          # Sliding window size
    "overlap": 128,               # Window overlap
    "min_cell_size": 50,          # Minimum cell size (pixels)
    "cellpose_diameter": 30,      # Expected cell diameter
    
    # Model paths
    "unet_model_path": "/path/to/unet_model.pth",
    "cellpose_model_path": "/path/to/cellpose_model",
    
    # Processing options
    "skip_background": True,      # Skip background-only windows
    "save_per_window_artifacts": False,
    "save_stitched_core_mask": False,
    "save_stitched_instance_mask": False,
}
```

**See `full_wsi_run/README.md` for detailed architecture and output specifications.**

### Triple Image Layer Pipeline

For initial image preprocessing and alignment.

```bash
cd thriple_image_layer

# Configure pipeline
cp config_example.py config.py

# Run complete pipeline
python run_full_pipeline.py

# Or run individual modules
python module1_preprocess.py
python module2_alignment.py
python module3_roi_evaluation.py
python module4_thumbnail.py
python module5_tile_generator.py
```

### Standalone Scripts

Utility scripts in `scripts/` directory:

```bash
# Check TIFF file dimensions
python scripts/check_tiff_size.py --file image.tiff

# Test CUDA availability and info
python scripts/cuda_test.py

# Start TIFF preview server (web-based viewer)
python scripts/tiff_preview_server.py --port 8000

# Generate image tiles
python scripts/tile_generator.py --input image.tiff --output tiles/
```

---

## Pipeline Architecture

### Processing Pipeline (M1→M5)

The pipeline consists of 5 integrated modules:

```
┌─────────────────────────────────────────────────────────┐
│ Input: IHC + DISH Whole Slide Images (WSI)             │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ M1: Overlay & Preprocessing                            │
│ - Register IHC to DISH images                          │
│ - Apply tissue masks                                   │
│ - Normalize intensities                                │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ M2: Cell Segmentation (Cellpose)                       │
│ - Instance segmentation                                │
│ - Cell boundary detection                              │
│ - Clear border cells                                   │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ M3: Cell Analysis & Dot Detection                      │
│ - Extract cell properties (centroid, area)             │
│ - Detect dots (red/black markers)                      │
│ - Filter by size/quality thresholds                    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ M4: Export & Statistics                                │
│ - Generate per-cell CSV report                         │
│ - Calculate slide-level statistics                     │
│ - Create summary CSV                                   │
│ - Render overlay visualizations                        │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ M5: Mask Writing (Optional)                            │
│ - Write stitched core mask (BigTIFF)                   │
│ - Write stitched instance mask (BigTIFF)               │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Output: CSV Reports, Masks, Statistics                 │
└─────────────────────────────────────────────────────────┘
```

### Sliding Window Processing

Large WSI are processed in overlapping tiles to manage memory:

- **Window Size**: Configurable (default 1024×1024 pixels)
- **Overlap**: Prevents edge artifacts (default 128 pixels)
- **Lazy Loading**: Only current window loaded into memory
- **GPU Processing**: Batch inference on GPU
- **Post-processing**: Merge results across windows with coordinate transformation

---

## Configuration

### Main Configuration (config.py)

```python
# Input/Output
ihc_wsi_path = "path/to/ihc_slide.tiff"
dish_wsi_path = "path/to/dish_slide.tiff"
slide_id = "SLIDE_001"
output_dir = "results"

# Pipeline Parameters
window_size = 1024
overlap = 128
min_cell_size = 50
max_cell_size = 10000
cellpose_diameter = 30
cellpose_flow_threshold = 0.4
cellpose_prob_threshold = 0.0

# Models
unet_model_path = "models/unet_v2.pth"
cellpose_model_path = "models/cellpose_model"

# Processing
skip_background = True
num_workers = 4
batch_size = 4
save_per_window_artifacts = False
save_stitched_core_mask = True
save_stitched_instance_mask = True
```

---

## Output Formats

### CSV Outputs

**`{slide_id}_report.csv`** - Per-cell detailed analysis:
- `global_cell_id`: Unique cell identifier
- `centroid_y`, `centroid_x`: Cell center coordinates (WSI space)
- `area`: Cell area in pixels
- `red_dots`: Number of red markers in cell
- `black_dots`: Number of black markers in cell
- `ratio`: red_dots / (red_dots + black_dots)
- `status`: Cell quality indicator

**`{slide_id}_summary.csv`** - Slide-level statistics:
- `total_cells`: Total cells detected
- `valid_bichromatic`: Cells with both red and black dots
- `ratio_lt2`: Cells with ratio < 2
- `ratio_ge2`: Cells with ratio ≥ 2
- `red_copy_distribution`: Distribution of red dot counts
- `black_copy_distribution`: Distribution of black dot counts

### Mask Outputs (Optional)

**`{slide_id}_core_mask.tiff`** - Tissue segmentation mask
- Format: uint8 BigTIFF
- Size: Full WSI dimensions
- Value: 0 (background) or 1 (tissue)

**`{slide_id}_instance_mask.tiff`** - Cell instance segmentation
- Format: uint32 BigTIFF
- Size: Full WSI dimensions
- Value: 0 (background) or cell_id (1, 2, 3, ...)

### Per-Window Artifacts (Optional)

When `save_per_window_artifacts=True`, each window's results saved to:
```
windows/
├── w_y{Y}_x{X}/
│   ├── overlay.png
│   ├── core_mask.npy
│   ├── instance_mask.npy
│   ├── cells_analysis.json
│   └── dots_detection.json
```

---

## Hardware Requirements

### Minimum Configuration

- CPU: 4 cores (8 cores recommended)
- RAM: 16 GB
- GPU: NVIDIA with 4 GB VRAM (8 GB recommended)
- Storage: 100 GB SSD

### Recommended Configuration

- CPU: 8+ cores
- RAM: 32 GB
- GPU: NVIDIA A100 / RTX 4090 with 24 GB VRAM
- Storage: 1 TB NVMe SSD

### Performance Estimates

For a typical 114k × 141k WSI (~2,000 windows):

| Component | Time/Window | Total Time |
|-----------|------------|-----------|
| UNet++ segmentation | 0.1–0.3 s | ~3–10 min |
| Cellpose detection | 1–3 s | ~30–100 min |
| Dot detection | 0.05–0.2 s | ~2–7 min |
| Post-processing | 0.05–0.1 s | ~2–5 min |
| **Total** | **~1.5–4 s** | **~1–2 hours** |

**Notes:**
- Times are with GPU acceleration
- Background skipping can reduce processing by 30–70%
- Instance mask stitching adds ~50% disk I/O overhead if enabled
- Large instance mask files (~64 GB for 114k×141k image)

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

# Run pipeline
docker run --rm --gpus all \
  -v /path/to/data:/data \
  -v /path/to/output:/output \
  tsgh:latest \
  python full_wsi_run/full_wsi_pipeline.py

# Using docker-compose
docker-compose run --rm tsgh python full_wsi_run/full_wsi_pipeline.py
```

### Dockerfile Features

- NVIDIA CUDA 13.0 base image
- Ubuntu 24.04 OS
- Python 3.11
- Pre-configured GPU support
- All system dependencies included
- Ready for deployment

---

## Development

### Setting Up Development Environment

```bash
# Create development environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies with dev tools
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Install in editable mode
pip install -e .
```

### Project Structure for Development

```
tsgh/
├── full_wsi_run/          # Main pipeline modules
├── thriple_image_layer/   # Alternative pipeline
├── cell_mask/             # Core algorithms
├── scripts/               # Utility tools
├── tests/                 # Unit tests (if added)
└── docs/                  # Documentation (if added)
```

### Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=full_wsi_run
```

### Code Style

```bash
# Format code with black
black full_wsi_run/ thriple_image_layer/ cell_mask/

# Check code style with flake8
flake8 full_wsi_run/ thriple_image_layer/ cell_mask/
```

### Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test
3. Commit with descriptive message: `git commit -m "feat: description"`
4. Push to branch: `git push origin feature/your-feature`
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
python full_wsi_run/full_wsi_pipeline.py
```

### Memory Issues

```bash
# Reduce window size
python full_wsi_run/full_wsi_pipeline.py --window 512

# Reduce batch size in config.py
batch_size = 2  # default 4

# Process with limit for testing
python full_wsi_run/full_wsi_pipeline.py --limit 10
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
| `CUDA out of memory` | Reduce `window_size` or `batch_size` in config |
| `OpenSlide not found` | Install system dependencies via apt/brew |
| `Model not found` | Check model paths in config.py |
| `Permission denied` | Ensure write permissions on output directory |

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{tsgh2024,
  title={TSGH: Whole Slide Image Analysis Pipeline},
  author={Your Institution},
  year={2024},
  url={https://github.com/diadia0000/tsgh}
}
```

---

## License

[Add your license information here]

---

## Support & Contact

For issues, questions, or contributions:

- **Issues**: [GitHub Issues](https://github.com/diadia0000/tsgh/issues)
- **Discussions**: [GitHub Discussions](https://github.com/diadia0000/tsgh/discussions)

---

## Acknowledgments

This project builds upon:
- [UNet++ for Medical Image Segmentation](https://github.com/4uiiurz1/pytorch-unet-plus-plus)
- [Cellpose: A generalist algorithm for cellular segmentation](https://www.cellpose.org/)
- [VALIS: Registration Framework for Whole Slide Images](https://github.com/MannLabs/valis)
- [OpenSlide for WSI Reading](https://openslide.org/)

---

**Last Updated:** May 2026  
**Version:** 0.1.0
