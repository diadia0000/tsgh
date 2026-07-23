# TSGH - Whole Slide Image Analysis Pipeline

A deep learning pipeline for automated HER2/CEP17 amplification analysis in histopathology. Precuts IHC (HER2) and DISH whole-slide-image pairs into overlapping tiles and runs an M0→M4 chain to detect cells, count signal dots, and classify amplification status. Ships with a FastAPI backend (`backend/api/`) and a React/OpenSeadragon viewer (`frontend/`) for browsing slides and triggering analysis, in addition to the CLI.

**Language:** Traditional Chinese / English
**Python Version:** 3.11 (pinned — `>=3.11,<3.12`)
**GPU:** NVIDIA CUDA required for practical runtime; this project's reference machine is an RTX 5090 (Blackwell, sm_120), which only runs the `cu130` PyTorch build (see [Hardware Requirements](#hardware-requirements))

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
- **Tiled, Bounded-Memory Architecture**: Any ROI or full WSI is precut to disk into overlapping 1024px tiles (M0) before analysis, so memory stays bounded regardless of slide size; per-tile results are deduplicated by centroid core-ownership and merged globally
- **Elastic IHC-DISH Matching**: Nearest-first, cell-centered nucleus locking to align cells across stain channels
- **Amplification Criteria**: Per-cell `score = HER2/CEP17`, amplified if `score ≥ 2.0` (cells with CEP17 count below `score_cep17_min_count` are excluded, not scored as 0); the slide-level `summary.txt` additionally reports a full ASCO/CAP 2013 case-level verdict (ratio + average HER2 copy number, incl. the equivocal band)
- **Flexible Output**: Global per-cell CSV report, an ASCO/CAP 2013 summary report, a stitched slide-level overlay TIFF, and per-cell crops
- **GPU Acceleration**: CUDA support for high-performance inference; optional cross-tile multiprocessing (`run_batch(workers=N)`) for further scaling — see [Usage](#hybrid-pipeline-main)
- **Containerized Deployment**: Docker support with GPU runtime
- **Image Preprocessing**: VALIS-based CZI→BigTIFF registration and tile generation
- **Web Viewer / API**: FastAPI backend + React/OpenSeadragon frontend for slide upload, alignment, and viewing results (see `docs/UI/`)

---

## Project Structure

```
tsgh/
├── backend/
│   ├── algorithms/
│   │   ├── hybrid/                       # Main IHC-DISH analysis pipeline (M0→M4)
│   │   │   ├── hybrid_pipeline.py        # Entry point + CLI (--test / --ihc --dish / --output)
│   │   │   ├── m0_reader.py              # Precut: WSI/ROI → overlapping tile files on disk
│   │   │   ├── m0_stitch.py              # Tile geometry, global dedup/renumber, slide-level stitch
│   │   │   ├── m1_overlay.py             # Module 1: UNet++ core mask → IHC-DISH fusion
│   │   │   ├── m2_segmentation.py        # Module 2: Cellpose cell segmentation
│   │   │   ├── m3_cell_detection.py      # Module 3: shim re-exporting m3_module
│   │   │   ├── m3_module/                # Module 3 sub-package
│   │   │   │   ├── m3_cells_generator.py     # Cell centroid extraction
│   │   │   │   ├── m3_elastic_matching.py    # IHC-DISH nucleus matching
│   │   │   │   ├── m3_dot_detection.py       # HER2/CEP17 dot detection
│   │   │   │   └── m3_dot_kernels.py         # Pixel-level dot kernels
│   │   │   ├── m4_export.py              # Module 4: facade for CSV + overlays + crops
│   │   │   ├── m4_module/
│   │   │   │   ├── csv.py                # report.csv + ASCO/CAP summary.txt export
│   │   │   │   ├── overlay.py            # Overlay visualization rendering
│   │   │   │   └── cell_crops.py         # Per-cell crop export
│   │   │   ├── hybrid_data_types.py      # Shared dataclasses
│   │   │   ├── unet_inference.py         # UNetPPInference (sliding-window)
│   │   │   ├── config_example.py         # Configuration template (copy → config.py, gitignored)
│   │   │   ├── test_picture/             # Bundled ROI pair for `--test`
│   │   │   └── CLAUDE.md                 # Module-level spec (authoritative, terse)
│   │   │
│   │   └── thriple_image_layer/          # VALIS-based preprocessing pipeline
│   │       ├── run_full_pipeline.py      # Orchestration entry point
│   │       ├── module1_preprocess.py     # CZI → BigTIFF conversion
│   │       ├── module2_alignment.py      # VALIS image alignment
│   │       ├── module3_roi_evaluation.py # ROI quality evaluation
│   │       ├── module4_thumbnail.py      # Thumbnail generation
│   │       ├── config_example.py
│   │       └── artifacts/                # Intermediate alignment artifacts
│   │
│   ├── api/                              # FastAPI routers (alignment, hybrid, jobs, tiles)
│   ├── schemas/                          # Pydantic request/response models
│   ├── io/                               # Shared I/O helpers
│   ├── tests/                            # pytest suite (chunked upload, alignment, resume, OpenSlide)
│   └── main.py                           # FastAPI app entrypoint (uvicorn, port 8000)
│
├── frontend/                             # React + Vite + OpenSeadragon viewer UI
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
│
├── cell_mask/                            # Model *training* only (not part of the runtime pipeline)
│   ├── unet_mask/                        # UNet++ training & inference tooling
│   └── dish_mask/                        # DISH Cellpose training & prediction tooling
│
├── scripts/                              # Utility & performance-measurement scripts
│   ├── check_tiff_size.py                # Compare two TIFFs' dimensions
│   ├── cuda_test.py                      # Report CUDA/GPU availability & info
│   ├── tile_generator.py                 # Ad-hoc tile-cutting helper (edit paths in __main__)
│   ├── perf_measure.py                   # Timed pipeline run + resource/GPU sampling
│   ├── mp_scaling_report.py / mp_concurrency_probe.py / verify_mp_failfast.py  # Cross-tile multiprocessing scaling & correctness (see docs/hybrid-pipeline/21-...)
│   ├── gc_ablation_report.py / arm_report.py / aggregate_report.py / generate_report.py  # Bottleneck measurement tooling
│   └── ...
│
├── docs/                                 # Project documentation
│   ├── hybrid-pipeline/                  # Handoff docs for the M0–M4 pipeline: architecture,
│   │                                     #   benchmarks, optimization rounds, open backlog — start
│   │                                     #   at docs/hybrid-pipeline/README.md
│   ├── UI/                               # FastAPI+React handoff docs — start at docs/UI/README.md
│   ├── algo/                             # Algorithm design explainers (elastic matching, seam stitch)
│   └── BACKLOG.md                        # Cross-cutting open-items list (pipeline perf + UI)
│
├── Dockerfile                            # NVIDIA CUDA 13.0 base + uv-managed venv
├── docker-compose.yml
└── pyproject.toml                        # Dependencies + uv config (no requirements.txt — uv.lock is the lockfile)
```

---

## Quick Start

### Using Docker (Recommended)

```bash
# Build image
docker compose build

# Start the container (stays up; default command is bash)
docker compose up -d
docker compose exec tsgh bash

# Inside the container: run the FastAPI backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# ...or run the hybrid pipeline CLI directly
python backend/algorithms/hybrid/hybrid_pipeline.py --test
```

### Local Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd tsgh

# 2. Install dependencies with uv (manages its own Python 3.11 venv per pyproject.toml)
pip install uv
uv sync

# 3. Configure and run the hybrid pipeline
cd backend/algorithms/hybrid
cp config_example.py config.py
# Edit config.py with your model and tile paths
cd ../../..
uv run python backend/algorithms/hybrid/hybrid_pipeline.py --test
```

There is no `requirements.txt` — `pyproject.toml` + `uv.lock` is the single source of truth for dependencies.

---

## Installation

### System Requirements

- **OS**: Ubuntu 20.04+ (Docker image targets Ubuntu 24.04), macOS, or Windows (with WSL2)
- **Python**: 3.11 only (`requires-python = ">=3.11,<3.12"` in `pyproject.toml`)
- **GPU**: NVIDIA GPU with CUDA support strongly recommended; Blackwell cards (RTX 50-series) **require** the `cu130` PyTorch build — see [Hardware Requirements](#hardware-requirements)
- **RAM**: 16+ GB (32+ GB recommended for large WSI / multi-process runs)
- **Storage**: SSD with 100+ GB free space (for intermediate results)

### Dependencies

**Key Libraries:**
- PyTorch 2.11.0 + torchvision 0.26.0, pinned to the `cu130` build via a dedicated `[[tool.uv.index]]` (`download.pytorch.org/whl/cu130`)
- OpenSlide (for WSI reading)
- Cellpose (cell segmentation), Segmentation-models-pytorch + timm (UNet++/EfficientNet-B4), `dinov3` (git-pinned)
- VALIS (image alignment/registration) — installed from a git fork (`tool.uv.sources`), not PyPI
- scikit-image, scipy, pandas, OpenCV (image processing & analysis)
- pyvips, tifffile, imagecodecs (large image I/O)
- FastAPI + uvicorn (backend API); React + Vite + OpenSeadragon (frontend, separate `npm` toolchain in `frontend/`)

**Full dependency list:** see `pyproject.toml` (there is no `requirements.txt`).

### Installation Steps

#### Option 1: Using uv (Recommended)

```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install dependencies (Python 3.11 + cu130 torch)
uv sync
```

#### Option 2: Using pip

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Install PyTorch pinned to the cu130 build (matches pyproject.toml's tool.uv.index)
pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu130

# Install the rest of the dependencies from pyproject.toml
pip install .
```

`uv sync` (Option 1) is strongly preferred: it also resolves the VALIS git fork and the pinned `dinov3` commit that plain `pip install .` may not reproduce identically.

#### Option 3: Using Docker

```bash
docker build -t tsgh:latest .
```

---

## Usage

### Hybrid Pipeline (Main)

The hybrid pipeline precuts an IHC (HER2) + DISH image pair (a single tile, an arbitrary ROI, or a full WSI of any size) into overlapping tiles on disk (M0), then analyzes every tile through M1→M4, merges results globally, and stitches an annotated slide-level overlay.

#### Setup

```bash
cd backend/algorithms/hybrid
cp config_example.py config.py
# Edit config.py — set model paths, tile directories, output_dir, slide_id
```

#### Basic Usage

```bash
# Bundled smoke-test ROI pair (test_picture/), full precut+analysis path
python hybrid_pipeline.py --test

# Single ROI/WSI image pair, any size — precut then analyzed
python hybrid_pipeline.py \
  --ihc roi_ihc.tiff \
  --dish roi_dish.tiff

# Custom output directory
python hybrid_pipeline.py --test --output /path/to/output
```

Tile pairing during precut is by filename coordinate: `tile_x{int}_y{int}`. There is no `--batch` flag — `--ihc`/`--dish` already accepts a whole WSI and precuts it internally (`PrecutStream`), so there is no separate "batch over a directory of pre-cut tiles" mode.

**Cross-tile multiprocessing (`workers=N`)**: `run_batch()` (called internally, not yet exposed as a CLI flag on `hybrid_pipeline.py`) accepts a `workers` argument that runs `N` `spawn`-ed processes, each with its own model set and CUDA context, over a shared dynamic tile queue. Measured **3.09x** at `workers=3` on the reference RTX 5090. It defaults to `workers=1` (today's sequential behavior; the API path never passes it) and is **not yet cleared for production** — it's gated on full-WSI-scale validation. Full detail: [`docs/hybrid-pipeline/21-cross-tile-multiprocessing-implementation.md`](docs/hybrid-pipeline/21-cross-tile-multiprocessing-implementation.md). `scripts/perf_measure.py --mp-workers N` is the current way to exercise it.

#### Key Configuration Parameters

A representative subset of `Config` (see `config_example.py` for the full, commented list — it also includes ~40 DISH dot-detection tuning parameters):

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

@dataclass
class Config:
    # Tile input directories / test ROI / output
    ihc_tile_dir: Path = ...              # default: <hybrid dir>/tile/her2
    dish_tile_dir: Path = ...             # default: <hybrid dir>/tile/dish
    ihc_test_path: Path = ...             # bundled ROI used by --test
    dish_test_path: Path = ...
    output_dir: Path = ...                # default: <hybrid dir>/output

    # Model paths (M1/M2/M3b)
    unet_model_path: Path = ...
    cellpose_model_path: Path = ...
    cellpose_dish_model_path: Path = ...

    # UNet++ parameters (M1)
    unet_encoder_name: str = "timm-efficientnet-b4"
    unet_image_size: Tuple[int, int] = (1024, 1024)

    # Cellpose parameters (M2: cell segmentation)
    cellpose_diameter: Optional[float] = None   # auto-detect
    cellpose_flow_threshold: float = 0.6
    cellpose_cellprob_threshold: float = -0.8
    cellpose_batch_size: int = 16               # per-tile internal patch batch, not cross-tile

    # Tiling / sliding-window deduplication (M0/M2/M3b)
    default_tile_size: int = 1024
    window_overlap_px: int = 256          # overlap between adjacent windows
    window_dedup_iomin: float = 0.5       # IoMin threshold for deduplication

    # Amplification criteria (M3) — see m3_dot_detection.py for the full ASCO/CAP logic
    score_cep17_min_count: int = 2        # CEP17 < this and not 0/0 → excluded
    dot_amplification_ratio: float = 2.0  # score = HER2/CEP17 ≥ 2.0 → amplified

    # Export (M4)
    cell_crop_size: int = 100
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
# Compare two TIFF files' dimensions (positional args, not flags)
python scripts/check_tiff_size.py file1.tiff file2.tiff

# Report CUDA/GPU availability and info (no args)
python scripts/cuda_test.py

# tile_generator.py has no CLI — it's an ad-hoc script; edit the paths in its
# `if __name__ == "__main__":` block before running:
python scripts/tile_generator.py
```

Performance-measurement scripts (`perf_measure.py`, `mp_scaling_report.py`, `mp_concurrency_probe.py`, `arm_report.py`, `gc_ablation_report.py`, …) are the tooling behind `docs/hybrid-pipeline/measurement/`'s benchmark record — see [`docs/hybrid-pipeline/05-dev-testing-guide.md`](docs/hybrid-pipeline/05-dev-testing-guide.md) and [`docs/hybrid-pipeline/README.md`](docs/hybrid-pipeline/README.md) before reaching for them directly.

---

## Pipeline Architecture

### Hybrid Analysis Pipeline (M0→M4)

```
┌──────────────────────────────────────────────────────────┐
│ Input: IHC (HER2) + DISH image pair (tile, ROI, or WSI)  │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ M0: Precut (m0_reader.py / m0_stitch.py)                 │
│ - Cut IHC+DISH into overlapping tile files on disk        │
│ - Streamed into the analysis loop (not a blocking pass)   │
│ - Global dedup by centroid core-ownership, renumber once  │
│ - Optional: N `spawn`-ed worker processes (workers=N)     │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ M1: Overlay & Core Mask (m1_overlay.py)                 │
│ - UNet++ (EfficientNet-B4) → IHC core mask              │
│ - Apply mask to IHC and DISH channels                   │
│ - Alpha blend (overlay_alpha) → fused IHC-DISH overlay   │
│ - Empty core mask → short-circuit to empty CSV           │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ M2: Cell Segmentation (m2_segmentation.py)              │
│ - Cellpose on fused overlay → cell instance mask        │
│ - Sliding-window with overlap deduplication              │
│ - Interior seams left for M0's cross-tile dedup           │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ M3: Cell Analysis & Dot Detection (m3_module/)          │
│ - Extract cell centroids                                 │
│ - Elastic, cell-centered IHC-DISH nucleus matching        │
│ - Detect HER2 (black) and CEP17 (red) signal dots        │
│   via LAB color space + H-morphology on per-cell patch    │
│ - Per-cell score = HER2/CEP17; amplified if score ≥ 2.0   │
│ - Exclude: boundary-contaminated or CEP17-insufficient    │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ M4: Export (m4_export.py)                                │
│ - Global report.csv (per-cell dot counts & score)         │
│ - summary.txt (ASCO/CAP 2013 slide-level verdict)         │
│ - overlay_annotated/ per-tile + stitched overlay_slide.tiff│
│ - Per-cell crops with dot annotations                     │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ Output: report.csv, summary.txt, overlay_slide.tiff,      │
│         overlay_annotated/, cell_crops/                   │
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

The hybrid pipeline uses a Python dataclass `Config` in `config.py` (gitignored — copy from `config_example.py` and edit):

- **Paths**: `ihc_tile_dir`, `dish_tile_dir`, `ihc_test_path`/`dish_test_path`, `output_dir`, model paths
- **UNet++ (M1)**: `unet_encoder_name`, `unet_image_size`, `core_close_kernel`, `overlay_alpha`
- **Cellpose (M2)**: `cellpose_diameter`, `cellpose_flow_threshold`, `cellpose_cellprob_threshold`, `cellpose_batch_size`
- **Cellpose DISH (M3b)**: `cellpose_dish_diameter`, `cellpose_dish_flow_threshold`
- **Dot detection (M3)**: ~40 LAB/morphology/ring-statistics thresholds per color (red=CEP17, black=HER2) — see the fully commented block in `config_example.py`
- **Elastic matching (M3)**: `cell_enlarge_area_factor`, `dish_elastic_expand_factor`, `dish_elastic_min_reach_px`
- **Amplification criteria (M3)**: `score_cep17_min_count`, `dot_amplification_ratio`
- **Tiling / sliding window**: `default_tile_size`, `window_overlap_px`, `window_dedup_iomin`
- **Tracing**: `slide_id`, `model_version` (written into every output CSV via `compute_config_hash()`)

---

## Output Formats

Per `run_batch()` invocation, output lands under `output_dir/` (one global result set, not per-tile files):

### `report.csv` — global per-cell analysis

Columns: `cell_id`, `centroid_x`, `centroid_y`, `reddot` (CEP17 signal count), `blackdot` (HER2 signal count), `score` (HER2/CEP17; `NaN` for excluded cells — boundary-contaminated or CEP17 below `score_cep17_min_count`).

### `summary.txt` — ASCO/CAP 2013 slide-level verdict

Case-level ratio (`ΣHER2/ΣCEP17`) and average HER2 copy number over valid cells, the resulting verdict (amplified / equivocal / not amplified / insufficient cells), and a per-cell distribution breakdown (ratio buckets, HER2 copy-number buckets).

### Visual Outputs

- **`overlay_slide.tiff`** — pyramidal, QuPath-openable slide-level overlay: cell boundaries, DISH nucleus contours, drift arrows, labels, HER2/CEP17 dot markers, stitched from per-tile `overlay_annotated/tile_x{x}_y{y}.tiff`
- **`cell_crops/tile_x{x}_y{y}/`** — per-cell crops (`cell_crop_size`) with dot annotations

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
- GPU: NVIDIA with 24+ GB VRAM (the reference/measured machine is an RTX 5090, 32 GB)
- Storage: 1 TB NVMe SSD

**Blackwell (RTX 50-series) caveat**: this architecture (`sm_120`) only runs with the `cu130` PyTorch build (`torch==2.11.0+cu130`, pinned in `pyproject.toml`). Do not downgrade torch or install an older CUDA wheel — see [`docs/hybrid-pipeline/06-versions-dependencies.md`](docs/hybrid-pipeline/06-versions-dependencies.md).

### Performance

Real, measured end-to-end numbers (RTX 5090, real WSI crops) live in [`docs/hybrid-pipeline/measurement/bottleneck-list.md`](docs/hybrid-pipeline/measurement/bottleneck-list.md) and are kept current there rather than duplicated here. Headline: single-process (`workers=1`) is ~1.1 s/tile at the 441-tile anchor (~10.5 h upper-bound full-WSI estimate); with cross-tile multiprocessing at `workers=3` that drops to ~0.33 s/tile (~3.3 h upper-bound), pending the production gate noted in [Usage](#hybrid-pipeline-main).

---

## Docker Support

### Building Docker Image

```bash
# Build with default settings (base image: nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04)
docker build -t tsgh:latest .

# Or using docker compose
docker compose build
```

### Running with Docker

```bash
# Interactive shell (default command; project dir + a host storage dir are mounted — see docker-compose.yml)
docker compose up -d
docker compose exec tsgh bash

# Inside the container: run the FastAPI backend (port 8000, mapped to the host)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# ...or the hybrid pipeline CLI
python backend/algorithms/hybrid/hybrid_pipeline.py --test

# One-off equivalent without an interactive shell
docker compose run --rm tsgh \
  python backend/algorithms/hybrid/hybrid_pipeline.py --test
```

Dependencies are installed at build time via `uv sync --frozen --no-dev --no-install-project` against `uv.lock` — there is no `pip install -r requirements.txt` step.

---

## Development

### Setting Up Development Environment

```bash
uv sync            # installs the pinned Python 3.11 + cu130 torch environment
```

Frontend (`frontend/`) has its own toolchain — see [`docs/UI/06-dev-setup.md`](docs/UI/06-dev-setup.md):

```bash
cd frontend
npm install
npm run dev      # Vite dev server
npm run build     # tsc -b && vite build
npm run lint      # oxlint
```

### Code Style

`pyproject.toml` does not currently pin a formatter/linter config; if you use `black`/`flake8`/`ruff` locally, target the actual source trees:

```bash
black backend/ scripts/
flake8 backend/ scripts/
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

# Run with CPU only (slow, not validated against the current pipeline)
export CUDA_VISIBLE_DEVICES=""
python backend/algorithms/hybrid/hybrid_pipeline.py --test
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
| `CUDA out of memory` | Reduce `window_overlap_px` or `unet_image_size` in config, or lower `workers` if using cross-tile multiprocessing |
| `OpenSlide not found` | Install system dependencies via apt/brew |
| `Model not found` | Check `unet_model_path`, `cellpose_model_path` in config.py |
| `找不到 tile` (tile not found) | Verify tile filenames match `tile_x{int}_y{int}` convention |
| `config not found` | Run `cp config_example.py config.py` in `backend/algorithms/hybrid/` |

---

## Acknowledgments

This project builds upon:
- [Cellpose: A generalist algorithm for cellular segmentation](https://www.cellpose.org/)
- [Segmentation Models PyTorch (UNet++)](https://github.com/qubvel/segmentation_models.pytorch)
- [VALIS: Registration Framework for Whole Slide Images](https://github.com/MannLabs/valis)
- [OpenSlide for WSI Reading](https://openslide.org/)

---

**Last Updated:** 2026-07-23
**Version:** 0.1.0
