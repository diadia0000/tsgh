# TSGH — Whole Slide Image Analysis Pipeline

Automated HER2/CEP17 amplification analysis for histopathology. Precuts an IHC (HER2) + DISH whole-slide-image pair into overlapping tiles and runs an M0→M4 chain to detect cells, count signal dots, and classify amplification. Ships a CLI, a FastAPI backend (`backend/api/`), and a React/OpenSeadragon viewer (`frontend/`).

- **Python:** 3.11 only (`>=3.11,<3.12`)
- **GPU:** NVIDIA CUDA. Reference machine is an RTX 5090 (Blackwell, `sm_120`), which requires the `cu130` PyTorch build — `torch==2.11.0+cu130` is pinned in `pyproject.toml`. Do not downgrade.

---

## Quick Start

```bash
# uv manages its own Python 3.11 venv from pyproject.toml (no requirements.txt)
uv sync

cd backend/algorithms/hybrid
cp config_example.py config.py     # config.py is gitignored; edit model + tile paths
cd ../../..

uv run python backend/algorithms/hybrid/hybrid_pipeline.py --test
```

Docker:

```bash
docker compose build
docker compose up -d && docker compose exec tsgh bash
# inside: uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
tsgh/
├── backend/
│   ├── algorithms/
│   │   ├── hybrid/                     # Main IHC-DISH pipeline (M0→M4) — see its CLAUDE.md
│   │   │   ├── hybrid_pipeline.py      # Entry point + CLI
│   │   │   ├── m0_slide.py             # Facade over m0_module/ (import this, not the internals)
│   │   │   ├── m0_module/              # Precut, tile geometry/dedup/stitch, tile runner,
│   │   │   │                           #   spawn multiprocessing, resume checkpoints
│   │   │   ├── m1_overlay.py           # UNet++ core mask → IHC-DISH fusion
│   │   │   ├── m2_segmentation.py      # Cellpose cell segmentation
│   │   │   ├── m3_cell_detection.py    # Re-export shim for m3_module/
│   │   │   ├── m3_module/              # Centroids, elastic nucleus matching, dot detection
│   │   │   ├── m4_export.py            # Facade over m4_module/{csv,overlay}.py
│   │   │   ├── unet_inference.py       # UNetPPInference (sliding-window)
│   │   │   ├── hybrid_data_types.py    # Shared dataclasses
│   │   │   └── config_example.py       # Config template → copy to config.py
│   │   └── thriple_image_layer/        # VALIS preprocessing: CZI → BigTIFF → aligned slides
│   ├── api/                            # FastAPI routers: alignment, hybrid, jobs, tiles
│   ├── schemas/                        # Pydantic request/response models
│   ├── io/pyramid.py                   # Pyramidal TIFF I/O helpers
│   ├── tests/                          # API-side pytest suite
│   └── main.py                         # FastAPI app (uvicorn, port 8000)
├── frontend/                           # React + Vite + OpenSeadragon viewer
├── cell_mask/                          # Model *training* only, not part of the runtime pipeline
├── scripts/                            # Standalone utilities + perf/probe scripts
├── tests/                              # Pipeline pytest suite
├── docs/UI/                            # FastAPI + React handoff docs (start at README.md)
├── docs/BACKLOG.md                     # Open items
├── Dockerfile / docker-compose.yml     # nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
└── pyproject.toml                      # Deps + uv config; uv.lock is the lockfile
```

---

## Usage

### Hybrid pipeline (main)

Takes a single tile, an arbitrary ROI, or a full WSI of any size. `PrecutStream` cuts overlapping 1024px tiles into `output_dir/_precut_scratch/` *while* the batch analyzes them, so memory stays bounded regardless of slide size.

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

**Single-process default got faster too**: a one-line config fix (`dot_detect_n_jobs: int = 1`, replacing an unbounded joblib fan-out inside `detect_all_dots`) measured **1.60x** at `workers=1` — the production default — because the removed background threads were starving the GPU main thread of the GIL, not because the CPU stage itself got cheaper. Detail: [`docs/hybrid-pipeline/23-next-optimization-cycle-implementation.md`](docs/hybrid-pipeline/23-next-optimization-cycle-implementation.md) §4.

**Cross-tile multiprocessing (`workers=N`)**: `run_batch()` (called internally, not yet exposed as a CLI flag on `hybrid_pipeline.py`) accepts a `workers` argument that runs `N` `spawn`-ed processes, each with its own model set and CUDA context, over a shared dynamic tile queue. Measured **3.09x** at `workers=3` on the reference RTX 5090 (round 5); after the `dot_detect_n_jobs` fix landed, a worker-count re-sweep (round 6) revised the recommendation down to **`workers=4`** for unattended jobs / `workers=5` when a restart is cheap. `run_batch()` defaults to `workers=4`, the production setting below; `backend/api/hybrid.py`'s single-tile endpoint explicitly overrides it to `workers=1` so a one-tile request doesn't pay N workers' model-init cost. **Round 8 (2026-07-27) ran the full-WSI-scale validation that gated production and it passed**: on the real 27,565-tile slide, `workers=1` took 3.82 h and `workers=4` took 1.73 h (**2.216x measured speedup**, correctness veto passed). **Round 12 (2026-07-29) re-ran `workers=4` on current code — the first re-run since round 8 — and the speedup is now 1.745x** (5,854.9 s against round 10's 10,217.7 s at `workers=1`; correctness veto passed and tighter than round 8's, −0.002% rows). The tile-parallel arm itself did not regress (2.279x with the stitch excluded, inside the 2.1x–2.5x band the slide's 55.8% background composition imposes); the loss is entirely **Phase D stitch, now 32.3% of the `workers=4` wall** — see [`docs/hybrid-pipeline/39-round-12-multiprocess-scaling-ceiling-implementation.md`](docs/hybrid-pipeline/39-round-12-multiprocess-scaling-ceiling-implementation.md). `workers=4` is recommended for production; it peaked at **93.3% of the reference card's 32 GB (~2.2 GB headroom)**, so treat 32 GB VRAM as a hard floor. The intermittent CUDA allocator OOM this recommendation used to be gated on (first seen at `workers≥6`, later also observed at the shipped `workers=4`) is now **root-caused and mitigated (round 11)**: `config.cuda_alloc_conf = "expandable_segments:True"` eliminated it in a 12-repeat sweep at `workers=4` (4/12 → 0/12 OOM, +0.67% wall). **This is now the shipped default in `config_example.py`** (commit `b3fa47d`), so production `workers=4` runs on the tighter peak-framebuffer figure the knob produces — **92.2% of the card**, not round 8's 93.3%/~2.2 GB-headroom figure measured without it. The knob removes a failure mode; it does not create headroom. Full detail: [`docs/hybrid-pipeline/21-cross-tile-multiprocessing-implementation.md`](docs/hybrid-pipeline/21-cross-tile-multiprocessing-implementation.md) (round 5), [`docs/hybrid-pipeline/23-next-optimization-cycle-implementation.md`](docs/hybrid-pipeline/23-next-optimization-cycle-implementation.md) §6 (round 6 re-tune), [`docs/hybrid-pipeline/27-remaining-work-implementation.md`](docs/hybrid-pipeline/27-remaining-work-implementation.md) §5–§6 (round 8 full-slide validation), and [`docs/hybrid-pipeline/37-round-11-backlog-implementation.md`](docs/hybrid-pipeline/37-round-11-backlog-implementation.md) §3 (round 11 allocator fix). `scripts/perf_measure.py --mp-workers N` is the current way to exercise it; `run_batch(..., checkpoint=True)` (or `--resume`) enables opt-in partial-resume so an interrupted long-running batch doesn't have to restart from tile 0.

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
    dot_detect_n_jobs: int = 1                  # joblib workers for detect_all_dots; keep at 1 (see below)

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

python run_full_pipeline.py                # alignment + ROI evaluation + thumbnail
python run_full_pipeline.py --preprocess   # also run Module 1: CZI → BigTIFF

# or individually
python module1_preprocess.py
python module2_alignment.py
python module3_roi_evaluation.py
python module4_thumbnail.py
```

### Standalone scripts

```bash
python scripts/check_tiff_size.py file1.tiff file2.tiff   # positional args, not flags
python scripts/cuda_test.py                               # CUDA/GPU availability, no args
python scripts/tile_generator.py                          # no CLI — edit paths in __main__ first
```

The rest of `scripts/` (`mp_scaling_report.py`, `gc_ablation_report.py`, `full_wsi_validate.py`, `*_probe.py`, …) are one-off measurement tools, not part of the pipeline.

---

## Pipeline Architecture

```
IHC (HER2) + DISH pair (tile, ROI, or WSI)
   │
   ├─ M0  m0_module/ ── precut into overlapping 1024px tile files on disk, streamed into
   │                    the analysis loop; per-tile GPU/CPU stages overlap; cross-tile dedup
   │                    by centroid core-ownership; global cell_id assigned once in the parent
   │
   ├─ M1  m1_overlay.py ── UNet++ (EfficientNet-B4) core mask → applied to IHC & DISH →
   │                       alpha blend (overlay_alpha). Empty mask short-circuits to empty CSV
   │
   ├─ M2  m2_segmentation.py ── Cellpose on the fused overlay → cell instance mask
   │                            (interior seams left to M0, remove_border=False)
   │
   ├─ M3  m3_module/ ── cell centroids → elastic cell-centered DISH nucleus matching
   │                    (reach = sqrt(factor×area/π), nearest-first with one-to-one locking) →
   │                    HER2 (black) / CEP17 (red) dot detection on a local LAB patch
   │
   └─ M4  m4_export.py ── report.csv, summary.txt, and the rendered overlay arrays;
                          m0_stitch joins _stitch_scratch/ into overlay_slide.tiff
```

`_stitch_scratch/` is the one deliberate disk round-trip — pyvips reads it lazily (`access="sequential"`). Stitching from memory resurrects a ~400 GB full-canvas OOM.

**Scoring (M3):** per-cell `score = HER2/CEP17`, amplified if `score ≥ dot_amplification_ratio` (2.0). Cells are excluded (X) on drop-out, boundary contamination, or `CEP17 < score_cep17_min_count` (2) — except 0/0, which counts normally. `summary.txt` adds a case-level ASCO/CAP 2013 verdict.

**Preprocessing chain:** `CZI → module1 (BigTIFF) → module2 (VALIS alignment) → module3 (ROI eval) → module4 (thumbnail) → hybrid pipeline`.

**Invariants:** images between modules are RGB `uint8 (H,W,3)`; core mask is `uint8{0,1} (H,W)`; instance mask is `int32 (H,W)` with background 0 and cells 1..N.

---

## Output

A run leaves exactly three files in `output_dir/`. No per-tile intermediates — masks stay in memory.

| File | Contents |
|---|---|
| `report.csv` | Per cell: `cell_id, centroid_x, centroid_y, reddot` (CEP17), `blackdot` (HER2), `score` |
| `summary.txt` | ASCO/CAP 2013 case-level verdict: ratio `ΣHER2/ΣCEP17`, average HER2 copy number, per-cell distribution buckets |
| `overlay_slide.tiff` | Pyramidal, QuPath-openable slide overlay: cell boundaries, DISH nucleus contours, drift arrows, labels, dot markers |

`_precut_scratch/` is kept for inspection and never auto-deleted. `_stitch_scratch/` is deleted only on a successful stitch, so a failed stitch can be re-run without recomputing the batch.

---

## Hardware

|  | Minimum (small ROI via API) | Full-slide batch |
|---|---|---|
| CPU | 8 cores | 16+ cores |
| RAM | 32 GB | 64 GB (measured ~60 GB peak RSS on a 27k-tile slide) |
| GPU | 18 GB VRAM | 32 GB (`workers=4` peaked ~30.4 GB) |
| Disk | 100 GB SSD | ~350 GB output, NVMe |

A full-slide run also wants `RLIMIT_NOFILE ≥ ~28,000`; the pipeline raises the soft limit itself when the hard limit allows.

---

## Development

```bash
uv sync                      # backend

cd frontend
npm install
npm run dev                  # Vite dev server
npm run build                # tsc -b && vite build
npm run lint                 # oxlint
```

Frontend setup detail: [`docs/UI/06-dev-setup.md`](docs/UI/06-dev-setup.md).

Dependencies are managed by uv only — `uv add` / `uv sync`, never `pip`. Docker installs via `uv sync --frozen --no-dev --no-install-project` against `uv.lock`. VALIS comes from a git fork and `dinov3` from a pinned commit (`tool.uv.sources`), so plain `pip install .` will not reproduce the environment.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `CUDA out of memory` | Lower `--workers`, or reduce `window_overlap_px` / `unet_image_size` |
| `OpenSlide not found` | `sudo apt-get install libopenslide0 libopenslide-dev` (or `brew install openslide`) |
| `Model not found` | Check `unet_model_path` / `cellpose_model_path` in `config.py` |
| `找不到 tile` | Tile filenames must match `tile_x{int}_y{int}` |
| `config not found` | `cp config_example.py config.py` in `backend/algorithms/hybrid/` |

`python scripts/cuda_test.py` reports CUDA/GPU availability.

---

## Built on

[Cellpose](https://www.cellpose.org/) · [Segmentation Models PyTorch (UNet++)](https://github.com/qubvel/segmentation_models.pytorch) · [VALIS](https://github.com/MannLabs/valis) · [OpenSlide](https://openslide.org/)

**Version:** 0.1.0
