# TSGH — Whole Slide Image Analysis Pipeline

Automated HER2/CEP17 amplification analysis for histopathology. Precuts an IHC (HER2) + DISH whole-slide-image pair into overlapping tiles and runs an M0→M4 chain to detect cells, count signal dots, and classify amplification. Ships a CLI, a FastAPI backend (`backend/api/`), and a React/OpenSeadragon viewer (`frontend/`).

- **Python:** 3.11 only (`>=3.11,<3.12`)
- **GPU:** NVIDIA CUDA. Reference machine is an RTX 5090 (Blackwell, `sm_120`), which requires the `cu130` PyTorch build — `torch==2.11.0+cu130` is pinned in `pyproject.toml`. Do not downgrade.

---

## Quick Start

### Open the UI (Linux / WSL2)

```bash
./run-ui.sh
```

One command, first run included: it installs dependencies, creates `config.py`
from the example if missing, points `TSGH_STORAGE_DIR` / `TSGH_SLIDES_DIR` at
`~/tsgh_data/` (outside the checkout, deliberately), starts backend + frontend,
waits for the backend to finish loading, and opens <http://localhost:5173>.
Ctrl-C stops both. `--skip-deps` skips installation on later runs;
`BACKEND_PORT` / `FRONTEND_PORT` override the ports.

Analysis needs an NVIDIA GPU and the model files named in `config.py`; without
them the UI still opens and slides are still viewable.

### CLI pipeline

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

python hybrid_pipeline.py --test                            # bundled test_picture ROI pair
python hybrid_pipeline.py --ihc a.tiff --dish b.tiff        # any size
python hybrid_pipeline.py --ihc a.tiff --dish b.tiff --output /path/to/out
python hybrid_pipeline.py --ihc a.tiff --dish b.tiff --workers 4 --resume   # unattended full slide
```

| Flag | Meaning |
|---|---|
| `--test` | Run the bundled `test_picture/` ROI pair through the full precut+analysis path |
| `--ihc` / `--dish` | Input pair (tile, ROI, or WSI) |
| `--output` | Output directory (default `config.output_dir`) |
| `--workers N` | Cross-tile parallelism: N `spawn`-ed processes, each with its own models and CUDA context. Default 1 |
| `--resume` | Checkpoint each finished tile to `output/_resume/`; a re-run skips completed tiles |

Tiles pair by filename coordinate `tile_x{int}_y{int}`. There is no `--batch` mode — `--ihc`/`--dish` already precuts internally.

`workers=4` is the practical setting for a full slide on a 32 GB card; it needs materially more RAM/VRAM than a small ROI request (a full ~35,700-tile slide measured ~14 GB peak RSS and ~30 GB VRAM at `workers=4`, see Hardware below). `config.cuda_alloc_conf = "expandable_segments:True"` (already the default in `config_example.py`) is required at `workers>1` to avoid intermittent CUDA allocator OOM — even so, a `workers=4` batch can still OOM on rare occasion (~1 in 10 runs measured on a crop 48x smaller than a full slide); combined VRAM headroom across all four workers is tight (~2.5 GB).

For a full-slide run, point `--ihc`/`--dish` at the **registered** pair `module4_thumbnail.py` already produces (`her2_warped_lv0.tiff` / `dish_warped_lv0.tiff`), not the raw `*_processed.tiff` Module 1 output — the two are naturally equal-sized (`crop="overlap"`) and analysis-ready. Feeding the raw pair silently analyzes misaligned tissue (see [`docs/hybrid-pipeline/44-conform-intersection-shift-investigation.md`](docs/hybrid-pipeline/44-conform-intersection-shift-investigation.md)).

### Triple image layer pipeline (preprocessing)

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

`_stitch_scratch/` is the one deliberate disk round-trip — it is read lazily, one horizontal band at a time, never as a whole image. Stitching from memory resurrects a ~400 GB full-canvas OOM.

**Phase D stitch backend (`config.stitch_backend`):** defaults to **`"tifffile"`** as of round 13 — the overlay tiles are streamed band-by-band with the read overlapped onto a background thread, the pyramid built with a CPU box filter, and each 128px container tile LZW-encoded with TIFF Predictor 2. Measured on the real 27,565-tile slide at `workers=4`: **Phase D 1,889.8 → 987.8 s (1.913x)** and **end-to-end 5,854.9 → 4,877.4 s (1.200x)**, above the 1.135x projection and its 1.184x perfect-overlap floor, with the correctness veto passing on all four gates (`report.csv` +0.0014% rows, `overlay_pyramid_audit.py` PASS at every level, every pyramid level bit-exact against a 2×2 box shrink of the level above, and a by-hand QuPath render check). Two side effects worth knowing: **peak RSS dropped 45.6 → 17.0 GB** (band streaming replaces a lazy join holding all 27,565 tiles open, which relaxes the 61–62 GB host requirement round 8 recorded), and the artifact shrinks 7.50 → 5.85 GB because Predictor 2 applies where the old path wrote `predictor="none"`. The previous `"pyvips"` path (one `tiffsave`) is retained as a fallback and as the control arm for future measurement. Full detail: [`docs/hybrid-pipeline/41-round-13-phase-d-pipelined-stitch-implementation.md`](docs/hybrid-pipeline/41-round-13-phase-d-pipelined-stitch-implementation.md).

That round-13 measurement, like every full-slide run through round 14, was taken on an
**unregistered** canvas (see the Hardware section below); round 15 re-confirmed the same mechanism
on the correct, registered ~35,700-tile canvas: Phase D 24.1% of the `workers=4` wall, end-to-end
`workers=1` 3.023 h / `workers=4` 1.478 h. See
[`docs/hybrid-pipeline/46-round-15-eta-estimation-implementation.md`](docs/hybrid-pipeline/46-round-15-eta-estimation-implementation.md).

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
| RAM | 32 GB | measured **~14 GB peak RSS** on the real ~35,700-tile registered slide (round 15) |
| GPU | 18 GB VRAM | 32 GB (`workers=4` peaked ~30.4 GB) |
| Disk | 100 GB SSD | ~350 GB output, NVMe |

A full-slide run also wants `RLIMIT_NOFILE` comfortably above the tile count (~35,700 on the real
registered slide); the pipeline raises the soft limit itself when the hard limit allows.

The RAM figure dropped sharply from an earlier "~60 GB" estimate for two independent reasons: the
round-13 `tifffile` stitch backend (default since then) replaced a lazy join holding every overlay
tile open with band streaming (45.6 → 17.0 GB on its own), and the ~60 GB figure itself was measured
on an **unregistered, 27,565-tile canvas** later found to be the wrong input for a full-slide run —
see [`docs/hybrid-pipeline/44-conform-intersection-shift-investigation.md`](docs/hybrid-pipeline/44-conform-intersection-shift-investigation.md).
The real registered slide is larger (35,700 tiles) but its measured peak RSS is lower still (13.66 GB,
[`docs/hybrid-pipeline/46-round-15-eta-estimation-implementation.md`](docs/hybrid-pipeline/46-round-15-eta-estimation-implementation.md) §3.7).

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
