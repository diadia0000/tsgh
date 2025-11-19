# GPU Warp Engine - Technical Overview & Visual Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT STAGE                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────┐      ┌──────────────────┐                     │
│  │  CZI Files      │      │  Valis Registrar │                     │
│  │                 │      │  (Pickle)        │                     │
│  │ • HER2_40X.czi  │      │                  │                     │
│  │ • DISH_40X.czi  │      │ • M matrix       │                     │
│  │ • HE_40X.czi    │      │ • DVF (dx, dy)   │                     │
│  └─────────────────┘      └──────────────────┘                     │
│           │                         │                                │
└───────────┼─────────────────────────┼────────────────────────────────┘
            │                         │
            └─────────┬───────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GPU WARP ENGINE CORE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Step 1: Target Grid Generation                              │  │
│  │  ──────────────────────────────────────────────────────────  │  │
│  │                                                               │  │
│  │  (tile_x, tile_y) ──> [grid_x, grid_y] meshgrid (GPU)       │  │
│  │                                                               │  │
│  │  Output: Target coordinates (H, W)                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           │                                           │
│                           ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Step 2: Inverse Rigid Transform                             │  │
│  │  ──────────────────────────────────────────────────────────  │  │
│  │                                                               │  │
│  │  [x, y, 1]^T  ──>  M^(-1) @ [x, y, 1]^T  ──>  [x', y']      │  │
│  │                                                               │  │
│  │  Output: Rigid-transformed coordinates                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           │                                           │
│                           ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Step 3: Inverse Non-Rigid Transform                         │  │
│  │  ──────────────────────────────────────────────────────────  │  │
│  │                                                               │  │
│  │  Sample DVF at (x', y') ──> [dx, dy]                        │  │
│  │  x_src = x' + dx                                             │  │
│  │  y_src = y' + dy                                             │  │
│  │                                                               │  │
│  │  Output: Final source coordinates                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           │                                           │
│                           ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Step 4: Dynamic Source Cropping                             │  │
│  │  ──────────────────────────────────────────────────────────  │  │
│  │                                                               │  │
│  │  bbox = (x_min, y_min, x_max, y_max)                        │  │
│  │  source_crop = czi_reader.read_mosaic(bbox)                 │  │
│  │                                                               │  │
│  │  Output: Cropped source region (GPU tensor)                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           │                                           │
│                           ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Step 5: Pixel Sampling (grid_sample)                        │  │
│  │  ──────────────────────────────────────────────────────────  │  │
│  │                                                               │  │
│  │  Normalize (x_src, y_src) to [-1, 1]                        │  │
│  │  warped = F.grid_sample(source_crop, grid)                  │  │
│  │                                                               │  │
│  │  Output: Warped tile (GPU tensor)                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           │                                           │
└───────────────────────────┼───────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT STAGE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────┐      ┌──────────────────┐                    │
│  │  Individual      │      │  Merged RGB      │                    │
│  │  Warped TIFFs    │      │  Output          │                    │
│  │                  │      │                  │                    │
│  │ • dish_lv2.tiff  │      │ Merged_lv2.tiff  │                    │
│  │ • her2_lv2.tiff  │      │                  │                    │
│  └──────────────────┘      └──────────────────┘                    │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## Coordinate Transformation Flow

```
Target Space (Aligned)                Source Space (CZI Native)
┌─────────────────────┐              ┌─────────────────────┐
│                     │              │                     │
│  ┌──────────┐       │    Rigid    │       ┌──────┐      │
│  │          │       │   Transform │       │      │      │
│  │  Tile    │──────────────────────────────>│      │      │
│  │  (x,y)   │       │              │       │      │      │
│  │          │       │              │       └──────┘      │
│  └──────────┘       │              │                     │
│                     │   Non-Rigid  │                     │
│                     │   Transform  │                     │
│                     │   (DVF)      │                     │
│                     │──────────────────────>              │
│                     │              │                     │
└─────────────────────┘              └─────────────────────┘
     GPU Grid                           CZI Region
   Generation                          Smart Crop
```

## Memory Management Strategy

```
Traditional Approach (valis built-in):
┌────────────────────────────────────────────────┐
│  Load Entire Level 0 Image to Memory          │ ❌ OOM Error!
│  (~50,000 x 40,000 x 3 = 6 GB)                │
└────────────────────────────────────────────────┘

GPU Warp Engine Approach:
┌────────────────────────────────────────────────┐
│  Tile 1: (0, 0)      ──> Process ──> Save     │ ✅ ~200 MB
├────────────────────────────────────────────────┤
│  Tile 2: (2048, 0)   ──> Process ──> Save     │ ✅ ~200 MB
├────────────────────────────────────────────────┤
│  Tile 3: (4096, 0)   ──> Process ──> Save     │ ✅ ~200 MB
├────────────────────────────────────────────────┤
│  ...                                            │
└────────────────────────────────────────────────┘
   Total GPU Memory: < 1 GB at any time
```

## Processing Pipeline Comparison

### Before: valis Built-in Warping

```
┌──────────────┐
│ Load Full    │ ──> 5-10 minutes (I/O bound)
│ Resolution   │
└──────────────┘
       │
       ▼
┌──────────────┐
│ Warp Entire  │ ──> OOM Error or 30+ minutes
│ Image        │
└──────────────┘
       │
       ▼
┌──────────────┐
│ Save Output  │ ──> 2-5 minutes
└──────────────┘

Total: 40+ minutes or FAILURE
```

### After: GPU Warp Engine

```
┌──────────────┐
│ Initialize   │ ──> 2-3 seconds
│ Engine       │
└──────────────┘
       │
       ▼
┌──────────────┐
│ Process      │ ──> ~1 second per tile
│ Tiles        │     (GPU accelerated)
│ (n=1000)     │
└──────────────┘
       │
       ▼
┌──────────────┐
│ Assemble &   │ ──> 1-2 minutes
│ Save         │
└──────────────┘

Total: ~20 minutes, NO OOM
```

## Data Flow: Single Tile Example

```
Input: tile_x=0, tile_y=0, tile_size=2048, level=2

Step 1: Target Grid
┌────────────────┐
│ 0   1   2  ... │  Shape: (2048, 2048)
│ 1   2   3  ... │  Device: CUDA
│ 2   3   4  ... │  Dtype: float32
│ ...            │
└────────────────┘
      grid_x, grid_y

Step 2: Rigid Transform (matrix multiplication)
┌────────────────┐
│ Inv Matrix     │   ┌──────────┐
│ [m11 m12 m13]  │ @ │ grid_x   │ = x_rigid
│ [m21 m22 m23]  │   │ grid_y   │   y_rigid
│ [  0   0   1]  │   │    1     │
└────────────────┘   └──────────┘

Step 3: Non-Rigid (grid_sample on DVF)
DVF Shape: (1, 2, H_dvf, W_dvf)
Sample at (x_rigid, y_rigid) ──> (dx, dy)
x_src = x_rigid + dx
y_src = y_rigid + dy

Step 4: Smart Crop
x_min = 1024, x_max = 3072  ──> Read region (1024, 768, 2048, 2048)
y_min = 768,  y_max = 2816       from CZI (not full image!)

Step 5: Grid Sample
Normalize (x_src, y_src) to crop region ──> grid_normalized
F.grid_sample(crop, grid_normalized) ──> warped_tile

Output: (2048, 2048, 3) uint8
```

## Performance Optimization Techniques

### 1. Batched Coordinate Operations
```python
# ✅ Good: Vectorized GPU operations
coords = torch.stack([grid_x, grid_y, ones], dim=-1)
transformed = torch.matmul(coords, matrix.T)  # Single GPU call

# ❌ Bad: Loop over pixels
for i in range(H):
    for j in range(W):
        transformed[i,j] = matrix @ coords[i,j]  # Millions of calls!
```

### 2. Smart I/O (Dynamic Cropping)
```python
# ✅ Good: Read only what's needed
bbox = (x_min, y_min, width, height)
crop = czi_reader.read_mosaic(region=bbox)  # ~2048x2048 region

# ❌ Bad: Read entire image
full_img = czi_reader.read_mosaic()  # ~50000x40000 pixels!
```

### 3. Memory Reuse
```python
# ✅ Good: Reuse GPU tensors
for tile in tiles:
    warped = engine.process_tile(...)  # Reuses GPU memory
    save_tile(warped)
    torch.cuda.empty_cache()  # Explicit cleanup

# ❌ Bad: Accumulate in memory
all_tiles = []
for tile in tiles:
    all_tiles.append(process(tile))  # Accumulates!
```

## Tile Grid Calculation

For an image of size (H, W) at level L with tile size T:

```
Target dimensions:
  H_target = H / (2^L)
  W_target = W / (2^L)

Number of tiles:
  n_tiles_x = ceil(W_target / T)
  n_tiles_y = ceil(H_target / T)
  total_tiles = n_tiles_x × n_tiles_y

Example: 70829×57207 image, Level 2, tile_size=2048
  H_target = 57207 / 4 = 14,302
  W_target = 70829 / 4 = 17,707
  n_tiles_x = ceil(17707 / 2048) = 9
  n_tiles_y = ceil(14302 / 2048) = 7
  total_tiles = 9 × 7 = 63 tiles

Processing time (assuming 1.5s per tile):
  Total = 63 × 1.5s = 94.5 seconds (~1.6 minutes)
```

## Grid Sample Coordinate Normalization

Critical concept: `torch.nn.functional.grid_sample` expects normalized coordinates.

```
Image Space          grid_sample Space
(0, 0)               (-1, -1)
   ┌─────────────>      ┌─────────────>
   │                    │
   │  Image             │     [-1, 1]
   │  (H, W)            │     range
   │                    │
   ▼                    ▼
(W-1, H-1)           (1, 1)

Normalization formula:
  x_norm = 2.0 * x / (W - 1) - 1.0
  y_norm = 2.0 * y / (H - 1) - 1.0

Example: x=512, W=2048
  x_norm = 2.0 * 512 / 2047 - 1.0 = -0.4998
```

## Error Handling Flow

```
┌─────────────────────────────────────────────┐
│         Try Process Tile                    │
└───────────┬─────────────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │ Check Bounds  │ ──No──> Return Black Tile
    └───────┬───────┘
            │ Yes
            ▼
    ┌───────────────┐
    │ Read CZI      │ ──Error──> Log + Return Black
    └───────┬───────┘
            │ OK
            ▼
    ┌───────────────┐
    │ GPU Transform │ ──OOM──> Reduce Tile Size
    └───────┬───────┘          + Retry
            │ OK
            ▼
    ┌───────────────┐
    │ Return Warped │
    │ Tile          │
    └───────────────┘
```

## GPU Memory Layout

```
GPU Memory (8 GB example):
┌────────────────────────────────────────────┐
│  PyTorch CUDA Context        (~500 MB)     │
├────────────────────────────────────────────┤
│  DVF Tensor                  (~100 MB)     │
├────────────────────────────────────────────┤
│  Rigid Matrix                (~1 KB)       │
├────────────────────────────────────────────┤
│  Working Memory (per tile):                │
│    - Target grid             (~32 MB)      │
│    - Source coordinates      (~64 MB)      │
│    - Source crop             (~50 MB)      │
│    - Warped output           (~25 MB)      │
│  Total per tile:             (~171 MB)     │
├────────────────────────────────────────────┤
│  Free                        (~7.2 GB)     │
└────────────────────────────────────────────┘

Conclusion: Can easily handle 2048×2048 tiles
           Could even process 4096×4096 if needed
```

## Multi-Slide Processing Strategy

```
Slide 1 (DISH_40X_2.czi)
    │
    ├──> Initialize Engine ──> Warp ──> Save ──> Clear GPU
    │
    ▼
Slide 2 (HER2_40X.czi)
    │
    ├──> Initialize Engine ──> Warp ──> Save ──> Clear GPU
    │
    ▼
Slide 3 (HE_40X.czi)
    │
    ├──> Initialize Engine ──> Warp ──> Save ──> Clear GPU
    │
    ▼
Merge All ──> RGB Output

Key: Process slides sequentially to avoid GPU memory accumulation
```

## Summary of Key Innovations

1. **Inverse Mapping on GPU**: Manual implementation using PyTorch operations
2. **Dynamic Source Cropping**: Read only required CZI regions
3. **Tile-Based Processing**: Avoid loading full gigapixel images
4. **Memory Efficiency**: Reuse GPU tensors across tiles
5. **Smart Coordinate Handling**: Proper normalization for grid_sample
6. **Error Resilience**: Graceful handling of out-of-bounds regions

This design enables processing of arbitrarily large images limited only by storage, not RAM/VRAM.

