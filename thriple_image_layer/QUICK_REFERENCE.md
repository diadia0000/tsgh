# GPU Warp Engine - Quick Reference Card

## 🚀 One-Liner Commands

### Quick Start (First Time Users)
```bash
python thriple_image_layer/quickstart_gpu_warp.py
```

### Run Full Test Suite
```bash
python thriple_image_layer/test_gpu_warp_engine.py
```

### Check CUDA Status
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Monitor GPU
```bash
nvidia-smi -l 1
```

---

## 📖 File Quick Index

| File | Purpose | When to Use |
|------|---------|-------------|
| `quickstart_gpu_warp.py` | Simple example | First time, testing |
| `module5.py` | Batch processing | Production use |
| `gpu_warp_engine.py` | Core engine | Advanced/custom |
| `test_gpu_warp_engine.py` | Testing | Validation |
| `INSTALLATION_CHECKLIST.md` | Setup guide | Before first run |
| `GPU_WARP_ENGINE_README.md` | User manual | Reference |
| `GPU_WARP_ENGINE_ARCHITECTURE.md` | Tech details | Understanding |
| `IMPLEMENTATION_SUMMARY.md` | Overview | Big picture |
| `DELIVERABLES.md` | File list | Navigation |

---

## 🎯 Common Tasks

### Task: Warp Single Slide at Level 2
```python
from pathlib import Path
from gpu_warp_engine import GPUWarpEngine

engine = GPUWarpEngine(
    registrar_path=Path("output/Transform_Params/data/registrar.pickle"),
    slide_name="DISH_40X_2.czi"
)
engine.warp_full_slide(
    output_path=Path("output/dish_warped_lv2.tiff"),
    level=2,
    tile_size=2048
)
```

### Task: Warp Multiple Slides and Merge
```python
from pathlib import Path
from module5 import warp_and_merge_slides

warp_and_merge_slides(
    registrar_path=Path("output/Transform_Params/data/registrar.pickle"),
    output_dir=Path("output"),
    slides_to_warp=["HER2_40X.czi", "DISH_40X_2.czi"],
    level=2,
    merge=True
)
```

### Task: Process Custom Tile Region
```python
from gpu_warp_engine import GPUWarpEngine

engine = GPUWarpEngine(...)
tile = engine.process_tile(
    level=2,
    tile_x=1024,
    tile_y=512,
    tile_w=2048,
    tile_h=2048
)
# tile is numpy array (2048, 2048, 3)
```

---

## ⚙️ Configuration Parameters

### Pyramid Levels
```python
level=0  # Full resolution (slowest, highest quality)
level=1  # Half resolution
level=2  # Quarter resolution (recommended for testing)
level=3  # 1/8 resolution (fast preview)
```

### Tile Sizes
```python
tile_size=512   # Small (fast, lower GPU utilization)
tile_size=1024  # Medium
tile_size=2048  # Recommended (best GPU utilization)
tile_size=4096  # Large (requires 8+ GB GPU memory)
```

### Compression Options
```python
compression='jpeg'     # Best compression (recommended)
compression='lzw'      # Lossless compression
compression='deflate'  # Alternative lossless
compression=None       # No compression (huge files)

quality=90  # JPEG quality (50-100)
```

---

## 🐛 Troubleshooting Quick Fixes

### Issue: CUDA not available
```bash
# Check driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of Memory
```python
# Option 1: Reduce tile size
tile_size=1024  # Instead of 2048

# Option 2: Lower pyramid level
level=3  # Instead of 2

# Option 3: Clear cache
import torch
torch.cuda.empty_cache()
```

### Issue: Slow processing
```bash
# Check GPU usage (should be 80-95%)
nvidia-smi -l 1

# If low, ensure CUDA is being used
python -c "from gpu_warp_engine import GPUWarpEngine; print('Check output for device: cuda')"
```

### Issue: Import error
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check specific package
pip show aicspylibczi torch pyvips valis
```

---

## 📊 Performance Reference

### Expected Processing Times (RTX 3090)

| Level | Resolution | Tile Size | Time per Tile | Full Slide (60 tiles) |
|-------|------------|-----------|---------------|----------------------|
| 0 | Full | 2048 | ~5s | ~5 min |
| 1 | 1/2 | 2048 | ~2s | ~2 min |
| 2 | 1/4 | 2048 | ~1s | ~1 min |
| 3 | 1/8 | 2048 | ~0.5s | ~30 sec |

### GPU Memory Usage

| Tile Size | GPU Memory | Recommended GPU |
|-----------|------------|-----------------|
| 512×512 | ~100 MB | 4 GB+ |
| 1024×1024 | ~200 MB | 4 GB+ |
| 2048×2048 | ~400 MB | 6 GB+ |
| 4096×4096 | ~1.5 GB | 8 GB+ |

---

## 🔍 Validation Checklist

Before running on production data:

- [ ] CUDA available: `torch.cuda.is_available() == True`
- [ ] Registrar exists: `registrar.pickle` file present
- [ ] CZI files accessible: Check paths
- [ ] Output directory writable: Test write permissions
- [ ] Sufficient GPU memory: Check `nvidia-smi`
- [ ] Test passed: Run `test_gpu_warp_engine.py`

---

## 📞 Quick Help

### Error Messages & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Tile too large | Reduce `tile_size` |
| `No module named 'torch'` | PyTorch not installed | `pip install torch` |
| `Registrar not found` | Wrong path | Check `registrar_path` |
| `Slide not in registrar` | Wrong name | Check `slide_name` |
| `CziFile not found` | Wrong CZI path | Check CZI file location |

### Where to Look for Help

1. **Quick issues**: This card
2. **Setup problems**: `INSTALLATION_CHECKLIST.md`
3. **Usage questions**: `GPU_WARP_ENGINE_README.md`
4. **Technical details**: `GPU_WARP_ENGINE_ARCHITECTURE.md`
5. **Code errors**: Check inline comments in `gpu_warp_engine.py`

---

## 💡 Pro Tips

### Tip 1: Start Small
Always test with Level 2 or 3 before processing Level 0.

### Tip 2: Monitor GPU
Keep `nvidia-smi -l 1` running in a separate terminal.

### Tip 3: Save Memory
Clear GPU cache between slides:
```python
import torch
torch.cuda.empty_cache()
```

### Tip 4: Batch Process
Use Module 5 for multiple slides instead of running engine multiple times.

### Tip 5: Check Output
Always verify output file size > 0 and can be opened.

---

## 🎓 Learning Path

### Beginner (Day 1)
1. Read `INSTALLATION_CHECKLIST.md`
2. Run `quickstart_gpu_warp.py`
3. View output in image viewer

### Intermediate (Day 2)
1. Read `GPU_WARP_ENGINE_README.md`
2. Modify `quickstart_gpu_warp.py` parameters
3. Run `test_gpu_warp_engine.py`

### Advanced (Week 1)
1. Read `GPU_WARP_ENGINE_ARCHITECTURE.md`
2. Study `gpu_warp_engine.py` source
3. Implement custom processing with `process_tile()`

---

## 📈 Optimization Guide

### For Speed
- ✅ Use larger tile sizes (2048-4096)
- ✅ Process at lower pyramid levels first
- ✅ Ensure GPU utilization > 80%
- ✅ Use JPEG compression for faster I/O

### For Quality
- ✅ Process at Level 0 (full resolution)
- ✅ Use smaller tiles (1024) for precision
- ✅ Use lossless compression (LZW)
- ✅ Enable non-rigid transformation

### For Memory
- ✅ Use smaller tile sizes (512-1024)
- ✅ Process at higher pyramid levels (2-3)
- ✅ Clear cache between slides
- ✅ Close other GPU applications

---

## 🔗 Quick Links

| Resource | Location |
|----------|----------|
| Core engine code | `gpu_warp_engine.py` |
| High-level API | `module5.py` |
| Quick start script | `quickstart_gpu_warp.py` |
| Test suite | `test_gpu_warp_engine.py` |
| User guide | `GPU_WARP_ENGINE_README.md` |
| Technical docs | `GPU_WARP_ENGINE_ARCHITECTURE.md` |
| Setup checklist | `INSTALLATION_CHECKLIST.md` |
| Project summary | `IMPLEMENTATION_SUMMARY.md` |
| File list | `DELIVERABLES.md` |
| This card | `QUICK_REFERENCE.md` |

---

## 🎯 Success Indicators

Your setup is working correctly if:
- ✅ `torch.cuda.is_available()` returns `True`
- ✅ Quick start completes without errors
- ✅ Output TIFF files are created
- ✅ File sizes look reasonable (not 0 KB)
- ✅ GPU usage shows ~90% during processing
- ✅ Processing time matches benchmarks above
- ✅ All 5 tests pass

---

## 📝 Command Cheat Sheet

```bash
# Setup
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Quick start
python thriple_image_layer/quickstart_gpu_warp.py

# Full test
python thriple_image_layer/test_gpu_warp_engine.py

# Production run
python thriple_image_layer/module5.py

# Monitor GPU
nvidia-smi -l 1

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Check versions
python --version
pip show torch
pip show aicspylibczi
```

---

**Print this card and keep it handy!** 🖨️

**Last Updated**: 2025-05-19 | **Version**: 1.0.0

