# GPU Warp Engine - Installation & Verification Checklist

## ✅ Pre-Flight Checklist

Complete this checklist before running the GPU Warp Engine for the first time.

---

## 1️⃣ System Requirements

### Hardware
- [ ] GPU with CUDA support (NVIDIA GTX 1060 or better)
- [ ] At least 8 GB GPU VRAM (16 GB recommended)
- [ ] At least 32 GB system RAM (64 GB recommended)
- [ ] Fast SSD storage (500+ MB/s read/write)

### Software
- [ ] Windows 10/11 or Linux
- [ ] Python 3.8+ installed
- [ ] CUDA Toolkit 11.8+ installed
- [ ] NVIDIA drivers up to date

---

## 2️⃣ Python Environment

### Check Python Version
```powershell
python --version
# Expected: Python 3.8.x or higher
```
- [ ] Python version >= 3.8

### Check Virtual Environment (Optional but Recommended)
```powershell
# Create virtual environment
python -m venv venv_gpu_warp

# Activate
.\venv_gpu_warp\Scripts\Activate.ps1

# Verify
(venv_gpu_warp) PS>
```
- [ ] Virtual environment created and activated

---

## 3️⃣ Dependencies

### Install Required Packages
```powershell
cd H:\tsgh
pip install -r requirements.txt
```
- [ ] All packages installed without errors

### Critical Packages Checklist
```powershell
# Check PyTorch with CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Expected: 2.0.0 or higher

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Expected: True

python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Expected: Your GPU name (e.g., NVIDIA RTX 3090)
```
- [ ] PyTorch installed
- [ ] CUDA available in PyTorch
- [ ] GPU detected

```powershell
# Check other critical packages
python -c "import aicspylibczi; print('aicspylibczi OK')"
python -c "import pyvips; print('pyvips OK')"
python -c "import valis; print('valis OK')"
python -c "import tifffile; print('tifffile OK')"
```
- [ ] aicspylibczi installed
- [ ] pyvips installed
- [ ] valis installed
- [ ] tifffile installed

---

## 4️⃣ File Structure Verification

### Check Files Exist
```powershell
# Navigate to project directory
cd H:\tsgh\thriple_image_layer

# List key files
ls gpu_warp_engine.py
ls module5.py
ls test_gpu_warp_engine.py
ls quickstart_gpu_warp.py
```
- [ ] gpu_warp_engine.py exists
- [ ] module5.py exists
- [ ] test_gpu_warp_engine.py exists
- [ ] quickstart_gpu_warp.py exists

### Check Input Data
```powershell
# Check CZI files
ls H:\tsgh\picture\whole_size\40X\*.czi

# Check registrar (if alignment already done)
ls H:\tsgh\thriple_image_layer\output\Transform_Params\data\Transform_Params_registrar.pickle
```
- [ ] CZI files present (HER2_40X.czi, DISH_40X_2.czi, etc.)
- [ ] Registrar pickle exists (or ready to run Module 2 alignment)

---

## 5️⃣ Import Test

### Test Imports
```powershell
cd H:\tsgh\thriple_image_layer
python -c "from gpu_warp_engine import GPUWarpEngine; print('✅ GPUWarpEngine imported successfully')"
```
- [ ] GPUWarpEngine imports without errors

```powershell
python -c "from module5 import warp_and_merge_slides; print('✅ Module5 imported successfully')"
```
- [ ] Module5 imports without errors

---

## 6️⃣ GPU Health Check

### Monitor GPU Status
```powershell
# Open new PowerShell window and run:
nvidia-smi -l 1
```
- [ ] nvidia-smi shows GPU info
- [ ] GPU utilization visible
- [ ] GPU memory info visible

### Check GPU Memory
```powershell
python -c "import torch; print(f'Free GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')"
```
- [ ] At least 6 GB free GPU memory

---

## 7️⃣ Prerequisites: Run Module 2 (Alignment)

If you haven't run the alignment yet:

```powershell
cd H:\tsgh\thriple_image_layer
python run_full_pipeline.py
```

**OR** just run Module 2:

```powershell
python -c "from module2_alignment import align_images; from pathlib import Path; align_images(Path('H:/tsgh/picture/whole_size/40X'), Path('H:/tsgh/thriple_image_layer/output'))"
```

- [ ] Module 2 completed successfully
- [ ] Registrar pickle file created
- [ ] No error messages in output

---

## 8️⃣ Quick Test: Single Tile

### Test Single Tile Processing

```powershell
cd H:\tsgh\thriple_image_layer

python -c "
from pathlib import Path
from gpu_warp_engine import GPUWarpEngine

registrar_path = Path('output/Transform_Params/data/Transform_Params_registrar.pickle')
engine = GPUWarpEngine(registrar_path, 'DISH_40X_2.czi', device='cuda')
tile = engine.process_tile(level=2, tile_x=0, tile_y=0, tile_w=512, tile_h=512)
print(f'✅ Tile shape: {tile.shape}, dtype: {tile.dtype}')
"
```

**Expected output:**
```
Initializing GPUWarpEngine on device: cuda
✓ Loaded registrar from ...
✓ Loaded non-rigid DVF: (H, W)
✓ Engine initialized for 'DISH_40X_2.czi'
✅ Tile shape: (512, 512, 3), dtype: uint8
```

- [ ] Single tile processed successfully
- [ ] No errors or warnings
- [ ] Output shape is (512, 512, 3)

---

## 9️⃣ Quick Start Test

### Run Quick Start Script

```powershell
cd H:\tsgh\thriple_image_layer
python quickstart_gpu_warp.py
```

**Watch for:**
- GPU initialization message
- Processing progress bar
- Success message with output path

**Expected duration:** 2-5 minutes for Level 2

- [ ] Quick start completed successfully
- [ ] Output TIFF file created
- [ ] File size > 0 MB
- [ ] No error messages

---

## 🔟 Full Test Suite

### Run Comprehensive Tests

```powershell
cd H:\tsgh\thriple_image_layer
python test_gpu_warp_engine.py
```

**Expected results:**
- ✅ Test 1: Single Tile Processing - PASS
- ✅ Test 2: Grid Warping - PASS
- ✅ Test 3: Full Slide Warping - PASS
- ✅ Test 4: Performance Benchmark - PASS
- ✅ Test 5: Memory Usage - PASS

**Expected duration:** 5-10 minutes

- [ ] All 5 tests passed
- [ ] Benchmark shows reasonable performance (~1-2s per tile)
- [ ] GPU memory usage < 2 GB

---

## 🎯 Final Verification

### Check Output Files

```powershell
ls H:\tsgh\thriple_image_layer\output\*.tiff
ls H:\tsgh\thriple_image_layer\output\test_results\*.tiff
```

- [ ] Warped TIFF files created
- [ ] File sizes look reasonable (not 0 bytes)
- [ ] Can open files in image viewer (optional)

### Verify Performance

Expected performance (NVIDIA RTX 3090 or similar):
- Single 2048×2048 tile: ~1-2 seconds
- Full slide at Level 2: ~3-5 minutes
- GPU utilization during processing: 80-95%

- [ ] Processing speed is reasonable
- [ ] GPU utilization is high during processing
- [ ] No performance warnings

---

## 🚨 Troubleshooting

### If CUDA Not Available

1. **Check NVIDIA driver:**
   ```powershell
   nvidia-smi
   ```
   - Update drivers if needed: https://www.nvidia.com/drivers

2. **Reinstall PyTorch with CUDA:**
   ```powershell
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify CUDA Toolkit:**
   ```powershell
   nvcc --version
   ```
   - Install CUDA Toolkit if missing: https://developer.nvidia.com/cuda-downloads

### If Out of Memory (OOM)

1. **Reduce tile size in quickstart_gpu_warp.py:**
   ```python
   tile_size = 1024  # Instead of 2048
   ```

2. **Use lower pyramid level:**
   ```python
   level = 3  # Instead of 2
   ```

3. **Clear GPU cache:**
   ```powershell
   python -c "import torch; torch.cuda.empty_cache()"
   ```

### If Import Errors

1. **Check requirements.txt is up to date:**
   ```powershell
   pip install -r requirements.txt --upgrade
   ```

2. **Check Python path:**
   ```powershell
   python -c "import sys; print('\n'.join(sys.path))"
   ```

3. **Reinstall specific package:**
   ```powershell
   pip install --force-reinstall aicspylibczi
   ```

### If Processing is Slow

1. **Verify GPU usage:**
   ```powershell
   nvidia-smi -l 1
   # Check GPU-Util column should be ~90% during processing
   ```

2. **Check if running on CPU:**
   - Look for "device: cuda" in console output
   - If says "device: cpu", check CUDA installation

3. **Close other GPU applications:**
   - Close games, video editors, or ML training jobs

---

## 📝 Success Criteria

### ✅ All Checks Passed!

If all items are checked, you're ready to process your gigapixel images!

**Next steps:**
1. Run `python quickstart_gpu_warp.py` for single slide
2. Use `module5.py` for batch processing
3. Integrate into your pipeline

### ⚠️ Some Checks Failed

**Don't proceed until all checks pass!**

Common issues:
- CUDA not available → Install/update GPU drivers and CUDA toolkit
- Import errors → Reinstall dependencies with `pip install -r requirements.txt`
- Missing registrar → Run Module 2 alignment first
- OOM errors → Reduce tile size or use lower pyramid level

---

## 📞 Need Help?

1. **Check error logs:** Read error messages carefully
2. **Review documentation:**
   - `GPU_WARP_ENGINE_README.md` - User guide
   - `GPU_WARP_ENGINE_ARCHITECTURE.md` - Technical details
3. **Run verbose mode:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

---

## 📊 Checklist Summary

Count your ✅ marks:

- [ ] **System Requirements** (4 items)
- [ ] **Python Environment** (2 items)
- [ ] **Dependencies** (6 items)
- [ ] **File Structure** (6 items)
- [ ] **Import Test** (2 items)
- [ ] **GPU Health Check** (4 items)
- [ ] **Prerequisites** (3 items)
- [ ] **Quick Test** (3 items)
- [ ] **Quick Start Test** (4 items)
- [ ] **Full Test Suite** (3 items)
- [ ] **Final Verification** (3 items)

**Total: 40 items**

### Status:
- 40/40 ✅ → **READY TO GO!** 🚀
- 35-39 ✅ → **Almost there**, fix remaining issues
- 30-34 ✅ → **Getting close**, check documentation
- <30 ✅ → **Need more setup**, review installation steps

---

**Date Completed:** _______________

**Verified By:** _______________

**GPU Model:** _______________

**Notes:** _______________________________________________

