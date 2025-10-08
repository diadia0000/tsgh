# Phase 0 - 影像前處理 (灰階轉換)

## 功能說明

將 DISH.tiff 和 Her2.tiff 轉換為灰階影像並增強對比度，用於後續對齊。

## 核心技術

1. **灰階轉換** - 使用 OpenCV cvtColor(RGB2GRAY)
2. **自適應 CLAHE** - 根據影像統計特性自動調整增強參數
3. **分塊處理** - 支援 Gigapixel 影像的記憶體優化處理 (2048×2048)
4. **多進程加速** - 使用 n-1 個 CPU 核心並行處理

## 使用方法

```bash
# 安裝依賴
pip install -r requirements.txt

# 執行完整 Phase 0
python run_phase0.py

# 或單獨執行
python preprocess_dish_image.py
python preprocess_her2_image.py
```

## 輸入/輸出

### 輸入
- `../picture/WSI/DISH_20X_ED7.tiff` (5 層金字塔 TIFF)
- `../picture/WSI/HER2_20X_ED7.tiff` (5 層金字塔 TIFF)

### 輸出
- `output/DISH_Gray.tiff` (單通道灰階影像)
- `output/Her2_Gray.tiff` (單通道灰階影像)

## 處理流程

### Phase 0.1 - DISH
1. 從 Level 1 讀取影像
2. RGB 轉灰階
3. 自適應 CLAHE 增強並儲存

### Phase 0.2 - Her2
1. 從 Level 1 讀取影像
2. RGB 轉灰階
3. 自適應 CLAHE 增強並儲存

## 參數說明

- `output_level`: 處理的金字塔層級 (預設 1)
- `tile_size`: 分塊大小 (2048 像素)
- CLAHE `clipLimit`: 自動根據影像標準差計算 (範圍 1.0-4.0)
- CLAHE `tileGridSize`: 根據影像大小自動選擇 (8×8 或 16×16)
- `n_workers`: CPU 核心數 - 1
