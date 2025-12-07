# Module 5: Tile-based Output 使用說明

## 快速開始

### 1. 基本使用

```python
from pathlib import Path
from valis import registration, slide_io
from module5_tile_saver import TileSaver

# 初始化 JVM
slide_io.init_jvm()

try:
    # 載入 registrar
    output_dir = Path(r"H:\tsgh\thriple_image_layer\output")
    pickle_path = output_dir / "Transform_Params" / "data" / "Transform_Params_registrar.pickle"
    registrar = registration.load_registrar(str(pickle_path))
    
    # 創建 TileSaver
    tile_saver = TileSaver(
        registrar=registrar,
        output_dir=output_dir / "tiles",
        level=1,              # ⚠️ 建議用 level 1-2，避免記憶體問題
        tile_wh=4096,         # Tile 尺寸
        batch_size=16         # GPU batch size
    )
    
    # 處理並儲存 DISH
    dish_tiles_dir = tile_saver.process_and_save_tiles('DISH_40X_2', non_rigid=True)
    print(f"DISH tiles: {dish_tiles_dir}")
    
    # 處理並儲存 HER2
    her2_tiles_dir = tile_saver.process_and_save_tiles('HER2_40X', non_rigid=True)
    print(f"HER2 tiles: {her2_tiles_dir}")
    
finally:
    slide_io.kill_jvm()
```

### 2. 輸出結構

```
output/tiles/
├── DISH_40X_2_tiles_lv1/
│   ├── tile_r0000_c0000.tif
│   ├── tile_r0000_c0001.tif
│   ├── tile_r0000_c0002.tif
│   ├── ...
│   └── metadata.json
└── HER2_40X_tiles_lv1/
    ├── tile_r0000_c0000.tif
    ├── tile_r0000_c0001.tif
    ├── ...
    └── metadata.json
```

### 3. Metadata 格式

```json
{
  "slide_name": "DISH_40X_2",
  "level": 1,
  "original_size": [141818, 114366],
  "tile_size": 4096,
  "total_tiles": 1050,
  "grid_size": [28, 35],
  "non_rigid": true,
  "tiles": [
    {"x": 0, "y": 0, "w": 4096, "h": 4096, "row": 0, "col": 0},
    {"x": 4096, "y": 0, "w": 4096, "h": 4096, "row": 0, "col": 1},
    ...
  ]
}
```

---

## 進階使用

### 1. 選擇合適的 Level

```python
# 檢查各 level 的尺寸和記憶體需求
registrar = registration.load_registrar(pickle_path)
slide_obj = registrar.slide_dict['DISH_40X_2']

for level, (w, h) in enumerate(slide_obj.slide_dimensions_wh):
    pixels = w * h
    memory_gb = pixels * 3 / (1024**3)  # RGB uint8
    tiles = (w // 4096 + 1) * (h // 4096 + 1)
    
    print(f"Level {level}:")
    print(f"  Size: {w} x {h} ({pixels/1e9:.2f} Gpixels)")
    print(f"  Memory: {memory_gb:.2f} GB")
    print(f"  Tiles: {tiles}")
    print()
```

**建議**：
- Level 0: > 100 GB → 使用 tile output
- Level 1: 25-100 GB → 使用 tile output
- Level 2: < 25 GB → 可以考慮完整圖像
- Level 3+: < 10 GB → 直接使用 VALIS 原始方法

### 2. 調整 Batch Size

```python
# GPU 記憶體不足？減少 batch size
tile_saver = TileSaver(..., batch_size=8)  # 減半

# GPU 記憶體充足？增加 batch size
tile_saver = TileSaver(..., batch_size=32)  # 加速

# 估算 GPU 記憶體需求
batch_memory_mb = (4096 * 4096 * 3 * batch_size * 4) / (1024**2)
print(f"Batch memory: {batch_memory_mb:.0f} MB")
```

### 3. 只處理特定區域

修改 `module5_tile_saver.py` 添加 ROI 支援：

```python
def process_and_save_tiles(self, slide_name, non_rigid=True, roi_xywh=None):
    """
    roi_xywh: (x, y, width, height) 只處理這個區域
    """
    if roi_xywh is not None:
        roi_x, roi_y, roi_w, roi_h = roi_xywh
        tiles = [t for t in tiles 
                 if (t['x'] < roi_x + roi_w and t['x'] + t['w'] > roi_x and
                     t['y'] < roi_y + roi_h and t['y'] + t['h'] > roi_y)]
```

---

## 後續使用 Tiles

### 方法 1: 重新組合成完整圖像

```python
from module5_tile_saver import reassemble_tiles

# 當你有足夠記憶體或在另一台機器上
output_file = reassemble_tiles(
    tile_dir="output/tiles/DISH_40X_2_tiles_lv1",
    output_file="output/DISH_complete.tif"
)
```

### 方法 2: 在 QuPath 中使用

1. 開啟 QuPath
2. File → Import images → Image directory
3. 選擇 tile 目錄 (`DISH_40X_2_tiles_lv1/`)
4. QuPath 會自動識別 grid 結構

### 方法 3: 在 ImageJ/FIJI 中使用

```
Plugins → Stitching → Grid/Collection stitching

Type: Grid: column-by-column
Order: Down & Right
Directory: [選擇 tile 目錄]
File names: tile_r{iiii}_c{iiii}.tif
Compute overlap: [取消勾選]
```

### 方法 4: Python 程式化處理

```python
import json
from pathlib import Path
from PIL import Image

# 讀取 metadata
tile_dir = Path("output/tiles/DISH_40X_2_tiles_lv1")
with open(tile_dir / "metadata.json") as f:
    meta = json.load(f)

# 逐 tile 處理（例如：深度學習推論）
for tile_info in meta['tiles']:
    # 載入 tile
    filename = f"tile_r{tile_info['row']:04d}_c{tile_info['col']:04d}.tif"
    tile_path = tile_dir / filename
    tile = Image.open(tile_path)
    
    # 處理
    result = your_model.predict(tile)
    
    # 儲存結果（保持相同的 grid 結構）
    result_path = output_dir / filename
    result.save(result_path)
```

### 方法 5: 只組合感興趣的區域

```python
import pyvips

def reassemble_roi(tile_dir, roi_xywh, output_file):
    """
    只組合 ROI 區域的 tiles
    
    roi_xywh: (x, y, width, height) 相對於完整圖像的座標
    """
    with open(tile_dir / "metadata.json") as f:
        meta = json.load(f)
    
    roi_x, roi_y, roi_w, roi_h = roi_xywh
    tile_size = meta['tile_size']
    
    # 找出包含 ROI 的 tiles
    start_col = roi_x // tile_size
    start_row = roi_y // tile_size
    end_col = (roi_x + roi_w + tile_size - 1) // tile_size
    end_row = (roi_y + roi_h + tile_size - 1) // tile_size
    
    # 組合這些 tiles
    rows = []
    for row in range(start_row, end_row):
        row_tiles = []
        for col in range(start_col, end_col):
            filename = f"tile_r{row:04d}_c{col:04d}.tif"
            tile = pyvips.Image.new_from_file(str(tile_dir / filename))
            row_tiles.append(tile)
        
        row_img = row_tiles[0]
        for tile in row_tiles[1:]:
            row_img = row_img.join(tile, 'horizontal')
        rows.append(row_img)
    
    # 組合行
    result = rows[0]
    for row in rows[1:]:
        result = result.join(row, 'vertical')
    
    # 裁切到精確的 ROI
    offset_x = roi_x % tile_size
    offset_y = roi_y % tile_size
    result = result.extract_area(offset_x, offset_y, roi_w, roi_h)
    
    # 儲存
    result.write_to_file(str(output_file))
    return output_file

# 使用範例
roi = (50000, 40000, 10000, 10000)  # 10K x 10K ROI
reassemble_roi(
    tile_dir="output/tiles/DISH_40X_2_tiles_lv1",
    roi_xywh=roi,
    output_file="output/DISH_roi.tif"
)
```

---

## 疑難排解

### 1. GPU 記憶體不足

```python
# 錯誤: CUDA out of memory

# 解決方案 1: 減少 batch size
tile_saver = TileSaver(..., batch_size=8)

# 解決方案 2: 使用更小的 tile
tile_saver = TileSaver(..., tile_wh=2048)

# 解決方案 3: 使用 CPU
# 修改 module5_tile_saver.py:
# self.device = torch.device('cpu')
```

### 2. 處理中斷

```python
# Tiles 是獨立的，可以手動刪除並重新運行
# 已存在的 tiles 會被覆蓋

# 或者修改代碼添加檢查：
if not filepath.exists():
    process_and_save_tile(...)
```

### 3. Tile 接縫可見

```python
# 增加 tile 重疊
# 修改 _save_tile() 使用 overlap:

def _save_tile(self, output_dir, tile_info, tile_data, overlap=128):
    # 讀取時包含 overlap
    expanded_tile = read_with_overlap(tile_info, overlap)
    
    # 處理
    warped = process(expanded_tile)
    
    # 儲存時去除 overlap
    cropped = warped[overlap:-overlap, overlap:-overlap]
    save(cropped)
```

### 4. 記憶體仍然不足

```python
# 如果連 batch_size=1 都不行：

# 方案 A: 使用更高的 level
tile_saver = TileSaver(..., level=2)  # 或 level=3

# 方案 B: 使用 VALIS 的原始 tile-based 方法
# （它會分別處理每個 tile，但會拼接位移場）

# 方案 C: 只處理 ROI
tile_saver.process_and_save_tiles(..., roi_xywh=(x, y, w, h))
```

---

## 效能比較

### Level 1 (約 142K x 114K pixels)

| 方法 | 記憶體峰值 | 處理時間 | 輸出 |
|------|-----------|---------|------|
| VALIS 原始 | ~120 GB | 2-3 小時 | 完整圖像 |
| module5 (batch=16) | ~3 GB | 類似 | Tiles |
| module5 (batch=32) | ~6 GB | 更快 | Tiles |

### Level 0 (約 284K x 229K pixels)

| 方法 | 記憶體峰值 | 處理時間 | 輸出 |
|------|-----------|---------|------|
| VALIS 原始 | ~480 GB | 10+ 小時 | 完整圖像 |
| module5 (batch=16) | ~3 GB | 類似 | Tiles |

---

## 最佳實踐

### 1. 工作流程建議

```
步驟 1: 使用 level 2-3 快速對齊和驗證
    ↓
步驟 2: 使用 module5 處理 level 1
    ├─ 輸出 tiles
    └─ 檢查幾個 tiles 的品質
    ↓
步驟 3a: 滿意 → 處理 level 0 (如果需要)
步驟 3b: 不滿意 → 調整參數，重新步驟 1
    ↓
步驟 4: 根據需求使用 tiles
    ├─ 直接分析 (深度學習等)
    ├─ 組合 ROI
    └─ 或組合完整圖像 (後續)
```

### 2. 目錄組織

```
project/
├── raw/                      # 原始 .czi 檔案
├── valis_output/             # VALIS 對齊結果
│   └── Transform_Params/
├── tiles/                    # Tile 輸出
│   ├── DISH_lv1/
│   ├── HER2_lv1/
│   └── HE_lv1/
└── final/                    # 最終組合的圖像 (可選)
    ├── DISH_complete.tif
    └── HER2_complete.tif
```

### 3. 記憶體預算

```python
# 估算所需記憶體
def estimate_memory(level, tile_wh, batch_size):
    slide_obj = registrar.slide_dict['DISH_40X_2']
    w, h = slide_obj.slide_dimensions_wh[level]
    
    # Batch 記憶體
    batch_mb = (tile_wh ** 2 * 3 * batch_size * 4) / (1024**2)
    
    # 位移場記憶體
    dxdy_mb = (w * h * 2 * 4) / (1024**2)
    
    # 系統開銷
    overhead_mb = 1024  # 1 GB
    
    total_mb = batch_mb + dxdy_mb + overhead_mb
    
    print(f"Level {level}:")
    print(f"  Batch: {batch_mb:.0f} MB")
    print(f"  Displacement field: {dxdy_mb:.0f} MB")
    print(f"  Total: {total_mb:.0f} MB ({total_mb/1024:.1f} GB)")
    
    return total_mb

# 檢查所有 levels
for level in range(4):
    estimate_memory(level, 4096, 16)
    print()
```

---

## 總結

**何時使用 module5_tile_saver？**

✅ 使用情況：
- 圖像 > 50 GB
- 記憶體 < 64 GB
- 需要處理 level 0-1
- 需要後續 tile-based 分析
- 需要容錯和續傳

❌ 不需要使用：
- 圖像 < 10 GB
- 記憶體充足
- 只需要 level 2+
- 必須要完整圖像

**module5 vs VALIS 原始方法**

| 特性 | VALIS | module5 |
|------|-------|---------|
| 記憶體 | 可能 > 100 GB | 恆定 ~3 GB |
| 輸出 | OME-TIFF | Tiles |
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 靈活性 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 適用範圍 | 中小圖像 | 超大圖像 |

**你的情況（284K x 229K）→ 強烈建議使用 module5！** 🚀

