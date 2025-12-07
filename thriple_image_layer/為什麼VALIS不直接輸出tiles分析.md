# 為什麼 VALIS 不直接輸出 Tiles？

## 問題分析

你的觀察非常敏銳！VALIS 的設計確實有一個**矛盾**：

### VALIS 的流程

```
1. Tile-based 計算位移場
   ├─ 分割成 62,048 個 tiles
   ├─ 逐個計算位移場 (10 小時)
   └─ 儲存所有位移場到記憶體

2. 拼接位移場
   └─ stitch_tiles() 使用 pyvips.merge()
       └─ 需要大量記憶體！❌

3. 應用變換到完整圖像
   └─ warp_slide() 一次性變換整張圖
       └─ 又需要大量記憶體！❌
```

### 記憶體需求估算

```python
# 你的圖像: 283,637 × 228,733 pixels

# 位移場 (float32)
dxdy_size = 283637 * 228733 * 2 * 4 / (1024**3)
print(f"位移場: {dxdy_size:.2f} GB")  # ≈ 121 GB

# 原始圖像 (RGB uint8)
img_size = 283637 * 228733 * 3 / (1024**3)
print(f"原始圖像: {img_size:.2f} GB")  # ≈ 181 GB

# 變換後的圖像 (RGB uint8)
warped_size = 283637 * 228733 * 3 / (1024**3)
print(f"變換後圖像: {warped_size:.2f} GB")  # ≈ 181 GB

# 總計
total = dxdy_size + img_size + warped_size
print(f"總記憶體需求: {total:.2f} GB")  # ≈ 483 GB！
```

---

## 為什麼 VALIS 不直接輸出 Tiles？

### 1. 歷史原因 - 設計目標不同

VALIS 的設計目標是：
- **研究用途**：產生對齊後的完整圖像供可視化和分析
- **互操作性**：輸出標準的 OME-TIFF 格式
- **易用性**：用戶期望得到一個完整的對齊圖像

### 2. Tiles 的缺點

如果直接輸出 tiles：
- ❌ **不連續**：tiles 之間可能有接縫
- ❌ **難以使用**：大多數軟體期望完整圖像
- ❌ **需要額外步驟**：用戶需要自己組合 tiles
- ❌ **Metadata 複雜**：需要記錄每個 tile 的位置和變換

### 3. PyVIPS 的串流處理

VALIS 使用 PyVIPS 是因為：
- ✅ PyVIPS 使用**惰性求值** (lazy evaluation)
- ✅ 不會立即將整張圖載入記憶體
- ✅ 在 `write_to_file()` 時才逐塊處理

**但實際上**：
- ⚠️ `stitch_tiles()` 使用 `pyvips.merge()` 仍然需要大量記憶體
- ⚠️ 對於超大圖像（你的情況），還是會 OOM

---

## 我們的改進方案：module5_tile_saver.py

### 核心思想

```
不要拼接！直接輸出 tiles！

1. 逐 batch 讀取原始 tiles
   ├─ Batch size = 16 (可控記憶體)
   └─ 使用 GPU 加速變換

2. 應用變換到 batch
   ├─ 剛性變換 (affine)
   └─ 非剛性變換 (displacement field)

3. 直接儲存每個 tile
   └─ tile_r0000_c0000.tif, tile_r0000_c0001.tif, ...

4. 儲存 metadata.json
   └─ 記錄 grid 資訊，方便後續使用
```

### 優勢

#### 1. 記憶體使用恆定

```python
# 只需要處理當前 batch
batch_memory = tile_size^2 * 3 * batch_size * 4 / (1024**3)

# 例如：4096x4096, batch=16, float32
batch_memory = 4096^2 * 3 * 16 * 4 / (1024**3)
             ≈ 3.07 GB

# vs VALIS 的 483 GB！
```

#### 2. 可以暫停和恢復

```python
# 檢查已存在的 tiles
for tile in tiles:
    filename = f"tile_r{tile['row']:04d}_c{tile['col']:04d}.tif"
    if os.path.exists(filename):
        continue  # 跳過已完成的
    else:
        process_tile(tile)  # 處理新的
```

#### 3. 靈活使用

```python
# 選項 1: 直接使用 tiles
# - QuPath 可以載入 tile 目錄
# - ImageJ/FIJI 可以用 Grid/Collection Stitching
# - 深度學習可以逐 tile 推論

# 選項 2: 需要時再組合
reassemble_tiles(tile_dir, output_file)

# 選項 3: 只組合感興趣的區域
reassemble_roi(tile_dir, roi_bbox, output_file)
```

#### 4. 容錯性強

```python
# 如果某個 tile 處理失敗
try:
    process_tile(tile)
except Exception as e:
    print(f"Tile {tile} 失敗: {e}")
    # 其他 tiles 仍然可以繼續！
```

---

## 使用範例

### 1. 直接輸出 tiles（不組合）

```python
from module5_tile_saver import TileSaver

tile_saver = TileSaver(
    registrar=registrar,
    output_dir="output/tiles",
    level=1,
    tile_wh=4096,
    batch_size=16
)

# 處理並儲存 tiles
dish_tiles_dir = tile_saver.process_and_save_tiles('DISH_40X_2', non_rigid=True)

# 結果：
# output/tiles/DISH_40X_2_tiles_lv1/
#   ├─ tile_r0000_c0000.tif
#   ├─ tile_r0000_c0001.tif
#   ├─ ...
#   └─ metadata.json
```

### 2. 後續組合（可選）

```python
from module5_tile_saver import reassemble_tiles

# 當你有足夠記憶體時，或在另一台機器上
reassemble_tiles(
    tile_dir="output/tiles/DISH_40X_2_tiles_lv1",
    output_file="output/DISH_reassembled.tif"
)
```

### 3. 在其他軟體中使用

#### QuPath
```python
# QuPath 可以讀取 tile 目錄作為虛擬 slide
# File -> Import images -> Image directory
```

#### ImageJ/FIJI
```
Plugins -> Stitching -> Grid/Collection stitching
選擇 tile 目錄和 metadata.json
```

#### Python 深度學習
```python
import json
from pathlib import Path

# 讀取 metadata
with open('metadata.json') as f:
    meta = json.load(f)

# 逐 tile 推論
for tile_info in meta['tiles']:
    filename = f"tile_r{tile_info['row']:04d}_c{tile_info['col']:04d}.tif"
    tile = load_tile(filename)
    prediction = model.predict(tile)
    save_prediction(prediction, filename)
```

---

## 比較

| 特性 | VALIS 原始方法 | module5_tile_saver |
|------|----------------|-------------------|
| **記憶體使用** | ~483 GB | ~3 GB (恆定) |
| **輸出格式** | 完整圖像 (OME-TIFF) | 獨立 tiles + metadata |
| **處理時間** | 10+ 小時 | 類似（但可暫停） |
| **容錯性** | 失敗需重來 | 可續傳 |
| **靈活性** | 固定輸出 | 多種使用方式 |
| **後續處理** | 直接使用 | 需要組合或支援 tiles 的軟體 |

---

## 結論

### 為什麼 VALIS 不這樣做？

1. **設計目標**：面向研究人員，優先考慮易用性
2. **標準格式**：輸出標準 OME-TIFF，與現有工具相容
3. **PyVIPS 假設**：假設 PyVIPS 的串流處理足夠
4. **歷史包袱**：早期版本的設計延續至今

### 為什麼我們應該這樣做？

1. ✅ **記憶體效率**：從 483 GB → 3 GB
2. ✅ **可靠性**：支援斷點續傳
3. ✅ **靈活性**：多種使用方式
4. ✅ **擴展性**：適合超大圖像

### 建議

對於你的情況（283K × 228K pixels）：

```python
# 方案 A: 使用 module5_tile_saver (推薦)
# - Level 1 或 2
# - 輸出 tiles
# - 需要時再組合

# 方案 B: 使用更小的 level
# - Level 2 或 3
# - 記憶體需求 ÷ 16
# - VALIS 原始方法可行

# 方案 C: 升級硬體
# - 512 GB RAM
# - 但這不實際...
```

**你的觀察完全正確！tile-based 處理應該徹底貫徹到底，而不是中途拼接！** 🎯

