# VALIS 區域讀取與 Tile 輸出完整指南

本文件詳細說明如何使用 VALIS 讀取特定區域並輸出 tile，避免處理整個大型影像。

---

## 目錄
1. [核心概念](#核心概念)
2. [方法 1: slide2vips() - 讀取原始區域](#方法-1-slide2vips)
3. [方法 2: warp_slide() - 對齊整個影像](#方法-2-warp_slide)
4. [方法 3: 組合策略 - 對齊後裁切](#方法-3-組合策略)
5. [座標系統轉換](#座標系統轉換)
6. [實際應用範例](#實際應用範例)
7. [效能比較](#效能比較)

---

## 核心概念

### VALIS 的兩種座標系統

1. **原始座標系統** (Unwarped/Source)
   - 原始 slide 檔案的座標
   - 未經對齊變換

2. **對齊座標系統** (Warped/Registered)
   - 經過剛性/非剛性變換後的座標
   - 所有影像對齊到同一空間

### 關鍵問題
- `warp_and_save_slide()` 會輸出**整個對齊後的 slide**
- 如果只需要特定區域，需要使用其他方法

---

## 方法 1: slide2vips()

### 功能
從**原始 slide** 讀取特定區域（未對齊）

### 方法簽名
```python
slide_obj.slide2vips(level, series=None, xywh=None)
```

### 參數詳解

#### `level` (int, 必需)
- 金字塔層級
- `0` = 最高解析度（最大檔案）
- 數字越大，解析度越低
- 範例：
  ```python
  # Level 0: 40000 x 30000 像素
  # Level 1: 20000 x 15000 像素
  # Level 2: 10000 x 7500 像素
  ```

#### `series` (int, optional)
- Slide series 編號
- 預設：`None`（使用預設 series）
- 某些多 series 檔案格式需要指定

#### `xywh` (tuple of int, optional)
- 要讀取的矩形區域
- 格式：`(x, y, width, height)`
- 座標單位：**像素**
- 座標系統：**該 level 的座標系統**
- `None` = 讀取整個 slide

### 返回值
- `pyvips.Image` 物件
- 可進行串流處理，記憶體友善

### 使用範例

#### 範例 1: 讀取整個 slide
```python
from valis import registration

# 載入 registrar
registrar = registration.load_registrar("registrar.pickle")
slide_obj = registrar.slide_dict['DISH_40X_2']

# 讀取整個 level 2
full_img = slide_obj.slide2vips(level=2)
print(f"影像尺寸: {full_img.width} x {full_img.height}")
```

#### 範例 2: 讀取特定區域
```python
# 讀取 level 0 的一個 1024x1024 區域
# 從座標 (5000, 3000) 開始
roi = slide_obj.slide2vips(
    level=0,
    xywh=(5000, 3000, 1024, 1024)
)

# 儲存這個區域
roi.write_to_file("roi_unwarped.tiff", compression='deflate')
```

#### 範例 3: 不同 level 的相同區域
```python
# Level 0 的區域
roi_lv0 = slide_obj.slide2vips(level=0, xywh=(4000, 2000, 2048, 2048))

# Level 1 的相同區域（座標需要除以 2）
roi_lv1 = slide_obj.slide2vips(level=1, xywh=(2000, 1000, 1024, 1024))

# Level 2 的相同區域（座標需要除以 4）
roi_lv2 = slide_obj.slide2vips(level=2, xywh=(1000, 500, 512, 512))
```

### 注意事項
⚠️ **此方法讀取的是原始未對齊的影像**
- 如果需要對齊後的區域，需要額外處理
- 適合：檢查原始影像、預處理、品質控制

---

## 方法 2: warp_slide()

### 功能
對齊整個 slide 並返回 `pyvips.Image`

### 方法簽名
```python
slide_obj.warp_slide(
    level, 
    non_rigid=True, 
    crop=True,
    src_f=None, 
    interp_method="bicubic", 
    reader=None
)
```

### 參數詳解

#### `level` (int, 必需)
- 要對齊的金字塔層級
- 同 `slide2vips()`

#### `non_rigid` (bool, optional)
- `True`: 使用剛性 + 非剛性變換（預設）
- `False`: 僅使用剛性變換
- 非剛性變換更精確但較慢

#### `crop` (bool or str, optional)
- 裁切方式：
  - `True`: 使用初始化時的設定（預設）
  - `False`: 不裁切，保留完整對齊影像
  - `"overlap"`: 裁切到所有影像重疊區域
  - `"reference"`: 裁切到參考影像區域

#### `src_f` (str, optional)
- 要對齊的 slide 檔案路徑
- `None`: 使用 `slide_obj.src_f`（預設）
- 可指定其他檔案（如預處理版本）

#### `interp_method` (str, optional)
- 插值方法：
  - `"nearest"`: 最快，適合遮罩
  - `"bilinear"`: 快速，品質中等
  - `"bicubic"`: 預設，品質好
  - `"lanczos"`: 最慢，品質最好

#### `reader` (SlideReader, optional)
- 自訂 slide reader
- `None`: 使用預設 reader

### 返回值
- `pyvips.Image` 物件
- 已對齊的完整影像

### 使用範例

#### 範例 1: 基本對齊
```python
# 對齊 level 1，使用非剛性變換
warped = slide_obj.warp_slide(level=1)

# 儲存
warped.write_to_file("warped_full.tiff", compression='deflate')
```

#### 範例 2: 僅剛性變換（更快）
```python
# 僅使用剛性變換，速度較快
warped_rigid = slide_obj.warp_slide(
    level=2,
    non_rigid=False
)
```

#### 範例 3: 不裁切
```python
# 保留完整對齊影像，不裁切
warped_full = slide_obj.warp_slide(
    level=1,
    crop=False
)
print(f"完整尺寸: {warped_full.width} x {warped_full.height}")
```

#### 範例 4: 使用最快插值
```python
# 使用 nearest 插值，最快速度
warped_fast = slide_obj.warp_slide(
    level=2,
    interp_method="nearest"
)
```

### 注意事項
⚠️ **此方法會對齊整個 slide**
- 即使只需要小區域，也會處理整張影像
- 對於大型影像可能很慢
- 建議配合 `extract_area()` 裁切需要的區域

---

## 方法 3: 組合策略

### 策略 A: 對齊整個 slide → 裁切區域

**適用情境**：需要多個對齊後的 tile

#### 步驟

1. **對齊整個 slide**
```python
warped_full = slide_obj.warp_slide(
    level=0,
    non_rigid=True,
    crop=False  # 不裁切，保留完整影像
)
```

2. **裁切多個 tile**
```python
# 定義 tile 區域（在對齊後座標系統中）
tiles = [
    (0, 0, 1024, 1024),
    (1024, 0, 1024, 1024),
    (0, 1024, 1024, 1024),
    (1024, 1024, 1024, 1024)
]

# 批次裁切
for i, (x, y, w, h) in enumerate(tiles):
    tile = warped_full.extract_area(x, y, w, h)
    tile.write_to_file(f"tile_{i}.tiff", compression='deflate')
```

#### 優點
- 對齊一次，裁切多次
- 適合需要多個 tile 的情況

#### 缺點
- 需要對齊整個 slide（可能很慢）
- 記憶體使用較高

---

### 策略 B: 讀取原始區域 → 手動對齊

**適用情境**：只需要單一小區域，且對齊速度要求高

#### 步驟

1. **從原始 slide 讀取區域**
```python
# 讀取原始 slide 的特定區域
roi_original = slide_obj.slide2vips(
    level=0,
    xywh=(5000, 3000, 1024, 1024)
)
```

2. **轉換為 numpy 陣列**
```python
from valis import warp_tools

roi_np = warp_tools.vips2numpy(roi_original)
```

3. **手動對齊**
```python
# 計算對齊後的輸出尺寸
# 這需要根據變換矩陣計算
warped_roi = warp_tools.warp_img(
    img=roi_np,
    M=slide_obj.M,
    bk_dxdy=slide_obj.bk_dxdy,
    transformation_src_shape_rc=slide_obj.processed_img_shape_rc,
    transformation_dst_shape_rc=slide_obj.reg_img_shape_rc,
    out_shape_rc=(1024, 1024)  # 輸出尺寸
)
```

#### 優點
- 只處理需要的區域
- 記憶體使用最低

#### 缺點
- 需要手動計算座標轉換
- 較複雜，容易出錯

---

## 座標系統轉換

### 問題
如何知道原始 slide 的某個區域在對齊後的位置？

### 解決方案: `warp_xy()`

#### 方法簽名
```python
slide_obj.warp_xy(
    xy,
    M=None,
    slide_level=0,
    pt_level=0,
    non_rigid=True,
    crop=True
)
```

#### 參數說明

- **`xy`** (ndarray): 座標陣列
  - 格式: `[[x1, y1], [x2, y2], ...]`
  - Shape: `(N, 2)`

- **`M`** (ndarray, optional): 變換矩陣
  - `None`: 使用 `slide_obj.M`

- **`slide_level`** (int or tuple): 對齊後影像的層級
  - int: 金字塔層級
  - tuple: 影像尺寸 `(height, width)`

- **`pt_level`** (int or tuple): 原始座標的層級
  - int: 金字塔層級
  - tuple: 影像尺寸 `(height, width)`

- **`non_rigid`** (bool): 是否使用非剛性變換

- **`crop`** (bool or str): 是否應用裁切偏移

#### 使用範例

##### 範例 1: 單點轉換
```python
import numpy as np

# 原始 slide 中的一個點（level 0）
original_point = np.array([[5000, 3000]])

# 轉換到對齊後的座標
warped_point = slide_obj.warp_xy(
    xy=original_point,
    slide_level=0,
    pt_level=0,
    non_rigid=True,
    crop=False
)

print(f"原始座標: {original_point[0]}")
print(f"對齊後座標: {warped_point[0]}")
```

##### 範例 2: 矩形區域轉換
```python
# 原始矩形的四個角點
original_rect = np.array([
    [5000, 3000],  # 左上
    [6024, 3000],  # 右上
    [6024, 4024],  # 右下
    [5000, 4024]   # 左下
])

# 轉換所有角點
warped_rect = slide_obj.warp_xy(
    xy=original_rect,
    slide_level=0,
    pt_level=0,
    non_rigid=True
)

# 計算對齊後的 bounding box
x_min = int(warped_rect[:, 0].min())
y_min = int(warped_rect[:, 1].min())
x_max = int(warped_rect[:, 0].max())
y_max = int(warped_rect[:, 1].max())

width = x_max - x_min
height = y_max - y_min

print(f"對齊後區域: x={x_min}, y={y_min}, w={width}, h={height}")
```

##### 範例 3: 不同 level 的轉換
```python
# Level 2 的座標
point_lv2 = np.array([[1000, 750]])

# 轉換到 level 0 對齊後的座標
warped_lv0 = slide_obj.warp_xy(
    xy=point_lv2,
    slide_level=0,    # 輸出為 level 0 座標
    pt_level=2,       # 輸入為 level 2 座標
    non_rigid=True
)
```

---

## 實際應用範例

### 範例 1: 輸出單一對齊 Tile

```python
from pathlib import Path
import numpy as np
from valis import registration, slide_io, warp_tools

# 初始化
slide_io.init_jvm()
registrar = registration.load_registrar("registrar.pickle")
slide_obj = registrar.slide_dict['DISH_40X_2']

# 定義原始 slide 中的 tile 區域（level 0）
tile_x = 5000
tile_y = 3000
tile_w = 1024
tile_h = 1024

# 方法 1: 對齊整個 slide 再裁切（簡單但慢）
warped_full = slide_obj.warp_slide(level=0, non_rigid=True, crop=False)
tile = warped_full.extract_area(tile_x, tile_y, tile_w, tile_h)
tile.write_to_file("tile_method1.tiff", compression='deflate')

# 方法 2: 使用座標轉換（複雜但快）
# 轉換 tile 的角點座標
corners = np.array([
    [tile_x, tile_y],
    [tile_x + tile_w, tile_y],
    [tile_x + tile_w, tile_y + tile_h],
    [tile_x, tile_y + tile_h]
])

warped_corners = slide_obj.warp_xy(
    xy=corners,
    slide_level=0,
    pt_level=0,
    non_rigid=True,
    crop=False
)

# 計算對齊後的 bounding box
x_min = int(warped_corners[:, 0].min())
y_min = int(warped_corners[:, 1].min())
x_max = int(warped_corners[:, 0].max())
y_max = int(warped_corners[:, 1].max())

# 對齊整個 slide 並裁切計算出的區域
warped_full = slide_obj.warp_slide(level=0, non_rigid=True, crop=False)
tile = warped_full.extract_area(x_min, y_min, x_max - x_min, y_max - y_min)
tile.write_to_file("tile_method2.tiff", compression='deflate')

slide_io.kill_jvm()
```

---

### 範例 2: 批次輸出多個 Tiles

```python
from pathlib import Path
from valis import registration, slide_io

slide_io.init_jvm()
registrar = registration.load_registrar("registrar.pickle")
slide_obj = registrar.slide_dict['DISH_40X_2']

# 定義多個 tile（在對齊後座標系統中）
tiles = [
    {"name": "tile_topleft", "xywh": (0, 0, 2048, 2048)},
    {"name": "tile_topright", "xywh": (2048, 0, 2048, 2048)},
    {"name": "tile_bottomleft", "xywh": (0, 2048, 2048, 2048)},
    {"name": "tile_bottomright", "xywh": (2048, 2048, 2048, 2048)}
]

# 對齊一次
print("對齊 slide...")
warped = slide_obj.warp_slide(level=0, non_rigid=True, crop=False)

# 批次輸出
output_dir = Path("tiles")
output_dir.mkdir(exist_ok=True)

for tile_info in tiles:
    name = tile_info["name"]
    x, y, w, h = tile_info["xywh"]
    
    print(f"裁切 {name}...")
    tile = warped.extract_area(x, y, w, h)
    
    output_path = output_dir / f"{name}.tiff"
    tile.write_to_file(
        str(output_path),
        compression='deflate',
        tile=True,
        tile_width=256,
        tile_height=256
    )
    print(f"已儲存: {output_path}")

slide_io.kill_jvm()
```

---

### 範例 3: 合併兩張影像的對齊 Tile

```python
from pathlib import Path
import pyvips
from valis import registration, slide_io

slide_io.init_jvm()
registrar = registration.load_registrar("registrar.pickle")

dish_obj = registrar.slide_dict['DISH_40X_2']
her2_obj = registrar.slide_dict['HER2_40X']

# 定義 tile 區域（對齊後座標）
tile_x = 1000
tile_y = 1500
tile_w = 2048
tile_h = 2048

# 對齊兩張 slide
print("對齊 DISH...")
dish_warped = dish_obj.warp_slide(level=0, non_rigid=True, crop=False)

print("對齊 HER2...")
her2_warped = her2_obj.warp_slide(level=0, non_rigid=True, crop=False)

# 裁切相同區域
dish_tile = dish_warped.extract_area(tile_x, tile_y, tile_w, tile_h)
her2_tile = her2_warped.extract_area(tile_x, tile_y, tile_w, tile_h)

# 合併（平均）
merged_tile = (dish_tile * 0.5 + her2_tile * 0.5).cast('uchar')

# 儲存
merged_tile.write_to_file(
    "merged_tile.tiff",
    compression='deflate',
    tile=True,
    tile_width=256,
    tile_height=256
)

print("完成！")
slide_io.kill_jvm()
```

---

## 效能比較

### 情境 1: 輸出單一小 Tile (1024x1024)

| 方法 | 速度 | 記憶體 | 複雜度 |
|------|------|--------|--------|
| `warp_slide()` + `extract_area()` | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| `slide2vips()` + 手動對齊 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| `warp_and_save_slide()` + 外部裁切 | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**推薦**: `slide2vips()` + 手動對齊（如果熟悉座標轉換）

---

### 情境 2: 輸出多個 Tiles (10+ tiles)

| 方法 | 速度 | 記憶體 | 複雜度 |
|------|------|--------|--------|
| `warp_slide()` 一次 + 多次 `extract_area()` | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 多次 `slide2vips()` + 手動對齊 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

**推薦**: `warp_slide()` 一次 + 多次 `extract_area()`

---

### 情境 3: 輸出整個對齊 Slide

| 方法 | 速度 | 記憶體 | 複雜度 |
|------|------|--------|--------|
| `warp_and_save_slide()` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| `warp_slide()` + `write_to_file()` | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

**推薦**: `warp_and_save_slide()`（記憶體友善）

---

## 總結

### 選擇指南

#### 我想要...

1. **讀取原始 slide 的特定區域（未對齊）**
   - 使用：`slide2vips(level, xywh=(x, y, w, h))`

2. **輸出整個對齊後的 slide**
   - 使用：`warp_and_save_slide()`

3. **輸出對齊後的單一 tile**
   - 簡單方法：`warp_slide()` + `extract_area()`
   - 高效方法：`slide2vips()` + 手動對齊

4. **輸出對齊後的多個 tiles**
   - 使用：`warp_slide()` 一次 + 多次 `extract_area()`

5. **轉換座標（原始 → 對齊）**
   - 使用：`warp_xy()`

6. **合併多張影像的對齊 tiles**
   - 使用：對每張影像執行 `warp_slide()` + `extract_area()`
   - 然後用 pyvips 合併

---

## 常見問題

### Q1: `warp_and_save_slide()` 可以指定輸出區域嗎？
**A**: 不行。此方法會輸出整個對齊後的 slide。如需特定區域，請使用 `warp_slide()` + `extract_area()`。

### Q2: 如何知道對齊後影像的尺寸？
**A**: 
```python
warped = slide_obj.warp_slide(level=0, crop=False)
print(f"尺寸: {warped.width} x {warped.height}")
```

### Q3: 座標轉換後的區域可能變形，如何處理？
**A**: 使用 `warp_xy()` 轉換矩形的四個角點，然後計算 bounding box。

### Q4: 如何加速對齊過程？
**A**: 
- 使用較低的 level（較低解析度）
- 設定 `non_rigid=False`（僅剛性變換）
- 使用 `interp_method="nearest"`（最快插值）

### Q5: 記憶體不足怎麼辦？
**A**:
- 使用 `warp_and_save_slide()` 而非 `warp_slide()`
- 使用較高的 level（較低解析度）
- 分批處理 tiles

---

## 參考資源

- [VALIS GitHub](https://github.com/MathOnco/valis)
- [VALIS 文檔](https://valis.readthedocs.io/)
- [pyvips 文檔](https://libvips.github.io/pyvips/)
