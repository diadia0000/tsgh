# VALIS Warp Functions 使用指南

本文件說明 VALIS 套件中 `Slide` 類別的各種 warp (變形/對齊) 相關函數的用途與使用方式。

---

## 1. `warp_slide()`

### 用途
將整張投影片影像進行對齊變換，返回對齊後的 NumPy 陣列。

### 使用方式
```python
warped_img = slide_obj.warp_slide(
    level=3,              # 金字塔層級 (0=最高解析度)
    non_rigid=True,       # 是否使用非剛性變換
    crop=True,            # 是否裁切到重疊區域
    interp_method='bicubic'  # 插值方法
)
```

### 特點
- 返回 NumPy 陣列 (會佔用記憶體)
- 適合需要進一步處理影像數據的情況
- **記憶體密集**：大型影像可能導致記憶體不足

---

## 2. `warp_and_save_slide()`

### 用途
將整張投影片進行對齊變換，並直接儲存到檔案，避免在記憶體中載入完整影像。

### 使用方式
```python
slide_obj.warp_and_save_slide(
    dst_f="output.tiff",     # 輸出檔案路徑
    level=3,                 # 金字塔層級
    non_rigid=True,          # 是否使用非剛性變換
    crop=True,               # 是否裁切
    compression='deflate',   # 壓縮方式 (注意：某些版本可能不支援此參數)
    pyramid=True,            # 是否建立金字塔結構
    interp_method='bicubic'  # 插值方法
)
```

### 特點
- **記憶體友善**：使用串流處理，不會一次載入整張影像
- 適合大型影像的對齊與儲存
- VALIS 會自動將檔名改為 `.ome.tiff` 格式
- **推薦用於大型影像處理**

---

## 3. `warp_img()`

### 用途
對任意 NumPy 影像陣列進行對齊變換（使用此 Slide 對象的變換參數）。

### 使用方式
```python
warped = slide_obj.warp_img(
    img,                  # NumPy 影像陣列
    non_rigid=True,       # 是否使用非剛性變換
    crop=True,            # 是否裁切
    interp_method='bicubic'  # 插值方法
)
```

### 特點
- 可以對自定義影像應用對齊變換
- 適合處理遮罩、標註等衍生影像

---

## 4. `warp_img_from_to()`

### 用途
將影像從一個空間變換到另一個空間（例如從 Slide A 的空間變換到 Slide B 的空間）。

### 使用方式
```python
warped = slide_obj.warp_img_from_to(
    img,                  # NumPy 影像陣列
    from_level,           # 來源金字塔層級
    to_level,             # 目標金字塔層級
    from_slide_obj,       # 來源 Slide 對象
    to_slide_obj,         # 目標 Slide 對象
    non_rigid=True,       # 是否使用非剛性變換
    crop=True,            # 是否裁切
    interp_method='bicubic'  # 插值方法
)
```

### 特點
- 跨投影片的影像變換
- 適合將標註從一張影像對應到另一張影像

---

## 5. `warp_xy()`

### 用途
對座標點 (x, y) 進行對齊變換。

### 使用方式
```python
warped_coords = slide_obj.warp_xy(
    xy,                   # 座標陣列 [[x1, y1], [x2, y2], ...]
    M=None,               # 變換矩陣（可選，預設使用 slide 的變換）
    non_rigid=True,       # 是否使用非剛性變換
    slide_level=0         # 座標所在的金字塔層級
)
```

### 特點
- 適合對齊標註點、ROI 邊界等
- 返回變換後的座標

---

## 6. `warp_xy_from_to()`

### 用途
將座標從一個投影片空間變換到另一個投影片空間。

### 使用方式
```python
warped_coords = slide_obj.warp_xy_from_to(
    xy,                   # 座標陣列
    from_slide_obj,       # 來源 Slide 對象
    to_slide_obj,         # 目標 Slide 對象
    from_level=0,         # 來源層級
    to_level=0,           # 目標層級
    non_rigid=True        # 是否使用非剛性變換
)
```

### 特點
- 跨投影片的座標對應
- 適合多模態影像的標註轉換

---

## 7. `warp_geojson()`

### 用途
對 GeoJSON 格式的幾何標註進行對齊變換。

### 使用方式
```python
warped_geojson = slide_obj.warp_geojson(
    geojson,              # GeoJSON 字典或檔案路徑
    non_rigid=True,       # 是否使用非剛性變換
    slide_level=0         # 標註所在的金字塔層級
)
```

### 特點
- 支援多邊形、點、線等幾何標註
- 適合整合病理影像標註工具（如 QuPath）

---

## 8. `warp_geojson_from_to()`

### 用途
將 GeoJSON 標註從一個投影片空間變換到另一個投影片空間。

### 使用方式
```python
warped_geojson = slide_obj.warp_geojson_from_to(
    geojson,              # GeoJSON 字典或檔案路徑
    from_slide_obj,       # 來源 Slide 對象
    to_slide_obj,         # 目標 Slide 對象
    from_level=0,         # 來源層級
    to_level=0,           # 目標層級
    non_rigid=True        # 是否使用非剛性變換
)
```

### 特點
- 跨投影片的標註對應
- 適合多模態影像的標註整合

---

## 插值方法 (Interpolation Methods) 比較

VALIS 支援多種插值方法，影響變換的速度與品質：

### 1. **Nearest Neighbor** (`'nearest'`)
- **速度**: ⭐⭐⭐⭐⭐ (最快)
- **品質**: ⭐ (最低)
- **適用**: 遮罩、分割結果（避免產生中間值）
- **記憶體**: 最低

### 2. **Bilinear** (`'bilinear'`)
- **速度**: ⭐⭐⭐⭐
- **品質**: ⭐⭐⭐
- **適用**: 一般影像、快速預覽
- **記憶體**: 低

### 3. **Bicubic** (`'bicubic'`) — **預設值**
- **速度**: ⭐⭐⭐
- **品質**: ⭐⭐⭐⭐
- **適用**: 高品質影像變換、出版品質
- **記憶體**: 中等

### 4. **Lanczos** (`'lanczos'`)
- **速度**: ⭐⭐
- **品質**: ⭐⭐⭐⭐⭐ (最高)
- **適用**: 最高品質需求、影像縮放
- **記憶體**: 高

### 速度排名（從快到慢）
1. `nearest` (最快)
2. `bilinear`
3. `bicubic`
4. `lanczos` (最慢但品質最好)

---

## 使用建議

### 大型影像處理
- **優先使用**: `warp_and_save_slide()` — 避免記憶體不足
- **插值方法**: `bilinear` 或 `nearest` — 平衡速度與品質

### 高品質輸出
- **使用**: `warp_slide()` 或 `warp_and_save_slide()`
- **插值方法**: `bicubic` 或 `lanczos`

### 遮罩/標註處理
- **使用**: `warp_img()` 或 `warp_xy()`
- **插值方法**: `nearest` — 保持標籤的離散性

### 記憶體受限環境
- **避免**: `warp_slide()` — 會載入完整陣列到記憶體
- **使用**: `warp_and_save_slide()` + `pyvips` 串流處理

---

## 範例：記憶體友善的大型影像合併

```python
from pathlib import Path
import pyvips
from valis import registration, slide_io

# 初始化
slide_io.init_jvm()
registrar = registration.load_registrar("registrar.pickle")

# 取得對齊物件
slide1 = registrar.slide_dict['slide1']
slide2 = registrar.slide_dict['slide2']

# 使用最快的插值方法 (nearest) 對齊並儲存
slide1.warp_and_save_slide(
    "temp_slide1.tiff",
    level=2,
    non_rigid=False,
    crop=True,
    interp_method='nearest'  # 最快
)

slide2.warp_and_save_slide(
    "temp_slide2.tiff",
    level=2,
    non_rigid=False,
    crop=True,
    interp_method='nearest'  # 最快
)

# 使用 pyvips 串流合併（不會一次載入全部）
img1 = pyvips.Image.new_from_file("temp_slide1.ome.tiff", access='sequential')
img2 = pyvips.Image.new_from_file("temp_slide2.ome.tiff", access='sequential')

merged = (img1.cast('float') + img2.cast('float')) / 2
merged = merged.cast('uchar')

merged.write_to_file("merged_output.tiff", compression='deflate', 
                     tile=True, pyramid=True)

slide_io.kill_jvm()
```

---

## 注意事項

1. **`warp_and_save_slide()` 參數支援**
   - 某些版本的 VALIS 可能不支援 `compression` 參數
   - 建議先測試或查閱您使用的 VALIS 版本文檔

2. **檔名自動修改**
   - VALIS 會自動將輸出檔名改為 `.ome.tiff`
   - 讀取時需使用 `.ome.tiff` 副檔名

3. **JVM 管理**
   - 使用完畢後務必呼叫 `slide_io.kill_jvm()`
   - 建議使用 `try...finally` 確保清理

4. **記憶體管理**
   - Level 數字越小，解析度越高，記憶體需求越大
   - 建議從高 level (低解析度) 開始測試

