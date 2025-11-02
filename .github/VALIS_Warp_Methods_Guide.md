# VALIS Warp Methods 完整指南

本文件說明 VALIS 庫中 `Slide` 類別的所有 warp 相關方法的用途和使用方式。

---

## 目錄

1. [warp_slide](#1-warp_slide) - 對齊影像並返回 numpy 陣列
2. [warp_and_save_slide](#2-warp_and_save_slide) - 對齊影像並直接儲存（省記憶體）
3. [warp_img](#3-warp_img) - 對齊任意影像
4. [warp_img_from_to](#4-warp_img_from_to) - 在兩張切片間對齊影像
5. [warp_xy](#5-warp_xy) - 對齊座標點
6. [warp_xy_from_to](#6-warp_xy_from_to) - 在兩張切片間對齊座標點
7. [warp_geojson](#7-warp_geojson) - 對齊 GeoJSON 標註
8. [warp_geojson_from_to](#8-warp_geojson_from_to) - 在兩張切片間對齊 GeoJSON

---

## 1. warp_slide

### 📋 功能說明
將切片影像對齊到參考影像，返回對齊後的 **numpy 陣列**。

### ⚠️ 注意事項
- 會將整個影像載入記憶體
- 大型影像（如 level 0-2）可能導致記憶體不足
- 適合小型影像或高 level（低解析度）

### 📝 參數簽名
```python
warp_slide(self, level, non_rigid=True, crop=True, 
           src_f=None, interp_method='bicubic', reader=None)
```

### 🔧 參數說明
- `level` (int): 金字塔層級 (0=最高解析度)
- `non_rigid` (bool): 是否使用非剛性變換，預設 True
- `crop` (bool): 是否裁切影像，預設 True
- `src_f` (str): 替代來源檔案路徑
- `interp_method` (str): 插值方法，預設 'bicubic'
- `reader`: 自訂讀取器

### 💡 使用範例
```python
from valis import registration

# 載入 registrar
registrar = registration.load_registrar("path/to/registrar.pickle")
slide = registrar.slide_dict['slide_name']

# 對齊影像並獲取 numpy 陣列
aligned_img = slide.warp_slide(level=3, non_rigid=True, crop=True)

# 現在可以用 numpy 或 PIL 處理
print(aligned_img.shape)  # (height, width, channels)
```

### 🎯 適用場景
- 需要進一步處理影像（numpy 運算）
- 影像尺寸較小（level 3-6）
- 記憶體充足的情況

---

## 2. warp_and_save_slide

### 📋 功能說明
將切片影像對齊到參考影像，**直接儲存為 OME-TIFF 檔案**，不經過記憶體。

### ✅ 優點
- **省記憶體**：串流處理，不會一次載入整個影像
- 適合處理超大型影像（level 0-2）
- 自動生成金字塔 TIFF

### 📝 參數簽名
```python
warp_and_save_slide(self, dst_f, level=0, non_rigid=True, crop=True, 
                    src_f=None, channel_names=None, colormap='auto', 
                    interp_method='bicubic', tile_wh=None, 
                    compression='deflate', Q=100, pyramid=True, reader=None)
```

### 🔧 參數說明
- `dst_f` (str): **輸出檔案路徑**（必填）
- `level` (int): 金字塔層級，預設 0
- `non_rigid` (bool): 是否使用非剛性變換，預設 True
- `crop` (bool/str): 裁切方式，可選 True/False/"overlap"/"reference"
- `src_f` (str): 替代來源檔案（例如處理過的版本）
- `channel_names` (list): 通道名稱列表
- `colormap` (dict): 通道顏色映射
- `interp_method` (str): 插值方法，預設 'bicubic'
- `tile_wh` (int): 磁磚大小
- `compression` (str): 壓縮方式，預設 'deflate'
- `Q` (int): JPEG 品質（如使用 JPEG 壓縮）
- `pyramid` (bool): 是否生成金字塔，預設 True
- `reader`: 自訂讀取器

### 💡 使用範例
```python
from valis import registration
from pathlib import Path

# 載入 registrar
registrar = registration.load_registrar("path/to/registrar.pickle")
slide = registrar.slide_dict['slide_name']

# 直接對齊並儲存（省記憶體）
output_path = Path("output/aligned_slide.tiff")
slide.warp_and_save_slide(
    str(output_path),
    level=2,                    # 使用 level 2
    non_rigid=True,             # 啟用非剛性變換
    crop=True,                  # 裁切影像
    compression='deflate',      # 壓縮格式
    pyramid=True                # 生成金字塔
)

print(f"已儲存到: {output_path}")
```

### 🎯 適用場景
- **處理大型影像**（level 0-2）
- 記憶體有限
- 需要儲存對齊結果供後續使用
- 批次處理多張影像

### 📊 記憶體比較
| 方法 | Level 2 記憶體使用 |
|------|-------------------|
| `warp_slide()` | ~46 GB |
| `warp_and_save_slide()` | < 1 GB |

---

## 3. warp_img

### 📋 功能說明
對齊**任意影像**（不限於原始切片），可用於對齊處理過的影像、遮罩等。

### 📝 參數簽名
```python
warp_img(self, img=None, non_rigid=True, crop=True, 
         interp_method='bicubic')
```

### 🔧 參數說明
- `img` (ndarray): 要對齊的影像（numpy 陣列），如為 None 則使用原始影像
- `non_rigid` (bool): 是否使用非剛性變換
- `crop` (bool): 是否裁切
- `interp_method` (str): 插值方法

### 💡 使用範例
```python
import numpy as np
from valis import registration

registrar = registration.load_registrar("path/to/registrar.pickle")
slide = registrar.slide_dict['slide_name']

# 讀取原始影像的某個處理版本（例如分割遮罩）
mask = np.load("segmentation_mask.npy")

# 對齊這個遮罩
aligned_mask = slide.warp_img(
    img=mask,
    non_rigid=True,
    crop=True
)

# 儲存對齊後的遮罩
np.save("aligned_mask.npy", aligned_mask)
```

### 🎯 適用場景
- 對齊影像分割結果
- 對齊細胞核遮罩
- 對齊任何處理過的影像
- 對齊熱圖 (heatmap)

---

## 4. warp_img_from_to

### 📋 功能說明
將影像從**一張切片對齊到另一張切片**，用於跨切片的影像對齊。

### 📝 參數簽名
```python
warp_img_from_to(self, img, to_slide_obj, dst_slide_level=0, 
                 non_rigid=True, interp_method='bicubic', bg_color=None)
```

### 🔧 參數說明
- `img` (ndarray): 來源影像
- `to_slide_obj` (Slide): 目標切片物件
- `dst_slide_level` (int): 目標切片的金字塔層級
- `non_rigid` (bool): 是否使用非剛性變換
- `interp_method` (str): 插值方法
- `bg_color` (tuple): 背景顏色

### 💡 使用範例
```python
from valis import registration

registrar = registration.load_registrar("path/to/registrar.pickle")
source_slide = registrar.slide_dict['HE']
target_slide = registrar.slide_dict['IHC']

# 從 HE 切片讀取標註區域
annotation = source_slide.warp_slide(level=4, crop=False)

# 將標註對齊到 IHC 切片
aligned_annotation = source_slide.warp_img_from_to(
    img=annotation,
    to_slide_obj=target_slide,
    dst_slide_level=4,
    non_rigid=True
)

# 現在 annotation 已對齊到 IHC 切片的座標系統
```

### 🎯 適用場景
- 在不同染色切片間轉移標註
- 將 HE 的分割結果對齊到 IHC
- 跨切片的特徵對齊
- 多模態影像分析

---

## 5. warp_xy

### 📋 功能說明
對齊**座標點**，將點座標從原始位置變換到對齊後的位置。

### 📝 參數簽名
```python
warp_xy(self, xy, M=None, slide_level=0, pt_level=0, 
        non_rigid=True, crop=True)
```

### 🔧 參數說明
- `xy` (ndarray): 座標點陣列，形狀為 (N, 2) 或 (2,)
- `M` (ndarray): 變換矩陣（通常為 None，自動使用註冊的變換）
- `slide_level` (int): 切片的金字塔層級
- `pt_level` (int): 座標點所在的層級
- `non_rigid` (bool): 是否使用非剛性變換
- `crop` (bool): 是否考慮裁切偏移

### 💡 使用範例
```python
import numpy as np
from valis import registration

registrar = registration.load_registrar("path/to/registrar.pickle")
slide = registrar.slide_dict['slide_name']

# 原始座標點（例如細胞核中心）
original_points = np.array([
    [1000, 2000],
    [1500, 2500],
    [2000, 3000]
])

# 對齊這些座標點
aligned_points = slide.warp_xy(
    xy=original_points,
    slide_level=0,
    pt_level=0,
    non_rigid=True
)

print("原始座標:", original_points)
print("對齊後座標:", aligned_points)
```

### 🎯 適用場景
- 對齊細胞核座標
- 對齊關鍵點（keypoints）
- 對齊標註點
- 對齊 ROI 中心點
- 追蹤特定位置在對齊後的變化

---

## 6. warp_xy_from_to

### 📋 功能說明
將座標點從**一張切片對齊到另一張切片**。

### 📝 參數簽名
```python
warp_xy_from_to(self, xy, to_slide_obj, src_slide_level=0, 
                src_pt_level=0, dst_slide_level=0, non_rigid=True)
```

### 🔧 參數說明
- `xy` (ndarray): 來源座標點
- `to_slide_obj` (Slide): 目標切片物件
- `src_slide_level` (int): 來源切片層級
- `src_pt_level` (int): 來源座標點層級
- `dst_slide_level` (int): 目標切片層級
- `non_rigid` (bool): 是否使用非剛性變換

### 💡 使用範例
```python
import numpy as np
from valis import registration

registrar = registration.load_registrar("path/to/registrar.pickle")
he_slide = registrar.slide_dict['HE']
ihc_slide = registrar.slide_dict['IHC']

# 在 HE 影像上標註的腫瘤邊界點
tumor_boundary_he = np.array([
    [1000, 2000],
    [1100, 2100],
    [1200, 2200]
])

# 將這些點對齊到 IHC 切片
tumor_boundary_ihc = he_slide.warp_xy_from_to(
    xy=tumor_boundary_he,
    to_slide_obj=ihc_slide,
    src_slide_level=0,
    dst_slide_level=0,
    non_rigid=True
)

print("HE 座標:", tumor_boundary_he)
print("IHC 座標:", tumor_boundary_ihc)
```

### 🎯 適用場景
- 跨切片轉移標註點
- 在不同染色間對應相同位置
- 多模態分析中的座標對齊
- 病理學家標註的跨切片轉移

---

## 7. warp_geojson

### 📋 功能說明
對齊 **GeoJSON 格式的標註**（多邊形、點、線等），常用於病理影像標註工具（如 QuPath）。

### 📝 參數簽名
```python
warp_geojson(self, geojson_f, M=None, slide_level=0, pt_level=0, 
             non_rigid=True, crop=True)
```

### 🔧 參數說明
- `geojson_f` (str/dict): GeoJSON 檔案路徑或 dict
- `M` (ndarray): 變換矩陣（通常為 None）
- `slide_level` (int): 切片層級
- `pt_level` (int): 標註所在層級
- `non_rigid` (bool): 是否使用非剛性變換
- `crop` (bool): 是否考慮裁切

### 💡 使用範例
```python
import json
from valis import registration

registrar = registration.load_registrar("path/to/registrar.pickle")
slide = registrar.slide_dict['slide_name']

# 對齊 GeoJSON 標註（例如 QuPath 匯出的標註）
aligned_geojson = slide.warp_geojson(
    geojson_f="annotations.geojson",
    slide_level=0,
    non_rigid=True,
    crop=True
)

# 儲存對齊後的標註
with open("aligned_annotations.geojson", 'w') as f:
    json.dump(aligned_geojson, f)
```

### 🎯 適用場景
- 對齊 QuPath 標註
- 對齊病理學家的手動標註
- 對齊 ROI 區域（多邊形）
- 對齊組織結構邊界

---

## 8. warp_geojson_from_to

### 📋 功能說明
將 GeoJSON 標註從**一張切片對齊到另一張切片**。

### 📝 參數簽名
```python
warp_geojson_from_to(self, geojson_f, to_slide_obj, src_slide_level=0, 
                     src_pt_level=0, dst_slide_level=0, non_rigid=True)
```

### 🔧 參數說明
- `geojson_f` (str/dict): GeoJSON 檔案或 dict
- `to_slide_obj` (Slide): 目標切片物件
- `src_slide_level` (int): 來源切片層級
- `src_pt_level` (int): 標註所在層級
- `dst_slide_level` (int): 目標切片層級
- `non_rigid` (bool): 是否使用非剛性變換

### 💡 使用範例
```python
import json
from valis import registration

registrar = registration.load_registrar("path/to/registrar.pickle")
he_slide = registrar.slide_dict['HE']
ihc_slide = registrar.slide_dict['IHC']

# 在 HE 上標註的腫瘤區域
he_annotations = "he_tumor_regions.geojson"

# 將標註對齊到 IHC 切片
ihc_annotations = he_slide.warp_geojson_from_to(
    geojson_f=he_annotations,
    to_slide_obj=ihc_slide,
    src_slide_level=0,
    dst_slide_level=0,
    non_rigid=True
)

# 儲存到 IHC 對應的標註檔案
with open("ihc_tumor_regions.geojson", 'w') as f:
    json.dump(ihc_annotations, f)
```

### 🎯 適用場景
- 將 HE 上的標註轉移到 IHC
- 跨染色的標註對應
- 多模態影像的 ROI 對齊
- 自動化病理分析流程

---

## 📊 方法選擇指南

### 根據使用場景選擇

| 場景 | 推薦方法 | 原因 |
|------|---------|------|
| 處理大型影像 (level 0-2) | `warp_and_save_slide` | 省記憶體 |
| 處理小型影像 (level 3+) | `warp_slide` | 簡單直接 |
| 對齊處理過的影像/遮罩 | `warp_img` | 支援任意影像 |
| 跨切片對齊影像 | `warp_img_from_to` | 支援多模態 |
| 對齊座標點 | `warp_xy` | 精確座標變換 |
| 跨切片對齊座標 | `warp_xy_from_to` | 座標系統轉換 |
| 對齊標註（同一切片） | `warp_geojson` | 支援多邊形 |
| 跨切片對齊標註 | `warp_geojson_from_to` | 標註轉移 |

### 根據記憶體限制選擇

| 可用記憶體 | Level 0-2 | Level 3-4 | Level 5+ |
|-----------|-----------|-----------|----------|
| < 8 GB | `warp_and_save_slide` | `warp_slide` | `warp_slide` |
| 8-32 GB | `warp_and_save_slide` | `warp_slide` | `warp_slide` |
| > 32 GB | `warp_and_save_slide`* | `warp_slide` | `warp_slide` |

*即使記憶體充足，`warp_and_save_slide` 仍然更高效

---

## 💡 最佳實踐

### 1. 處理超大型影像
```python
# ✅ 好的做法：使用 warp_and_save_slide
slide.warp_and_save_slide(
    "output.tiff",
    level=2,
    compression='deflate',
    pyramid=True
)

# ❌ 避免：使用 warp_slide 會記憶體不足
# img = slide.warp_slide(level=2)  # 需要 46+ GB 記憶體！
```

### 2. 批次處理多張影像
```python
from pathlib import Path

output_dir = Path("aligned_slides")
output_dir.mkdir(exist_ok=True)

for slide_name, slide_obj in registrar.slide_dict.items():
    output_path = output_dir / f"{slide_name}_aligned.tiff"
    slide_obj.warp_and_save_slide(
        str(output_path),
        level=2,
        non_rigid=True,
        compression='deflate'
    )
    print(f"✓ {slide_name} 完成")
```

### 3. 使用 pyvips 合併多張對齊影像
```python
import pyvips
from pathlib import Path

# 先分別對齊並儲存
dish_obj.warp_and_save_slide("dish_aligned.tiff", level=2)
her2_obj.warp_and_save_slide("her2_aligned.tiff", level=2)

# 使用 pyvips 串流合併（省記憶體）
dish = pyvips.Image.new_from_file("dish_aligned.tiff", access='sequential')
her2 = pyvips.Image.new_from_file("her2_aligned.tiff", access='sequential')

merged = (dish.cast('float') + her2.cast('float')) / 2
merged = merged.cast('uchar')
merged.write_to_file("merged.tiff", compression='deflate')
```

### 4. 跨切片標註轉移
```python
# 將 HE 上的腫瘤標註轉移到所有 IHC 切片
he_slide = registrar.slide_dict['HE']
he_annotations = "he_tumor_annotations.geojson"

for stain_name, stain_slide in registrar.slide_dict.items():
    if stain_name == 'HE':
        continue
    
    aligned_annotations = he_slide.warp_geojson_from_to(
        geojson_f=he_annotations,
        to_slide_obj=stain_slide,
        non_rigid=True
    )
    
    with open(f"{stain_name}_tumor_annotations.geojson", 'w') as f:
        json.dump(aligned_annotations, f)
```

---

## 🔗 相關資源

- [VALIS 官方文檔](https://github.com/MathOnco/valis)
- [QuPath GeoJSON 格式](https://qupath.github.io/)
- [pyvips 文檔](https://libvips.github.io/pyvips/)

---

## 📝 版本資訊

- 文件建立日期：2025-10-24
- VALIS 版本：適用於 1.0+
- 作者：AI Assistant

---

## ❓ 常見問題

### Q: 為什麼 `warp_slide` 會記憶體不足？
A: `warp_slide` 會將整個影像載入為 numpy 陣列。Level 2 的影像可能超過 40 GB。使用 `warp_and_save_slide` 代替。

### Q: `non_rigid=True` 和 `False` 有什麼差別？
A: 
- `True`: 使用非剛性變換，可以處理組織變形，更準確但更慢
- `False`: 只使用剛性變換（平移、旋轉、縮放），更快但對於有變形的組織可能不夠準確

### Q: `crop=True` 和 `False` 有什麼差別？
A: 
- `True`: 裁切到重疊區域，去除邊緣的空白
- `False`: 保留完整影像，包括變換後的空白區域
- `"overlap"`: 裁切到所有影像都重疊的區域
- `"reference"`: 裁切到與參考影像重疊的區域

### Q: 如何選擇合適的 `level`？
A: 
- Level 0: 最高解析度，檔案最大（適合最終輸出）
- Level 1-2: 高解析度（適合詳細分析）
- Level 3-4: 中等解析度（適合一般分析）
- Level 5+: 低解析度（適合快速預覽）

### Q: `compression='deflate'` 和其他壓縮方式的差別？
A: 
- `'deflate'`: 無損壓縮，檔案較大，品質最佳（推薦）
- `'jpeg'`: 有損壓縮，檔案較小，可能有壓縮失真
- `'lzw'`: 無損壓縮，相容性好
- `None`: 不壓縮，檔案最大

### Q: 哪種插值方法最快？
A: **速度排序（快→慢）**：
1. **`'nearest'`** ⚡ - 最快，但品質較差（鋸齒狀邊緣）
2. **`'linear'`** - 快速且品質可接受
3. **`'area'`** - 中等速度，適合縮小影像
4. **`'cubic'` / `'bicubic'`** - 較慢，預設值，品質好
5. **`'lanczos4'`** - 最慢，品質最好

**建議**：
- 快速預覽：使用 `'nearest'` 或 `'linear'`
- 正式分析：使用 `'bicubic'`（預設）或 `'lanczos4'`
- Level 2 大型影像：使用 `'nearest'` 可節省 50%+ 處理時間

**範例：使用最快插值方法**
```python
# 快速處理大型影像
slide.warp_and_save_slide(
    "output.tiff",
    level=2,
    interp_method='nearest',  # 最快！
    compression='deflate'
)
```

---

**提示**: 在處理大型病理影像時，優先選擇 `warp_and_save_slide` 以避免記憶體問題！

