# Module 5 重構總結

## 問題描述

原始代碼使用 `slide2vips()` 方法時，即使指定了 `xywh` 參數，VALIS 仍然會處理整張影像的所有 tiles（超過 62000 個 tasks），導致處理時間極長且記憶體消耗巨大。

## 解決方案

根據 VALIS 文檔的建議，改用以下工作流程：

1. **載入註冊結果**：使用 `registration.load_registrar()` 載入預先計算的變換參數
2. **讀取指定 level 的完整影像**：使用 `slide2image(level=N)` 讀取需要處理的金字塔層級
3. **使用 `warp_img()` 配合 `crop` 參數**：只對指定區域進行對齊變換

### 核心改進

```python
# 步驟 1: 讀取完整影像（一次性）
dish_img_full = dish_obj.slide2image(level=level)
her2_img_full = her2_obj.slide2image(level=level)

# 步驟 2: 迭代處理每個 tile
for y in range(0, height, tile_size):
    for x in range(0, width, tile_size):
        # 使用 warp_img() 配合 crop 參數，只處理指定區域
        dish_tile = dish_obj.warp_img(
            img=dish_img_full,
            non_rigid=True,
            crop=(x, y, w, h)
        )
```

## 優勢

✅ **記憶體高效**：只處理需要的區域，不會載入整張對齊後的大圖  
✅ **速度快**：避免處理不必要的 tiles  
✅ **靈活性高**：可以指定任意位置和大小的區域進行輸出  
✅ **支援高解析度**：可以處理 Level 0~9 的任意層級

## 使用方法

```python
from pathlib import Path
from module5 import generate_aligned_tiles

output_dir = Path(r"E:\Class\tsgh\thriple_image_layer\output")

# 生成 Level 2 的 2048x2048 tiles
generate_aligned_tiles(
    output_dir=output_dir,
    level=2,              # 金字塔層級（0=最高解析度）
    non_rigid=True,       # 使用非剛性變換
    tile_wh=2048          # Tile 尺寸
)
```

## 測試結果

- **Level 5** (8863 x 7150 像素)：成功生成 20 個 tiles
- **處理速度**：每個 tile 約 2-3 秒
- **記憶體使用**：合理範圍內，無 OOM 問題

## 注意事項

1. **Level 選擇**：
   - Level 0/1：解析度極高，需要大量記憶體和時間
   - Level 2-4：推薦用於高解析度驗證
   - Level 5-7：適合快速測試和預覽

2. **警告信息**：運行時可能會出現 "scaling transformation for image with different shape" 的警告，這是正常的，因為 DISH 和 HER2 影像的原始尺寸略有不同。

3. **輸出格式**：
   - 檔案名格式：`Merged_Tile_lv{level}_x{x}_y{y}_w{w}_h{h}.tiff`
   - 壓縮方式：deflate
   - 座標系統：對齊後的座標空間

## 相關檔案

- `module5.py`：主要實作
- `module3_roi_evaluation.py`：參考的 ROI 提取方法
- `WORKFLOW_V7.1.md`：整體工作流程文檔

