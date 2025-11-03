# Module 5 優化筆記

## 問題分析

原始代碼在處理 Level 2 (70829 x 57207 像素) 時會：
1. 使用 `slide2vips(level=2)` 讀取**完整**的 Level 2 影像到記憶體
2. 這會導致大量記憶體消耗和長時間的 I/O 等待

## 解決方案

採用**延遲載入 (Lazy Loading)** 策略：

### 核心改進

1. **使用 `slide_obj.image` 而不是 `slide2vips()`**
   - `slide_obj.image` 是 Level 0 的 pyvips 延遲載入參考
   - pyvips 只會在實際需要時才載入影像數據

2. **使用 `warp_img()` 的 `crop` 參數**
   ```python
   dish_tile = dish_obj.warp_img(
       img=dish_obj.image,  # Level 0 延遲載入
       non_rigid=True,
       crop=(x_lv0, y_lv0, w_lv0, h_lv0)  # Level 0 座標空間
   )
   ```

3. **Level 0 座標轉換**
   ```python
   scale_factor = 2 ** level  # Level 2 = 4
   x_lv0 = x * scale_factor
   y_lv0 = y * scale_factor
   w_lv0 = w * scale_factor
   h_lv0 = h * scale_factor
   ```

4. **下採樣到目標 Level**
   ```python
   if level > 0:
       dish_tile = dish_tile.resize(1.0 / scale_factor, kernel='lanczos3')
       her2_tile = her2_tile.resize(1.0 / scale_factor, kernel='lanczos3')
   ```

## API 修正

根據 VALIS 文檔和實際測試：

1. `warp_img()` 方法簽名：
   ```python
   slide_obj.warp_img(
       img,          # pyvips.Image - 原始影像
       non_rigid,    # bool - 是否使用非剛性變換
       crop          # tuple (x, y, w, h) - 裁切區域 (Level 0 座標)
   )
   ```

2. **不支持的參數**：
   - ❌ `level` - warp_img 不接受此參數
   - ❌ `bg_color` - warp_img 不接受此參數
   - ❌ `slide_obj` - warp_img 是實例方法，不是靜態函數

## 記憶體優化效果

### 之前 (讀取完整影像)
```
正在讀取 Level 2 的完整影像...
Converting slide to pyvips image
QUEUEING TASKS: 100% |████| 3920/3920 [00:05<00:00, 669.46tiles/s]
PROCESSING TASKS:   1% |█   | 38/3920 [00:23<39:17,  1.65tiles/s]
```
- 需要載入整個 Level 2 影像 (~70829 x 57207 像素)
- 預估處理時間: ~40 分鐘

### 之後 (延遲載入)
```
--- 開始逐 Tile 處理（延遲載入模式）---
處理 Tile #1: (0, 0, 2048x2048)...
處理 Tile #2: (2048, 0, 2048x2048)...
...
```
- 只在需要時載入小區域
- 每個 tile 獨立處理
- 記憶體使用量大幅降低

## 工作流程

1. 載入 registrar 和 slide 物件
2. 計算對齊後的工作區域尺寸
3. 計算 Level 0 到目標 Level 的縮放因子
4. 逐 tile 處理：
   a. 計算 Level N 的 tile 座標
   b. 轉換到 Level 0 座標空間
   c. 使用 `warp_img()` 讀取並變換 tile
   d. 下採樣到目標 Level
   e. 合併 DISH 和 HER2 tiles
   f. 儲存結果

## 文件變更

修改文件: `module5.py`

主要改動:
1. 移除 `slide2vips(level=level)` 的完整影像載入
2. 使用 `slide_obj.image` 進行延遲載入
3. 添加 Level 0 座標轉換邏輯
4. 添加下採樣步驟
5. 移除不支持的 `bg_color` 參數

## 注意事項

1. **座標空間**: `warp_img()` 的 `crop` 參數使用 **Level 0 的對齊後座標空間**
2. **延遲載入**: `slide_obj.image` 是 pyvips 延遲載入參考，不會立即載入數據
3. **記憶體管理**: pyvips 會自動管理記憶體，但仍建議處理大型影像時監控記憶體使用

## 效能建議

- Level 2: 適合高品質驗證，記憶體需求適中
- Level 3: 平衡品質與效能
- Level 4: 快速預覽，記憶體需求最低

