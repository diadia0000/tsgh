# VALIS 記憶體問題深度分析與解決方案

## 📊 您的環境分析

### 硬體環境
| 項目 | 數值 |
|------|------|
| RAM | **48 GB** |
| valis-wsi 版本 | 1.2.0 (最新) |

### 圖片規格 (來自 analyze.txt)

| 檔案 | 瓦片數 | 馬賽克邊界框 | 估計大小 |
|------|--------|-------------|----------|
| DISH_40X_2.czi | 15,759 | 283,317 × 228,831 px | 669.17 GB |
| HER2_40X.czi | 15,759 | 283,637 × 228,733 px | 669.17 GB |
| HE_40X.czi | 15,759 | 283,435 × 232,800 px | 669.17 GB |

### 金字塔層級 (Pyramid Levels)

| Scale Factor | 估計尺寸 (pixels) | 估計大小 |
|--------------|-------------------|----------|
| 1.0 (40x) | 283,317 × 228,831 | 669 GB |
| 0.5 (20x) | 141,658 × 114,415 | 167 GB |
| 0.25 (10x) | 70,829 × 57,207 | 42 GB |
| 0.125 (5x) | 35,414 × 28,603 | 10.5 GB |
| 0.0625 (2.5x) | 17,707 × 14,301 | 2.6 GB |

### 您的限制條件

1. ✅ 必須使用 Non-rigid registration (醫療分析)
2. ✅ 不可使用 20x 以下的解析度
3. ⚠️ RAM 48GB，但 20x 解析度預估需要 **~180 GB** 記憶體

---

## 🔴 核心問題：記憶體完全不足

### 在 20x (scale_factor=0.5) 下的記憶體需求估算

```
圖片尺寸: ~141,658 × 114,415 = 16.2 billion pixels
RGB 圖片: 16.2B × 3 bytes = 48.6 GB
位移場 (dx, dy float32): 16.2B × 2 × 4 bytes = 129.6 GB  
索引圖: 16.2B × 2 × 4 bytes = 129.6 GB
────────────────────────────────────────
預估總需求: ~180-200 GB
您的 RAM: 48 GB
差距: 約 4 倍不足
```

### 問題根源 (來自 warp_tools.py 分析)

```python
# warp_tools.py 第1200-1210行
# 這是記憶體爆炸的主因

# 1. 創建索引圖 (與目標尺寸相同)
index = pyvips.Image.xyz(affine_warped.width, affine_warped.height)

# 2. 計算 warp 索引 (又是相同尺寸)
warp_index = (index[0] + warp_dxdy[0]).bandjoin(index[1] + warp_dxdy[1])

# 3. 執行 mapim (需要所有數據在記憶體中)
warped = affine_warped.mapim(warp_index, ...)
```

即使使用 pyvips，`mapim` 操作在處理非剛性變換時仍需要大量記憶體。

---

## 🟢 解決方案

### 方案 A: 分塊處理 (Tiled Warping) - **推薦**

**原理**: 將大圖片分成小塊，分別 warp 每個塊，然後拼接回去。

**優點**:
- 每次只處理一小塊，記憶體使用可控
- 可以達到全解析度
- 不犧牲 non-rigid registration 精度

**缺點**:
- 需要自己實作
- 塊邊界可能需要處理重疊

**實作思路**:

```python
def warp_slide_tiled(slide_obj, level, tile_size=4096, overlap=256):
    """
    分塊 warp slide
    
    步驟:
    1. 獲取目標尺寸和 warping 參數
    2. 將目標區域分成 tiles
    3. 對每個 tile:
       a. 計算該 tile 對應的原始區域 (考慮位移場)
       b. 讀取原始區域 + 邊界 padding
       c. Warp 該區域
       d. 裁切到目標 tile 尺寸
       e. 寫入 BigTIFF
    4. 使用 pyvips 將所有 tiles 合併
    """
    pass
```

### 方案 B: 只在 Registration 時使用 Non-Rigid，Warp 時使用 Rigid Only

**原理**: 
- 在低解析度下計算 non-rigid 位移場 (這步驟記憶體沒問題)
- 在全解析度 warp 時，只使用 rigid transformation (記憶體可行)
- Non-rigid 對齊效果會略差，但 rigid 部分是精確的

```python
# 完整 registration (包含 non-rigid)
rigid_registrar, non_rigid_registrar, error_df = registrar.register()

# Warp 時只使用 rigid
slide_obj.warp_and_save_slide(
    dst_f="output.ome.tiff",
    level=1,  # 20x
    non_rigid=False,  # 只使用 rigid transformation
)
```

**⚠️ 這可能不符合您的精度需求**

### 方案 C: 使用虛擬記憶體 (Swap/Pagefile)

**原理**: 讓 Windows 使用硬碟作為虛擬記憶體

**設置方式** (Windows):
1. 右鍵「此電腦」→「內容」→「進階系統設定」
2. 「效能」→「設定」→「進階」→「虛擬記憶體」→「變更」
3. 設定自訂大小: 初始 **150 GB**，最大 **250 GB**
4. 確保該磁碟有足夠空間

**優點**: 不需要改 code
**缺點**: 極慢 (可能需要數小時處理一張圖)

### 方案 D: 雲端運算

使用 AWS/GCP/Azure 的高記憶體實例:

| 雲端 | 實例類型 | RAM | 估計費用 |
|------|---------|-----|---------|
| AWS | x2idn.16xlarge | 512 GB | ~$2/hr |
| GCP | n2-highmem-64 | 512 GB | ~$2.5/hr |
| Azure | Standard_E64_v5 | 512 GB | ~$2.3/hr |

---

## 🏆 推薦策略: 分塊處理 (Tiled Warping)

這是唯一能在您的硬體上達到您需求的方案。

### 實作概述

```
原始圖片 (20x, ~141K × 114K)
    ↓
分成 n × m 個 tiles (例如 4096 × 4096)
    ↓
對每個 tile:
  1. 從位移場獲取該 tile 對應的位移
  2. 計算需要讀取的原始區域 (包含 padding)
  3. 讀取原始 tile
  4. 應用 rigid + non-rigid 變換
  5. 寫入臨時 tile
    ↓
使用 pyvips 拼接所有 tiles → 輸出 BigTIFF
```

### 記憶體估算 (使用 4096×4096 tiles)

```
每個 tile: 4096 × 4096 × 3 = 50 MB
位移場 tile: 4096 × 4096 × 2 × 4 = 134 MB  
padding (假設 512px): 額外 ~20-30 MB
────────────────────────────────────────
每個 tile 處理: ~250 MB
峰值記憶體: ~2-4 GB (非常安全)
```

### 挑戰

1. **位移場的連續性**: 需要確保 tile 邊界的位移場平滑過渡
2. **tile 邊界處理**: 需要重疊區域來避免接縫
3. **實作複雜度**: 需要仔細處理座標轉換

---

## 📋 下一步

### 我建議的行動計畫

1. **短期 (立即可行)**:
   - 使用方案 C (虛擬記憶體) 進行測試
   - 這可以讓您立即嘗試 20x 解析度，雖然會很慢

2. **中期 (1-2 天開發)**:
   - 實作方案 A (分塊處理)
   - 我可以幫您設計和實作這個解決方案

3. **備選**:
   - 如果分塊處理不可行，考慮方案 D (雲端運算)

### 請確認

1. 您是否想讓我幫您實作分塊 warp 的解決方案？
2. 您目前的硬碟可用空間有多少？(用於虛擬記憶體或臨時 tiles)
3. 處理一張圖片需要多長時間可以接受？

---

## 附錄: VALIS 源碼中的相關發現

### 源碼位置

| 檔案 | 函數 | 說明 |
|------|------|------|
| `registration.py:894-1001` | `Slide.warp_slide()` | 主要 warp 入口 |
| `slide_tools.py:291-361` | `warp_slide()` | 調用 warp_tools |
| `warp_tools.py:1100-1230` | `warp_img()` | 實際 warp 邏輯 - **記憶體問題根源** |

### VALIS 的 tiled 處理

VALIS 有內建 tiled 處理邏輯 (`TILER_THRESH_GB = 10 GB`)，但根據 GitHub 討論，這個功能可能不夠完善。您可以嘗試:

```python
# 降低觸發 tiled 處理的閾值
import valis.registration as registration
registration.TILER_THRESH_GB = 5  # 改為 5 GB 觸發

# 然後執行 warp
```

但這可能不夠可靠，所以自己實作分塊處理是更穩妥的方案。
