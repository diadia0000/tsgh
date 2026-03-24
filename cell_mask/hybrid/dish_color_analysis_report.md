# DISH 影像顏色分析報告

## 1. 抽樣分析概要

- 來源：`test_picture/dish/`（共 80 張 2048×2048 TIFF tiles）
- 抽樣：隨機取 8 張進行 RGB + HSV 雙色彩空間分析
- 分析目標：藍色細胞核、紅色 dot (CEP17)、黑色 dot (HER2)

---

## 2. 三類目標的顏色特徵

### 2.1 藍色細胞核（Hematoxylin counterstain）

| 指標 | 值 |
|------|-----|
| RGB 中位值 | R=72-79, G=90-96, B=131-136 |
| RGB 均值 | R=69-76, G=87-93, B=127-133 |
| HSV Hue 範圍 | 200°–270°（平均 ~222°）|
| HSV Saturation | 0.10–0.80（平均 ~0.22）|
| HSV Value | 0.30–0.86（平均 ~0.62）|
| 佔比（寬鬆 HSV 定義）| 約 15-20% |
| 佔比（嚴格 RGB dominant）| 約 1.4-3.7% |

**特徵**：藍色通道明顯高於紅/綠，但整體是偏淡的紫藍色而非飽和藍，Saturation 偏低。

### 2.2 紅色 Dot（CEP17 / Fast Red）

| 指標 | 值 |
|------|-----|
| **Raw RGB 範圍** | R=121-218, G=16-62, B=28-93 |
| **RGB 中位值** | R=133-165, G=33-55, B=66-93 |
| **RGB 均值** | R=132-160, G=35-54, B=67-93 |
| HSV Hue | 320°–355°（平均 ~331°，偏品紅/洋紅）|
| HSV Saturation | 0.30–0.88（平均 ~0.54）|
| HSV Value | 0.35–0.90（平均 ~0.59）|
| 佔比 | 0.02-0.12%（非常稀疏）|

**關鍵發現**：紅色 dot 並不是純紅，而是偏向 **品紅/洋紅 (magenta)** 色調。B 通道值（raw 28-93, gamma 後 0.21-0.49）比預期的「純紅」高很多。這是因為 Fast Red 染劑本身帶有紫紅色調。

### 2.3 黑色 Dot（HER2 / Silver ISH）

| 指標 | 值 |
|------|-----|
| RGB 均值 | R=34-45, G=28-40, B=48-59 |
| RGB 中位值 | R=32-46, G=26-40, B=50-59 |
| HSV Hue | 分散（平均 ~230°，略帶藍）|
| HSV Value | 0.02–0.25（平均 ~0.16）|
| 佔比 | 0.6-1.1% |

**特徵**：非純黑，而是帶有微弱藍色偏移的暗色（B 通道略高於 R、G），這是銀增強訊號在 Hematoxylin 背景下的正常表現。

### 2.4 背景

| 指標 | 值 |
|------|-----|
| RGB 均值 | R=173-179, G=174-179, B=181-185 |
| 佔比 | 95-98% |

背景是淡灰/淡藍灰色，佔絕大部分面積。

---

## 3. 當前 M3 閾值的問題

### 3.1 紅色偵測嚴重不足（核心問題）

**當前設定**（`config.py`）：
```python
red_r_min: float = 0.45      # gamma 後 R 下限
red_b_max: float = 0.35      # gamma 後 B 上限  ← 問題在這裡
red_diff_min: float = 0.10   # R-B 最小差值
```

**實測結果**：

| Tile | M3 偵測到的紅 px | 實際紅 px | **漏檢率** |
|------|-------------------|-----------|------------|
| tile_x4096_y12288 | 720 | 3,732 | **81%** |
| tile_x8192_y4096 | 275 | 3,367 | **92%** |
| tile_x6144_y4096 | 526 | 3,609 | **85%** |
| tile_x14336_y0 | 470 | 720 | **35%** |

**原因分析**：
- DISH 的紅色 dot 是 Fast Red 染劑，實際顏色偏品紅，B 通道在 gamma 後約 0.21-0.49
- `red_b_max=0.35` 把 B 通道 > 0.35 的紅色 dot 全部漏掉了
- 被漏掉的紅色 dot 的 gamma 後 B 均值約 **0.46-0.49**，R-B 差值約 **0.23**
- `red_r_min=0.45` 和 `red_diff_min=0.10` 問題不大

### 3.2 黑色偵測偏保守

**當前設定**：
```python
black_brightness_thresh: float = 0.30  # gamma 後平均亮度上限
```

**實測**：gamma 後偵測到的黑色 px 約 15k-30k，而用 raw RGB < 80 的定義約 26k-48k。差距約 40-50%。不過黑色閾值影響相對較小，因為真正的 HER2 silver dot 確實很暗。

---

## 4. 文獻回顧：DISH 紅黑點比例分析方法

### 4.1 ASCO/CAP HER2 評分標準（2018/2023）

DISH 中：**黑色 dot = HER2（SISH 銀增強）**，**紅色 dot = CEP17（Fast Red）**

需計算至少 **20 個非重疊腫瘤細胞**的 HER2/CEP17 比值：

| 群組 | HER2/CEP17 Ratio | 平均 HER2 Copy 數 | 判定 |
|------|-------------------|-------------------|------|
| Group 1 | ≥ 2.0 | ≥ 4.0 | **ISH 陽性** |
| Group 2 | ≥ 2.0 | < 4.0 | 需併 IHC 判讀 |
| Group 3 | < 2.0 | ≥ 6.0 | 需併 IHC 判讀 |
| Group 4 | < 2.0 | 4.0–5.9 | 需併 IHC 判讀 |
| Group 5 | < 2.0 | < 4.0 | **ISH 陰性** |

> 目前 `m3_dot_quant.py` 的 `compute_ratio()` 計算 black/red（HER2/CEP17），方向正確。

### 4.2 商用系統方法

| 系統 | 方法 |
|------|------|
| **Roche uPath HER2 Dual ISH IA** | ML 排名演算法選 20 個最佳細胞；產生 heat map |
| **Visiopharm HER2-SISH** | Hematoxylin 色彩解卷積 → 多項式 blob filter（只保留圓形物件）→ cluster 分小（6）/大（12）|
| **HALO ISH Module** | 同時量化最多 3 個 chromogenic/silver ISH probe |
| **medRxiv 2025 DL 系統** | 半監督 random forest 分割訊號 + Cellpose 分割細胞 → 多參數排名選 top 20 |

### 4.3 關鍵方法論差異

1. **色彩解卷積（Color Deconvolution）**：將 RGB 轉 OD 空間分離 Hematoxylin / Silver / Fast Red，比直接 RGB 閾值更穩健
2. **形狀過濾**：用圓度（circularity）過濾，只保留圓形 dot，排除邊緣碎片
3. **距離約束**：Visiopharm 排除離最近另一色 dot > 5μm 的訊號（artifact rejection）
4. **細胞排名**：不是分析所有細胞，而是用球度、面積、訊號品質排名，選 top 20

---

## 5. 建議的 Code 修改

### 5.1 緊急修正：紅色閾值（影響最大）

**`config.py`**：
```python
# ========== Dot 定量參數 (M3) ==========
# 原始值（太嚴格）：
# red_r_min: float = 0.45
# red_b_max: float = 0.35     ← 主要問題
# red_diff_min: float = 0.10

# 建議修正值：
red_r_min: float = 0.50       # 稍微提高，避免抓到淡粉色背景
red_b_max: float = 0.52       # 大幅放寬，容納品紅色調的 Fast Red
red_diff_min: float = 0.15    # 稍微提高，確保 R 明顯高於 B
```

**理由**：
- 實測紅 dot gamma 後 B 值範圍 0.21-0.49，大部分在 0.35-0.49
- `red_b_max` 從 0.35 提升到 0.52 可以捕獲 >90% 的紅 dot
- 同時提高 `red_r_min` 到 0.50 和 `red_diff_min` 到 0.15 來控制 false positive

### 5.2 改進：加入 HSV 色彩空間輔助偵測

**`m3_dot_quant.py`** 新增：
```python
def threshold_red_hsv(
    image: np.ndarray,
    hue_ranges: list = [(0, 30), (310, 360)],
    sat_min: float = 0.25,
    val_min: float = 0.35,
) -> np.ndarray:
    """使用 HSV 色彩空間分離紅/品紅色 dot。

    相較 RGB 閾值對染色強度變異更穩健。
    """
    import cv2
    # 將 float64 [0,1] 轉為 uint8 再轉 HSV
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:,:,0].astype(float) * 2, hsv[:,:,1].astype(float) / 255, hsv[:,:,2].astype(float) / 255
    # OpenCV HSV: H=0-180 → 乘 2 轉為 0-360

    mask = np.zeros(h.shape, dtype=bool)
    for lo, hi in hue_ranges:
        mask |= (h >= lo) & (h <= hi)
    mask &= (s >= sat_min) & (v >= val_min)
    return mask.astype(np.uint8)
```

然後在 `quantify_overlay_signals()` 中將 RGB 和 HSV 結果取聯集：
```python
red_binary_rgb = threshold_red(img_gamma, red_r_min, red_b_max, red_diff_min)
red_binary_hsv = threshold_red_hsv(img_gamma, hue_ranges=[(0, 30), (310, 360)], sat_min=0.25, val_min=0.35)
red_binary = np.maximum(red_binary_rgb, red_binary_hsv)  # 聯集
```

### 5.3 改進：加入圓度過濾

```python
def filter_by_circularity(
    binary: np.ndarray,
    min_circularity: float = 0.4,
) -> np.ndarray:
    """過濾非圓形的 connected component，排除邊緣碎片。

    circularity = 4π × area / perimeter²
    完美圓 = 1.0，dot 通常 > 0.4。
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(binary)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity >= min_circularity:
            cv2.drawContours(filtered, [cnt], -1, 1, -1)
    return filtered
```

在 `morphological_postprocess` 之後呼叫，放在 cluster counting 之前。

### 5.4 改進優先順序

| 優先級 | 修改 | 預期影響 | 工作量 |
|--------|------|----------|--------|
| **P0** | 修正 `red_b_max` 閾值 | 紅 dot 偵測率從 ~10% → ~90% | 改 1 個數字 |
| **P1** | 加入 HSV 輔助偵測 | 對染色變異更穩健 | ~30 行 |
| **P1** | 加入圓度過濾 | 減少 false positive | ~20 行 |
| **P3** | Color deconvolution | 最穩健的方法但需標定 stain vector | ~100 行 |

---

## 6. 驗證圖片

分析過程產生的視覺化圖片存放在 `test_picture/dish_analysis/`：
- `*_crop_center.png`：原始影像中心 512×512 裁切
- `*_crop_overlay.png`：顏色標注（青色=藍核, 黃色=紅dot, 綠色=黑dot）
- `*_overlay.png`：全圖顏色標注

---

## 參考文獻

1. ASCO/CAP 2018 HER2 Guideline - JCO (doi: 10.1200/JCO.2018.77.8738)
2. ASCO/CAP 2023 Update - JCO (doi: 10.1200/JCO.22.02864)
3. Roche uPath HER2 Dual ISH Image Analysis
4. Visiopharm HER2-SISH Breast Cancer Algorithm
5. HALO ISH Module (Indica Labs)
6. medRxiv 2025 - Deep learning auto-quantification for HER2 DISH
7. Ruifrok & Johnston 2001 - Color deconvolution for IHC
