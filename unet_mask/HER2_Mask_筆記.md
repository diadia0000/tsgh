# HER2 Mask 生成方法筆記

> 日期: 2024-12-24
> 目的: 從 HER2 IHC 染色影像中提取 DAB (棕色) 區域的 mask

---

## 📌 問題描述

我們需要從 HER2 免疫組織化學 (IHC) 染色影像中，自動分割出**所有棕色 (DAB) 染色區域**，用於後續的分析或作為 UNet 的訓練資料。

### 挑戰:
1. 棕色 (DAB) 和藍色 (Hematoxylin) 染色會互相干擾
2. 不同區域的染色強度不一致
3. 背景不是純白色，會影響閾值判斷

---

## 🔬 使用的方法: Color Deconvolution

### 參考論文:
> "Color Deconvolution applied to Domain Adaptation in HER2 Histopathological Images"  
> Anglada-Rotger et al., 2023

### 核心概念:
將 RGB 影像轉換到 **Optical Density (OD) 空間**，然後用染色向量分離不同的染色成分。

---

## 📐 實作步驟

### Step 1: 轉換到 OD 空間
```
OD = -log10(I / I0)
```
- `I` = 像素值
- `I0` = 背景值 (從 QuPath 提取: **[206, 206, 212]**)

### Step 2: 使用 QuPath 提取的染色向量
從 QuPath 的 Visual Stain Editor 取得精確的染色向量:

| 染色成分 | 向量 [R, G, B] |
|----------|----------------|
| Hematoxylin (藍) | [0.651, 0.701, 0.29] |
| DAB (棕) | [0.269, 0.568, 0.778] |

### Step 3: 計算各成分濃度
將 OD 影像投影到染色向量上:
```python
dab_concentration = dot(OD, dab_vector)
hema_concentration = dot(OD, hema_vector)
```

### Step 4: 分割邏輯
使用固定閾值 + 比例條件:

```python
# 參數
MIN_TOTAL_OD = 0.05   # 排除純白背景
MIN_DAB_OD = 0.08     # 最小 DAB 濃度
DAB_HEMA_RATIO = 0.8  # DAB/Hema 比例

# 條件
dab_positive = (dab > MIN_DAB_OD) AND 
               ((dab > hema * 0.8) OR (dab > 0.15))

# 排除背景
mask = dab_positive AND (total_OD > 0.05)

# 排除純藍色細胞核
mask = mask AND NOT (hema > 0.2 AND hema > dab * 2)
```

### Step 5: 形態學處理
1. 移除小雜訊 (< 50 pixels)
2. Closing 操作連接鄰近區域
3. 填補小孔洞

---

## 📊 結果

- **Mask 覆蓋率**: ~64% (針對測試圖片)
- **輸出格式**: 
  - 黑色 (0) = DAB 陽性區域
  - 白色 (255) = 背景/其他

---

## 📁 程式碼檔案

| 檔案 | 功能 |
|------|------|
| `her2_mask.py` | 核心 mask 生成演算法 |
| `overlap_to_dish.py` | 生成 mask 並疊加到 DISH 圖片 |
| `test.py` | 舊版測試程式 (參考用) |

---

## 🔧 可調整的參數

如果效果不理想，可以調整這些閾值:

```python
MIN_TOTAL_OD = 0.05   # ↓ 會包含更多淺色區域
MIN_DAB_OD = 0.08     # ↓ 會包含更淡的棕色
DAB_HEMA_RATIO = 0.8  # ↓ 允許更多 Hema 混合
```

---

## 🎯 下一步

1. 在更多樣本上測試
2. 調整參數以達到最佳效果
3. 用生成的 mask 訓練 UNet 模型

---

## 📝 備註

- QuPath 的染色向量是針對我們實驗室的染色條件校正的
- 如果使用不同來源的圖片，可能需要重新校正染色向量
- 背景值 [206, 206, 212] 也是 QuPath 提取的，不是假設純白 255
