# HER2 染色細胞膜遮罩工具

## 概述

本工具專門用於從 HER2 染色的組織病理學影像中提取咖啡色細胞膜，基於 HED (Hematoxylin-Eosin-DAB) 色彩分離技術，提供精確的細胞膜分割和分析功能。

## 核心技術

### 1. HED 色彩分離
- 使用 `scikit-image` 的 `rgb2hed` 函式將 RGB 影像分離為：
  - **Hematoxylin 通道**：藍色細胞核
  - **DAB 通道**：咖啡色細胞膜 (HER2 陽性染色)

### 2. DAB 通道處理
- **反轉與正規化**：取負號並正規化到 [0,1] 範圍
- **高斯模糊去噪**：可選的預處理步驟
- **二值化**：支援手動閾值或 Otsu 自動閾值

### 3. 邊緣檢測
- **Canny 邊緣檢測**：適合細胞膜邊界檢測 (預設)
- **Laplacian 邊緣檢測**：強調細節變化
- **Sobel 邊緣檢測**：方向性邊緣檢測

### 4. 核周圍 ROI 限制
- 以細胞核為中心建立環形 ROI
- 內圈半徑：排除核本身 (預設 5 像素)
- 外圈半徑：膜檢測範圍 (預設 20 像素)

## 檔案結構

```
testing/Her2/
├── Her2.md                    # 工作流程說明
├── her2_mask_core.py         # 核心處理邏輯
├── her2_mask_gui.py          # GUI 介面
├── README.md                 # 本文件
└── output/                   # 輸出資料夾
    ├── membrane/             # 細胞膜相關結果
    ├── nuclei/               # 細胞核相關結果
    ├── combined/             # 組合結果
    └── channels/             # HED 通道圖像
```

## 使用方法

### 1. 啟動 GUI

```bash
cd testing/Her2
python her2_mask_gui.py
```

### 2. 主要功能區塊

#### DAB 通道控制
- **DAB 閾值**：調整細胞膜檢測敏感度 (預設 0.20)
- **使用 Otsu**：自動選擇最佳閾值
- **反轉 DAB**：是否對 DAB 通道取負號 (通常需要)

#### Hematoxylin 通道控制
- **Hematoxylin 閾值**：細胞核檢測敏感度 (預設 0.30)
- **使用 Otsu**：自動選擇細胞核閾值

#### 形態學處理
- **膜處理**：Kernel 大小、開運算、閉運算次數
- **核處理**：針對細胞核的形態學參數

#### 邊緣檢測
- **啟用開關**：是否使用邊緣檢測增強膜結構
- **檢測方法**：Canny、Laplacian、Sobel
- **Canny 參數**：低閾值 (50)、高閾值 (150)

#### 核周圍 ROI
- **內圈半徑**：排除核心區域的半徑
- **外圈半徑**：膜檢測的最大範圍

### 3. 輸出結果

#### 遮罩檔案
- `*_mask_membrane.png`：完整細胞膜遮罩
- `*_mask_membrane_roi.png`：ROI 限制的膜遮罩
- `*_mask_nuclei.png`：細胞核遮罩

#### 疊加圖
- `*_overlay_membrane.png`：膜遮罩疊加在原圖 (紅色)
- `*_overlay_nuclei.png`：核遮罩疊加在原圖 (藍色)
- `*_overlay_combined.png`：膜+核組合疊加

#### 提取結果
- `*_extract_membrane.png`：透明背景的膜提取圖 (RGBA)
- `*_extract_nuclei.png`：透明背景的核提取圖 (RGBA)

#### 通道圖像
- `*_dab_channel.png`：DAB 通道
- `*_hema_channel.png`：Hematoxylin 通道
- `*_edges.png`：邊緣檢測結果

#### 統計報告
- `*_report.csv`：量化分析結果
- `*_params.json`：處理參數設定

## 統計指標

### 覆蓋率統計
- **細胞膜覆蓋率**：膜像素數 / 總像素數 × 100%
- **細胞核覆蓋率**：核像素數 / 總像素數 × 100%
- **膜ROI覆蓋率**：ROI內膜像素數 / 總像素數 × 100%

### DAB 強度分析
- **平均值**：膜區域內 DAB 通道平均強度
- **中位數**：強度分佈中位數
- **百分位數**：25%、75%、95% 百分位數值

## 效能最佳化

### 1. 影像縮放策略
- **工作影像**：25% 縮放進行即時處理
- **輸出影像**：使用原始尺寸確保品質

### 2. 防抖動機制
- GUI 更新延遲 500ms，避免頻繁重新計算
- 背景執行緒處理，維持介面響應性

### 3. 記憶體管理
- 分階段處理大型影像
- 適時釋放中間結果

## 參數調整建議

### 細胞膜檢測不佳時
1. 調低 DAB 閾值 (0.10-0.15)
2. 啟用邊緣檢測
3. 調整 Canny 參數至較低值

### 雜訊過多時
1. 增加形態學開運算次數
2. 提高最小面積過濾
3. 啟用高斯模糊

### 膜連接過度時
1. 增加形態學開運算
2. 減小 ROI 外圈半徑
3. 調高邊緣檢測閾值

## 故障排除

### 常見問題

1. **無法載入 HER2 影像**
   - 確認 `picture/tiff/` 目錄存在
   - 檢查檔名包含 `_Her2_region` 字樣

2. **scikit-image 相關錯誤**
   - 確認已安裝：`pip install scikit-image>=0.19.0`

3. **處理速度過慢**
   - 調整工作影像縮放比例
   - 減少形態學運算次數
   - 關閉邊緣檢測

4. **記憶體不足**
   - 使用較小的工作影像比例
   - 分批處理大型影像

### 開發注意事項

- HED 分離需要 RGB 輸入，確保色彩空間正確
- DAB 通道通常需要反轉處理
- 邊緣檢測參數需根據影像品質調整
- ROI 參數應根據細胞大小調整

## 未來功能

1. **強度分級 (0/1+/2+/3+)**：根據 DAB 強度自動分級
2. **批次處理**：支援多檔案批次分析
3. **機器學習整合**：深度學習輔助膜檢測
4. **3D 分析**：支援厚切片 Z-stack 影像

## 參考資料

- Ruifrok AC, Johnston DA. Quantification of histochemical staining by color deconvolution. Anal Quant Cytol Histol. 2001;23(4):291-9.
- HER2 Testing in Breast Cancer: ASCO/CAP Clinical Practice Guideline