# HE 染色細胞遮罩工具

本工具專門用於 H&E (Hematoxylin and Eosin) 染色玻片的細胞質與細胞核分離處理。

## 檔案說明

- `he_mask_core.py` - 核心處理邏輯，包含 HSV 分離、形態學處理、面積過濾等功能
- `he_mask_gui_v3.py` - **新版簡化 GUI 介面**，提供互動式參數調整與即時預覽
- `he_gui_utils.py` - GUI 通用工具模組，包含滑桿管理、影像顯示、樣式等可重用元件
- `he_mask_gui_v2.py` - 原版 GUI 介面 (保留，但建議使用 v3)
- `test_he_mask.py` - 快速測試腳本
- `HE.md` - 技術文件，說明處理流程與算法原理

## 功能特色

### 雙重 HSV 分離
- **細胞質 (Eosin)**：支援雙 H 範圍聯集 (H1: 0-20, H2: 160-179)
- **細胞核 (Hematoxylin)**：單一 H 範圍 (H: 0-140)
- 獨立的 S、V 範圍調整

### 形態學處理
- 獨立的細胞質/細胞核 kernel 大小與迭代次數
- 開運算去雜訊 + 閉運算填補空洞
- Gaussian 與 Median 平滑前處理

### 面積過濾
- 基於連通元件的小物件移除
- 獨立的細胞質/細胞核最小面積閾值

### 排他性細胞核提取
- 自動移除與細胞質重疊的細胞核像素
- 專為「只保留細胞核，排除細胞質」需求設計

## 快速開始

### 1. 環境要求
```bash
pip install opencv-python numpy pillow PyQt5
```

### 2. 啟動 GUI
```bash
cd testing/HE
python he_mask_gui_v3.py
```

或使用原版 GUI：
```bash
python he_mask_gui_v2.py
```

### 3. 操作流程
1. **載入影像**：點擊「重新載入 HE」自動載入 `picture/tiff/P2525729F_HE_region.tiff`
2. **調整參數**：
   - **細胞質分頁**：調整 Eosin HSV 範圍與形態學參數
   - **細胞核分頁**：調整 Hematoxylin HSV 範圍與形態學參數  
   - **形態學分頁**：調整平滑處理參數
3. **即時預覽**：使用「顯示模式」切換不同結果檢視
4. **儲存結果**：點擊「儲存所有結果」輸出到 `testing/HE/output`

## 輸出結果

### 檔案結構
```
testing/HE/output/
├── cytoplasm/
│   ├── P2525729F_HE_region_mask_cyto.png           # 細胞質遮罩
│   ├── P2525729F_HE_region_overlay_cyto.png        # 細胞質疊加圖
│   └── P2525729F_HE_region_extract_cyto.png        # 細胞質提取 (RGBA)
├── nuclei/
│   ├── P2525729F_HE_region_mask_nuclei.png         # 細胞核遮罩
│   └── P2525729F_HE_region_overlay_nuclei.png      # 細胞核疊加圖
├── nuclei_exclusive/
│   ├── P2525729F_HE_region_mask_nuclei_exclusive.png    # 排他核遮罩
│   ├── P2525729F_HE_region_overlay_nuclei_exclusive.png # 排他核疊加圖
│   └── P2525729F_HE_region_extract_nuclei_exclusive.png # 排他核提取 (RGBA)
└── P2525729F_HE_region_params.json                 # 參數記錄
```

### 檔案類型
- **遮罩檔案**：8-bit PNG 灰階 (0=背景, 255=前景)
- **疊加檔案**：24-bit PNG 彩色 (原圖+半透明遮罩)
- **提取檔案**：32-bit PNG RGBA (透明背景+保留區域)
- **參數檔案**：JSON 格式，記錄所有滑桿設定

## 參數說明

### 細胞質 (Eosin) 預設值
- H1 範圍：0-20 (紅色區域)
- H2 範圍：160-179 (紅色區域)
- S 範圍：30-150
- V 範圍：80-255
- Kernel 大小：3
- 開/閉運算：各 1 次
- 最小面積：50 像素

### 細胞核 (Hematoxylin) 預設值
- H 範圍：0-140 (藍紫色區域)
- S 範圍：0-255
- V 範圍：0-255
- Kernel 大小：3
- 開/閉運算：各 1 次
- 最小面積：20 像素

### 平滑處理預設值
- Gaussian Kernel：5×5
- Median Kernel：3×3

## 效能考量

- **記憶體優化**：顯示時使用 50% 縮小影像，輸出時用全尺寸處理
- **即時更新**：1ms 防抖動，確保滑桿調整流暢
- **背景處理**：多執行緒處理，避免 UI 凍結

## 快捷鍵

- `Ctrl+S`：儲存結果
- `Ctrl+R`：重置參數
- `Ctrl+Q`：退出程式

## 注意事項

1. **HSV 雙範圍**：細胞質使用雙 H 範圍 (H1+H2) 聯集，適合粉紅色 Eosin 染色
2. **排他性處理**：「排他核遮罩」會自動排除與細胞質重疊的像素
3. **參數儲存**：建議在找到最佳參數後儲存為 JSON 檔案以便重複使用
4. **影像尺寸**：支援大影像處理，自動進行記憶體優化

## 疑難排解

- **無法載入影像**：檢查 `picture/tiff/` 目錄是否存在 HE 影像檔案
- **處理過慢**：減少 Kernel 大小或迭代次數
- **結果不佳**：嘗試調整 HSV 範圍，特別是 H 範圍的邊界值
- **記憶體不足**：確保系統有足夠 RAM 處理大型 TIFF 檔案