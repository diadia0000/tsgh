# 細胞影像對位系統 (Cell Image Registration System) v2.1

本專案為一個自動化的全組織病理影像 (Whole Slide Image, WSI) 對位系統，採用先進的 `valis` 函式庫，專為處理同一組織切片的三種不同染色影像而設計：
- **HE (H&E)** - 蘇木精-伊紅染色
- **Her2** - HER2 免疫組化染色 (作為對位的基準)
- **DISH** - 雙重原位雜交染色

此系統透過一個高效、模組化的 Python 工作流程，解決因人工染色過程產生的影像偏移、旋轉、縮放及非線性變形，實現像素級的精準對齊。

## 系統特色

### 全自動化 Python 工作流程
本系統以 `valis` 函式庫為核心，搭配 `aicspylibczi 3.3.1` 處理 CZI 格式影像，提供一個從影像對位到結果驗證的完整 Python 解決方案。

### 高效記憶體管理
- **無需載入完整影像**: 系統利用 `pyvips` 函式庫，直接在磁碟上對 Gigapixel 等級的 CZI 檔案進行處理，無需將整個高解析度影像載入記憶體。
- **金字塔層級處理**: 所有耗時的對位運算都在影像金字塔的低解析度層級完成，大幅提升運算速度並降低硬體需求。
- **串流式影像處理**: 使用 pyvips 的 sequential access 模式，實現真正的串流處理。

### GPU 加速支援
- **CUDA 加速**: 支援使用 CUDA 加速特徵偵測與匹配，大幅提升對位速度。
- **LightGlue + DISK**: 採用最新的 LightGlue 匹配器搭配 DISK 特徵偵測器，提供更準確的對位結果。

### 五階段模組化工作流程
1.  **模組 1: 影像對準 (Alignment)** - 使用 valis 內建前處理，計算剛性與非剛性變換參數，並將結果儲存為 `Transform_Params.h5` 檔案。
2.  **模組 2: ROI 品質評估 (ROI Evaluation)** - 從原始高解析度影像中，根據計算出的參數提取已對齊的感興趣區域 (ROI)，並計算 MI、NCC 等品質指標。
3.  **模組 3: 產生對齊縮圖 (Thumbnail Generation)** - 產生一個已對齊的全尺寸疊合縮圖，使用拉普拉斯金字塔融合技術保留細節。
4.  **模組 4: 切割影像磚 (Tile Generation)** - 高效切割對齊後的大型 TIFF 影像為小型磚塊，支援多執行緒並行處理。

## 系統需求

### 硬體需求
- **CPU**: Intel i5 或 AMD Ryzen 5 以上
- **記憶體**: 16GB RAM 以上 (建議 32GB)
- **GPU**: NVIDIA GPU (支援 CUDA，可選但強烈建議)
- **儲存空間**: 至少 50GB 可用空間 (用於存放原始影像與輸出結果)

### 軟體需求
- **作業系統**: Windows 10/11, Linux, macOS
- **Python**: 3.9+
- **CUDA Toolkit**: 11.8+ (若使用 GPU 加速)

## 安裝指南

1.  **克隆專案**
    ```bash
    git clone <repository-url>
    cd tsgh
    ```

2.  **安裝 Python 依賴**
    建議在虛擬環境中安裝，以避免套件版本衝突。
    ```bash
    # 建立虛擬環境 (可選)
    python -m venv venv
    # 啟動虛擬環境
    # Windows
    .\venv\Scripts\activate
    # Linux / macOS
    source venv/bin/activate

    # 安裝所有必要的套件
    pip install -r requirements.txt
    ```

3.  **驗證 CUDA 安裝 (可選)**
    ```python
    import torch
    print(torch.cuda.is_available())  # 應該顯示 True
    ```

## 使用方法

### 快速開始 (推薦)
執行單一腳本即可完成完整工作流程：
```bash
python thriple_image_layer/run_full_pipeline.py
```
此腳本會自動執行從對位到產生最終縮圖的所有步驟。

### 分步執行
您也可以依照需求，單獨執行每個模組的腳本。

#### 模組 1: 影像對準
```bash
python thriple_image_layer/module2_alignment.py
```
- **輸入**: `picture/whole_size/40X/` 目錄下的 `*.czi` 檔案。
- **輸出**: `thriple_image_layer/output/Transform_Params/` (對位參數)
- **特色**: 
  - 自動偵測並使用 GPU 加速
  - 使用 LightGlue + DISK 進行特徵匹配
  - 以 HER2 影像作為參考基準

#### 模組 2: ROI 品質評估
```bash
python thriple_image_layer/module3_roi_evaluation.py
```
- **輸入**: `Transform_Params` 目錄
- **輸出**:
    - `thriple_image_layer/output/Merged_ROI.png` (2048x2048 高解析度對齊區域疊合圖)
    - `thriple_image_layer/output/Metrics.csv` (量化評估指標)
- **特色**: 
  - 從影像中心提取 ROI
  - 計算 NCC 和 MI 指標
  - 生成三色疊合圖 (R=Her2, G=HE, B=DISH)

#### 模組 3: 產生對齊縮圖
```bash
python thriple_image_layer/module4_thumbnail.py
```
- **輸入**: `Transform_Params` 目錄
- **輸出**: `thriple_image_layer/output/Merged_Aligned_lv{level}.tiff` (全局對齊效果縮圖)
- **特色**: 
  - 使用拉普拉斯金字塔融合技術
  - 保留兩張影像的細節
  - 支援金字塔 TIFF 格式
  - 可調整金字塔層級 (level 參數)

#### 模組 4: 切割影像磚
```bash
python thriple_image_layer/module5_tile_generator.py
```
- **輸入**: `Merged_Aligned_lv{level}.tiff`
- **輸出**: 多個 `tile_x{x}_y{y}.tiff` 檔案
- **特色**: 
  - 多執行緒並行處理
  - 可自訂磚塊尺寸
  - 適合後續深度學習分析

## 目錄結構與輸出檔案

### 專案目錄結構
```
tsgh/
├── thriple_image_layer/         # 主要工作流程目錄
│   ├── module2_alignment.py         # 模組1: 影像對準
│   ├── module3_roi_evaluation.py    # 模組2: ROI 品質評估
│   ├── module4_thumbnail.py         # 模組3: 產生對齊縮圖
│   ├── module5_tile_generator.py    # 模組4: 切割影像磚
│   ├── run_full_pipeline.py         # 完整流程執行腳本
│   ├── check_output.py              # 輸出檢查工具
│   └── output/                      # 結果輸出目錄
│       ├── Transform_Params/        # 對位參數目錄
│       │   └── data/
│       │       └── Transform_Params_registrar.pickle
│       ├── temp/                    # 暫存檔案目錄
│       │   ├── dish_warped_lv*.ome.tiff
│       │   └── her2_warped_lv*.ome.tiff
│       ├── Merged_ROI.png           # 高解析度 ROI 疊合圖
│       ├── Metrics.csv              # 量化評估指標
│       └── Merged_Aligned_lv*.tiff  # 全局對齊縮圖
│
├── picture/                     # 影像資料目錄
│   └── whole_size/
│       └── 40X/
│           ├── DISH_40X_2.czi
│           ├── HE_40X.czi
│           └── HER2_40X.czi
│
├── analyze/                     # 影像分析相關文件
│   ├── analysis_20X.txt
│   └── analysis_40X.txt
│
├── .amazonq/                    # AI 輔助開發規則
│   └── rules/
│       ├── aicspylibczi_rule.instructions.md
│       └── copilot-instructions.md
│
├── analyze_czi.py               # CZI 檔案分析工具
├── czi_to_tiff.py               # CZI 轉 TIFF 工具
├── image_aligner.py             # 影像對位工具
├── tiff_to_pyramid.py           # TIFF 金字塔生成工具
├── requirements.txt             # Python 依賴列表
└── README.md                    # 專案說明文件
```

### 輸出檔案說明
- **Transform_Params/**: 包含所有計算出的變換矩陣和位移場，是對位結果的核心，可被後續模組重複使用。
- **Merged_ROI.png**: 一個 2048x2048 像素的高解析度疊合圖，用於精確檢查細胞層級的對位細節。
- **Metrics.csv**: 包含正規化互相關 (NCC) 和互信息 (MI) 指標，量化不同影像間的相似度。
- **Merged_Aligned_lv*.tiff**: 降採樣的全尺寸疊合縮圖，使用拉普拉斯金字塔融合技術保留細節。
- **temp/**: 暫存對齊後的單張影像，避免重複計算。

## 品質評估指標

### 正規化互相關 (NCC)
- **範圍**: [-1, 1]
- **意義**: 衡量兩影像的線性相關性。值越接近 1，表示結構越相似，對位效果越好。
- **應用**: 適合評估相似染色方式的影像對位品質。

### 互信息 (MI)
- **範圍**: [0, +∞)
- **意義**: 衡量兩影像間資訊量的共享程度，特別適用於不同染色 (多模態) 的影像。值越高表示相關性越強，對位品質越好。
- **應用**: 適合評估不同染色方式 (如 HE vs DISH) 的影像對位品質。

## 核心技術

### 影像對位技術
- **特徵偵測**: DISK (Deep Image Structure and Keypoint)
- **特徵匹配**: LightGlue (輕量級圖匹配網路)
- **變換模型**: 剛性變換 + 非剛性變換 (B-spline)
- **參考影像**: HER2 染色影像

### 影像融合技術
- **拉普拉斯金字塔融合**: 多尺度融合技術，保留兩張影像的細節資訊
- **金字塔層級**: 6 層 (可調整)
- **優勢**: 避免簡單平均造成的細節損失

### 記憶體優化技術
- **串流式處理**: pyvips sequential access
- **暫存機制**: 避免重複計算已對齊的影像
- **多執行緒**: 切割影像磚時使用多執行緒並行處理

## 效能優化

本系統的設計已將效能優化考慮在內：
- **GPU 加速**: 使用 CUDA 加速特徵偵測與匹配，速度提升 5-10 倍。
- **磁碟流式處理**: 歸功於 `valis` 和 `pyvips`，即使是數 GB 大小的 CZI 檔案也能在記憶體有限的機器上流暢處理。
- **暫存機制**: 已對齊的影像會暫存在 `temp/` 目錄，避免重複計算。
- **參數化設計**: 所有關鍵步驟，如 ROI 大小、縮圖降採樣率、磚塊尺寸等，都可以在腳本中輕鬆調整。
- **多執行緒支援**: 切割影像磚時支援多執行緒並行處理，充分利用多核心 CPU。

## 故障排除

1.  **套件安裝失敗**
    - 確認 Python 版本為 3.9+。
    - 嘗試更新 `pip` 與 `setuptools`：`pip install --upgrade pip setuptools`。
    - 在 Windows 上，部分套件可能需要 Visual C++ Build Tools。
    - 若 `aicspylibczi` 安裝失敗，請確認已安裝 Microsoft Visual C++ 14.0+。

2.  **CUDA 相關錯誤**
    - 確認已安裝 CUDA Toolkit 11.8+。
    - 確認 PyTorch 版本與 CUDA 版本相容：`torch.cuda.is_available()`。
    - 若無 GPU，系統會自動切換到 CPU 模式，但速度會較慢。

3.  **記憶體不足 (OutOfMemory Error)**
    - 雖然系統經過優化，但在極端情況下仍可能發生。
    - 嘗試在 `module3_roi_evaluation.py` 中減小 `roi_size` 參數 (例如，從 `(2048, 2048)` 降為 `(1024, 1024)`)。
    - 在 `module4_thumbnail.py` 中使用更高的 `level` 參數 (例如，從 `level=1` 改為 `level=4`)。
    - 確保沒有其他耗費大量記憶體的程式正在運行。

4.  **對位品質不佳**
    - 檢查輸入影像品質，確保組織區域清晰可見。
    - 影像內容差異過大 (例如，組織嚴重缺失或變形) 可能導致對位失敗。
    - 可以嘗試調整 `module2_alignment.py` 中的 `num_features` 參數 (預設 2048)。
    - 檢查 `Metrics.csv` 中的 NCC 和 MI 指標，評估對位品質。

5.  **JVM 相關錯誤**
    - 若出現 JVM 初始化錯誤，請確認已安裝 Java Runtime Environment (JRE)。
    - 在腳本結束時會自動清理 JVM，若仍有問題可手動呼叫 `slide_io.kill_jvm()`。

## 進階使用

### 調整對位參數
在 `module2_alignment.py` 中可調整以下參數：
```python
# 特徵點數量 (越多越準確但越慢)
fd = feature_detectors.DiskFD(num_features=2048, device=device)

# 參考影像 (預設為 HER2)
reference_img_name = "HER2_40X.czi"
```

### 調整融合品質
在 `module4_thumbnail.py` 中可調整以下參數：
```python
# 金字塔層級 (越多細節保留越好但越慢)
merged = laplacian_blend(dish_vips, her2_vips, levels=6)

# 輸出解析度 (level 越小解析度越高)
generate_thumbnail(output_dir, level=1)
```

### 調整切割參數
在 `module5_tile_generator.py` 中可調整以下參數：
```python
# 磚塊尺寸
tile_width = 2056
tile_height = 2464

# 執行緒數量 (根據 CPU 核心數調整)
workers = 7
```

## 更新日誌

### v2.1.0 (2025-01-XX)
- **新增模組 5**: 高效切割影像磚功能，支援多執行緒並行處理
- **GPU 加速**: 整合 LightGlue + DISK 特徵匹配，支援 CUDA 加速
- **融合技術升級**: 採用拉普拉斯金字塔融合技術，保留更多細節
- **暫存機制**: 新增暫存目錄，避免重複計算已對齊的影像
- **文件更新**: 更新 README 以反映最新的專案結構和功能
- **依賴更新**: 更新至 aicspylibczi 3.3.1, valis 0.1, valis-wsi 1.2.0

### v2.0.0 (2024-10-17)
- **架構重構**: 專案完全重寫，採用以 `valis` 函式庫為核心的 Python 工作流程
- **模組化設計**: 將流程拆分為四個獨立、可單獨執行的模組
- **移除 C++ 依賴**: 簡化安裝與部署流程，不再需要 C++ 編譯器和相關函式庫
- **效能提升**: 引入基於 `pyvips` 的高效記憶體管理，實現對超大影像的快速處理
- **文件更新**: 全面更新 README 以反映新的架構、使用方法和目錄結構

### v1.1.0 (2025-01-13)
- 重構專案架構，採用模組化設計
- 新增詳細的模組分析文檔
- 優化代碼結構和可維護性
- 改進 README 文檔結構

### v1.0.0 (2024-12-09)
- 實現四階段配準工作流程
- 支援 CUDA 加速
- 新增 Python GUI 查看器
- 完整的品質評估指標
- 支援多種影像格式

## 授權與引用

本專案使用的主要開源函式庫：
- **valis**: Virtual Alignment of pathoLogy Image Series
- **aicspylibczi**: Allen Institute for Cell Science CZI reader
- **pyvips**: Python binding for libvips image processing library

## 聯絡資訊

如有問題或建議，請透過 GitHub Issues 回報。
