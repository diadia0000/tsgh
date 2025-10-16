# 細胞影像對位系統 (Cell Image Registration System) v2.0

本專案為一個自動化的全組織病理影像 (Whole Slide Image, WSI) 對位系統，採用先進的 `valis` 函式庫，專為處理同一組織切片的三種不同染色影像而設計：
- **HE (H&E)** - 蘇木精-伊紅染色 (通常作為對位的基準)
- **Her2** - HER2 免疫組化染色
- **DISH** - 雙重原位雜交染色

此系統透過一個高效、模組化的 Python 工作流程，解決因人工染色過程產生的影像偏移、旋轉、縮放及非線性變形，實現像素級的精準對齊。

## 系統特色

### 全自動化 Python 工作流程
本系統以 `valis` 函式庫為核心，取代了傳統複雜的多語言環境，提供一個從影像前處理到結果驗證的完整 Python 解決方案。

### 高效記憶體管理
- **無需載入完整影像**: 系統利用 `pyvips` 函式庫，直接在磁碟上對 Gigapixel 等級的 CZI 檔案進行處理，無需將整個高解析度影像載入記憶體。
- **金字塔層級處理**: 所有耗時的對位運算都在影像金字塔的低解析度層級完成，大幅提升運算速度並降低硬體需求。

### 四階段模組化工作流程
1.  **模組 1: 影像前處理 (Preprocessing)** - 在低解析度下讀取影像，進行灰階正規化並產生組織遮罩，為對準做準備。
2.  **模組 2: 計算對位參數 (Alignment)** - 使用低解析度影像計算剛性與非剛性變換參數，並將結果儲存為一個輕量的 `Transform_Params.h5` 檔案。
3.  **模組 3: ROI 品質評估 (ROI Evaluation)** - 從原始高解析度影像中，根據計算出的參數提取已對齊的感興趣區域 (ROI)，並計算 MI、NCC 等品質指標。
4.  **模組 4: 產生對齊縮圖 (Thumbnail Generation)** - 產生一個已對齊的全尺寸三色疊合縮圖，用於快速、直觀地評估全局對位效果。

## 系統需求

### 硬體需求
- **CPU**: Intel i5 或 AMD Ryzen 5 以上
- **記憶體**: 16GB RAM 以上
- **儲存空間**: 至少 10GB 可用空間 (用於存放原始影像與輸出結果)

### 軟體需求
- **作業系統**: Windows 10/11, Linux, macOS
- **Python**: 3.8+

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

## 使用方法

### 快速開始 (推薦)
執行單一腳本即可完成完整四階段工作流程：
```bash
python thriple_image_layer/run_full_pipeline.py
```
此腳本會自動執行從前處理到產生最終縮圖的所有步驟。

### 分步執行
您也可以依照需求，單獨執行每個模組的腳本。

#### 階段 1: 影像前處理
```bash
python thriple_image_layer/module1_preprocessing.py
```
- **輸入**: `picture/whole_size/40X/` 目錄下的 `*.czi` 檔案。
- **輸出**: 產生供後續模組使用的內部暫存檔案。

#### 階段 2: 計算對位參數
```bash
python thriple_image_layer/module2_alignment.py
```
- **輸入**: 模組 1 的處理結果。
- **輸出**: `thriple_image_layer/output/Transform_Params.h5`

#### 階段 3: ROI 品質評估
```bash
python thriple_image_layer/module3_roi_evaluation.py
```
- **輸入**: `Transform_Params.h5`
- **輸出**:
    - `thriple_image_layer/output/Merged_ROI.png` (高解析度對齊區域疊合圖)
    - `thriple_image_layer/output/Metrics.csv` (量化評估指標)

#### 階段 4: 產生對齊縮圖
```bash
python thriple_image_layer/module4_thumbnail.py
```
- **輸入**: `Transform_Params.h5`
- **輸出**: `thriple_image_layer/output/Merged_Thumbnail_12.5pct.png` (全局對齊效果縮圖)


## 目錄結構與輸出檔案

### 專案目錄結構
```
tsgh/
├── thriple_image_layer/         # 主要工作流程目錄
│   ├── module1_preprocessing.py     # 模組1: 影像前處理
│   ├── module2_alignment.py         # 模組2: 計算對位參數
│   ├── module3_roi_evaluation.py    # 模組3: ROI 品質評估
│   ├── module4_thumbnail.py         # 模組4: 產生對齊縮圖
│   ├── run_full_pipeline.py         # 完整流程執行腳本
│   └── output/                      # 結果輸出目錄
│       ├── Transform_Params.h5      # 對位參數檔案
│       ├── Merged_ROI.png           # 高解析度 ROI 疊合圖
│       ├── Metrics.csv              # 量化評估指標
│       └── Merged_Thumbnail_12.5pct.png # 全局對齊縮圖
│
├── picture/                     # 影像資料目錄
│   └── whole_size/
│       └── 40X/
│           ├── DISH_40X_2.czi
│           ├── HE_40X.czi
│           └── HER2_40X.czi
│
├── analyze/                     # 影像分析相關文件
│   └── analysis_40X.txt
│
└── requirements.txt             # Python 依賴列表
```

### 輸出檔案說明
- **Transform_Params.h5**: 包含所有計算出的變換矩陣和位移場，是對位結果的核心，可被後續模組重複使用。
- **Merged_ROI.png**: 一個 2048x2048 像素的高解析度疊合圖，用於精確檢查細胞層級的對位細節。
- **Metrics.csv**: 包含正規化互相關 (NCC) 和互信息 (MI) 指標，量化 HE 影像與另外兩張影像的相似度。
- **Merged_Thumbnail_12.5pct.png**: 降採樣 8 倍的全尺寸疊合縮圖，用於快速概覽全局對位的準確性。

## 品質評估指標

### 正規化互相關 (NCC)
- **範圍**: [-1, 1]
- **意義**: 衡量兩影像的線性相關性。值越接近 1，表示結構越相似，對位效果越好。

### 互信息 (MI)
- **範圍**: [0, +∞)
- **意義**: 衡量兩影像間資訊量的共享程度，特別適用於不同染色 (多模態) 的影像。值越高表示相關性越強，對位品質越好。

## 效能優化
本系統的設計已將效能優化考慮在內：
- **磁碟流式處理**: 歸功於 `valis` 和 `pyvips`，即使是數 GB 大小的 CZI 檔案也能在記憶體有限的機器上流暢處理。
- **參數化設計**: 所有關鍵步驟，如 ROI 大小、縮圖降採樣率等，都可以在腳本中輕鬆調整，以在速度和精度之間取得平衡。
- **單一解析度層級優化**: 即使 CZI 檔案僅提供單一最高解析度層級，`valis` 也能在執行期間動態生成所需的低解析度版本進行運算，確保流程高效運行。

## 故障排除

1.  **套件安裝失敗**
    - 確認 Python 版本符合 `requirements.txt` 中某些套件的要求。
    - 嘗試更新 `pip` 與 `setuptools`：`pip install --upgrade pip setuptools`。
    - 在 Windows 上，部分套件可能需要 Visual C++ Build Tools。

2.  **記憶體不足 (OutOfMemory Error)**
    - 雖然系統經過優化，但在極端情況下仍可能發生。
    - 嘗試在 `module3_roi_evaluation.py` 中減小 `get_aligned_roi` 的 `size` 參數 (例如，從 `(2048, 2048)` 降為 `(1024, 1024)`)。
    - 確保沒有其他耗費大量記憶體的程式正在運行。

3.  **對位品質不佳**
    - 檢查 `module1_preprocessing.py` 中的組織遮罩是否正確生成。不正確的遮罩會嚴重影響特徵提取。
    - 影像內容差異過大 (例如，組織嚴重缺失或變形) 可能導致 `valis` 自動排序失敗。可以嘗試在 `valis.Valis` 初始化時手動指定 `reference_img_f`。

## 更新日誌

### v2.0.0 (2025-10-17)
- **架構重構**: 專案完全重寫，採用以 `valis` 函式庫為核心的 Python 工作流程。
- **模組化設計**: 將流程拆分為四個獨立、可單獨執行的模組。
- **移除 C++ 依賴**: 簡化安裝與部署流程，不再需要 C++ 編譯器和相關函式庫。
- **效能提升**: 引入基於 `pyvips` 的高效記憶體管理，實現對超大影像的快速處理。
- **文件更新**: 全面更新 README 以反映新的架構、使用方法和目錄結構。

### v1.1.0 (2025-01-13)
- 重構專案架構，採用模組化設計
- 新增詳細的模組分析文檔
- 優化代碼結構和可維護性
- 改進README文檔結構

### v1.0.0 (2024-12-09)
- 實現四階段配準工作流程
- 支援 CUDA 加速
- 新增 Python GUI 查看器
- 完整的品質評估指標
- 支援多種影像格式
