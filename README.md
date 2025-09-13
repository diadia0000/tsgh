# 細胞影像對位系統 (Cell Image Registration System)

本專案為細胞影像對位 (Image Registration) 應用系統，專門處理同一細胞的三種不同染色影像：
- **Her2** - HER2 免疫組化染色
- **HE** - H&E (蘇木精-伊紅) 染色
- **DISH** - 雙重原位雜交染色

由於人工染色過程會產生偏移、旋轉、平移、縮放及非線性變形，本系統採用多階段配準演算法來精確對齊這些影像。

## 系統特色

### 三階段工作流程
1. **Python 影像轉換** - CZI 格式轉換為 TIFF 格式
2. **C++ 配準運算** - 四步驟配準演算法 (特徵點對齊 → 互信息對齊 → B-spline FFD → 品質評估)
3. **PyQt5 GUI 顯示** - 視覺化結果檢視與品質評估

### 配準演算法流程
1. **特徵點粗對齊** - SIFT/ORB + RANSAC
2. **互信息精準對齊** - 互信息 + 仿射變換
3. **B-spline 非剛體對齊** - B-spline FFD 像素級對齊
4. **品質評估** - MI、NMI、TRE 指標計算

### 模組化架構設計

#### Core 核心模組
- **多階段配準流程**: 從粗對齊到精細配準的完整流程
- **WSI專用處理**: 針對大型病理切片影像優化
- **豐富評估指標**: 提供多種配準品質評估方法
- **細胞級配準**: 支持細胞層級的精確配準

#### GPU 加速模組
- **CUDA並行計算**: 利用GPU大幅提升處理速度
- **記憶體優化**: 智能記憶體管理，支持大型影像處理
- **自動回退機制**: GPU不可用時自動切換到CPU處理

#### IO 處理模組
- **多格式支持**: 支援WSI、TIFF、CZI等多種醫學影像格式
- **高效載入**: 針對大型影像優化的分塊載入機制
- **擴展性設計**: 易於添加新的影像格式支持

### 支援的影像類型
- **HE 染色** (基準影像)
- **Her2 免疫組化**
- **DISH 雙重原位雜交**

## 系統需求

### 硬體需求
- **CPU**: Intel i5 或 AMD Ryzen 5 以上
- **記憶體**: 8GB RAM 以上 (建議 16GB)
- **GPU**: NVIDIA GPU (支援 CUDA 11.0+) - 可選但建議
- **儲存空間**: 至少 2GB 可用空間

### 軟體需求
- **作業系統**: Windows 10/11, Linux, macOS
- **編譯器**: 
  - Windows: Visual Studio 2019/2022 或 MinGW
  - Linux: GCC 7.0+
  - macOS: Xcode 12+
- **CMake**: 3.16 或更新版本
- **Python**: 3.8+ (用於 GUI 查看器)

## 安裝指南

### 1. 安裝依賴庫

#### Windows (使用 vcpkg)
```bash
# 安裝 vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# 安裝 OpenCV
.\vcpkg install opencv[contrib,cuda]:x64-windows

# 安裝 CUDA (可選)
# 從 NVIDIA 官網下載並安裝 CUDA Toolkit 11.8+
```

#### Linux (Ubuntu/Debian)
```bash
# 安裝基本依賴
sudo apt update
sudo apt install build-essential cmake git

# 安裝 OpenCV
sudo apt install libopencv-dev libopencv-contrib-dev

# 安裝 CUDA (可選)
# 參考 NVIDIA 官方文檔安裝 CUDA Toolkit
```

### 2. 編譯 C++ 核心

```bash
# 克隆專案
git clone <repository-url>
cd wsi-registration

# 建立編譯目錄
mkdir build
cd build

# 配置 CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 編譯
cmake --build . --config Release

# 執行檔將生成在 dist/ 目錄中
```

### 3. 安裝 Python GUI 依賴

```bash
# 安裝 Python 依賴
pip install -r requirements.txt
```

## 使用方法

### 快速開始 (推薦)

執行完整三階段工作流程：

```bash
# Windows
run_full_workflow.bat

# 或手動執行各階段
```

### 階段 1: Python 影像轉換

```bash
# 轉換 CZI 檔案為 TIFF 格式
python convert_czi_to_tiff.py --input picture/ --output picture/tiff/
```

### 階段 2: C++ 配準運算

```bash
# 編譯專案 (Windows)
build_and_run.bat

# 或手動編譯
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release

# 執行配準
.\dist\Release\wsi_registration.exe --input ..\picture\tiff\ --output ..\picture\output\ --gpu
```

### 階段 3: PyQt5 GUI 查看器

```bash
# 啟動 GUI 查看器
python gui/main.py

# 或先測試 GUI (使用範例資料)
python test_gui.py
```

GUI 功能：
- 三區域佈局：檔案清單、影像顯示、結果資訊
- 基本影像操作：縮放、平移、旋轉
- 比較模式：雙影像疊合顯示
- 三圖合成模式：多通道疊合檢視
- 即時品質評估指標顯示 (MI, NMI, TRE)
- 重新整理功能：無需重啟即可更新結果

### 3. 參數說明

#### 配準參數
- `--transform`: 變換類型 (Affine, BSpline)
- `--metric`: 相似性度量 (MI, NCC, SSD)
- `--pyramid`: 金字塔層級 (例: 4,2,1)
- `--grid`: B-spline 網格間距 (例: 32,16,8)
- `--iterations`: 最大迭代次數

#### GPU 參數
- `--gpu`: 啟用 GPU 加速
- `--device`: GPU 設備 ID (預設: 0)

## 目錄結構與輸出檔案

### 專案目錄結構
```
tsgh/
├── picture/                      # 影像資料目錄
│   ├── *.czi                    # 原始 CZI 檔案
│   ├── tiff/                    # 轉換後的 TIFF 檔案
│   └── output/                  # 配準結果輸出
├── src/                         # C++ 源碼
│   ├── core/                    # 核心配準演算法
│   │   ├── CellRegistration.cpp              # 細胞級別配準
│   │   ├── ImagePreprocessing.cpp            # 圖像預處理
│   │   ├── RegistrationMetrics.cpp           # 配準評估指標
│   │   ├── RegistrationStagesBSpline.cpp     # B樣條配準階段
│   │   ├── RegistrationStagesCore.cpp        # 配準核心邏輯
│   │   ├── RegistrationStagesOptimization.cpp # 配準優化
│   │   ├── WSIRegistrationCore.cpp           # WSI配準核心
│   │   ├── WSIRegistrationMetrics.cpp        # WSI配準指標
│   │   ├── WSIRegistrationPreprocessing.cpp  # WSI預處理
│   │   ├── WSIRegistrationStages.cpp         # WSI配準階段管理
│   │   └── WSIRegistrationTransforms.cpp     # 配準變換實現
│   ├── gpu/                     # CUDA 加速模組
│   │   ├── CudaRegistrationCore.cpp          # CUDA配準核心
│   │   ├── CudaRegistrationUtils.cpp         # CUDA配準工具
│   │   └── CudaUtils.cpp                     # CUDA通用工具
│   ├── io/                      # 檔案 I/O 處理
│   │   ├── WSILoader.cpp                     # WSI圖像載入器
│   │   └── WSILoaderExtensions.cpp           # WSI載入器擴展
│   └── main.cpp                 # 主程式入口
├── gui/                         # PyQt5 GUI 程式
│   ├── main.py                  # GUI 主程式
│   ├── viewer.py                # 影像顯示控制
│   ├── metrics.py               # 評估指標處理
│   └── ui/                      # Qt Designer UI 檔案
├── .amazonq/                    # 專案分析文檔
│   └── rules/                   # 模組結構分析
│       ├── core.md              # 核心模組分析
│       ├── gpu.md               # GPU模組分析
│       └── io.md                # IO模組分析
└── CMakeLists.txt               # CMake 建置檔案
```

### 輸出檔案
配準完成後會在 `picture/output/` 目錄生成：

```
picture/output/
├── aligned_HE.tiff              # 基準 HE 影像
├── aligned_Her2.tiff            # 對齊後的 Her2 影像
├── aligned_DISH.tiff            # 對齊後的 DISH 影像
├── overlay_triple.tiff          # 三通道疊合影像
├── registration_metrics.json    # 評估指標 (JSON 格式)
└── registration_report.txt      # 詳細配準報告
```

## 品質評估指標

### 互信息 (MI)
- **範圍**: [0, +∞)
- **意義**: 值越高表示兩影像間相關性越強，配準品質越好
- **典型值**: 0.5-2.0 為良好配準

### 正規化互信息 (NMI)
- **範圍**: [0, 1]
- **意義**: MI 的正規化版本，消除影像大小影響
- **品質標準**:
  - NMI > 0.7: 優秀
  - 0.5 < NMI ≤ 0.7: 良好
  - NMI ≤ 0.5: 需要改善

### 目標配準誤差 (TRE)
- **單位**: 像素
- **意義**: 配準後的空間誤差，值越小越好
- **品質標準**:
  - TRE < 2.0: 優秀
  - 2.0 ≤ TRE < 5.0: 良好
  - TRE ≥ 5.0: 需要改善

## 效能優化建議

### 1. 硬體優化
- 使用 NVIDIA GPU 進行 CUDA 加速
- 增加系統記憶體以處理大型 WSI
- 使用 SSD 儲存以提高 I/O 效能

### 2. 參數調整
- 對於高解析度影像，增加金字塔層級數
- 調整特徵點數量以平衡精度與速度
- 根據影像特性選擇合適的相似性度量

### 3. 預處理優化
- 對螢光影像進行適當的對比度增強
- 使用色彩正規化改善 H&E 染色一致性
- 適當的去噪處理以提高特徵檢測品質

## 故障排除

### 常見問題

1. **CUDA 初始化失敗**
   - 檢查 NVIDIA 驅動程式版本
   - 確認 CUDA Toolkit 正確安裝
   - 檢查 GPU 記憶體是否足夠

2. **配準品質不佳**
   - 檢查影像預處理是否適當
   - 調整特徵檢測參數
   - 嘗試不同的相似性度量

3. **記憶體不足**
   - 減少影像解析度
   - 調整瓦片大小參數
   - 使用更多的金字塔層級

4. **處理速度慢**
   - 啟用 GPU 加速
   - 減少最大迭代次數
   - 使用較粗的網格間距

## 技術支援

如遇到問題，請提供以下資訊：
- 作業系統版本
- 硬體配置 (CPU, GPU, 記憶體)
- 輸入影像資訊 (大小, 格式, 類型)
- 完整的錯誤訊息
- 使用的參數設定
## 
開發與貢獻

### 專案架構說明
本專案採用模組化設計，詳細的模組分析文檔位於 `.amazonq/rules/` 目錄：
- `core.md` - 核心配準算法模組分析
- `gpu.md` - GPU加速模組分析  
- `io.md` - 圖像IO處理模組分析

### 代碼結構
- **核心算法**: 11個核心組件，涵蓋完整的配準流程
- **GPU加速**: 3個CUDA組件，提供高性能並行計算
- **IO處理**: 2個專用組件，處理各種醫學影像格式



## 更新日誌

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