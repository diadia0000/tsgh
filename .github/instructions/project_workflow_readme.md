# 細胞影像對位系統 (Cell Image Registration System)

## 專案概述

本專案為細胞影像對位 (Image Registration) 應用系統，專門處理同一細胞的三種不同染色影像：
- **Her2** - HER2 免疫組化染色
- **HE** - H&E (蘇木精-伊紅) 染色  
- **DISH** - 雙重原位雜交染色

由於人工染色過程會產生偏移、旋轉、平移、縮放及非線性變形，本系統採用多階段配準演算法來精確對齊這些影像。

## 系統架構

### 三階段工作流程

#### 1. Python 影像轉換階段
- **輸入**: `picture/` 目錄下的 `.czi` 格式檔案
- **處理**: 將 CZI 檔案轉換為 TIFF 格式
- **輸出**: `picture/tiff/` 目錄下的 TIFF 檔案

#### 2. C++ 配準運算階段
- **輸入**: `picture/tiff/` 目錄下的 TIFF 檔案
- **基準影像**: HE 染色影像作為配準基準
- **配準流程**:
  1. **特徵點粗對齊**: SIFT/ORB + RANSAC
  2. **互資訊精準對齊**: 互資訊 + 仿射變換
  3. **非剛體對齊**: B-spline FFD (自由形變)
  4. **品質評估**: MI、NMI、TRE 指標計算
- **CUDA 加速**: 所有運算優先使用 CUDA 加速
- **輸出**: `picture/output/` 目錄下的對齊結果

#### 3. PyQt5 GUI 顯示階段
- **輸入**: `picture/output/` 目錄下的處理結果
- **功能**: 視覺化顯示、品質評估、互動操作
- **介面**: 三區域佈局 (檔案清單、影像顯示、結果資訊)

## 目錄結構

```
tsgh/
├── picture/                    # 影像資料目錄
│   ├── *.czi                  # 原始 CZI 檔案
│   ├── tiff/                  # 轉換後的 TIFF 檔案
│   └── output/                # 配準結果輸出
├── src/                       # C++ 原始碼
│   ├── core/                  # 核心配準演算法
│   ├── gpu/                   # CUDA 加速模組
│   ├── io/                    # 檔案 I/O 處理
│   ├── python/                # Python 介面模組
│   └── main.cpp               # 主程式入口
├── gui/                       # PyQt5 GUI 程式
│   ├── main.py                # GUI 主程式
│   ├── viewer.py              # 影像顯示控制
│   ├── metrics.py             # 評估指標處理
│   └── ui/                    # Qt Designer UI 檔案
├── CMakeLists.txt             # CMake 建置檔案
├── requirements.txt           # Python 依賴套件
└── README.md                  # 專案說明文件
```

## 系統需求

### 硬體需求
- **CPU**: Intel i5 或 AMD Ryzen 5 以上
- **記憶體**: 16GB RAM 以上 (建議 32GB)
- **GPU**: NVIDIA GPU 支援 CUDA 11.0+ (建議 RTX 3060 以上)
- **儲存**: 10GB 可用空間

### 軟體需求
- **作業系統**: Windows 10/11 (64-bit)
- **編譯器**: Visual Studio 2019/2022
- **CMake**: 3.16+
- **Python**: 3.8+
- **CUDA Toolkit**: 11.8+

## 安裝與建置

### 1. 環境準備

```bash
# 安裝 vcpkg 套件管理器
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# 安裝 OpenCV (含 CUDA 支援)
.\vcpkg install opencv[contrib,cuda]:x64-windows
```

### 2. 編譯 C++ 核心

```bash
# 建立建置目錄
mkdir build
cd build

# 配置 CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake

# 編譯
cmake --build . --config Release
```

### 3. 安裝 Python 依賴

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 影像轉換 (Python)

```python
# 執行 CZI 到 TIFF 轉換
python convert_czi_to_tiff.py --input picture/ --output picture/tiff/
```

### 2. 配準運算 (C++)

```bash
# 執行配準演算法
./build/dist/Release/wsi_registration --input picture/tiff/ --output picture/output/ --gpu
```

### 3. GUI 查看器 (PyQt5)

```bash
# 啟動 GUI 介面
python gui/main.py
```
