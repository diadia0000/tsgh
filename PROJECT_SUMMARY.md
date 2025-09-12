# WSI 四階段配準系統 - 項目總結

## 系統概述

本項目實現了一個完整的 WSI (Whole Slide Image) 四階段配準系統，專門用於處理組織病理學影像的多模態配準，包括 H&E 染色、HER2 免疫組化和 FISH 螢光原位雜交影像。

## 核心技術架構

### 四階段配準工作流程

1. **階段 1: 特徵點粗對齊**
   - 使用 SIFT/ORB 特徵檢測器
   - RANSAC 算法進行魯棒性匹配
   - 估計初始仿射變換

2. **階段 2: 互信息精準對齊**
   - 多解析度金字塔優化
   - 基於互信息的相似性度量
   - 仿射變換參數精細調整

3. **階段 3: B-spline 非剛體對齊**
   - B-spline 自由形變 (FFD)
   - 像素級精準對齊
   - 處理局部組織變形

4. **階段 4: 品質評估與驗證**
   - 計算互信息 (MI)
   - 計算正規化互信息 (NMI)
   - 計算目標配準誤差 (TRE)

### CUDA 加速支援

- 自動檢測 GPU 可用性
- CUDA 核心函數並行處理
- 記憶體使用優化
- CPU 回退機制

## 文件結構

```
wsi-registration/
├── CMakeLists.txt                 # CMake 配置文件
├── README.md                      # 詳細使用說明
├── PROJECT_SUMMARY.md             # 項目總結 (本文件)
├── requirements.txt               # Python 依賴
├── build.bat                      # Windows 編譯腳本
├── build.sh                       # Linux/macOS 編譯腳本
├── test_registration.py           # 系統測試腳本
├── wsi_registration_viewer.py     # Python GUI 查看器
├── src/
│   ├── main.cpp                   # 主程式入口
│   ├── core/
│   │   ├── WSIRegistration.h      # 核心配準類標頭檔
│   │   └── WSIRegistration.cpp    # 核心配準實現
│   ├── io/                        # I/O 處理模組
│   └── gpu/                       # GPU 加速模組
└── vcpkg/                         # 依賴管理 (Windows)
```

## 核心類別設計

### WSIRegistration 類

```cpp
class WSIRegistration {
public:
    // 主要工作流程
    bool loadWSI(const std::string& her2_path, 
                 const std::string& he_path, 
                 const std::string& fish_path);
    bool performRegistration();
    bool saveAlignedImages(const std::string& output_dir);
    
    // 品質評估
    double calculateMutualInformation(const cv::Mat& img1, const cv::Mat& img2);
    double calculateNormalizedMutualInformation(const cv::Mat& img1, const cv::Mat& img2);
    double calculateTargetRegistrationError(const cv::Mat& fixed, const cv::Mat& moving, 
                                          const cv::Mat& transform);

private:
    // 四階段配準實現
    StageResult performFeatureBasedCoarseAlignment(...);
    StageResult performMutualInfoAffineAlignment(...);
    StageResult performBSplineNonRigidAlignment(...);
};
```

### 配置參數結構

```cpp
struct RegistrationParams {
    // 基本參數
    std::string referenceType = "HE";
    bool enableCudaAcceleration = true;
    
    // 階段 1: 特徵點參數
    std::string featureDetector = "SIFT";
    int maxFeatures = 5000;
    double ransacThreshold = 3.0;
    
    // 階段 2: 互信息參數
    std::vector<int> pyramidLevels = {4, 2, 1};
    int maxIterationsAffine = 500;
    
    // 階段 3: B-spline 參數
    std::vector<int> gridSpacing = {32, 16, 8};
    int maxIterationsBSpline = 200;
};
```

## Python GUI 功能

### 主要組件

1. **ImageComparisonWidget**
   - 影像前後對比顯示
   - 多種顯示模式 (原始、熱圖、差異圖)
   - 疊合模式與透明度調整

2. **MetricsWidget**
   - 即時計算品質指標
   - 圖表化顯示結果
   - 品質等級評估

3. **WSIRegistrationViewer**
   - 主視窗管理
   - 標籤頁組織
   - 報告匯出功能

### 品質評估指標

- **互信息 (MI)**: 衡量影像間統計相關性
- **正規化互信息 (NMI)**: 範圍 [0,1]，消除影像大小影響
- **目標配準誤差 (TRE)**: 空間配準誤差，單位像素

## 技術特色

### 1. 多模態影像支援
- H&E 明場影像預處理
- HER2 免疫組化適應
- FISH 螢光影像特殊處理

### 2. 魯棒性設計
- 多階段配準確保穩定性
- 錯誤處理與回退機制
- 參數自適應調整

### 3. 效能優化
- CUDA 並行加速
- 多解析度金字塔處理
- 記憶體使用優化

### 4. 使用者友善
- 圖形化結果查看
- 詳細品質報告
- 參數調整指導

## 系統需求

### 硬體需求
- CPU: Intel i5 或 AMD Ryzen 5+
- 記憶體: 8GB+ (建議 16GB)
- GPU: NVIDIA GPU (CUDA 11.0+) - 可選
- 儲存: 2GB+ 可用空間

### 軟體需求
- 作業系統: Windows 10/11, Linux, macOS
- 編譯器: VS2019+, GCC 7.0+, Xcode 12+
- CMake: 3.16+
- Python: 3.8+ (GUI)

## 使用流程

### 1. 編譯系統
```bash
# Windows
build.bat

# Linux/macOS
./build.sh
```

### 2. 執行配準
```bash
./wsi_registration her2.jpg he.jpg fish.jpg --output results/ --gpu
```

### 3. 查看結果
```bash
python wsi_registration_viewer.py
```

### 4. 測試系統
```bash
python test_registration.py
```

## 輸出結果

配準完成後生成以下檔案：
- `reference_HE.jpg` - 基準影像
- `aligned_HER2.jpg` - 對齊後 HER2 影像
- `aligned_FISH.jpg` - 對齊後 FISH 影像
- `overlay_*.jpg` - 各種疊合影像
- `registration_report.txt` - 詳細配準報告

## 品質標準

### 優秀配準 (🟢)
- TRE < 2.0 像素
- NMI > 0.7
- MI > 1.0

### 良好配準 (🟡)
- 2.0 ≤ TRE < 5.0 像素
- 0.5 < NMI ≤ 0.7
- 0.5 < MI ≤ 1.0

### 需要改善 (🔴)
- TRE ≥ 5.0 像素
- NMI ≤ 0.5
- MI ≤ 0.5

## 優化建議

### 參數調整
1. **特徵檢測**
   - 增加 `maxFeatures` 提高精度
   - 調整 `ransacThreshold` 控制魯棒性

2. **互信息優化**
   - 增加 `pyramidLevels` 處理大尺寸影像
   - 調整 `maxIterationsAffine` 平衡速度與精度

3. **B-spline 配準**
   - 減小 `gridSpacing` 提高局部精度
   - 增加 `maxIterationsBSpline` 改善收斂

### 硬體優化
- 使用 NVIDIA GPU 加速
- 增加系統記憶體
- 使用 SSD 提高 I/O 效能

## 故障排除

### 常見問題
1. **CUDA 初始化失敗** - 檢查驅動程式和 CUDA 版本
2. **配準品質不佳** - 調整預處理參數和特徵檢測設定
3. **記憶體不足** - 減少影像解析度或使用更多金字塔層級
4. **處理速度慢** - 啟用 GPU 加速或調整迭代參數

## 未來改進方向

1. **深度學習整合**
   - 使用 CNN 進行特徵提取
   - 端到端配準網路

2. **多尺度處理**
   - WSI 瓦片化處理
   - 分層配準策略

3. **品質控制**
   - 自動品質評估
   - 配準失敗檢測

4. **使用者介面**
   - Web 介面支援
   - 批次處理功能

## 技術貢獻

本系統的主要技術貢獻包括：

1. **完整的四階段配準工作流程**，從粗對齊到精細配準
2. **多模態影像適應性處理**，針對不同染色類型優化
3. **CUDA 加速實現**，顯著提升大型 WSI 處理效能
4. **綜合品質評估體系**，提供多維度配準品質指標
5. **使用者友善的視覺化工具**，便於結果分析和驗證

這個系統為組織病理學影像分析提供了一個強大而靈活的配準解決方案，特別適用於需要高精度多模態影像對齊的研究和臨床應用。