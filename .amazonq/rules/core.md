# Core 模組結構分析

## 概述
Core模組是專案的核心功能實現，主要負責圖像配準的核心算法和處理邏輯。

## 文件結構

### 主要組件

#### 圖像配準核心
- `WSIRegistrationCore.cpp` - WSI(Whole Slide Image)配準的核心實現
- `RegistrationStagesCore.cpp` - 配準階段的核心邏輯
- `RegistrationStagesBSpline.cpp` - B樣條配準階段實現
- `RegistrationStagesOptimization.cpp` - 配準優化階段

#### 圖像預處理
- `ImagePreprocessing.cpp` - 通用圖像預處理功能
- `WSIRegistrationPreprocessing.cpp` - WSI專用預處理

#### 配準算法
- `WSIRegistrationStages.cpp` - WSI配準的各個階段管理
- `WSIRegistrationTransforms.cpp` - 配準變換實現

#### 評估與度量
- `RegistrationMetrics.cpp` - 配準質量評估指標
- `WSIRegistrationMetrics.cpp` - WSI專用配準指標

#### 細胞配準
- `CellRegistration.cpp` - 細胞級別的配準功能

## 架構特點
- 模組化設計，各功能分離清晰
- 支持多階段配準流程
- 包含完整的預處理到後處理流程
- 提供豐富的配準評估指標