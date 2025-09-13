# GPU 模組結構分析

## 概述
GPU模組提供CUDA加速的圖像配準功能，利用GPU並行計算能力提升配準性能。

## 文件結構

### 主要組件

#### CUDA配準核心
- `CudaRegistrationCore.cpp` - CUDA版本的配準核心實現
- `CudaRegistrationUtils.cpp` - CUDA配準相關的工具函數

#### CUDA工具
- `CudaUtils.cpp` - 通用CUDA工具函數和輔助功能

## 架構特點
- 專門針對GPU加速優化
- 提供與CPU版本對應的CUDA實現
- 包含CUDA記憶體管理和錯誤處理
- 支持大規模並行圖像處理

## 性能優勢
- 利用GPU並行計算能力
- 大幅提升大尺寸圖像配準速度
- 適合處理WSI等大型醫學圖像