# 醫療影像配準 Pipeline

## 概述
這是一個用於對齊兩張醫療影像（DISH 和 HER2）的配準系統，支援從縮圖（PNG）到原圖（CZI）的參數轉換。

## 系統設計

### 座標空間轉換流程
```
移動影像(HER2) CZI 座標
    ↓ (縮小 5.0x)
移動影像(HER2) PNG 座標
    ↓ (應用配準變換 H_png)
參考影像(DISH) PNG 座標
    ↓ (放大 4.0x)
參考影像(DISH) CZI 座標
```

### 數學公式
完整的變換矩陣計算：
```
H_czi = S_ref × H_png × S_mov^(-1)
```

其中：
- `S_ref`: 參考影像的縮放矩陣 (4.0x)
- `H_png`: PNG 空間的配準變換矩陣
- `S_mov^(-1)`: 移動影像的逆縮放矩陣 (1/5.0x)

## 配準方法

### 1. SIFT (推薦)
- **優點**: 高精度，尺度不變，旋轉不變
- **缺點**: 計算較慢
- **適用**: 影像有明顯特徵點

### 2. ORB
- **優點**: 快速，免費
- **缺點**: 精度略低於 SIFT
- **適用**: 快速驗證

### 3. ECC (Enhanced Correlation Coefficient)
- **優點**: 適合灰度配準
- **缺點**: 需要良好的初始對齊
- **適用**: 已大致對齊的影像

## 使用方法

### 安裝依賴
```bash
pip install -r requirements.txt
```

### 執行配準
```bash
cd thriple_image_layer
python image_registration_pipeline.py
```

### 互動選擇
程式會提示選擇配準方法：
- 輸入 `1`: SIFT (推薦)
- 輸入 `2`: ORB
- 輸入 `3`: ECC
- 直接按 Enter: 使用預設 SIFT

## 輸出檔案

### PNG 配準階段

#### 1. Her2_aligned_to_DISH.png
配準後的 HER2 影像（對齊到 DISH 座標系，PNG 解析度）

#### 2. registration_results.json
完整的配準參數，包含：

```json
{
  "metadata": {
    "timestamp": "配準時間",
    "registration_method": "使用的方法"
  },
  "image_info": {
    "reference": {
      "png_size": {"width": 40314, "height": 30544},
      "czi_size": {"width": 161259, "height": 122176},
      "scale_factor": 4.0
    },
    "moving": {
      "png_size": {"width": 31877, "height": 24444},
      "czi_size": {"width": 159388, "height": 122224},
      "scale_factor": 5.0
    }
  },
  "transformation": {
    "png_space": {
      "matrix": [[...], [...], [...]],
      "description": "用於 PNG 影像的 3x3 變換矩陣"
    },
    "czi_space": {
      "matrix": [[...], [...], [...]],
      "description": "用於 CZI 原圖的 3x3 變換矩陣 (完整解析度)"
    }
  },
  "quality_metrics": {
    "MSE": "均方誤差",
    "PSNR_dB": "峰值信噪比",
    "SSIM": "結構相似性 (0-1，越接近1越好)",
    "NCC": "正規化互相關"
  }
}
```

#### 3. registration_visualization.png
PNG 層級的視覺化對比

### CZI 驗證階段

#### 4. Her2_aligned_to_DISH_CZI_scaled.png
配準後的 HER2 CZI 影像（縮放版本，例如 0.2x）

#### 5. czi_validation_results.png
CZI 層級的詳細視覺化對比，包含：
- 原始 CZI 影像對比
- 棋盤格疊合（多種尺度）
- 色彩疊合（紅-綠）
- 邊緣對齊檢查
- 局部放大檢查

#### 6. czi_validation_results.json
CZI 驗證的品質指標和縮放參數

## 如何使用輸出的參數

### 關鍵概念: 多層級參數轉換

系統產生三組變換矩陣：

1. **H_png**: PNG 空間的變換矩陣
2. **H_czi_full**: CZI 完整解析度的變換矩陣
3. **H_czi_scaled**: CZI 縮放版本的變換矩陣 (例如 0.2x)

### 參數轉換公式

```
# PNG -> CZI 完整解析度
H_czi_full = S_ref × H_png × S_mov^(-1)

# CZI 完整解析度 -> CZI 縮放版本
H_czi_scaled = S_scale × H_czi_full × S_scale^(-1)

其中:
- S_ref = diag(4.0, 4.0, 1.0)      # DISH PNG->CZI 縮放
- S_mov = diag(5.0, 5.0, 1.0)      # HER2 PNG->CZI 縮放
- S_scale = diag(0.2, 0.2, 1.0)    # CZI 載入縮放 (例如)
```

### Python 使用範例

#### 1. 在 PNG 上應用 (快速測試)
```python
import cv2
import numpy as np
import json

# 載入參數
with open('registration_results.json', 'r') as f:
    results = json.load(f)

# 獲取 PNG 變換矩陣
H_png = np.array(results['transformation']['png_space']['matrix'])

# 載入 PNG 影像
moving_png = cv2.imread('Her2_mask.png', cv2.IMREAD_GRAYSCALE)
ref_png = cv2.imread('DISH_mask.png', cv2.IMREAD_GRAYSCALE)

# 應用變換
aligned_png = cv2.warpPerspective(
    moving_png, 
    H_png, 
    (ref_png.shape[1], ref_png.shape[0])
)
```

#### 2. 在 CZI 完整解析度上應用 (記憶體需求大)
```python
from aicspylibczi import CziFile
import cv2
import numpy as np
import json

# 載入參數
with open('registration_results.json', 'r') as f:
    results = json.load(f)

# 獲取 CZI 完整解析度變換矩陣
H_czi = np.array(results['transformation']['czi_space']['matrix'])

# 載入 CZI 影像 (完整解析度 - 需要大量記憶體!)
czi_mov = CziFile('HER2.czi')
bbox_mov = czi_mov.get_mosaic_bounding_box()
mov_image = czi_mov.read_mosaic(
    (bbox_mov.x, bbox_mov.y, bbox_mov.w, bbox_mov.h),
    scale_factor=1.0,  # 完整解析度
    C=0
)

czi_ref = CziFile('DISH.czi')
bbox_ref = czi_ref.get_mosaic_bounding_box()

# 應用變換
aligned_czi = cv2.warpPerspective(
    mov_image, 
    H_czi, 
    (bbox_ref.w, bbox_ref.h)
)
```

#### 3. 在 CZI 縮放版本上應用 (推薦)
```python
from aicspylibczi import CziFile
import cv2
import numpy as np
import json

# 載入參數
with open('registration_results.json', 'r') as f:
    results = json.load(f)

# 獲取 CZI 完整解析度矩陣
H_czi_full = np.array(results['transformation']['czi_space']['matrix'])

# 設定縮放比例
scale = 0.2

# 計算縮放後的變換矩陣
S = np.array([
    [scale, 0, 0],
    [0, scale, 0],
    [0, 0, 1]
])
S_inv = np.linalg.inv(S)
H_czi_scaled = S @ H_czi_full @ S_inv

# 載入 CZI 影像 (縮放版本)
czi_mov = CziFile('HER2.czi')
bbox_mov = czi_mov.get_mosaic_bounding_box()
mov_image = czi_mov.read_mosaic(
    (bbox_mov.x, bbox_mov.y, bbox_mov.w, bbox_mov.h),
    scale_factor=scale,
    C=0
)

czi_ref = CziFile('DISH.czi')
bbox_ref = czi_ref.get_mosaic_bounding_box()
ref_image = czi_ref.read_mosaic(
    (bbox_ref.x, bbox_ref.y, bbox_ref.w, bbox_ref.h),
    scale_factor=scale,
    C=0
)

# 應用變換
aligned_czi = cv2.warpPerspective(
    mov_image, 
    H_czi_scaled, 
    (ref_image.shape[1], ref_image.shape[0])
)
```

### C++ (OpenCV)
```cpp
#include <opencv2/opencv.hpp>
#include <fstream>
#include "json.hpp"  // 使用 nlohmann/json

using json = nlohmann::json;

// 載入參數
std::ifstream f("registration_results.json");
json results = json::parse(f);

// 獲取 CZI 變換矩陣
auto matrix = results["transformation"]["czi_space"]["matrix"];
cv::Mat H = (cv::Mat_<double>(3, 3) << 
    matrix[0][0], matrix[0][1], matrix[0][2],
    matrix[1][0], matrix[1][1], matrix[1][2],
    matrix[2][0], matrix[2][1], matrix[2][2]);

// 載入 CZI 影像
cv::Mat moving_czi = ...; // 159388 x 122224
cv::Mat aligned_czi;

// 應用變換
cv::warpPerspective(moving_czi, aligned_czi, H, 
                    cv::Size(161259, 122176),
                    cv::INTER_LINEAR);
```

## 品質評估指標

### SSIM (Structural Similarity Index)
- 範圍: 0-1
- **> 0.9**: 優秀
- **0.7-0.9**: 良好
- **< 0.7**: 需要改進

### PSNR (Peak Signal-to-Noise Ratio)
- 單位: dB
- **> 30 dB**: 優秀
- **20-30 dB**: 良好
- **< 20 dB**: 較差

### 特徵匹配
- **good_matches**: 良好匹配點數量
- **ransac_inliers**: RANSAC 內點數量（越多越好）

## 進階使用

### 批次處理
修改 `main()` 函數以處理多組影像：

```python
image_pairs = [
    ("DISH_mask.png", "Her2_mask.png"),
    ("DISH_mask.png", "HE_mask.png"),
    # 更多配對...
]

for ref, mov in image_pairs:
    registrator = ImageRegistration(ref, mov, ...)
    # ... 執行配準
```

### 視覺化結果
```python
import matplotlib.pyplot as plt

# 載入影像
ref = cv2.imread("DISH_mask.png", cv2.IMREAD_GRAYSCALE)
aligned = cv2.imread("Her2_aligned_to_DISH.png", cv2.IMREAD_GRAYSCALE)

# 顯示疊合結果
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(ref, cmap='gray')
plt.title('DISH (Reference)')
plt.subplot(132)
plt.imshow(aligned, cmap='gray')
plt.title('HER2 (Aligned)')
plt.subplot(133)
plt.imshow(ref, cmap='Reds', alpha=0.5)
plt.imshow(aligned, cmap='Greens', alpha=0.5)
plt.title('Overlay')
plt.show()
```

## 疑難排解

### 問題: 匹配點數量不足
**解決方案**:
- 增加 `max_features` 參數
- 嘗試不同的配準方法
- 檢查影像是否有足夠的紋理資訊

### 問題: ECC 配準失敗
**解決方案**:
- 先使用 SIFT/ORB 獲得初始對齊
- 減少迭代次數
- 調整 `gaussFiltSize` 參數

### 問題: 記憶體不足
**解決方案**:
- 使用更小的 `max_features`
- 對 CZI 使用較低的 scale_factor
- 分塊處理大型影像

## 轉換到 C++ 的建議

1. **使用 OpenCV C++ API**: 所有函數都有對應的 C++ 版本
2. **JSON 解析**: 使用 `nlohmann/json` 或 `RapidJSON`
3. **效能優化**: 
   - 使用 `cv::parallel_for_` 平行化
   - 考慮 GPU 加速 (`cv::cuda`)
4. **記憶體管理**: 
   - 使用智慧指標 (`std::unique_ptr`, `std::shared_ptr`)
   - 及時釋放大型矩陣

## 下一步

1. **三影像配準**: 將 HE 影像加入配準流程
2. **影像融合**: 實作多通道融合顯示
3. **ROI 分析**: 在對齊後的影像上進行區域分析
4. **效能優化**: 使用 GPU 加速或多執行緒處理

## 參考資料

- OpenCV Documentation: https://docs.opencv.org/
- SIFT Paper: Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints"
- Image Registration Review: Zitova, B., & Flusser, J. (2003). "Image registration methods: a survey"
