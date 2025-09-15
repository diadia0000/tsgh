# DISH 染色細胞遮罩工具使用說明 (架構分離版)

## 功能概述
這是一個基於 PyQt5 + OpenCV 的互動式細胞遮罩工具，專門用於 DISH 染色影像的細胞分離與標記。
採用模組化設計，將 UI 介面與核心邏輯完全分離。

## 檔案結構
```
testing/
├── dish_mask_core.py      # 核心處理邏輯 (無 UI 依賴)
├── dish_mask_gui_v2.py    # PyQt5 圖形介面
├── dish_mask_cli.py       # 命令列介面
└── README_dish_mask.md    # 說明文件
```

## 安裝需求
```bash
# 使用專案虛擬環境
pip install opencv-python PyQt5 numpy
```

## 使用方式

### 1. 圖形介面版本 (推薦)
```bash
cd E:\Class\tsgh
.\.venv\Scripts\python.exe testing/dish_mask_gui_v2.py
```

### 2. 命令列版本
```bash
# 使用預設參數自動載入 DISH 影像
.\.venv\Scripts\python.exe testing/dish_mask_cli.py

# 指定輸入檔案和參數
.\.venv\Scripts\python.exe testing/dish_mask_cli.py --input picture/tiff/P2525729F_DISH_region.tiff --h-min 20 --h-max 160 --output my_results/

# 高品質模式 (使用原圖處理)
.\.venv\Scripts\python.exe testing/dish_mask_cli.py --high-quality

# 檢視所有參數說明
.\.venv\Scripts\python.exe testing/dish_mask_cli.py --help
```

### 3. 程式模組使用
```python
from dish_mask_core import DishMaskProcessor, MaskingParams

# 建立處理器
processor = DishMaskProcessor()

# 載入影像
processor.load_dish_from_directory()  # 自動載入
# 或
processor.load_image("path/to/image.tiff")  # 指定檔案

# 設定參數
params = MaskingParams(h_min=20, h_max=160, s_min=50)

# 執行處理
result = processor.process_mask(params)

# 儲存結果
processor.save_results(result, "output_directory")
```

## 架構優勢

### 核心邏輯分離 (`dish_mask_core.py`)
- **無 UI 依賴**: 可獨立運行，易於測試和整合
- **資料結構化**: 使用 `dataclass` 管理參數和結果
- **多尺度處理**: 自動選擇最適合的處理解析度
- **記憶體優化**: 智能縮放策略避免大圖卡頓
- **錯誤處理**: 完善的例外處理機制

### GUI 介面分離 (`dish_mask_gui_v2.py`)
- **背景處理**: 使用 `QThread` 避免 UI 凍結
- **防抖動機制**: 滑桿調整時避免頻繁重新計算
- **即時預覽**: 透明度調整等不需重新處理的操作即時更新
- **進度回饋**: 清楚的狀態訊息和錯誤提示
- **檔案管理**: 支援自動載入和手動選擇檔案

### 命令列介面 (`dish_mask_cli.py`)
- **批次處理**: 適合自動化工作流程
- **參數化**: 所有處理參數都可透過命令列指定
- **輸出控制**: 靈活的輸入輸出路徑設定
- **品質選項**: 可選快速模式或高品質模式

## 效能最佳化

### 智能縮放策略
- **工作圖像**: 自動縮放至適合大小進行運算 (預設最大 1000px)
- **顯示分離**: UI 顯示與運算解析度獨立
- **高品質輸出**: 儲存時可選擇基於原圖重新計算

### 記憶體管理
- **按需載入**: 只載入必要的圖像資料
- **結果快取**: 避免重複計算相同參數
- **垃圾回收**: 適時釋放不需要的記憶體

### 並行處理
- **背景執行緒**: UI 和處理邏輯分離執行
- **非阻塞 UI**: 處理期間 UI 保持響應
- **中斷機制**: 支援取消長時間運算

## 輸出檔案

所有版本都會產生相同的輸出結果：

```
檔名前綴_01_HSV遮罩.png              # 初始 HSV 顏色篩選
檔名前綴_02_形態學清理.png            # 形態學處理後結果  
檔名前綴_03_確定背景.png              # Watershed 確定背景
檔名前綴_04_確定前景.png              # Watershed 確定前景
檔名前綴_05_距離變換.png              # 距離變換視覺化
檔名前綴_06_細胞分離.png              # 最終分離結果
檔名前綴_07_彩色疊圖.png              # 紅色遮罩疊圖
檔名前綴_參數設定.json                # 參數記錄
檔名前綴_統計報告.txt                 # 詳細統計
```

## 開發與擴展

### 添加新的處理方法
在 `dish_mask_core.py` 中新增方法：

```python
def custom_processing_method(self, params: MaskingParams) -> ProcessingResult:
    # 自訂處理邏輯
    pass
```

### 自訂 UI 元件
繼承 `DishMaskGUI` 類別：

```python
class CustomDishMaskGUI(DishMaskGUI):
    def create_custom_control_panel(self):
        # 自訂控制面板
        pass
```

### 批次處理腳本
使用核心模組建立批次處理：

```python
import os
from dish_mask_core import DishMaskProcessor, MaskingParams

processor = DishMaskProcessor()
params = MaskingParams(h_min=20, h_max=160)

for image_file in os.listdir("input_dir"):
    processor.load_image(f"input_dir/{image_file}")
    result = processor.process_mask(params)
    processor.save_results(result, f"output_dir/{image_file}_results")
```

## 故障排除

### 常見問題

1. **模組導入錯誤**
   ```
   ModuleNotFoundError: No module named 'dish_mask_core'
   ```
   確保在 `testing/` 目錄中執行，或將該目錄加入 Python 路徑。

2. **記憶體不足**
   - 使用命令列版本的快速模式
   - 減少影像解析度
   - 檢查系統可用記憶體

3. **處理速度慢**
   - 避免在 GUI 中頻繁調整參數
   - 使用命令列版本進行批次處理
   - 確保使用工作圖像模式 (非原圖模式)

4. **Qt 相關錯誤**
   ```
   QApplication: invalid style override passed
   ```
   移除 `app.setStyle('Fusion')` 或使用系統預設樣式。

### 除錯模式

在核心模組中啟用詳細輸出：

```python
# 在 dish_mask_core.py 開頭加入
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 技術規格

- **Python**: 3.8+
- **OpenCV**: 4.0+
- **PyQt5**: 5.15+
- **NumPy**: 1.19+
- **支援格式**: TIFF, PNG, JPEG, BMP
- **記憶體需求**: 建議 8GB+ (處理大型 WSI)
- **平台支援**: Windows, Linux, macOS

## 介面說明

### 左側控制面板

#### 1. HSV 顏色閾值
- **Hmin/Hmax**: 色相範圍 (0-179)
- **Smin/Smax**: 飽和度範圍 (0-255)
- **Vmin/Vmax**: 明度範圍 (0-255)
- **用途**: 過濾出特定顏色的細胞區域

#### 2. 形態學處理
- **Kernel Size**: 結構元素大小 (奇數, 1-15)
- **Open Iter**: 開運算次數 (去除雜訊)
- **Close Iter**: 閉運算次數 (填補小孔)
- **Dilate Iter**: 膨脹次數 (擴大區域)

#### 3. Watershed 分離
- **距離變換閾值**: 控制細胞分離敏感度 (0.1-0.8)
- 數值越小，分離越細緻
- 數值越大，只分離明顯分開的細胞

#### 4. 顯示設定
- **遮罩透明度**: 0-100%，控制紅色遮罩的不透明度

#### 5. 統計資訊
顯示即時統計數據：
- 總像素數
- HSV 遮罩像素數與比例
- 細胞區域像素數與比例
- 估算細胞數量
- 原圖與顯示圖尺寸資訊

### 右側影像顯示
- 即時顯示處理結果
- 紅色區域表示偵測到的細胞
- 支援自動縮放適應視窗

## 操作流程

### 1. 基本調參流程
1. **粗調 HSV 範圍**: 先調整 HSV 參數，讓紅色遮罩大致覆蓋細胞區域
2. **形態學清理**: 調整 kernel 大小和迭代次數，去除雜訊和填補空洞
3. **細胞分離**: 調整距離變換閾值，分離黏連的細胞
4. **顯示調整**: 調整透明度以便觀察效果

### 2. 參數調整建議
- **HSV 範圍**: 從寬鬆範圍開始，逐步縮小到目標區域
- **形態學**: Kernel 大小通常 3-7，迭代次數 1-3 次
- **距離閾值**: 從 0.3-0.4 開始，根據分離效果調整

### 3. 儲存結果
點擊「儲存所有結果」按鈕，會自動儲存：
- `01_HSV遮罩.png`: 初始 HSV 顏色篩選結果
- `02_形態學清理.png`: 形態學處理後的清理結果  
- `03_確定背景.png`: Watershed 確定背景區域
- `04_確定前景.png`: Watershed 確定前景區域
- `05_距離變換.png`: 距離變換視覺化
- `06_細胞分離.png`: 最終細胞分離結果 (二值圖)
- `07_彩色疊圖.png`: 紅色遮罩疊在原圖上的效果
- `參數設定.json`: 所有滑桿參數的 JSON 記錄
- `統計報告.txt`: 詳細統計資訊

### 4. 重置功能
點擊「重置參數」可將所有參數恢復為預設值

## 輸入檔案
工具會自動載入 `picture/tiff/` 目錄中第一個包含 "DISH" 字樣的 TIFF 檔案。

## 輸出位置
所有結果儲存在 `testing/output/middle-gen/` 目錄中。

## 技術細節

### 處理管線
1. **HSV 顏色空間轉換**: `cv.cvtColor(img, cv.COLOR_BGR2HSV)`
2. **顏色範圍篩選**: `cv.inRange(hsv, lower, upper)`  
3. **形態學開運算**: 去除小雜訊
4. **形態學閉運算**: 填補小空洞
5. **距離變換**: `cv.distanceTransform()` 找細胞中心
6. **Watershed 演算法**: 分離相黏細胞
7. **彩色疊圖**: 半透明顯示最終結果

### 效能最佳化
- 顯示圖像自動縮放至 75% 以提升互動響應速度
- 支援 8-bit 和 16-bit TIFF 格式自動轉換
- 處理結果保存在記憶體中，避免重複計算

## 常見問題

### Q: 紅色遮罩覆蓋不完整
A: 調整 HSV 範圍，特別是 H (色相) 和 S (飽和度) 的最小/最大值

### Q: 有很多雜訊點
A: 增加開運算次數或增大 kernel 大小

### Q: 細胞分離效果不好
A: 調整距離變換閾值，或檢查形態學清理是否足夠

### Q: 程式無回應
A: 大圖像處理可能需要時間，請耐心等待，或考慮先裁切影像

## 注意事項
- 工具僅支援 DISH 染色影像，其他染色類型可能需要調整參數
- 建議在調參前先觀察影像特徵，了解目標細胞的顏色分布
- 儲存結果前確保參數調整完成，避免重複輸出