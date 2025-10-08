# GitHub Copilot Instructions — Python / aicspylibczi 3.3.1 專案

## 版本與依賴要求
- 使用 **Python 3.9+** （或專案指定版本）  
- 專案必須使用 **aicspylibczi 3.3.1** 版本，不使用其他版本  
- 在 import 時要明確指出版本性質或在 requirements 中鎖版：`aicspylibczi==3.3.1`

## 總體風格與規範
- 遵守 **PEP8** 標準風格（縮排、命名、空格等）  
- 函式與方法一定要加 **type hints**（參數與回傳）  
- 變數、函式命名要清晰、有意義
- 減少重複程式碼，盡量封裝共用邏輯

## 與 aicspylibczi 的互動規範

- 所有和 CZI 檔案操作的程式碼都要用 aicspylibczi 提供的接口，而不是手寫二進位解析  
- 在程式碼中示範使用以下方法與屬性：

  1. **`CziFile` 類別**  
     - import 要來自 `aicspylibczi`：  
       ```python
       from aicspylibczi import CziFile
       ```  
     - 建構：接受 `Path` / 檔名 / bytes IO 等  
     - 支援參數 `verbose: bool = False` 作為選擇性輸出訊息  
     
  2. **讀取影像與維度資訊**  
     - 使用 `get_dims_shape()` 取得維度範圍列表  
     - 使用 `czi.dims` 和 `czi.size` 屬性  
     - 使用 `czi.read_image(**kwargs)` 來讀取指定維度子區塊  
       - `read_image` 回傳一個 `(numpy.ndarray, dims_list)`  
     - 若是 mosaic 檔案，使用 `czi.read_mosaic(...)` 方法  
     - 使用 `czi.is_mosaic()` 來判別是否為 mosaic 模式  
     - 若需要切圖或 bounding box：使用 `get_mosaic_tile_bounding_box(...)`、`get_tile_bounding_box(...)`、`get_scene_bounding_box(...)` 等方法
  3. ZI 檔案在讀取 mosaic 時，region 參數需要使用 CZI 內部的 mosaic tile bounding boxes 座標系統，而不是簡單地從 (0,0) 開始切割。
## 範例（正確）  
```python
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from aicspylibczi import CziFile

def load_czi_slice(
    file_path: Path,
    S: int,
    Z: int,
    C: int
) -> np.ndarray:
    czi = CziFile(file_path, verbose=False)
    # dims_shape = [{'X': (0, w), 'Y': (0, h), ...}, …]
    dims_info = czi.get_dims_shape()
    img, dims = czi.read_image(S=S, Z=Z, C=C)
    # dims 是類似 [('S',1),('C',1),('Z',1),('Y',h),('X',w)]
    # img.shape 對應這個 dims 結構
    # 例如做 normalization
    arr = img.astype(np.float32)
    p05, p99 = np.percentile(arr, [5, 99])
    norm = np.clip((arr - p05) / (p99 - p05), 0.0, 1.0)
    return norm
---
applyTo: '**'
description: 'description'
---
Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.