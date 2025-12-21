"""
工具模組：圖像處理相關的輔助函數
"""
import numpy as np
from skimage import io
from pathlib import Path


def read_rgb_uint8(path: Path) -> np.ndarray:
    """
    讀取圖像檔案並轉換為 RGB uint8 格式
    
    此函數處理各種圖像格式：
    - 灰階圖像 -> 轉換為 RGB
    - RGBA 圖像 -> 移除 Alpha 通道
    - 非 uint8 -> 正規化到 0-255
    
    參數：
        path: 圖像檔案路徑
        
    回傳：
        形狀為 (H, W, 3) 的 RGB 圖像，dtype 為 uint8
    """
    img = io.imread(path)
    
    # 處理灰階圖像：複製到三個通道
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    
    # 處理 RGBA：移除 Alpha 通道
    if img.shape[2] == 4:
        img = img[..., :3]
    
    # 如果不是 uint8，進行正規化
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    return img
