"""
資料集模組：定義 UNet 訓練用的 Dataset 類別
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from time import perf_counter

from config import CROP_SIZE, NUM_CLASSES
from mask_generation import generate_pseudo_mask_v2


class LargeImagePseudoMaskDataset(Dataset):
    """
    大圖像資料集：支援 Pseudo Mask 快取機制
    
    功能：
    - 自動掃描指定目錄下的所有圖像
    - 首次存取時生成 pseudo mask 並快取
    - 後續存取直接讀取快取的 mask
    - 支援即時裁切和資料增強
    
    輸出：
    - 圖像：[3, H, W] float32 tensor
    - Mask：[NUM_CLASSES, H, W] float32 one-hot tensor
    """
    
    def __init__(
        self, 
        image_root: Path, 
        mask_root: Path, 
        transform=None, 
        cache_ext: str = '.png'
    ):
        """
        初始化資料集
        
        參數：
            image_root: 原始圖像所在目錄
            mask_root: Mask 快取儲存目錄
            transform: Albumentations 資料增強流程
            cache_ext: 快取 mask 的檔案副檔名
        """
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.transform = transform
        self.cache_ext = cache_ext
        
        # 收集所有支援格式的圖像檔案
        self.samples = sorted([
            p for p in self.image_root.rglob('*') 
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        ])
        
        if not self.samples:
            print(f"警告：在 {image_root} 下找不到任何圖像")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path = self.samples[idx]
        
        # 計算相對路徑，用於在快取目錄中維持相同的目錄結構
        try:
            rel = img_path.relative_to(self.image_root)
        except ValueError:
            rel = Path(img_path.name)
            
        mask_path = (self.mask_root / rel).with_suffix(self.cache_ext)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 讀取圖像
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"讀取圖像錯誤 {img_path}: {e}")
            img = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)

        # 讀取或生成 Mask
        if mask_path.exists():
            try:
                mask = np.array(Image.open(mask_path))
            except Exception:
                # 快取讀取失敗，重新生成
                mask = generate_pseudo_mask_v2(img_path)
                Image.fromarray(mask).save(mask_path)
        else:
            # 快取不存在，生成並儲存
            mask = generate_pseudo_mask_v2(img_path)
            Image.fromarray(mask).save(mask_path)
            
        # 應用資料增強
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
            # 確保 mask 是 2D tensor
            if mask.ndim == 3:
                mask = mask.squeeze()
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask)
        
        # 轉換為 one-hot 編碼 [NUM_CLASSES, H, W]
        # mask 的值：0=背景, 1=細胞內, 2=細胞膜
        mask = mask.long()
        mask_onehot = F.one_hot(mask, num_classes=NUM_CLASSES)  # [H, W, C]
        mask_onehot = mask_onehot.permute(2, 0, 1).float()      # [C, H, W]

        return img, mask_onehot


class PreloadedDataset(Dataset):
    """
    預載入資料集：將整個資料集載入 RAM
    
    功能：
    - 消除訓練時的 I/O 瓶頸
    - 適用於記憶體充足的情況
    - 顯著提升訓練速度
    """
    
    def __init__(self, base_dataset: Dataset, desc: str = "載入中"):
        """
        初始化預載入資料集
        
        參數：
            base_dataset: 要預載入的基礎資料集
            desc: 進度顯示的描述文字
        """
        print(f"[{desc}] 正在將 {len(base_dataset)} 個樣本預載入 RAM...")
        t0 = perf_counter()
        
        self.data = []
        for i in range(len(base_dataset)):
            try:
                self.data.append(base_dataset[i])
            except Exception as e:
                print(f"  載入索引 {i} 失敗: {e}")
            
            if (i + 1) % 500 == 0:
                print(f"  已載入 {i+1}/{len(base_dataset)}...")
        
        print(f"[{desc}] 完成！耗時 {perf_counter()-t0:.1f} 秒，共載入 {len(self.data)} 個樣本")
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
