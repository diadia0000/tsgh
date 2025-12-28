"""
HER2 語義分割 Dataset 類

支援:
- TIFF/PNG 影像讀取
- Albumentations 數據增強
- 自動數據集分割 (訓練/驗證/測試)
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from config import config


def get_training_augmentation():
    """
    獲取訓練時的數據增強
    
    包含:
    - RandomRotate90: 隨機 90 度旋轉
    - HorizontalFlip: 水平翻轉
    - VerticalFlip: 垂直翻轉
    - ColorJitter: 顏色抖動 (針對病理影像)
    - GaussianBlur: 高斯模糊
    - ElasticTransform: 彈性變形
    """
    aug_config = config.augmentation
    
    transforms = [
        # 統一尺寸 (必須在最前面，確保所有圖片大小一致)
        A.Resize(height=config.image_size[0], width=config.image_size[1]),
        
        # 幾何變換
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # 顏色抖動 (針對病理影像的 H&E/DAB 染色)
        A.ColorJitter(
            brightness=aug_config["color_jitter"]["brightness"],
            contrast=aug_config["color_jitter"]["contrast"],
            saturation=aug_config["color_jitter"]["saturation"],
            hue=aug_config["color_jitter"]["hue"],
            p=0.5
        ),
        
        # 高斯模糊
        A.GaussianBlur(
            blur_limit=aug_config["gaussian_blur"]["blur_limit"],
            p=aug_config["gaussian_blur"]["p"]
        ),
        
        # 彈性變形 (模擬組織變形)
        A.ElasticTransform(
            alpha=aug_config["elastic_transform"]["alpha"],
            sigma=aug_config["elastic_transform"]["sigma"],
            p=aug_config["elastic_transform"]["p"]
        ),
        
        # HueSaturationValue (針對病理影像染色變異)
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),
        
        # 正規化和轉換為 Tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 標準化
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ]
    
    return A.Compose(transforms)


def get_validation_augmentation():
    """
    獲取驗證/測試時的數據增強 (僅正規化)
    """
    return A.Compose([
        # 統一尺寸
        A.Resize(height=config.image_size[0], width=config.image_size[1]),
        
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


class HER2SegmentationDataset(Dataset):
    """
    HER2 語義分割 Dataset
    
    Args:
        image_paths: 影像路徑列表
        mask_paths: mask 路徑列表
        transform: Albumentations 變換
        num_classes: 類別數量
    """
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[A.Compose] = None,
        num_classes: int = 3
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.num_classes = num_classes
        
        # 驗證路徑數量一致
        assert len(image_paths) == len(mask_paths), \
            f"影像數量 ({len(image_paths)}) 與 mask 數量 ({len(mask_paths)}) 不一致"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 讀取影像
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise ValueError(f"無法讀取影像: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 讀取 mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"無法讀取 mask: {self.mask_paths[idx]}")
        
        # 確保 mask 是正確的格式 (0, 1, 2)
        mask = mask.astype(np.int64)
        
        # 應用數據增強
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 確保 mask 是 LongTensor
        mask = mask.long()
        
        return image, mask
    
    def get_class_weights(self) -> torch.Tensor:
        """
        計算類別權重 (用於處理類別不平衡)
        
        Returns:
            類別權重 tensor
        """
        print("計算類別權重...")
        class_counts = np.zeros(self.num_classes)
        
        for mask_path in self.mask_paths[:100]:  # 使用部分樣本估計
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is not None:
                for c in range(self.num_classes):
                    class_counts[c] += np.sum(mask == c)
        
        # 計算權重 (inverse frequency)
        total = class_counts.sum()
        weights = total / (self.num_classes * class_counts + 1e-6)
        
        # 正規化
        weights = weights / weights.sum() * self.num_classes
        
        return torch.FloatTensor(weights)


def create_datasets(
    image_dir: str = None,
    mask_dir: str = None,
    train_ratio: float = None,
    val_ratio: float = None,
    random_seed: int = None
) -> Tuple[HER2SegmentationDataset, HER2SegmentationDataset, HER2SegmentationDataset]:
    """
    建立訓練、驗證、測試 Dataset
    
    Args:
        image_dir: 影像資料夾路徑
        mask_dir: mask 資料夾路徑
        train_ratio: 訓練集比例
        val_ratio: 驗證集比例
        random_seed: 隨機種子
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    image_dir = image_dir or config.train_image_dir
    mask_dir = mask_dir or config.mask_dir
    train_ratio = train_ratio or config.train_ratio
    val_ratio = val_ratio or config.val_ratio
    random_seed = random_seed or config.random_seed
    
    # 收集所有影像和對應的 mask
    image_path = Path(image_dir)
    mask_path = Path(mask_dir)
    
    # 支援的影像格式
    image_extensions = ['.tiff', '.tif', '.png', '.jpg', '.jpeg']
    
    # 收集影像檔案
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_path.glob(f"*{ext}"))
        image_files.extend(image_path.glob(f"*{ext.upper()}"))
    image_files = sorted(set(image_files))
    
    # 配對影像和 mask
    paired_data = []
    for img_file in image_files:
        # 對應的 mask 檔案名
        mask_filename = img_file.stem + "_mask.png"
        mask_file = mask_path / mask_filename
        
        if mask_file.exists():
            paired_data.append((str(img_file), str(mask_file)))
    
    print(f"找到 {len(paired_data)} 對影像-mask 配對")
    
    if len(paired_data) == 0:
        raise ValueError("沒有找到任何影像-mask 配對！請先運行 dataset_generator.py")
    
    # 分割數據集
    image_paths = [p[0] for p in paired_data]
    mask_paths = [p[1] for p in paired_data]
    
    # 第一次分割: 訓練集 vs (驗證+測試)
    train_images, temp_images, train_masks, temp_masks = train_test_split(
        image_paths, mask_paths,
        train_size=train_ratio,
        random_state=random_seed
    )
    
    # 第二次分割: 驗證集 vs 測試集
    val_size = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
    val_images, test_images, val_masks, test_masks = train_test_split(
        temp_images, temp_masks,
        train_size=val_size,
        random_state=random_seed
    )
    
    print(f"訓練集: {len(train_images)} 張")
    print(f"驗證集: {len(val_images)} 張")
    print(f"測試集: {len(test_images)} 張")
    
    # 建立 Dataset
    train_dataset = HER2SegmentationDataset(
        train_images, train_masks,
        transform=get_training_augmentation(),
        num_classes=config.num_classes
    )
    
    val_dataset = HER2SegmentationDataset(
        val_images, val_masks,
        transform=get_validation_augmentation(),
        num_classes=config.num_classes
    )
    
    test_dataset = HER2SegmentationDataset(
        test_images, test_masks,
        transform=get_validation_augmentation(),
        num_classes=config.num_classes
    )
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: HER2SegmentationDataset,
    val_dataset: HER2SegmentationDataset,
    test_dataset: HER2SegmentationDataset = None,
    batch_size: int = None,
    num_workers: int = None,
    pin_memory: bool = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    建立 DataLoader
    
    Args:
        train_dataset: 訓練 Dataset
        val_dataset: 驗證 Dataset
        test_dataset: 測試 Dataset (可選)
        batch_size: batch 大小
        num_workers: worker 數量
        pin_memory: 是否 pin memory
        
    Returns:
        train_loader, val_loader, test_loader
    """
    batch_size = batch_size or config.batch_size
    num_workers = num_workers or config.num_workers
    pin_memory = pin_memory if pin_memory is not None else config.pin_memory
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 確保每個 batch 大小一致
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 測試 Dataset
    print("測試 Dataset 建立...")
    
    try:
        train_dataset, val_dataset, test_dataset = create_datasets()
        
        print(f"\n測試讀取第一筆資料...")
        image, mask = train_dataset[0]
        
        print(f"影像 shape: {image.shape}")
        print(f"影像 dtype: {image.dtype}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask dtype: {mask.dtype}")
        print(f"Mask 唯一值: {torch.unique(mask).tolist()}")
        
        # 計算類別權重
        weights = train_dataset.get_class_weights()
        print(f"\n類別權重: {weights.tolist()}")
        
    except Exception as e:
        print(f"錯誤: {e}")
        print("請確保已經運行 dataset_generator.py 生成 mask 檔案")
