"""
資料增強模組：定義訓練和驗證用的資料轉換流程
"""
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CROP_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transform():
    """
    取得訓練用的資料增強流程
    
    包含以下增強操作：
    1. 必要時填充到最小尺寸
    2. 隨機裁切（增加多樣性）
    3. 顏色抖動（模擬不同染色條件）
    4. 彈性變形（模擬組織變形）
    5. 水平/垂直翻轉
    6. ImageNet 標準化
    7. 轉換為 PyTorch Tensor
    """
    return A.Compose([
        A.PadIfNeeded(
            min_height=CROP_SIZE, 
            min_width=CROP_SIZE,
            border_mode=cv2.BORDER_REFLECT_101  # 鏡像填充
        ),
        A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE),
        A.ColorJitter(
            brightness=0.2,   # 亮度變化 ±20%
            contrast=0.2,     # 對比度變化 ±20%
            saturation=0.2,   # 飽和度變化 ±20%
            hue=0.05          # 色相變化 ±5%
        ),
        A.ElasticTransform(alpha=50, sigma=5, p=0.5),  # 50% 機率彈性變形
        A.HorizontalFlip(p=0.5),   # 50% 機率水平翻轉
        A.VerticalFlip(p=0.5),     # 50% 機率垂直翻轉
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_val_transform():
    """
    取得驗證用的資料轉換流程
    
    驗證時只做確定性的轉換：
    1. 必要時填充到最小尺寸
    2. 中心裁切（確保一致性）
    3. ImageNet 標準化
    4. 轉換為 PyTorch Tensor
    """
    return A.Compose([
        A.PadIfNeeded(
            min_height=CROP_SIZE,
            min_width=CROP_SIZE,
            border_mode=cv2.BORDER_REFLECT_101
        ),
        A.CenterCrop(height=CROP_SIZE, width=CROP_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
