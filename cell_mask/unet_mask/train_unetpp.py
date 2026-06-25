"""
UNet++ HER2 細胞膜語義分割訓練腳本

使用 LAB 色彩空間生成的偽標籤進行訓練
模型架構: UNet++ with DenseNet121 Encoder (ImageNet pretrained)
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # headless 環境只存檔,不開 GUI 視窗
import matplotlib.pyplot as plt

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """
    載入配置檔案
    
    Returns:
        Config: 配置物件
    
    Raises:
        ImportError: 若 config.py 不存在
    """
    try:
        from config import config
        return config
    except ImportError:
        raise ImportError(
            "找不到 config.py！\n"
            "請複製 config_example.py 為 config.py 並設定參數"
        )


class HER2MembraneDataset(Dataset):
    """
    HER2 細胞膜分割資料集
    
    讀取影像和對應的 LAB 生成偽標籤
    
    Attributes:
        image_paths: 影像路徑列表
        mask_paths: 對應的 mask 路徑列表
        transform: Albumentations 轉換
        threshold: 二值化閾值 (0-255)
    """
    
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        transform: Optional[A.Compose] = None,
        threshold: int = 20,
    ) -> None:
        """
        初始化資料集
        
        Args:
            image_paths: 影像路徑列表
            mask_paths: 對應的 mask 路徑列表
            transform: Albumentations 資料增強
            threshold: Mask 二值化閾值 (R 通道值 > threshold 視為膜)
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.threshold = threshold
        
        # 驗證路徑配對
        assert len(image_paths) == len(mask_paths), \
            f"影像數量 ({len(image_paths)}) 與 Mask 數量 ({len(mask_paths)}) 不符"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        取得單筆資料
        
        Args:
            idx: 索引
            
        Returns:
            image: 影像 Tensor (C, H, W)
            mask: Mask Tensor (H, W), 二值 0 或 1
        """
        # 讀取影像
        image = io.imread(str(self.image_paths[idx]))
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # 讀取 Mask (RGB 格式，R 通道為膜強度)
        mask_rgb = io.imread(str(self.mask_paths[idx]))
        if mask_rgb.ndim == 3:
            mask_gray = mask_rgb[:, :, 0]  # 取 R 通道
        else:
            mask_gray = mask_rgb
        
        # 二值化 (R > threshold 為膜)
        mask = (mask_gray > self.threshold).astype(np.uint8)
        
        # 資料增強
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # 確保 mask 為 long 類型 (CrossEntropyLoss 需要)
        mask = mask.long()
        
        return image, mask


def get_train_transforms(image_size: Tuple[int, int], augmentation: dict) -> A.Compose:
    """
    取得訓練時的資料增強

    Args:
        image_size: 輸出影像尺寸 (H, W)
        augmentation: config.augmentation dict

    Returns:
        Albumentations Compose 物件
    """
    cj = augmentation.get("color_jitter", {})
    gb = augmentation.get("gaussian_blur", {})
    et = augmentation.get("elastic_transform", {})

    return A.Compose([
        # 確保最小尺寸 (處理邊緣 tile)
        A.PadIfNeeded(
            min_height=image_size[0],
            min_width=image_size[1],
            border_mode=0,  # cv2.BORDER_CONSTANT
            fill=255,  # 白色填充 (背景)
            fill_mask=0,  # mask 填充 0 (非膜)
        ),

        # 幾何變換
        A.RandomRotate90(p=0.5 if augmentation.get("random_rotate90", True) else 0.0),
        A.HorizontalFlip(p=0.5 if augmentation.get("horizontal_flip", True) else 0.0),
        A.VerticalFlip(p=0.5 if augmentation.get("vertical_flip", True) else 0.0),
        A.Affine(
            translate_percent=(-0.1, 0.1),
            scale=(0.9, 1.1),
            rotate=(-45, 45),
            border_mode=0,
            p=0.5
        ),

        # 彈性變形 (模擬細胞形變)
        A.ElasticTransform(
            alpha=et.get("alpha", 50),
            sigma=et.get("sigma", 5),
            p=et.get("p", 0.3),
        ),

        # 顏色增強 (應對染色差異)
        A.ColorJitter(
            brightness=cj.get("brightness", 0.2),
            contrast=cj.get("contrast", 0.2),
            saturation=cj.get("saturation", 0.2),
            hue=cj.get("hue", 0.05),
            p=0.5,
        ),

        # 模糊 (應對對焦差異)
        A.GaussianBlur(
            blur_limit=gb.get("blur_limit", (3, 7)),
            p=gb.get("p", 0.3),
        ),

        # 正規化 + 轉 Tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: Tuple[int, int]) -> A.Compose:
    """
    取得驗證時的資料轉換 (無增強)
    
    Args:
        image_size: 輸出影像尺寸 (H, W)
        
    Returns:
        Albumentations Compose 物件
    """
    return A.Compose([
        # 確保最小尺寸 (處理邊緣 tile)
        A.PadIfNeeded(
            min_height=image_size[0],
            min_width=image_size[1],
            border_mode=0,
            fill=255,
            fill_mask=0,
        ),
        
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


class CombinedLoss(nn.Module):
    """
    組合損失函數: Focal Loss + Dice Loss

    Focal Loss 自動降低易分類像素的權重，聚焦邊界等困難區域，
    比 CE 更適合類別不平衡的分割任務。
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.dice_loss = smp.losses.DiceLoss(
            mode='multiclass',
            from_logits=True,
        )

        self.focal_loss = smp.losses.FocalLoss(
            mode='multiclass',
            gamma=focal_gamma,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal


def build_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
) -> torch.Tensor:
    """
    計算混淆矩陣 (留在 GPU 上，不呼叫 .item())

    Args:
        pred: 模型輸出 (B, C, H, W)
        target: Ground Truth (B, H, W)
        num_classes: 類別數量

    Returns:
        混淆矩陣 (num_classes, num_classes)，cm[i, j] = 真實 i 預測 j 的像素數
    """
    pred_labels = pred.argmax(dim=1).flatten()  # (B*H*W,)
    target_flat = target.flatten()               # (B*H*W,)

    # 用線性索引一次算出整個混淆矩陣
    indices = target_flat * num_classes + pred_labels
    cm = torch.bincount(indices, minlength=num_classes ** 2)
    return cm.reshape(num_classes, num_classes).float()


def miou_from_confusion_matrix(cm: torch.Tensor) -> float:
    """
    從累積的混淆矩陣計算 Mean IoU

    當某 class 的 union == 0（真實與預測皆無該 class），視為 IoU = 1.0。

    Args:
        cm: 混淆矩陣 (num_classes, num_classes)

    Returns:
        Mean IoU 值
    """
    intersection = cm.diag()
    union = cm.sum(dim=1) + cm.sum(dim=0) - intersection

    ious = torch.where(
        union > 0,
        intersection / union,
        torch.ones_like(intersection),  # 預測也是空，算正確
    )
    return ious.mean().item()


def split_dataset(
    image_dir: Path,
    mask_dir: Path,
    train_ratio: float = 0.85,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    """
    分割資料集為訓練/驗證集

    Args:
        image_dir: 影像目錄
        mask_dir: Mask 目錄
        train_ratio: 訓練集比例，剩餘為驗證集
        seed: 隨機種子

    Returns:
        (train_images, train_masks, val_images, val_masks)
    """
    # 收集所有影像
    extensions = ['.tiff', '.tif', '.png', '.jpg', '.jpeg']
    image_paths = []
    for ext in extensions:
        image_paths.extend(image_dir.glob(f'*{ext}'))
        image_paths.extend(image_dir.glob(f'*{ext.upper()}'))

    image_paths = sorted(set(image_paths))

    # 配對 Mask
    paired_data = []
    for img_path in image_paths:
        mask_name = img_path.stem + '_mask.png'
        mask_path = mask_dir / mask_name
        if mask_path.exists():
            paired_data.append((img_path, mask_path))

    logger.info(f"找到 {len(paired_data)} 筆配對資料")

    # 隨機打亂
    np.random.seed(seed)
    np.random.shuffle(paired_data)

    # 分割
    n_train = int(len(paired_data) * train_ratio)

    train_data = paired_data[:n_train]
    val_data = paired_data[n_train:]

    # 解包
    train_images = [x[0] for x in train_data]
    train_masks = [x[1] for x in train_data]
    val_images = [x[0] for x in val_data]
    val_masks = [x[1] for x in val_data]

    logger.info(f"資料分割: 訓練={len(train_images)}, 驗證={len(val_images)}")

    return train_images, train_masks, val_images, val_masks


@dataclass
class TrainState:
    """訓練狀態追蹤"""
    epoch: int = 0
    best_miou: float = 0.0
    patience_counter: int = 0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int = 1,
) -> Tuple[float, float]:
    """
    訓練一個 Epoch
    
    Args:
        model: 模型
        loader: DataLoader
        criterion: 損失函數
        optimizer: 優化器
        scaler: GradScaler (AMP)
        device: 計算設備
        accumulation_steps: 梯度累積步數
        
    Returns:
        (平均損失, 平均 mIoU)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    cm_sum = None  # 累積混淆矩陣

    optimizer.zero_grad()

    pbar = tqdm(loader, desc='Training', leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # 混合精度前向傳播
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks) / accumulation_steps

        # 反向傳播
        scaler.scale(loss).backward()

        # 梯度累積
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 計算指標（累積混淆矩陣，不逐 batch 呼叫 .item()）
        total_loss += loss.item() * accumulation_steps
        with torch.no_grad():
            cm = build_confusion_matrix(outputs, masks)
            cm_sum = cm if cm_sum is None else cm_sum + cm
        n_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'mIoU': f'{miou_from_confusion_matrix(cm_sum):.4f}'
        })

    epoch_miou = miou_from_confusion_matrix(cm_sum) if cm_sum is not None else 0.0
    return total_loss / n_batches, epoch_miou


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    驗證模型
    
    Args:
        model: 模型
        loader: DataLoader
        criterion: 損失函數
        device: 計算設備
        
    Returns:
        (平均損失, 平均 mIoU)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    cm_sum = None

    pbar = tqdm(loader, desc='Validating', leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)

        total_loss += loss.item()
        cm = build_confusion_matrix(outputs, masks)
        cm_sum = cm if cm_sum is None else cm_sum + cm
        n_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mIoU': f'{miou_from_confusion_matrix(cm_sum):.4f}'
        })

    epoch_miou = miou_from_confusion_matrix(cm_sum) if cm_sum is not None else 0.0
    return total_loss / n_batches, epoch_miou


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    state: TrainState,
    save_path: Path,
    is_best: bool = False,
) -> None:
    """
    儲存模型檢查點
    
    Args:
        model: 模型
        optimizer: 優化器
        scheduler: 學習率調度器
        state: 訓練狀態
        save_path: 儲存目錄
        is_best: 是否為最佳模型
    """
    save_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': state.epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_miou': state.best_miou,
    }
    
    # 儲存最新檢查點
    torch.save(checkpoint, save_path / 'last_checkpoint.pth')
    
    # 儲存最佳模型 (只保存模型權重)
    if is_best:
        torch.save(model.state_dict(), save_path / 'best_model.pth')
        logger.info(f"✓ 保存最佳模型 (mIoU: {state.best_miou:.4f})")


def plot_training_history(history: Dict[str, List[float]], save_path: Path) -> None:
    """
    繪製訓練統計曲線並存成 PNG (Loss / mIoU / Learning Rate)

    每個 epoch 結束後覆寫同一張圖,訓練中即可即時查看,
    early stopping 或中途崩潰也能保留已訓練部分的曲線。

    Args:
        history: 包含 train_loss/train_miou/val_loss/val_miou/lr 的歷史紀錄
        save_path: 圖片儲存路徑 (含檔名)
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss 曲線
    axes[0].plot(epochs, history['train_loss'], label='Train', marker='o', markersize=3)
    axes[0].plot(epochs, history['val_loss'], label='Val', marker='o', markersize=3)
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # mIoU 曲線
    axes[1].plot(epochs, history['train_miou'], label='Train', marker='o', markersize=3)
    axes[1].plot(epochs, history['val_miou'], label='Val', marker='o', markersize=3)
    axes[1].set_title('Mean IoU')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning Rate 曲線 (對數軸)
    axes[2].plot(epochs, history['lr'], color='tab:green', marker='o', markersize=3)
    axes[2].set_title('Learning Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()


def main() -> None:
    """主程式入口"""
    # 載入配置
    config = load_config()
    
    logger.info("=" * 60)
    logger.info("UNet++ HER2 細胞膜分割訓練")
    logger.info("=" * 60)
    logger.info(f"設備: {config.device}")
    logger.info(f"模型: {config.model_name} + {config.encoder_name}")
    logger.info(f"影像尺寸: {config.image_size}")
    logger.info(f"Batch Size: {config.batch_size} (有效: {config.effective_batch_size})")
    logger.info(f"Epochs: {config.epochs}")
    logger.info("=" * 60)
    
    # 分割資料集
    train_images, train_masks, val_images, val_masks = split_dataset(
        image_dir=config.train_image_dir,
        mask_dir=config.pseudo_label_mask_dir,
        train_ratio=config.train_ratio,
        seed=config.random_seed,
    )
    
    # 建立 Dataset
    train_dataset = HER2MembraneDataset(
        image_paths=train_images,
        mask_paths=train_masks,
        transform=get_train_transforms(config.image_size, config.augmentation),
    )
    
    val_dataset = HER2MembraneDataset(
        image_paths=val_images,
        mask_paths=val_masks,
        transform=get_val_transforms(config.image_size),
    )
    
    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True,
        prefetch_factor=4,
    )
    
    # 建立模型
    model = smp.UnetPlusPlus(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=3,
        classes=config.num_classes,
    )
    model = model.to(config.device)
    
    logger.info(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 損失函數
    criterion = CombinedLoss(
        dice_weight=config.dice_weight,
        focal_weight=config.focal_weight,
    )
    
    # 優化器
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    
    # 學習率調度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.t_max,
        eta_min=config.min_lr,
    )
    
    # 混合精度
    scaler = GradScaler('cuda', enabled=config.use_amp)
    
    # 訓練狀態
    state = TrainState()

    # 訓練統計歷史 (用於繪製曲線)
    history: Dict[str, List[float]] = {
        'train_loss': [], 'train_miou': [],
        'val_loss': [], 'val_miou': [], 'lr': [],
    }

    # 訓練迴圈
    logger.info("開始訓練...")

    for epoch in range(config.epochs):
        state.epoch = epoch + 1

        # 此 epoch 訓練時實際使用的學習率 (在 scheduler.step() 之前)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"\nEpoch {state.epoch}/{config.epochs}")
        logger.info(f"學習率: {current_lr:.2e}")
        
        # 訓練
        train_loss, train_miou = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=config.device,
            accumulation_steps=config.gradient_accumulation_steps,
        )
        
        # 驗證
        val_loss, val_miou = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=config.device,
        )
        
        # 更新學習率
        scheduler.step()
        
        # 輸出指標
        logger.info(
            f"Train - Loss: {train_loss:.4f}, mIoU: {train_miou:.4f} | "
            f"Val - Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}"
        )

        # 紀錄歷史並更新訓練曲線圖
        history['train_loss'].append(train_loss)
        history['train_miou'].append(train_miou)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)
        history['lr'].append(current_lr)
        plot_training_history(history, config.model_save_dir / 'training_curves.png')

        # 檢查是否為最佳模型
        is_best = val_miou > state.best_miou
        if is_best:
            state.best_miou = val_miou
            state.patience_counter = 0
        else:
            state.patience_counter += 1
        
        # 保存檢查點
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            save_path=config.model_save_dir,
            is_best=is_best,
        )
        
        # Early Stopping
        if state.patience_counter >= config.early_stopping_patience:
            logger.info(
                f"Early Stopping! 驗證 mIoU 已有 {config.early_stopping_patience} 個 epoch 未改善"
            )
            break
    
    logger.info("=" * 60)
    logger.info(f"訓練完成! 最佳 mIoU: {state.best_miou:.4f}")
    logger.info(f"模型已儲存至: {config.model_save_dir / 'best_model.pth'}")
    logger.info(f"訓練曲線已儲存至: {config.model_save_dir / 'training_curves.png'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
