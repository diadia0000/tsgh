"""
HER2 語義分割訓練腳本

使用 segmentation_models_pytorch (SMP) 進行訓練
架構: Unet++ + Swin-Transformer (Base) 編碼器
損失函數: Dice Loss + Cross-Entropy Loss
優化器: AdamW + CosineAnnealingLR

針對 NVIDIA RTX 5090 32GB + Intel Ultra 265K + 64GB RAM 優化:
- 混合精度訓練 (AMP)
- pin_memory 和多 worker 數據載入
- 顯存管理
"""
import os
import sys
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse
from datetime import datetime

# 添加當前目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from dataset import create_datasets, create_dataloaders
from utils import (
    CombinedLoss, 
    compute_iou, 
    compute_dice_score,
    MetricsTracker,
    CheckpointManager,
    TrainingLogger,
    clear_cuda_cache
)


def create_model():
    """
    建立 Unet++ 模型，使用 EfficientNet-B4 編碼器
    
    EfficientNet 優點:
    - 支援任意尺寸輸入 (適合 1024x1024)
    - SMP 原生支援，穩定性高
    - 參數效率高，效能優秀
    
    Returns:
        model: segmentation_models_pytorch 模型
    """
    print(f"建立模型: {config.model_name} + {config.encoder_name}")
    
    # 使用 SMP 原生支援的 EfficientNet
    model = smp.UnetPlusPlus(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=3,
        classes=config.num_classes,
        aux_params=config.aux_params
    )
    
    return model


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    gradient_accumulation_steps: int = 1
):
    """
    訓練一個 epoch
    
    Args:
        model: 模型
        train_loader: 訓練 DataLoader
        criterion: 損失函數
        optimizer: 優化器
        scaler: GradScaler (用於 AMP)
        device: 計算設備
        epoch: 當前 epoch
        gradient_accumulation_steps: 梯度累積步數
        
    Returns:
        metrics: 訓練指標
    """
    model.train()
    tracker = MetricsTracker(num_classes=config.num_classes, class_names=config.class_names)
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # 混合精度前向傳播
        with autocast('cuda', enabled=config.use_amp):
            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks)
            loss = loss / gradient_accumulation_steps
        
        # 混合精度反向傳播
        scaler.scale(loss).backward()
        
        # 梯度累積
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # 計算指標 (不需要梯度)
        with torch.no_grad():
            iou_dict = compute_iou(outputs, masks, config.num_classes)
            dice_dict = compute_dice_score(outputs, masks, config.num_classes)
        
        # 更新追蹤器
        tracker.update(loss_dict, iou_dict, dice_dict)
    
    # 處理最後不完整的梯度累積
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return tracker.get_metrics()


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion: CombinedLoss,
    device: torch.device,
    epoch: int
):
    """
    驗證
    
    Args:
        model: 模型
        val_loader: 驗證 DataLoader
        criterion: 損失函數
        device: 計算設備
        epoch: 當前 epoch
        
    Returns:
        metrics: 驗證指標
    """
    model.eval()
    tracker = MetricsTracker(num_classes=config.num_classes, class_names=config.class_names)
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
    
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # 混合精度推論
        with autocast('cuda', enabled=config.use_amp):
            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks)
        
        # 計算指標
        iou_dict = compute_iou(outputs, masks, config.num_classes)
        dice_dict = compute_dice_score(outputs, masks, config.num_classes)
        
        tracker.update(loss_dict, iou_dict, dice_dict)
    
    return tracker.get_metrics()


def train(args):
    """
    主訓練流程
    """
    print("=" * 60)
    print("HER2 語義分割訓練")
    print("=" * 60)
    
    # 設置設備
    device = config.device
    print(f"計算設備: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"顯存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 設置隨機種子
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
    
    # 建立 Dataset 和 DataLoader
    print("\n建立 Dataset...")
    train_dataset, val_dataset, test_dataset = create_datasets()
    
    # 覆寫 batch_size (如果有命令列參數)
    batch_size = args.batch_size if args.batch_size else config.batch_size
    print(f"Batch size: {batch_size}")
    print(f"有效 Batch size: {batch_size * config.gradient_accumulation_steps}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=batch_size
    )
    
    # 建立模型
    print("\n建立模型...")
    model = create_model()
    model = model.to(device)
    
    # 列印模型資訊
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數量: {total_params:,}")
    print(f"可訓練參數量: {trainable_params:,}")
    
    # 建立損失函數
    print("\n建立損失函數...")
    criterion = CombinedLoss(
        num_classes=config.num_classes,
        dice_weight=config.dice_weight,
        ce_weight=config.ce_weight,
        class_weights=config.class_weights
    ).to(device)  # 移動到 GPU
    
    # 建立優化器
    print("建立優化器: AdamW")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    
    # 建立學習率調度器
    print("建立學習率調度器: CosineAnnealingLR")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.min_lr
    )
    
    # 建立 GradScaler (用於混合精度訓練)
    scaler = GradScaler('cuda', enabled=config.use_amp)
    print(f"混合精度訓練 (AMP): {'啟用' if config.use_amp else '停用'}")
    
    # 建立 Checkpoint 管理器
    experiment_name = f"her2_unetpp_swin_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_manager = CheckpointManager(
        save_dir=config.model_save_dir,
        monitor_metric="val_miou",
        monitor_mode="max",
        save_top_k=config.save_top_k
    )
    
    # 建立訓練日誌
    logger = TrainingLogger(
        log_dir=config.log_dir,
        experiment_name=experiment_name
    )
    
    # 早停計數器
    early_stopping_counter = 0
    best_val_miou = 0.0
    
    # 訓練迴圈
    print("\n" + "=" * 60)
    print("開始訓練...")
    print("=" * 60)
    
    for epoch in range(1, config.epochs + 1):
        print(f"\n{'='*20} Epoch {epoch}/{config.epochs} {'='*20}")
        
        # 訓練
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )
        
        # 驗證
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # 更新學習率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # 記錄日誌
        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)
        logger.print_epoch_summary(epoch, train_metrics, val_metrics, current_lr)
        
        # 保存 Checkpoint
        val_miou = val_metrics["miou"]
        save_metrics = {"val_miou": val_miou, **val_metrics}
        is_best = checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, save_metrics
        )
        
        if is_best:
            print("  🎉 新的最佳模型！")
            best_val_miou = val_miou
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # 早停檢查
        if early_stopping_counter >= config.early_stopping_patience:
            print(f"\n早停觸發！連續 {config.early_stopping_patience} 個 epoch 沒有改善。")
            break
        
        # 清理顯存
        clear_cuda_cache()
    
    # 訓練完成
    print("\n" + "=" * 60)
    print("訓練完成！")
    print("=" * 60)
    print(f"最佳驗證 mIoU: {best_val_miou:.4f}")
    print(f"模型保存位置: {config.model_save_dir}")
    print(f"訓練日誌: {logger.log_file}")
    
    # 在測試集上評估最佳模型
    if test_loader is not None:
        print("\n在測試集上評估最佳模型...")
        checkpoint_manager.load_best_checkpoint(model)
        test_metrics = validate(model, test_loader, criterion, device, epoch=0)
        
        print(f"\n測試集結果:")
        print(f"  Loss: {test_metrics['total_loss']:.4f}")
        print(f"  mIoU: {test_metrics['miou']:.4f}")
        print(f"  IoU - 背景: {test_metrics['iou_Background']:.4f}, "
              f"內部: {test_metrics['iou_Interior']:.4f}, 膜: {test_metrics['iou_Membrane']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="HER2 語義分割訓練")
    parser.add_argument("--batch_size", type=int, default=None,
                        help=f"Batch size (預設: {config.batch_size})")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"訓練 epochs (預設: {config.epochs})")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"學習率 (預設: {config.learning_rate})")
    parser.add_argument("--no_amp", action="store_true",
                        help="停用混合精度訓練")
    
    args = parser.parse_args()
    
    # 更新配置
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.no_amp:
        config.use_amp = False
    
    train(args)


if __name__ == "__main__":
    main()
