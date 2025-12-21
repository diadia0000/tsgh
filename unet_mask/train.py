"""
主程式：UNet 分割模型訓練腳本

這是整個專案的入口點，執行 `python train.py` 來開始訓練。
"""
import sys
from pathlib import Path

# 將當前目錄加入 Python 路徑，確保能正確 import 本地模組
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from time import perf_counter

# 本地模組匯入
from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    FULL_DATA_DIR, MASK_CACHE_DIR, REP_DIR, MODEL_DIR,
    TRAIN_VAL_SPLIT, RANDOM_SEED, ensure_dirs
)
from transforms import get_train_transform, get_val_transform
from dataset import LargeImagePseudoMaskDataset, PreloadedDataset
from model import create_model
from trainer import train_one_epoch, evaluate


def main():
    """
    主訓練函數
    
    訓練流程：
    1. 初始化設定與目錄
    2. 載入並預處理資料集
    3. 建立模型與優化器
    4. 執行訓練迴圈
    5. 儲存最佳模型
    """
    print("=" * 60)
    print("UNet 細胞膜分割模型訓練")
    print("=" * 60)
    print(f"裝置: {DEVICE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"訓練輪數: {NUM_EPOCHS}")
    print(f"學習率: {LEARNING_RATE}")
    print("=" * 60)
    
    # 確保必要的目錄存在
    ensure_dirs()
    
    # 取得資料增強轉換器
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    # ===== 初始化資料集 =====
    print("\n正在初始化資料集...")
    rep_mask_dir = MASK_CACHE_DIR / 'rep_eval'
    
    # 建立三個資料集：訓練、驗證、代表性評估
    full_ds_train = LargeImagePseudoMaskDataset(
        FULL_DATA_DIR, MASK_CACHE_DIR, transform=train_transform
    )
    full_ds_val = LargeImagePseudoMaskDataset(
        FULL_DATA_DIR, MASK_CACHE_DIR, transform=val_transform
    )
    rep_ds = LargeImagePseudoMaskDataset(
        REP_DIR, rep_mask_dir, transform=val_transform
    )
    
    # 檢查資料集大小
    num_samples = len(full_ds_train)
    if num_samples == 0:
        print("錯誤：找不到訓練資料，程式結束。")
        return
    
    print(f"總樣本數: {num_samples}")
    
    # ===== 訓練/驗證集分割 =====
    train_size = int(TRAIN_VAL_SPLIT * num_samples)
    val_size = num_samples - train_size
    print(f"訓練集: {train_size} 張，驗證集: {val_size} 張")
    
    # 使用固定隨機種子確保可重現性
    indices = torch.randperm(
        num_samples, 
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    ).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_ds_subset = torch.utils.data.Subset(full_ds_train, train_indices)
    val_ds_subset = torch.utils.data.Subset(full_ds_val, val_indices)
    
    # ===== 預載入資料到 RAM =====
    print("\n" + "=" * 60)
    print("正在將資料預載入 RAM（這可能需要幾分鐘）...")
    print("=" * 60)
    
    train_ds_preloaded = PreloadedDataset(train_ds_subset, desc="訓練集")
    val_ds_preloaded = PreloadedDataset(val_ds_subset, desc="驗證集")
    rep_ds_preloaded = PreloadedDataset(rep_ds, desc="代表集")
    
    # ===== 建立資料載入器 =====
    train_loader = DataLoader(
        train_ds_preloaded,
        batch_size=BATCH_SIZE,
        shuffle=True,           # 訓練時打亂順序
        num_workers=0,          # 資料已在 RAM，不需要額外 worker
        pin_memory=True         # 加速 GPU 傳輸
    )
    val_loader = DataLoader(
        val_ds_preloaded,
        batch_size=BATCH_SIZE,
        shuffle=False,          # 驗證時不打亂
        num_workers=0,
        pin_memory=True
    )
    rep_loader = DataLoader(
        rep_ds_preloaded,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\n訓練批次數: {len(train_loader)}")
    print(f"驗證批次數: {len(val_loader)}")
    print(f"代表批次數: {len(rep_loader)}")
    
    # ===== 初始化模型 =====
    print("\n正在初始化模型...")
    model = create_model()
    
    # ===== 優化器與學習率調度器 =====
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',      # 監控 IoU（越大越好）
        factor=0.5,      # 學習率降為原來的一半
        patience=2       # 連續 2 個 epoch 沒改善就降低學習率
    )
    scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))
    
    # ===== 訓練迴圈 =====
    print("\n" + "=" * 60)
    print("開始訓練...")
    print("=" * 60)
    
    best_val_iou = 0.0
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = perf_counter()
        
        # 訓練一個 epoch
        train_loss, train_iou, train_bt = train_one_epoch(
            train_loader, model, optimizer, scaler, DEVICE
        )
        
        # 驗證
        val_loss, val_iou = evaluate(val_loader, model, DEVICE)
        rep_loss, rep_iou = evaluate(rep_loader, model, DEVICE)
        
        # 更新學習率
        lr_scheduler.step(val_iou)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = perf_counter() - epoch_start
        
        # 輸出訓練進度
        print(
            f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
            f"訓練損失: {train_loss:.4f} 訓練IoU: {train_iou:.4f} | "
            f"驗證損失: {val_loss:.4f} 驗證IoU: {val_iou:.4f} | "
            f"代表IoU: {rep_iou:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{epoch_time:.1f}秒"
        )
        
        # 儲存最佳模型
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            print(f"  --> 已儲存最佳模型 (IoU: {val_iou:.4f})")
    
    # ===== 訓練完成，儲存最終檢查點 =====
    print("\n" + "=" * 60)
    print("訓練完成！")
    print("=" * 60)
    
    final_ckpt = {
        "epoch": NUM_EPOCHS,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "val_iou": val_iou,
        "best_val_iou": best_val_iou,
    }
    torch.save(final_ckpt, MODEL_DIR / "final_checkpoint.pt")
    print(f"最終檢查點已儲存至: {MODEL_DIR / 'final_checkpoint.pt'}")
    print(f"最佳模型已儲存至: {MODEL_DIR / 'best_model.pt'}")
    print(f"最佳驗證 IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()