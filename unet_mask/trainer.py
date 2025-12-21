"""
訓練器模組：訓練和評估函數
"""
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from time import perf_counter

from model import combined_loss, mean_iou


def train_one_epoch(loader, model, optimizer, scaler, device):
    """
    訓練模型一個 epoch
    
    使用混合精度訓練（AMP）加速訓練過程
    
    參數：
        loader: 訓練資料載入器
        model: 要訓練的模型
        optimizer: 優化器
        scaler: GradScaler（用於 AMP）
        device: 訓練裝置（cuda/cpu）
        
    回傳：
        (平均損失, 平均 mIoU, 平均批次時間) 的元組
    """
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    batch_times = []
    
    for imgs, masks in loader:
        t0 = perf_counter()
        
        # 將資料移動到指定裝置
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # 清除梯度（使用 set_to_none 更高效）
        optimizer.zero_grad(set_to_none=True)
        
        # AMP 前向傳播（自動混合精度）
        with autocast():
            preds = model(imgs)
            loss = combined_loss(preds, masks)
        
        # 縮放後的反向傳播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 累計損失
        running_loss += loss.item() * imgs.size(0)
        
        # 計算 mIoU（不計算梯度）
        with torch.no_grad():
            running_iou += mean_iou(preds.detach(), masks)
        
        batch_times.append(perf_counter() - t0)
    
    # 計算平均值
    avg_loss = running_loss / len(loader.dataset)
    avg_iou = running_iou / len(loader)
    avg_bt = float(np.mean(batch_times)) if batch_times else 0.0
    
    return avg_loss, avg_iou, avg_bt


@torch.no_grad()
def evaluate(loader, model, device):
    """
    在驗證資料上評估模型
    
    使用 @torch.no_grad() 裝飾器禁用梯度計算，節省記憶體
    
    參數：
        loader: 驗證資料載入器
        model: 要評估的模型
        device: 裝置
        
    回傳：
        (平均損失, 平均 mIoU) 的元組
    """
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # AMP 推論
        with autocast():
            preds = model(imgs)
            loss = combined_loss(preds, masks)
        
        running_loss += loss.item() * imgs.size(0)
        running_iou += mean_iou(preds, masks)
    
    avg_loss = running_loss / len(loader.dataset)
    avg_iou = running_iou / len(loader)
    
    return avg_loss, avg_iou
