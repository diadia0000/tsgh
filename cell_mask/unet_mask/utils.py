"""
HER2 語義分割訓練工具函數

包含:
- 損失函數 (Dice Loss + Cross-Entropy Loss)
- 評估指標 (mIoU, Dice Score)
- 模型保存/載入
- 訓練日誌
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class DiceLoss(nn.Module):
    """
    Dice Loss
    
    用於處理類別不平衡問題，特別適合分割任務
    """
    
    def __init__(self, num_classes: int = 3, smooth: float = 1e-6, weight: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.weight = weight  # 類別權重
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 預測結果 (B, C, H, W) - logits
            target: 真實標籤 (B, H, W) - 類別索引
            
        Returns:
            dice_loss: Dice 損失值
        """
        # 將 logits 轉為機率
        pred_prob = F.softmax(pred, dim=1)
        
        # One-hot 編碼 target
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # 計算每個類別的 Dice
        intersection = (pred_prob * target_one_hot).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # 應用類別權重
        if self.weight is not None:
            weight = self.weight.to(dice.device)
            dice = dice * weight.unsqueeze(0)
            dice_loss = 1 - dice.sum(dim=1) / weight.sum()
        else:
            dice_loss = 1 - dice.mean(dim=1)
        
        return dice_loss.mean()


class CombinedLoss(nn.Module):
    """
    結合 Dice Loss 和 Cross-Entropy Loss
    
    用於應對細胞膜類別佔比極小的類別不平衡問題
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: List[float] = None
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.num_classes = num_classes
        
        # 使用 register_buffer 確保權重隨模型一起移動到 GPU
        if class_weights is not None:
            self.register_buffer('class_weights', torch.FloatTensor(class_weights))
        else:
            self.class_weights = None
        
        self.dice_loss = DiceLoss(num_classes=num_classes, weight=None)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred: 預測結果 (B, C, H, W)
            target: 真實標籤 (B, H, W)
            
        Returns:
            total_loss: 總損失
            loss_dict: 各損失項的值
        """
        # Dice Loss
        dice = self.dice_loss(pred, target)
        
        # Cross-Entropy Loss (權重會自動在正確的設備上)
        ce = F.cross_entropy(pred, target, weight=self.class_weights)
        
        total = self.dice_weight * dice + self.ce_weight * ce
        
        loss_dict = {
            "dice_loss": dice.item(),
            "ce_loss": ce.item(),
            "total_loss": total.item()
        }
        
        return total, loss_dict


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> Dict[str, float]:
    """
    計算 IoU (Intersection over Union)
    
    Args:
        pred: 預測結果 (B, C, H, W) - logits
        target: 真實標籤 (B, H, W)
        num_classes: 類別數量
        
    Returns:
        iou_dict: 各類別的 IoU 和 mIoU
    """
    # 獲取預測類別
    pred_classes = pred.argmax(dim=1)
    
    ious = []
    iou_dict = {}
    
    for c in range(num_classes):
        pred_c = (pred_classes == c)
        target_c = (target == c)
        
        intersection = (pred_c & target_c).float().sum()
        union = (pred_c | target_c).float().sum()
        
        if union > 0:
            iou = (intersection / union).item()
        else:
            iou = 1.0 if intersection == 0 else 0.0
        
        ious.append(iou)
        iou_dict[f"iou_class_{c}"] = iou
    
    iou_dict["miou"] = np.mean(ious)
    
    return iou_dict


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> Dict[str, float]:
    """
    計算 Dice Score
    
    Args:
        pred: 預測結果 (B, C, H, W) - logits
        target: 真實標籤 (B, H, W)
        num_classes: 類別數量
        
    Returns:
        dice_dict: 各類別的 Dice Score 和平均 Dice
    """
    pred_classes = pred.argmax(dim=1)
    
    dices = []
    dice_dict = {}
    
    for c in range(num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        total = pred_c.sum() + target_c.sum()
        
        if total > 0:
            dice = (2 * intersection / total).item()
        else:
            dice = 1.0 if intersection == 0 else 0.0
        
        dices.append(dice)
        dice_dict[f"dice_class_{c}"] = dice
    
    dice_dict["mean_dice"] = np.mean(dices)
    
    return dice_dict


class MetricsTracker:
    """訓練指標追蹤器"""
    
    def __init__(self, num_classes: int = 3, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """重置所有指標"""
        self.total_loss = 0.0
        self.dice_loss = 0.0
        self.ce_loss = 0.0
        self.iou_sum = [0.0] * self.num_classes
        self.dice_sum = [0.0] * self.num_classes
        self.count = 0
    
    def update(
        self,
        loss_dict: Dict[str, float],
        iou_dict: Dict[str, float],
        dice_dict: Dict[str, float]
    ):
        """更新指標"""
        self.total_loss += loss_dict["total_loss"]
        self.dice_loss += loss_dict["dice_loss"]
        self.ce_loss += loss_dict["ce_loss"]
        
        for c in range(self.num_classes):
            self.iou_sum[c] += iou_dict[f"iou_class_{c}"]
            self.dice_sum[c] += dice_dict[f"dice_class_{c}"]
        
        self.count += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """獲取平均指標"""
        if self.count == 0:
            return {}
        
        metrics = {
            "total_loss": self.total_loss / self.count,
            "dice_loss": self.dice_loss / self.count,
            "ce_loss": self.ce_loss / self.count,
        }
        
        # 各類別 IoU
        ious = []
        for c in range(self.num_classes):
            iou = self.iou_sum[c] / self.count
            metrics[f"iou_{self.class_names[c]}"] = iou
            ious.append(iou)
        metrics["miou"] = np.mean(ious)
        
        # 各類別 Dice
        dices = []
        for c in range(self.num_classes):
            dice = self.dice_sum[c] / self.count
            metrics[f"dice_{self.class_names[c]}"] = dice
            dices.append(dice)
        metrics["mean_dice"] = np.mean(dices)
        
        return metrics


class CheckpointManager:
    """模型 Checkpoint 管理器"""
    
    def __init__(
        self,
        save_dir: str,
        monitor_metric: str = "val_miou",
        monitor_mode: str = "max",
        save_top_k: int = 3
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.save_top_k = save_top_k
        
        self.best_metrics = []
        self.checkpoint_paths = []
    
    def _is_better(self, current: float, best: float) -> bool:
        """判斷是否更好"""
        if self.monitor_mode == "max":
            return current > best
        return current < best
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler,
        epoch: int,
        metrics: Dict[str, float],
        model_config: Dict[str, any] = None
    ) -> bool:
        """
        保存 Checkpoint
        
        Returns:
            is_best: 是否是最佳模型
        """
        current_metric = metrics.get(self.monitor_metric, 0.0)
        
        # 檢查是否應該保存
        should_save = False
        is_best = False
        
        if len(self.best_metrics) < self.save_top_k:
            should_save = True
            is_best = len(self.best_metrics) == 0 or self._is_better(current_metric, self.best_metrics[0])
        else:
            worst_idx = 0 if self.monitor_mode == "max" else -1
            worst_metric = self.best_metrics[worst_idx]
            
            if self._is_better(current_metric, worst_metric):
                should_save = True
                is_best = self._is_better(current_metric, self.best_metrics[-1 if self.monitor_mode == "max" else 0])
                
                # 刪除最差的 checkpoint
                old_path = self.checkpoint_paths[worst_idx]
                if old_path.exists():
                    old_path.unlink()
                
                self.best_metrics.pop(worst_idx)
                self.checkpoint_paths.pop(worst_idx)
        
        if should_save:
            # 保存 checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "metrics": metrics,
                "model_config": model_config or {}
            }
            
            filename = f"checkpoint_epoch{epoch:03d}_{self.monitor_metric}_{current_metric:.4f}.pth"
            filepath = self.save_dir / filename
            
            torch.save(checkpoint, filepath)
            
            # 更新追蹤列表
            self.best_metrics.append(current_metric)
            self.checkpoint_paths.append(filepath)
            
            # 排序
            if self.monitor_mode == "max":
                pairs = sorted(zip(self.best_metrics, self.checkpoint_paths), reverse=True)
            else:
                pairs = sorted(zip(self.best_metrics, self.checkpoint_paths))
            
            self.best_metrics = [p[0] for p in pairs]
            self.checkpoint_paths = [p[1] for p in pairs]
            
            print(f"  💾 Checkpoint 已保存: {filename}")
        
        return is_best
    
    def load_best_checkpoint(self, model: nn.Module) -> Dict:
        """載入最佳 Checkpoint"""
        if len(self.checkpoint_paths) == 0:
            raise ValueError("沒有可用的 checkpoint")
        
        best_path = self.checkpoint_paths[0]
        print(f"載入最佳 checkpoint: {best_path}")
        
        checkpoint = torch.load(best_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return checkpoint


class TrainingLogger:
    """訓練日誌記錄器"""
    
    def __init__(self, log_dir: str, experiment_name: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}_training.json"
        
        self.history = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "epochs": []
        }
    
    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float):
        """記錄一個 epoch"""
        epoch_data = {
            "epoch": epoch,
            "learning_rate": lr,
            "train": train_metrics,
            "val": val_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        self.history["epochs"].append(epoch_data)
        self._save()
    
    def _save(self):
        """保存日誌"""
        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def print_epoch_summary(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float):
        """列印 epoch 摘要"""
        print(f"\nEpoch {epoch} 摘要:")
        print(f"  學習率: {lr:.6f}")
        print(f"  訓練 - Loss: {train_metrics['total_loss']:.4f}, mIoU: {train_metrics['miou']:.4f}")
        print(f"  驗證 - Loss: {val_metrics['total_loss']:.4f}, mIoU: {val_metrics['miou']:.4f}")
        print(f"  訓練 IoU - 背景: {train_metrics['iou_Background']:.4f}, "
              f"內部: {train_metrics['iou_Interior']:.4f}, 膜: {train_metrics['iou_Membrane']:.4f}")
        print(f"  驗證 IoU - 背景: {val_metrics['iou_Background']:.4f}, "
              f"內部: {val_metrics['iou_Interior']:.4f}, 膜: {val_metrics['iou_Membrane']:.4f}")


class TeeLogger:
    """
    同時將輸出導向 terminal 和 txt 檔案的日誌記錄器
    
    使用方式:
        logger = TeeLogger(log_dir, experiment_name)
        logger.start()  # 開始捕獲所有 print 輸出
        ... 訓練過程 ...
        logger.stop()   # 停止捕獲，恢復正常輸出
    """
    
    def __init__(self, log_dir: str, experiment_name: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.txt_log_file = self.log_dir / f"{experiment_name}_console.txt"
        
        self._original_stdout = None
        self._original_stderr = None
        self._file = None
    
    def start(self):
        """開始捕獲輸出"""
        import sys
        
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._file = open(self.txt_log_file, "w", encoding="utf-8", buffering=1)  # line buffering
        
        sys.stdout = _TeeStream(self._original_stdout, self._file)
        sys.stderr = _TeeStream(self._original_stderr, self._file)
        
        # 寫入開始時間
        start_msg = f"=" * 60 + "\n"
        start_msg += f"訓練日誌開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        start_msg += f"日誌檔案: {self.txt_log_file}\n"
        start_msg += f"=" * 60 + "\n"
        self._file.write(start_msg)
        self._file.flush()
    
    def stop(self):
        """停止捕獲輸出，恢復正常"""
        import sys
        
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr
        
        if self._file is not None:
            # 寫入結束時間
            end_msg = f"\n" + "=" * 60 + "\n"
            end_msg += f"訓練日誌結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            end_msg += f"=" * 60 + "\n"
            self._file.write(end_msg)
            self._file.close()
            self._file = None
    
    def __enter__(self):
        """支援 with 語法"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支援 with 語法"""
        self.stop()
        return False


class _TeeStream:
    """
    內部類別：同時寫入兩個串流
    """
    
    def __init__(self, stream1, stream2):
        self.stream1 = stream1  # terminal
        self.stream2 = stream2  # file
    
    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)
        self.stream1.flush()
        self.stream2.flush()
    
    def flush(self):
        self.stream1.flush()
        self.stream2.flush()
    
    def isatty(self):
        return self.stream1.isatty()


def clear_cuda_cache():
    """清理 CUDA 快取"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
