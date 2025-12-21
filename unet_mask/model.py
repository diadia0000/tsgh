"""
模型模組：UNet 模型定義、損失函數、評估指標

三類別分割：
- 0 = 背景（細胞外區域）
- 1 = 細胞內部
- 2 = 細胞膜（棕色 DAB 染色）
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from config import ENCODER_NAME, ENCODER_WEIGHTS, IN_CHANNELS, NUM_CLASSES, DEVICE


def create_model():
    """
    建立並回傳 UNet 模型
    
    使用 EfficientNet-B4 作為編碼器，載入 ImageNet 預訓練權重
    輸出 3 個通道，對應三個類別
    """
    model = smp.Unet(
        encoder_name=ENCODER_NAME,       # 編碼器架構
        encoder_weights=ENCODER_WEIGHTS,  # 預訓練權重
        in_channels=IN_CHANNELS,          # 輸入通道數
        classes=NUM_CLASSES,              # 輸出類別數（3）
        activation=None                   # 輸出原始 logits
    )
    return model.to(DEVICE)


# ================= 損失函數 =================

class MultiClassSegmentationLoss(nn.Module):
    """
    多類別分割組合損失函數
    
    結合 Dice Loss 和 Cross Entropy Loss：
    - Dice Loss：處理類別不平衡問題
    - Cross Entropy：提供穩定的梯度
    """
    
    def __init__(self, class_weights=None):
        """
        參數：
            class_weights: 各類別權重，用於處理類別不平衡
                          例如 [1.0, 2.0, 3.0] 表示細胞膜權重最高
        """
        super().__init__()
        
        # 多類別 Dice Loss
        self.dice_loss = smp.losses.DiceLoss(
            mode='multiclass',
            from_logits=True
        )
        
        # Cross Entropy Loss（可加入類別權重）
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
            self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算組合損失
        
        參數：
            pred: 模型預測 [B, NUM_CLASSES, H, W]（logits）
            target: 真實標籤 [B, NUM_CLASSES, H, W]（one-hot）
            
        回傳：
            組合損失值
        """
        # Dice Loss 需要 one-hot target
        dice = self.dice_loss(pred, target)
        
        # Cross Entropy 需要 class indices target
        target_indices = target.argmax(dim=1)  # [B, H, W]
        ce = self.ce_loss(pred, target_indices)
        
        return dice + ce


# 建立損失函數實例（給細胞膜較高權重，因為它通常面積最小）
combined_loss = MultiClassSegmentationLoss(class_weights=[1.0, 1.5, 2.0])


# ================= 評估指標 =================

def iou_score(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    eps: float = 1e-6
) -> dict:
    """
    計算多類別 IoU（Intersection over Union）分數
    
    參數：
        pred: 模型預測 [B, NUM_CLASSES, H, W]（logits）
        target: 真實標籤 [B, NUM_CLASSES, H, W]（one-hot）
        eps: 防止除零的小數值
        
    回傳：
        字典包含：
        - 'mean': 平均 IoU（mIoU）
        - 'background': 背景 IoU
        - 'inside': 細胞內部 IoU
        - 'membrane': 細胞膜 IoU
    """
    # 轉換為預測類別
    pred_classes = pred.argmax(dim=1)  # [B, H, W]
    target_classes = target.argmax(dim=1)  # [B, H, W]
    
    ious = []
    class_names = ['background', 'inside', 'membrane']
    result = {}
    
    for cls_id in range(NUM_CLASSES):
        pred_mask = (pred_classes == cls_id).float()
        target_mask = (target_classes == cls_id).float()
        
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum() - intersection
        
        iou = (intersection + eps) / (union + eps)
        ious.append(iou.item())
        result[class_names[cls_id]] = iou.item()
    
    result['mean'] = sum(ious) / len(ious)
    
    return result


def mean_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    計算平均 IoU（簡化版，只回傳 mIoU）
    
    用於訓練過程中的快速評估
    """
    return iou_score(pred, target)['mean']
