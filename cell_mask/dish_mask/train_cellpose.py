#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cellpose 訓練腳本
使用 Cellpose GUI 標註的 1024x1024 tiles 進行訓練
"""

import os
import sys
import shutil
import logging
import numpy as np
import albumentations as A
from pathlib import Path
from skimage.io import imread
from cellpose import models, train

# 設定 logging 讓訓練進度可見
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# 確保 cellpose 的 logger 也輸出
logging.getLogger('cellpose').setLevel(logging.INFO)

# ==================== 配置參數 ====================

BASE_DIR = Path(__file__).resolve().parent

# 訓練資料目錄
TRAIN_DATA_DIR = BASE_DIR / "train" / "train_data"      # npy 標註檔
TRAIN_IMAGE_DIR = BASE_DIR / "train" / "train_picture"   # tiff 圖片檔
MODEL_DIR = str(BASE_DIR / "models")                     # 模型輸出目錄

# 訓練參數
MODEL_NAME = "cellpose"   # 模型名稱
# 使用之前訓練的模型繼續訓練 (設為 None 或 "cyto3" 則從頭開始)
INITIAL_MODEL = "cyto3"  # 繼續訓練
N_EPOCHS = 200                    # 訓練輪數
LEARNING_RATE = 5e-6              # 繼續訓練時使用較小學習率
WEIGHT_DECAY = 0.1                # 權重衰減
BATCH_SIZE = 16                    # 批次大小
MIN_TRAIN_MASKS = 4               # 最少訓練 mask 數量

# 資料增強參數
SCALE_RANGE = 1.0                 # 縮放範圍: 圖片會縮放 (1-range/2) ~ (1+range/2)，即 0.5x ~ 1.5x
RESCALE = True                    # 正規化細胞到統一大小
USE_COLOR_AUGMENTATION = True     # 是否使用顏色增強
N_AUG_COPIES = 3                  # 每張圖片的增強副本數 (總訓練量 = 原始 + N 份增強)
SAVE_EVERY = 10                   # 每 N 個 epoch 儲存 checkpoint

# 顏色增強設定 (適合組織染色圖像)
COLOR_AUGMENTATION = A.Compose([
    A.ColorJitter(
        brightness=0.15,          # 亮度變化 ±15%
        contrast=0.15,            # 對比度變化 ±15%
        saturation=0.1,           # 飽和度變化 ±10%
        hue=0.05,                 # 色調變化 ±5%
        p=0.5                     # 50% 機率套用
    ),
    A.GaussNoise(
        std_range=(0.02, 0.1),    # 高斯雜訊 (正規化範圍 0-1)
        p=0.3                     # 30% 機率套用
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.1,     # 額外亮度調整
        contrast_limit=0.1,       # 額外對比度調整
        p=0.3                     # 30% 機率套用
    ),
])


# ==================== 資料增強函數 ====================

def augment_single_image(img: np.ndarray) -> np.ndarray:
    """對單張圖片套用顏色增強。

    Args:
        img: HWC 格式的圖片 (uint8 或 float)。

    Returns:
        增強後的 uint8 圖片。
    """
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
    return COLOR_AUGMENTATION(image=img)['image']


def expand_with_augmentation(
    images: list[np.ndarray],
    masks: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """產生 N 份顏色增強副本來擴充訓練集。

    每份副本的隨機增強參數不同，搭配 Cellpose 內建的幾何增強，
    讓模型在每個 epoch 都能看到不同的顏色變化。

    Args:
        images: 原始圖片列表。
        masks: 對應的 mask 列表。

    Returns:
        (擴充後的圖片列表, 擴充後的 mask 列表)。
    """
    if not USE_COLOR_AUGMENTATION or N_AUG_COPIES <= 0:
        return images, masks

    aug_images = list(images)    # 先保留原始圖片
    aug_masks = list(masks)

    for copy_idx in range(N_AUG_COPIES):
        for img, mask in zip(images, masks):
            aug_images.append(augment_single_image(img))
            aug_masks.append(mask)  # mask 不需要顏色增強
        print(f"  增強副本 {copy_idx + 1}/{N_AUG_COPIES}: +{len(images)} 張")

    print(f"  總計: {len(images)} 原始 + {len(images) * N_AUG_COPIES} 增強 = {len(aug_images)} 張")
    return aug_images, aug_masks


# ==================== 資料準備 ====================

def prepare_training_data() -> tuple[list, list, list]:
    """從 train_data/ 和 train_picture/ 載入訓練資料。

    Returns:
        (images, masks, file_names) 三元組。
    """
    print("=" * 60)
    print("準備訓練資料...")
    print(f"  標註目錄: {TRAIN_DATA_DIR}")
    print(f"  圖片目錄: {TRAIN_IMAGE_DIR}")
    print("=" * 60)

    images = []
    masks = []
    file_names = []

    seg_files = sorted(TRAIN_DATA_DIR.glob("*_seg.npy"))
    print(f"\n找到 {len(seg_files)} 個標註檔")

    for seg_file in seg_files:
        base_name = seg_file.stem.replace("_seg", "")
        # 支援 tiff 和 png 格式
        image_file = TRAIN_IMAGE_DIR / f"{base_name}.tiff"
        if not image_file.exists():
            image_file = TRAIN_IMAGE_DIR / f"{base_name}.png"
        if not image_file.exists():
            print(f"  跳過 (無對應圖片): {base_name}")
            continue

        img = imread(str(image_file))
        seg_data = np.load(str(seg_file), allow_pickle=True).item()
        mask = seg_data.get('masks', None)

        if mask is None:
            continue

        n_objects = len(np.unique(mask)) - 1
        if n_objects < MIN_TRAIN_MASKS:
            continue

        images.append(img)
        masks.append(mask)
        file_names.append(base_name)
        print(f"  {base_name}: {img.shape}, {n_objects} 物件")

    print(f"\n有效訓練資料: {len(images)} 組")
    return images, masks, file_names


# ==================== Best Model 選擇 ====================

def select_best_model(model_dir: str, model_name: str,
                      train_losses: np.ndarray) -> tuple[str, int, float]:
    """從所有 checkpoint 中選出 training loss 最低的作為 best model。

    Args:
        model_dir: 模型儲存目錄。
        model_name: 模型名稱前綴。
        train_losses: 每個 epoch 的 training loss 陣列。

    Returns:
        (best_model_path, best_epoch, best_loss) 三元組。
    """
    model_path = Path(model_dir)

    # 收集所有 checkpoint epoch 及其 loss
    # save_each=True 時，Cellpose 儲存格式: {model_name}_epoch_{epoch:04d}
    # 最後一個 epoch 額外存為 {model_name} (無後綴)
    candidates = {}

    # 最終模型 (最後一個 epoch)
    final_model = model_path / model_name
    if final_model.exists():
        final_epoch = len(train_losses) - 1
        candidates[final_epoch] = str(final_model)

    # 各 checkpoint
    for ckpt in model_path.glob(f"{model_name}_epoch_*"):
        try:
            epoch = int(ckpt.stem.split("_epoch_")[-1])
            candidates[epoch] = str(ckpt)
        except ValueError:
            continue

    if not candidates:
        print("  未找到任何 checkpoint")
        return str(final_model), len(train_losses) - 1, train_losses[-1]

    # 找最低 loss 的 epoch
    best_epoch = min(candidates, key=lambda e: train_losses[e])
    best_loss = train_losses[best_epoch]
    best_src = candidates[best_epoch]

    # 複製為 best model
    best_dst = str(model_path / f"{model_name}_best")
    shutil.copy2(best_src, best_dst)

    print(f"\n  Best model: epoch {best_epoch + 1}, loss = {best_loss:.4f}")
    print(f"  來源: {Path(best_src).name}")
    print(f"  儲存: {best_dst}")

    # 刪除其他 checkpoint，只保留 best
    removed = 0
    for ckpt_path_str in candidates.values():
        if ckpt_path_str != best_dst:
            Path(ckpt_path_str).unlink(missing_ok=True)
            removed += 1
    print(f"  已清除 {removed} 個非最佳 checkpoint")

    return best_dst, best_epoch, best_loss


# ==================== 訓練函數 ====================

def train_cellpose_model(
    images: list[np.ndarray],
    masks: list[np.ndarray],
) -> tuple[str, np.ndarray, np.ndarray]:
    """訓練 Cellpose 模型，含顏色增強擴充與 best model 選擇。

    Args:
        images: 原始訓練圖片。
        masks: 對應的 mask。

    Returns:
        (best_model_path, train_losses, test_losses) 三元組。
    """
    print("\n" + "=" * 60)
    print("開始訓練 Cellpose 模型...")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 初始化模型
    print(f"\n基礎模型: {INITIAL_MODEL}")
    print(f"學習率: {LEARNING_RATE}")
    print(f"訓練輪數: {N_EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")

    if os.path.exists(str(INITIAL_MODEL)):
        print(f"載入已訓練模型: {INITIAL_MODEL}")
        model = models.CellposeModel(
            gpu=True,
            pretrained_model=INITIAL_MODEL
        )
    else:
        print(f"使用預設模型: {INITIAL_MODEL}")
        model = models.CellposeModel(
            gpu=True,
            pretrained_model=INITIAL_MODEL
        )

    # 擴充訓練集 (原始 + N 份顏色增強副本)
    print(f"\n產生顏色增強副本 (x{N_AUG_COPIES})...")
    train_images, train_masks = expand_with_augmentation(images, masks)

    print(f"\n縮放範圍: {SCALE_RANGE}")
    print(f"重新縮放: {RESCALE}")
    print(f"\n開始訓練...")

    # train_seg 會在 save_path 下建立 models 子目錄
    save_base = str(Path(MODEL_DIR).parent)

    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_masks,
        save_path=save_base,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        model_name=MODEL_NAME,
        min_train_masks=MIN_TRAIN_MASKS,
        save_every=SAVE_EVERY,
        save_each=True,             # 每個 checkpoint 存為獨立檔案
        scale_range=SCALE_RANGE,
        rescale=RESCALE,
    )

    # 輸出訓練曲線摘要
    print(f"\n訓練 Loss 範圍: {train_losses.min():.4f} ~ {train_losses.max():.4f}")
    print(f"最終 Loss (epoch {N_EPOCHS}): {train_losses[-1]:.4f}")

    # 選出 best model
    print("\n" + "-" * 40)
    print("選擇最佳模型...")
    best_path, best_epoch, best_loss = select_best_model(
        MODEL_DIR, MODEL_NAME, train_losses
    )

    print("\n" + "=" * 60)
    print("訓練完成!")
    print("=" * 60)
    print(f"最終模型: {model_path}")
    print(f"最佳模型: {best_path} (epoch {best_epoch + 1}, loss {best_loss:.4f})")

    return best_path, train_losses, test_losses


# ==================== 主程式 ====================

def main():
    print("\n")
    print("+" + "=" * 58 + "+")
    print("|" + " " * 15 + "Cellpose 訓練腳本" + " " * 25 + "|")
    print("|" + " " * 15 + "Dish Mask Training" + " " * 24 + "|")
    print("+" + "=" * 58 + "+")
    print()

    # 1. 準備訓練資料
    images, masks, file_names = prepare_training_data()

    if len(images) == 0:
        print("沒有找到有效的訓練資料！")
        return

    # 2. 訓練模型 (含增強擴充 + best model 選擇)
    best_path, train_losses, _ = train_cellpose_model(images, masks)

    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)
    print(f"\n最佳模型位置: {best_path}")
    print(f"可以使用以下命令進行推論:")
    print(f"  python -m cellpose --pretrained_model {best_path} --dir <IMAGE_DIR>")


if __name__ == "__main__":
    main()
