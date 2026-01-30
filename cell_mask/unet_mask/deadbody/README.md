# HER2 語義分割訓練指南

## 專案結構

```
unet_mask/
├── config.py              # 配置檔案 (路徑、超參數)
├── dataset_generator.py   # 批量生成 pseudo mask
├── dataset.py            # Dataset 和 DataLoader
├── utils.py              # 損失函數、指標、工具
├── train.py              # 訓練腳本
├── inference.py          # 推論腳本
├── her2_mask.py          # HER2 DAB mask 生成
└── membrane_interior_segmentation.py  # 細胞膜/內部分割
```

## 快速開始

### 1. 啟動 Python 環境

```bash
source /home/sec312/tsgh/.venv/bin/activate
cd /home/sec312/tsgh/unet_mask
```

### 2. 生成 Pseudo Masks (預處理)

首次訓練前，需要將 24,000 張 HER2 影像預處理為 0-1-2 標籤的 PNG mask：

```bash
python3 dataset_generator.py
```

**參數選項:**
```bash
python3 dataset_generator.py \
    --input_dir /home/sec312/tsgh/unet_mask/tile/train/her2/ \
    --output_dir /home/sec312/tsgh/unet_mask/presudomask/her2/ \
    --membrane_thickness 5 \
    --min_interior_size 100 \
    --num_workers 8
```

**驗證生成的 masks:**
```bash
python3 dataset_generator.py --verify
```

### 3. 訓練模型

```bash
python3 train.py
```

**參數選項:**
```bash
python3 train.py \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --no_amp  # 如需停用混合精度訓練
```

**訓練輸出:**
- 模型 checkpoint: `/home/sec312/tsgh/unet_mask/models/`
- 訓練日誌: `/home/sec312/tsgh/unet_mask/logs/`

### 4. 推論

**單張影像推論:**
```bash
python3 inference.py \
    --input /path/to/image.tiff \
    --output ./inference_output
```

**批量推論:**
```bash
python3 inference.py \
    --input /path/to/image_folder \
    --output ./inference_output
```

**指定 checkpoint:**
```bash
python3 inference.py \
    --checkpoint /home/sec312/tsgh/unet_mask/models/checkpoint_xxx.pth \
    --input /path/to/image.tiff \
    --output ./inference_output
```

## 配置說明

修改 `config.py` 可自訂以下設定：

### 路徑設定
```python
train_image_dir = "/home/sec312/tsgh/unet_mask/tile/train/her2/"
mask_dir = "/home/sec312/tsgh/unet_mask/presudomask/her2/"
model_save_dir = "/home/sec312/tsgh/unet_mask/models/"
```

### 訓練參數
```python
batch_size = 4                    # 針對 32GB 顯存優化
gradient_accumulation_steps = 2   # 有效 batch size = 8
epochs = 100
learning_rate = 1e-4
```

### 類別權重 (處理類別不平衡)
```python
class_weights = [0.5, 1.0, 3.0]  # [背景, 內部, 細胞膜]
```

## 標籤定義

| 數值 | 類別 | 描述 |
|------|------|------|
| 0 | Background | 背景 |
| 1 | Interior | 細胞內部 (Cytoplasm/Nucleus) |
| 2 | Membrane | 細胞膜 (HER2 染色區域) |

## 模型架構

- **架構**: Unet++ 
- **編碼器**: Swin-Transformer Base (`swin_base_patch4_window7_224`)
- **預訓練權重**: ImageNet
- **參數量**: ~96M

## 損失函數

結合兩種損失函數以應對細胞膜類別不平衡：

- **Dice Loss**: 權重 0.5
- **Cross-Entropy Loss**: 權重 0.5
- **類別權重**: 背景 0.5, 內部 1.0, 細胞膜 3.0

## 硬體優化

針對 NVIDIA RTX 5090 32GB + Intel Ultra 265K + 64GB RAM：

- ✅ 混合精度訓練 (AMP)
- ✅ pin_memory=True
- ✅ num_workers=8
- ✅ 梯度累積 (有效 batch size = 8)
- ✅ 顯存自動清理

## 數據增強

訓練時使用以下增強策略：

- RandomRotate90
- HorizontalFlip / VerticalFlip
- ColorJitter (病理影像顏色抖動)
- GaussianBlur
- ElasticTransform
- HueSaturationValue

## 評估指標

- **mIoU**: 各類別 IoU 的平均值
- **Dice Score**: 各類別的 Dice 係數
- **損失**: Dice Loss + Cross-Entropy Loss

## 早停機制

當驗證集 mIoU 連續 15 個 epoch 沒有改善時自動停止訓練。

## 注意事項

1. **首次訓練前**必須先運行 `dataset_generator.py` 生成 masks
2. 確保有足夠的磁碟空間存放生成的 masks (~1GB)
3. 建議在 `screen` 或 `tmux` 中運行訓練，防止 SSH 斷線
4. Checkpoint 會自動保存最佳的 3 個模型
