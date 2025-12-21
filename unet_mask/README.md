# UNet 三類別細胞分割模型

這是一個使用 UNet 架構進行 HER2 細胞分割的深度學習專案。

## 🎯 目標

分割 HER2+SISH 疊合圖中的三個區域：
- **背景**（類別 0）：細胞外區域
- **細胞內部**（類別 1）：用於計算 SISH 紅黑點
- **細胞膜**（類別 2）：HER2 陽性區域（棕色 DAB 染色）

## 📁 專案結構

```
unet_mask/
├── config.py           # 設定檔：路徑、超參數、模型設定
├── dataset.py          # 資料集類別定義
├── mask_generation.py  # Pseudo mask 自動生成
├── model.py            # UNet 模型與損失函數
├── train.py            # 🚀 主訓練程式（入口點）
├── trainer.py          # 訓練和評估函數
├── transforms.py       # 資料增強流程
├── utils.py            # 輔助工具函數
├── README.md           # 本文件
└── process/            # 資料與輸出目錄
    ├── train-512-lv1/  # 代表性評估圖像
    ├── tile/           # 完整訓練資料
    ├── output/         # Mask 快取
    ├── pseudo_masks/   # 生成的 pseudo masks
    └── models/         # 儲存的模型
```

---

## 🔧 各檔案功能說明

### `train.py` - 主程式（入口點）
**功能**：整個訓練流程的入口點

**執行方式**：
```bash
python train.py
```

**流程**：
1. 初始化設定與目錄
2. 載入並預處理資料集
3. 將資料預載入 RAM（加速訓練）
4. 建立 UNet 模型與優化器
5. 執行訓練迴圈
6. 自動儲存最佳模型

---

### `config.py` - 設定檔
**功能**：集中管理所有可調參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `CROP_SIZE` | 輸入圖像尺寸 | 512 |
| `BATCH_SIZE` | 批次大小 | 32 |
| `NUM_EPOCHS` | 訓練輪數 | 30 |
| `LEARNING_RATE` | 學習率 | 5e-4 |
| `WEIGHT_DECAY` | L2 正則化 | 1e-4 |
| `TRAIN_VAL_SPLIT` | 訓練集比例 | 0.9 |
| `ENCODER_NAME` | 編碼器架構 | efficientnet-b4 |

---

### `dataset.py` - 資料集模組
**功能**：定義 PyTorch Dataset 類別

**兩個主要類別**：

1. **`LargeImagePseudoMaskDataset`**
   - 自動掃描目錄下的所有圖像
   - 首次存取時生成 pseudo mask 並快取
   - 支援即時裁切和資料增強

2. **`PreloadedDataset`**
   - 將整個資料集預載入 RAM
   - 消除 I/O 瓶頸，大幅加速訓練

---

### `mask_generation.py` - Pseudo Mask 生成
**功能**：自動生成細胞分割的訓練標籤

**處理流程**：
```
原始圖像
    ↓
排除紅色信號點（SISH 染色）
    ↓
排除黑色信號點（SISH 染色）
    ↓
檢測棕色區域（HER2 DAB 染色）
    ↓ HSV + Lab 雙重驗證
形態學處理（閉合、開運算）
    ↓
孔洞填充
    ↓
輸出三類別 mask
```

**輸出類別**：
- `0` = 背景（細胞外區域）
- `1` = 細胞內部
- `2` = 細胞膜（棕色 DAB 染色區域）

**獨立執行**（批次生成 masks）：
```bash
python mask_generation.py
```

---

### `model.py` - 模型模組
**功能**：定義 UNet 模型架構、損失函數、評估指標

**模型架構**：
- 使用 `segmentation_models_pytorch` 套件
- 編碼器：EfficientNet-B4（ImageNet 預訓練）
- 輸出：二元分割（細胞膜 vs 其他）

**損失函數**：
- **Dice Loss**：直接優化 IoU，處理類別不平衡
- **Focal Loss**：聚焦於難分類的邊界像素
- 最終損失 = Dice Loss + Focal Loss

**評估指標**：
- **IoU (Intersection over Union)**：分割任務標準指標

---

### `trainer.py` - 訓練器模組
**功能**：封裝訓練和評估邏輯

**主要函數**：
- `train_one_epoch()`：訓練一個 epoch
- `evaluate()`：在驗證集上評估模型

**特點**：
- 使用 AMP（自動混合精度）加速訓練
- 支援 GPU 和 CPU

---

### `transforms.py` - 資料增強模組
**功能**：定義資料增強流程

**訓練增強**：
| 操作 | 說明 |
|------|------|
| `PadIfNeeded` | 填充到最小尺寸 |
| `RandomCrop` | 隨機裁切（增加多樣性） |
| `ColorJitter` | 顏色抖動（模擬不同染色條件） |
| `ElasticTransform` | 彈性變形（模擬組織變形） |
| `HorizontalFlip` | 水平翻轉 |
| `VerticalFlip` | 垂直翻轉 |
| `Normalize` | ImageNet 標準化 |

**驗證增強**：
- 只做確定性轉換（中心裁切、標準化）

---

### `utils.py` - 工具模組
**功能**：圖像處理輔助函數

- `read_rgb_uint8()`：讀取圖像並轉換為 RGB uint8 格式

---

## 🚀 快速開始

### 1. 安裝依賴
```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install opencv-python
pip install scikit-image
pip install scipy
pip install matplotlib
```

### 2. 準備資料
將訓練圖像放入對應目錄：
```
unet_mask/process/
├── train-512-lv1/     # 代表性圖像（用於評估）
│   ├── blank/
│   ├── negative/
│   ├── strong/
│   └── weak/
└── tile/level-train-batch4/  # 完整訓練資料
```

### 3. 生成 Pseudo Masks（可選）
如果要預先生成 masks：
```bash
python mask_generation.py
```

### 4. 開始訓練
```bash
python train.py
```

---

## 📊 訓練過程

訓練時會顯示以下資訊：
```
Epoch  1/30 | 訓練損失: 0.4521 訓練IoU: 0.5234 | 驗證損失: 0.3892 驗證IoU: 0.6123 | 代表IoU: 0.5892 | LR: 5.00e-04 | 45.2秒
  --> 已儲存最佳模型 (IoU: 0.6123)
```

| 欄位 | 說明 |
|------|------|
| 訓練損失 | 訓練集的平均損失 |
| 訓練IoU | 訓練集的 IoU 分數 |
| 驗證損失 | 驗證集的平均損失 |
| 驗證IoU | 驗證集的 IoU 分數 |
| 代表IoU | 代表性評估集的 IoU |
| LR | 當前學習率 |

---

## 📦 輸出檔案

訓練完成後，模型會儲存在：
```
unet_mask/process/models/
├── best_model.pt          # 最佳驗證 IoU 的模型
└── final_checkpoint.pt    # 最終檢查點（含優化器狀態）
```

---

## 🔬 技術細節

### 為什麼使用 Pseudo Mask？
- 標註細胞膜非常耗時且昂貴
- 利用 DAB 染色的顏色特徵自動生成標籤
- 這是一種**弱監督學習**方法

### 模型選擇
- **UNet**：經典的分割架構，適合醫學影像
- **EfficientNet-B4**：平衡效能與效率的編碼器
- **預訓練權重**：使用 ImageNet 預訓練加速收斂

### 損失函數設計
- **Dice Loss**：針對類別不平衡（細胞膜面積小）
- **Focal Loss**：強調邊界等難分類區域

---

## ❓ 常見問題

### Q: 記憶體不足？
修改 `config.py`：
```python
BATCH_SIZE = 16  # 減小批次大小
```

### Q: 訓練速度太慢？
1. 確保使用 GPU：`torch.cuda.is_available()` 應返回 `True`
2. 資料已預載入 RAM，I/O 不應是瓶頸

### Q: IoU 太低？
1. 檢查 pseudo mask 品質：`python mask_generation.py` 查看視覺化
2. 調整 `mask_generation.py` 中的 HSV/Lab 閾值
3. 增加訓練輪數

---

## 📝 作者備註

這個專案用於 HER2 病理圖像的細胞膜分割。主要目的是從免疫組織化學（IHC）染色的病理切片中自動識別 HER2 陽性的細胞膜區域。
