# TSGH 病理影像分析系統 v2.3

本專案為一個自動化的全組織病理影像 (Whole Slide Image, WSI) 分析系統，整合多項先進技術，提供從影像對位到 HER2 染色分析的完整解決方案。

## 主要功能

### 🔬 多染色影像對位

採用 `valis` 函式庫，精準對齊同一組織切片的三種不同染色影像：

- **HE (H&E)** - 蘇木精-伊紅染色
- **Her2** - HER2 免疫組化染色 (作為對位的基準)
- **DISH** - 雙重原位雜交染色

### 🎯 HER2 細胞分割分析

採用 UNet++ 深度學習模型 + Watershed 分割：

- LAB 色彩空間生成僞標籤 (Pseudo Labels)
- UNet++ (DenseNet121 Encoder) 細胞膜分割
- 配置化管理 (`config.py`)

### 🧩 影像磚切割與處理

高效處理 Gigapixel 等級的大型病理影像：

- 自動切割為小型影像磚
- 支援多執行緒並行處理
- 適合後續深度學習分析

## 系統特色

### 全自動化 Python 工作流程

本系統以 `valis` 函式庫為核心，搭配 `aicspylibczi 3.3.1` 處理 CZI 格式影像，提供一個從影像對位到結果驗證的完整 Python 解決方案。

### 高效記憶體管理

- **無需載入完整影像**: 系統利用 `pyvips` 函式庫，直接在磁碟上對 Gigapixel 等級的 CZI 檔案進行處理，無需將整個高解析度影像載入記憶體。
- **金字塔層級處理**: 所有耗時的對位運算都在影像金字塔的低解析度層級完成，大幅提升運算速度並降低硬體需求。
- **串流式影像處理**: 使用 pyvips 的 sequential access 模式，實現真正的串流處理。

### GPU 加速支援

- **CUDA 加速**: 支援使用 CUDA 加速特徵偵測與匹配，大幅提升對位速度。
- **LightGlue + DISK**: 採用最新的 LightGlue 匹配器搭配 DISK 特徵偵測器，提供更準確的對位結果。
- **PyTorch 2.11+**: 支援最新的 PyTorch nightly 版本，搭配 CUDA 12.8。

### Docker 容器化支援

- **完整環境封裝**: 提供 Dockerfile 和 docker-compose.yml，一鍵部署完整環境。
- **GPU 支援**: 容器內完整支援 NVIDIA GPU 加速。
- **跨平台運行**: 在任何支援 Docker 的系統上皆可運行。

### 五階段模組化工作流程

1. **模組 1: 影像對準 (Alignment)** - 使用 valis 內建前處理，計算剛性與非剛性變換參數。
2. **模組 2: ROI 品質評估 (ROI Evaluation)** - 從原始高解析度影像中提取已對齊的感興趣區域。
3. **模組 3: 產生對齊縮圖 (Thumbnail Generation)** - 產生已對齊的全尺寸疊合縮圖。
4. **模組 4: 切割影像磚 (Tile Generation)** - 高效切割對齊後的大型 TIFF 影像為小型磚塊。
5. **模組 5: HER2 細胞分割 (Cell Segmentation)** - UNet++ + Watershed 細胞分割。

## 安裝指南

### 方法一：使用 Docker (推薦)

最簡單的安裝方式，無需手動安裝任何依賴。

#### 前置需求

- 安裝 [Docker](https://docs.docker.com/get-docker/)
- 安裝 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (GPU 支援)

#### 建立與運行容器

```bash
# 克隆專案
git clone <repository-url>
cd tsgh

# 使用 docker compose 建立並啟動容器
docker compose build
docker compose up -d

# 進入容器
docker compose exec tsgh bash

# 停止容器
docker compose down
```

#### 使用傳統 Docker 命令

```bash
# 建立映像
docker build -t tsgh-pytorch:latest .

# 運行容器（支援 GPU）
docker run --gpus all -it --rm \
    -v $(pwd):/app \
    -v /path/to/your/data:/app/data \
    tsgh-pytorch:latest
```

### 方法二：本地安裝

適合開發者或需要自訂環境的使用者。

```bash
# 克隆專案
git clone <repository-url>
cd tsgh

# 建立虛擬環境
python3.11 -m venv .venv

# 啟動虛擬環境
source .venv/bin/activate  # Linux/macOS
# .\.venv\Scripts\activate  # Windows

# 安裝 PyTorch (CUDA 12.8)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 安裝其他依賴
pip install -r requirements.txt
```

### 驗證安裝

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
```

## 使用方法

### 快速開始 (推薦)

執行單一腳本即可完成完整工作流程：

```bash
python thriple_image_layer/run_full_pipeline.py
```

### 分步執行

#### 模組 1: 影像對準

```bash
python thriple_image_layer/module2_alignment.py
```

- **輸入**: `picture/whole_size/40X/` 目錄下的 `*.czi` 檔案
- **輸出**: `thriple_image_layer/output/Transform_Params/` (對位參數)

#### 模組 2: ROI 品質評估

```bash
python thriple_image_layer/module3_roi_evaluation.py
```

- **輸出**: `Merged_ROI.png` + `Metrics.csv`

#### 模組 3: 產生對齊縮圖

```bash
python thriple_image_layer/module4_thumbnail.py
```

- **輸出**: `Merged_Aligned_lv*.tiff`

#### 模組 4: 切割影像磚

```bash
python thriple_image_layer/module5_tile_generator.py
```

- **輸出**: 多個 `tile_x{x}_y{y}.tiff` 檔案

#### 模組 5: HER2 細胞分割

```bash
# 1. 複製配置範例檔
cp cell_mask/unet_mask/config_example.py cell_mask/unet_mask/config.py

# 2. 生成 LAB 僞標籤 (Pseudo Labels)
python cell_mask/unet_mask/lab_mask_generator.py

# 3. 訓練 UNet++ 模型
python cell_mask/unet_mask/train_unetpp.py

# 4. Watershed 細胞分割
python cell_mask/unet_mask/watershed_test.py
```

- **配置**: `cell_mask/unet_mask/config.py`
- **模型輸出**: `cell_mask/unet_mask/output/model/best_model.pth`
- **分割結果**: `cell_mask/unet_mask/output/result/`

## 目錄結構

```
tsgh/
├── thriple_image_layer/             # 主要工作流程目錄
│   ├── module2_alignment.py         # 模組1: 影像對準
│   ├── module3_roi_evaluation.py    # 模組2: ROI 品質評估
│   ├── module4_thumbnail.py         # 模組3: 產生對齊縮圖
│   ├── module5_tile_generator.py    # 模組4: 切割影像磚
│   ├── run_full_pipeline.py         # 完整流程執行腳本
│   ├── reorganize_tiles.py          # 影像磚整理工具
│   ├── check_output.py              # 輸出檢查工具
│   └── output/                      # 結果輸出目錄
│
├── cell_mask/                       # HER2 細胞分割模組
│   ├── unet_mask/                   # UNet++ 訓練與推論
│   │   ├── config_example.py        # 配置範例檔 (複製為 config.py)
│   │   ├── config.py                # 實際配置檔 (git ignored)
│   │   ├── lab_mask_generator.py    # LAB 僞標籤生成器
│   │   ├── train_unetpp.py          # UNet++ 訓練腳本
│   │   ├── watershed_test.py        # Watershed 細胞分割
│   │   ├── train/                   # 訓練影像目錄
│   │   └── output/                  # 輸出目錄 (model, mask, result)
│   └── dish_mask/                   # DISH 影像處理
│
├── scripts/                         # 工具腳本
│   ├── check_tiff_size.py           # TIFF 尺寸檢查
│   ├── cuda_test.py                 # CUDA 測試
│   └── tiff to png.py               # 格式轉換
│
├── Dockerfile                       # Docker 映像定義
├── docker-compose.yml               # Docker Compose 配置
├── .dockerignore                    # Docker 忽略檔案
├── requirements.txt                 # Python 依賴列表
└── README.md                        # 專案說明文件
```

## Docker 環境說明

### 基礎映像資訊

| 項目 | 版本 |
|------|------|
| 基礎映像 | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` |
| Python | 3.11 |
| PyTorch | 2.11+ (nightly, CUDA 12.8) |
| Java | OpenJDK 17 |
| CUDA | 12.8.1 |
| cuDNN | 已包含 |

### 系統依賴

- **libvips**: 高效影像處理 (libvips42t64)
- **OpenSlide**: WSI 全玻片影像讀取
- **OpenCV**: 電腦視覺處理

### 掛載資料卷

```yaml
volumes:
  - .:/app                    # 專案目錄
  - /path/to/data:/app/data   # 資料目錄
```

## 品質評估指標

### 正規化互相關 (NCC)

- **範圍**: [-1, 1]
- **意義**: 衡量兩影像的線性相關性。值越接近 1，表示對位效果越好。

### 互信息 (MI)

- **範圍**: [0, +∞)
- **意義**: 衡量兩影像間資訊量的共享程度，特別適用於多模態影像。

## 核心技術

### 影像對位技術

- **特徵偵測**: DISK (Deep Image Structure and Keypoint)
- **特徵匹配**: LightGlue (輕量級圖匹配網路)
- **變換模型**: 剛性變換 + 非剛性變換 (B-spline)

### 影像融合技術

- **拉普拉斯金字塔融合**: 多尺度融合技術，保留細節資訊
- **金字塔層級**: 6 層 (可調整)

### HER2 細胞分割

- **僞標籤生成**: LAB 色彩空間 + DAB (HED) 融合
- **模型架構**: UNet++ with DenseNet121 Encoder
- **細胞分割**: Marker-Controlled Watershed
- **配置管理**: 集中式 `config.py` 參數管理

## 授權與引用

本專案使用的主要開源函式庫：

- **valis**: Virtual Alignment of pathoLogy Image Series
- **aicspylibczi**: Allen Institute for Cell Science CZI reader
- **pyvips**: Python binding for libvips image processing library
- **PyTorch**: Deep learning framework
