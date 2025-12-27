# TSGH 病理影像分析系統 (Pathology Image Analysis System) v2.2

本專案為一個自動化的全組織病理影像 (Whole Slide Image, WSI) 分析系統，整合多項先進技術，提供從影像對位到 HER2 染色分析的完整解決方案。

## 主要功能

### 🔬 多染色影像對位
採用 `valis` 函式庫，精準對齊同一組織切片的三種不同染色影像：
- **HE (H&E)** - 蘇木精-伊紅染色
- **Her2** - HER2 免疫組化染色 (作為對位的基準)
- **DISH** - 雙重原位雜交染色

### 🎯 HER2 染色分析
使用色彩解卷積技術，自動分割 HER2 陽性區域：
- 細胞膜 HER2 染色識別
- 染色強度量化分析
- 陽性/陰性區域分割

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
5. **模組 5: HER2 遮罩分析 (HER2 Mask Analysis)** - 使用色彩解卷積進行 HER2 染色區域分割。

## 系統需求

### 硬體需求
- **CPU**: Intel i5 或 AMD Ryzen 5 以上
- **記憶體**: 16GB RAM 以上 (建議 32GB)
- **GPU**: NVIDIA GPU (支援 CUDA 12.8+，可選但強烈建議)
- **儲存空間**: 至少 50GB 可用空間 (用於存放原始影像與輸出結果)

### 軟體需求
- **作業系統**: Ubuntu 24.04 LTS (建議) / Windows 10/11 / macOS
- **Python**: 3.11
- **CUDA Toolkit**: 12.8+ (若使用 GPU 加速)
- **Java**: OpenJDK 17 (用於 scyjava/jgo)
- **Docker**: 24.0+ (可選，用於容器化部署)

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

#### 模組 5: HER2 遮罩分析
```bash
python unet_mask/her2_mask.py
```
- **輸出**: HER2 染色區域分割遮罩

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
├── unet_mask/                       # HER2 遮罩分析模組
│   ├── her2_mask.py                 # HER2 染色區域分割
│   ├── membrane_interior_segmentation.py  # 細胞膜/內部分割
│   ├── overlap_to_dish.py           # DISH 影像疊合
│   └── HER2_Mask_筆記.md            # 技術筆記
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

### HER2 遮罩分析
- **色彩解卷積**: 分離 HER2 棕色染色通道
- **細胞膜分割**: 識別 HER2 陽性細胞膜區域

## 故障排除

### Docker 相關問題

1. **GPU 不可用**
   ```bash
   # 確認 NVIDIA Container Toolkit 已安裝
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
   ```

2. **映像建立失敗**
   - 確認網路連線正常
   - 檢查磁碟空間是否足夠 (需約 20GB)

3. **PyTorch nightly 版本不存在**
   - nightly 版本會定期更新，若特定版本不存在，會自動安裝最新版本

### 一般問題

1. **CUDA 相關錯誤**
   - 確認 CUDA 驅動版本 >= 525
   - 執行 `python scripts/cuda_test.py` 測試

2. **記憶體不足**
   - 減小 `roi_size` 參數
   - 使用更高的 `level` 參數

3. **JVM 相關錯誤**
   - 確認已安裝 Java 17+
   - 檢查 `JAVA_HOME` 環境變數

## 更新日誌

### v2.2.0 (2025-12-27)
- **新增 Docker 支援**: 提供 Dockerfile 和 docker-compose.yml
- **環境升級**: 
  - Ubuntu 24.04 LTS
  - Python 3.11
  - PyTorch 2.11+ (CUDA 12.8)
  - OpenJDK 17
- **新增 HER2 遮罩分析模組**: 使用色彩解卷積進行染色區域分割
- **文件更新**: 更新 README 反映最新專案結構

### v2.1.0 (2025-01-XX)
- **新增模組 5**: 高效切割影像磚功能
- **GPU 加速**: 整合 LightGlue + DISK 特徵匹配
- **融合技術升級**: 採用拉普拉斯金字塔融合技術

### v2.0.0 (2024-10-17)
- **架構重構**: 採用以 `valis` 函式庫為核心的 Python 工作流程
- **模組化設計**: 將流程拆分為獨立模組
- **移除 C++ 依賴**: 簡化安裝與部署流程

## 授權與引用

本專案使用的主要開源函式庫：
- **valis**: Virtual Alignment of pathoLogy Image Series
- **aicspylibczi**: Allen Institute for Cell Science CZI reader
- **pyvips**: Python binding for libvips image processing library
- **PyTorch**: Deep learning framework

## 聯絡資訊

如有問題或建議，請透過 GitHub Issues 回報。
