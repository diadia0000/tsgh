"""
設定檔：UNet 模型訓練的各項參數設定
"""
from pathlib import Path
import torch

# ================= 路徑設定 =================
BASE_DIR = Path(__file__).parent                          # 專案根目錄
REP_DIR = BASE_DIR / 'process/train-512-lv1'             # 代表性評估圖像目錄
FULL_DATA_DIR = BASE_DIR / 'process/tile/level-train-batch4'  # 完整訓練資料目錄
MASK_CACHE_DIR = BASE_DIR / 'process/output'             # Mask 快取目錄
MODEL_DIR = BASE_DIR / 'process/models'                  # 模型儲存目錄

# ================= 訓練超參數 =================
CROP_SIZE = 512           # 裁切尺寸（輸入圖像大小）
BATCH_SIZE = 32           # 批次大小
NUM_EPOCHS = 30           # 訓練輪數
LEARNING_RATE = 5e-4      # 學習率
WEIGHT_DECAY = 1e-4       # 權重衰減（L2 正則化）
TRAIN_VAL_SPLIT = 0.9     # 訓練/驗證集比例（90% 訓練，10% 驗證）
RANDOM_SEED = 42          # 隨機種子（確保可重現性）

# ================= 裝置設定 =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 圖像正規化 =================
# 使用 ImageNet 預訓練模型的標準化參數
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ================= 模型設定 =================
ENCODER_NAME = 'efficientnet-b4'  # 編碼器架構
ENCODER_WEIGHTS = 'imagenet'       # 預訓練權重
IN_CHANNELS = 3                    # 輸入通道數（RGB）
NUM_CLASSES = 3                    # 輸出類別數（0=背景, 1=細胞內, 2=細胞膜）

# ================= 建立目錄 =================
def ensure_dirs():
    """建立必要的目錄（如果不存在）"""
    MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MASK_CACHE_DIR / 'rep_eval').mkdir(parents=True, exist_ok=True)
