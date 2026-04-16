# ============================================================
# Dockerfile for tsgh project
# 基於您的 venv 環境（Python 3.11.14）建立
# ============================================================

# 基礎映像：NVIDIA CUDA 13.0 + cuDNN + Ubuntu 24.04
# 來源：https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

# 環境變數設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=Asia/Taipei

# ============================================================
# 系統依賴安裝
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.11（與您的 venv 環境一致）
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
    # 基本工具
    git \
    wget \
    curl \
    unzip \
    # OpenCV 依賴（Ubuntu 24.04 套件名稱）
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # 影像處理：libvips（Ubuntu 24.04 使用 libvips42t64）
    libvips42t64 \
    libvips-dev \
    # 影像處理：OpenSlide（用於 WSI 全玻片影像）
    libopenslide0 \
    libopenslide-dev \
    # Java 17（用於 scyjava/jgo，與您的系統環境一致）
    openjdk-17-jdk \
    openjdk-17-jdk-headless \
    # 編譯工具（部分 Python 套件需要）
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 設定 Python 3.11 為預設
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# 升級 pip
RUN python -m pip install --upgrade pip setuptools wheel

# ============================================================
# 設定工作目錄
# ============================================================
WORKDIR /app

# ============================================================
# 安裝 Python 套件
# ============================================================

# 複製 requirements.txt
COPY requirements.txt .

# 1. 先安裝 PyTorch nightly（CUDA 12.8）
#    來源：https://pytorch.org/get-started/locally/
#    您的版本：torch==2.11.0.dev20251215+cu128
RUN pip install --no-cache-dir \
    --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# 2. 建立過濾後的 requirements（排除 PyTorch 相關和 nvidia-* 套件）
#    nvidia-* 套件已包含在基礎映像中
RUN grep -v -E "^torch==|^torchvision==|^triton==|^nvidia-|^cuda-" requirements.txt > requirements_filtered.txt

# 3. 安裝其他 Python 套件
RUN pip install --no-cache-dir -r requirements_filtered.txt

# ============================================================
# 複製專案檔案
# ============================================================
COPY . .

# ============================================================
# 環境變數
# ============================================================
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# ============================================================
# 預設命令
# ============================================================
CMD ["/bin/bash"]
