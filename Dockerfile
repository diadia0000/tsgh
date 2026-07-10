# syntax=docker/dockerfile:1

# ============================================================
# Dockerfile for tsgh — WSI / medical image pipeline
# 以 uv (專案原生工具) + uv.lock 重現已驗證的 .venv 環境：
#   Python 3.11 · torch 2.11.0+cu130 · torchvision 0.26.0 · CUDA 13.0
# ============================================================

# 基礎映像：NVIDIA CUDA 13.0 + cuDNN (devel) / Ubuntu 24.04
# 對應 requirements/uv.lock 內的 cu130 版 PyTorch。
# 註：torch wheel 已自帶 CUDA runtime，此 base 主要提供 nvidia-container
#     runtime hook 與系統 CUDA；若要縮小體積可改用 -runtime 版。
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

# uv：在 build cache 掛載下用 copy 避免 hardlink 警告，並預先編譯 bytecode
# 將 venv 放在 /app 之外，避免 docker-compose 的 .:/app 掛載把它蓋掉
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Taipei \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# ============================================================
# 系統依賴（影像處理 / WSI / bioformats 的 Java）
# 這些是 pip wheel 在執行期動態載入的原生函式庫。
# ============================================================
# OpenCV / OpenGL runtime（Ubuntu 24.04：libgl1-mesa-glx 已移除）
# libvips（Ubuntu 24.04 為 t64 ABI）— pyvips 執行期需要
# OpenSlide（openslide-bin 已內含 binary，這裡保留系統版本較保險）
# Java 17：scyjava / bioformats（valis-wsi、wsireg）
# 少數套件可能需由 sdist 編譯
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git curl \
      libgl1 libglx-mesa0 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
      libvips42t64 libvips-dev \
      libopenslide0 \
      openjdk-17-jre-headless \
      build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# ============================================================
# uv（官方靜態 binary，版本對齊專案的 uv 0.11.6）
# ============================================================
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /usr/local/bin/

WORKDIR /app

# ============================================================
# 安裝 Python 依賴
# 先只複製 lock 相關檔案，讓依賴層可被 cache（原始碼變動不會使其失效）。
# uv 依 requires-python (>=3.11,<3.12) 自動下載並管理 Python 3.11。
# --no-install-project：本專案以指令稿路徑執行，不需安裝為套件。
# ============================================================
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 讓 venv 內的可執行檔（python 等）進入 PATH
ENV PATH=/opt/venv/bin:${PATH} \
    VIRTUAL_ENV=/opt/venv

# ============================================================
# 複製專案原始碼
# ============================================================
COPY . .

# ============================================================
# CUDA 環境變數
# ============================================================
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# ============================================================
# 預設命令
# ============================================================
CMD ["/bin/bash"]
