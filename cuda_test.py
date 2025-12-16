import torch
import time
import sys

def print_header(msg):
    print(f"\n{'='*10} {msg} {'='*10}")

def check_gpu():
    print_header("系統環境檢查")
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 檢查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("\n❌ 錯誤: PyTorch 偵測不到 GPU！")
        print("請檢查是否安裝了正確版本的 PyTorch (Nightly/Preview for RTX 5090)")
        return

    # 獲取 GPU 資訊
    device_id = 0
    gpu_name = torch.cuda.get_device_name(device_id)
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()
    
    # 獲取 VRAM 資訊
    total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9
    allocated_memory = torch.cuda.memory_allocated(device_id) / 1e9
    reserved_memory = torch.cuda.memory_reserved(device_id) / 1e9

    # 獲取算力 (Compute Capability)
    capability = torch.cuda.get_device_capability(device_id)

    print(f"✅ 偵測到 GPU: {gpu_name}")
    print(f"CUDA 版本 (PyTorch): {cuda_version}")
    print(f"cuDNN 版本: {cudnn_version}")
    print(f"算力架構 (Compute Capability): {capability[0]}.{capability[1]}")
    print(f"總 VRAM: {total_memory:.2f} GB")
    print("-" * 30)
check_gpu()