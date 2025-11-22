import torch


def check_cuda_status():
    """
    檢查 CUDA 的安裝和狀態，並顯示 GPU 資訊。
    """
    print("--- 檢查 CUDA 狀態 ---")

    # 1. 檢查 PyTorch 是否支援 CUDA
    is_available = torch.cuda.is_available()
    print(f"CUDA 可用嗎 (torch.cuda.is_available())? {is_available}")

    if is_available:
        # 2. 獲取 GPU 數量
        gpu_count = torch.cuda.device_count()
        print(f"偵測到的 GPU 數量: {gpu_count}")

        # 3. 遍歷並顯示每個 GPU 的資訊
        for i in range(gpu_count):
            print(f"\n--- GPU {i} 資訊 ---")

            # GPU 名稱
            gpu_name = torch.cuda.get_device_name(i)
            print(f"名稱: {gpu_name}")

            # 總記憶體
            total_memory_bytes = torch.cuda.get_device_properties(i).total_memory
            # 轉換為 GB
            total_memory_gb = total_memory_bytes / (1024 ** 3)
            print(f"總記憶體: {total_memory_gb:.2f} GB")

            # 計算能力 (Compute Capability)
            capability = torch.cuda.get_device_capability(i)
            print(f"Compute Capability: {capability[0]}.{capability[1]}")

            # 簡單的 GPU 運算測試：將兩個張量相加並放在 GPU 上
            try:
                device = torch.device(f"cuda:{i}")
                a = torch.randn(5, 5).to(device)
                b = torch.randn(5, 5).to(device)
                c = a + b
                print(f"\n✅ 成功在 GPU {i} 上執行張量相加運算。")
                # print("結果張量 (部分):", c[:2, :2])
            except Exception as e:
                print(f"\n❌ 在 GPU {i} 上執行運算失敗。錯誤: {e}")

    else:
        print("\n❌ 失敗：沒有偵測到 CUDA 或 GPU。")
        print("請檢查您的 NVIDIA 驅動程式和 CUDA Toolkit 是否正確安裝。")


if __name__ == "__main__":
    # 執行檢查函數
    check_cuda_status()