"""Module 2: Alignment Pipeline"""
from pathlib import Path

import torch
from valis import registration, feature_detectors, feature_matcher


def align_images(
    czi_dir: Path,
    output_dir: Path,
    reference_img_name: str = "HER2_40X.czi"
):
    """
    Module 2: 執行影像對準並儲存變換參數
    
    Args:
        czi_dir: CZI 檔案目錄
        output_dir: 輸出目錄
        reference_img_name: 參考影像檔名 (HER2 染色圖)
    
    Returns:
        registrar: VALIS 對準器物件
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("警告：找不到 CUDA 裝置，將繼續使用 CPU。")
    else:
        print(f"使用裝置：{device}")
    # 2. 建立一個特徵「偵測器」
    # LightGlue 推薦搭配 DISK 或 SuperPoint。valis 預設使用 DISK。
    # 您也可以在這裡指定 device，因為偵測器也需要在 GPU 上運行
    fd = feature_detectors.DiskFD(num_features=2048, device=device)

    # 3. 建立「匹配器」(LightGlueMatcher)
    #    在這裡傳入您想要的 device，以及剛剛建立的偵測器
    #    valis 的 LightGlueMatcher 會把 'device' 參數傳給底層的 kornia 模型
    matcher = feature_matcher.LightGlueMatcher(
        fd,  # 告訴 LightGlue 它要搭配哪個偵測器
        device=device  # <--- 這才是指定 GPU 的關鍵參數！
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # 初始化 VALIS (自動選擇對位演算法)
    registrar = registration.Valis(
        src_dir=str(czi_dir),
        dst_dir=str(output_dir),
        name="Transform_Params",
        reference_img_f=reference_img_name,
        align_to_reference=True,
        matcher=matcher
    )
    
    # 執行對準
    print("開始執行對準...")
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()
    
    print("對準完成，變換參數已儲存")
    return registrar

if __name__ == "__main__":
    czi_dir = Path(r"H:\tsgh\picture\whole_size\40X")
    output_dir = Path(r"H:\tsgh\thriple_image_layer\output")
    
    registrar = align_images(czi_dir, output_dir)
    print(f"\n對準完成，結果儲存於: {output_dir}")
