"""Module 2: Alignment Pipeline"""
from pathlib import Path
import pickle
import os

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
        print("using cpu")
    else:
        print(f"使用裝置：{device}")
    
    # 使用 BRISK 特徵檢測器 + 傳統 Matcher
    # BRISK 是穩定的傳統特徵檢測器，不會造成影像扭曲
    fd = feature_detectors.BriskFD()
    
    matcher = feature_matcher.Matcher(
        feature_detector=fd,
        match_filter_method='GMS',      # Grid-based Motion Statistics 過濾
        gms_threshold=25,               # 較高閾值確保匹配品質
        ransac_thresh=5
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

    # 明確地將 registrar 物件保存為 pickle 檔案
    pickle_dir = output_dir / "Transform_Params" / "data"
    pickle_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = pickle_dir / "Transform_Params_registrar.pickle"

    # 根據 AI 檔案生成規則：如果檔案已存在則先刪除
    if os.path.exists(pickle_path):
        print(f"正在刪除舊的 pickle 檔案: {pickle_path}")
        os.remove(pickle_path)

    print(f"正在保存 registrar 物件到: {pickle_path}")
    with open(pickle_path, 'wb') as f:
        pickle.dump(registrar, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("✓ Registrar 物件已成功保存為 pickle 檔案")

    return registrar

if __name__ == "__main__":
    czi_dir = Path(r"E:\Class\tsgh\picture\whole_size\40X")
    output_dir = Path(r"E:\Class\tsgh\thriple_image_layer\output")
    
    registrar = align_images(czi_dir, output_dir)
    print(f"\n對準完成，結果儲存於: {output_dir}")
