"""Module 2: Alignment Pipeline"""
from pathlib import Path
from valis import registration

def align_images(
    czi_dir: Path,
    output_dir: Path,
    reference_img_name: str = "HE_40X.czi"
):
    """
    Module 2: 執行影像對準並儲存變換參數
    
    Args:
        czi_dir: CZI 檔案目錄
        output_dir: 輸出目錄
        reference_img_name: 參考影像檔名 (HE 染色圖)
    
    Returns:
        registrar: VALIS 對準器物件
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化 VALIS (自動選擇對位演算法)
    registrar = registration.Valis(
        src_dir=str(czi_dir),
        dst_dir=str(output_dir),
        name="Transform_Params",
        reference_img_f=reference_img_name,
        align_to_reference=True
    )
    
    # 執行對準
    print("開始執行對準...")
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()
    
    print("對準完成，變換參數已儲存")
    return registrar

if __name__ == "__main__":
    czi_dir = Path(r"E:\Class\tsgh\picture\whole_size\40X")
    output_dir = Path(r"E:\Class\tsgh\thriple_image_layer\output")
    
    registrar = align_images(czi_dir, output_dir)
    print(f"\n對準完成，結果儲存於: {output_dir}")
