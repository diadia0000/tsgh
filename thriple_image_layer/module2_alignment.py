"""Module 2: Alignment Pipeline"""
from pathlib import Path
from valis import registration

def align_images(
    czi_dir: Path,
    output_dir: Path,
    scale_factor: int = 32
):
    """
    Module 2: 執行影像對準並儲存變換參數
    
    Args:
        czi_dir: CZI 檔案目錄
        output_dir: 輸出目錄
        scale_factor: 縮放因子
    
    Returns:
        registrar: VALIS 對準器物件
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化 VALIS
    registrar = registration.Valis(
        src_dir=str(czi_dir),
        dst_dir=str(output_dir),
        name="Transform_Params"
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
