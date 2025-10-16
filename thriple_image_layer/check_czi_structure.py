"""檢查 CZI 檔案結構"""
from pathlib import Path
from aicspylibczi import CziFile

def check_czi_structure(czi_path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"檔案: {czi_path.name}")
    print(f"{'='*60}")
    
    czi = CziFile(czi_path)
    print(f"維度: {czi.dims}")
    print(f"大小: {czi.size}")
    
    dims_shape = czi.get_dims_shape()
    print(f"\n維度形狀資訊:")
    for i, dim_info in enumerate(dims_shape):
        print(f"  Scene {i}: {dim_info}")
    
    print(f"\n是否為 Mosaic: {czi.is_mosaic()}")
    
    try:
        img, dims = czi.read_image()
        print(f"\n影像形狀: {img.shape}")
        print(f"影像維度: {dims}")
        print(f"資料類型: {img.dtype}")
        print(f"值範圍: [{img.min()}, {img.max()}]")
    except Exception as e:
        print(f"\n讀取影像時發生錯誤: {e}")

if __name__ == "__main__":
    czi_dir = Path(r"E:\Class\tsgh\picture\whole_size\40X")
    
    for czi_file in ["DISH_40X_2.czi", "HE_40X.czi", "HER2_40X.czi"]:
        czi_path = czi_dir / czi_file
        if czi_path.exists():
            check_czi_structure(czi_path)
