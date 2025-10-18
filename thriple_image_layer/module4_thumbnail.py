"""Module 4: Full Merged Image Thumbnail"""
from pathlib import Path
import numpy as np
from PIL import Image
from valis import registration, slide_io

def generate_thumbnail(
    output_dir: Path,
    level: int = 4,
    non_rigid: bool = True,
) -> None:
    """
    Module 4: 產生對齊疊合縮圖並輸出為 TIFF

    Args:
        output_dir: 輸出目錄
        level: 金字塔層級 (0=最高解析度，數字越大解析度越低)
        non_rigid: 是否使用非剛性變換
    """
    try:
        slide_io.init_jvm()
    except:
        pass
    
    pickle_path = output_dir / "Transform_Params" / "data" / "Transform_Params_registrar.pickle"
    registrar = registration.load_registrar(str(pickle_path))

    ref_slide = registrar.get_ref_slide()
    print(f"使用金字塔 level {level}, 尺寸: {ref_slide.slide_dimensions_wh[level]}")
    print(f"非剛性變換: {'啟用' if non_rigid else '停用'}")

    # 手動對齊兩張影像並合併
    print("生成對齊疊合圖...")
    dish_obj = registrar.slide_dict['DISH_40X_2']
    her2_obj = registrar.slide_dict['HER2_40X']
    
    dish_thumb = dish_obj.warp_slide(level=level, non_rigid=non_rigid, crop=True)
    if hasattr(dish_thumb, 'height'):
        from valis import warp_tools
        dish_thumb = warp_tools.vips2numpy(dish_thumb)
    
    her2_thumb = her2_obj.warp_slide(level=level, non_rigid=non_rigid, crop=True)
    if hasattr(her2_thumb, 'height'):
        from valis import warp_tools
        her2_thumb = warp_tools.vips2numpy(her2_thumb)
    
    merged = (her2_thumb.astype(np.float32) + dish_thumb.astype(np.float32)) / 2
    merged = np.clip(merged, 0, 255).astype(np.uint8)
    
    output_path = output_dir / f"Merged_Aligned_lv{level}.tiff"
    Image.fromarray(merged).save(str(output_path), compression="tiff_deflate")
    
    print(f"已儲存: {output_path.name}")
    

if __name__ == "__main__":
    output_dir = Path(r"E:\Class\tsgh\thriple_image_layer\output")
    try:
        generate_thumbnail(output_dir, level=3, non_rigid=True)
    finally:
        try:
            slide_io.kill_jvm()
        except:
            pass
