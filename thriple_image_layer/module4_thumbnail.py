"""Module 4: Full Merged Image Thumbnail"""
from pathlib import Path
import numpy as np
from PIL import Image
from valis import registration, slide_io

def generate_thumbnail(
    output_dir: Path,
    level: int = 3
) -> None:
    """
    Module 4: 產生全局對齊縮圖

    Args:
        output_dir: 輸出目錄
        level: 金字塔層級 (0=最高解析度)
    """
    # 初始化 JVM
    try:
        slide_io.init_jvm()
    except:
        pass
    
    # 載入變換參數
    pickle_path = output_dir / "Transform_Params" / "data" / "Transform_Params_registrar.pickle"
    registrar = registration.load_registrar(str(pickle_path))

    ref_slide = registrar.get_ref_slide()
    print(f"使用金字塔 level {level}, 尺寸: {ref_slide.slide_dimensions_wh[level]}")

    # 產生三張對齊後縮圖
    print("產生對齊後縮圖...")
    slide_names = sorted(registrar.slide_dict.keys())
    thumbs = {}

    for name in slide_names:
        slide_obj = registrar.slide_dict[name]
        print(f"  處理 {name}...")
        thumb = slide_obj.warp_slide(
            level=level,
            non_rigid=True,
            crop=True
        )
        # 轉換為 numpy
        if hasattr(thumb, 'height'):
            from valis import warp_tools
            thumb = warp_tools.vips2numpy(thumb)
        thumbs[name] = thumb

    dish_thumb = thumbs['DISH_40X_2']
    her2_thumb = thumbs['HER2_40X']
    print(f"縮圖尺寸: {dish_thumb.shape}")

    # 合併為 RGB (RGB=Her2, RGB=DISH)
    merged = (her2_thumb.astype(np.float32) + dish_thumb.astype(np.float32)) / 2
    merged = np.clip(merged, 0, 255).astype(np.uint8)

    # 儲存
    merged_img = Image.fromarray(merged)
    merged_img.save(output_dir / "Merged_DISH_Her2.png")
    print(f"已儲存: Merged_DISH_Her2.png (Her2 + DISH 平均疊合)")
    
    # 也儲存單獨的縮圖
    Image.fromarray(dish_thumb.astype(np.uint8)).save(output_dir / "DISH_thumbnail.png")
    Image.fromarray(her2_thumb.astype(np.uint8)).save(output_dir / "Her2_thumbnail.png")
    print("已儲存單獨縮圖")
    

if __name__ == "__main__":
    output_dir = Path(r"E:\Class\tsgh\thriple_image_layer\output")
    try:
        generate_thumbnail(output_dir, level=4)
    finally:
        # 清理 JVM
        try:
            slide_io.kill_jvm()
        except:
            pass
