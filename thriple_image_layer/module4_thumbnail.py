"""Module 4: Full Merged Image Thumbnail"""
from pathlib import Path
import pyvips
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

    # 使用 warp_and_save_slide 直接儲存到暫存檔案，避免大型陣列佔用記憶體
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    # VALIS 會自動將檔名改為 .ome.tiff
    dish_temp_ome = temp_dir / f"dish_warped_lv{level}.ome.tiff"
    her2_temp_ome = temp_dir / f"her2_warped_lv{level}.ome.tiff"

    # 檢查 DISH 暫存檔案是否存在
    if dish_temp_ome.exists():
        print(f"找到現有的 DISH 暫存檔案，跳過重新生成: {dish_temp_ome.name}")
    else:
        print("對齊並儲存 DISH 影像...")
        dish_temp = temp_dir / f"dish_warped_lv{level}.tiff"
        dish_obj.warp_and_save_slide(
            str(dish_temp),
            level=level,
            non_rigid=non_rigid,
            crop=True,
            compression='deflate',
            pyramid=True
        )

    # 檢查 HER2 暫存檔案是否存在
    if her2_temp_ome.exists():
        print(f"找到現有的 HER2 暫存檔案，跳過重新生成: {her2_temp_ome.name}")
    else:
        print("對齊並儲存 HER2 影像...")
        her2_temp = temp_dir / f"her2_warped_lv{level}.tiff"
        her2_obj.warp_and_save_slide(
            str(her2_temp),
            level=level,
            non_rigid=non_rigid,
            crop=True,
            compression='deflate',
            pyramid=True
        )


    # 使用 pyvips 讀取並合併（串流處理，不會一次載入全部記憶體）
    print("合併影像中...")
    dish_vips = pyvips.Image.new_from_file(str(dish_temp_ome), access='sequential')
    her2_vips = pyvips.Image.new_from_file(str(her2_temp_ome), access='sequential')

    merged = (dish_vips.cast('float') + her2_vips.cast('float')) / 2
    merged = merged.cast('uchar')

    # 儲存為 TIFF (使用 BigTIFF 格式支援超過 4GB 的檔案)
    output_path = output_dir / f"Merged_Aligned_lv{level}.tiff"
    print("儲存合併影像...")
    merged.write_to_file(
        str(output_path),
        compression='jpeg',      # 使用 JPEG 壓縮（比 deflate 更小）
        Q=95,                    # JPEG 品質 (1-100)
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
        bigtiff=True             # 啟用 BigTIFF 格式
    )

    # 清理暫存檔案
    print("清理暫存檔案...")
    dish_temp_ome.unlink(missing_ok=True)
    her2_temp_ome.unlink(missing_ok=True)

    print(f"已儲存: {output_path.name}")


if __name__ == "__main__":
    output_dir = Path(r"E:\Class\tsgh\thriple_image_layer\output")
    try:
        generate_thumbnail(output_dir, level=2, non_rigid=True)
    finally:
        try:
            slide_io.kill_jvm()
        except:
            pass
