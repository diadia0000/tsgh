"""Module 4: Full Merged Image Thumbnail"""
from pathlib import Path
import pyvips
from valis import registration, slide_io

def generate_thumbnail(
    output_dir: Path,
    level: int = 4,
    non_rigid: bool = True,
    blend_mode: str = 'max',
) -> None:
    """
    Module 4: 產生對齊疊合縮圖並輸出為 TIFF

    Args:
        output_dir: 輸出目錄
        level: 金字塔層級 (0=最高解析度，數字越大解析度越低)
        non_rigid: 是否使用非剛性變換
        blend_mode: 混合模式 ('max', 'color', 'screen', 'average')
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

    # 根據層級選擇插值方法: level>=3 用 bilinear (快速), level<3 用 lanczos (高品質)
    interp = "bilinear" if level >= 3 else "lanczos"
    print(f"插值方法: {interp}")

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
            interp_method=interp,
            tile_wh=4096
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
            interp_method=interp
        )
    # 使用 pyvips 讀取並合併（串流處理，不會一次載入全部記憶體）
    print(f"合併影像中 (模式: {blend_mode})...")
    dish_vips = pyvips.Image.new_from_file(str(dish_temp_ome), access='sequential')
    her2_vips = pyvips.Image.new_from_file(str(her2_temp_ome), access='sequential')
    
    # 根據混合模式選擇合併方式
    if blend_mode == 'max':
        # 最大值投影 - 保留最高對比度
        merged = dish_vips.max(her2_vips)
    elif blend_mode == 'color':
        # 色彩通道疊合 - DISH(綠) + HER2(紅)
        zero_channel = pyvips.Image.black(dish_vips.width, dish_vips.height)
        merged = her2_vips.bandjoin([dish_vips, zero_channel])
    elif blend_mode == 'screen':
        # Screen 混合 - 保留亮部細節
        dish_norm = dish_vips / 255.0
        her2_norm = her2_vips / 255.0
        merged = (1 - (1 - dish_norm) * (1 - her2_norm)) * 255
    else:  # 'average'
        # 加權平均 (原始方法)
        merged = (dish_vips * 0.5 + her2_vips * 0.5)
    output_path = output_dir / f"Merged_Aligned_lv{level}.tiff"
    merged = merged.cast('uchar')
    print("儲存合併影像...")
    merged.write_to_file(
        str(output_path),
        compression='defaults',      # 使用 JPEG 壓縮（比 deflate 更小）
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
        # 推薦使用 'max' 或 'color' 模式以獲得更好的視覺效果
        generate_thumbnail(output_dir, level=2, non_rigid=True, blend_mode='max')
    finally:
        try:
            slide_io.kill_jvm()
        except:
            pass
