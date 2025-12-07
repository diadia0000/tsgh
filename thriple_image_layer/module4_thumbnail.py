"""Module 4: Full Merged Image Thumbnail"""
from pathlib import Path
import pyvips
from valis import registration, slide_io


def laplacian_blend(img1: pyvips.Image, img2: pyvips.Image, levels: int = 5) -> pyvips.Image:
    """
    多尺度拉普拉斯金字塔融合，保留兩張影像的細節
    
    Args:
        img1: 第一張影像 (DISH)
        img2: 第二張影像 (HER2)
        levels: 金字塔層級數
    
    Returns:
        融合後的影像
    """
    # 建立高斯金字塔
    gauss1, gauss2 = [img1], [img2]
    for _ in range(levels):
        gauss1.append(gauss1[-1].shrink(2, 2))
        gauss2.append(gauss2[-1].shrink(2, 2))
    
    # 從最粗層級開始融合
    result = gauss1[-1] * 0.5 + gauss2[-1] * 0.5
    
    # 逐層重建並融合拉普拉斯細節
    for i in range(levels - 1, -1, -1):
        result = result.resize(2, kernel='cubic')
        # 計算拉普拉斯層（細節）
        lap1 = gauss1[i] - gauss1[i].shrink(2, 2).resize(2, kernel='cubic')
        lap2 = gauss2[i] - gauss2[i].shrink(2, 2).resize(2, kernel='cubic')
        # 融合細節並加回
        result = result + (lap1 * 0.5 + lap2 * 0.5)
    
    return result.cast('uchar')

def generate_thumbnail(
    output_dir: Path,
    level: int = 4,
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
            non_rigid=True,
            crop="overlap",
            compression='lzw'
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
            non_rigid=False,
            crop="overlap",
            compression='lzw'
        )
    # 使用 pyvips 讀取並合併（串流處理，不會一次載入全部記憶體）
    print("合併影像中 (多尺度拉普拉斯金字塔融合)...")
    dish_vips = pyvips.Image.new_from_file(str(dish_temp_ome), access='sequential')
    her2_vips = pyvips.Image.new_from_file(str(her2_temp_ome), access='sequential')
    
    # 使用拉普拉斯金字塔融合保留細節
    merged = laplacian_blend(dish_vips, her2_vips, levels=6)
    output_path = f"G:\output\Merged_Aligned_lv{level}.tiff"
    print("儲存合併影像...")
    merged.write_to_file(
        str(output_path),
        pyramid=True,
        bigtiff=True,
        compression='lzw'
    )

    print(f"已儲存: {output_path}")
    print(f"影像尺寸: {merged.width} x {merged.height}, 通道數: {merged.bands}")


if __name__ == "__main__":
    output_dir = Path(r"H:\tsgh\thriple_image_layer\output")
    try:
        pyvips.cache_set_max(0)
        generate_thumbnail(output_dir, level=0)
    finally:
        try:
            slide_io.kill_jvm()
        except:
            pass
