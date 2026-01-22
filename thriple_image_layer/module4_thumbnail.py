"""Module 4: Full Merged Image Thumbnail

產生對齊疊合縮圖
"""
from pathlib import Path
import pyvips
from valis import registration, slide_io

try:
    from .config import RegistrationConfig, create_default_config, get_slide_key
except ImportError:
    from config import RegistrationConfig, create_default_config, get_slide_key


def laplacian_blend(
    img1: pyvips.Image,
    img2: pyvips.Image,
    levels: int = 5
) -> pyvips.Image:
    """
    多尺度拉普拉斯金字塔融合，保留兩張影像的細節
    
    Args:
        img1: 第一張影像 (DISH)
        img2: 第二張影像 (HER2)
        levels: 金字塔層級數
    
    Returns:
        pyvips.Image: 融合後的影像
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
    config: RegistrationConfig,
) -> None:
    """
    Module 4: 產生對齊疊合縮圖並輸出為 TIFF

    Args:
        config: 配準流程配置
    """
    try:
        slide_io.init_jvm()
    except Exception:
        pass

    output_dir = config.output_dir
    level = config.thumbnail.level
    use_non_rigid = config.thumbnail.use_non_rigid
    laplacian_levels = config.thumbnail.laplacian_levels
    
    print(f"載入變換參數: {config.pickle_path}")
    registrar = registration.load_registrar(str(config.pickle_path))

    ref_slide = registrar.get_ref_slide()
    print(f"使用金字塔 level {level}, 尺寸: {ref_slide.slide_dimensions_wh[level]}")

    # 從配置獲取模態的 slide key
    dish_key = get_slide_key(config.get_modality_by_name("DISH").filename)
    her2_key = get_slide_key(config.get_modality_by_name("HER2").filename)
    
    # 手動對齊兩張影像並合併
    print("生成對齊疊合圖...")
    dish_obj = registrar.slide_dict[dish_key]
    her2_obj = registrar.slide_dict[her2_key]

    # 使用 warp_and_save_slide 直接儲存到暫存檔案，避免大型陣列佔用記憶體
    temp_dir = config.temp_dir
    temp_dir.mkdir(exist_ok=True)

    # VALIS 會自動將檔名改為 .ome.tiff
    dish_temp_ome = temp_dir / f"dish_warped_lv{level}.ome.tiff"
    her2_temp_ome = temp_dir / f"her2_warped_lv{level}.ome.tiff"

    # 檢查 DISH 暫存檔案是否存在
    if dish_temp_ome.exists():
        print(f"找到現有的 DISH 暫存檔案，跳過重新生成: {dish_temp_ome.name}")
    else:
        print(f"對齊並儲存 DISH 影像 (non_rigid={use_non_rigid})...")
        dish_temp = temp_dir / f"dish_warped_lv{level}.tiff"
        dish_obj.warp_and_save_slide(
            str(dish_temp),
            level=level,
            non_rigid=use_non_rigid,
            crop="overlap",
            compression='jpeg',
            Q=100,
            pyramid=True
        )

    # 檢查 HER2 暫存檔案是否存在
    if her2_temp_ome.exists():
        print(f"找到現有的 HER2 暫存檔案，跳過重新生成: {her2_temp_ome.name}")
    else:
        print("對齊並儲存 HER2 影像 (non_rigid=False, 參考影像無需非剛性變換)...")
        her2_temp = temp_dir / f"her2_warped_lv{level}.tiff"
        her2_obj.warp_and_save_slide(
            str(her2_temp),
            level=level,
            non_rigid=False,
            crop="overlap",
            compression='jpeg',
            Q=100,
            pyramid=True,
        )
    
    # 使用 pyvips 讀取並合併（串流處理，不會一次載入全部記憶體）
    print(f"合併影像中 (使用一般 0.5/0.5 融合)...")
    dish_vips = pyvips.Image.new_from_file(str(dish_temp_ome), access='sequential')
    her2_vips = pyvips.Image.new_from_file(str(her2_temp_ome), access='sequential')
    
    # 記錄原始尺寸（用於最終裁剪）
    target_width = dish_vips.width
    target_height = dish_vips.height
    print(f"目標尺寸: {target_width} x {target_height}")
    
    # 一般 0.5 0.5 融合
    print("使用一般 0.5/0.5 融合...")
    merged = (dish_vips * 0.5 + her2_vips * 0.5).cast('uchar')
    
    print(f"融合後尺寸: {merged.width} x {merged.height}")
    
    # 如果融合後尺寸不同，裁剪到目標尺寸
    if merged.width != target_width or merged.height != target_height:
        print(f"裁剪到目標尺寸: {target_width} x {target_height}")
        # 從左上角裁剪（0, 0）
        merged = merged.crop(0, 0, target_width, target_height)
    
    output_path = output_dir / f"Merged_Aligned_lv{level}.tiff"
    print("儲存合併影像...")
    merged.write_to_file(
        str(output_path),
        pyramid=True,
        bigtiff=True,
        compression='jpeg',
        Q=100
    )

    print(f"已儲存: {output_path}")
    print(f"影像尺寸: {merged.width} x {merged.height}, 通道數: {merged.bands}")


if __name__ == "__main__":
    config = create_default_config()
    
    print("=" * 60)
    print("Module 4: Thumbnail Generation")
    print("=" * 60)
    print(f"金字塔層級: {config.thumbnail.level}")
    print(f"使用非剛性變換: {config.thumbnail.use_non_rigid}")
    print()
    
    try:
        pyvips.cache_set_max(1024 * 1024 * 1024)
        generate_thumbnail(config)
    finally:
        try:
            slide_io.kill_jvm()
        except Exception:
            pass
