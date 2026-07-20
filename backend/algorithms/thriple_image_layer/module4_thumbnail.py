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
        raise RuntimeError("無法啟動 Java 虛擬機，請確認已安裝 Java 並設定環境變數。")

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

    # 使用 warp_slide + tiffsave(subifd=False) 直接儲存到暫存檔案，避免大型陣列佔用記憶體。
    # 不走 warp_and_save_slide：它內部的 valis.slide_io.save_ome_tiff 把 subifd 綁死等於
    # pyramid（見 <venv>/valis/slide_io.py:3685），OpenSlide 讀不到金字塔，OSD 開圖會把後端讀爆
    # （docs/UI/09-viewer-tiff-subifd.md 的事故）。這兩份暫存檔只會被下面的 pyvips 直讀，不會再
    # 餵回 VALIS，所以不需要 OME-TIFF/subifd，直接存成 OpenSlide 看得懂的一般金字塔 TIFF。
    temp_dir = config.temp_dir
    temp_dir.mkdir(exist_ok=True)

    dish_temp = temp_dir / f"dish_warped_lv{level}.tiff"
    her2_temp = temp_dir / f"her2_warped_lv{level}.tiff"

    # 每次都從最新的對齊結果重新 warp，不復用舊暫存檔：復用只以 level 為 key，
    # 無法辨別 registrar 是否已重跑，會把上一次對齊的 warped 影像混進新縮圖。
    # tiffsave 會覆寫既有檔案，這裡先刪除以防殘留半寫入的檔案。
    for temp_file in (dish_temp, her2_temp):
        temp_file.unlink(missing_ok=True)

    print(f"對齊並儲存 DISH 影像 (non_rigid={use_non_rigid})...")
    dish_warped = dish_obj.warp_slide(level=level, non_rigid=True, crop="overlap")
    dish_warped.tiffsave(
        str(dish_temp),
        compression='jpeg',
        Q=95,
        rgbjpeg=True,
        tile=True,
        tile_width=1024,
        tile_height=1024,
        pyramid=True,
        subifd=False,
        bigtiff=True,
    )

    print("對齊並儲存 HER2 影像 (non_rigid=False)...")
    her2_warped = her2_obj.warp_slide(level=level, non_rigid=False, crop="overlap")
    her2_warped.tiffsave(
        str(her2_temp),
        compression='jpeg',
        Q=95,
        rgbjpeg=True,
        tile=True,
        tile_width=1024,
        tile_height=1024,
        pyramid=True,
        subifd=False,
        bigtiff=True,
    )

    # 使用 pyvips 讀取並合併（串流處理，不會一次載入全部記憶體）
    print(f"合併影像中 (使用一般 0.5/0.5 融合)...")
    dish_vips = pyvips.Image.new_from_file(str(dish_temp), access='random')
    her2_vips = pyvips.Image.new_from_file(str(her2_temp), access='random')
    
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
        Q=95,
        compression='jpeg',
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
