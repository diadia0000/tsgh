"""Module 5: Tile Generator

同時切割 HER2、DISH、Merged 三組對齊的 Tiles
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
import pyvips

from config import RegistrationConfig, create_default_config

def _save_triple_tiles(args: Tuple) -> int:
    """
    同時保存三組對應的 tiles (HER2, DISH, Merged)
    
    Args:
        args: (her2_path, dish_path, merged_path, output_dirs, x, y, w, h, compression)
    
    Returns:
        int: 1 表示成功
    """
    her2_path, dish_path, merged_path, output_dirs, x, y, w, h, compression = args
    
    her2_dir, dish_dir, merged_dir = output_dirs
    
    # 讀取並切割 HER2 tile
    her2_img = pyvips.Image.new_from_file(str(her2_path), access='sequential')
    her2_tile = her2_img.crop(x, y, w, h)
    her2_tile.write_to_file(
        str(her2_dir / f"tile_x{x}_y{y}.tiff"), 
        compression=compression
    )
    
    # 讀取並切割 DISH tile
    dish_img = pyvips.Image.new_from_file(str(dish_path), access='sequential')
    dish_tile = dish_img.crop(x, y, w, h)
    dish_tile.write_to_file(
        str(dish_dir / f"tile_x{x}_y{y}.tiff"), 
        compression=compression
    )
    
    # 讀取並切割 Merged tile
    merged_img = pyvips.Image.new_from_file(str(merged_path), access='sequential')
    merged_tile = merged_img.crop(x, y, w, h)
    merged_tile.write_to_file(
        str(merged_dir / f"tile_x{x}_y{y}.tiff"), 
        compression=compression
    )
    
    return 1


def generate_triple_tiles(
    config: RegistrationConfig,
    level: int = 1,
) -> None:
    """
    同時切割三組對齊的 TIFF 影像 (HER2, DISH, Merged)
    
    Args:
        config: 配準流程配置
        level: 使用的金字塔層級
    """
    output_dir = config.output_dir
    temp_dir = config.temp_dir
    tile_config = config.tile
    
    # 設定路徑
    her2_tiff = temp_dir / f"her2_warped_lv{level}.ome.tiff"
    dish_tiff = temp_dir / f"dish_warped_lv{level}.ome.tiff"
    merged_tiff = output_dir / f"Merged_Aligned_lv{level}.tiff"
    
    tile_output_dir = output_dir / f"tiles_lv{level}-{tile_config.tile_width}"
    
    tile_width = tile_config.tile_width
    tile_height = tile_config.tile_height
    workers = tile_config.workers
    compression = tile_config.compression
    
    # 創建輸出目錄
    her2_dir = tile_output_dir / "her2"
    dish_dir = tile_output_dir / "dish"
    merged_dir = tile_output_dir / "merged"
    
    her2_dir.mkdir(parents=True, exist_ok=True)
    dish_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    output_dirs = (her2_dir, dish_dir, merged_dir)
    
    # 檢查檔案是否存在
    for name, path in [("HER2", her2_tiff), ("DISH", dish_tiff), ("Merged", merged_tiff)]:
        if not path.exists():
            raise FileNotFoundError(f"{name} 檔案不存在: {path}")
    
    # 讀取影像尺寸（使用 Merged 作為參考）
    img = pyvips.Image.new_from_file(str(merged_tiff), access='sequential')
    width, height = img.width, img.height
    
    print("=" * 60)
    print(f"影像尺寸: {width} x {height}")
    print(f"Tile 尺寸: {tile_width} x {tile_height}")
    print(f"壓縮格式: {compression}")
    print(f"使用 {workers} 個執行緒")
    print("=" * 60)
    
    # 驗證三張圖尺寸相同
    her2_img = pyvips.Image.new_from_file(str(her2_tiff), access='sequential')
    dish_img = pyvips.Image.new_from_file(str(dish_tiff), access='sequential')
    
    print(f"\n尺寸驗證:")
    print(f"  Merged: {width} x {height}")
    print(f"  HER2:   {her2_img.width} x {her2_img.height}")
    print(f"  DISH:   {dish_img.width} x {dish_img.height}")
    
    if (her2_img.width != width or her2_img.height != height or
        dish_img.width != width or dish_img.height != height):
        print("\n❌ 錯誤：三張影像尺寸不同！")
        if her2_img.width != width or her2_img.height != height:
            print(f"  HER2 不匹配：{her2_img.width} x {her2_img.height} != {width} x {height}")
        if dish_img.width != width or dish_img.height != height:
            print(f"  DISH 不匹配：{dish_img.width} x {dish_img.height} != {width} x {height}")
        raise ValueError("三張影像尺寸不同！請使用 check_tiff_size.py 檢查檔案。")
    
    print("✓ 驗證通過：三張圖像尺寸相同\n")
    
    # 生成切割任務
    tasks = []
    for y in range(0, height, tile_height):
        for x in range(0, width, tile_width):
            w = min(tile_width, width - x)
            h = min(tile_height, height - y)
            tasks.append((
                her2_tiff, dish_tiff, merged_tiff, 
                output_dirs, x, y, w, h, compression
            ))
    
    print(f"預計生成 {len(tasks)} 組 tiles（每組包含 HER2, DISH, Merged）\n")
    
    # 多執行緒處理
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tile_count = 0
        for _ in executor.map(_save_triple_tiles, tasks):
            tile_count += 1
            if tile_count % 500 == 0:
                print(f"進度: {tile_count}/{len(tasks)} ({tile_count/len(tasks)*100:.1f}%)")
    
    print()
    print("=" * 60)
    print(f"完成！共生成 {tile_count} 組 tiles")
    print(f"   - HER2 tiles: {her2_dir}")
    print(f"   - DISH tiles: {dish_dir}")
    print(f"   - Merged tiles: {merged_dir}")
    print("=" * 60)


if __name__ == "__main__":
    config = create_default_config()
    
    print("=" * 60)
    print("Module 5: Tile Generator")
    print("=" * 60)
    print(f"Tile 尺寸: {config.tile.tile_width} x {config.tile.tile_height}")
    print(f"執行緒數: {config.tile.workers}")
    print()
    
    # 使用 level 1 切割 tiles
    generate_triple_tiles(config, level=0)
