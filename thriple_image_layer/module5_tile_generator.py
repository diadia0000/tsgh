"""Module 5: Tile Generator

同時切割 HER2、DISH、Merged 三組對齊的 Tiles
使用 pyvips 讀取大型金字塔 TIFF 影像
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
import pyvips

from config import RegistrationConfig, create_default_config


def _crop_tile(image: pyvips.Image, x: int, y: int, tile_w: int, tile_h: int) -> pyvips.Image:
    w = min(tile_w, image.width - x)
    h = min(tile_h, image.height - y)
    tile = image.crop(x, y, w, h)
    if w < tile_w or h < tile_h:
        tile = tile.gravity("north-west", tile_w, tile_h, extend="white")
    return tile


def _save_triple_tiles(args: Tuple) -> int:
    her2_path, dish_path, merged_path, output_dirs, x, y, tile_w, tile_h, compression = args
    her2_dir, dish_dir, merged_dir = output_dirs

    for slide_path, out_dir in (
        (her2_path, her2_dir),
        (dish_path, dish_dir),
        (merged_path, merged_dir),
    ):
        img = pyvips.Image.new_from_file(str(slide_path), access="random")
        _crop_tile(img, x, y, tile_w, tile_h).tiffsave(
            str(out_dir / f"tile_x{x}_y{y}.tiff"),
            compression=compression,
            rgbjpeg=True,
        )
    return 1


def generate_triple_tiles(config: RegistrationConfig, level: int = 0) -> None:
    """同時切割三組對齊的 TIFF 影像 (HER2, DISH, Merged)"""
    output_dir = config.output_dir
    temp_dir = config.temp_dir
    tile_config = config.tile

    her2_tiff = temp_dir / f"her2_warped_lv{level}.ome.tiff"
    dish_tiff = temp_dir / f"dish_warped_lv{level}.ome.tiff"
    merged_tiff = output_dir / f"Merged_Aligned_lv{level}.tiff"
    tile_output_dir = output_dir / f"tiles_lv{level}-{tile_config.tile_width}"

    tile_w = tile_config.tile_width
    tile_h = tile_config.tile_height
    workers = tile_config.workers
    compression = tile_config.compression

    her2_dir = tile_output_dir / "her2"
    dish_dir = tile_output_dir / "dish"
    merged_dir = tile_output_dir / "merged"
    for d in (her2_dir, dish_dir, merged_dir):
        d.mkdir(parents=True, exist_ok=True)
    output_dirs = (her2_dir, dish_dir, merged_dir)

    for name, path in [("HER2", her2_tiff), ("DISH", dish_tiff), ("Merged", merged_tiff)]:
        if not path.exists():
            raise FileNotFoundError(f"{name} 檔案不存在: {path}")

    merged_ref = pyvips.Image.new_from_file(str(merged_tiff), access="random")
    her2_ref = pyvips.Image.new_from_file(str(her2_tiff), access="random")
    dish_ref = pyvips.Image.new_from_file(str(dish_tiff), access="random")
    width, height = merged_ref.width, merged_ref.height

    print("=" * 60)
    print(f"影像尺寸: {width} x {height}")
    print(f"Tile 尺寸: {tile_w} x {tile_h}")
    print(f"壓縮格式: {compression}")
    print(f"使用 {workers} 個執行緒")
    print("=" * 60)

    print(f"\n尺寸驗證:")
    print(f"  Merged: {width} x {height}")
    print(f"  HER2:   {her2_ref.width} x {her2_ref.height}")
    print(f"  DISH:   {dish_ref.width} x {dish_ref.height}")

    if (her2_ref.width != width or her2_ref.height != height or
            dish_ref.width != width or dish_ref.height != height):
        if her2_ref.width != width or her2_ref.height != height:
            print(f"  HER2 不匹配：{her2_ref.width} x {her2_ref.height} != {width} x {height}")
        if dish_ref.width != width or dish_ref.height != height:
            print(f"  DISH 不匹配：{dish_ref.width} x {dish_ref.height} != {width} x {height}")
        raise ValueError("三張影像尺寸不同！請使用 check_tiff_size.py 檢查檔案。")

    print("✓ 驗證通過：三張圖像尺寸相同\n")
    del merged_ref, her2_ref, dish_ref

    tasks = [
        (her2_tiff, dish_tiff, merged_tiff, output_dirs, x, y, tile_w, tile_h, compression)
        for y in range(0, height, tile_h)
        for x in range(0, width, tile_w)
    ]
    print(f"預計生成 {len(tasks)} 組 tiles（每組包含 HER2, DISH, Merged）\n")

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
    print("Module 5: Tile Generator (pyvips)")
    print("=" * 60)
    print(f"Tile 尺寸: {config.tile.tile_width} x {config.tile.tile_height}")
    print(f"執行緒數: {config.tile.workers}")
    print()
    generate_triple_tiles(config, level=0)
