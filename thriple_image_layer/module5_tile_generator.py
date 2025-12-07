"""Module 5: Tile Generator - 高效切割對齊影像"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pyvips


def _save_tile(args):
    """單一 tile 處理函式"""
    input_tiff, output_dir, x, y, w, h = args
    img = pyvips.Image.openslideload(str(input_tiff), level=0)
    tile = img.crop(x, y, w, h)
    tile.write_to_file(str(output_dir / f"tile_x{x}_y{y}.tiff"), compression='deflate')
    return 1


def generate_tiles(
    input_tiff: Path,
    output_dir: Path,
    tile_width: int = 2056,
    tile_height: int = 2464,
    workers: int = 4,
) -> None:
    """多執行緒切割 TIFF 影像"""
    output_dir.mkdir(parents=True, exist_ok=True)

    img = pyvips.Image.openslideload(str(input_tiff), level=0)
    width, height = img.width, img.height

    print(f"影像尺寸: {width} x {height}")
    print(f"使用 {workers} 個執行緒")

    tasks = []
    for y in range(0, height, tile_height):
        for x in range(0, width, tile_width):
            w = min(tile_width, width - x)
            h = min(tile_height, height - y)
            tasks.append((input_tiff, output_dir, x, y, w, h))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        tile_count = 0
        for _ in executor.map(_save_tile, tasks):
            tile_count += 1
            if tile_count % 100 == 0:
                print(f"已生成 {tile_count}/{len(tasks)} 個 tiles...")

    print(f"完成！共生成 {tile_count} 個 tiles")


if __name__ == "__main__":
    input_tiff = Path(r"G:\output\Merged_Aligned_lv0.tiff")
    output_dir = Path(r"G:\output\level0")

    # SSD 用 4，NVMe 可用 8
    generate_tiles(input_tiff, output_dir, tile_width=4112, tile_height=4928, workers=7)
