"""Module 5: Tile Generator - 同時切割 HER2、DISH、Merged 三組對齊的 Tiles"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pyvips


def _save_triple_tiles(args):
    """
    同時保存三組對應的 tiles (HER2, DISH, Merged)
    
    Args:
        args: (her2_path, dish_path, output_dirs, x, y, w, h)
    """
    her2_path, dish_path, output_dirs, x, y, w, h = args
    
    her2_dir, dish_dir = output_dirs
    
    # 讀取並切割 HER2 tile
    her2_img = pyvips.Image.new_from_file(str(her2_path), access='sequential')
    her2_tile = her2_img.crop(x, y, w, h)
    her2_tile.write_to_file(
        str(her2_dir / f"tile_x{x}_y{y}.tiff"), 
        compression='deflate'
    )
    
    # 讀取並切割 DISH tile
    dish_img = pyvips.Image.new_from_file(str(dish_path), access='sequential')
    dish_tile = dish_img.crop(x, y, w, h)
    dish_tile.write_to_file(
        str(dish_dir / f"tile_x{x}_y{y}.tiff"), 
        compression='deflate'
    )
    return 1


def generate_triple_tiles(
    her2_tiff: Path,
    dish_tiff: Path,
    output_base_dir: Path,
    tile_width: int = 512,
    tile_height: int = 512,
    workers: int = 4,
) -> None:
    """
    同時切割三組對齊的 TIFF 影像 (HER2, DISH, Merged)
    
    Args:
        her2_tiff: HER2 影像路徑
        dish_tiff: DISH 影像路徑
        output_base_dir: 輸出基礎目錄
        tile_width: Tile 寬度
        tile_height: Tile 高度
        workers: 執行緒數量
    """
    # 創建輸出目錄
    her2_dir = output_base_dir / "her2"
    dish_dir = output_base_dir / "dish"
    
    her2_dir.mkdir(parents=True, exist_ok=True)
    dish_dir.mkdir(parents=True, exist_ok=True)
    
    output_dirs = (her2_dir, dish_dir)
    
    # 檢查檔案是否存在
    for name, path in [("HER2", her2_tiff), ("DISH", dish_tiff)]:
        if not path.exists():
            raise FileNotFoundError(f"{name} 檔案不存在: {path}")
    
    # 讀取影像尺寸 (使用 HER2 作為參考)
    ref_img = pyvips.Image.new_from_file(str(her2_tiff), access='sequential')
    width = ref_img.width
    height = ref_img.height
    print(f"影像尺寸: {width} x {height}")
    del ref_img  # 釋放記憶體
    
    # 生成切割任務
    tasks = []
    for y in range(0, height, tile_height):
        for x in range(0, width, tile_width):
            w = min(tile_width, width - x)
            h = min(tile_height, height - y)
            tasks.append((her2_tiff, dish_tiff, output_dirs, x, y, w, h))
    
    print(f"預計生成 {len(tasks)} 組 tiles（每組包含 HER2, DISH）\n")
    
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
    print("=" * 60)


if __name__ == "__main__":
    # 設定路徑
    output_base = Path("/home/sec312/tsgh/picture/tiff/reigion/")
    
    her2_tiff = output_base / "HER2_region.tiff"
    dish_tiff = output_base / "DISH_region.tiff"
    output_dir = output_base / "tiles_region-2048"
    
    # 切割 tiles（使用 2048x2048)
    generate_triple_tiles(
        her2_tiff=her2_tiff,
        dish_tiff=dish_tiff,
        output_base_dir=output_dir,
        tile_width=2048,
        tile_height=2048,
        workers=16  # 根據您的 CPU 核心數調整
    )
