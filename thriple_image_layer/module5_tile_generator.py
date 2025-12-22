"""Module 5: Tile Generator - 同時切割 HER2、DISH、Merged 三組對齊的 Tiles"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pyvips


def _save_triple_tiles(args):
    """
    同時保存三組對應的 tiles (HER2, DISH, Merged)
    
    Args:
        args: (her2_path, dish_path, merged_path, output_dirs, x, y, w, h)
    """
    her2_path, dish_path, merged_path, output_dirs, x, y, w, h = args
    
    her2_dir, dish_dir, merged_dir = output_dirs
    
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
    
    # 讀取並切割 Merged tile
    merged_img = pyvips.Image.new_from_file(str(merged_path), access='sequential')
    merged_tile = merged_img.crop(x, y, w, h)
    merged_tile.write_to_file(
        str(merged_dir / f"tile_x{x}_y{y}.tiff"), 
        compression='deflate'
    )
    
    return 1


def generate_triple_tiles(
    her2_tiff: Path,
    dish_tiff: Path,
    merged_tiff: Path,
    output_base_dir: Path,
    tile_width: int = 512,
    tile_height: int = 512,
    workers: int = 4,
) -> None:
    """
    同時切割三組對齊的 TIFF 影像 (HER2, DISH, Merged)
    
    Args:
        her2_tiff: HER2 影像路徑 (用於生成細胞膜 mask)
        dish_tiff: DISH 影像路徑 (用於檢測紅黑點)
        merged_tiff: 疊合影像路徑 (用於訓練輸入)
        output_base_dir: 輸出基礎目錄
        tile_width: Tile 寬度
        tile_height: Tile 高度
        workers: 執行緒數量
    """
    # 創建輸出目錄
    her2_dir = output_base_dir / "her2"
    dish_dir = output_base_dir / "dish"
    merged_dir = output_base_dir / "merged"
    
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
            tasks.append((her2_tiff, dish_tiff, merged_tiff, output_dirs, x, y, w, h))
    
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
    # 設定路徑
    output_base = Path("/home/sec312/tsgh/thriple_image_layer/output")
    
    her2_tiff = output_base / "temp" / "her2_warped_lv1.ome.tiff"
    dish_tiff = output_base / "temp" / "dish_warped_lv1.ome.tiff"
    merged_tiff = output_base / "Merged_Aligned_lv1.tiff"
    
    output_dir = output_base / "tiles_lv1"
    
    # 切割 tiles（使用 512x512，適合 UNet 訓練）
    generate_triple_tiles(
        her2_tiff=her2_tiff,
        dish_tiff=dish_tiff,
        merged_tiff=merged_tiff,
        output_base_dir=output_dir,
        tile_width=512,
        tile_height=512,
        workers=16  # 根據您的 CPU 核心數調整
    )
