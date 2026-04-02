"""Module 5: Tile Generator

同時切割 HER2、DISH、Merged 三組對齊的 Tiles
使用 OpenSlide 讀取大型金字塔 TIFF 影像
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict
import numpy as np
from PIL import Image
import openslide
import tifffile

from config import RegistrationConfig, create_default_config


# TIFF 壓縮格式對應
COMPRESSION_MAP: Dict[str, str] = {
    "deflate": "ADOBE_DEFLATE",
    "lzw": "LZW",
    "jpeg": "JPEG",
    "none": None,
}


def _read_region_openslide(
    slide: openslide.OpenSlide,
    x: int,
    y: int,
    width: int,
    height: int,
    level: int = 0,
    pad_to: Tuple[int, int] = None,
) -> np.ndarray:
    """使用 OpenSlide 讀取指定區域，邊界不足時以白色填充至指定尺寸

    Args:
        slide: OpenSlide 物件
        x: 左上角 X 座標 (pixels, level 0)
        y: 左上角 Y 座標 (pixels, level 0)
        width: 實際讀取寬度 (pixels, 已裁切至影像邊界)
        height: 實際讀取高度 (pixels, 已裁切至影像邊界)
        level: 金字塔層級
        pad_to: (target_w, target_h) 若提供，將結果填充至此尺寸

    Returns:
        np.ndarray: RGB 影像陣列 (H, W, 3)
    """
    # OpenSlide 的 read_region 返回 RGBA，需轉換為 RGB
    region = slide.read_region((x, y), level, (width, height))
    tile = np.array(region.convert("RGB"))

    if pad_to is not None:
        target_w, target_h = pad_to
        if width < target_w or height < target_h:
            padded = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
            padded[:height, :width] = tile
            return padded

    return tile


def _save_tile_tiff(
    tile_array: np.ndarray,
    output_path: Path,
    compression: str = "deflate",
) -> None:
    """將 tile 陣列保存為 TIFF

    Args:
        tile_array: RGB 影像陣列 (H, W, 3)
        output_path: 輸出路徑
        compression: 壓縮格式
    """
    tiff_compression = COMPRESSION_MAP.get(compression, "ADOBE_DEFLATE")
    tifffile.imwrite(
        str(output_path),
        tile_array,
        compression=tiff_compression,
        photometric="rgb",
    )


def _save_triple_tiles_openslide(args: Tuple) -> int:
    """同時保存三組對應的 tiles (HER2, DISH, Merged)

    Args:
        args: (her2_path, dish_path, merged_path, output_dirs,
               x, y, w, h, tile_w, tile_h, compression, level)

    Returns:
        int: 1 表示成功
    """
    (her2_path, dish_path, merged_path,
     output_dirs, x, y, w, h, tile_w, tile_h, compression, level) = args

    her2_dir, dish_dir, merged_dir = output_dirs
    pad_to = (tile_w, tile_h)

    # 開啟並讀取 HER2 tile
    with openslide.OpenSlide(str(her2_path)) as her2_slide:
        her2_tile = _read_region_openslide(her2_slide, x, y, w, h, level, pad_to)
    _save_tile_tiff(
        her2_tile,
        her2_dir / f"tile_x{x}_y{y}.tiff",
        compression,
    )

    # 開啟並讀取 DISH tile
    with openslide.OpenSlide(str(dish_path)) as dish_slide:
        dish_tile = _read_region_openslide(dish_slide, x, y, w, h, level, pad_to)
    _save_tile_tiff(
        dish_tile,
        dish_dir / f"tile_x{x}_y{y}.tiff",
        compression,
    )

    # 開啟並讀取 Merged tile
    with openslide.OpenSlide(str(merged_path)) as merged_slide:
        merged_tile = _read_region_openslide(merged_slide, x, y, w, h, level, pad_to)
    _save_tile_tiff(
        merged_tile,
        merged_dir / f"tile_x{x}_y{y}.tiff",
        compression,
    )

    return 1


def generate_triple_tiles(
    config: RegistrationConfig,
    level: int = 0,
) -> None:
    """同時切割三組對齊的 TIFF 影像 (HER2, DISH, Merged)

    使用 OpenSlide 讀取金字塔 TIFF，支援 BigTIFF 格式

    Args:
        config: 配準流程配置
        level: 使用的金字塔層級 (0 = 最高解析度)
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

    # 使用 OpenSlide 讀取影像尺寸
    with openslide.OpenSlide(str(merged_tiff)) as merged_slide:
        width, height = merged_slide.dimensions

    print("=" * 60)
    print(f"影像尺寸: {width} x {height}")
    print(f"Tile 尺寸: {tile_width} x {tile_height}")
    print(f"壓縮格式: {compression}")
    print(f"使用 {workers} 個執行緒")
    print("=" * 60)

    # 驗證三張圖尺寸相同
    with openslide.OpenSlide(str(her2_tiff)) as her2_slide:
        her2_dims = her2_slide.dimensions

    with openslide.OpenSlide(str(dish_tiff)) as dish_slide:
        dish_dims = dish_slide.dimensions

    print(f"\n尺寸驗證:")
    print(f"  Merged: {width} x {height}")
    print(f"  HER2:   {her2_dims[0]} x {her2_dims[1]}")
    print(f"  DISH:   {dish_dims[0]} x {dish_dims[1]}")

    if (her2_dims[0] != width or her2_dims[1] != height or
            dish_dims[0] != width or dish_dims[1] != height):
        print("\n❌ 錯誤：三張影像尺寸不同！")
        if her2_dims[0] != width or her2_dims[1] != height:
            print(f"  HER2 不匹配：{her2_dims[0]} x {her2_dims[1]} != {width} x {height}")
        if dish_dims[0] != width or dish_dims[1] != height:
            print(f"  DISH 不匹配：{dish_dims[0]} x {dish_dims[1]} != {width} x {height}")
        raise ValueError("三張影像尺寸不同！請使用 check_tiff_size.py 檢查檔案。")

    print("✓ 驗證通過：三張圖像尺寸相同\n")

    # 生成切割任務
    tasks = []
    for y_pos in range(0, height, tile_height):
        for x_pos in range(0, width, tile_width):
            w = min(tile_width, width - x_pos)
            h = min(tile_height, height - y_pos)
            tasks.append((
                her2_tiff, dish_tiff, merged_tiff,
                output_dirs, x_pos, y_pos, w, h,
                tile_width, tile_height, compression, level
            ))

    print(f"預計生成 {len(tasks)} 組 tiles（每組包含 HER2, DISH, Merged）\n")

    # 多執行緒處理
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tile_count = 0
        for _ in executor.map(_save_triple_tiles_openslide, tasks):
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
    print("Module 5: Tile Generator (OpenSlide)")
    print("=" * 60)
    print(f"Tile 尺寸: {config.tile.tile_width} x {config.tile.tile_height}")
    print(f"執行緒數: {config.tile.workers}")
    print()

    # 使用 level 0 切割 tiles
    generate_triple_tiles(config, level=0)
