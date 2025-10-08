"""
Phase 0.2: Her2 影像前處理 - 灰階轉換
"""
import openslide
import numpy as np
import cv2
import tifffile
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def _process_tile_worker_her2(args):
    """多進程 worker 函數"""
    x, y, w, h, path, level, downsample = args
    slide_local = openslide.OpenSlide(str(path))
    # 將 level 座標轉換為 level 0 座標
    x0 = int(x * downsample)
    y0 = int(y * downsample)
    tile = slide_local.read_region((x0, y0), level, (w, h))
    tile_rgb = np.array(tile.convert('RGB'))
    
    # 轉灰階
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    
    # 自適應 CLAHE
    std_dev = np.std(gray)
    clip_limit = max(1.0, min(4.0, std_dev / 20.0))
    tile_size_clahe = 8 if gray.shape[0] < 1024 else 16
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size_clahe, tile_size_clahe))
    enhanced = clahe.apply(gray)
    
    slide_local.close()
    return x, y, w, h, enhanced


def preprocess_her2_image(
    input_path: Path,
    output_path: Path,
    output_level: int = 1
) -> None:
    """Her2 影像前處理 - 灰階轉換"""
    print(f"[Phase 0.2] 開始處理 Her2 影像: {input_path}")
    
    slide = openslide.OpenSlide(str(input_path))
    print(f"  層級數: {slide.level_count}")
    print(f"  Level {output_level} 尺寸: {slide.level_dimensions[output_level]}")
    print(f"  輸出將縮小至 1/8 面積")
    
    print(f"  [步驟 1/2] 分塊處理 Level {output_level} 影像...")
    level_size = slide.level_dimensions[output_level]
    tile_size = 2048
    output_shape = (level_size[1], level_size[0])
    gray_full = np.zeros(output_shape, dtype=np.uint8)
    
    tiles = [(x, y, min(tile_size, level_size[0] - x), min(tile_size, level_size[1] - y))
             for y in range(0, level_size[1], tile_size)
             for x in range(0, level_size[0], tile_size)]
    
    n_workers = max(1, cpu_count() - 1)
    print(f"    使用 {n_workers} 個核心處理")
    
    # 取得 downsample 倍率
    downsample = slide.level_downsamples[output_level]
    tasks = [(x, y, w, h, input_path, output_level, downsample) for x, y, w, h in tiles]
    
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(_process_tile_worker_her2, tasks), total=len(tasks), desc="    處理分塊"))
    
    for x, y, w, h, enhanced in results:
        gray_full[y:y+h, x:x+w] = enhanced
    
    print(f"  [步驟 2/2] 縮小並儲存輸出影像...")
    # 縮小 1/8 (寬高各縮小至 1/√8 ≈ 0.354)
    new_h = int(gray_full.shape[0] / 2.828)
    new_w = int(gray_full.shape[1] / 2.828)
    gray_resized = cv2.resize(gray_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output_path), gray_resized, photometric='minisblack')
    
    slide.close()
    print(f"[Phase 0.2] 完成！輸出: {output_path}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    
    input_file = Path("../picture/WSI/Her2.tiff")
    output_file = Path("output/Her2_Gray.tiff")
    
    preprocess_her2_image(input_file, output_file)
