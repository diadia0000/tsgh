from pathlib import Path
from typing import Tuple
import numpy as np
import openslide
from skimage import morphology
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def create_tissue_mask(
    rgb_image: np.ndarray,
    saturation_threshold: int = 20,
    open_kernel_size: int = 3,
    close_kernel_size: int = 5
) -> np.ndarray:
    """
    從一張 RGB 組織病理影像中，生成一個精確的二值化組織遮罩 (binary tissue mask)。
    此函式專為分離組織與淺色/白色背景而優化。

    Args:
        rgb_image (np.ndarray): 輸入的 RGB 影像 (HxWx3)。
        saturation_threshold (int): HSV 色彩空間中飽和度(S)的閾值。
                                   低於此值的像素被視為背景。
        open_kernel_size (int): 形態學開運算的核心大小，用於移除小的噪點。
        close_kernel_size (int): 形態學閉運算的核心大小，用於填補組織內部的小空洞。

    Returns:
        np.ndarray: 一個二值化的 uint8 遮罩 (HxW)，組織區域為 255，背景為 0。
    """
    # 步驟 1: 確保輸入是 3 通道的 RGB 影像
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        # 如果是 RGBA 影像，則移除 Alpha 通道
        if rgb_image.shape[2] == 4:
            rgb_image = rgb_image[:, :, :3]
        else:
            raise ValueError("輸入必須是 3 通道的 RGB 影像")

    # 步驟 2: 將影像從 RGB 轉換為 HSV 色彩空間
    # HSV 能更好地分離顏色強度(V)與色彩飽和度(S)，後者對區分背景非常有效
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # 步驟 3: 提取飽和度(S)通道
    s_channel = hsv_image[:, :, 1]

    # 步驟 4: 根據飽和度閾值進行二值化，產生初始遮罩
    _, binary_mask = cv2.threshold(s_channel, saturation_threshold, 255, cv2.THRESH_BINARY)

    # 步驟 5: 進行形態學清洗，以優化遮罩品質
    # 開運算 (Opening): 先侵蝕後膨脹，可以有效移除孤立的小噪點
    open_kernel = morphology.disk(open_kernel_size)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel)

    # 閉運算 (Closing): 先膨脹後侵蝕，可以填補組織內部的小空洞
    close_kernel = morphology.disk(close_kernel_size)
    final_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, close_kernel)

    return final_mask


def process_single_tile(args: Tuple) -> Tuple:
    """處理單個 tile (用於多進程)"""
    wsi_path, level, tile_size, tx, ty, width, height, downsample, saturation_threshold = args

    slide = openslide.OpenSlide(str(wsi_path))

    x = tx * tile_size
    y = ty * tile_size
    w = min(tile_size, width - x)
    h = min(tile_size, height - y)

    x0 = int(x * downsample)
    y0 = int(y * downsample)
    tile_image = slide.read_region((x0, y0), level, (w, h))
    tile = np.array(tile_image.convert('RGB'))

    slide.close()

    # 跳過空白 tile
    if np.mean(tile) > 245:
        return None
    try:
        tissue_mask = create_tissue_mask(tile, saturation_threshold=saturation_threshold, 
                                        open_kernel_size=3, close_kernel_size=5)
        # 返回組織遮罩 (組織=255, 背景=0)
        return (x, y, w, h, tissue_mask)
    except:
        return None


def process_wsi_tissue_mask(
    wsi_path: Path,
    output_dir: Path,
    level: int = 2,
    tile_size: int = 2048,
    n_workers: int = None,
    saturation_threshold: int = 40
) -> None:
    """
    處理整個 WSI 圖像並生成組織遮罩

    Args:
        wsi_path: WSI 圖像路徑
        output_dir: 輸出目錄
        level: 處理的金字塔層級 (0=最高解析度)
        tile_size: tile 大小
        n_workers: 並行處理數量 (None=自動)
        saturation_threshold: 飽和度閾值 (值越高越嚴格，預設40)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    # 開啟 WSI
    slide = openslide.OpenSlide(str(wsi_path))
    
    print(f"WSI 層級數: {slide.level_count}")
    print(f"各層級尺寸: {slide.level_dimensions}")
    print(f"處理層級 {level}: {slide.level_dimensions[level]}")
    print(f"飽和度閾值: {saturation_threshold}")

    width, height = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    slide.close()

    # 計算 tile 數量
    n_tiles_x = (width + tile_size - 1) // tile_size
    n_tiles_y = (height + tile_size - 1) // tile_size
    total_tiles = n_tiles_x * n_tiles_y
    
    print(f"總共需要處理 {n_tiles_x} x {n_tiles_y} = {total_tiles} 個 tiles")
    print(f"使用 {n_workers} 個進程並行處理")

    # 初始化輸出遮罩（白色背景）
    final_mask = np.full((height, width), 255, dtype=np.uint8)

    # 準備所有 tile 參數
    tile_args = [
        (wsi_path, level, tile_size, tx, ty, width, height, downsample, saturation_threshold)
        for ty in range(n_tiles_y)
        for tx in range(n_tiles_x)
    ]

    # 多進程處理
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_tile, tile_args),
            total=total_tiles,
            desc="處理 tiles"
        ))

    # 組合結果
    for result in tqdm(results, desc="組合結果"):
        if result is None:
            continue
        x, y, w, h, background_mask = result

        # 將 tissue_mask (組織為255) 中為 255 的部分，在 final_mask 對應位置設為 0 (黑色)
        # 這樣可以將組織區域標記為黑色，背景保持白色
        final_mask[y:y+h, x:x+w][background_mask == 255] = 0

    # 儲存遮罩
    cv2.imwrite(str(output_dir / "DISH_nuclear_mask.png"), final_mask)

    print("完成!")


if __name__ == "__main__"
    wsi_path = Path("../picture/WSI/DISH_20X_ED7.tiff")
    output_dir = Path("./output")
    process_wsi_tissue_mask(
        wsi_path=wsi_path,
        output_dir=output_dir,
        level=1,
        tile_size=4096,
        n_workers=None,  # 自動使用 CPU 核心數-1
        saturation_threshold=15  # 將閾值提高到 20，避免將背景誤判為組織
    )
