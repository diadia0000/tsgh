"""
CZI to TIFF 轉換腳本
使用 aicspylibczi 分塊讀取 mosaic tiles，避免記憶體爆掉
"""
from pathlib import Path
import numpy as np
from tifffile import TiffWriter
from aicspylibczi import CziFile
from tqdm import tqdm
import gc


def czi_to_tiff_mosaic(czi_path: Path, output_path: Path, tile_size: int = 512):
    """
    將 CZI 檔案轉換為 TIFF，處理 mosaic tiles
    
    Args:
        czi_path: CZI 檔案路徑
        output_path: 輸出 TIFF 檔案路徑
        tile_size: TIFF tile 大小
    """
    print(f"正在處理: {czi_path.name}")
    
    czi = CziFile(czi_path)
    
    # 取得維度資訊
    dims = czi.get_dims_shape()[0]
    print(f"  維度資訊: {dims}")
    
    # 取得 mosaic 的邊界框
    bbox = czi.get_mosaic_bounding_box()
    if bbox is None:
        print("  警告: 無法取得 mosaic bounding box，嘗試直接讀取...")
        # 嘗試直接讀取單一 tile
        img_data, _ = czi.read_image()
        img_data = np.squeeze(img_data)
        if img_data.ndim == 3 and img_data.shape[0] <= 4:
            img_data = np.moveaxis(img_data, 0, -1)
    else:
        print(f"  Mosaic 邊界框: x={bbox.x}, y={bbox.y}, w={bbox.w}, h={bbox.h}")
        
        # 使用 read_mosaic 分塊讀取
        # 這個方法會在內部處理 mosaic 拼接
        # 設定 scale_factor 來降低解析度，減少記憶體使用
        scale_factor = 1.0  # 可以設成 0.5 或 0.25 來進一步減少記憶體
        
        print(f"  正在讀取 mosaic (scale_factor={scale_factor})...")
        img_data = czi.read_mosaic(scale_factor=scale_factor, C=0)  # 只讀取第一個 channel
        img_data = np.squeeze(img_data)
        print(f"  讀取完成，形狀: {img_data.shape}")
    
    # 處理維度順序
    if img_data.ndim == 3 and img_data.shape[0] <= 4:
        img_data = np.moveaxis(img_data, 0, -1)
        print(f"  轉換後形狀: {img_data.shape}")
    
    # 確保是 uint8
    if img_data.dtype != np.uint8:
        if img_data.max() > 0:
            img_data = (img_data.astype(np.float32) / img_data.max() * 255).astype(np.uint8)
        else:
            img_data = img_data.astype(np.uint8)
    
    # BGR -> RGB 轉換 (CZI 通常是 BGR 順序)
    if img_data.ndim == 3 and img_data.shape[-1] >= 3:
        img_data = img_data[..., ::-1].copy()  # BGR -> RGB
        print(f"  已轉換 BGR -> RGB")
    
    # 寫入 TIFF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"  正在寫入 TIFF...")
    with TiffWriter(str(output_path), bigtiff=True) as tif:
        tif.write(
            img_data,
            tile=(min(tile_size, img_data.shape[0]), min(tile_size, img_data.shape[1])),
            compression='lzw',
            photometric='rgb' if img_data.ndim == 3 and img_data.shape[-1] >= 3 else 'minisblack',
        )
    
    print(f"  完成! 輸出: {output_path}")
    print(f"  檔案大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 清理記憶體
    del img_data
    del czi
    gc.collect()


def main():
    wsi_dir = Path("/home/sec312/tsgh/picture/czi/reigion/")
    output_dir = Path("/home/sec312/tsgh/picture/tiff/reigion/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    czi_files = list(wsi_dir.glob("*.czi"))
    print(f"找到 {len(czi_files)} 個 CZI 檔案")
    
    for czi_file in tqdm(czi_files, desc="轉換進度"):
        output_path = output_dir / czi_file.with_suffix(".tiff").name
        
        # 跳過已存在的檔案
        if output_path.exists():
            print(f"跳過 (已存在): {output_path.name}")
            continue
        
        try:
            czi_to_tiff_mosaic(czi_file, output_path)
        except Exception as e:
            print(f"錯誤處理 {czi_file.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()