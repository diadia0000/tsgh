"""
壓縮輸出影像至 1/4 大小
"""
import cv2
import tifffile
from pathlib import Path
from tqdm import tqdm


def compress_image(input_path: Path, output_path: Path, scale: float = 0.5):
    """
    壓縮影像至指定比例
    
    Args:
        input_path: 輸入 TIFF 路徑
        output_path: 輸出 TIFF 路徑
        scale: 縮放比例 (0.5 = 1/4 面積)
    """
    print(f"壓縮影像: {input_path.name}")
    
    # 讀取影像
    img = tifffile.imread(str(input_path))
    print(f"  原始尺寸: {img.shape}")
    
    # 計算新尺寸
    new_h = int(img.shape[0] * scale)
    new_w = int(img.shape[1] * scale)
    
    # 使用 INTER_AREA 進行高品質縮小
    compressed = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"  壓縮尺寸: {compressed.shape}")
    
    # 儲存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output_path), compressed, photometric='minisblack')
    
    # 顯示檔案大小
    original_size = input_path.stat().st_size / (1024**3)
    compressed_size = output_path.stat().st_size / (1024**3)
    print(f"  原始大小: {original_size:.2f} GB")
    print(f"  壓縮大小: {compressed_size:.2f} GB")
    print(f"  壓縮率: {(1 - compressed_size/original_size)*100:.1f}%")


if __name__ == "__main__":
    # 壓縮 DISH
    compress_image(
        Path("output/DISH_Hematoxylin.tiff"),
        Path("output/DISH_Hematoxylin_compressed.tiff")
    )
    print()
    
    # 壓縮 Her2
    compress_image(
        Path("output/Her2_Hematoxylin.tiff"),
        Path("output/Her2_Hematoxylin_compressed.tiff")
    )
