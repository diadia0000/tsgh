"""
檢查兩個 TIFF 檔案的尺寸是否相同
"""
import argparse
from pathlib import Path

try:
    import pyvips
    USE_PYVIPS = True
except ImportError:
    USE_PYVIPS = False

try:
    import tifffile
    USE_TIFFFILE = True
except ImportError:
    USE_TIFFFILE = False


def get_tiff_info_pyvips(path: Path) -> dict:
    """使用 pyvips 讀取 TIFF 資訊（支援大型 TIFF）"""
    img = pyvips.Image.new_from_file(str(path), access='sequential')
    return {
        'width': img.width,
        'height': img.height,
        'bands': img.bands,
        'format': img.format,
        'interpretation': img.interpretation,
    }


def get_tiff_info_tifffile(path: Path) -> dict:
    """使用 tifffile 讀取 TIFF 資訊"""
    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        return {
            'width': page.imagewidth,
            'height': page.imagelength,
            'bands': page.samplesperpixel,
            'dtype': str(page.dtype),
            'pages': len(tif.pages),
        }


def compare_tiff_files(path1: Path, path2: Path) -> None:
    """比較兩個 TIFF 檔案的尺寸"""
    print("=" * 60)
    print("TIFF 檔案尺寸比較工具")
    print("=" * 60)
    
    # 檢查檔案是否存在
    if not path1.exists():
        print(f"❌ 檔案不存在: {path1}")
        return
    if not path2.exists():
        print(f"❌ 檔案不存在: {path2}")
        return
    
    # 選擇讀取方式
    if USE_PYVIPS:
        print("使用 pyvips 讀取...\n")
        info1 = get_tiff_info_pyvips(path1)
        info2 = get_tiff_info_pyvips(path2)
    elif USE_TIFFFILE:
        print("使用 tifffile 讀取...\n")
        info1 = get_tiff_info_tifffile(path1)
        info2 = get_tiff_info_tifffile(path2)
    else:
        print("❌ 需要安裝 pyvips 或 tifffile")
        print("   pip install pyvips tifffile")
        return
    
    # 顯示檔案 1 資訊
    print(f"📁 檔案 1: {path1.name}")
    print(f"   路徑: {path1}")
    for key, value in info1.items():
        print(f"   {key}: {value}")
    
    print()
    
    # 顯示檔案 2 資訊
    print(f"📁 檔案 2: {path2.name}")
    print(f"   路徑: {path2}")
    for key, value in info2.items():
        print(f"   {key}: {value}")
    
    print()
    print("-" * 60)
    
    # 比較尺寸
    width_match = info1['width'] == info2['width']
    height_match = info1['height'] == info2['height']
    bands_match = info1['bands'] == info2['bands']
    
    if width_match and height_match:
        print("✅ 尺寸相同!")
        print(f"   寬度: {info1['width']} px")
        print(f"   高度: {info1['height']} px")
        if bands_match:
            print(f"   通道數: {info1['bands']}")
        else:
            print(f"   ⚠️  通道數不同: {info1['bands']} vs {info2['bands']}")
    else:
        print("❌ 尺寸不同!")
        print(f"   檔案 1: {info1['width']} x {info1['height']}")
        print(f"   檔案 2: {info2['width']} x {info2['height']}")
        
        # 計算差異
        width_diff = abs(info1['width'] - info2['width'])
        height_diff = abs(info1['height'] - info2['height'])
        print(f"   差異: {width_diff} x {height_diff} px")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='比較兩個 TIFF 檔案的尺寸')
    parser.add_argument('file1', type=Path, help='第一個 TIFF 檔案路徑')
    parser.add_argument('file2', type=Path, help='第二個 TIFF 檔案路徑')
    
    args = parser.parse_args()
    compare_tiff_files(args.file1, args.file2)


if __name__ == "__main__":
    # 如果沒有提供參數，使用範例路徑
    import sys
    
    if len(sys.argv) < 3:
        print("使用方式:")
        print("  python check_tiff_size.py <tiff1> <tiff2>")
        print()
        print("範例:")
        print("  python check_tiff_size.py dish_warped_lv0.ome.tiff her2_warped_lv0.ome.tiff")
        print()
        
        # 嘗試自動尋找 temp 目錄中的檔案
        possible_dirs = [
            Path("/home/sec312/tsgh/thriple_image_layer/output/temp"),
            Path("H:/tsgh/thriple_image_layer/output"),
        ]
        
        for temp_dir in possible_dirs:
            if temp_dir.exists():
                tiff_files = list(temp_dir.glob("*.tiff")) + list(temp_dir.glob("*.ome.tiff"))
                if len(tiff_files) >= 2:
                    print(f"找到檔案在: {temp_dir}")
                    for f in tiff_files:
                        print(f"  - {f.name}")
                break
    else:
        main()
