#!/usr/bin/env python3
"""
快速轉換 TIFF 為 SubIFD 格式

將現有的金字塔 TIFF 轉換為 SubIFD 格式，避免重跑 Module 1
這樣 VALIS 可以正確讀取這些檔案
"""
import os
import shutil
from pathlib import Path
import pyvips

def convert_to_subifd(input_path: Path, output_path: Path = None):
    """
    將 TIFF 轉換為 SubIFD 金字塔格式
    
    Args:
        input_path: 輸入 TIFF 路徑
        output_path: 輸出路徑 (預設為覆蓋原檔)
    """
    input_path = Path(input_path)
    
    if output_path is None:
        # 使用暫存檔案
        temp_path = input_path.with_suffix('.temp.tiff')
        output_path = input_path
        use_temp = True
    else:
        output_path = Path(output_path)
        temp_path = output_path
        use_temp = False
    
    print(f"讀取: {input_path}")
    
    # 只讀取第一頁 (全解析度)
    # 不要用 n=-1，因為那會讀取所有頁面並失敗
    img = pyvips.Image.new_from_file(str(input_path), page=0)
    
    print(f"  尺寸: {img.width} x {img.height}")
    print(f"  通道: {img.bands}")
    
    print(f"寫入: {temp_path}")
    
    # 使用 SubIFD 格式重新生成金字塔
    img.tiffsave(
        str(temp_path),
        compression="jpeg",
        Q=95,
        tile=True,
        tile_width=512,
        tile_height=512,
        bigtiff=True,
        pyramid=True,
        subifd=True,  # 關鍵：使用 SubIFD 格式
    )
    
    if use_temp:
        # 刪除原檔，重命名暫存檔
        print(f"替換原檔...")
        os.remove(str(input_path))
        shutil.move(str(temp_path), str(input_path))
    
    # 檢查檔案大小
    size_gb = os.path.getsize(str(output_path if not use_temp else input_path)) / (1024**3)
    print(f"✓ 完成: {size_gb:.2f} GB")
    
    return True


def main():
    """轉換所有處理過的 TIFF"""
    output_dir = Path("/home/hispadmin/tsgh/thriple_image_layer/output")
    
    tiff_files = [
        "DISH_processed.tiff",
        "HE_processed.tiff", 
        "HER2_processed.tiff",
    ]
    
    print("=" * 60)
    print("TIFF 格式轉換器 (轉換為 SubIFD 金字塔格式)")
    print("=" * 60)
    print()
    
    for filename in tiff_files:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"\n處理 {filename}...")
            try:
                convert_to_subifd(filepath)
            except Exception as e:
                print(f"✗ 失敗: {e}")
        else:
            print(f"跳過 {filename}: 檔案不存在")
    
    print("\n" + "=" * 60)
    print("轉換完成！現在可以執行 Module 2")
    print("=" * 60)


if __name__ == "__main__":
    main()
