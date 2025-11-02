"""
TIFF to Pyramidal TIFF Converter (Using pyvips for large files)
將普通 TIFF 圖像轉換為多層金字塔 TIFF 格式
使用 pyvips 處理大型圖像，記憶體效率更高
"""

import os
import sys
import argparse
import pyvips
from pathlib import Path


def create_pyramid_tiff(input_path, output_path, tile_size=256, compression='jpeg', quality=85, depth='onepixel'):
    """
    將 TIFF 圖像轉換為多層金字塔 TIFF（使用 pyvips）

    參數:
        input_path (str): 輸入圖像檔案路徑
        output_path (str): 輸出金字塔 TIFF 檔案路徑
        tile_size (int): 瓷磚大小（預設 256）
        compression (str): 壓縮方式 ('jpeg', 'lzw', 'deflate', 'none')
        quality (int): JPEG 壓縮質量 (1-100)
        depth (str/int): 金字塔深度 ('onepixel', 'onetile') 或指定層數
    """

    print(f"讀取圖像: {input_path}")

    # 使用 pyvips 載入圖像（支援多種格式，記憶體效率高）
    try:
        image = pyvips.Image.new_from_file(input_path, access='sequential')
    except Exception as e:
        print(f"錯誤: 無法載入圖像 - {str(e)}")
        sys.exit(1)

    print(f"原始圖像大小: {image.width} x {image.height}")
    print(f"色彩通道: {image.bands}")
    print(f"圖像格式: {image.format}")

    # 刪除舊檔案
    if os.path.exists(output_path):
        print(f"刪除舊檔案: {output_path}")
        os.remove(output_path)

    # 設定壓縮參數
    compression_map = {
        'jpeg': pyvips.ForeignTiffCompression.JPEG,
        'lzw': pyvips.ForeignTiffCompression.LZW,
        'deflate': pyvips.ForeignTiffCompression.DEFLATE,
        'none': pyvips.ForeignTiffCompression.NONE
    }

    vips_compression = compression_map.get(compression, pyvips.ForeignTiffCompression.JPEG)

    # 計算金字塔層數
    import math
    use_subifd = False
    
    if depth == 'onepixel':
        depth_mode = 'onepixel'
        min_dim = min(image.width, image.height)
        calc_layers = math.ceil(math.log2(min_dim))
        print(f"金字塔深度: {calc_layers} 層 (直到 1 像素)")
    elif depth == 'onetile':
        depth_mode = 'onetile'
        min_dim = min(image.width, image.height)
        calc_layers = max(0, math.ceil(math.log2(min_dim / tile_size)))
        print(f"金字塔深度: {calc_layers} 層 (直到一個瓷磚)")
    else:
        try:
            subifd_layers = int(depth)
            use_subifd = True
            depth_mode = None
            print(f"金字塔深度: {subifd_layers} 層 (自訂)")
        except:
            depth_mode = 'onepixel'
            use_subifd = False
            print(f"金字塔深度: 自動 (直到 1 像素)")

    print(f"\n開始生成金字塔 TIFF...")
    print(f"  瓷磚大小: {tile_size}x{tile_size}")
    print(f"  壓縮方式: {compression.upper()}")
    if compression == 'jpeg':
        print(f"  JPEG 質量: {quality}")

    # 保存為金字塔 TIFF
    try:
        save_options = {
            'compression': vips_compression,
            'tile': True,
            'tile_width': tile_size,
            'tile_height': tile_size,
            'pyramid': True,
            'bigtiff': True
        }
        
        if use_subifd:
            save_options['subifd'] = True
            # 建立金字塔層級
            pyramid_layers = [image]
            for i in range(subifd_layers - 1):
                prev = pyramid_layers[-1]
                next_layer = prev.shrink(2, 2)
                pyramid_layers.append(next_layer)
            
            # 設定 subifd 頁面
            if len(pyramid_layers) > 1:
                save_options['page_height'] = image.height
        else:
            save_options['depth'] = depth_mode

        if compression == 'jpeg':
            save_options['Q'] = quality

        if use_subifd and len(pyramid_layers) > 1:
            # 使用 arrayjoin 合併所有層級
            combined = pyvips.Image.arrayjoin(pyramid_layers, across=1)
            combined.write_to_file(output_path, **save_options)
        else:
            image.write_to_file(output_path, **save_options)

        print(f"\n✓ 完成！金字塔 TIFF 已保存")

        # 顯示檔案大小
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"檔案大小: {file_size:.2f} MB")

        # 顯示金字塔資訊
        print(f"\n檢查生成的金字塔:")
        result_image = pyvips.Image.new_from_file(output_path)
        print(f"  層級 0: {result_image.width} x {result_image.height}")

        # 嘗試讀取其他層級
        level = 1
        while True:
            try:
                level_img = pyvips.Image.new_from_file(output_path, page=level)
                print(f"  層級 {level}: {level_img.width} x {level_img.height}")
                level += 1
                if level > 20:  # 安全限制
                    break
            except:
                break

        if level > 1:
            print(f"\n總共 {level} 層金字塔")

    except Exception as e:
        print(f"錯誤: 保存金字塔 TIFF 失敗 - {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='將 TIFF 圖像轉換為多層金字塔 TIFF (使用 pyvips)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python tiff_to_pyramid.py input.tiff -o output_pyramid.tiff
  python tiff_to_pyramid.py input.tiff -o output.tiff -c jpeg -q 90
  python tiff_to_pyramid.py input.tiff -o output.tiff -t 512 -c lzw
  python tiff_to_pyramid.py input.tiff -o output.tiff -d onetile
        """
    )

    parser.add_argument('input', help='輸入圖像檔案路徑（支援 TIFF, PNG, JPEG 等）')
    parser.add_argument('-o', '--output', help='輸出金字塔 TIFF 檔案路徑（預設為 input_pyramid.tiff）')
    parser.add_argument('-t', '--tile-size', type=int, default=256,
                        help='瓷磚大小（預設為 256）')
    parser.add_argument('-d', '--depth', default='onepixel',
                        help='金字塔深度：onepixel（預設）, onetile, 或指定層數')
    parser.add_argument('-c', '--compression', default='jpeg',
                        choices=['jpeg', 'lzw', 'deflate', 'none'],
                        help='壓縮方式（預設為 jpeg）')
    parser.add_argument('-q', '--quality', type=int, default=85,
                        help='JPEG 壓縮質量 1-100（預設為 85）')

    args = parser.parse_args()

    # 檢查輸入檔案
    if not os.path.exists(args.input):
        print(f"錯誤: 找不到輸入檔案 '{args.input}'")
        sys.exit(1)

    # 設定輸出路徑
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_pyramid.tiff")

    # 執行轉換
    try:
        create_pyramid_tiff(
            input_path=args.input,
            output_path=args.output,
            tile_size=args.tile_size,
            compression=args.compression,
            quality=args.quality,
            depth=args.depth
        )
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

