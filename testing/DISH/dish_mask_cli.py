#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DISH 染色細胞遮罩工具 - 命令列版本
Command Line Interface for DISH Cell Masking
"""

import argparse
import sys
from pathlib import Path

from dish_mask_core import DishMaskProcessor, MaskingParams

def main():
    parser = argparse.ArgumentParser(description='DISH 染色細胞遮罩處理工具')
    
    # 基本參數
    parser.add_argument('--input', '-i', type=str, 
                       help='輸入影像路徑 (如未指定則自動載入 picture/tiff/ 中的 DISH 影像)')
    parser.add_argument('--output', '-o', type=str, default='testing/output/middle-gen',
                       help='輸出目錄 (預設: testing/output/middle-gen)')
    
    # HSV 參數
    parser.add_argument('--h-min', type=int, default=0, help='H 最小值 (0-179)')
    parser.add_argument('--h-max', type=int, default=179, help='H 最大值 (0-179)')
    parser.add_argument('--s-min', type=int, default=0, help='S 最小值 (0-255)')
    parser.add_argument('--s-max', type=int, default=255, help='S 最大值 (0-255)')
    parser.add_argument('--v-min', type=int, default=0, help='V 最小值 (0-255)')
    parser.add_argument('--v-max', type=int, default=255, help='V 最大值 (0-255)')
    
    # 形態學參數
    parser.add_argument('--kernel-size', type=int, default=3, help='Kernel 大小 (奇數)')
    parser.add_argument('--open-iter', type=int, default=1, help='開運算迭代次數')
    parser.add_argument('--close-iter', type=int, default=1, help='閉運算迭代次數')
    parser.add_argument('--dilate-iter', type=int, default=2, help='膨脹迭代次數')
    
    # Watershed 參數
    parser.add_argument('--dist-threshold', type=float, default=0.35, 
                       help='距離變換閾值 (0.1-0.8)')
    
    # 顯示參數
    parser.add_argument('--alpha', type=int, default=50, help='透明度 (0-100)')
    
    # 遮罩模式
    parser.add_argument('--invert-mask', action='store_true', 
                       help='反轉遮罩 (保留細胞核而非細胞質)')
    
    # 其他選項
    parser.add_argument('--high-quality', action='store_true', 
                       help='使用原圖進行處理 (較慢但品質更好)')
    
    args = parser.parse_args()
    
    # 建立處理器
    processor = DishMaskProcessor()
    
    # 載入影像
    if args.input:
        success = processor.load_image(args.input)
    else:
        success = processor.load_dish_from_directory()
    
    if not success:
        print("錯誤: 無法載入影像")
        return 1
    
    # 設定參數
    params = MaskingParams(
        h_min=args.h_min, h_max=args.h_max,
        s_min=args.s_min, s_max=args.s_max,
        v_min=args.v_min, v_max=args.v_max,
        kernel_size=args.kernel_size,
        open_iter=args.open_iter,
        close_iter=args.close_iter,
        dilate_iter=args.dilate_iter,
        dist_threshold=args.dist_threshold,
        alpha=args.alpha,
        invert_mask=args.invert_mask
    )
    
    print(f"處理影像: {processor.image_filename}")
    
    # 執行處理
    try:
        result = processor.process_mask(params, use_original=args.high_quality)
        
        print(f"處理完成 ({result.processing_time:.3f}s)")
        print(f"• 遮罩覆蓋率: {result.mask_area_percent:.1f}%")
        print(f"• 細胞區域: {result.cell_area_percent:.1f}%")
        print(f"• 估算細胞數: {result.cell_count}")
        
        # 儲存結果
        if processor.save_results(result, args.output):
            print(f"結果已儲存至: {args.output}")
            return 0
        else:
            print("錯誤: 無法儲存結果")
            return 1
            
    except Exception as e:
        print(f"處理失敗: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())