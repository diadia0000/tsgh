#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HE細胞膜分離功能測試腳本
Test script for HE membrane separation functionality
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2 as cv

# 導入核心處理邏輯
from he_mask_core import HEMaskProcessor, HEMaskingParams

def test_he_membrane_processing():
    """測試HE細胞膜處理功能"""
    print("開始測試HE細胞膜分離功能...")
    
    # 建立處理器
    processor = HEMaskProcessor()
    
    # 嘗試自動載入HE圖像
    print("嘗試載入HE圖像...")
    success = processor.load_he_from_directory("../../picture/tiff")
    
    if not success:
        print("無法載入HE圖像，請檢查圖像路徑")
        return False
    
    # 建立測試參數
    params = HEMaskingParams()
    print(f"使用測試參數: {params.to_dict()}")
    
    try:
        # 執行處理
        print("開始處理細胞膜分離...")
        result = processor.process_mask(params, use_original=False)
        
        # 顯示結果統計
        print(f"\n處理結果:")
        print(f"• 總像素數: {result.total_pixels:,}")
        print(f"• 細胞膜像素數: {result.membrane_pixels:,}")
        print(f"• 細胞膜覆蓋率: {result.membrane_area_percent:.2f}%")
        print(f"• 處理時間: {result.processing_time:.2f}秒")
        
        # 保存結果圖像
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # 保存疊加圖
        cv.imwrite(str(output_dir / "membrane_overlay.jpg"), result.overlay_membrane)
        print(f"疊加圖已保存: {output_dir / 'membrane_overlay.jpg'}")
        
        # 保存提取圖
        cv.imwrite(str(output_dir / "membrane_extract.png"), result.extract_membrane)
        print(f"提取圖已保存: {output_dir / 'membrane_extract.png'}")
        
        # 保存遮罩
        cv.imwrite(str(output_dir / "membrane_mask.jpg"), result.mask_membrane_clean)
        print(f"遮罩已保存: {output_dir / 'membrane_mask.jpg'}")
        
        print("\n✅ HE細胞膜分離功能測試成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_he_membrane_processing()