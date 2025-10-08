"""
Phase 0 完整執行腳本 - 灰階轉換
"""
from pathlib import Path
from preprocess_dish_image import preprocess_dish_image
from preprocess_her2_image import preprocess_her2_image


def run_phase0():
    """執行完整的 Phase 0 前處理流程"""
    print("=" * 60)
    print("Phase 0: 影像前處理 - 灰階轉換")
    print("=" * 60)
    
    dish_input = Path("../picture/WSI/DISH_20X_ED7.tiff")
    her2_input = Path("../picture/WSI/HER2_20X_ED7.tiff")
    dish_output = Path("output/DISH_Gray.tiff")
    her2_output = Path("output/Her2_Gray.tiff")
    
    # Phase 0.1: DISH 前處理
    preprocess_dish_image(dish_input, dish_output)
    print()
    
    # Phase 0.2: Her2 前處理
    preprocess_her2_image(her2_input, her2_output)
    print()
    
    print("=" * 60)
    print("Phase 0 完成！")
    print(f"輸出檔案:")
    print(f"  - {dish_output}")
    print(f"  - {her2_output}")
    print("=" * 60)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    run_phase0()
