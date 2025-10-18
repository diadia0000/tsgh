"""完整流程執行腳本"""
from pathlib import Path
import sys

# 導入各模組
from module2_alignment import align_images
from module3_roi_evaluation import evaluate_roi
from module4_thumbnail import generate_thumbnail

def main():
    """執行完整的三模組流程"""
    czi_dir = Path(r"E:\Class\tsgh\picture\whole_size\40X")
    output_dir = Path(r"E:\Class\tsgh\thriple_image_layer\output")
    
    print("="*60)
    print("開始執行完整流程")
    print("="*60)
    
    # Module 1: Alignment (valis 內建前處理)
    print("\n[Module 1] 執行影像對準...")
    try:
        registrar = align_images(czi_dir, output_dir)
        print("✓ Module 1 完成")
    except Exception as e:
        print(f"✗ Module 1 失敗: {e}")
        sys.exit(1)
    
    # Module 2: ROI Evaluation
    print("\n[Module 2] 評估 ROI 品質...")
    try:
        evaluate_roi(output_dir)
        print("✓ Module 2 完成")
    except Exception as e:
        print(f"✗ Module 2 失敗: {e}")
        sys.exit(1)
    
    # Module 3: Thumbnail
    print("\n[Module 3] 產生全局縮圖...")
    try:
        generate_thumbnail(output_dir, level=4)
        print("✓ Module 3 完成")
    except Exception as e:
        print(f"✗ Module 3 失敗: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("完整流程執行完畢")
    print(f"結果儲存於: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
