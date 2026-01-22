"""完整流程執行腳本

執行完整的 VALIS 配準管線：
- Module 2: VALIS Alignment
- Module 3: ROI Quality Evaluation
- Module 4: Thumbnail Generation
"""
from pathlib import Path
import sys

# 導入各模組
from module2_alignment import align_images
from module3_roi_evaluation import evaluate_roi
from module4_thumbnail import generate_thumbnail
from config import create_default_config


def main() -> None:
    """執行完整的配準流程"""
    config = create_default_config()
    
    print("=" * 60)
    print("VALIS 配準管線")
    print("=" * 60)
    print(f"專案名稱: {config.project_name}")
    print(f"輸入目錄: {config.input_dir}")
    print(f"輸出目錄: {config.output_dir}")
    print(f"參考模態: {config.reference_modality}")
    print("=" * 60)
    
    # Module 2: Alignment
    print("\n[Module 2] 執行影像對準...")
    try:
        registrar = align_images(config)
        print("✓ Module 2 完成")
    except Exception as e:
        print(f"✗ Module 2 失敗: {e}")
        sys.exit(1)
    
    # Module 3: ROI Evaluation
    print("\n[Module 3] 評估 ROI 品質...")
    try:
        evaluate_roi(config)
        print("✓ Module 3 完成")
    except Exception as e:
        print(f"✗ Module 3 失敗: {e}")
        sys.exit(1)
    
    # Module 4: Thumbnail
    print("\n[Module 4] 產生全局縮圖...")
    try:
        generate_thumbnail(config)
        print("✓ Module 4 完成")
    except Exception as e:
        print(f"✗ Module 4 失敗: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("完整流程執行完畢")
    print(f"結果儲存於: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
