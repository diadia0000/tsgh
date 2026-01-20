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
    
    # Module 2: Alignment (使用 VALIS 原生演算法)
    print("\n[Module 2] 執行影像對準...")
    print(f"  演算法: DISK + LightGlueMatcher (VALIS 原生)")
    print(f"  特徵檢測解析度: {config.valis.max_processed_image_dim_px}px")
    print(f"  非剛性配準解析度: {config.valis.max_non_rigid_registration_dim_px}px")
    try:
        registrar = align_images(config)
        print("✓ Module 2 完成")
    except Exception as e:
        print(f"✗ Module 2 失敗: {e}")
        sys.exit(1)
    
    # Module 3: ROI Evaluation
    print("\n[Module 3] 評估 ROI 品質...")
    print(f"  ROI 尺寸: {config.roi.roi_size}")
    try:
        evaluate_roi(config)
        print("✓ Module 3 完成")
    except Exception as e:
        print(f"✗ Module 3 失敗: {e}")
        sys.exit(1)
    
    # Module 4: Thumbnail
    print("\n[Module 4] 產生全局縮圖...")
    print(f"  金字塔層級: {config.thumbnail.level}")
    print(f"  使用非剛性變換: {config.thumbnail.use_non_rigid}")
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
