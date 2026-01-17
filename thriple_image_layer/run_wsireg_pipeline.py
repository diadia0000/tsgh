"""
WSIReg 完整配準流程執行腳本

執行四模組流程：
1. CZI 預處理（降採樣轉 TIFF） - module1_preprocess.py
2. 影像配準 (WSIReg) - module2_wsireg_alignment.py  
3. ROI 品質評估 - module3_wsireg_evaluation.py
4. 縮圖生成 - module4_wsireg_thumbnail.py

前提條件：Module 1 已執行，HER2_processed.tif 等檔案已存在於 output 目錄
"""
from pathlib import Path
import sys
import argparse
import pyvips

from config import create_default_config, RegistrationConfig, ElastixParams, ModalityConfig
from module1_preprocess import CziPreprocessor
from module2_wsireg_alignment import WSIRegAligner
from module3_wsireg_evaluation import ROIEvaluator
from module4_wsireg_thumbnail import ThumbnailGenerator, ThumbnailConfig


def parse_args() -> argparse.Namespace:
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description="WSIReg 全玻片影像配準流程"
    )
    
    # 使用 config.py 的預設值
    default_config = create_default_config()
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_config.input_dir,
        help="預處理後的 TIFF 輸入目錄 (Module 1 輸出)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_config.output_dir.parent,  # output 目錄 (非 registered)
        help="輸出目錄"
    )
    
    parser.add_argument(
        "--preprocess-downsampling",
        type=int,
        default=8,
        help="預處理降採樣倍率 (減少記憶體使用)"
    )
    
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=20.0,
        help="B-spline 網格間距 (越大越平滑)"
    )
    
    parser.add_argument(
        "--bending-weight",
        type=float,
        default=5.0,
        help="彎曲能量正則化權重"
    )
    
    parser.add_argument(
        "--thumbnail-level",
        type=int,
        default=0,  # 影像已經是 level 1，使用 level 0 保持原解析度
        help="縮圖金字塔層級 (0=原解析度)"
    )
    
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="跳過預處理（使用現有 TIFF）"
    )
    
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="跳過配準步驟（使用現有結果）"
    )
    
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="跳過評估步驟"
    )
    
    parser.add_argument(
        "--skip-thumbnail",
        action="store_true",
        help="跳過縮圖生成"
    )
    
    return parser.parse_args()


def run_preprocess(config: RegistrationConfig) -> None:
    """執行預處理模組 (Module 1)
    
    將 CZI 轉換為 BigTIFF
    """
    print("\n" + "=" * 60)
    print("[Module 1] CZI 預處理...")
    print("=" * 60)
    
    try:
        processor = CziPreprocessor(config)
        processor.run()
        print("✓ Module 1 完成")
    except Exception as e:
        print(f"✗ Module 1 失敗: {e}")
        raise




def run_alignment(config: RegistrationConfig) -> None:
    """執行配準模組 (Module 2)"""
    print("\n" + "=" * 60)
    print("[Module 2] 執行影像配準...")
    print("=" * 60)
    
    try:
        aligner = WSIRegAligner(config)
        aligner.run()
        print("✓ Module 2 完成")
    except Exception as e:
        print(f"✗ Module 2 失敗: {e}")
        raise


def run_evaluation(output_dir: Path) -> None:
    """執行評估模組 (Module 3)"""
    print("\n" + "=" * 60)
    print("[Module 3] 評估 ROI 品質...")
    print("=" * 60)
    
    try:
        evaluator = ROIEvaluator(output_dir)
        evaluator.run()
        print("✓ Module 3 完成")
    except Exception as e:
        print(f"✗ Module 3 失敗: {e}")
        raise


def run_thumbnail(output_dir: Path, level: int) -> None:
    """執行縮圖生成模組 (Module 4)"""
    print("\n" + "=" * 60)
    print("[Module 4] 產生全局縮圖...")
    print("=" * 60)
    
    try:
        thumb_config = ThumbnailConfig(level=level)
        generator = ThumbnailGenerator(output_dir, thumb_config)
        generator.run()
        print("✓ Module 4 完成")
    except Exception as e:
        print(f"✗ Module 4 失敗: {e}")
        raise


def main() -> int:
    """主執行函數
    
    Returns:
        退出碼 (0=成功, 1=失敗)
    """
    # 禁用 pyvips 快取以節省記憶體
    pyvips.cache_set_max(0)
    
    args = parse_args()
    
    # 使用預設配置並根據參數調整
    config = create_default_config()
    config.elastix_params.grid_spacing = args.grid_spacing
    config.elastix_params.bending_energy_weight = args.bending_weight
    
    print("=" * 60)
    print("WSIReg 配準流程")
    print("=" * 60)
    print(f"\n輸入目錄:      {config.input_dir}")
    print(f"配準輸出目錄:  {config.output_dir}")
    print(f"網格間距:      {config.elastix_params.grid_spacing}")
    print(f"正則化權重:    {config.elastix_params.bending_energy_weight}")
    
    try:
        # Module 1: 預處理 (CZI -> BigTIFF)
        if not args.skip_preprocess:
            run_preprocess(config)
        else:
            print("\n[跳過] Module 1 預處理")
        
        # Module 2: 配準
        if not args.skip_alignment:
            run_alignment(config)
        else:
            print("\n[跳過] Module 2 配準")
        
        # Module 3: 評估
        if not args.skip_evaluation:
            run_evaluation(args.output_dir)
        else:
            print("\n[跳過] Module 3 評估")
        
        # Module 4: 縮圖
        if not args.skip_thumbnail:
            run_thumbnail(args.output_dir, args.thumbnail_level)
        else:
            print("\n[跳過] Module 4 縮圖")
        
        print("\n" + "=" * 60)
        print("完整流程執行完畢")
        print(f"結果儲存於: {config.output_dir}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n流程執行失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
