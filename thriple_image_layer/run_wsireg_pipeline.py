"""
WSIReg 完整配準流程執行腳本

執行四模組流程：
0. CZI 預處理（降採樣轉 TIFF）
1. 影像配準 (WSIReg)
2. ROI 品質評估
3. 縮圖生成
"""
from pathlib import Path
import sys
import argparse
import pyvips

from config import RegistrationConfig, ElastixParams, ModalityConfig
from module1_preprocess import preprocess_czi_files
from module2_wsireg_alignment import WSIRegAligner
from module3_wsireg_evaluation import ROIEvaluator
from module4_wsireg_thumbnail import ThumbnailGenerator, ThumbnailConfig


def parse_args() -> argparse.Namespace:
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description="WSIReg 全玻片影像配準流程"
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/sec312/tsgh/picture/czi/40X"),
        help="CZI 影像輸入目錄"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/sec312/tsgh/thriple_image_layer/output"),
        help="配準結果輸出目錄"
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
        default=4,
        help="縮圖金字塔層級"
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


def run_preprocess(
    input_dir: Path,
    output_dir: Path,
    downsampling: int
) -> Path:
    """執行預處理模組
    
    Returns:
        預處理輸出目錄
    """
    print("\n" + "=" * 60)
    print("[Module 0] CZI 預處理...")
    print("=" * 60)
    
    preprocess_dir = output_dir / "preprocessed"
    
    # CZI 檔案列表
    czi_files = ["HER2_40X.czi", "DISH_40X_2.czi", "HE_40X.czi"]
    
    try:
        preprocess_czi_files(
            input_dir=input_dir,
            output_dir=preprocess_dir,
            filenames=czi_files,
            downsampling=downsampling
        )
        print("✓ Module 0 完成")
        return preprocess_dir
    except Exception as e:
        print(f"✗ Module 0 失敗: {e}")
        raise


def create_tiff_config(
    preprocess_dir: Path,
    output_dir: Path,
    downsampling: int,
    elastix_params: ElastixParams
) -> RegistrationConfig:
    """創建使用預處理 TIFF 的配置"""
    
    # 計算輸出解析度 (原始 40X = 0.25 µm/px)
    input_res = 0.25 * downsampling  # 降採樣後的解析度
    
    config = RegistrationConfig(
        project_name="thriple_registration",
        input_dir=preprocess_dir,
        output_dir=output_dir,
        elastix_params=elastix_params,
        modalities=[
            ModalityConfig(
                name="HER2",
                filename=f"HER2_40X_ds{downsampling}.tiff",
                resolution=input_res,
                output_resolution=None,  # 保持輸入解析度
                downsampling=1,  # 已經降採樣過，不需要再降
                channel_names=["HER2"],
                channel_colors=["red"]
            ),
            ModalityConfig(
                name="DISH",
                filename=f"DISH_40X_2_ds{downsampling}.tiff",
                resolution=input_res,
                output_resolution=None,
                downsampling=1,
                channel_names=["DISH"],
                channel_colors=["blue"]
            ),
            ModalityConfig(
                name="HE",
                filename=f"HE_40X_ds{downsampling}.tiff",
                resolution=input_res,
                output_resolution=None,
                downsampling=1,
                channel_names=["HE"],
                channel_colors=["green"]
            ),
        ]
    )
    
    return config


def run_alignment(config: RegistrationConfig) -> None:
    """執行配準模組"""
    print("\n" + "=" * 60)
    print("[Module 1] 執行影像配準...")
    print("=" * 60)
    
    try:
        aligner = WSIRegAligner(config)
        aligner.run()
        print("✓ Module 1 完成")
    except Exception as e:
        print(f"✗ Module 1 失敗: {e}")
        raise


def run_evaluation(output_dir: Path) -> None:
    """執行評估模組"""
    print("\n" + "=" * 60)
    print("[Module 2] 評估 ROI 品質...")
    print("=" * 60)
    
    try:
        evaluator = ROIEvaluator(output_dir)
        evaluator.run()
        print("✓ Module 2 完成")
    except Exception as e:
        print(f"✗ Module 2 失敗: {e}")
        raise


def run_thumbnail(output_dir: Path, level: int) -> None:
    """執行縮圖生成模組"""
    print("\n" + "=" * 60)
    print("[Module 3] 產生全局縮圖...")
    print("=" * 60)
    
    try:
        config = ThumbnailConfig(level=level)
        generator = ThumbnailGenerator(output_dir, config)
        generator.run()
        print("✓ Module 3 完成")
    except Exception as e:
        print(f"✗ Module 3 失敗: {e}")
        raise


def main() -> int:
    """主執行函數
    
    Returns:
        退出碼 (0=成功, 1=失敗)
    """
    # 禁用 pyvips 快取以節省記憶體
    pyvips.cache_set_max(0)
    
    args = parse_args()
    
    print("=" * 60)
    print("WSIReg 配準流程 (記憶體優化版)")
    print("=" * 60)
    print(f"\n輸入目錄:      {args.input_dir}")
    print(f"輸出目錄:      {args.output_dir}")
    print(f"預處理降採樣:  {args.preprocess_downsampling}x")
    print(f"網格間距:      {args.grid_spacing}")
    print(f"正則化權重:    {args.bending_weight}")
    
    # 建立 elastix 配置
    elastix_params = ElastixParams(
        grid_spacing=args.grid_spacing,
        bending_energy_weight=args.bending_weight,
    )
    
    try:
        # Module 0: 預處理
        if not args.skip_preprocess:
            preprocess_dir = run_preprocess(
                args.input_dir,
                args.output_dir,
                args.preprocess_downsampling
            )
        else:
            preprocess_dir = args.output_dir / "preprocessed"
            print("\n[跳過] Module 0 預處理")
        
        # 創建使用預處理 TIFF 的配置
        config = create_tiff_config(
            preprocess_dir,
            args.output_dir,
            args.preprocess_downsampling,
            elastix_params
        )
        
        # Module 1: 配準
        if not args.skip_alignment:
            run_alignment(config)
        else:
            print("\n[跳過] Module 1 配準")
        
        # Module 2: 評估
        if not args.skip_evaluation:
            run_evaluation(args.output_dir)
        else:
            print("\n[跳過] Module 2 評估")
        
        # Module 3: 縮圖
        if not args.skip_thumbnail:
            run_thumbnail(args.output_dir, args.thumbnail_level)
        else:
            print("\n[跳過] Module 3 縮圖")
        
        print("\n" + "=" * 60)
        print("完整流程執行完畢")
        print(f"結果儲存於: {args.output_dir}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n流程執行失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
