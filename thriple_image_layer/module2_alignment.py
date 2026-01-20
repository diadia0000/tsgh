"""Module 2: VALIS Alignment Pipeline

使用 VALIS 原生演算法 (DISK + LightGlueMatcher) 進行高精度配準
"""
from pathlib import Path
import pickle
import os

import torch
from valis import registration

try:
    from .config import RegistrationConfig, create_default_config
except ImportError:
    from config import RegistrationConfig, create_default_config


def align_images(
    config: RegistrationConfig,
) -> registration.Valis:
    """
    Module 2: 執行影像對準並儲存變換參數
    
    使用 VALIS 原生演算法 (DISK + LightGlueMatcher) 進行高精度配準
    
    Args:
        config: 配準流程配置
    
    Returns:
        registration.Valis: VALIS 對準器物件
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠️  使用 CPU，速度可能較慢")
    else:
        print(f"✓ 使用裝置：{device}")
    
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 從配置獲取 VALIS 參數
    valis_config = config.valis
    
    print("初始化 VALIS 配準器...")
    print(f"  - 特徵檢測解析度: {valis_config.max_processed_image_dim_px}px")
    print(f"  - 非剛性配準解析度: {valis_config.max_non_rigid_registration_dim_px}px")
    print(f"  - 參考影像: {valis_config.reference_img_f}")
    print(f"  - 演算法: DISK + LightGlueMatcher (VALIS 原生)")
    
    registrar = registration.Valis(
        src_dir=str(output_dir),
        dst_dir=str(output_dir),
        name="Transform_Params",
        reference_img_f=valis_config.reference_img_f,
        align_to_reference=valis_config.align_to_reference,
        max_processed_image_dim_px=valis_config.max_processed_image_dim_px,
        max_non_rigid_registration_dim_px=valis_config.max_non_rigid_registration_dim_px
    )
    
    # 執行對準
    print("\n開始執行對準...")
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()
    
    print("✓ 對準完成，變換參數已儲存")

    # 明確地將 registrar 物件保存為 pickle 檔案
    pickle_path = config.pickle_path
    pickle_path.parent.mkdir(parents=True, exist_ok=True)

    # 根據 AI 檔案生成規則：如果檔案已存在則先刪除
    if pickle_path.exists():
        print(f"正在刪除舊的 pickle 檔案: {pickle_path}")
        os.remove(pickle_path)

    print(f"正在保存 registrar 物件到: {pickle_path}")
    with open(pickle_path, 'wb') as f:
        pickle.dump(registrar, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("✓ Registrar 物件已成功保存為 pickle 檔案")

    return registrar


if __name__ == "__main__":
    config = create_default_config()
    
    print("=" * 60)
    print("Module 2: VALIS Alignment")
    print("=" * 60)
    print(f"輸入目錄: {config.output_dir}")
    print(f"參考影像: {config.valis.reference_img_f}")
    print()
    
    registrar = align_images(config)
    print(f"\n✓ 對準完成，結果儲存於: {config.output_dir}")
