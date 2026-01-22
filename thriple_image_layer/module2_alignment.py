"""Module 2: VALIS Alignment Pipeline"""
from pathlib import Path
from valis import registration

try:
    from .config import RegistrationConfig, create_default_config
except ImportError:
    from config import RegistrationConfig, create_default_config


def align_images(
    config: RegistrationConfig,
) -> registration.Valis:
    """執行影像對準並儲存變換參數
    
    Args:
        config: 配準流程配置
    
    Returns:
        registration.Valis: VALIS 對準器物件
    """
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 建立影像列表
    img_list = [str(output_dir / m.filename) for m in config.modalities]
    
    # 設定參考影像
    reference_img_f = None
    if config.valis.reference_img_f:
        reference_img_f = str(output_dir / config.valis.reference_img_f)
    
    # 初始化 VALIS 配準器
    registrar = registration.Valis(
        src_dir=str(output_dir),
        dst_dir=str(output_dir),
        name="Transform_Params",
        reference_img_f=reference_img_f,
        align_to_reference=config.valis.align_to_reference,
        max_processed_image_dim_px=config.valis.max_processed_image_dim_px,
        max_non_rigid_registration_dim_px=config.valis.max_non_rigid_registration_dim_px,
        img_list=img_list,
    )
    
    # 執行配準
    print("開始執行配準...")
    registrar.register()
    print("✓ 配準完成")

    return registrar


if __name__ == "__main__":
    config = create_default_config()
    registrar = align_images(config)
    print(f"✓ 結果儲存於: {config.output_dir}")
