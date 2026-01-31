"""Module 2: VALIS Alignment Pipeline"""
from pathlib import Path
from valis import registration, feature_detectors, feature_matcher,non_rigid_registrars
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
    
    # 自訂特徵檢測器 - 增加特徵點數量
    detector = feature_detectors.DiskFD(num_features=10000)  # 預設 7500
    matcher = feature_matcher.LightGlueMatcher(feature_detector=detector)
    
    # 初始化 VALIS 配準器
    # 注意：matcher 內部已經包含 detector，不需要額外傳入 feature_detector_cls
    # VALIS 會自動判斷是否需要使用 NonRigidTileRegistrar 進行分塊處理（基於記憶體需求）
    registrar = registration.Valis(
        src_dir=str(output_dir),
        dst_dir=str(output_dir),
        name="Transform_Params",
        reference_img_f=reference_img_f,
        align_to_reference=config.valis.align_to_reference,
        max_processed_image_dim_px=config.valis.max_processed_image_dim_px,
        max_non_rigid_registration_dim_px=config.valis.max_non_rigid_registration_dim_px,
        img_list=img_list,
        compose_non_rigid=True,
        max_image_dim_px=2048,
        non_rigid_registrar_cls=non_rigid_registrars.SimpleElastixWarper,  # B-spline 配準
        non_rigid_reg_params={
            # 傳遞給 SimpleElastixWarper 的參數
            # B-spline 天然無翻折問題，更適合醫學影像配準
            "ammi_weight": 0.5,  # AdvancedMattesMutualInformation 互信息權重
            "bending_penalty_weight": 0.3,  # 變形平滑度懲罰（減少過度變形）
            "kp_weight": 0.2,  # 控制點權重（無控制點時會自動忽略）
        },
        matcher=matcher,
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
