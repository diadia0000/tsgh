"""Module 2: VALIS Alignment Pipeline"""
from valis import registration, feature_detectors, feature_matcher, non_rigid_registrars

from config import RegistrationConfig, create_default_config


def _patch_valis_pickle():
    """Prevent VALIS pickle crash caused by unpicklable SWIG (pyvips/OpenSlide) objects.

    VALIS calls pickle.dump(self) internally to cache the registrar. SwigPyObject
    instances in slide readers can't be pickled, so we strip them at __getstate__.
    """
    import pickle as _pickle

    def _safe_getstate(self):
        state = {}
        for k, v in self.__dict__.items():
            try:
                _pickle.dumps(v)
                state[k] = v
            except Exception:
                pass
        return state

    registration.Valis.__getstate__ = _safe_getstate


_patch_valis_pickle()


def _build_elastix_params(max_image_dim_px: int):
    """建立較高自由度的 B-spline 非剛性配準參數。"""
    params = non_rigid_registrars.SimpleElastixWarper.get_default_params(
        img_shape=(max_image_dim_px, max_image_dim_px),
        grid_spacing_ratio=0.015,
    )

    # 提高優化強度，讓局部形變更容易被學到。
    params["MaximumNumberOfIterations"] = ["2500"]
    params["NumberOfSpatialSamples"] = ["8000"]
    params["NumberOfResolutions"] = ["5"]

    return params

def align_images(
    config: RegistrationConfig,
) -> registration.Valis:
    """執行影像對準並儲存變換參數
    
    Args:
        config: 配準流程配置
    
    Returns:
        registration.Valis: VALIS 對準器物件
    """
    input_dir = config.input_dir
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 建立影像列表 (從 input_dir 讀取 Module 1 輸出的 TIFF)
    img_list = [str(input_dir / m.filename) for m in config.modalities]

    # 設定參考影像 (從 reference_modality 自動推導)
    reference_img_f = None
    if config.reference_img_f:
        reference_img_f = str(input_dir / config.reference_img_f)
    
    # 自訂特徵檢測器 - 增加特徵點數量
    detector = feature_detectors.DiskFD(num_features=20000)  # 預設 7500
    matcher = feature_matcher.LightGlueMatcher(feature_detector=detector)
    
    max_image_dim_px = 1024
    elastix_params = _build_elastix_params(max_image_dim_px)

    # 初始化 VALIS 配準器
    # 注意：matcher 內部已經包含 detector，不需要額外傳入 feature_detector_cls
    # VALIS 會自動判斷是否需要使用 NonRigidTileRegistrar 進行分塊處理（基於記憶體需求）
    registrar = registration.Valis(
        src_dir=str(input_dir),
        dst_dir=str(output_dir),
        name="Transform_Params",
        reference_img_f=reference_img_f,
        align_to_reference=config.valis.align_to_reference,
        img_list=img_list,
        compose_non_rigid=True,
        max_processed_image_dim_px=1024,
        max_image_dim_px=max_image_dim_px,
        non_rigid_registrar_cls=non_rigid_registrars.SimpleElastixWarper,  # B-spline 配準
        non_rigid_reg_params={
            # 傳遞給 SimpleElastixWarper 的參數
            # B-spline 天然無翻折問題，更適合醫學影像配準
            "params": elastix_params,
            "ammi_weight": 0.8,  # AdvancedMattesMutualInformation 互信息權重
            "bending_penalty_weight": 0.12,  # 降低平滑懲罰以增加非剛性形變
            "kp_weight": 0.08,  # 控制點權重（無控制點時會自動忽略）
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
