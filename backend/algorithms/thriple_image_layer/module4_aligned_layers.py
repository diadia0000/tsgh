"""Module 4: Aligned Modality Layers

將 VALIS registrar 內的每個模態各自 warp 到參考座標，輸出三張對齊的全解析度
TIFF（HER2 / DISH / HE）。三張皆以 crop="overlap" 裁到相同的重疊區域，因此尺寸
一致、可像素級疊合。

與 module4_thumbnail 的差異：thumbnail 只 warp DISH+HER2 並融合成單一
Merged_Aligned；本模組不融合，而是保留三個模態各自的對齊影像。
"""
from pathlib import Path
from typing import Dict, Optional

from valis import registration, slide_io

try:
    from .config import RegistrationConfig, create_default_config, get_slide_key
except ImportError:
    from config import RegistrationConfig, create_default_config, get_slide_key


def generate_aligned_layers(
    config: RegistrationConfig,
    level: Optional[int] = None,
) -> Dict[str, Path]:
    """將每個模態 warp 到參考座標並輸出對齊 TIFF。

    Args:
        config: 配準流程配置。
        level: 金字塔層級；None 時採用 config.thumbnail.level。

    Returns:
        Dict[str, Path]: {模態名稱: 對齊後 TIFF 路徑}。
    """
    try:
        slide_io.init_jvm()
    except Exception:
        raise RuntimeError("無法啟動 Java 虛擬機，請確認已安裝 Java 並設定環境變數。")

    if level is None:
        level = config.thumbnail.level
    # 若 Module 2 沒做非剛性（method="none"），registrar 無形變場，warp 只能用剛性。
    nr_available = config.valis.non_rigid_method != "none"
    use_non_rigid = config.thumbnail.use_non_rigid and nr_available
    ref_name = config.reference_modality

    print(f"載入變換參數: {config.pickle_path}")
    registrar = registration.load_registrar(str(config.pickle_path))

    ref_slide = registrar.get_ref_slide()
    print(f"使用金字塔 level {level}, 尺寸: {ref_slide.slide_dimensions_wh[level]}")

    outputs: Dict[str, Path] = {}
    for modality in config.modalities:
        key = get_slide_key(modality.filename)
        slide_obj = registrar.slide_dict[key]

        # 參考模態本身不需非剛性形變；其餘模態依 config 決定。
        non_rigid = use_non_rigid and modality.name != ref_name

        # warp_and_save_slide 會將副檔名改為 .ome.tiff。
        out_stub = config.output_dir / f"{modality.name}_aligned_lv{level}.tiff"
        out_ome = out_stub.with_suffix("").with_suffix(".ome.tiff")

        if out_ome.exists():
            print(f"找到現有對齊檔案，跳過: {out_ome.name}")
            outputs[modality.name] = out_ome
            continue

        print(f"對齊並儲存 {modality.name} 影像 (non_rigid={non_rigid})...")
        slide_obj.warp_and_save_slide(
            str(out_stub),
            level=level,
            non_rigid=non_rigid,
            crop="overlap",
            compression="jpeg",
            pyramid=True,
        )
        outputs[modality.name] = out_ome
        print(f"✓ 已儲存: {out_ome}")

    print("\n" + "=" * 60)
    print(f"完成！共產生 {len(outputs)} 張對齊模態圖：")
    for name, path in outputs.items():
        print(f"   - {name}: {path}")
    print("=" * 60)
    return outputs


if __name__ == "__main__":
    config = create_default_config()

    print("=" * 60)
    print("Module 4: Aligned Modality Layers")
    print("=" * 60)
    print(f"金字塔層級: {config.thumbnail.level}")
    print(f"使用非剛性變換: {config.thumbnail.use_non_rigid}")
    print(f"參考模態: {config.reference_modality}")
    print()

    try:
        generate_aligned_layers(config)
    finally:
        try:
            slide_io.kill_jvm()
        except Exception:
            pass
