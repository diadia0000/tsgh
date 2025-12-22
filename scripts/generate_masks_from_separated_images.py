#!/usr/bin/env python3
"""
完整流程：使用 HER2 和 DISH 分離圖像生成高品質 Pseudo Masks

步驟：
1. 切割三組對齊的 tiles (HER2, DISH, Merged)
2. 使用 HER2 + DISH tiles 生成精確的 pseudo masks
3. 視覺化檢查結果
"""
from pathlib import Path
import sys

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    print("=" * 80)
    print("📊 HER2 + DISH Pseudo Mask 生成流程")
    print("=" * 80)
    print()
    
    # ===== 設定路徑 =====
    base_output_dir = Path("/home/sec312/tsgh/thriple_image_layer/output")
    
    # 對齊後的原始圖像
    her2_tiff = base_output_dir / "temp" / "her2_warped_lv1.ome.tiff"
    dish_tiff = base_output_dir / "temp" / "dish_warped_lv1.ome.tiff"
    merged_tiff = base_output_dir / "Merged_Aligned_lv1.tiff"
    
    # 輸出目錄
    tiles_output_dir = base_output_dir / "tiles_lv1"
    masks_output_dir = Path("/home/sec312/tsgh/unet_mask/process/pseudo_masks_v3")
    
    # ===== 驗證檔案存在 =====
    print("🔍 檢查原始檔案...")
    for name, path in [("HER2", her2_tiff), ("DISH", dish_tiff), ("Merged", merged_tiff)]:
        if path.exists():
            print(f"   ✅ {name}: {path.name}")
        else:
            print(f"   ❌ {name} 不存在: {path}")
            return
    print()
    
    # ===== 步驟 1：切割 Tiles =====
    print("=" * 80)
    print("步驟 1/2: 切割三組對齊的 Tiles")
    print("=" * 80)
    
    # 檢查是否已有 tiles
    her2_tiles_dir = tiles_output_dir / "her2"
    if her2_tiles_dir.exists() and len(list(her2_tiles_dir.glob("*.tiff"))) > 0:
        print(f"⚠️  發現現有的 tiles 在 {tiles_output_dir}")
        choice = input("是否跳過切割步驟？(y/n, 預設 y): ").strip().lower()
        if choice != 'n':
            print("✅ 跳過切割步驟")
        else:
            print("🔧 開始切割 tiles...")
            from thriple_image_layer.module5_tile_generator import generate_triple_tiles
            generate_triple_tiles(
                her2_tiff=her2_tiff,
                dish_tiff=dish_tiff,
                merged_tiff=merged_tiff,
                output_base_dir=tiles_output_dir,
                tile_width=512,
                tile_height=512,
                workers=8
            )
    else:
        print("🔧 開始切割 tiles...")
        from thriple_image_layer.module5_tile_generator import generate_triple_tiles
        generate_triple_tiles(
            her2_tiff=her2_tiff,
            dish_tiff=dish_tiff,
            merged_tiff=merged_tiff,
            output_base_dir=tiles_output_dir,
            tile_width=512,
            tile_height=512,
            workers=8
        )
    
    print()
    
    # ===== 步驟 2：生成 Pseudo Masks =====
    print("=" * 80)
    print("步驟 2/2: 生成 Pseudo Masks")
    print("=" * 80)
    
    from unet_mask.mask_generation import batch_generate_masks_v3
    
    batch_generate_masks_v3(
        her2_dir=tiles_output_dir / "her2",
        dish_dir=tiles_output_dir / "dish",
        output_dir=masks_output_dir,
        visualize_samples=10  # 隨機抽 10 張檢查
    )
    
    print()
    print("=" * 80)
    print("🎉 完成！")
    print("=" * 80)
    print(f"📁 HER2 Tiles: {tiles_output_dir / 'her2'}")
    print(f"📁 DISH Tiles: {tiles_output_dir / 'dish'}")
    print(f"📁 Merged Tiles: {tiles_output_dir / 'merged'}")
    print(f"📁 Pseudo Masks: {masks_output_dir}")
    print("=" * 80)
    print()
    print("下一步：")
    print("1. 將 Merged tiles 作為訓練輸入")
    print("2. 將生成的 Pseudo Masks 作為訓練目標")
    print("3. 開始訓練 UNet 模型")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用戶中斷")
    except Exception as e:
        print(f"\n\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
