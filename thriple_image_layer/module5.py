from pathlib import Path
from valis import registration, slide_io, warp_tools
import pyvips

# 2048x2048 像素
TILE_WH = 2048


def generate_aligned_tiles(
        output_dir: Path,
        level: int = 2,  # 預設使用 Level 2 (比 Level 0/1 記憶體消耗少得多)
        non_rigid: bool = True,
        tile_wh: int = TILE_WH
) -> None:
    """
    根據 VALIS 註冊結果，將對齊後的 DISH 和 HER2 影像切割成
    指定大小 (預設 2048x2048) 的 Tile，並儲存到輸出目錄。

    此方法會先將 Level N 的整張 Slide 對齊並載入 pyvips 影像物件，
    請注意 Level 0 或 Level 1 解析度極高，可能導致記憶體不足 (OOM)。
    建議使用 Level 2 (約 400x) 進行驗證。

    Args:
        output_dir: 輸出目錄。會在此目錄下建立一個 'aligned_tiles_lvX' 子目錄。
        level: 金字塔層級 (0=最高解析度，數字越大解析度越低)。
        non_rigid: 是否使用非剛性變換。
        tile_wh: 輸出 Tile 的邊長（像素）。
    """
    try:
        # VALIS 需要 JVM 初始化
        slide_io.init_jvm()
    except Exception as e:
        print(f"VALIS/JVM 初始化失敗: {e}")
        return

    pickle_path = output_dir / "Transform_Params" / "data" / "Transform_Params_registrar.pickle"
    if not pickle_path.exists():
        print(f"錯誤: 找不到 Registrar 檔案: {pickle_path}")
        slide_io.kill_jvm()
        return

    # 載入註冊結果
    registrar = registration.load_registrar(str(pickle_path))

    # 設置輸出目錄
    tiles_output_dir = output_dir / f"aligned_tiles_lv{level}_wh{tile_wh}"
    tiles_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tile 輸出目錄: {tiles_output_dir}")
    print(f"Tile 尺寸: {tile_wh}x{tile_wh} 像素, Level: {level}")
    print(f"非剛性變換: {'啟用' if non_rigid else '停用'}")

    try:
        # 取得需要對齊的 Slide 物件
        dish_obj = registrar.slide_dict['DISH_40X_2']
        her2_obj = registrar.slide_dict['HER2_40X']
    except KeyError:
        print("錯誤: 找不到 DISH_40X_2 或 HER2_40X 投影片物件。請檢查註冊時使用的名稱是否正確。")
        slide_io.kill_jvm()
        return

    # 取得指定 level 的原始影像尺寸
    # slide_dimensions_wh 格式: [[w0, h0], [w1, h1], ...]
    dish_dims = dish_obj.slide_dimensions_wh[level]
    her2_dims = her2_obj.slide_dimensions_wh[level]

    # 使用較大的尺寸作為對齊後的工作區域
    width_lv_n = max(dish_dims[0], her2_dims[0])
    height_lv_n = max(dish_dims[1], her2_dims[1])

    print(f"Level {level} 原始影像尺寸:")
    print(f"  DISH: {dish_dims[0]} x {dish_dims[1]} 像素")
    print(f"  HER2: {her2_dims[0]} x {her2_dims[1]} 像素")
    print(f"Level {level} 對齊後工作區域 (W x H): {width_lv_n} x {height_lv_n} 像素")

    # --- 使用 slide2image(xywh=...) 直接切割區域 ---
    print(f"\n--- 開始使用 slide2image(xywh=...) 方法處理指定區域 ---")

    tile_count = 0

    # x, y 座標是 Level N 上的對齊後座標空間
    for y in range(0, height_lv_n, tile_wh):
        for x in range(0, width_lv_n, tile_wh):
            # 計算當前 Tile 的實際寬高
            w = min(tile_wh, width_lv_n - x)
            h = min(tile_wh, height_lv_n - y)

            if w <= 0 or h <= 0:
                continue

            tile_count += 1
            print(f"處理 Tile #{tile_count}: ({x}, {y}, {w}x{h})...")

            # 使用 slide2image(xywh=...) 直接切割區域
            # xywh 參數格式: (top_left_x, top_left_y, width, height)
            try:
                # 直接從原始影像切割指定區域
                dish_tile_img = dish_obj.slide2image(level=level, xywh=(x, y, w, h))
                her2_tile_img = her2_obj.slide2image(level=level, xywh=(x, y, w, h))

                # 對切割後的區域進行對齊變換
                dish_tile = dish_obj.warp_img(
                    img=dish_tile_img,
                    non_rigid=non_rigid
                )
                her2_tile = her2_obj.warp_img(
                    img=her2_tile_img,
                    non_rigid=non_rigid
                )

                # 轉換為 pyvips.Image
                if not isinstance(dish_tile, pyvips.Image):
                    dish_tile = warp_tools.numpy2vips(dish_tile)
                if not isinstance(her2_tile, pyvips.Image):
                    her2_tile = warp_tools.numpy2vips(her2_tile)

            except Exception as e:
                print(f"錯誤: 處理區域 ({x}, {y}, {w}x{h}) 失敗. {e}")
                continue

            # 合併 Tiles (簡單平均)
            merged_tile = (dish_tile * 0.5 + her2_tile * 0.5)
            merged_tile = merged_tile.cast('uchar')

            # 儲存 (使用 deflate 壓縮)
            tile_filename = f"Merged_Tile_lv{level}_x{x}_y{y}_w{w}_h{h}.tiff"
            output_path = tiles_output_dir / tile_filename

            merged_tile.write_to_file(
                str(output_path),
                compression='deflate'
            )

    print(f"\n--- Tile 生成完成 ---")
    print(f"總共生成 {tile_count} 個 Tile，儲存於: {tiles_output_dir}")


if __name__ == "__main__":
    # --- 請修改此處的路徑 ---
    # 假設您的註冊結果儲存於 E:\Class\tsgh\thriple_image_layer\output
    output_dir = Path(r"E:\Class\tsgh\thriple_image_layer\output")

    # --- 驗證層級建議 ---
    # Level 2 適用於高解析度驗證。若記憶體不足，請嘗試 Level 3 或 Level 4。
    validation_level = 2

    try:
        generate_aligned_tiles(output_dir, level=validation_level, non_rigid=True)
    finally:
        # 清理 JVM
        try:
            slide_io.kill_jvm()
        except:
            pass
