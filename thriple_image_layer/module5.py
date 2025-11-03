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

    # --- 使用黑色填充處理邊界 tile ---
    print(f"\n--- 開始處理 Tiles（超出範圍使用黑色填充）---")

    tile_count = 0
    import numpy as np

    # 遍歷對齊後的座標空間
    for y in range(0, height_lv_n, tile_wh):
        for x in range(0, width_lv_n, tile_wh):
            # 計算當前 Tile 的實際寬高
            w = min(tile_wh, width_lv_n - x)
            h = min(tile_wh, height_lv_n - y)

            if w <= 0 or h <= 0:
                continue

            tile_count += 1
            print(f"處理 Tile #{tile_count}: ({x}, {y}, {w}x{h})...")

            try:
                # === 處理 DISH ===
                # 計算實際可讀取的區域
                dish_read_x = min(x, dish_dims[0] - 1) if x < dish_dims[0] else dish_dims[0] - 1
                dish_read_y = min(y, dish_dims[1] - 1) if y < dish_dims[1] else dish_dims[1] - 1
                dish_read_w = min(w, dish_dims[0] - dish_read_x) if dish_read_x < dish_dims[0] else 0
                dish_read_h = min(h, dish_dims[1] - dish_read_y) if dish_read_y < dish_dims[1] else 0
                
                # 如果完全超出範圍，創建黑色影像
                if dish_read_w <= 0 or dish_read_h <= 0 or x >= dish_dims[0] or y >= dish_dims[1]:
                    # 先讀取一小塊來判斷通道數
                    sample = dish_obj.slide2image(level=level, xywh=(0, 0, 1, 1))
                    if sample.ndim == 3:
                        dish_tile_img = np.zeros((h, w, sample.shape[2]), dtype=np.uint8)
                    else:
                        dish_tile_img = np.zeros((h, w), dtype=np.uint8)
                else:
                    # 讀取有效區域
                    dish_partial = dish_obj.slide2image(level=level, xywh=(dish_read_x, dish_read_y, dish_read_w, dish_read_h))
                    # 根據讀取的影像創建對應通道數的黑色影像
                    if dish_partial.ndim == 3:
                        dish_tile_img = np.zeros((h, w, dish_partial.shape[2]), dtype=np.uint8)
                    else:
                        dish_tile_img = np.zeros((h, w), dtype=np.uint8)
                    # 將讀取的部分放入正確位置
                    offset_x = dish_read_x - x
                    offset_y = dish_read_y - y
                    dish_tile_img[offset_y:offset_y+dish_read_h, offset_x:offset_x+dish_read_w] = dish_partial
                
                # === 處理 HER2 ===
                her2_read_x = min(x, her2_dims[0] - 1) if x < her2_dims[0] else her2_dims[0] - 1
                her2_read_y = min(y, her2_dims[1] - 1) if y < her2_dims[1] else her2_dims[1] - 1
                her2_read_w = min(w, her2_dims[0] - her2_read_x) if her2_read_x < her2_dims[0] else 0
                her2_read_h = min(h, her2_dims[1] - her2_read_y) if her2_read_y < her2_dims[1] else 0
                
                if her2_read_w <= 0 or her2_read_h <= 0 or x >= her2_dims[0] or y >= her2_dims[1]:
                    sample = her2_obj.slide2image(level=level, xywh=(0, 0, 1, 1))
                    if sample.ndim == 3:
                        her2_tile_img = np.zeros((h, w, sample.shape[2]), dtype=np.uint8)
                    else:
                        her2_tile_img = np.zeros((h, w), dtype=np.uint8)
                else:
                    her2_partial = her2_obj.slide2image(level=level, xywh=(her2_read_x, her2_read_y, her2_read_w, her2_read_h))
                    if her2_partial.ndim == 3:
                        her2_tile_img = np.zeros((h, w, her2_partial.shape[2]), dtype=np.uint8)
                    else:
                        her2_tile_img = np.zeros((h, w), dtype=np.uint8)
                    offset_x = her2_read_x - x
                    offset_y = her2_read_y - y
                    her2_tile_img[offset_y:offset_y+her2_read_h, offset_x:offset_x+her2_read_w] = her2_partial

                # 對切割後的區域進行對齊變換
                dish_tile = dish_obj.warp_img(
                    img=dish_tile_img,
                    non_rigid=non_rigid,
                )
                her2_tile = her2_obj.warp_img(
                    img=her2_tile_img,
                    non_rigid=non_rigid,
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
    print(f"總共生成 {tile_count} 個 Tile（超出範圍的區域已用黑色填充）")
    print(f"儲存於: {tiles_output_dir}")


if __name__ == "__main__":
    # --- 請修改此處的路徑 ---
    # 假設您的註冊結果儲存於 E:\Class\tsgh\thriple_image_layer\output
    output_dir = Path(r"E:\Class\tsgh\thriple_image_layer\output")

    # --- 驗證層級建議 ---
    # Level 2 適用於高解析度驗證。若記憶體不足，請嘗試 Level 3 或 Level 4。
    validation_level = 5

    try:
        generate_aligned_tiles(output_dir, level=validation_level, non_rigid=True)
    finally:
        # 清理 JVM
        try:
            slide_io.kill_jvm()
        except:
            pass
