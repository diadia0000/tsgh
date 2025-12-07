from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import time
import gc
import pyvips
# 匯入 valis 模組
from valis import registration, slide_io

TILE_WIDTH = 4096
TILE_HEIGHT = 4096


def get_transformation_matrix(registrar, slide_name: str, level: int) -> np.ndarray:
    """從 registrar 物件中提取指定 slide 的剛性變換矩陣。"""
    slide_obj = registrar.slide_dict[slide_name]

    # 仿射變換矩陣 (通常是 3x3)
    M = slide_obj.M.copy()

    # 考慮 level 縮放
    level_scale = slide_obj.slide_dimensions_wh[0][0] / slide_obj.slide_dimensions_wh[level][0]

    # 調整變換矩陣以適應 level
    # 縮放矩陣
    S = np.array([
        [1 / level_scale, 0, 0],
        [0, 1 / level_scale, 0],
        [0, 0, 1]
    ])
    # 反向縮放矩陣
    S_inv = np.array([
        [level_scale, 0, 0],
        [0, level_scale, 0],
        [0, 0, 1]
    ])

    # 新的變換矩陣 M' = S * M * S_inv
    M_scaled = S @ M @ S_inv

    return M_scaled


def extract_and_warp_batch_gpu(
    tiles_batch: list,
    rigid_matrix_gpu: torch.Tensor,
    bk_dxdy_gpu: torch.Tensor,
    coords_batch: list,
    level_scale_non_rigid: float,
    device: torch.device
) -> tuple[list, list]:
    """
    批次處理：按尺寸分組，最大化 GPU 利用率。
    按照 VALIS 底層算法順序：先剛性變換，再非剛性變換。

    VALIS 順序 (warp_tools.py line 1021-1232):
    1. affine_warped = img.affine(M, ...)  # 剛性變換
    2. warped = affine_warped.mapim(dxdy, ...)  # 非剛性變換
    """
    with torch.no_grad():
        # 按尺寸分組
        size_groups = {}
        for i, (tile, coord) in enumerate(zip(tiles_batch, coords_batch)):
            size_key = tile.shape[:2]
            if size_key not in size_groups:
                size_groups[size_key] = []
            size_groups[size_key].append((i, tile, coord))
        
        results = [None] * len(tiles_batch)
        
        for size_key, group in size_groups.items():
            indices, tiles, coords = zip(*group)
            
            batch_tensors = []
            for tile in tiles:
                tile_copy = np.array(tile, copy=True)
                if tile_copy.ndim == 2:
                    t = torch.from_numpy(tile_copy).unsqueeze(0).unsqueeze(0)
                else:
                    t = torch.from_numpy(tile_copy).permute(2, 0, 1).unsqueeze(0)
                batch_tensors.append(t)
            
            tiles_tensor = torch.cat(batch_tensors, dim=0).float().to(device, non_blocking=True)
            B, C, H, W = tiles_tensor.shape

            # ===== 步驟 1: 剛性變換 (Affine/Rigid Transformation) =====
            affine_grid = F.affine_grid(rigid_matrix_gpu[:, :2], size=(B, C, H, W), align_corners=False)
            affine_warped = F.grid_sample(
                tiles_tensor, affine_grid,
                mode='bilinear', padding_mode='zeros', align_corners=False
            )

            # ===== 步驟 2: 非剛性變換 (Non-rigid Deformation) =====
            if bk_dxdy_gpu is not None:
                # 提取非剛性變形場（相對於已經剛性變換後的圖像）
                batch_fields = []
                for (x, y, w, h) in coords:
                    # 變形場的座標需要相對於變換後的空間
                    field_x = int(x * level_scale_non_rigid)
                    field_y = int(y * level_scale_non_rigid)
                    field_w_scaled = int(w * level_scale_non_rigid)
                    field_h_scaled = int(h * level_scale_non_rigid)

                    x_start = max(0, min(field_x, bk_dxdy_gpu.shape[3] - 1))
                    y_start = max(0, min(field_y, bk_dxdy_gpu.shape[2] - 1))
                    x_end = min(bk_dxdy_gpu.shape[3], field_x + max(1, field_w_scaled))
                    y_end = min(bk_dxdy_gpu.shape[2], field_y + max(1, field_h_scaled))

                    local_field = bk_dxdy_gpu[:, :, y_start:y_end, x_start:x_end]
                    local_field_resized = F.interpolate(
                        local_field, size=(H, W), mode='bilinear', align_corners=False
                    )
                    batch_fields.append(local_field_resized)

                fields_tensor = torch.cat(batch_fields, dim=0) * (1.0 / level_scale_non_rigid)

                # 創建像素網格索引 (等同於 pyvips 的 xyz)
                y_coords = torch.arange(H, device=device).view(H, 1).expand(H, W)
                x_coords = torch.arange(W, device=device).view(1, W).expand(H, W)

                # 應用位移場 (等同於 pyvips 的 mapim)
                displaced_x = x_coords + fields_tensor[:, 0]
                displaced_y = y_coords + fields_tensor[:, 1]

                # 正規化到 [-1, 1] 範圍 (grid_sample 的要求)
                norm_x = 2.0 * displaced_x / (W - 1) - 1.0
                norm_y = 2.0 * displaced_y / (H - 1) - 1.0
                non_rigid_grid = torch.stack([norm_x, norm_y], dim=-1)

                # 對剛性變換後的圖像應用非剛性變換
                warped_batch = F.grid_sample(
                    affine_warped, non_rigid_grid,
                    mode='bilinear', padding_mode='zeros', align_corners=False
                )
            else:
                # 如果沒有非剛性變換，直接使用剛性變換的結果
                warped_batch = affine_warped

            warped_np = warped_batch.cpu().numpy()
            
            for i, (idx, tile) in enumerate(zip(indices, tiles)):
                if tile.ndim == 2:
                    result = warped_np[i, 0]
                else:
                    result = warped_np[i].transpose(1, 2, 0)
                results[idx] = result.astype(tile.dtype)
        
        valid_flags = [True] * len(results)
        return results, valid_flags


def generate_aligned_tiles(
    input_params_dir: Path,
    output_dir: Path,
    level: int = 0,
    non_rigid: bool = True,
    tile_width: int = TILE_WIDTH,
    tile_height: int = TILE_HEIGHT,
    use_gpu: bool = True,
    batch_size: int = 4,
    num_workers: int = 2
) -> None:
    """
    使用 pyvips 讀取，GPU 批次處理，並用 pyvips 寫入 BigTIFF。
    """
    if level != 0:
        print("警告: 此腳本已為 Level 0 進行優化。其他 level 可能無法正確工作。")

    pickle_path = input_params_dir / "Transform_Params" / "data" / "Transform_Params_registrar.pickle"
    if not pickle_path.exists():
        print(f"錯誤: 找不到 Registrar 檔案: {pickle_path}")
        return

    # 使用 valis 提供的方法初始化 JVM
    print("初始化 JVM...")
    try:
        slide_io.init_jvm()
        print("[OK] JVM 初始化成功")
    except:
        print("[WARNING] JVM 可能已經啟動，繼續執行...")
        pass

    try:
        # 載入 registrar
        print("正在載入 Registrar...")
        registrar = registration.load_registrar(str(pickle_path))
        print("[OK] Registrar 載入成功")

        final_output_path = output_dir / f"Merged_Aligned_lv{level}.tiff"
        # 根據使用者指示，如果檔案已存在則刪除
        if final_output_path.exists():
            print(f"正在刪除已存在的輸出檔案: {final_output_path}")
            final_output_path.unlink()

        print(f"最終輸出將儲存於: {final_output_path}")
        print(f"Tile 尺寸: {tile_width}x{tile_height} 像素, Level: {level}")
        print(f"批次大小: {batch_size}")

        try:
            dish_obj = registrar.slide_dict['DISH_40X_2']
            her2_obj = registrar.slide_dict['HER2_40X']
        except KeyError:
            print("錯誤: 找不到 'DISH_40X_2' 或 'HER2_40X' 投影片物件")
            return

        # 使用 valis 的 slide reader 來讀取 .czi 檔案
        print(f"開啟投影片: {her2_obj.src_f}")
        her2_reader = her2_obj.slide
        print(f"開啟投影片: {dish_obj.src_f}")
        dish_reader = dish_obj.slide

        # 獲取 Level 0 的尺寸
        width_lv_n, height_lv_n = her2_obj.slide_dimensions_wh[level]

        print(f"Level {level} 參考影像尺寸 (HER2): {width_lv_n} x {height_lv_n} 像素")

        # 設置 GPU
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # 提取剛性變換矩陣
        print("\n提取剛性變換矩陣...")
        rigid_matrix = get_transformation_matrix(registrar, 'DISH_40X_2', level)
        rigid_matrix_gpu = torch.from_numpy(rigid_matrix[:2]).unsqueeze(0).float().to(device)
        print(f"DISH->HER2 的 Level {level} 剛性變換矩陣:\n{rigid_matrix[:2]}")

        # 載入非剛性變形場到 GPU
        bk_dxdy_gpu = None
        level_scale_non_rigid = 1.0
        if non_rigid and hasattr(dish_obj, 'bk_dxdy') and dish_obj.bk_dxdy is not None:
            print("\n載入 B-spline 變形場到 GPU...")
            bk_dxdy_np = dish_obj.bk_dxdy

            if isinstance(bk_dxdy_np, (list, tuple)):
                bk_dxdy_np = np.stack(bk_dxdy_np, axis=0)
            elif bk_dxdy_np.ndim == 3 and bk_dxdy_np.shape[2] == 2:
                bk_dxdy_np = bk_dxdy_np.transpose(2, 0, 1)

            if bk_dxdy_np.ndim == 3:
                bk_dxdy_gpu = torch.from_numpy(bk_dxdy_np).unsqueeze(0).float().to(device)
            else:
                bk_dxdy_gpu = torch.from_numpy(bk_dxdy_np).float().to(device)

            # 變形場是相對於哪個 level 計算的？通常是較低的 level。
            # 我們需要知道這個 level 的尺寸來計算縮放比例。
            # 使用 DISH 的 processed_img_shape 作為參考
            reg_img_shape_rc = dish_obj.processed_img_shape_rc
            reg_img_h, reg_img_w = reg_img_shape_rc
            level_scale_non_rigid = reg_img_h / height_lv_n
            print(f"GPU 變形場尺寸: {bk_dxdy_gpu.shape}")
            print(f"非剛性變形場縮放比例: {level_scale_non_rigid:.4f}")
        else:
            print("\n未找到或不使用非剛性變形場")

        # 生成 tile 座標
        tile_coords = []
        for y in range(0, height_lv_n, tile_height):
            for x in range(0, width_lv_n, tile_width):
                w = min(tile_width, width_lv_n - x)
                h = min(tile_height, height_lv_n - y)
                if w > 0 and h > 0:
                    tile_coords.append((x, y, w, h))

        total_tiles = len(tile_coords)
        print(f"\n總共需要處理 {total_tiles} 個 tiles")

        # 建立一個空的 pyvips 影像來存放合併後的結果
        # 使用 float32 以避免在混合時精度損失
        output_image = pyvips.Image.black(width_lv_n, height_lv_n, bands=3)

        # 主處理循環
        start_time = time.time()
        for i in range(0, total_tiles, batch_size):
            batch_start_time = time.time()
            batch_coords = tile_coords[i:i+batch_size]

            # 讀取批次
            io_start = time.time()
            her2_tiles = []
            dish_tiles = []
            for x, y, w, h in batch_coords:
                # 使用 valis slide reader 的 read_region 方法
                # read_region(location, level, size) - location 是 (x, y)，size 是 (w, h)
                her2_region = np.array(her2_reader.read_region((x, y), level, (w, h)))
                dish_region = np.array(dish_reader.read_region((x, y), level, (w, h)))

                # 移除 alpha 通道（如果有的話）
                if her2_region.shape[2] == 4:
                    her2_region = her2_region[:, :, :3]
                if dish_region.shape[2] == 4:
                    dish_region = dish_region[:, :, :3]

                her2_tiles.append(her2_region)
                dish_tiles.append(dish_region)
            io_time = time.time() - io_start

            # GPU 批次處理
            gpu_start = time.time()
            warped_tiles, _ = extract_and_warp_batch_gpu(
                dish_tiles, rigid_matrix_gpu, bk_dxdy_gpu, batch_coords, level_scale_non_rigid, device
            )
            gpu_time = time.time() - gpu_start

            # 合併與插入
            insert_start = time.time()
            for j, (x, y, w, h) in enumerate(batch_coords):
                # 將 warped_tile 和 her2_tile 轉為 pyvips 影像
                warped_vips = pyvips.Image.new_from_memory(warped_tiles[j].astype(np.float32), w, h, 3, 'float')
                her2_vips = pyvips.Image.new_from_memory(her2_tiles[j].astype(np.float32), w, h, 3, 'float')

                # 混合
                merged_vips = (warped_vips * 0.5 + her2_vips * 0.5)

                # 插入到輸出影像
                output_image = output_image.insert(merged_vips, x, y)
            insert_time = time.time() - insert_start

            # 清理記憶體
            del her2_tiles, dish_tiles, warped_tiles
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # 進度報告
            processed_tiles = i + len(batch_coords)
            elapsed = time.time() - start_time
            tiles_per_sec = processed_tiles / elapsed
            eta = (total_tiles - processed_tiles) / tiles_per_sec if tiles_per_sec > 0 else 0

            print(f"批次 {i//batch_size + 1}/{(total_tiles + batch_size - 1)//batch_size}: "
                  f"Tiles {processed_tiles}/{total_tiles} | "
                  f"總耗時: {time.time() - batch_start_time:.2f}s (I/O: {io_time:.2f}s, GPU: {gpu_time:.2f}s, Insert: {insert_time:.2f}s) | "
                  f"速度: {tiles_per_sec:.2f} tiles/s | ETA: {eta:.0f}s")

        # 儲存最終影像
        print("\n所有 tile 處理完畢，正在儲存最終的大圖檔...")
        save_start = time.time()
        output_image.cast('uchar').write_to_file(
            str(final_output_path),
            tile=True,
            pyramid=True,
            compression='jpeg',
            Q=90,
            bigtiff=True
        )
        save_time = time.time() - save_start
        print(f"影像儲存完畢，耗時: {save_time:.2f}s")

        total_time = time.time() - start_time
        print(f"\n完成！總時間: {total_time:.1f}s, 平均速度: {total_tiles/total_time:.2f} tiles/s")
        print(f"儲存於: {final_output_path}")

        # 清理 GPU 記憶體
        if device.type == 'cuda':
            del rigid_matrix_gpu
            if bk_dxdy_gpu is not None:
                del bk_dxdy_gpu
            torch.cuda.empty_cache()
            print("GPU 記憶體已清理")

    finally:
        print("關閉 JVM...")
        try:
            slide_io.kill_jvm()
        except:
            pass



if __name__ == "__main__":
    # 確保 pyvips 的快取設定不會過度使用記憶體
    pyvips.cache_set_max(100)  # 最多快取 100 個操作
    pyvips.cache_set_max_mem(1024 * 1024 * 1024) # 1 GB vips 快取記憶體

    # 定義輸入參數目錄和輸出目錄
    input_params_dir = Path(r"H:\tsgh\thriple_image_layer\output")
    output_dir = Path(r"G:\output\level0")

    # 確保輸出目錄存在
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"所有輸出將儲存於: {output_dir}")

    generate_aligned_tiles(
        input_params_dir,
        output_dir,
        level=0,
        non_rigid=True,
        use_gpu=True,
        batch_size=16,   # Level 0 tile 很大，需要較小的 batch size
        num_workers=6
    )
