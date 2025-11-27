from pathlib import Path
from valis import registration, slide_io
import numpy as np
import torch
import torch.nn.functional as F
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

TILE_WIDTH = 2056
TILE_HEIGHT = 2464


def extract_and_warp_batch_gpu(
    tiles_batch: list,
    bk_dxdy_gpu: torch.Tensor,
    coords_batch: list,
    level_scale: float,
    device: torch.device
) -> list:
    """
    批次處理多個 tiles，提高 GPU 利用率。
    """
    results = []
    
    with torch.no_grad():
        for tile, (x, y, w, h) in zip(tiles_batch, coords_batch):
            # 提取局部變形場
            field_h, field_w = bk_dxdy_gpu.shape[2:]
            field_x = int(x * level_scale)
            field_y = int(y * level_scale)
            
            x_start = max(0, field_x - 2)
            y_start = max(0, field_y - 2)
            x_end = min(field_w, field_x + int(w * level_scale) + 6)
            y_end = min(field_h, field_y + int(h * level_scale) + 6)
            
            local_field = bk_dxdy_gpu[:, :, y_start:y_end, x_start:x_end]
            local_field_resized = F.interpolate(
                local_field, size=(h, w), mode='bilinear', align_corners=False
            )
            
            # 轉換 tile 到 GPU (複製避免只讀警告)
            tile_copy = np.array(tile, copy=True)
            if tile_copy.ndim == 2:
                tile_tensor = torch.from_numpy(tile_copy).unsqueeze(0).unsqueeze(0).float().to(device)
            else:
                tile_tensor = torch.from_numpy(tile_copy).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            # 建立變形網格
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, h, device=device),
                torch.linspace(-1, 1, w, device=device),
                indexing='ij'
            )
            
            norm_dx = local_field_resized[0, 0] / (w / 2)
            norm_dy = local_field_resized[0, 1] / (h / 2)
            
            warped_grid = torch.stack([
                grid_x + norm_dx,
                grid_y + norm_dy
            ], dim=-1).unsqueeze(0)
            
            # GPU 變形
            warped = F.grid_sample(
                tile_tensor, warped_grid,
                mode='bilinear', padding_mode='zeros', align_corners=False
            )
            
            # 轉回 CPU
            if tile.ndim == 2:
                result = warped[0, 0].cpu().numpy()
            else:
                result = warped[0].permute(1, 2, 0).cpu().numpy()
            
            results.append(result.astype(tile.dtype))
    
    return results


def generate_aligned_tiles(
    output_dir: Path,
    level: int = 2,
    non_rigid: bool = True,
    tile_width: int = TILE_WIDTH,
    tile_height: int = TILE_HEIGHT,
    use_gpu: bool = True,
    batch_size: int = 4,  # 批次大小
    num_workers: int = 2   # I/O 執行緒數
) -> None:
    """
    使用 GPU 批次處理和多執行緒 I/O 優化。
    """
    try:
        slide_io.init_jvm()
    except Exception as e:
        print(f"VALIS/JVM 初始化失敗: {e}")
        return

    pickle_path = output_dir / "Transform_Params" / "data" / "Transform_Params_registrar.pickle"
    if not pickle_path.exists():
        print(f"錯誤: 找不到 Registrar 檔案: {pickle_path}")
        slide_io.kill_jvm()
        return

    registrar = registration.load_registrar(str(pickle_path))
    tiles_output_dir = Path(f"G:\\output\\level0_tile\\aligned_tiles_lv{level}_{tile_width}x{tile_height}")
    tiles_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tile 輸出目錄: {tiles_output_dir}")
    print(f"Tile 尺寸: {tile_width}x{tile_height} 像素, Level: {level}")
    print(f"批次大小: {batch_size}, I/O 執行緒: {num_workers}")

    try:
        dish_obj = registrar.slide_dict['DISH_40X_2']
        her2_obj = registrar.slide_dict['HER2_40X']
    except KeyError:
        print("錯誤: 找不到投影片物件")
        slide_io.kill_jvm()
        return

    her2_dims = her2_obj.slide_dimensions_wh[level]
    dish_dims = dish_obj.slide_dimensions_wh[level]

    # Use the minimum dimensions to ensure all tiles are valid for both images
    width_lv_n = min(her2_dims[0], dish_dims[0])
    height_lv_n = min(her2_dims[1], dish_dims[1])

    print(f"Level {level} 參考影像尺寸:")
    print(f"  HER2: {her2_dims[0]} x {her2_dims[1]} 像素")
    print(f"  DISH: {dish_dims[0]} x {dish_dims[1]} 像素")
    print(f"  使用: {width_lv_n} x {height_lv_n} 像素 (取較小值)")

    # 設置 GPU
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 載入變形場到 GPU
    bk_dxdy_gpu = None
    if non_rigid and hasattr(dish_obj, 'bk_dxdy') and dish_obj.bk_dxdy is not None:
        print("\n載入 B-spline 變形場到 GPU...")
        bk_dxdy_np = dish_obj.bk_dxdy
        
        if isinstance(bk_dxdy_np, (list, tuple)):
            bk_dxdy_np = np.stack(bk_dxdy_np, axis=0)
        elif bk_dxdy_np.ndim == 3 and bk_dxdy_np.shape[2] == 2:
            bk_dxdy_np = bk_dxdy_np.transpose(2, 0, 1)
        
        print(f"原始變形場形狀: {bk_dxdy_np.shape}")
        
        if bk_dxdy_np.ndim == 3:
            bk_dxdy_gpu = torch.from_numpy(bk_dxdy_np).unsqueeze(0).float().to(device)
        else:
            bk_dxdy_gpu = torch.from_numpy(bk_dxdy_np).float().to(device)
        
        print(f"GPU 變形場尺寸: {bk_dxdy_gpu.shape}")
        level_scale = bk_dxdy_gpu.shape[2] / height_lv_n
    else:
        print("\n未找到非剛性變形場")
        level_scale = 1.0

    # 生成所有 tile 座標 (修正邊界檢查)
    tile_coords = []
    for y in range(0, height_lv_n, tile_height):
        for x in range(0, width_lv_n, tile_width):
            # 確保不超出邊界
            if x >= width_lv_n or y >= height_lv_n:
                continue
            w = min(tile_width, width_lv_n - x)
            h = min(tile_height, height_lv_n - y)
            if w > 0 and h > 0:
                tile_coords.append((x, y, w, h))
    
    total_tiles = len(tile_coords)
    print(f"\n總共需要處理 {total_tiles} 個 tiles")
    
    if level == 0:
        est_hours = total_tiles * 7 / 3600
        print(f"\n警告: Level 0 的 I/O 非常慢！")
        print(f"  預估時間: ~{est_hours:.1f} 小時")
        print(f"  強烈建議使用 level=2 (加速 14x) 或 level=3 (加速 70x)\n")
    
    # Helper function for parallel loading
    def load_tile_pair(coord_idx):
        """Load both HER2 and DISH tiles for a given coordinate index."""
        idx, (x, y, w, h) = coord_idx

        # Validate dimensions before attempting to load
        if w <= 0 or h <= 0:
            return idx, None, None, f"Invalid tile dimensions: w={w}, h={h}"

        try:
            her2_tile = her2_obj.slide2image(level=level, xywh=(x, y, w, h))
            if hasattr(her2_tile, 'numpy'):
                her2_tile = her2_tile.numpy()
            
            # Additional validation: check actual loaded tile size
            if her2_tile.shape[0] == 0 or her2_tile.shape[1] == 0:
                return idx, None, None, f"Loaded tile has zero dimension: {her2_tile.shape}"

            dish_tile = dish_obj.slide2image(level=level, xywh=(x, y, w, h))
            if hasattr(dish_tile, 'numpy'):
                dish_tile = dish_tile.numpy()
            
            # Validate dish tile as well
            if dish_tile.shape[0] == 0 or dish_tile.shape[1] == 0:
                return idx, None, None, f"Loaded tile has zero dimension: {dish_tile.shape}"

            return idx, her2_tile, dish_tile, None
        except Exception as e:
            return idx, None, None, str(e)
    
    # Producer-Consumer: Prefetch queue
    prefetch_queue = Queue(maxsize=2)
    stop_prefetch = threading.Event()
    
    def prefetch_worker():
        """Background thread to prefetch next batch while GPU processes current."""
        batch_idx = 0
        batches_queued = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            while not stop_prefetch.is_set() and batch_idx * batch_size < total_tiles:
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_tiles)
                batch_coords = tile_coords[batch_start:batch_end]
                
                # Parallel load within batch
                io_start = time.time()
                indexed_coords = list(enumerate(batch_coords))
                futures = executor.map(load_tile_pair, indexed_coords)
                
                # Collect results in order
                results = list(futures)
                results.sort(key=lambda x: x[0])  # Ensure order preservation
                
                her2_tiles = []
                dish_tiles = []
                valid_coords = []
                for idx, her2, dish, error in results:
                    if error:
                        print(f"Warning: Skipping tile {batch_coords[idx]}: {error}")
                        continue
                    else:
                        her2_tiles.append(her2)
                        dish_tiles.append(dish)
                        valid_coords.append(batch_coords[idx])

                io_time = time.time() - io_start

                # Always increment batch_idx
                batch_idx += 1

                # Put batch in queue only if there are valid tiles
                if valid_coords:
                    try:
                        prefetch_queue.put((valid_coords, her2_tiles, dish_tiles, io_time), timeout=30)
                        batches_queued += 1
                    except:
                        print("Warning: Prefetch queue timeout")
                        break

        print(f"Prefetch worker finished: {batches_queued} batches queued")
        prefetch_queue.put(None)  # Sentinel to signal completion
    
    # Start prefetch thread
    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()
    
    # Main processing loop
    tile_count = 0
    start_time = time.time()
    batch_num = 0
    last_progress_time = time.time()

    while True:
        # Get prefetched batch with timeout
        try:
            batch_data = prefetch_queue.get(timeout=300)  # 5 minutes timeout
        except:
            print("\nError: Queue timeout - no data received for 5 minutes")
            print(f"Processed {tile_count}/{total_tiles} tiles before timeout")
            break

        if batch_data is None:  # Sentinel received
            break
        
        batch_coords, her2_tiles, dish_tiles, io_time = batch_data
        batch_num += 1
        
        # GPU 批次處理
        gpu_start = time.time()
        if bk_dxdy_gpu is not None:
            warped_tiles = extract_and_warp_batch_gpu(
                dish_tiles, bk_dxdy_gpu, batch_coords, level_scale, device
            )
        else:
            warped_tiles = dish_tiles
        gpu_time = time.time() - gpu_start
        
        # 儲存
        save_start = time.time()
        from PIL import Image
        for i, (x, y, w, h) in enumerate(batch_coords):
            tile_count += 1
            merged = (warped_tiles[i].astype(np.float32) * 0.5 + 
                     her2_tiles[i].astype(np.float32) * 0.5).astype(np.uint8)
            
            tile_filename = f"Merged_Tile_lv{level}_x{x}_y{y}_w{w}_h{h}.tiff"
            output_path = tiles_output_dir / tile_filename
            # 使用無壓縮加速儲存 (70s -> 5s)
            Image.fromarray(merged).save(output_path, compression=None)
        
        save_time = time.time() - save_start
        
        # 進度報告
        elapsed = time.time() - start_time
        tiles_per_sec = tile_count / elapsed
        eta = (total_tiles - tile_count) / tiles_per_sec if tiles_per_sec > 0 else 0
        
        print(f"Batch {batch_num}/{(total_tiles + batch_size - 1)//batch_size}: "
              f"Tiles {tile_count}/{total_tiles} | "
              f"I/O: {io_time:.2f}s | GPU: {gpu_time:.2f}s | Save: {save_time:.2f}s | "
              f"Speed: {tiles_per_sec:.1f} tiles/s | ETA: {eta:.0f}s")
        
        # 更新最後進度時間
        last_progress_time = time.time()

        # 清理
        torch.cuda.empty_cache()
    
    # Stop prefetch thread
    stop_prefetch.set()
    prefetch_thread.join(timeout=5)

    total_time = time.time() - start_time
    print(f"\n完成！總時間: {total_time:.1f}s, 平均速度: {total_tiles/total_time:.2f} tiles/s")
    print(f"儲存於: {tiles_output_dir}")
    
    # 清理 GPU
    if bk_dxdy_gpu is not None:
        del bk_dxdy_gpu
        torch.cuda.empty_cache()


if __name__ == "__main__":
    output_dir = Path(r"H:\tsgh\thriple_image_layer\output")
    
    try:
        # 關鍵修正: 使用 level=2 或 3 以減少 I/O 瓶頸
        # Level 0: 283637x228733 像素 -> 每個 tile 讀取 ~7秒
        # Level 2: ~70909x57183 像素 -> 每個 tile 讀取 ~0.5秒 (14x 加速)
        # Level 3: ~35454x28591 像素 -> 每個 tile 讀取 ~0.1秒 (70x 加速)
        
        generate_aligned_tiles(
            output_dir, 
            level=0,
            non_rigid=True, 
            use_gpu=True,
            batch_size=64,  # 增加 batch size
            num_workers=7  # 增加並行度
        )
    finally:
        try:
            slide_io.kill_jvm()
        except:
            pass
