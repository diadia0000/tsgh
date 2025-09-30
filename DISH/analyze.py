from aicspylibczi import CziFile
import numpy as np
import tifffile
import os
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# --- 參數設定 ---
# 輸入 CZI 檔案路徑
FILE_PATH = "E:/Class/tsgh/picture/whole_size/DISH_20X_ED7.czi"
# 輸出資料夾路徑
OUTPUT_DIR = "picture/tiles"
# 檔案名前綴
FILE_PREFIX = os.path.splitext(os.path.basename(FILE_PATH))[0]


def export_tile_worker(tile_data, file_path, output_dir, file_prefix):
    """
    這是在單一 CPU 核心上運行的工作函式。
    它接收簡單的數據類型（整數、元組），而不是複雜的物件。
    """
    m_index, bbox_coords = tile_data
    try:
        # 每個工作核心都獨立開啟一次 CZI 檔案
        czi = CziFile(file_path)

        # 直接使用傳入的座標元組
        region = bbox_coords

        # 讀取單一圖塊的影像數據
        tile_image_data = czi.read_mosaic(region, C=0)

        if tile_image_data is None or tile_image_data.size == 0:
            return (m_index, "empty")

        tile_image_data = np.squeeze(tile_image_data)

        output_filename = f"{file_prefix}_tile_{m_index:05d}.tiff"
        output_path = os.path.join(output_dir, output_filename)

        tifffile.imwrite(output_path, tile_image_data)

        return (m_index, "success")

    except Exception as e:
        return (m_index, f"error: {e}")


def main():
    """主程式：設置平行處理池並分派任務"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"開始處理檔案: {os.path.basename(FILE_PATH)}")
    start_time = time.time()

    czi = CziFile(FILE_PATH)
    if not czi.is_mosaic():
        print("錯誤：此 CZI 檔案不是馬賽克格式。")
        return

    # --- 關鍵修改在此 ---
    # 我們建立一個只包含簡單數據類型的任務列表
    tasks = []
    all_tiles = czi.get_all_mosaic_tile_bounding_boxes()
    for tile_info, bbox in all_tiles.items():
        # 從 bbox 物件中提取簡單的座標元組
        bbox_coords = (bbox.x, bbox.y, bbox.w, bbox.h)
        # 從 tile_info 物件中提取簡單的索引號
        m_index = tile_info.m_index
        # 將簡單數據打包成一個任務
        tasks.append((m_index, bbox_coords))

    total_tiles = len(tasks)
    del czi

    if total_tiles == 0:
        print("警告：找不到任何圖塊。")
        return

    print(f"找到 {total_tiles} 個 tile，將使用 {cpu_count()} 個 CPU 核心進行平行匯出...")

    worker_func = partial(export_tile_worker,
                          file_path=FILE_PATH,
                          output_dir=OUTPUT_DIR,
                          file_prefix=FILE_PREFIX)

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(worker_func, tasks), total=total_tiles))

    end_time = time.time()

    success_count = sum(1 for r in results if r[1] == "success")
    empty_count = sum(1 for r in results if r[1] == "empty")
    error_count = total_tiles - success_count - empty_count

    print("\n--- 處理完成 ---")
    print(f"成功匯出: {success_count} 張")
    print(f"空白跳過: {empty_count} 張")
    print(f"發生錯誤: {error_count} 張")
    print(f"所有 tile 已匯出至資料夾: {OUTPUT_DIR}")
    print(f"總耗時: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main()