from aicspylibczi import CziFile
import numpy as np
import tifffile
import os
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import xml.etree.ElementTree as ET

# --- 參數設定 ---
FILE_PATH = "E:/Class/tsgh/picture/whole_size/DISH_20X_ED7.czi"
OUTPUT_DIR = "picture/tiles/"
FILE_PREFIX = os.path.splitext(os.path.basename(FILE_PATH))[0]


def normalize_to_uint8(image_array):
    """
    正確地將影像陣列正規化到 0-255 範圍並轉換為 uint8
    處理不同的資料類型和範圍 (float32 [0,1], uint16 [0,4095], 等等)
    """
    # 確保是 numpy array 並轉為 float32 進行計算
    img = np.array(image_array, dtype=np.float32)
    
    # 如果已經是空的或全零，直接返回
    if img.size == 0 or (img.min() == img.max() == 0):
        return img.astype(np.uint8)
    
    # 正規化到 0-255 範圍
    img_min = img.min()
    img_max = img.max()
    
    # 避免除以零
    if img_max > img_min:
        img_normalized = 255.0 * (img - img_min) / (img_max - img_min)
    else:
        img_normalized = np.zeros_like(img)
    
    # 確保在有效範圍內並轉換為 uint8
    img_normalized = np.clip(img_normalized, 0, 255)
    return img_normalized.astype(np.uint8)


def detect_channel_types(czi_file):
    """
    從 CZI metadata 中偵測通道類型
    返回字典：{channel_index: channel_type}
    """
    channel_info = {}
    
    try:
        # 獲取 metadata XML
        metadata_xml = czi_file.meta
        
        # 解析 XML 來找通道資訊
        root = ET.fromstring(metadata_xml)
        
        # 尋找通道定義 (可能在不同的 XML 路徑中)
        channel_elements = []
        
        # 常見的通道資訊路徑
        for path in [
            ".//Channel",
            ".//Channels/Channel", 
            ".//Track/Channel",
            ".//Tracks/Track/Channel"
        ]:
            elements = root.findall(path)
            if elements:
                channel_elements.extend(elements)
                break
        
        # 解析通道資訊
        for i, channel in enumerate(channel_elements):
            channel_name = ""
            channel_type = "unknown"
            
            # 嘗試不同的屬性名稱來獲取通道資訊
            for attr in ['Name', 'name', 'Id', 'id']:
                if channel.get(attr):
                    channel_name = channel.get(attr).lower()
                    break
            
            # 根據名稱判斷通道類型
            if any(keyword in channel_name for keyword in ['dapi', 'hoechst', 'blue']):
                channel_type = "dapi"
            elif any(keyword in channel_name for keyword in ['fitc', 'gfp', 'green']):
                channel_type = "fitc"
            elif any(keyword in channel_name for keyword in ['texas', 'red', 'cy3', 'tritc']):
                channel_type = "texas_red"
            elif any(keyword in channel_name for keyword in ['bright', 'trans', 'dic']):
                channel_type = "brightfield"
            
            channel_info[i] = {
                'name': channel_name,
                'type': channel_type
            }
    
    except Exception as e:
        print(f"警告：無法解析通道 metadata: {e}")
    
    # 如果無法從 metadata 獲取資訊，嘗試通過測試小區域來推測
    if not channel_info:
        channel_info = detect_channels_by_sampling(czi_file)
    
    return channel_info


def detect_channels_by_sampling(czi_file):
    """
    通過採樣小區域來推測通道類型
    """
    channel_info = {}
    
    try:
        # 獲取第一個 tile 的小區域進行測試
        tiles = list(czi_file.get_all_mosaic_tile_bounding_boxes().items())
        if not tiles:
            return channel_info
            
        first_tile_info, first_bbox = tiles[0]
        test_region = (first_bbox.x, first_bbox.y, min(100, first_bbox.w), min(100, first_bbox.h))
        
        # 測試每個通道
        num_channels = czi_file.size[czi_file.dims.index('C')] if 'C' in czi_file.dims else 2
        
        for c in range(num_channels):
            try:
                tile = czi_file.read_mosaic(test_region, C=c)
                if tile is not None:
                    tile_data = np.squeeze(tile)
                    
                    # 根據統計特性推測通道類型
                    mean_intensity = tile_data.mean()
                    std_intensity = tile_data.std()
                    
                    # 簡單的啟發式規則（可能需要根據實際資料調整）
                    if mean_intensity > 0.7 * tile_data.max():  # 高亮度，可能是亮場
                        channel_type = "brightfield"
                    elif std_intensity < 0.1 * mean_intensity:  # 低變異，可能是背景
                        channel_type = "background"
                    else:
                        channel_type = f"staining_{c}"  # 染色通道
                    
                    channel_info[c] = {
                        'name': f"channel_{c}",
                        'type': channel_type
                    }
                        
            except Exception as e:
                print(f"警告：無法讀取通道 {c}: {e}")
                
    except Exception as e:
        print(f"警告：通道偵測失敗: {e}")
    
    return channel_info


def export_tile_worker(tile_data, file_path, output_dir, file_prefix, channel_mapping):
    """
    改進的 tile 匯出函數，支援動態通道偵測和正確的資料類型轉換
    """
    m_index, bbox_coords = tile_data
    czi = CziFile(file_path)

    region = bbox_coords
    channels_data = {}
    
    # 獲取實際的通道數量
    actual_channels = czi.size[czi.dims.index('C')] if 'C' in czi.dims else 1
    num_channels = max(len(channel_mapping), actual_channels) if channel_mapping else actual_channels
        
    for c in range(num_channels):
        try:
            tile = czi.read_mosaic(region, C=c)
            if tile is not None and tile.size > 0:
                # 正確處理資料類型轉換
                tile_squeezed = np.squeeze(tile)
                tile_normalized = normalize_to_uint8(tile_squeezed)
                channels_data[c] = tile_normalized
            else:
                # 創建空白通道
                channels_data[c] = np.zeros((bbox_coords[3], bbox_coords[2]), dtype=np.uint8)
        except Exception as e:
            print(f"警告：讀取通道 {c} 失敗: {e}")
            channels_data[c] = np.zeros((bbox_coords[3], bbox_coords[2]), dtype=np.uint8)

    # 根據偵測到的通道類型組合 RGB 影像
    rgb = create_rgb_image(channels_data, channel_mapping)
    
    # 儲存為 TIFF
    output_path = f"{output_dir}/{file_prefix}_tile_{m_index:05d}.tiff"
    tifffile.imwrite(output_path, rgb, photometric='rgb')
    
    return (m_index, "success")


def create_rgb_image(channels_data, channel_mapping):
    """
    根據通道類型創建 RGB 影像
    """
    if not channels_data:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 獲取影像尺寸
    first_channel = next(iter(channels_data.values()))
    height, width = first_channel.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 尋找特定類型的通道
    brightfield_channel = None
    red_channel = None
    green_channel = None
    blue_channel = None
    
    for ch_idx, ch_data in channels_data.items():
        if ch_idx in channel_mapping:
            ch_type = channel_mapping[ch_idx]['type']
            
            if ch_type == "brightfield":
                brightfield_channel = ch_data
            elif ch_type in ["texas_red", "red"]:
                red_channel = ch_data
            elif ch_type in ["fitc", "green"]:
                green_channel = ch_data
            elif ch_type in ["dapi", "blue"]:
                blue_channel = ch_data
    
    # 組合 RGB 影像
    if brightfield_channel is not None:
        # 如果有亮場，將其作為背景
        if red_channel is not None:
            rgb[:, :, 0] = red_channel  # R: 紅色染色
            rgb[:, :, 1] = brightfield_channel  # G: 亮場
            rgb[:, :, 2] = brightfield_channel  # B: 亮場
        else:
            # 沒有紅色染色，使用亮場作為灰階
            rgb[:, :, 0] = brightfield_channel
            rgb[:, :, 1] = brightfield_channel
            rgb[:, :, 2] = brightfield_channel
    else:
        # 沒有亮場，直接分配顏色通道
        if red_channel is not None:
            rgb[:, :, 0] = red_channel
        if green_channel is not None:
            rgb[:, :, 1] = green_channel
        if blue_channel is not None:
            rgb[:, :, 2] = blue_channel
        
        # 如果都沒有，使用第一個可用通道作為灰階
        if red_channel is None and green_channel is None and blue_channel is None:
            first_available = next(iter(channels_data.values()))
            rgb[:, :, 0] = first_available
            rgb[:, :, 1] = first_available
            rgb[:, :, 2] = first_available
    
    return rgb


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"開始處理檔案: {os.path.basename(FILE_PATH)}")
    start_time = time.time()

    czi = CziFile(FILE_PATH)
    if not czi.is_mosaic():
        print("錯誤：此 CZI 檔案不是馬賽克格式。")
        return

    pixel_type = czi.pixel_type
    print(f"偵測到影像像素格式：{pixel_type}")

    # 偵測通道類型
    print("正在分析通道資訊...")
    channel_mapping = detect_channel_types(czi)
    
    print("=== 偵測到的通道資訊 ===")
    for ch_idx, ch_info in channel_mapping.items():
        print(f"通道 {ch_idx}: {ch_info['name']} (類型: {ch_info['type']})")
    
    if not channel_mapping:
        print("警告：無法偵測通道資訊，將使用預設設定")
        # 使用預設的通道對應
        channel_mapping = {
            0: {'name': 'channel_0', 'type': 'brightfield'},
            1: {'name': 'channel_1', 'type': 'texas_red'}
        }

    tasks = []
    for tile_info, bbox in czi.get_all_mosaic_tile_bounding_boxes().items():
        bbox_coords = (bbox.x, bbox.y, bbox.w, bbox.h)
        tasks.append((tile_info.m_index, bbox_coords))

    total_tiles = len(tasks)
    if total_tiles == 0:
        print("警告：找不到任何圖塊。")
        return

    print(f"找到 {total_tiles} 個 tile，將使用 {cpu_count()} 個 CPU 核心進行平行匯出...")

    worker_func = partial(export_tile_worker,
                          file_path=FILE_PATH,
                          output_dir=OUTPUT_DIR,
                          file_prefix=FILE_PREFIX,
                          channel_mapping=channel_mapping)

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