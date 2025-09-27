#!/usr/bin/env python3
"""
CZI文件詳細分析腳本
按照工作流程要求分析picture/內的3張CZI檔案

分析項目:
1. 基本影像屬性 (尺寸、像素類型、解析度、金字塔層數)
2. 色彩通道資訊
3. 座標系統 & Metadata (包含stage position檢查)
4. 像素統計資訊 (直方圖、Mean、Std、Min、Max)

記憶體管理策略: 逐張讀取 → 分析 → 釋放
"""

import os
import gc
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from aicspylibczi import CziFile
import xml.etree.ElementTree as ET


def format_bytes(bytes_count):
    """將位元組轉換為人類可讀的格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.2f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.2f} PB"


def extract_basic_properties(czi_file):
    """
    提取基本影像屬性
    - 圖像尺寸（Width, Height, Z-stack 數量, Channel 數量）
    - Pixel type
    - Resolution / Pixel spacing  
    - 圖像金字塔層數 (改進版)
    """
    basic_props = {}
    
    # 基本尺寸和維度
    dims = czi_file.dims
    size = czi_file.size
    
    basic_props['維度'] = dims
    basic_props['尺寸'] = size
    basic_props['像素類型'] = czi_file.pixel_type
    
    # 解析維度
    dim_mapping = {}
    for i, dim in enumerate(dims):
        if i < len(size):
            dim_mapping[dim] = size[i]
    
    # 提取關鍵尺寸
    width = dim_mapping.get('X', 0)
    height = dim_mapping.get('Y', 0)
    z_stack = dim_mapping.get('Z', 1)
    channels = dim_mapping.get('C', 1)
    
    basic_props['寬度(Width)'] = width
    basic_props['高度(Height)'] = height
    basic_props['Z-stack數量'] = z_stack
    basic_props['通道數量(Channels)'] = channels
    
    # 檢查是否為馬賽克
    basic_props['是否為馬賽克'] = czi_file.is_mosaic()
    if czi_file.is_mosaic():
        basic_props['馬賽克瓦片數'] = dim_mapping.get('M', 0)

    # 改進的金字塔層數分析
    pyramid_info = analyze_pyramid_levels(czi_file)
    basic_props.update(pyramid_info)

    # 計算總像素數和估算大小
    total_pixels = 1
    for s in size:
        total_pixels *= s
    basic_props['總像素數'] = total_pixels
    
    # 估算記憶體大小
    pixel_bytes_map = {
        'bgr24': 3, 'rgb24': 3, 'gray16': 2, 'bgr48': 6, 
        'rgb48': 6, 'gray8': 1, 'gray32': 4, 'bgr96': 12, 'rgb96': 12
    }
    pixel_bytes = pixel_bytes_map.get(czi_file.pixel_type, 1)
    estimated_size = total_pixels * pixel_bytes
    basic_props['每像素位元組數'] = pixel_bytes
    basic_props['估算影像大小'] = format_bytes(estimated_size)
    
    # 記憶體安全警告
    if estimated_size > 50 * 1024 * 1024 * 1024:  # 50GB
        basic_props['⚠️記憶體警告'] = f'估算大小超過50GB，直接載入可能導致記憶體不足'

    return basic_props


def analyze_pyramid_levels(czi_file):
    """
    詳細分析CZI檔案的金字塔層數
    使用多種方法確保準確檢測
    """
    pyramid_info = {
        '金字塔檢測方法': [],
        '金字塔層數': 1,
        '金字塔': '否',
        '各層詳細資訊': {}
    }

    print("    - 正在進行詳細金字塔分析...")

    # 方法1: 檢查S維度 (Scene/Series)
    try:
        dims = czi_file.dims
        size = czi_file.size

        if 'S' in dims:
            s_index = dims.index('S')
            s_count = size[s_index]
            if s_count > 1:
                pyramid_info['金字塔檢測方法'].append('S_dimension')
                pyramid_info['金字塔層數'] = s_count
                pyramid_info['金字塔'] = '是'
                pyramid_info['S維度層數'] = s_count
                print(f"    - ✓ S維度檢測到 {s_count} 層")
            else:
                print("    - S維度：1層")
        else:
            print("    - 無S維度")
    except Exception as e:
        pyramid_info['s_dimension_error'] = str(e)
        print(f"    - S維度檢查失敗: {e}")

    # 方法2: 嘗試讀取不同解析度層級
    try:
        scale_factors = [1.0, 0.5, 0.25, 0.125, 0.0625]
        accessible_scales = []

        for scale in scale_factors:
            try:
                if czi_file.is_mosaic():
                    # 嘗試讀取一小塊來測試該層級是否存在
                    bbox_dict = czi_file.get_all_mosaic_tile_bounding_boxes()
                    if bbox_dict:
                        first_bbox = list(bbox_dict.values())[0]
                        # 測試讀取很小的區域
                        test_region = (first_bbox.x, first_bbox.y, min(100, first_bbox.w), min(100, first_bbox.h))
                        test_data = czi_file.read_mosaic(test_region, scale_factor=scale, C=0)
                        if test_data is not None and test_data.size > 0:
                            accessible_scales.append(scale)
                            del test_data  # 立即清理
                else:
                    # 非馬賽克圖像
                    test_data = czi_file.read_image(scene=0, scale_factor=scale, X=slice(0, 100), Y=slice(0, 100))
                    if test_data is not None and test_data.size > 0:
                        accessible_scales.append(scale)
                        del test_data  # 立即清理
            except:
                break  # 如果某個scale失敗，停止測試更小的scale

        if len(accessible_scales) > 1:
            pyramid_info['金字塔檢測方法'].append('scale_factor_test')
            if pyramid_info['金字塔層數'] == 1:  # 如果前面方法沒檢測到
                pyramid_info['金字塔層數'] = len(accessible_scales)
                pyramid_info['金字塔'] = '是'
            pyramid_info['可訪問縮放層級'] = accessible_scales
            print(f"    - ✓ 縮放測試檢測到 {len(accessible_scales)} 個可訪問層級: {accessible_scales}")
        else:
            print(f"    - 縮放測試：僅1個層級可訪問 {accessible_scales}")

    except Exception as e:
        pyramid_info['scale_test_error'] = str(e)
        print(f"    - 縮放層級測試失敗: {e}")

    # 方法3: 檢查圖塊數量和排列 (針對馬賽克圖像)
    try:
        if czi_file.is_mosaic():
            bbox_dict = czi_file.get_all_mosaic_tile_bounding_boxes()
            tile_count = len(bbox_dict)

            # 分析圖塊尺寸的變化來推斷金字塔層數
            if tile_count > 0:
                tile_sizes = []
                for bbox in list(bbox_dict.values())[:100]:  # 只檢查前100個圖塊
                    tile_sizes.append((bbox.w, bbox.h))

                unique_sizes = list(set(tile_sizes))
                if len(unique_sizes) > 1:
                    pyramid_info['金字塔檢測方法'].append('tile_size_analysis')
                    pyramid_info['圖塊尺寸變化'] = unique_sizes
                    print(f"    - ✓ 圖塊尺寸分析：發現 {len(unique_sizes)} 種不同尺寸")
                else:
                    print(f"    - 圖塊尺寸分析：所有圖塊尺寸相同 {unique_sizes[0] if unique_sizes else 'N/A'}")
    except Exception as e:
        pyramid_info['tile_analysis_error'] = str(e)
        print(f"    - 圖塊分析失敗: {e}")

    # 方法4: XML元數據分析（如果可用）
    try:
        metadata_xml = czi_file.meta
        if metadata_xml and len(metadata_xml) > 100:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(metadata_xml)

            # 搜尋金字塔相關標籤
            pyramid_elements = []
            for elem in root.iter():
                tag_lower = elem.tag.lower()
                if any(keyword in tag_lower for keyword in ['pyramid', 'level', 'resolution', 'scale']):
                    if elem.text and elem.text.strip():
                        pyramid_elements.append(f"{elem.tag}: {elem.text}")

            if pyramid_elements:
                pyramid_info['XML金字塔資訊'] = pyramid_elements[:5]  # 只保留前5個
                pyramid_info['金字塔檢測方法'].append('xml_metadata')
                print(f"    - ✓ XML元數據發現 {len(pyramid_elements)} 個金字塔相關元素")
            else:
                print("    - XML元數據：未發現金字塔相關資訊")
        else:
            print("    - XML元數據不可用或過短")
    except Exception as e:
        pyramid_info['xml_analysis_error'] = str(e)
        print(f"    - XML金字塔分析失敗: {e}")

    # 最終結論
    if pyramid_info['金字塔層數'] == 1 and not pyramid_info['金字塔檢測方法']:
        pyramid_info['金字塔'] = '否 (所有檢測方法均未發現多層結構)'
        pyramid_info['建議'] = '此文件可能僅包含單一解析度影像，直接載入會消耗大量記憶體'
    elif pyramid_info['金字塔層數'] > 1:
        pyramid_info['建議'] = f'建議使用適當的scale_factor載入，避免直接載入最高解析度'

    print(f"    - 金字塔分析完成：{pyramid_info['金字塔層數']}層，使用方法: {pyramid_info['金字塔檢測方法']}")

    return pyramid_info


def extract_channel_info(czi_file):
    """
    提取色彩通道資訊
    """
    channel_info = {}
    
    dims = czi_file.dims
    size = czi_file.size
    
    # 獲取通道數量
    if 'C' in dims:
        channel_count = size[dims.index('C')]
        channel_info['通道總數'] = channel_count
    else:
        channel_info['通道總數'] = 1
    
    # 像素類型分析
    pixel_type = czi_file.pixel_type
    channel_info['像素類型'] = pixel_type
    
    if 'bgr' in pixel_type.lower():
        channel_info['色彩格式'] = 'BGR'
        if '24' in pixel_type:
            channel_info['每通道位元數'] = 8
        elif '48' in pixel_type:
            channel_info['每通道位元數'] = 16
        elif '96' in pixel_type:
            channel_info['每通道位元數'] = 32
    elif 'rgb' in pixel_type.lower():
        channel_info['色彩格式'] = 'RGB'
        if '24' in pixel_type:
            channel_info['每通道位元數'] = 8
        elif '48' in pixel_type:
            channel_info['每通道位元數'] = 16
        elif '96' in pixel_type:
            channel_info['每通道位元數'] = 32
    elif 'gray' in pixel_type.lower():
        channel_info['色彩格式'] = 'Grayscale'
        if '8' in pixel_type:
            channel_info['每通道位元數'] = 8
        elif '16' in pixel_type:
            channel_info['每通道位元數'] = 16
        elif '32' in pixel_type:
            channel_info['每通道位元數'] = 32
    
    return channel_info


def extract_coordinate_metadata(czi_file):
    """改進版座標系統與元數據分析"""
    coord_info = {}

    # 獲取元數據基本資訊
    # 經確認，.meta屬性在此案例中不返回完整XML，但我們仍記錄其狀態
    metadata_xml = czi_file.meta
    coord_info['元數據來源'] = 'czi.meta property'
    coord_info['元數據長度'] = len(metadata_xml) if metadata_xml else 0
    
    if czi_file.is_mosaic():
        try:
            # 使用正確的API獲取圖塊座標
            all_tile_bboxes = czi_file.get_all_mosaic_tile_bounding_boxes()
            coord_info['圖塊總數'] = len(all_tile_bboxes)

            # 提取前幾個圖塊的座標作為範例
            sample_tiles = list(all_tile_bboxes.items())[:3]
            coord_info['圖塊座標範例'] = [
                {
                    'Tile_Index': str(tile_id), # 將 TileInfo 物件轉為字串
                    'X': bbox.x, 'Y': bbox.y,
                    'Width': bbox.w, 'Height': bbox.h
                }
                for tile_id, bbox in sample_tiles
            ]

            # 獲取整體馬賽克邊界框
            mosaic_bbox = czi_file.get_mosaic_bounding_box()
            coord_info['馬賽克邊界框'] = {
                'X': mosaic_bbox.x, 'Y': mosaic_bbox.y,
                'Width': mosaic_bbox.w, 'Height': mosaic_bbox.h
            }

            coord_info['元數據狀態'] = '完整 - 含圖塊座標'
            coord_info['對齊建議'] = '可以使用圖塊座標進行精確拼接'

        except Exception as e:
            coord_info['圖塊座標提取錯誤'] = str(e)
            coord_info['對齊建議'] = '圖塊座標提取失敗，建議使用feature-based alignment'

    # 檢查從 .meta 獲取的XML狀態
    if metadata_xml and len(metadata_xml) > 10:
        coord_info['XML_Status'] = '有效'
        try:
            # 解析XML元數據
            root = ET.fromstring(metadata_xml)
            
            # 尋找stage position資訊
            stage_positions = []
            resolution_info = {}
            
            # 搜尋各種可能的標籤
            for elem in root.iter():
                tag_lower = elem.tag.lower()
                
                # 尋找stage position
                if 'stage' in tag_lower or 'position' in tag_lower:
                    if elem.text and elem.text.strip():
                        stage_positions.append(f"{elem.tag}: {elem.text}")
                
                # 尋找解析度/像素間距資訊
                if any(keyword in tag_lower for keyword in ['resolution', 'pixel', 'scaling', 'size']):
                    if elem.text and elem.text.strip():
                        try:
                            value = float(elem.text)
                            resolution_info[elem.tag] = value
                        except ValueError:
                            resolution_info[elem.tag] = elem.text
            
            # 記錄找到的資訊
            if stage_positions:
                coord_info['XML_Stage_Position'] = stage_positions

            if resolution_info:
                coord_info['XML_解析度資訊'] = resolution_info

        except ET.ParseError as e:
            coord_info['XML_解析錯誤'] = str(e)
    else:
        coord_info['XML_Status'] = '無效或過短'
        # 如果沒有從 is_mosaic() 得到建議，則使用舊的建議
        if '對齊建議' not in coord_info:
            coord_info['對齊建議'] = '無元數據，建議使用feature-based alignment (SIFT、ORB)'

    return coord_info


def calculate_image_statistics(czi_file):
    """
    計算像素統計資訊 (更穩健的版本)
    - Histogram（灰階分佈）
    - Mean, Std, Min, Max
    
    策略：
    1. 如果是馬賽克影像，只讀取第一個圖塊進行分析，以節省記憶體。
    2. 如果不是馬賽克，且影像較大，則讀取中心區域。
    3. 如果影像較小，則讀取整個影像。
    """
    stats = {}
    
    try:
        print("    - 正在準備影像統計分析...")
        
        # 優先處理馬賽克影像
        if czi_file.is_mosaic():
            print("    - 檢測到馬賽克影像，將分析第一個圖塊以節省記憶體。")
            try:
                # 獲取第一個圖塊的邊界框
                tile_bboxes = czi_file.get_all_mosaic_tile_bounding_boxes()
                if not tile_bboxes:
                    stats['錯誤'] = '馬賽克影像不包含任何圖塊資訊'
                    return stats
                
                first_tile_bbox = list(tile_bboxes.values())[0]
                
                # 將 BBox 物件轉換為元組 (x, y, w, h)
                region_tuple = (first_tile_bbox.x, first_tile_bbox.y, first_tile_bbox.w, first_tile_bbox.h)
                
                stats['採樣說明'] = f'馬賽克影像：分析第一個圖塊 (Region {region_tuple})'
                print(f"    - 正在讀取第一個圖塊，邊界框: {region_tuple}...")
                
                # 讀取單個圖塊，傳入元組並指定通道
                region = czi_file.read_mosaic(region_tuple, scale_factor=1.0, C=0)
                print("    - ✓ 圖塊讀取完畢")

            except Exception as e:
                stats['錯誤'] = f'讀取馬賽克圖塊失敗: {str(e)}'
                print(f"    - ✗ {stats['錯誤']}")
                return stats
        else:
            # 非馬賽克影像的處理邏輯
            dims = czi_file.dims
            size = czi_file.size
            
            if 'Y' in dims and 'X' in dims:
                y_idx = dims.index('Y')
                x_idx = dims.index('X')
                height = size[y_idx]
                width = size[x_idx]
                
                max_sample_size = 1000  # 最大採樣尺寸
                
                if height > max_sample_size or width > max_sample_size:
                    center_y, center_x = height // 2, width // 2
                    half_sample = max_sample_size // 2
                    start_y, end_y = max(0, center_y - half_sample), min(height, center_y + half_sample)
                    start_x, end_x = max(0, center_x - half_sample), min(width, center_x + half_sample)
                    
                    stats['採樣說明'] = f'使用中心區域採樣 (Scene 0, Region Y=({start_y}:{end_y}), X=({start_x}:{end_x}))'
                    print("    - 正在讀取影像中心區域...")
                    region = czi_file.read_image(scene=0, Y=slice(start_y, end_y), X=slice(start_x, end_x))
                    print("    - ✓ 影像區域讀取完畢")
                else:
                    stats['採樣說明'] = '使用完整影像 (Scene 0)'
                    print("    - 正在讀取完整影像...")
                    region = czi_file.read_image(scene=0)
                    print("    - ✓ 完整影像讀取完畢")
            else:
                stats['錯誤'] = '無法識別Y和X維度'
                return stats

        # --- 接下來是統計計算 ---
        if 'region' in locals() and region is not None and region.size > 0:
            print("    - 正在計算統計數據...")
            # 轉換為numpy陣列並降維處理
            if hasattr(region, 'squeeze'):
                region = region.squeeze()
            
            # 統一轉為灰階進行基礎統計
            if len(region.shape) > 2 and region.shape[-1] in [3, 4]: # RGB/RGBA
                # 使用加權平均轉為灰階
                weights = [0.2989, 0.5870, 0.1140]
                if region.shape[-1] == 4: # RGBA
                    weights.append(0)
                gray_region = np.dot(region[...,:len(weights)], weights[:region.shape[-1]])
                stats['色彩處理'] = '轉換為灰階進行統計'
                flat_data = gray_region.flatten()
            else:
                flat_data = region.flatten()

            stats['影像形狀'] = region.shape
            stats['統計資訊'] = {
                '平均值': float(np.mean(flat_data)),
                '標準差': float(np.std(flat_data)),
                '最小值': int(np.min(flat_data)),
                '最大值': int(np.max(flat_data)),
                '像素總數': len(flat_data)
            }
            
            # 直方圖
            hist, bin_edges = np.histogram(flat_data, bins=50)
            stats['直方圖摘要'] = {
                '區間數': len(hist),
                '最高頻率': int(np.max(hist)),
                '值範圍': f'{bin_edges[0]:.2f} - {bin_edges[-1]:.2f}'
            }
            print("    - ✓ 統計數據計算完成")
            
            # 清理記憶體
            del region
            del flat_data
            gc.collect()
        else:
            stats['錯誤'] = '無法讀取或處理影像資料'
            print(f"    - ✗ {stats['錯誤']}")

    except Exception as e:
        error_message = f'統計計算過程中發生未知錯誤: {str(e)}'
        stats['錯誤'] = error_message
        print(f"    - ✗ {error_message}")
    
    return stats


def analyze_single_czi(filepath):
    """
    分析單個CZI檔案的完整函數
    實現記憶體管理策略: 讀取 → 分析 → 釋放
    """
    print(f"\n{'='*60}")
    print(f"分析文件: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    analysis_result = {
        '檔案資訊': {
            '檔案名稱': os.path.basename(filepath),
            '檔案路徑': str(filepath),
            '檔案大小': format_bytes(os.path.getsize(filepath)),
            '分析時間': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    czi = None
    
    try:
        # 1. 載入CZI檔案
        print("正在載入CZI檔案...")
        czi = CziFile(filepath)
        print("✓ CZI檔案載入成功")
        
        # 2. 基本影像屬性分析
        print("正在分析基本影像屬性...")
        analysis_result['基本影像屬性'] = extract_basic_properties(czi)
        print("✓ 基本影像屬性分析完成")
        
        # 3. 色彩通道資訊分析  
        print("正在分析色彩通道資訊...")
        analysis_result['色彩通道資訊'] = extract_channel_info(czi)
        print("✓ 色彩通道資訊分析完成")
        
        # 4. 座標系統與元數據分析
        print("正在分析座標系統與元數據...")
        analysis_result['座標系統與元數據'] = extract_coordinate_metadata(czi)
        print("✓ 座標系統與元數據分析完成")
        
        # 5. 像素統計資訊分析
        print("正在計算像素統計資訊...")
        analysis_result['像素統計資訊'] = calculate_image_statistics(czi)
        print("✓ 像素統計資訊計算完成")
        
        print("✓ 檔案分析完成")
        
    except Exception as e:
        error_msg = f"分析過程中發生錯誤: {str(e)}"
        print(f"✗ {error_msg}")
        analysis_result['錯誤'] = error_msg
    
    finally:
        # 強制記憶體清理
        if czi is not None:
            del czi
        gc.collect()
        print("✓ 記憶體已清理")
    
    return analysis_result


def generate_analysis_report(all_analyses, output_file="analysis.txt"):
    """
    生成詳細分析報告
    """
    print(f"\n正在生成分析報告: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 報告標題
        f.write("="*80 + "\n")
        f.write("CZI 檔案詳細分析報告\n")
        f.write("="*80 + "\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析工具: aicspylibczi + numpy\n")
        f.write(f"分析檔案數量: {len(all_analyses)}\n")
        f.write("\n")
        
        # 摘要
        f.write("分析摘要\n")
        f.write("-"*40 + "\n")
        successful_count = 0
        total_size = 0
        
        for filename, analysis in all_analyses.items():
            if '錯誤' not in analysis:
                successful_count += 1
            if '檔案資訊' in analysis:
                size_str = analysis['檔案資訊']['檔案大小']
                # 粗略計算總大小（僅用於顯示）
                if 'GB' in size_str:
                    total_size += float(size_str.split()[0])
        
        f.write(f"成功分析: {successful_count}/{len(all_analyses)} 個檔案\n")
        f.write(f"估計總大小: ~{total_size:.2f} GB\n")
        f.write("\n")
        
        # 詳細分析結果
        for filename, analysis in all_analyses.items():
            f.write("="*80 + "\n")
            f.write(f"檔案: {filename}\n")
            f.write("="*80 + "\n")
            
            # 遞迴寫入分析結果
            def write_dict(data, indent=0):
                prefix = "  " * indent
                for key, value in data.items():
                    if isinstance(value, dict):
                        f.write(f"{prefix}{key}:\n")
                        write_dict(value, indent + 1)
                    elif isinstance(value, list):
                        f.write(f"{prefix}{key}:\n")
                        for item in value:
                            f.write(f"{prefix}  - {item}\n")
                    else:
                        f.write(f"{prefix}{key}: {value}\n")
                f.write("\n")
            
            write_dict(analysis)
        
        f.write("="*80 + "\n")
        f.write("報告結束\n")
        f.write("="*80 + "\n")


def main():
    """
    主程序
    實現工作流程要求的記憶體管理策略
    """
    print("CZI檔案詳細分析程序")
    print("實現記憶體管理策略: 逐張讀取 → 分析 → 釋放")
    print("="*60)
    
    # 設定目錄和輸出檔案
    picture_dir = Path("E:/Class/tsgh/picture/whole_size/")  # 相對於testing目錄的picture目錄
    output_file = "analysis.txt"
    
    # 檢查picture目錄
    if not picture_dir.exists():
        print(f"錯誤: {picture_dir} 目錄不存在")
        sys.exit(1)
    
    # 尋找CZI檔案
    czi_files = list(picture_dir.glob("*.czi"))
    if not czi_files:
        print(f"錯誤: 在 {picture_dir} 中找不到CZI檔案")
        sys.exit(1)
    
    print(f"找到 {len(czi_files)} 個CZI檔案:")
    for f in czi_files:
        print(f"  - {f.name}")
    
    # 逐張分析檔案
    all_analyses = {}
    
    for i, czi_file in enumerate(czi_files, 1):
        print(f"\n處理第 {i}/{len(czi_files)} 個檔案...")
        
        # 分析單個檔案
        analysis = analyze_single_czi(czi_file)
        all_analyses[czi_file.name] = analysis
        
        # 每處理完一個檔案後強制垃圾收集
        gc.collect()
        print(f"第 {i} 個檔案處理完成，記憶體已清理")
    
    # 生成報告
    print(f"\n所有檔案分析完成！")
    generate_analysis_report(all_analyses, output_file)
    print(f"詳細分析報告已儲存至: {output_file}")
    
    # 最終記憶體清理
    del all_analyses
    gc.collect()
    print("✓ 程序執行完成，所有記憶體已清理")


if __name__ == "__main__":
    main()