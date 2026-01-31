#!/usr/bin/env python3
"""
高效能 CZI 轉 BigTIFF 模組
使用真正的多進程 (multiprocessing) 而非多執行緒，繞過 Python GIL 限制
每個進程使用獨立的 CPU 核心，實現真正的平行處理
"""

import os
import sys


import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Tuple

import numpy as np
import pyvips
import aicspylibczi
from tqdm import tqdm
import gc
try:
    from .config import create_default_config, ModalityConfig
except ImportError:
    from config import create_default_config, ModalityConfig


def process_strip_worker(task: Dict[str, Any]) -> Tuple[bool, str, int]:
    """
    多進程 Worker 函數 - 在獨立進程中執行
    讀取 CZI 區域並轉換為 VIPS 格式暫存檔
    
    這個函數會在獨立的 Python 進程中執行，
    每個進程使用獨立的記憶體空間和 CPU 核心
    
    Args:
        task: 包含以下鍵值的字典
            - input_path: CZI 檔案路徑
            - region: (x, y, w, h) 區域座標 (pixels)
            - scale_factor: 縮放比例
            - output_vips_path: 輸出暫存檔路徑
            - strip_index: 條狀區塊索引
            - modality: 影像模態名稱
    
    Returns:
        Tuple[bool, str, int]: (成功與否, 訊息或路徑, 區塊索引)
    """
    # 從 task 字典解構參數
    input_path: str = task["input_path"]
    region: Tuple[int, int, int, int] = task["region"]
    scale: float = task["scale_factor"]
    out_path: str = task["output_vips_path"]
    strip_idx: int = task["strip_index"]
    modality: str = task["modality"]
    
    try:
        # 開啟 CZI 檔案
        czi = aicspylibczi.CziFile(input_path)
        
        # 讀取指定區域的馬賽克影像
        # C=0 指定通道 (RGB brightfield 通常是 channel 0)
        chunk_data = czi.read_mosaic(region=region, scale_factor=scale, C=0)
        
        # 移除多餘的維度 (1, H, W, 3) -> (H, W, 3)
        chunk_data = np.squeeze(chunk_data)
        
        # 處理維度順序
        if chunk_data.ndim == 3:
            # 檢查是否需要轉置 (C, H, W) -> (H, W, C)
            if chunk_data.shape[0] < 5 and chunk_data.shape[2] > 5:
                chunk_data = chunk_data.transpose(1, 2, 0)
            
            # 關鍵修正：CZI 通常是 BGR 順序，需要轉換為 RGB
            # 否則顏色會錯（棕色變藍色）
            if chunk_data.shape[2] == 3:
                chunk_data = chunk_data[:, :, ::-1].copy()  # BGR -> RGB
        
        # 確保資料型別正確
        if chunk_data.dtype != np.uint8:
            chunk_data = chunk_data.astype(np.uint8)
        
        # 轉換為 pyvips 影像
        height, width = chunk_data.shape[:2]
        bands = chunk_data.shape[2] if chunk_data.ndim == 3 else 1
        
        # 使用 new_from_memory 更有效率
        vimg = pyvips.Image.new_from_memory(
            chunk_data.tobytes(),
            width, height, bands,
            'uchar'
        )
        
        # 寫入 VIPS 格式檔案 (記憶體映射，速度快)
        vimg.write_to_file(out_path)
        
        # 明確釋放記憶體
        del chunk_data, vimg, czi
        gc.collect()
        return (True, out_path, strip_idx)
        
    except Exception as e:
        return (False, f"Strip {strip_idx} ({modality}) error: {e}", strip_idx)


class CziPreprocessor:
    """
    高效能 CZI 轉 BigTIFF 處理器
    
    特點：
    - 使用真正的多進程 (multiprocessing.Pool) 繞過 GIL
    - 每個進程獨立使用一個 CPU 核心
    - 分割影像為條狀區塊平行處理
    - 使用 pyvips 記憶體映射減少 RAM 使用
    """

    def __init__(self, config, num_processes: int = None):
        self.config = config
        # Module 1: 從 czi_input_dir 讀取 CZI，輸出到 input_dir (作為 Module 2 的輸入)
        self.czi_input_dir = self.config.czi_input_dir
        self.output_dir = self.config.input_dir  # Module 1 輸出 = Module 2 輸入
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 暫存目錄
        self.temp_dir = self.output_dir / "temp_strips"
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_processes = num_processes if num_processes else cpu_count()-10
        print(f"將使用 {self.num_processes} 個獨立進程 (記憶體安全模式)")

    def get_conversion_tasks(self) -> List[Dict[str, Any]]:
        """準備轉換任務列表"""
        # 條狀區塊高度 (原始座標系)
        # 注意：這是在原始解析度下的像素數
        # read_mosaic 會根據 scale_factor 自動縮放輸出
        strip_height = 4096
        
        file_plans = []
        
        for modality in self.config.modalities:
            # 使用 czi_filename 作為輸入
            if not modality.czi_filename:
                print(f"跳過 {modality.name}: 未設定 czi_filename")
                continue
            
            input_path = self.czi_input_dir / modality.czi_filename
            output_filename = modality.filename  # 使用 config 中定義的輸出檔名
            output_path = self.output_dir / output_filename
            
            if not input_path.exists():
                print(f"跳過 {modality.name}: 找不到 {input_path}")
                continue
            
            try:
                czi = aicspylibczi.CziFile(str(input_path))
                bbox = czi.get_mosaic_bounding_box()
                x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
                
                # 直接使用 config 中的 scale_factor
                # 1.0 = 不縮放 (40X 原始解析度), 0.5 = 縮小一半 (20X)
                scale_factor = modality.scale_factor
                
                # 估算每個區塊的記憶體使用量
                output_width = int(w * scale_factor)
                output_strip_h = int(strip_height * scale_factor)
                mem_per_strip_gb = (output_width * output_strip_h * 3) / (1024**3)
                print(f"準備 {modality.name}: {w}x{h} px -> 輸出 {output_width}x{int(h*scale_factor)} px")
                print(f"  scale_factor={scale_factor:.2f}, 每個區塊約 {mem_per_strip_gb:.2f} GB")


                # 產生條狀區塊任務
                strips = []
                num_strips = (h + strip_height - 1) // strip_height
                
                for i in range(num_strips):
                    current_y = y + i * strip_height
                    current_h = min(strip_height, y + h - current_y)
                    
                    strip_vips_path = self.temp_dir / f"{modality.name}_strip_{i:04d}.v"
                    
                    strips.append({
                        "input_path": str(input_path),
                        "region": (x, current_y, w, current_h),
                        "scale_factor": scale_factor,
                        "output_vips_path": str(strip_vips_path),
                        "strip_index": i,
                        "modality": modality.name
                    })
                
                file_plans.append({
                    "modality": modality.name,
                    "output_path": str(output_path),
                    "strips": strips
                })
                
            except Exception as e:
                print(f"讀取 {input_path} 失敗: {e}")
                
        return file_plans

    def run(self):
        """執行轉換流程"""
        file_plans = self.get_conversion_tasks()
        
        if not file_plans:
            print("沒有任務需要執行")
            return

        # 收集所有條狀區塊任務
        all_strip_tasks = []
        for plan in file_plans:
            all_strip_tasks.extend(plan['strips'])
        
        total_tasks = len(all_strip_tasks)
        print(f"\n開始處理 {total_tasks} 個條狀區塊 (使用 {self.num_processes} 進程)")
        print("每個進程使用獨立的 CPU 核心，真正平行執行\n")
        
        # 使用多進程池執行
        # maxtasksperchild=10 讓每個 worker 處理一定數量後重啟，避免記憶體洩漏
        failed_strips = []
        
        with Pool(processes=self.num_processes, maxtasksperchild=60) as pool:
            # imap_unordered 不保證順序，但更快
            results = list(tqdm(
                pool.imap_unordered(process_strip_worker, all_strip_tasks),
                total=total_tasks,
                desc="處理進度"
            ))
        
        # 檢查錯誤
        for success, msg, idx in results:
            if not success:
                print(f"錯誤: {msg}")
                failed_strips.append(idx)
        
        if failed_strips:
            print(f"\n{len(failed_strips)} 個區塊處理失敗，中止組裝")
            return
            
        # --- 關鍵修正：組裝階段解放 CPU ---
        # 前面為了多進程限制了單核，現在要改回多執行緒模式
        # 這樣 tiffsave 才能利用所有核心進行壓縮和寫入
        print(f"切換為多執行緒模式 (Concurrency: {self.num_processes})")
        os.environ['VIPS_CONCURRENCY'] = str(self.num_processes)
        # 組裝最終影像
        print("\n組裝 BigTIFF 影像...")
        
        for plan in file_plans:
            modality = plan['modality']
            out_path = plan['output_path']
            
            # 依照索引排序
            modality_strips = sorted(plan['strips'], key=lambda x: x['strip_index'])
            strip_paths = [t['output_vips_path'] for t in modality_strips]
            
            if not strip_paths:
                continue

            try:
                print(f"組裝 {modality} ({len(strip_paths)} 個區塊)...")
                
                # 分批 join 策略：
                # 1. 每批處理 BATCH_SIZE 個 strips (減少單次 pipeline 深度)
                # 2. 再將各批次組合起來
                # 這比遞迴 join 更淺，比 arrayjoin 更穩定
                
                BATCH_SIZE = 16  # 每批處理的 strips 數量
                
                first_strip = pyvips.Image.new_from_file(strip_paths[0], access='sequential')
                print(f"  每個區塊: {first_strip.width} x {first_strip.height}")
                print(f"  預計總高度: {first_strip.height * len(strip_paths)}")
                del first_strip
                
                # 第一階段：分批組裝
                print(f"  分批組裝 ({BATCH_SIZE} strips/batch)...")
                batches = []
                for batch_start in range(0, len(strip_paths), BATCH_SIZE):
                    batch_paths = strip_paths[batch_start:batch_start + BATCH_SIZE]
                    
                    # 在批次內使用 join
                    batch_result = pyvips.Image.new_from_file(batch_paths[0], access='sequential')
                    for p in batch_paths[1:]:
                        strip = pyvips.Image.new_from_file(p, access='sequential')
                        batch_result = batch_result.join(strip, 'vertical', expand=True)
                    
                    batches.append(batch_result)
                    print(f"    批次 {len(batches)}/{(len(strip_paths) + BATCH_SIZE - 1) // BATCH_SIZE} 完成")
                
                # 第二階段：組合所有批次
                print(f"  組合 {len(batches)} 個批次...")
                result = batches[0]
                for i, batch in enumerate(batches[1:], 1):
                    result = result.join(batch, 'vertical', expand=True)
                    if i % 2 == 0:
                        print(f"    已合併 {i + 1}/{len(batches)} 批次")
                
                print(f"  最終尺寸: {result.width} x {result.height}")
                print(f"  寫入 {out_path}...")
                
                # 串流寫入 - VIPS 會自動處理
                # 重要：VALIS 需要 tile-based 金字塔 TIFF
                # 使用 subifd=True 將金字塔層級存為 SubIFD 格式
                # 這樣可以避免 "tiff2vips: page 1 differs from page 0" 錯誤
                result.tiffsave(
                    out_path,
                    compression="jpeg",  # JPEG 壓縮減少檔案大小
                    Q=95,                # JPEG 品質
                    tile=True,           # 必須使用 tile 格式
                    tile_width=1024,      # tile 尺寸
                    tile_height=1024,
                    bigtiff=True,        # 支援大於 4GB 檔案
                    pyramid=True,        # 生成金字塔層級
                    subifd=True,         # 使用 SubIFD 格式存放金字塔層級 (VALIS 相容)
                )
                
                # 檢查檔案大小
                import os as os_module
                size_gb = os_module.path.getsize(out_path) / (1024**3)
                print(f"✓ 完成: {out_path} ({size_gb:.2f} GB)")
                
            except Exception as e:
                print(f"✗ 組裝 {modality} 失敗: {e}")
                import traceback
                traceback.print_exc()

        # 清理暫存檔
        print("\n清理暫存檔...")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        print("全部完成!")


if __name__ == "__main__":
    print("=" * 60)
    print("CZI to BigTIFF 多進程轉換器")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"可用 CPU 核心: {cpu_count()}")
    print()
    
    config = create_default_config()
    processor = CziPreprocessor(config)
    processor.run()
