#!/usr/bin/env python3
"""
高效能 CZI 轉 BigTIFF 模組
使用真正的多進程 (multiprocessing) 而非多執行緒，繞過 Python GIL 限制
每個進程使用獨立的 CPU 核心，實現真正的平行處理
"""

import os
import sys

# 必須在 import pyvips 之前設定，libvips 啟動時才會讀
_VIPS_THREADS = os.cpu_count()
os.environ["VIPS_CONCURRENCY"] = str(_VIPS_THREADS)

import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pyvips
import aicspylibczi
from tqdm import tqdm
try:
    from .config import create_default_config
except ImportError:
    from backend.algorithms.thriple_image_layer.config import create_default_config

def process_strip_worker(task: Dict[str, Any]) -> Tuple[bool, str, int]:
    """
    多進程 Worker - 讀取 CZI 區域，寫成 VIPS 暫存檔 (.v)

    task 鍵值: input_path / region (x,y,w,h) / scale_factor /
               output_vips_path / strip_index / modality

    Returns:
        Tuple[bool, str, int]: (成功與否, 訊息或路徑, 區塊索引)
    """
    idx, out_path = task["strip_index"], task["output_vips_path"]
    if os.path.exists(out_path):
        return (True, out_path, idx)

    try:
        czi = aicspylibczi.CziFile(task["input_path"])
        # C=0: RGB brightfield 通常在 channel 0；squeeze 掉 (1,H,W,3) 的前導維度
        data = np.squeeze(czi.read_mosaic(
            region=task["region"], scale_factor=task["scale_factor"], C=0))

        if data.ndim == 3:
            if data.shape[0] < 5 < data.shape[2]:   # (C,H,W) -> (H,W,C)
                data = data.transpose(1, 2, 0)
            if data.shape[2] == 3:
                data = data[:, :, ::-1]             # CZI 是 BGR，不轉顏色會錯（棕變藍）

        # new_from_array 自行推導 width/height/bands 並處理非連續記憶體
        img = pyvips.Image.new_from_array(data.astype(np.uint8, copy=False))
        img.write_to_file(out_path)
        return (True, out_path, idx)

    except Exception as e:
        raise RuntimeError(f"處理 {task['modality']} 區塊 {idx} 失敗: {e}") from e


class CziPreprocessor:
    """
    高效能 CZI 轉 BigTIFF 處理器
    
    特點：
    - 使用真正的多進程 (multiprocessing.Pool) 繞過 GIL
    - 每個進程獨立使用一個 CPU 核心
    - 分割影像為條狀區塊平行處理
    - 使用 pyvips 記憶體映射減少 RAM 使用
    """

    def __init__(self, config):
        self.config = config
        # Module 1: 從 czi_input_dir 讀取 CZI，輸出到 input_dir (作為 Module 2 的輸入)
        self.czi_input_dir = self.config.czi_input_dir
        self.output_dir = self.config.input_dir  # Module 1 輸出 = Module 2 輸入
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.output_dir / "temp_vips"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_processes = config.preprocess.num_processes
        print(f"將使用 {self.num_processes} 個獨立進程 (記憶體安全模式)")

    def get_conversion_tasks(self) -> List[Dict[str, Any]]:
        """準備轉換任務列表 (每個模態一組條狀區塊)"""
        # strip_height 是原始座標系的像素數；read_mosaic 會依 scale_factor 縮放輸出
        strip_height = self.config.preprocess.strip_height
        file_plans = []

        for m in self.config.modalities:
            if not m.czi_filename:
                print(f"跳過 {m.name}: 未設定 czi_filename")
                continue

            input_path = self.czi_input_dir / m.czi_filename
            if not input_path.exists():
                raise FileNotFoundError(f"{m.name}: 找不到 CZI 輸入 {input_path}")

            try:
                bbox = aicspylibczi.CziFile(str(input_path)).get_mosaic_bounding_box()
            except Exception as e:
                raise RuntimeError(f"讀取 {input_path} 失敗: {e}") from e

            x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
            s = m.scale_factor  # 1.0 = 40X 原始解析度, 0.5 = 縮小一半 (20X)
            mem_gb = int(w * s) * int(strip_height * s) * 3 / 1024**3
            print(f"準備 {m.name}: {w}x{h} px -> 輸出 {int(w * s)}x{int(h * s)} px")
            print(f"  scale_factor={s:.2f}, 每個區塊約 {mem_gb:.2f} GB")

            file_plans.append({
                "modality": m.name,
                "output_path": str(self.output_dir / m.filename),
                "strips": [{
                    "input_path": str(input_path),
                    "region": (x, top, w, min(strip_height, y + h - top)),
                    "scale_factor": s,
                    "output_vips_path": str(self.temp_dir / f"{m.name}_strip_{i:04d}.v"),
                    "strip_index": i,
                    "modality": m.name,
                } for i, top in enumerate(range(y, y + h, strip_height))],
            })

        return file_plans

    def run(self):
        """執行轉換流程"""
        file_plans = self.get_conversion_tasks()
        
        if not file_plans:
            raise RuntimeError("預處理失敗：沒有可處理的模態（找不到 CZI 輸入或設定錯誤）")

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
            raise RuntimeError(
                f"預處理失敗：{len(failed_strips)} 個區塊處理失敗 (strip index: {sorted(failed_strips)})，中止組裝"
            )

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
                
                # 第一階段：分批組裝
                print(f"  分批組裝 ({BATCH_SIZE} strips/batch)...")
                batches = []
                for batch_start in range(0, len(strip_paths), BATCH_SIZE):
                    batch_paths = strip_paths[batch_start:batch_start + BATCH_SIZE]
                    
                    # 在批次內使用 join (.v 檔 memory-mapped，access='random' 可多 thread)
                    batch_result = pyvips.Image.new_from_file(batch_paths[0], access='random')
                    for p in batch_paths[1:]:
                        strip = pyvips.Image.new_from_file(p, access='random')
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
                    compression="jpeg",
                    tile=True,           # 必須使用 tile 格式
                    tile_width=1024,      # tile 尺寸
                    tile_height=1024,
                    bigtiff=True,        # 支援大於 4GB 檔案
                    pyramid=True,        # 生成金字塔層級
                    subifd=True,         # 使用 SubIFD 格式存放金字塔層級 (VALIS 相容)
                    Q=95,
                )
                
                size_gb = os.path.getsize(out_path) / (1024**3)
                print(f"✓ 完成: {out_path} ({size_gb:.2f} GB)")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"預處理失敗：組裝 {modality} 失敗: {e}") from e

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
