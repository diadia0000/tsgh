# `full_wsi_run` 過度設計分析與簡化建議

> 目的：盤點目前 pipeline 哪些區塊是「自己造輪子」、哪些只要換成熟工具就能解決，
> 哪些根本是死碼。為下一階段 UI 化把 pipeline 收斂成最小核心。

---

## TL;DR — 待辦事項

| # | 問題 | 影響 | 建議 | 狀態 |
|---|---|---|---|-|
| 1 | 三段式 thread + queue.Queue 串流 pipeline (~150 行) | 維護成本高、bug 面積大 | 改用 `DataLoader` | **[已完成]** |
| 2 | dot detection 用 skimage 重型操作 | skimage 比 OpenCV 慢 5–15×，是 post stage bottleneck | 改用 OpenCV | **[已完成]** |

> ~~BigTiffWriter 已於 2026-04-30 重寫：in-memory numpy + pyvips `new_from_array` + `tiffsave`，消除雙倍磁碟 I/O。~~

---

## 一、自己造輪子，可以換成熟工具

### 1.1 三段式 stream pipeline — [已完成]

**現況**：已改為 `torch.utils.data.DataLoader` + `_WSIWindowDataset`（`full_wsi_pipeline.py:107-191`）。
每個 DataLoader worker process 自行 lazy-init 一份 openslide handle，由 `persistent_workers` 持有整輪迭代。
主迴圈同步跑 GPU forward + post 處理（CPU 工作 << GPU，不會卡）。
`_split_batch` 處理空白 window 跳過。Post stage 的 dot detection 由 `ThreadPoolExecutor` 平行化。

**原問題**（已消除）：
- Python GIL 下，I/O thread 真正的 overlap 只有 `openslide.read_region`（C 部分釋放 GIL，OK）
  與 GPU forward；post thread 大半時間都拿著 GIL 在跑 numpy/skimage，跟 GPU thread 互搶 CPU。
- `queue` 容量、SENTINEL、try/finally、`_thread_errors.append(exc)` 例外手動回傳——
  整套就是一個 mini async runtime。
- 三 thread 共用 `state` 物件、`profiler`、`io_q`、`post_q`，未來要加東西容易踩到。

**實際採用方案**：方案 B（DataLoader）。

---

### 1.2 `BigTiffWriter`（`m5_tiffwriter.py`）— 已完成 2026-04-30

[已完成] 重寫為 in-memory numpy array 累積 + pyvips `new_from_array` + `tiffsave`。
原 `numpy.memmap` + `pyvips.rawload` 的雙倍磁碟 I/O 已消除。保留 `use_memmap=True`
降級路徑供記憶體有限的環境使用。

> 未來：upgrade libvips 8.16+ 後可改為真正的 tile-by-tile sequential sink，
> 進一步降低 peak RAM。GeoJSON polygon 輸出仍可作為品管選項。

---

### 1.3 dot detection 的 skimage → OpenCV — [已完成]

| 操作 | 原用 | 現用 | 狀態 |
|---|---|---|---|
| `binary_dilation(mask, disk(r))` | skimage | `cv2.dilate` + `_disk_kernel` | ✅ |
| `distance_transform_edt` | scipy.ndimage | `cv2.distanceTransform(..., DIST_L2, 5)` | ✅ |
| `regionprops(label_img, intensity_image=...)` | skimage | `_regionprops_cv`（`cv2.findContours` + `convexHull`） | ✅ |
| `h_maxima(a, h)` / `h_minima(L, h)` | skimage | 保留 skimage wrapper（pure cv2 iteration 在 2048×2048 需 100+ 次 dilation 才收斂，反而更慢） | ⚠️ 有意保留 |
| `binary_erosion(eroded == nid)` 逐 nid loop | skimage | `_erode_label_mask`（向量化距離變換） | ✅ |
| `label(..., connectivity=2)` | skimage | `cv2.connectedComponents` (connectivity=8) | ✅ |

**為什麼值得換**：dot detection 是每個 window 都跑、且是 ThreadPool 平行化的對象；
skimage 函式不一定釋放 GIL，導致 ThreadPool 收益打折。換成 OpenCV 後 GIL 釋放更乾淨，
ThreadPool 增益會更明顯。

**有意保留 skimage 的部分**：
- `rgb2lab` — OpenCV 的 LAB 校準不同，會迫使重新校準所有 dot_* threshold
- `h_maxima` / `h_minima` — morphological reconstruction 在 2048×2048 上 pure Python 迭代收斂太慢
- `distance_transform_edt(return_indices=True)` — 每 window 只跑一次，非熱路徑；OpenCV 無 indices 版本

---

### 1.4 owned-box 與邊界細胞處理 (`compute_owned_box`)

**現況**：手寫 `compute_owned_box` 計算 window 對 cell 的擁有權框；用 centroid 是否在
owned box 內判斷該 window 要不要寫該 cell。

**結論**：**保留**。這是 sliding-window with overlap 唯一正確的去重方法，沒有現成
工具可以直接套。skimage 的 `clear_border` 不能用——它會把貼邊的 cell 都丟掉，
但我們需要的是「重複 cell 在某一個 window 留住、其他丟掉」。

唯一可改進是：把 owned-box 計算結果包成 `np.ndarray` 後直接用向量化判斷，
不要逐 cell `for r in results` Python 迴圈。對 window 內 ~1000 cell 的 case
能省 1–2 ms。

---

### 1.5 CSV / summary 寫法

**現況**：直接用 stdlib `csv` module。

**結論**：**保留**。對 streaming 寫入（每個 window 處理完就 append 一段 row）來說 stdlib `csv`
最直接。改 `pandas` 沒好處——`pandas.DataFrame.to_csv` 不適合 streaming，要全部累積才能寫。

唯一的小優化：`_format_count` / `_format_ratio` 在 `m4_export.py` 跟 `full_wsi_pipeline.py`
重複定義（兩份一模一樣），抽成共用 module（或從 `m4_export` 直接 import）。

---

## 二、純粹的死碼 / UI 不需要 — 已清理

[已完成] 上述死碼已在之前的 PR 中全部清除。目前各檔案僅保留 active 功能：
- `m1_overlay.py` — 僅剩 `apply_mask`, `fuse_mask`, `overlay`
- `m2_segmentation.py` — 僅剩 `CellposeSegmenter` 類別
- `m4_export.py` — 僅剩 `export_overlay_visualization`, `DotStatsSummary`, `write_summary_csv`
- `unet_inference.py` — 僅剩 `UNetPPInference`, `postprocess_membrane_mask`
- `full_wsi_pipeline.py` — `_NullProfiler`, `_NullSection`, `_config_hash` 已移除

---

## 三、為了「順手好看」而生的雜訊

### 3.1 `gc.collect()` — 已刪除 ✓
Python 的 GC 對 numpy 大陣列基本無感；真正釋放 RAM 是 `del` + memmap flush。已移除。

### 3.2 `_config_hash` — 已刪除 ✓
為 trace 用，hash 寫進 benchmark.json。已隨 profiler 清理一併移除。

### 3.3 `wsi_skip_white_threshold` 預設值 — [已完成]
已從 `None` 改為 `245.0`（`config_example.py:87`），預設啟用空白 window 跳過。

### 3.4 `pipeline_queue_size = 64` — [已完成]
三段式 pipeline 已移除，此參數已不再存在。DataLoader 由 `wsi_io_workers` + `wsi_io_prefetch_factor` 取代。

### 3.5 `dots_workers = 0` (= os.cpu_count()) — [已完成]
`full_wsi_pipeline.py:531` 已有 `config.dots_workers or (os.cpu_count() or 4)` 的回退邏輯，保留 config 彈性但行為一致。

### 3.6 `slide_summary` 的 streaming merge — [已完成]
已改為 `summary_chunks: List[DotStatsSummary]` 累積，最後 `DotStatsSummary.aggregate(summary_chunks)` 一次合併（`full_wsi_pipeline.py:572:688`）。消除每 window 建新 dataclass 的 GC 壓力。

---

## 四、建議的最小核心架構（UI 化目標）

```
full_wsi_run/
├── core/                       # 純運算，無 I/O / 無 logging 噪音
│   ├── unet.py                 # UNetPPInference.predict_batch_arrays（保留）
│   ├── cellpose.py             # CellposeSegmenter.predict_batch（保留）
│   ├── overlay.py              # apply_mask, fuse, overlay 三個函式
│   ├── dots.py                 # detect_all_dots（用 OpenCV 重寫）
│   └── cells.py                # build_all_positive_results
├── pipeline.py                 # run(config, on_progress) — 給 UI 呼叫的單一入口
├── io/
│   ├── wsi_reader.py           # WSIReader（保留 openslide 版）
│   └── tiff_writer.py          # BigTiffWriter (pyvips in-memory + tiffsave)
├── export.py                   # write_summary_csv + write_report_csv
└── config.py                   # dataclass，UI 表單會直接 read/write
```

**關鍵差異**：
- `pipeline.run(config, on_progress=lambda done, total: ...)` ── UI 的進度 callback。
  pipeline 內部不再 `logger.info` 進度，全部走 callback。
- 沒有 `benchmark.json`、`per_window_timings.csv`、`StageProfiler`、`_log_progress`。
- 沒有 thread/queue ── 改用 `DataLoader` 或單 prefetch future。
- `BigTiffWriter` 已使用 pyvips in-memory 模式；改名為 `MaskWriter` 時保留 pyvips
  （需要 pyramidal JPEG）。未來 upgrade libvips 8.16+ 可改為 tile sink 降低 RAM。

---

## 五、行動優先順序

| 優先 | 動作 | 風險 | 預期效果 | 狀態 |
|---|---|---|-|---|
| P1 | 三段 stream pipeline → 單 prefetch | 中（要回歸測一輪） | 維護成本大幅下降 | **[已完成]** |
| P1 | dot detection: skimage → OpenCV | 中（要驗 dot 數一致） | 整體吞吐 +20–40% | **[已完成]** |
| P2 | per-window 細胞輸出 → GeoJSON polygon | 中 | UI 看 mask 容易、檔案小 100× | **待評估** |
| P3 | 共用格式工具去重（`_format_count` etc.） | 低 | 1 個 helper module | **[已完成]** |
| P3 | wsi_skip_white_threshold 預設值 | 低 | 預設啟用空白跳過 | **[已完成]** |

> 已完成：P0 timing/benchmark 清理、死碼刪除、gc.collect、_config_hash 移除、BigTiffWriter in-memory 重寫、三段 pipeline → DataLoader、dot detection OpenCV 化、summary accumulate + aggregate、wsi_skip_white_threshold 預設值

---

*Last updated: 2026-04-30 — 已移除完成項目：§1.1 三段 pipeline → DataLoader、§1.2 BigTiffWriter 重寫、§1.3 dot detection OpenCV 化、§II 死碼清理、§3.1 gc.collect、§3.2 _config_hash、§3.3 skip_white 預設值、§3.4 queue_size 移除、§3.6 summary aggregate*
