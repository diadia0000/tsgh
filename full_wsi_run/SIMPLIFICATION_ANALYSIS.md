# `full_wsi_run` 過度設計分析與簡化建議

> 目的：盤點目前 pipeline 哪些區塊是「自己造輪子」、哪些只要換成熟工具就能解決，
> 哪些根本是死碼。為下一階段 UI 化把 pipeline 收斂成最小核心。

---

## TL;DR — 真正該動的三件事

| # | 問題 | 影響 | 建議 |
|---|---|---|---|
| 1 | 三段式 thread + queue.Queue 串流 pipeline (~150 行) | 維護成本高、bug 面積大、實際只贏 GPU stream 1 個 batch | 改用 `torch.utils.data.DataLoader(num_workers, prefetch_factor)` 或單 prefetch thread，瘦身 80% |
| 2 | `BigTiffWriter` 用 `numpy.memmap` 暫存整張全圖再 `pyvips.rawload` 重寫 | 每張 WSI 多 ~64 GB 暫存 IO、寫檔兩次 | 改用 `tifffile.memmap(bigtiff=True, tile=...)` 直接 BigTIFF 落地，或 `pyvips` 的 sequential tile sink |
| 3 | dot detection 用 skimage 的 `binary_dilation`/`distance_transform_edt`/`regionprops`/`h_maxima` | skimage 在大張影像上比 OpenCV 慢 5–15×，是 post stage 主要 bottleneck | 改用 `cv2.dilate` / `cv2.distanceTransform` / `cv2.connectedComponentsWithStats` + 自製 H-maxima (一行 reconstruct) |

其他都是「可以順手刪」的小東西。

---

## 一、自己造輪子，可以換成熟工具

### 1.1 三段式 stream pipeline（`full_wsi_pipeline.py:521-789`）

**現況**：手寫 `io_worker → 主 GPU 迴圈 → post_worker` 三 thread + 兩個 `queue.Queue` +
sentinel + `_thread_errors` 例外傳播 + `_PipelineState` 共用狀態。約 270 行。

**問題**：
- Python GIL 下，I/O thread 真正的 overlap 只有 `openslide.read_region`（C 部分釋放 GIL，OK）
  與 GPU forward；post thread 大半時間都拿著 GIL 在跑 numpy/skimage，跟 GPU thread 互搶 CPU。
- `queue` 容量、SENTINEL、try/finally、`_thread_errors.append(exc)` 例外手動回傳——
  整套就是一個 mini async runtime。
- 三 thread 共用 `state` 物件、`profiler`、`io_q`、`post_q`，未來要加東西容易踩到。

**換法**：
- **方案 A（最少動）**：保留主 GPU 迴圈，只用一個 `concurrent.futures.ThreadPoolExecutor(max_workers=1)`
  做 prefetch。一個 future「下個 batch 的 patches 已讀好」就夠了——GPU 是瓶頸，
  prefetch 1 個 batch 已經把 I/O 藏完。
- **方案 B（更標準）**：把「window 列表 → IHC/DISH patch tensor」包成
  `torch.utils.data.Dataset`，用 `DataLoader(num_workers=2, prefetch_factor=2, pin_memory=True)`
  讀。post stage 改成 GPU 主迴圈跑完一個 batch 後直接同步處理（CPU 工作 < GPU，
  不會卡）。

**好處**：少 ~200 行、共用狀態變成 local variables、例外正常 raise。

---

### 1.2 `BigTiffWriter`（`m5_tiffwriter.py`）

**現況**：先把整張 mask 寫進 `numpy.memmap` 的原始 raw 檔，等 `close()` 才用
`pyvips.Image.rawload(...).tiffsave(...)` 重新編碼。

**問題**：
- 一張 114k × 141k 的 uint8 core mask = ~16 GB；instance mask uint32 = ~64 GB。
  這些都是「為了寫 TIFF 而建的中繼檔」，**寫進去之後立刻被讀出來再寫一次**。
  雙倍 I/O、雙倍磁碟 peak。
- pyvips 本身就支援 sequential / sink mode，可以邊餵 tile 邊寫 BigTIFF，
  根本不需要先落到 memmap。
- 或者乾脆用 `tifffile.memmap()` 直接以 BigTIFF tiled 格式 mmap 寫檔——一步到位。

**換法**：
- **方案 A（最簡單）**：`tifffile.memmap(path, shape=(H, W), dtype=..., bigtiff=True, tile=(512, 512))`
  直接拿到一個可隨機寫入的 mmap，每個 window 的 mask 直接寫進對應 slice。寫完
  就是合法 BigTIFF，不用 close 後再轉檔。
- **方案 B（要 pyramidal JPEG）**：用 `pyvips.Image.new_temp_file` + `tile sink`，
  或 `tifffile` 寫完非金字塔版再用 `vips tiffsave --pyramid` 一行 CLI 升級。

**好處**：少一倍磁碟使用、少一輪編解碼、不需要在 `close()` 時等好幾分鐘。

> 進階考慮：UI 化之後 stitched mask 的用途是給病理軟體看 (QuPath/ASAP)。
> 如果只是做品管「我看一下整張 segmentation 對不對」，**輸出 GeoJSON polygon
> 給 QuPath 載入**比 BigTIFF mask 輕量百倍。一張 WSI 約 100k 顆細胞，
> 多邊形大概 ~50 MB。`shapely + geojson` 兩個套件就能搞定。

---

### 1.3 dot detection 的 skimage 重型操作（`m3_dot_detection.py`）

| 操作 | 現用 | 建議 | 估計加速 |
|---|---|---|---|
| `binary_dilation(mask, disk(r))` | skimage | `cv2.dilate(mask, cv2.getStructuringElement(MORPH_ELLIPSE, ...))` | ×5–10 |
| `distance_transform_edt` | scipy.ndimage | `cv2.distanceTransform(..., DIST_L2, 5)` | ×3–5 |
| `regionprops(label_img, intensity_image=...)` | skimage | `cv2.connectedComponentsWithStats` 取 area/bbox/centroid + 必要時自己算 circularity | ×2–3 |
| `h_maxima(a, h)` / `h_minima(L, h)` | skimage（內部跑 reconstruction） | `cv2.morphologyEx` + reconstruction by dilation/erosion 自己組（< 10 行） | ×2 |
| `binary_erosion(eroded == nid)` 逐 nid loop | skimage | 直接對 `dish_nucleus_mask` 用 `cv2.erode` 一次處理 | ×N（其中 N=dish 核數） |

**為什麼值得換**：dot detection 是每個 window 都跑、且是 ThreadPool 平行化的對象；
skimage 函式不一定釋放 GIL，導致 ThreadPool 收益打折。換成 OpenCV 後 GIL 釋放更乾淨，
ThreadPool 增益會更明顯。

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

## 二、純粹的死碼 / UI 不需要

下面這些在 `full_wsi_pipeline.py` 完全沒被呼叫，是早期 tile 模式留下的化石：

| 檔案 | 函式/類別 | 狀態 |
|---|---|---|
| `m1_overlay.py` | `parse_tile_coords`, `find_paired_tiles`, `_build_coord_map`, `generate_ihc_core_mask` | **死碼**，UI 化後永遠不會走 tile 配對流程 |
| `m1_overlay.py` | `_TILE_COORD_PATTERN` regex | **死碼** |
| `m2_segmentation.py` | `predict()` 單張版 | **死碼**（pipeline 一律走 `predict_batch`） |
| `m4_export.py` | `export_per_cell_images`, `_extract_mask_shaped_cell`, `_fit_to_fixed_canvas`, `export_cell_dot_annotations`, `export_summary_statistics`, `export_tile_csv` | **死碼**（WSI 流程不切 per-cell PNG） |
| `unet_inference.py` | `predict_batch(image_paths, output_dir, save_proba, ...)` | **死碼**（UI 走 in-memory 路徑） |
| `unet_inference.py` | `predict_proba`, `_create_overlay`, `predict_batch` | **死碼** |
| `full_wsi_pipeline.py` | `_NullProfiler`, `_NullSection` | **死碼**（清掉 profiler 後一起清） |

**估計**：約 600 行可直接刪。UI 整合時這些都是噪音、增加 import 表面、
也會被 lint / IDE 拉出來干擾。

---

## 三、為了「順手好看」而生的雜訊

### 3.1 `gc.collect()` every 32 batches（`full_wsi_pipeline.py:780`）
Python 的 GC 對 numpy 大陣列基本無感；真正釋放 RAM 是 `del` + memmap flush。
這行只在「我擔心 RAM」時心安用，**刪掉**。

### 3.2 `_config_hash`（`full_wsi_pipeline.py:394`）
為 trace 用，hash 寫進 benchmark.json。UI 整合後 config 是動態（使用者填表），
hash 沒實際作用，**刪掉**。

### 3.3 `wsi_skip_white_threshold` 預設 `None`
config 留著沒問題，但目前沒人用——預設 `None` 等於關閉。建議直接設 `230.0`（標準
white-bg 閾值），預設啟用。

### 3.4 `pipeline_queue_size = 64`
搭配前面提的 stream pipeline 簡化，這個參數會跟著消失。

### 3.5 `dots_workers = 0` (= os.cpu_count())
SDD 已驗證有效，**保留**——但簡化後直接寫 `min(8, os.cpu_count())`，
不需要設成可調 config（一般沒人會調）。

### 3.6 `slide_summary` 的 streaming merge
目前每個 window 都 `state.slide_summary = state.slide_summary.merge(...)`，
等於每次都建一個新 dataclass。直接改成「累積 list，最後 `aggregate(list)`」就好。
微小的 GC 壓力差別。

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
│   └── tiff_writer.py          # tifffile.memmap 直接寫 BigTIFF
├── export.py                   # write_summary_csv + write_report_csv
└── config.py                   # dataclass，UI 表單會直接 read/write
```

**關鍵差異**：
- `pipeline.run(config, on_progress=lambda done, total: ...)` ── UI 的進度 callback。
  pipeline 內部不再 `logger.info` 進度，全部走 callback。
- 沒有 `benchmark.json`、`per_window_timings.csv`、`StageProfiler`、`_log_progress`。
- 沒有 thread/queue ── 改用 `DataLoader` 或單 prefetch future。
- `BigTiffWriter` 改名為 `MaskWriter`，內部直接 `tifffile.memmap`，刪掉 pyvips
  依賴（除非需要 pyramidal JPEG，那種情況再 fallback 到 vips CLI）。

---

## 五、行動優先順序

| 優先 | 動作 | 風險 | 預期效果 |
|---|---|---|---|
| P0 | 清掉 timing / benchmark code（本次 PR） | 低 | UI 化前置，code 乾淨 |
| P0 | 刪 `m1_overlay` / `m4_export` / `unet_inference` 死碼 | 低 | -600 行噪音 |
| P1 | 三段 stream pipeline → 單 prefetch | 中（要回歸測一輪） | 維護成本大幅下降 |
| P1 | dot detection: skimage → OpenCV | 中（要驗 dot 數一致） | 整體吞吐 +20–40% |
| P2 | `BigTiffWriter` → `tifffile.memmap` 直接寫 | 中（QuPath 相容測試） | 磁碟峰值砍半 |
| P2 | per-window 細胞輸出 → GeoJSON polygon | 中 | UI 看 mask 容易、檔案小 100× |
| P3 | 共用格式工具去重（`_format_count` etc.） | 低 | 1 個 helper module |

---

*Last updated: 2026-04-29*
