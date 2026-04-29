# Full WSI Run

單入口、sliding-window 方式，對整張 WSI 跑完整 `full_wsi_run` pipeline (M1→M4)。


## 架構

- 讀取：`tifffile + zarr` 對 tiled BigTIFF 做 lazy 讀取，不會一次把 19 GB 全圖載入 RAM
- 推論：每個 window 直接使用 `full_wsi_run` 在地 M1→M4 模組
- UNet++：原本就有 sliding window，window 內若仍大於 `unet_image_size` 會再細切
- Cellpose：以 `wsi_window_size` 為單位逐塊推論；切在 window 邊界的細胞由 `clear_border` 丟棄
- 合併：所有 window 的 `CellAnalysisResult` 把 centroid 從 window-local 轉為 WSI 全圖座標，寫入一份 slide-level CSV
- 選項：`save_stitched_core_mask` / `save_stitched_instance_mask` 會輸出 slide 尺寸的 BigTIFF mask

## 使用

```bash
cp config_example.py config.py
# 編輯 config.py：ihc_wsi_path / dish_wsi_path / 模型路徑 / slide_id / output_dir

python full_wsi_pipeline.py
# 或覆寫
python full_wsi_pipeline.py --window 2048 --overlap 128 --limit 20  # smoke test
```

## 輸出

位於 `{output_dir}/{slide_id}/`：

| 檔名 | 內容 |
|---|---|
| `{slide_id}_report.csv` | 全 slide 所有細胞（global_cell_id、WSI 座標、紅/黑點數、ratio） |
| `{slide_id}_summary.csv` | 全 slide 統計（有效雙色細胞、ratio<2/≥2、copy 分布） |
| `{slide_id}_core_mask.tiff` | （選） stitched UNet++ mask, uint8 BigTIFF |
| `{slide_id}_instance_mask.tiff` | （選） stitched Cellpose instance mask, uint32 BigTIFF |
| `windows/w_y{Y}_x{X}/...` | （選） 保留每個 window 的完整產物，開啟 `save_per_window_artifacts` 才有 |

## 硬體預估

以 114k × 141k 的 WSI（約 2 萬個 1024×1024 window）為例：

- 背景篩選開啟後，實際處理 window 數通常降到 30–60%
- UNet++: ~0.1–0.3 s/window (GPU)
- Cellpose (M2 + M3b 兩次): ~1–3 s/window (GPU)
- 合計 ~1.5–4 s/window × 有效 window 數 → 單張 WSI 3–12 小時量級，非常依賴 `cellpose_diameter` / 前景比例
- RAM 用量：每個 window 峰值 < 1 GB；全程 RSS 主要由 stitched BigTIFF memmap 決定
- VRAM：Cellpose + UNet++ 共用，1024 window 下約 3–6 GB

開 `save_stitched_instance_mask=True` 會在磁碟上多一個 `H*W*4 bytes` 的檔案（此 WSI 約 64 GB）。
建議第一次先用 `--limit 20 --window 1024` 跑 smoke test，再決定是否整張下去。
