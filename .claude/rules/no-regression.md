# Rule: no-regression（不得破壞既有功能）

**為什麼**：這個 repo 有其他人在用。既有的行為、輸出格式和 API 是組員手上工作的
前提；改壞任何一項，等於逼他們把已經完成的東西重做一遍。**「順手改好」造成的返工
成本遠高於它省下的時間。**

這條規則優先於重構、風格統一、效能微調、以及任何「我覺得這樣比較好」。

## 受保護的契約（動到就是 breaking change）

除非使用者明確要求改這一項，否則以下都不准動：

1. **輸出格式** — `report.csv` 的欄位與順序（`cell_id, centroid_x, centroid_y,
   reddot, blackdot, score`）、`summary.txt` 的 ASCO/CAP 版面、`overlay_slide.tiff`、
   `overlay_annotated/`、`cell_crops/` 的檔名規則。下游有人在讀這些。
2. **API 路徑與回應結構** — `/api/alignment/*`、`/api/hybrid/*`、`/api/jobs/{id}`、
   `/api/tiles*`。前端的 typed client 是從 `openapi.json` 產的，改了就要重產。
3. **既有函式的簽章與回傳型別** — m0–m4、module1–4、`backend/io/`、`backend/schemas/`。
4. **`config_example.py` 的既有預設值** — 那些數字是組員調過、驗證過的演算法行為，
   不是隨手填的。改預設值 = 靜默改變所有人的結果。
5. **tile 檔名慣例** `tile_x{int}_y{int}`。

## 怎麼做

- **預設「加」而不是「改」**：新功能開新檔案、新函式，或加**帶預設值**的新參數，
  讓不傳參數時行為與現在完全相同。
- **動既有函式前，先 `codegraph_impact` 看 blast radius**（見 codegraph-first.md）。
  呼叫者不只一個就先講，不要自己決定。
- **平台差異走 config / 環境變數，不要寫死路徑進程式碼**。
  `backend/algorithms/*/config.py` 是 gitignored 的機器專屬檔，Linux 的路徑只能寫在
  那裡；不要為了讓 Linux 跑起來就去改被 git 追蹤的檔案。
- **不要清理你沒動到的東西**：不相關的 dead code、格式、註解一律不碰，看到就回報。
- **非改不可時**：先說明為什麼、影響誰、有沒有相容的替代做法，**拿到同意再動**。

## 改動前後都要跑的煙霧測試

驗證過可用（2026-07-28，Linux 那台）。`--test` 不能用（`test_picture/` 不存在）：

```bash
source ~/projects/tsgh-env.sh && cd $TSGH_ROOT
.venv/bin/python backend/algorithms/hybrid/hybrid_pipeline.py \
  --ihc ~/tsgh_data/smoke/ihc/tile_x58188_y66690.tiff \
  --dish ~/tsgh_data/smoke/dish/tile_x58188_y66690.tiff \
  --output /tmp/smoke_after
```

合格標準：exit 0、`成功=1 跳過=0`、產出 `report.csv` / `summary.txt` /
`overlay_slide.tiff`。**改動前先跑一次存基準，改完 diff `report.csv`** —— 不該變的
時候變了，就是 regression。

後端契約也要驗（改到 API 或 schema 時）：

```bash
.venv/bin/python -c "import backend.main"                      # import 得過
curl -s http://127.0.0.1:8000/openapi.json | python3 -c "import json,sys; print(sorted(json.load(sys.stdin)['paths']))"
```

端點清單少一個或路徑變了 → 停下來問，不要自己改前端去配合。
