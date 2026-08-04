# tsgh — WSI / medical image pipeline

Codebase specific conventions, paths, and notes are written in this file.
This project backend algorithem is base on Linux, if you're on Windows, use WSL2 or a Linux VM, otherwise it might hit a lot unexpected errors.

## project rules

1. Before editing any file, read it first. Before modifying a function, use codegraph for all callers. Research before you edit.
2. @rules/codegraph-first.md
3. @rules/karpathy_rule.md
4. @rules/no-regression.md ← **不得破壞既有功能。這條優先於任何「順手改好」的衝動。**

## 執行環境（兩台，2026-07-28 起）

這個專案有**兩台**機器在跑，改動必須兩邊都成立。

| | Windows（開發／前端） | Linux（運算／後端） |
|---|---|---|
| 位置 | `C:\Users\RCLab\Desktop\tsgh` | `yoyo@210.240.160.154:~/projects/tsgh` |
| Python | conda env `tsgh311` | `.venv`（uv 管理，3.11.15） |
| GPU | RTX 4080 16GB | RTX 2080 Ti 22GB（sm_75） |
| RAM | 32 GB | 62 GB + 8GB swap |
| 資料 | `D:\tsgh_output\` | `~/tsgh_data/{picture,thriple_image_layer,storage,viewer}` |
| Node | 裝在 conda env 根目錄 | `~/.local/opt/node`（v22） |

**Linux 跑法**：`source ~/projects/tsgh-env.sh`（設好 `TSGH_STORAGE_DIR` / `TSGH_SLIDES_DIR` /
`PYTHONPATH` / PATH），或 `~/projects/tsgh-dev.sh` 一次起後端(8000)+前端(5173)。

**Linux 特有的雷**（都實測過）：
- **沒有免密 sudo**，不能 apt 裝東西。所有依賴走使用者層級（uv / pyvips 自帶 libvips / openslide-bin）。
- **GPU 是跟別人共用的**。跑任務前先 `nvidia-smi`；曾遇到別人的 Streamlit 佔卡導致
  `CUDA error: CUDA-capable device(s) is/are busy or unavailable`（Compute Mode 是 Default，等一下重跑就好）。
- **ufw 開著且看不到規則** → 外部連不到 5173/8000，用
  `ssh -L 5173:localhost:5173 -L 8000:localhost:8000 yoyo@210.240.160.154` 轉埠。
- **後端 import 很重，uvicorn 約 42 秒**才開始回應。健康檢查別等太短。
- **那台沒有 GitHub 憑證**，程式碼是 tar over ssh 傳過去的，`git pull/push` 不能用。
  改完 Windows 這邊要手動同步過去。
- **SimpleElastix 只有 Linux 有**（manylinux wheel），所以 `non_rigid_method="elastix"`
  只能在 Linux 跑；Windows 那台只有 stock SimpleITK。
- `--test` 跑不起來：README 寫的 `backend/algorithms/hybrid/test_picture/` **兩台都不存在**，
  從未提交過。要煙霧測試請用真實 tile pair（見 no-regression.md）。

**全片規模的硬性限制**（round 8 實測，5090 上量的）：peak RSS **61 GB**、輸出 **347 GB/片**、
`workers=1` 只吃 2.7GB VRAM 但 `workers=4` 要 **30.4GB** → **2080 Ti 只能 `workers=1`**，
Windows 那台 32GB RAM 根本跑不完一張全片。原始文件被 commit `b04cbfd` 拔掉且 `docs/*` 被
gitignore，用 `git show b04cbfd^:docs/hybrid-pipeline/27-remaining-work-implementation.md` 撈。
