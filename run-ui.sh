#!/usr/bin/env bash
# ============================================================
# run-ui.sh — 在自己的機器上開起 tsgh 的操作介面
#
#   ./run-ui.sh              首次執行也可以，缺的東西會自動補
#   ./run-ui.sh --skip-deps  已經裝過了，直接開（快很多）
#   ./run-ui.sh --no-open    不要自動開瀏覽器
#
# Linux / WSL2 用。Windows 請在 WSL2 裡跑（後端的影像處理相依只在 Linux 成立，
# 見 .claude/CLAUDE.md）。
#
# 這支腳本不改任何被 git 追蹤的檔案。它會建立的只有兩樣，兩樣都在 repo 外面或
# 被 gitignore：`.venv/`（uv 管的虛擬環境）與 `backend/algorithms/hybrid/config.py`
# （機器專屬設定，本來就 gitignored）。
# ============================================================
set -euo pipefail

# repo 根目錄由腳本自己的位置推出來，所以從哪個資料夾呼叫都可以。
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
# 後端 import 很重（torch + valis + 三個模型），實測約 42 秒才會開始回應。
# 這個上限是留給比較慢的機器的，不是預期值。
BACKEND_READY_TIMEOUT=240
BACKEND_LOG="$REPO/.run-ui-backend.log"

SKIP_DEPS=0
OPEN_BROWSER=1
for arg in "$@"; do
  case "$arg" in
    --skip-deps) SKIP_DEPS=1 ;;
    --no-open)   OPEN_BROWSER=0 ;;
    -h|--help)   sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "不認得的參數：$arg（用 --help 看用法）" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1;36m▸ %s\033[0m\n' "$*"; }
ok()   { printf '  \033[0;32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[0;33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[0;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# 連得上 = 那個 port 已經有人在聽。純 bash，不依賴 ss / lsof / netstat，
# 因為這三個在精簡的 WSL2 映像裡不一定都有。
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3<&- && return 0 || return 1; }

# ------------------------------------------------------------
# 1. 該裝的工具
# ------------------------------------------------------------
say "檢查工具"

command -v uv >/dev/null 2>&1 || die "找不到 uv。這個專案的 Python 環境只由 uv 管（不用 pip）。
  安裝：curl -LsSf https://astral.sh/uv/install.sh | sh
  裝完把 ~/.local/bin 加進 PATH，重開 shell 再跑一次。"
ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"

command -v npm >/dev/null 2>&1 || die "找不到 npm。前端是 Vite，需要 Node 20 以上。
  建議用 nvm 裝：https://github.com/nvm-sh/nvm
  裝完重開 shell 再跑一次。"
ok "node $(node --version 2>/dev/null)"

for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  ! port_busy "$port" || die "port $port 已經有東西在用。
  可能是上一次沒關乾淨，或別人在用這台機器。
  換 port：BACKEND_PORT=8001 FRONTEND_PORT=5174 ./run-ui.sh"
done
ok "port $BACKEND_PORT / $FRONTEND_PORT 都是空的"

# GPU 不是開 UI 的必要條件，但沒有的話送出去的分析一定會失敗，先講清楚比讓人
# 對著一個跑不動的按鈕好。
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  ok "GPU：$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  warn "沒偵測到 NVIDIA GPU — 介面開得起來、切片看得到，但『開始分析』會失敗。"
fi

# ------------------------------------------------------------
# 2. 設定檔與資料夾
# ------------------------------------------------------------
say "檢查設定"

# config.py 是 gitignored 的機器專屬檔；沒有它後端連 import 都過不了。
HYBRID_CFG="$REPO/backend/algorithms/hybrid/config.py"
if [[ ! -f "$HYBRID_CFG" ]]; then
  cp "$REPO/backend/algorithms/hybrid/config_example.py" "$HYBRID_CFG"
  ok "已建立 config.py（從 config_example.py 複製）"
  warn "要真的跑分析，得先編輯它裡面的模型路徑：$HYBRID_CFG"
else
  ok "config.py 已存在"
fi

# TSGH_STORAGE_DIR 是必要的，沒設後端 import 就 RuntimeError。預設**刻意**放在
# repo 外面：以前的 <repo>/storage 退路被拿掉，就是因為它會把幾 GB 的 CZI 和
# pipeline 輸出靜靜倒進 checkout 裡（見 backend/schemas/alignment.py 的註解）。
export TSGH_STORAGE_DIR="${TSGH_STORAGE_DIR:-$HOME/tsgh_data/storage}"
export TSGH_SLIDES_DIR="${TSGH_SLIDES_DIR:-$HOME/tsgh_data/viewer}"
mkdir -p "$TSGH_STORAGE_DIR" "$TSGH_SLIDES_DIR"
ok "上傳與輸出：$TSGH_STORAGE_DIR"
ok "切片來源：  $TSGH_SLIDES_DIR"

# ------------------------------------------------------------
# 3. 相依套件
# ------------------------------------------------------------
if [[ "$SKIP_DEPS" -eq 1 ]]; then
  say "略過相依安裝（--skip-deps）"
else
  say "安裝相依套件（第一次會久，要編 valis，十幾分鐘跑不掉）"
  # --frozen：完全照 uv.lock 裝，不重解相依也不改寫 lock。uv.lock 是全組共用的，
  # 開個 UI 不該動到它。
  uv sync --frozen || die "uv sync 失敗。上面的訊息是原因。"
  ok "Python 環境就緒"

  if [[ ! -d "$REPO/frontend/node_modules" ]]; then
    (cd "$REPO/frontend" && npm ci) || die "npm ci 失敗。上面的訊息是原因。"
    ok "前端套件就緒"
  else
    ok "前端套件已存在（要重裝就刪掉 frontend/node_modules）"
  fi
fi

# ------------------------------------------------------------
# 4. 起服務
# ------------------------------------------------------------
BACK_PID=""
FRONT_PID=""
TAIL_PID=""

# 開 job control，讓每個背景工作自成一個 process group。少了這行，Ctrl-C 只會殺掉
# `npm run dev` 那一層，它底下真正在聽 port 的 vite node 行程會活下來繼續佔著
# port——下次再跑就會被自己的殘留擋住。（實測過，不是理論上的顧慮。）
set -m

# 連同子孫一起收掉：kill 一個負的 pgid = 送訊號給整個 process group。
kill_group() {
  local pid="$1" pgid
  [[ -n "$pid" ]] || return 0
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  # 抓不到 pgid（行程已經死了），或那個 group 就是這支腳本自己（job control
  # 沒生效的話會這樣）——都退回只殺單一行程，免得把自己連同呼叫端的 shell 一起帶走。
  if [[ -n "$pgid" && "$pgid" != "$$" ]]; then
    kill -TERM "-$pgid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  trap - EXIT INT TERM
  printf '\n'
  say "收工，關掉服務"
  for pid in "$TAIL_PID" "$FRONT_PID" "$BACK_PID"; do
    kill_group "$pid"
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

say "啟動後端（port $BACKEND_PORT）"
# 後端啟動要 40 秒以上而且會刷一大堆 torch / 模型載入訊息，先收進 log 檔，
# 等它活了再把後續輸出接回畫面——不然使用者會盯著滿版雜訊等。
uv run --frozen python -m uvicorn backend.main:app \
  --host 127.0.0.1 --port "$BACKEND_PORT" > "$BACKEND_LOG" 2>&1 &
BACK_PID=$!

printf '  載入模型中（約 40 秒，第一次更久）'
for ((i = 0; i < BACKEND_READY_TIMEOUT; i++)); do
  # 行程先死掉就不必再等了，直接把 log 攤出來——這是最常見的失敗，而且原因
  # 幾乎都寫在最後幾行（模型路徑不對、port 被搶走、相依缺）。
  if ! kill -0 "$BACK_PID" 2>/dev/null; then
    printf '\n'
    echo "----- 後端最後 30 行 -----" >&2
    tail -30 "$BACKEND_LOG" >&2
    # 這個錯常見到值得直接給解法：valis 在 import 時就把模型搬上 GPU，所以卡被
    # 別人佔住時後端根本起不來（跟顯示記憶體夠不夠無關）。
    if grep -q "busy or unavailable" "$BACKEND_LOG"; then
      warn "GPU 正在被其他人使用。用 nvidia-smi 看是誰，或先用 CPU 開起來看介面："
      warn "  CUDA_VISIBLE_DEVICES= ./run-ui.sh --skip-deps"
    fi
    die "後端啟動失敗，完整記錄在 $BACKEND_LOG"
  fi
  if port_busy "$BACKEND_PORT"; then
    printf '\n'
    ok "後端就緒"
    break
  fi
  printf '.'
  sleep 1
done
port_busy "$BACKEND_PORT" || die "後端等了 ${BACKEND_READY_TIMEOUT} 秒還沒回應，記錄在 $BACKEND_LOG"

# 活了之後才把 log 接回畫面，加前綴才不會跟 Vite 的輸出混在一起分不出來。
tail -f "$BACKEND_LOG" 2>/dev/null | sed 's/^/[後端] /' &
TAIL_PID=$!

say "啟動前端（port $FRONTEND_PORT）"
(cd "$REPO/frontend" && npm run dev -- --port "$FRONTEND_PORT" --strictPort) &
FRONT_PID=$!

for ((i = 0; i < 60; i++)); do
  port_busy "$FRONTEND_PORT" && break
  kill -0 "$FRONT_PID" 2>/dev/null || die "前端啟動失敗，訊息在上面。"
  sleep 1
done

URL="http://localhost:$FRONTEND_PORT"
printf '\n\033[1;32m════════════════════════════════════════════\033[0m\n'
printf '\033[1;32m  介面已就緒： %s\033[0m\n' "$URL"
printf '\033[1;32m  按 Ctrl-C 關閉\033[0m\n'
printf '\033[1;32m════════════════════════════════════════════\033[0m\n\n'

if [[ "$OPEN_BROWSER" -eq 1 ]]; then
  # wslview 是 WSL2 用的（會開 Windows 那邊的瀏覽器）；開不起來不是錯誤，
  # 上面已經把網址印出來了。
  for opener in wslview xdg-open open; do
    command -v "$opener" >/dev/null 2>&1 && { "$opener" "$URL" >/dev/null 2>&1 & break; }
  done
fi

# 等到任一個服務結束（通常是使用者按 Ctrl-C），cleanup 會收掉另一個。
wait "$FRONT_PID"
