---
trigger: always_on
---

# 開發準則

## 環境配置

- **Python venv**: `/home/hispadmin/tsgh/.venv`
- **Python version**: 3.11.14
## 1. 架構優先 (Architecture First)

- 嚴禁在未產出「實作計畫」前撰寫業務邏輯代碼
- 新模組或重大演算法修改前，須先以 `.drawio` 輸出架構圖，並以 `Markdow` 解釋設計
- 等待使用者回覆「Approve」後方可執行

## 2. API 與文檔治理

- 調用第三方庫前，須透過 Browser Tool 查閱最新官方文件
- 嚴禁使用過時 API，若文檔與訓練數據衝突，以官方最新文檔為準
- 寫 Code 前須確認環境安裝的庫版本，確保語法相容
- 移除不必要的 `print` 語句，保持代碼簡潔

## 3. 程式碼質量標準

- **語意化命名**：`her2_image_path` 優於 `img1`
- **單一職責**：單個函式不超過 50 行
- **強制 Type Hints**：所有函式須標註輸入輸出型別
- **Google Docstrings**：每個 Class/Function 須包含參數說明

## 4. 醫療影像約束

- 禁止將 WSI 全圖載入記憶體，使用 `PyVips` Tile-based 處理
- 座標須明確標註單位（`pixels` / `microns`）

## 5. 錯誤處理

- 報錯時自動分析 Traceback，結合文檔修復並說明原因
