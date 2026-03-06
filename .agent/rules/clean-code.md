---
trigger: always_on
---

## Python Clean Code Rules

### 1. 跨平台路徑管理 (Path Independence)

* **原則：** 嚴禁硬編碼 (Hard-coding) 字串路徑，禁止使用 `+` 或 `f-string` 拼接路徑。
* **實作：** 必須統一使用 `pathlib.Path`。
* **規範：** * 生成的程式碼應能自動偵測作業系統環境。
* 涉及目錄跳轉必須使用 `path.parent` 或 `path.joinpath()`。
* 路徑物件應在程式入口處初始化，而非在函數深處生成。



### 2. 配置分離模式 (Config Pattern)

* **原則：** 環境變數與演算法邏輯分離。
* **實作：** * 建立 `config_example.py`：包含所有路徑設定（如 `DATA_DIR`, `OUTPUT_DIR`）、模型參數（如 `THRESHOLD`）的預設值與範例。
* 建立 `config.py` 加載邏輯：主程式應 `try: import config`，若失敗則拋出明確提示，引導使用者複製範例檔。


* **Agent 規範：** 生成新功能時，若涉及新參數，必須同步更新 `config_example.py` 並在程式中使用該變數。

### 3. 靜默與日誌規範 (Logging over Print)

* **原則：** 演算法開發不應產生大量無意義的 `print`，應具備可追蹤性。
* **實作：** * 禁止使用 `print()` 輸出中間過程。



* **Agent 規範：** 生成的迴圈邏輯（如處理幾百張 DICOM/TIFF 時）嚴禁在迴圈內 `print`。

### 4. 程式碼品質 (Code Quality)

* **Type Hinting：** 強制要求所有函數定義必須包含類型標記（特別是 NumPy ndarray 或 Tensor 的維度描述）。
* *範例：* `def segment_roi(image: np.ndarray) -> np.ndarray:`


* **Docstrings：** 使用 Google Style 說明演算法的數學邏輯或輸入輸出的影像規格。


---