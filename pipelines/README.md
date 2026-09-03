# Pipelines

重構後的資料與模型流程會依責任遷入下列目錄：

- `prepare/`：VIA 讀取、驗證、canonical export 與模型格式轉換。
- `train/`：訓練入口與可重現設定。
- `evaluate/`：Task A、Task B、skill assessment。
- `export/`：報表、模型 manifest 與發布產物。

遷移期間 `scripts/` 保留相容入口；每個入口應只負責參數解析與呼叫 package/pipeline API。

