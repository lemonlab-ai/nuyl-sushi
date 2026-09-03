# 中繼資料補全與人工確認流程

更新日期：2026-09-03

這一階段的目的，是把能由檔案 lineage 客觀判斷的欄位先補齊，同時把需要人判斷的資料留成明確工作清單。工具不會猜測師傅身分、熟練度、公開同意範圍，也不會把時鐘接近的 sensor capture 直接認定為同步資料。

## 目前結果

99 支 canonical 已標註影片的 session 均已確認。視角方面，67 支可由來源路徑直接判定，另外 32 支可由原始母檔或相機檔名家族推定；完成後為 view1 45 支、view2 54 支、未解析 0 支。

推定規則如下：

| Edited filename | View | 依據 | 信心 |
|---|---|---|---|
| `221210_01.mp4` 至 `221210_15.mp4` | view2 | 2022-12-10 唯一原始相機檔位於 `view2/221210.mp4` | medium |
| `IMG_6421_*.mp4`、`IMG_6423_*.mp4` | view2 | 母檔家族位於 2023-02-08 `view2` | high |
| `VID_20221215_192928.mp4` | view1 | 對應原檔為 `view1/VID_20221215_192928_1.mp4` | high |
| `VID_20230208_122207_1_*.mp4` | view1 | 由 `view1/VID_20230208_122207_1.mp4` 分段 | high |

39 支影片名稱帶有可解析的錄影時間，因此工具為每支列出同場次時間最接近的三筆 sensor capture。這些只是候選：相機與手機時鐘可能有偏差，且 2023-02-08 的 12 支分段影片共用母檔時間戳，不能靠檔名分別配對。

## 執行方式

先完成資料稽核，再建立候選：

```powershell
python scripts/audit_source_dataset.py `
  --source-root "D:\Project\ntnu-sushi\NUYL SUSHI" `
  --master-json artifacts/data_audit/master/master_annotations.json `
  --output-dir artifacts/data_audit

python scripts/build_metadata_candidates.py
```

輸出位於 `artifacts/data_audit/metadata/`：

- `metadata_candidates.csv`：session、view、evidence、confidence，以及待補的人工欄位。
- `sensor_link_candidates.csv`：每支影片最多三筆時間鄰近候選，全部為 `review_required`。
- `label_distribution.csv`：18 個 coarse、58 個 fine 標籤的片段數、影片數及標註秒數。
- `summary.json` 與 `REPORT_ZH.md`：摘要與限制。

## 人工處理順序

1. 先建立匿名 `subject_id` 對照；不可從姓名或影像自行猜測。
2. 確認每個拍攝 session 與 subject 的關係，同一人跨場次必須使用同一 ID。
3. 用 sensor 檔的實際起訖時間、動作波形或拍攝紀錄確認配對；確認前不要填 `sensor_capture_id`。
4. 定義 `skill_level` 的來源與評分標準。若原研究沒有可靠分級，保留 unknown，改把 skill assessment 視為未完成的後續研究問題。
5. 逐一確認 `consent_scope`，區分內部研究、展示 demo、公開衍生特徵與公開原始影像。
6. subject metadata 完成後才建立 subject-wise train/validation/test split，避免同一師傅同時出現在訓練與測試資料。

## 目前阻擋項目

資料轉換、驗證、視角分析與不依賴 subject 的 baseline 可以繼續；正式 subject-wise 評估仍被缺少的 `subject_id` 阻擋。GPS 必須維持 private，公開或分享 sensor 資料前要先移除位置資訊。
