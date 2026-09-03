# D 槽來源資料稽核

稽核日期：2026-09-03

來源：`D:\Project\ntnu-sushi\NUYL SUSHI`

本次只讀取 D 槽；canonical master、checksums 與報告輸出在本專案的 `artifacts/data_audit/`，該目錄不進 Git。

目前專案原有的 98 組 VIA 檔案與 D 槽內容相同（CSV 僅換行格式不同）；D 槽多出的第 99 組 `VID_20230208_122207_1_09_Making-Nigiri Sushi.csv/json` 已補入本專案。

## 已確認

| 項目 | 結果 |
|---|---:|
| VIA CSV | 99 |
| VIA JSON | 99 |
| Canonical video records | 99 |
| Canonical action segments | 1,232 |
| Coarse labels | 18 |
| Fine labels | 58 |
| `NUYLSushi-1M/videos` MP4 | 114 |
| Dataset video size | 2,423,166,715 bytes |
| 標註引用且存在 | 99 / 99 |
| 標註引用但缺檔 | 0 |
| 未引用影片 | 15 |
| 零位元影片 | 0 |
| SHA-256 完成 | 114 / 114 |
| 重複影片 hash groups | 0 |
| OpenCV boundary-frame probe 通過 | 114 / 114 |
| Sensor captures | 39 |
| 完整含 Motion/Gravity/GPS | 39 / 39 |

## Metadata 狀態

99 支 canonical videos 中：

- 可由原始來源直接判斷 `view1`：32
- 可由原始來源直接判斷 `view2`：35
- 只有 edited source、視角待補：32
- `subject_id` 未知：99
- `skill_level` 未知：99
- sensor ↔ video mapping：0

因此現在可以安全做資料轉換與單檔測試，但還不能宣稱 subject-wise 評估成立。現行 converter 在 subject 未知時會退回 video-level split，這不足以避免同一受試者跨 train/test。

## Source-of-truth 決定

新的資料核心採用：

- 標註：`NUYLSushi-1M/via-annotations` 的 99 組 VIA CSV。
- 影片：`NUYLSushi-1M/videos` 的 114 支 MP4。
- Canonical set：99 支有標註影片。
- Candidate/unlabeled set：15 支未引用影片，確認用途前不加入訓練。
- Sensor：保留為 private auxiliary modality，完成時間對齊前不進 baseline。

舊 `NUYLSushi-1M/annotations/annotation.json` 不作 source of truth。它的 taxonomy 有大量重複節點，segment serialization 也不符合新的 canonical contract。

## 隱私與發布限制

- 39 組 sensor capture 全部含 GPS 經緯度。
- GPS 不得進公開 dataset；後續 metadata 只保留必要的相對時間或已去識別化場次。
- 原始影片、訪談音訊、照片與師傅身份相關資料，在取得公開展示/散布同意前保持 private。

## 尚待人工確認

1. `20221216.xlsx` 的內容；目前指定的 spreadsheet runtime 不可用，尚未解析。
2. 拍攝日期/場次對應到哪位 subject。
3. 32 支 edited-only videos 的 view。
4. 39 組 sensor captures 與 video time ranges 的對應。
5. 15 支 unreferenced videos 是暖身、失敗片段、未標註資料或應排除資料。
6. skill level 是否真的有跨級別受試者資料；不可只靠拍攝日期推定。

## 已產生的稽核輸出

```text
artifacts/data_audit/
  REPORT_ZH.md
  summary.json
  video_manifest.csv
  missing_references.csv
  unreferenced_videos.csv
  sensor_manifest.csv
  metadata_worklist.csv
  master/
    master_annotations.json
    label_map_coarse.txt
    label_map_fine.txt
    stats.json
```

## 下一步

先建立一份人工可補值的 metadata mapping，欄位至少包含：

`video_id, subject_id, session_id, view_type, skill_level, sensor_capture_id, consent_scope, notes`

只有 `subject_id` 與資料使用權限完成後，才進入正式 k-fold baseline。

## 重跑方式

```powershell
python scripts/convert_via_to_master.py `
  --csv-dir "D:\Project\ntnu-sushi\NUYL SUSHI\NUYLSushi-1M\via-annotations" `
  --video-dir "D:\Project\ntnu-sushi\NUYL SUSHI\NUYLSushi-1M\videos" `
  --output-dir artifacts/data_audit/master

python scripts/audit_source_dataset.py `
  --source-root "D:\Project\ntnu-sushi\NUYL SUSHI" `
  --master-json artifacts/data_audit/master/master_annotations.json `
  --output-dir artifacts/data_audit
```

若環境有 `ffprobe` 會優先使用；否則使用 OpenCV 檢查影片 metadata、第一幀與最後一幀。
