# Canonical Data Model

更新日期：2026-09-03

`src/nuyl_sushi/domain/models.py` 現在是資料格式的單一 Python 表達層。模型刻意使用標準函式庫 dataclass，不綁定 FastAPI、Pydantic、訓練框架或特定標註工具。

## 核心類型

- `SegmentAnnotation`：起訖秒數、coarse label、fine label。
- `VideoRecord`：影片識別、媒體資訊、split、subject/session/view/skill、輔助 modality 與 annotations。
- `MasterDataset`：schema version、dataset metadata、label space 與影片索引。
- `DatasetSplit`：train、validation、test 的 video ID 集合。
- `ValidationIssue`：具有 severity、code、location、message 的可機器處理錯誤。

讀取與輸出時會保留尚未納入正式 schema 的額外欄位，讓舊研究產物可以漸進遷移；但 `videos`、每筆 video record 與 `annotations` 的容器型別會嚴格檢查，不會靜默略過損壞資料。

## 驗證規則

目前驗證器會檢查：

- dataset version、filename 與 dictionary key / `video_id` 一致性。
- 影片路徑是否存在（可關閉，供 CI fixtures 使用）。
- duration、segment 起訖、超出影片長度與完全重複片段。
- coarse/fine label 是否存在於 canonical label space。
- subset 合法值，以及已知 subject 是否跨越 train/val/test。
- 缺少 annotations 與未知 skill level。

執行真實資料：

```powershell
python scripts/validate_master_dataset.py `
  --master-json artifacts/data_audit/master/master_annotations.json `
  --no-check-files
```

也可以在安裝 package 後執行：

```powershell
nuyl-sushi-validate --master-json path/to/master_annotations.json
```

目前真實 canonical master 的結果是 99 支影片、1,232 個片段、0 errors、99 warnings；warnings 全部來自尚未有可靠依據的 `skill_level=unknown`。

## 下一步

下一個遷移單位是 VIA reader/converter。轉換器必須輸出這個 canonical model，並以 golden fixtures 證明新舊轉換結果語意相同；之後 ActivityNet、Task A 與其他格式都只能由 canonical model 匯出。

