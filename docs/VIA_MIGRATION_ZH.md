# VIA Converter 遷移結果

更新日期：2026-09-03

VIA CSV reader、label normalization、metadata merge、subset assignment 與 canonical master conversion 已由 `scripts/convert_via_to_master.py` 遷入 `src/nuyl_sushi/data/`。舊腳本仍保留原本的命令列參數，安裝 package 後也可使用 `nuyl-sushi-convert-via`。

## 模組責任

- `nuyl_sushi.data.via`：解析 VIA comment header、JSON-like 欄位、時間區段與 label。
- `nuyl_sushi.data.conversion`：整合影片 metadata、建立 canonical records、分配 split 與統計。
- `nuyl_sushi.cli.convert_via`：參數解析及四個輸出檔案的寫入。
- `scripts/convert_via_to_master.py`：舊工作流程的相容入口。

## Golden fixture

`tests/fixtures/via/` 保存一組不含真實影像或個資的小型測試資料，涵蓋：

- VIA `# CSV_HEADER` 格式。
- JSON 與 Python-literal 形式欄位。
- list、`start/end`、`from/to` 時間格式。
- 空白與底線 label normalization。
- `master` / `novice` skill aliases。
- metadata 以 filename 或 video ID 對應。
- CLI 的 master、label maps 與 stats 四項輸出。

## 真實資料 parity

新版 converter 已用 D 槽的 99 組 VIA CSV 重新產生 master，結果與遷移前保存的 JSON 做完整物件比較：

- Semantic equality：true
- Videos：99 / 99
- Segments：1,232 / 1,232
- Coarse labels：18 / 18
- Fine labels：58 / 58
- Split：train 69、validation 20、test 10
- Validation：0 errors、99 個已知 unknown-skill warnings

這項 parity 只證明遷移沒有改變舊行為。目前在缺少 subject metadata 時，舊行為會退回 video-level split；它不能作為正式研究評估 split。subject-wise split 仍須等匿名 subject 對照完成後重新產生。

## 下一步

依 canonical-only 原則，下一批將遷移 Task A clip manifest 與 Task B ActivityNet exporter。兩者都要讀取 `MasterDataset`，不得重新解析 VIA CSV。

