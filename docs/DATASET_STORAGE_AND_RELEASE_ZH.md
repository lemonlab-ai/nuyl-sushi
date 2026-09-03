# Dataset 儲存、壓縮與發布策略

更新日期：2026-09-03

## 決策摘要

NUYL Sushi 的資料不應只分成「原檔」與「壓縮檔」，而應分為三層：

1. **Archive originals（不可變原始封存）**：保存拍攝當下的完整位元組，不改名、不轉碼、不覆寫。
2. **Canonical working set（私有工作集）**：標註、清單、必要的 720p 訓練影片與去識別化 metadata。
3. **Release derivatives（可重建衍生版）**：經同意審查後，才產生研究版、預覽版或串流封裝。

Git repositories 只管理程式、標註、manifest、checksum 與資料集文件。大型影音本體放在 object storage 或離線磁碟，不把 GitHub LFS 當唯一封存位置。

## 目前盤點與判斷

- D 槽來源樹約 75 GB，視為原始封存候選，不做原地整理或轉碼。
- `NUYLSushi-1M/videos` 有 114 支 MP4，其中 99 支已有標註。
- 114 支影片合計約 2.26 GiB、3.83 小時，皆為 1280×720，平均碼率約 1.34 Mbps。
- 現有 MP4 已經是有損壓縮媒體；一般 ZIP/7z 幾乎不會再顯著縮小。
- 現有 720p 工作集已相當精簡。第一選擇是檢測後直接沿用；只有不符合發布規格的檔案才轉碼，避免不必要的 generation loss。

## 儲存配置

### A. 原始封存層

用途：災難復原、未來重新製作、證據與來源追溯。

- 保留 D 槽現有資料為唯讀來源；所有後續流程讀取但不修改它。
- 建立第二份離線或異地副本，例如外接硬碟或 NAS snapshot。
- 建立一份加密的 cloud object archive；可採 S3 相容儲存，低頻存取資料再進 cold/deep archive。
- 對每個檔案保存相對路徑、byte size、SHA-256、採集日期與媒體資訊。
- 啟用 bucket versioning；封存完成後進行 checksum 抽查與一次還原演練。

原始封存不等於公開資料集，也不因為有備份而自動取得發布授權。

### B. 私有工作層

用途：標註修正、訓練、評估與 demo 開發。

- `nuyl-sushi-dataset`：只放可版本控制的小型資料，包括 VIA/canonical annotations、manifest、taxonomy 與資料文件。
- 大型影片：放在本機/NAS 或 private object bucket，透過 manifest 對應，不直接 commit。
- `subject_id`、技術程度、同意範圍等敏感表格放在權限更嚴格的位置，不和一般開發資料混放。
- 感測器原始資料含 GPS；未去識別化前不得進 repository 或一般 release bucket。

### C. 發布衍生層

用途：核准後的研究重現、下載或網頁預覽。

- `research-720p`：主要 ML/research 版本。上限 1280×720、保留來源 fps、H.264、`yuv420p`、fast-start；符合規格者直接複製，不重壓。
- `preview-360p`：只供網頁 demo/人工瀏覽。上限 640×360、最多 15 fps、移除音訊；不得混入正式訓練或評估。
- `webdataset`：只有需要遠端串流訓練時才把發布檔封裝成約 1–2 GiB shards；114 支影片目前也可以直接以個別 MP4 發布。

音訊是否移除必須先查 consent；移除音訊不代表影片已匿名化，臉部、衣著、環境與動作仍可能識別個人。

## 壓縮原則

壓縮流程採 **probe → decide → transcode/copy → verify → manifest**：

1. 用 `ffprobe` 取得 codec、解析度、fps、duration、audio streams 與 bitrate。
2. 依 profile 判斷 `copy`、`transcode` 或 `reject`，預設 dry-run。
3. 寫入新的 derivative 目錄，絕不覆寫原檔。
4. 驗證可解碼、duration 容差、輸出解析度、frame count/時間基準與 checksum。
5. 產生 release manifest，記錄來源及衍生檔 SHA-256、工具版本、profile、命令、執行時間與驗證結果。

初始轉碼參數只是可重現的起點，不是盲目套用的最終品質標準：

| Profile | 建議設定 | 用途 |
|---|---|---|
| `research-720p` | H.264 CRF 21、preset slow、最高 1280×720、保留 fps | 訓練與研究交換 |
| `preview-360p` | H.264 CRF 28、preset medium、最高 640×360、最高 15 fps、無音訊 | Web 預覽 |

在全量執行前，以不同動作、光線與視角抽取 6–10 支代表影片做 pilot，檢查畫面細節、模型指標與縮減比例。若現有 720p 檔已符合規格，`research-720p` 應直接採用，不因流程存在就重壓。

## 版本與發布閘門

每個 dataset release 使用不可變版本，例如 `v0.1.0-private`、`v1.0.0`，並包含：

- dataset card、授權/存取條件、致謝方式與已知限制；
- annotations、label maps、split definitions 與 release manifest；
- source-to-derivative 對照與 checksum；
- consent decision、privacy review、GPS/audio/identity 檢查結果；
- 由乾淨環境重建衍生版的命令與工具版本。

只有下列條件全部通過才可公開：同意範圍明確、個資審查完成、subject-wise split 可重現、checksum 驗證成功、dataset card 完成。若授權只允許受控研究使用，則維持 private/gated，而不是製作公開壓縮版。

## 建議平台角色

- **GitHub**：程式、標註、manifest、文件；不承擔 75 GB 原始影音封存。
- **Private object storage**：原檔備份與工作影片的主要遠端位置。
- **Hugging Face Dataset**：日後核准的研究衍生版，或受控/private 的可下載資料集；搭配 dataset card。
- **Zenodo**：正式穩定版本的 DOI snapshot，可放 annotations、manifest、論文附件或大小合適的 release，不作日常工作區。

## 納入重構的順序

1. 鎖定 D 槽來源為唯讀並完成全量 checksum manifest。
2. 補齊 consent、匿名 subject metadata 與敏感資料存取規則。
3. 實作 media probe、profile 判定與 dry-run release plan。
4. 對代表樣本執行壓縮 pilot，量測大小、畫質與模型指標。
5. 產生 private `research-720p` working release，驗證 annotations 對齊。
6. subject-wise split 完成後再啟動正式訓練/評估遷移。
7. 公開前才建立 preview、streaming shards、dataset card 與 DOI snapshot。

轉碼設定的版本控制入口為 [`configs/datasets/media_profiles.toml`](../configs/datasets/media_profiles.toml)。實際影片路徑、cloud credentials 與個資不寫入設定檔。

## Dry-run planner

目前先提供唯讀規劃工具；它會寫出 `release_plan.csv/json`，但不會執行 FFmpeg 或改動來源：

```powershell
python scripts/plan_dataset_release.py `
  --video-manifest C:\path\to\video_manifest.csv `
  --source-dir C:\path\to\videos `
  --profile research-720p `
  --output-dir artifacts\release_plans\research-720p
```

加上 `--verify-checksum` 才會重新讀取全部影片並核對 SHA-256。若找不到 `ffprobe`，工具仍會產生計畫，但缺少 codec、pixel format 或 audio 資訊的檔案會標成 `blocked`，不會猜測其合規性。

### 2026-09-03 實際執行

- 範圍：99 支 canonical 已標註影片。
- SHA-256：99/99 與既有 manifest 相符。
- 影像：99/99 為 H.264、`yuv420p`、1280×720、24 fps。
- 音訊：99/99 各有 1 條 audio stream。
- 初次決策：99 支均為 `blocked`，唯一原因是 `audio_consent_review_required`。
- 後續決策：專案負責人指定移除音訊，改用 `research-720p-no-audio` profile。
- 結果：99/99 以 video stream copy remux 完成，輸出驗證為零音訊且影像參數不變。
- 容量：2,325,245,142 bytes 降為 1,796,316,619 bytes，約減少 22.7%。

影像本身符合 `research-720p`，沒有重新壓縮。衍生影音位於 Git 之外的 `NUYL-Sushi-Derived/v0.1.0-private/research-720p-no-audio`；release manifest 則進入 private dataset repository。這項音訊處理決策不等同公開發布 consent，後者仍維持 unresolved。

可重建命令：

```powershell
python scripts/strip_release_audio.py `
  --video-manifest C:\path\to\video_manifest.csv `
  --source-dir C:\path\to\videos `
  --output-dir D:\path\to\v0.1.0-private\research-720p-no-audio `
  --dataset-version v0.1.0-private `
  --execute
```

工具執行前會核對每支來源 SHA-256，只接受安全的 remux 決策；目的檔以 partial file 完成後才原子改名，並重新 probe 與計算 derivative SHA-256。

### 2026-09-03 原始來源 archive inventory

- 範圍：`NUYL SUSHI` 完整來源樹。
- 檔案：1,233。
- 容量：80,737,894,646 bytes（75.193 GiB）。
- SHA-256：1,233/1,233 完成，零錯誤。
- Tree SHA-256：`37bd73709d894dad683a84a2b1919212cec5d0fcf2155fa87f39f796b7c4ba47`。
- 相同內容 hash groups：199，但多餘容量只有 850,974 bytes，主要是小型 annotation copies；沒有值得冒風險刪除的影片級重複檔。

Inventory 工具支援逐檔 checkpoint 與續跑：

```powershell
python scripts/inventory_source_archive.py `
  --source-root D:\path\to\NUYL-SUSHI `
  --archive-id nuyl-sushi-original-2026-09-03 `
  --output-dir artifacts\archive_inventory\original-2026-09-03
```
