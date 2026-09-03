# NUYL Sushi 全專案重構順序

更新日期：2026-09-03

## 最終目標

將目前以研究腳本為中心的專案，重構為一個共用核心、三種入口的 monorepo：

```text
                       +------------------+
                       | src/nuyl_sushi   |
                       | schema + domain  |
                       +---------+--------+
                                 |
             +-------------------+-------------------+
             |                   |                   |
        pipelines/           apps/api            apps/research
      train/evaluate       production API         Streamlit tools
                                 |
                             apps/web
```

正式 demo 的主要流程是：影片上傳 → 非同步推論 → 標準化 prediction result → 時間軸呈現。研究 UI 與訓練程式也使用同一套 core，不再各自解析 JSON。

## 目標目錄

```text
NUYL-Sushi/
  apps/
    api/
    web/
    research/
  src/nuyl_sushi/
    domain/              # canonical schemas、label taxonomy
    data/                # VIA readers、validation、split、export adapters
    inference/           # predictor protocol、registry、pre/post-processing
    evaluation/          # Task A/B、skill-level metrics
    gesture/             # MediaPipe、MANO adapter、DTW comparison
  pipelines/
    prepare/
    train/
    evaluate/
    export/
  configs/
    datasets/
    models/
    runtime/
  tests/
    unit/
    integration/
    contract/
    fixtures/
  data/
    README.md
    metadata/
    samples/
  docs/
    architecture/
    dataset-card/
    model-cards/
    acknowledgements/
  legacy/
  pyproject.toml
```

## 確定的重構順序

目前進度：第 0 階段的自動化資料稽核已完成。結果見 `docs/DATA_AUDIT_D_DRIVE_ZH.md`；subject、skill level、sensor mapping 與 consent 仍需人工資料才能閉合。

### 0. 建立可回歸的現況基準

先不搬任何 Python module，固定目前能跑的行為。

工作：

- 保留 `python scripts/ci_smoke.py` 作為總體 smoke test。
- 產生完整資料盤點：video、annotation、subject、view、skill level、missing files。
- 將目前重要輸出 schema 存成 golden fixtures。
- 把 mock/perfect prediction 明確標成測試資產，避免和真實實驗結果混用。
- 補上 code/data license、拍攝同意與公開範圍決策。

完成條件：乾淨環境能跑 smoke；98 個 annotation records 都有明確的 present/missing 狀態；後續重構有可比較基準。

### 1. 建立 Python package 骨架

這是第一個程式碼重構，不先碰模型。

工作：

- 新建 `pyproject.toml` 與 `src/nuyl_sushi`。
- 建立設定、logging、path handling 與例外類型。
- 將散落的 `ROOT_DIR`、相對路徑與 Windows-specific wrapper 收斂。
- 舊 CLI 暫時保留，改成呼叫 package function 的 compatibility wrapper。

完成條件：package 可安裝；舊命令仍可執行；程式不依賴目前工作目錄才能找到資料。

### 2. 先重構 canonical data layer

資料格式是模型、API 與前端共同依賴，必須最先穩定。

工作：

- 定義 `VideoRecord`、`SegmentAnnotation`、`DatasetSplit`、`ModelManifest`、`PredictionResult`。
- VIA 只能轉入 canonical schema；ActivityNet、Kinetics、MMAction2、ActionFormer 只能由 canonical schema 匯出。
- 建立 label alias/taxonomy version、deterministic video ID 與 checksum。
- 驗證時間區段、缺檔、重複 annotation、未知 label 與 subject leakage。
- 將 raw/private、metadata、samples、generated artifacts 分開。

完成條件：相同輸入永遠產生相同輸出 checksum；所有 exporter 有 golden tests；subject 不跨資料切分。

### 2.5. 建立 dataset 儲存與 release preparation

在大量訓練或 demo 開發前，先把原始封存、私有工作集與可發布衍生版分開。

工作：

- D 槽約 75 GB 來源維持唯讀，建立全量 SHA-256 manifest、第二份離線副本與加密遠端封存。
- Git repositories 只保存程式、標註、manifest 與文件；影音本體由 object storage/NAS 管理。
- 建立 `research-720p` 與 `preview-360p` profiles，以及 probe、dry-run、轉碼、驗證與 release manifest 流程。
- 現有 114 支 720p MP4 先做 compliance probe；符合規格者直接沿用，避免二次有損壓縮。
- 先以 6–10 支代表影片做 pilot，比較檔案大小、畫面細節與模型指標，再決定是否全量轉碼。
- consent、個資、音訊與 GPS 審查未完成前，不建立公開 release。

完成條件：任一 derivative 都能從 immutable source 與受版控設定重建；source/derivative checksum 可追溯；原檔未被改動。完整決策見 [`DATASET_STORAGE_AND_RELEASE_ZH.md`](DATASET_STORAGE_AND_RELEASE_ZH.md)。

### 3. 建立正式測試與 CI

不能等全部搬完才補測試。

工作：

- `tests/unit`：schema、parser、label、metrics。
- `tests/integration`：VIA → master → Task A/B exports。
- `tests/contract`：API request/result schema。
- `tests/fixtures`：5–10 個合法的小型 sample，不使用完整私有影片。
- 將 smoke test 拆成快速 CI 與需要 GPU/影片的 nightly/local profile。

完成條件：CPU CI 不需要私有資料或外部 model repositories；錯誤資料會以可讀訊息失敗。

### 4. 統一 inference boundary

先定義模型如何被使用，再決定 API 如何呼叫它。

工作：

- 建立統一 `Predictor` protocol：load、predict、health、metadata。
- Task A、Task B、Gesture Coach 各自成為 adapter。
- 每個可用模型綁定 checkpoint checksum、label map、config、commit 與 metrics。
- 固定一個 Task A baseline、一個 Task B 主線；其他模型降為 research benchmark。
- 統一輸出 `PredictionResult`，包含 segment、label、score、latency、warnings。

完成條件：同一 sample 可透過 Python API 得到穩定結果；沒有模型格式洩漏到 UI。

### 5. 整理訓練與評估 pipeline

在 inference contract 穩定後，再整理大量研究腳本。

工作：

- `scripts/convert_*`、`export_*` → `pipelines/prepare`。
- `train_*` 與 `runners/` → `pipelines/train`。
- `eval_*`、`summarize_*` → `pipelines/evaluate`。
- paper tables 與 handoff → `pipelines/export`。
- configuration 與執行邏輯分離；保留薄 CLI，不在 PowerShell 中承載研究邏輯。

完成條件：單一設定檔可重現 split、train、evaluate 與報表；輸出包含環境、seed、資料與模型版本。

### 6. 重寫 FastAPI service

舊 server prototype 不直接搬入，只保留需求參考。

API：

- `GET /api/v1/health`
- `GET /api/v1/models`
- `POST /api/v1/jobs`
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs/{job_id}/result`
- `DELETE /api/v1/jobs/{job_id}`

工作：影片類型/大小限制、暫存期限、非同步 job、結構化錯誤、SSE progress、API contract tests。MVP 先用單機 queue，確定需要多人/GPU server 才加入 Redis worker。

完成條件：固定 sample 從 upload 到 result 全程通過；錯誤或取消不留下永久暫存檔。

### 7. 重寫 Web demo

API 完成後才寫正式前端，避免再次出現 client/server 契約不一致。

技術：Vue 3 + Vite + TypeScript；第一版不使用 Electron。

畫面：上傳、進度、影片播放器、segment timeline、label/score、模型資訊、JSON/CSV export、Gesture Coach comparison，以及師傅致謝與限制說明。

完成條件：前端只依賴 OpenAPI/schema generated types；非開發者能依 README 完成一次 demo。

### 8. 將 Streamlit 降為 research app

工作：

- 移到 `apps/research`。
- 移除重複的資料解析、模型載入與 metric logic，全部呼叫 core。
- 保留實驗診斷功能，不把 Streamlit 當正式產品 UI。

完成條件：Web 與 Streamlit 對同一影片、同一模型產生一致 prediction payload。

### 9. 移除 legacy 與發布

工作：

- 建立 legacy parity checklist，確認需求均已實作或明確放棄。
- 刪除 `legacy/server-prototype`、`legacy/client-prototype`、`legacy/via-converter`。
- 刪除到期 compatibility wrappers。
- 完成 architecture docs、dataset card、model cards、acknowledgements、Docker/啟動方式。
- 建立新的 code repository；若資料可發布，再獨立建立 dataset repository。

完成條件：`legacy/` 為空並移除；乾淨機器可以重建 demo；公開檔案均有授權與可追溯來源。

## 不可顛倒的依賴

```text
baseline
   ↓
package → canonical data → dataset preparation → tests/CI
                                                  ↓
                       inference contract
                          ↙          ↘
                 training/eval       API
                                        ↓
                                       Web
                                        ↓
                          research UI cleanup
                                        ↓
                              legacy removal/release
```

最重要的三條規則：

1. 不在 canonical data schema 前改寫模型輸入。
2. 不在 inference/API contract 前開寫正式 Web。
3. 不在 parity test 前刪除 legacy 原始碼。

## 建議的實作批次

每個批次保持可測試、可獨立 review：

1. `baseline-and-data-audit`
2. `python-package-foundation`
3. `canonical-data-model`
4. `dataset-storage-and-release-preparation`
5. `data-pipeline-tests`
6. `inference-contract-and-registry`
7. `training-pipeline-migration`
8. `fastapi-job-service`
9. `vue-web-demo`
10. `streamlit-core-migration`
11. `legacy-removal-and-release`

第 1 批「現況基準與資料盤點」已完成。下一個工程批次確定為 `python-package-foundation`；subject-wise split 則要等匿名 subject metadata 補齊後再啟用。

2026-09-03 進度：99 支 canonical 影片的 session 與 view 候選已完整建立，sensor 候選維持人工審查。

同日進度：`python-package-foundation` 已完成，canonical dataclass 與第一版 validation contract 已建立。下一個遷移單位為 VIA reader/converter 與 golden fixtures。

VIA reader/converter 與 golden fixtures 已完成；新版對 99 支影片、1,232 個片段的輸出與舊版 master 語意完全相同。下一個批次改為 Task A/B canonical exporters。
