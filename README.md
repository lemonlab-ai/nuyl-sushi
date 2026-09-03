# NUYL Sushi

Repositories:

- Public code and documentation: [lemonlab-ai/nuyl-sushi](https://github.com/lemonlab-ai/nuyl-sushi)
- Private annotations and metadata: [lemonlab-ai/nuyl-sushi-dataset](https://github.com/lemonlab-ai/nuyl-sushi-dataset)

Raw videos, sensor streams, GPS data, and participant metadata are not stored in the public repository.

NUYL Sushi 是一套針對壽司料理影片的動作理解專案，目標是完成：

- 料理動作分類（Task A）
- 未剪輯影片中的動作時間定位（Task B）
- 手部關鍵點與參考動作相似度比較（Gesture Coach）
- 可重現的資料轉換、訓練、評估與展示流程

目前專案已整併為單一工作樹。`legacy/` 保存三套早期 prototype 的純原始碼，不再各自作為 repository 維護。

## 目前入口

```powershell
python scripts/ci_smoke.py
```

研究流程、模型 wrapper 與既有命令仍維持原路徑，以確保重構期間可以持續驗證：

- `scripts/`：資料、實驗與評估腳本
- `runners/`：外部模型執行 wrapper
- `apps/`：目前的 Streamlit 與 Gesture Coach
- `templates/`：模型與實驗設定模板
- `data/`：目前資料與 metadata
- `docs/LEGACY_RESEARCH_GUIDE.md`：整併前的完整研究操作說明

## 重構方向

最終會整理成：

```text
apps/                  # api、web、research 三種介面
src/nuyl_sushi/        # 唯一共用 Python domain/data/inference 核心
pipelines/             # prepare、train、evaluate、export
configs/               # dataset/model/runtime 設定
tests/                 # unit、integration、contract、fixtures
data/                  # README、metadata 與小型合法 samples
docs/                  # 架構、dataset card、model cards、致謝
legacy/                # 遷移完成後刪除
```

詳細順序與每階段驗收條件請見 [重構計畫](docs/REFACTOR_PLAN_ZH.md)。

D 槽原始資料的第一輪完整性結果請見 [資料稽核](docs/DATA_AUDIT_D_DRIVE_ZH.md)。

原始封存、私有工作集、壓縮 profiles 與未來發布位置請見 [Dataset 儲存與發布策略](docs/DATASET_STORAGE_AND_RELEASE_ZH.md)。

視角補全、感測器候選與下一步人工確認方式請見 [中繼資料補全流程](docs/METADATA_COMPLETION_ZH.md)。

Python package 的安裝方式與遷移規則請見 [Package 基礎](docs/PACKAGE_FOUNDATION_ZH.md)。

Canonical schema 與資料驗證規則請見 [Canonical Data Model](docs/CANONICAL_DATA_MODEL_ZH.md)。

VIA reader/converter 的 golden fixture 與真實資料 parity 結果請見 [VIA 遷移報告](docs/VIA_MIGRATION_ZH.md)。

資料稽核工具：

```powershell
python scripts/audit_source_dataset.py --help
```

## 原則

1. 先建立資料契約與測試，再搬模型和介面。
2. 每次只搬一層，所有 compatibility wrapper 都有刪除期限。
3. 前端只依賴 API schema，不讀取模型或研究輸出格式。
4. 原始影片、個資與大型 checkpoint 不進一般 Git。
5. 所有公開內容先確認拍攝同意、授權與署名方式。
