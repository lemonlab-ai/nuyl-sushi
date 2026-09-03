# Python Package 基礎

更新日期：2026-09-03

專案採用 Python 3.11、PEP 517 `pyproject.toml` 與 `src/` layout。可重用的資料處理、評估與推論邏輯放在 `src/nuyl_sushi/`；`scripts/` 逐步縮減為相容 CLI，不再成為其他模組的 import 來源。

## 目前結構

```text
src/nuyl_sushi/
  config.py                 # 專案路徑探索與 NUYL_SUSHI_ROOT override
  logging.py                # 共用 logging 設定
  cli/                      # 可安裝的命令列入口
  data/                     # 資料處理核心
  domain/                   # canonical schema 與 taxonomy（下一批）
  evaluation/               # 評估邏輯（待遷移）
  inference/                # predictor contract（待建立）
  gesture/                  # 手勢處理（待遷移）
```

第一個完成遷移的功能是 metadata candidates：核心演算法位於 `nuyl_sushi.data.metadata_candidates`，CLI 位於 `nuyl_sushi.cli.metadata_candidates`，舊的 `scripts/build_metadata_candidates.py` 保留為薄 wrapper。

## 安裝

最小安裝不需要第三方 runtime dependency：

```powershell
python -m pip install -e .
```

依工作內容選擇 extras：

```powershell
python -m pip install -e ".[data]"
python -m pip install -e ".[vision]"
python -m pip install -e ".[research,demo]"
python -m pip install -e ".[all]"
```

安裝後可使用：

```powershell
nuyl-sushi-metadata --help
```

尚未安裝 package 時，既有命令仍可使用：

```powershell
python scripts/build_metadata_candidates.py --help
```

## 路徑規則

package 不保存 `C:` 或 `D:` 等個人絕對路徑。預設會由目前目錄或 package 位置往上尋找 `pyproject.toml`；若工具從 checkout 外執行，可設定：

```powershell
$env:NUYL_SUSHI_ROOT = "D:\path\to\checkout"
```

輸入資料位置仍應優先由 CLI 參數明確傳入。環境變數只用於指定專案 checkout，不應用來隱藏資料來源或輸出位置。

## 遷移規則

1. 先將純函式與資料契約搬入 package，再建立 CLI。
2. 舊腳本只做 bootstrap 與呼叫新入口，避免一次破壞既有研究命令。
3. package core 不得 import `scripts` 或 `apps`。
4. 重型套件必須延遲 import，或放在對應 optional extra。
5. 每搬一項功能，都要有 package-level test 與舊 CLI parity check。

下一批為 canonical data model：定義影片、片段、metadata 與 dataset validation contract，然後再遷移 VIA converter。

