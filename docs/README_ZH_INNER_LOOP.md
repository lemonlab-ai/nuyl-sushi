# NTNU Sushi 內圈新手指南（中文版）

這份文件是給「第一次碰 ML 專案」的人。
目標不是一次學完理論，而是先把專案跑起來，知道每一步在做什麼，最後能把結果交給外圈（更大規模研究/自動化系統）。

## 1. 你現在有什麼

這個專案的「內圈」已經具備：

1. 可重現的 subject-wise k-fold 切分（含品質檢查）
2. Task A / Task B 的 k-fold 訓練與評估入口
3. `--resume` 續跑能力（中斷後可跳過已完成 fold）
4. 多 seed 批次實驗與統計（含 95% CI、best/worst seed）
5. 論文表格輸出（CSV + Markdown）
6. CI smoke 測試（快速檢查流程是否壞掉）

## 2. 先決條件

建議使用 conda 環境 `ntnu-sushi`。

Windows（PowerShell）：

```powershell
conda activate ntnu-sushi
```

如果你本機 `conda activate` 不方便，可以直接用環境 Python：

```powershell
C:\Users\Public\conda-envs\ntnu-sushi\python.exe --version
```

## 3. 一鍵確認內圈健康度（必做）

先跑 smoke 測試，確認「流程骨架」正常：

```bash
python scripts/ci_smoke.py
```

如果這一步失敗，不要急著長時間訓練，先修這個。

## 4. 內圈標準手順（建議照順序）

## Step A: 產生 k-fold split（含品質檢查）

```bash
python scripts/generate_subject_kfold_masters.py \
  --master-json artifacts/master/master_annotations.json \
  --output-dir artifacts/splits_kfold \
  --num-folds 5 \
  --val-ratio 0.2 \
  --seed 42 \
  --stratify-level fine \
  --subject-leakage-policy fail \
  --imbalance-policy warn \
  --max-label-relative-deviation 0.8 \
  --min-test-label-count-per-fold 1 \
  --min-count-policy warn
```

你會得到：

1. `artifacts/splits_kfold/fold_00...fold_04`
2. `artifacts/splits_kfold/summary.json`（含 split 品質報告）

## Step B: 跑 k-fold 資料管線（Task A/Task B artifact）

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_kfold_pipeline.ps1 `
  -NumFolds 5 `
  -ValRatio 0.2 `
  -Seed 42 `
  -KFoldStratifyLevel fine `
  -KFoldSubjectLeakagePolicy fail `
  -KFoldImbalancePolicy warn `
  -KFoldMinTestLabelCountPerFold 1 `
  -KFoldMinCountPolicy warn `
  -PrepareRepoFormats `
  -PrepareActionFormer
```

## Step C: 跑 k-fold 訓練/評估（可續跑）

先從最簡單 baseline 開始：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_kfold_experiments.ps1 `
  -ModelName multimodal_baseline `
  -RunTaskA `
  -Epochs 100 `
  -Resume `
  -ExportPaperTable
```

`-Resume` 的意思：
如果某個 fold 已經有 `task_a_metrics.json` / `task_b_metrics.json`，就直接重用，不重跑該 fold。

## Step D: 跑多 seed（看穩定度）

```bash
python scripts/run_seed_sweep.py \
  --seeds 41,42,43 \
  --model-name multimodal_fusion \
  --run-task-a \
  --per-run-timeout-sec 21600 \
  --multimodal-epochs 120 \
  --multimodal-use-reliability-gating \
  --export-paper-table
```

輸出重點：

1. `outputs/seed_sweeps/<model>/seed_sweep_summary.json`
2. `outputs/seed_sweeps/<model>/seed_sweep_summary.csv`
3. `outputs/seed_sweeps/<model>/seed_sweep_aggregate.csv`

`seed_sweep_aggregate.csv` 會有：

1. mean / std
2. 95% CI
3. best seed / worst seed

## Step E: 匯出論文表格

只用 k-fold 結果：

```bash
python scripts/export_paper_tables.py \
  --experiment-summary outputs/kfold_experiments/multimodal_fusion/experiment_summary.json
```

同時整合 k-fold + seed sweep：

```bash
python scripts/export_paper_tables.py \
  --experiment-summary outputs/kfold_experiments/multimodal_fusion/experiment_summary.json \
  --seed-sweep-summary outputs/seed_sweeps/multimodal_fusion/seed_sweep_summary.json
```

## 5. 最常見錯誤與解法

## 問題 1: `ModuleNotFoundError: wandb`

原因：你用到系統 Python，不是專案 conda 環境。

解法：

1. `conda activate ntnu-sushi`
2. 或改用 `C:\Users\Public\conda-envs\ntnu-sushi\python.exe`

## 問題 2: 中斷後重跑又從頭開始

原因：沒加 `--resume` / `-Resume`。

解法：

1. Python 版加 `--resume`
2. PowerShell 版加 `-Resume`

## 問題 3: split 品質報警

看 `artifacts/splits_kfold/summary.json` 的 `split_quality`。
如果是探索階段可先用 `warn`，正式實驗建議用 `fail`。

## 6. 交給外圈前檢查清單

以下都過，再交接外圈：

1. `python scripts/ci_smoke.py` 成功
2. k-fold 實驗有完整 `experiment_summary.json`
3. seed sweep 有 `seed_sweep_summary.json` + `seed_sweep_aggregate.csv`
4. 論文表格 `paper_table.csv` / `paper_table.md` 已產出
5. commit hash 固定（例如 `main@<hash>`）

## 7. 你可以先記住的核心概念

1. 內圈目標是「可靠、可重現、可續跑」
2. 外圈目標是「大規模探索與自動化研究」
3. 內圈做得穩，外圈才不會浪費 GPU 時間

## 8. Skill Level 延伸實驗（2D/3D/MANO）

如果你要做「新手/進階/專家」分級，建議用這條線：

1. 在 `data/meta/video_meta.json` 補上 `skill_level`（`beginner/intermediate/expert`）
2. 重建 master：
```bash
python scripts/convert_via_to_master.py
```
3. 匯出 skill-level 清單：
```bash
python scripts/export_skill_level_manifest.py \
  --master-json artifacts/master/master_annotations.json \
  --output-csv artifacts/skill_level/skill_level_manifest.csv
```
4. 抽 keypoint（MediaPipe）：
```bash
python scripts/extract_skill_level_keypoints.py \
  --master-json artifacts/master/master_annotations.json \
  --output-dir artifacts/skill_level/keypoints \
  --subset all \
  --drop-unknown-skill
```
5. 跑 2D/3D/MANO 對比：
```bash
python scripts/eval_skill_level_from_keypoints.py \
  --index-path artifacts/skill_level/keypoints/skill_level_keypoint_index.csv \
  --coord-modes 2d,3d_image,3d_world,3d_mano \
  --hand-mode both \
  --num-folds 5 \
  --seed 42 \
  --allow-mano-proxy \
  --output-dir outputs/skill_level
```

結果檔案：

1. `outputs/skill_level/skill_level_keypoint_benchmark.json`
2. `outputs/skill_level/skill_level_keypoint_benchmark.csv`
3. 詳細設計文件：`docs/SKILL_LEVEL_EXPERIMENT_DESIGN_ZH.md`

---

外圈接入請看：
`docs/OUTER_LOOP_INTEGRATION_ZH.md`
