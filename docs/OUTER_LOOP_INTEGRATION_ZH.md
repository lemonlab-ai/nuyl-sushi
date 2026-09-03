# 外圈接入指南（中文版）

這份文件是給你在「內圈完成後」接外圈用的。
重點是：用固定契約交接，外圈不用猜路徑、不用人工整理。

## 1. 目標

把內圈輸出整理成一份 `handoff_manifest.json`，給外圈系統直接讀取：

1. k-fold 主結果
2. seed sweep 穩定度結果
3. 主要報表路徑
4. 重現命令與 commit 版本

## 2. 一條指令產生交接清單

```bash
python scripts/build_outer_loop_handoff.py --model-name multimodal_fusion
```

輸出檔案：

`outputs/outer_handoff/<model>/handoff_manifest.json`

## 3. 嚴格模式（正式交接建議）

如果必備檔案缺失就直接報錯：

```bash
python scripts/build_outer_loop_handoff.py \
  --model-name multimodal_fusion \
  --strict
```

必備檔案預設是：

1. `kfold_experiment_summary`
2. `seed_sweep_summary`
3. `seed_sweep_aggregate_csv`

如果你把 skill-level 分析也列為正式交接項目，請用：

```bash
python scripts/build_outer_loop_handoff.py \
  --model-name multimodal_fusion \
  --strict \
  --require-skill-level-metrics
```

此時必備項目會多兩個：

1. `skill_level_benchmark_json`
2. `skill_level_benchmark_csv`

## 4. 外圈應該讀哪些欄位

外圈最少讀：

1. `ready_for_outer_loop`
2. `missing_required_artifacts`
3. `outer_loop_inputs.primary_metrics_json`
4. `outer_loop_inputs.stability_metrics_json`
5. `outer_loop_inputs.stability_metrics_table_csv`

如果你有做 skill-level 分析，再讀：

6. `outer_loop_inputs.skill_level_metrics_json`
7. `outer_loop_inputs.skill_level_metrics_table_csv`

可攜性建議（跨機器/跨工作目錄）：

1. 優先讀 `outer_loop_inputs_relative.*`（相對於 repo root）
2. 若相對路徑為空，再 fallback 到 `outer_loop_inputs.*`（絕對路徑）
3. 執行命令可優先用 `repro_commands_python_explicit.*`

## 5. 推薦交接流程

1. 先跑 `python scripts/ci_smoke.py`
2. 跑你的正式 k-fold / seed sweep
3. 執行 `build_outer_loop_handoff.py --strict`
4. 把 `handoff_manifest.json + commit hash` 提供給外圈

如果此輪包含 skill-level 任務，步驟 3 改用：
`build_outer_loop_handoff.py --strict --require-skill-level-metrics`

## 6. 交給外圈的最小訊息模板

可以直接貼這段給外圈 agent：

```text
Project: ntnu-sushi
Commit: <your_commit_hash>
Manifest: outputs/outer_handoff/<model>/handoff_manifest.json
Please use manifest.outer_loop_inputs as the only source of artifact paths.
```

如果你的外圈支援相對路徑，建議改成：

```text
Please prefer manifest.outer_loop_inputs_relative, and fallback to manifest.outer_loop_inputs.
```

## 7. 常見問題

`ready_for_outer_loop = false`：

代表必備檔案不齊。
請先補跑：

1. `run_kfold_experiments.py --export-paper-table`
2. `run_seed_sweep.py --export-paper-table`

## 8. 外圈最小 Smoke 設定（可直接觸發內圈）

如果你要先驗證「外圈是否真的能驅動內圈」，可以先跑這個最小流程（單 seed、低 epoch）：

```json
{
  "project": "ntnu-sushi",
  "workdir": "C:/Users/<you>/Desktop/autoresearch/ntnu-sushi",
  "python": "C:/Users/Public/conda-envs/ntnu-sushi/python.exe",
  "steps": [
    "scripts/run_kfold_experiments.py --model-name multimodal_fusion --run-task-a --seed 41 --multimodal-epochs 1 --resume --export-paper-table",
    "scripts/run_seed_sweep.py --model-name multimodal_fusion --seeds 41 --multimodal-epochs 1 --resume --export-paper-table",
    "scripts/build_outer_loop_handoff.py --model-name multimodal_fusion --strict"
  ],
  "success_check": "outputs/outer_handoff/multimodal_fusion/handoff_manifest.json"
}
```

外圈執行時，等價命令如下（Windows）：

```powershell
C:\Users\Public\conda-envs\ntnu-sushi\python.exe scripts/run_kfold_experiments.py --model-name multimodal_fusion --run-task-a --seed 41 --multimodal-epochs 1 --resume --export-paper-table
C:\Users\Public\conda-envs\ntnu-sushi\python.exe scripts/run_seed_sweep.py --model-name multimodal_fusion --seeds 41 --multimodal-epochs 1 --resume --export-paper-table
C:\Users\Public\conda-envs\ntnu-sushi\python.exe scripts/build_outer_loop_handoff.py --model-name multimodal_fusion --strict
```

成功條件：

1. `handoff_manifest.json` 存在
2. `ready_for_outer_loop = true`
3. `missing_required_artifacts = []`

注意：

1. 請優先使用 `C:\Users\Public\conda-envs\ntnu-sushi\python.exe`（避免 `ModuleNotFoundError: wandb`）
2. smoke 只驗證管線連通，不代表最終模型效果
