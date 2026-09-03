# NTNU Sushi Research Workspace

> Repository status (2026-09-03): this is now the active NUYL Sushi codebase. See
> `docs/REPOSITORY_MIGRATION_ZH.md` for the local workspace layout and future
> repository migration plan.

Chinese beginner guide for the inner-loop workflow:
- `docs/README_ZH_INNER_LOOP.md`
Chinese outer-loop integration guide:
- `docs/OUTER_LOOP_INTEGRATION_ZH.md`
Chinese Gesture Coach demo guide:
- `docs/GESTURE_COACH_DEMO_ZH.md`
Chinese skill-level experiment design guide:
- `docs/SKILL_LEVEL_EXPERIMENT_DESIGN_ZH.md`

This folder is a clean workspace for your sushi action project. It is organized around:

1. `VIA -> master annotation` conversion
2. Task A: action recognition (coarse/fine)
3. Task B: temporal localization (start/end + class)
4. Training wrappers for the 4 model lines you selected:
   - TSM
   - SlowFast
   - VideoMAE V2
   - ActionFormer

## Folder Structure

```text
ntnu-sushi/
  data/
    raw/
      via_csv/                # put VIA CSV files here
      videos/                 # put source videos here
    meta/
      video_meta.json         # optional metadata (subject/view/gyro/weight/skill_level)
  docs/
  runners/
  scripts/
  templates/
  artifacts/
    master/                   # master_annotations.json + label maps
    task_a/                   # clip manifests / optional trimmed clips
    task_b/                   # ActivityNet-style JSON
```

## One-Time Setup (Miniconda)

Run one script to create the conda env and required folder layout:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_conda_env.ps1 -EnvName ntnu-sushi
```

This script will:
- create conda env `ntnu-sushi`
- install core deps (`ffmpeg`, `opencv`, `moviepy`, `jsonpickle`, `pyyaml`)
- create project directories:
  - `data/raw/via_csv`
  - `data/raw/videos`
  - `data/meta`
  - `artifacts/master`, `artifacts/task_a`, `artifacts/task_b`
- create `data/meta/video_meta.json` from template if missing

Note:
- the setup script creates the env at `C:\Users\Public\conda-envs\ntnu-sushi`
- this avoids permission/encoding issues from user profile paths with spaces/non-ASCII chars

If you also want repo-compatible `torch` on Windows (for local smoke tests), use Python 3.11:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_conda_env.ps1 `
  -EnvName ntnu-sushi `
  -PythonVersion 3.11 `
  -InstallTorch `
  -TorchChannel cpu
```

Use `-TorchChannel cu121` if your Windows GPU/CUDA runtime matches that wheel.
`wandb` is installed by default in this setup script (`-InstallWandb`).

## GPU Server Setup (Linux)

For next-step training on a GPU server, use:

```bash
cd ntnu-sushi
bash scripts/setup_gpu_server.sh
```

Important:
- Default is a repo-compatible stack: `torch==1.12.1+cu116`, `torchvision==0.13.1+cu116`, `torchaudio==0.12.1`.
- Override with `TORCH_INSTALL_CMD=...` if your server needs another CUDA wheel.
- Default conda env name is `ntnu-sushi-gpu` (override with `ENV_NAME=...`).
- If your server does not allow `sudo apt-get`, run with `INSTALL_SYSTEM_DEPS=0`.

Quick verify after install:
```bash
conda activate ntnu-sushi-gpu
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

Then run one-shot stack smoke test:
```bash
conda activate ntnu-sushi-gpu
bash scripts/server_smoke_test.sh
```

Optional short real-launch check (with timeout):
```bash
RUN_REAL_LAUNCH=1 LAUNCH_TIMEOUT_SEC=60 bash scripts/server_smoke_test.sh
```

## Data Placement (Fixed Convention)

Put your data here:
- VIA CSV files: `data/raw/via_csv`
- source videos: `data/raw/videos`
- optional metadata: `data/meta/video_meta.json` (`subject_id`, `view_type`, `weight/gyro`, `skill_level`)

Then activate the env:

```powershell
conda activate ntnu-sushi
```

Verify `ffmpeg` inside env:
```powershell
conda run -n ntnu-sushi ffmpeg -version
```

If `conda run` has Windows encoding issues on your machine, use env Python directly:
```powershell
C:\Users\Public\conda-envs\ntnu-sushi\python.exe --version
```

Or use wrapper:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_with_env_python.ps1 -Script scripts/validate_master_dataset.py
```

## Recommended Workflow

1. Convert VIA annotations into one master format (uses default paths above):
```bash
python scripts/convert_via_to_master.py
```

2. Validate the master dataset:
```bash
python scripts/validate_master_dataset.py
```

3. Export Task A data (classification clips/manifests):
```bash
python scripts/export_task_a_clips.py --trim-clips
```

4. Export Task B data (ActivityNet-style JSON):
```bash
python scripts/export_task_b_activitynet.py
```

5. Prepare repo-specific Task A formats (VideoMAEv2 / SlowFast):
```bash
python scripts/prepare_task_a_repo_formats.py
```

6. Prepare ActionFormer-compatible Task B JSON (+ config template):
```bash
python scripts/prepare_task_b_actionformer.py
```

7. Extract deep ActionFormer features (`.npy`, recommended):
```bash
python scripts/extract_task_b_features_deep.py \
  --gt-json artifacts/task_b/actionformer_fine.json \
  --output-dir artifacts/task_b/features_deep_r3d18 \
  --model r3d_18 \
  --weights kinetics400
```

Stronger Task B feature ensemble (concatenate multiple 3D backbones):
```bash
python scripts/extract_task_b_features_ensemble.py \
  --gt-json artifacts/task_b/actionformer_fine.json \
  --output-dir artifacts/task_b/features_deep_ensemble \
  --models r3d_18,mc3_18,r2plus1d_18 \
  --weights kinetics400
```

Baseline fallback (hand-crafted features):
```bash
python scripts/extract_task_b_features_basic.py
```

8. (Recommended) Generate subject-wise k-fold split masters:
```bash
python scripts/generate_subject_kfold_masters.py \
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

9. (Recommended) One-click k-fold pipeline:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_kfold_pipeline.ps1 `
  -NumFolds 5 `
  -ValRatio 0.2 `
  -Seed 42 `
  -KFoldSubjectLeakagePolicy fail `
  -KFoldImbalancePolicy warn `
  -KFoldMinTestLabelCountPerFold 1 `
  -KFoldMinCountPolicy warn `
  -PrepareRepoFormats `
  -PrepareActionFormer `
  -ExtractActionFormerFeatures `
  -ActionFormerFeatureExtractor deep `
  -DeepFeatureModel r3d_18
```

10. (Recommended) k-fold training/evaluation orchestration:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_kfold_experiments.ps1 `
  -ModelName multimodal_baseline `
  -RunTaskA `
  -Epochs 100 `
  -Resume `
  -ExportPaperTable
```
Note: Task A k-fold eval subset defaults to `test` (`-TaskASubset test`) to avoid train/val/test leakage.
Use `-TaskASubset all` only for diagnostics.
`experiment_summary.json` now includes a `reproducibility` section (argv, git commit/branch/dirty, python/pkg versions, UTC timestamp, fold list).

Improved Task A fusion model (missing-modality augmentation):
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_kfold_experiments.ps1 `
  -ModelName multimodal_fusion `
  -RunTaskA `
  -Epochs 120 `
  -FusionHiddenDim 128 `
  -FusionModDropVideo 0.05 `
  -FusionModDropWeight 0.20 `
  -FusionModDropGyro 0.20
```

Fusion + reliability gating (recommended innovation baseline):
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_kfold_experiments.ps1 `
  -ModelName multimodal_fusion `
  -RunTaskA `
  -UseReliabilityGating `
  -GatingWeightDecay 0.0001 `
  -Epochs 120
```

Task A video-only mainline (recommended when gyro is sparse or not time-aligned):
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_kfold_experiments.ps1 `
  -ModelName multimodal_fusion `
  -RunTaskA `
  -DisableWeight `
  -DisableGyro `
  -Epochs 120
```

Task B / ActionFormer k-fold orchestration:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_kfold_experiments.ps1 `
  -ModelName actionformer `
  -RunTaskB `
  -ActionFormerRepo external/actionformer_release `
  -ActionFormerConfigRel task_b/actionformer_fine_config.yaml `
  -ActionFormerLabelMapRel task_b/actionformer_label_map_fine.json
```
Note:
- `scripts/run_kfold_experiments.ps1` uses opt-in switches for both `-RunTaskA` and `-RunTaskB`.
- If you only pass `-RunTaskB`, Task A will not run.

## Frontend Integration (Streamlit)

For Task 7 (frontend/backend integration), this repo now includes a Streamlit app:
- `apps/streamlit_app.py`

What it provides:
- Task A UI: upload video (+ optional gyro CSV/weight) and run model inference (`linear_softmax` / `fusion_mlp` / `fusion_mlp_deepvideo`).
- Task B UI: upload raw prediction (`.csv/.json/.pkl`), convert to eval JSON, and run localization metrics.
- Task B (ActionFormer) UI: upload a video and generate prediction file (`pred_eval.json`) from a trained ActionFormer checkpoint.
- Gesture Coach (MVP) UI: compare two keypoint JSON files with DTW-based similarity score (2D/3D/3D-MANO-compatible mode).
- Gesture Coach (MVP) UI: optionally extract 2D/3D hand keypoints from two videos with MediaPipe, then compare.

Run directly:
```bash
streamlit run apps/streamlit_app.py
```

Or on Windows with env path wrapper:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_streamlit.ps1 -EnvName ntnu-sushi
```

Notes:
- Task A tab auto-detects model path (`outputs/multimodal_fusion/model.npz` -> `outputs/multimodal_baseline/model.npz` -> `outputs/multimodal_smoke/model.npz`) and you can still override in sidebar.
- Deepvideo Task A model is also supported when `model.npz` contains `model_type=fusion_mlp_deepvideo`.
- Task B tab reuses existing backend scripts: `convert_task_b_predictions.py` + `eval_task_b.py`.
- Task B ActionFormer single-video inference requires CUDA-ready ActionFormer runtime and a valid checkpoint path.
- Gesture Coach extraction mode requires `mediapipe` in your env (`pip install mediapipe`).
- Gesture Coach MANO-compatible mode supports precomputed `left_mano/right_mano`; optional fallback can proxy from `left_world_3d/right_world_3d`.

## One-Command Data Pipeline

After placing CSV/videos in the fixed folders, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_data_pipeline.ps1 -EnvName ntnu-sushi -TrimClips
```

This runs:
1. `convert_via_to_master.py`
2. `validate_master_dataset.py`
3. `export_task_a_clips.py` (with optional `--trim-clips`)
4. `export_task_b_activitynet.py`

For k-fold experiments, use:
- `scripts/run_kfold_pipeline.py` (Python)
- or `scripts/run_kfold_pipeline.ps1` (PowerShell wrapper)

Deep Task B feature extraction in k-fold pipeline (recommended):
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_kfold_pipeline.ps1 `
  -PrepareActionFormer `
  -ExtractActionFormerFeatures `
  -ActionFormerFeatureExtractor deep `
  -DeepFeatureModel r3d_18 `
  -DeepFeatureWeights kinetics400
```

Task B deep ensemble extraction in k-fold pipeline:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_kfold_pipeline.ps1 `
  -PrepareActionFormer `
  -ExtractActionFormerFeatures `
  -ActionFormerFeatureExtractor ensemble `
  -DeepFeatureWeights none `
  -DeepFeatureDevice cpu `
  -EnsembleFeatureModels "r3d_18,mc3_18,r2plus1d_18"
```

5. Clone external repos (one command):
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_repos.ps1 -BaseDir external
```

6. Start baseline training:
- Recognition: TSM / SlowFast / VideoMAE V2
- Localization: ActionFormer

Use wrappers in `runners/` and template configs in `templates/`.

## Repo Format Adapters

### Task A -> VideoMAEv2 / SlowFast

`scripts/prepare_task_a_repo_formats.py` converts `task_a_*_manifest.csv` into Kinetics-style split files:

- `artifacts/task_a/repo_formats/videomaev2/train.csv|val.csv|test.csv`
- `artifacts/task_a/repo_formats/slowfast/train.csv|val.csv|test.csv`

Default mode writes paths relative to `artifacts/task_a` (recommended).

Example:
```bash
python scripts/export_task_a_clips.py --trim-clips
python scripts/prepare_task_a_repo_formats.py --path-mode relative
```

Then:
- VideoMAEv2:
  - `--data_path artifacts/task_a/repo_formats/videomaev2`
  - `--data_root artifacts/task_a`
- SlowFast:
  - `DATA.PATH_TO_DATA_DIR = artifacts/task_a/repo_formats/slowfast`
  - `DATA.PATH_PREFIX = artifacts/task_a`

### Task B -> ActionFormer

`scripts/prepare_task_b_actionformer.py` adds `label_id` to each temporal annotation and writes:

- `artifacts/task_b/actionformer_fine.json`
- `artifacts/task_b/actionformer_label_map_fine.json`
- `artifacts/task_b/actionformer_fine_config.yaml` (template, includes `output_folder`)

Example:
```bash
python scripts/export_task_b_activitynet.py --level fine
python scripts/prepare_task_b_actionformer.py \
  --feature-dir artifacts/task_b/features_deep_r3d18 \
  --infer-input-dim
```

Important:
- ActionFormer needs per-video feature files (`.npy`) in `feat_folder`.
- Recommended deep feature set (torchvision 3D backbone):
```bash
python scripts/extract_task_b_features_deep.py \
  --gt-json artifacts/task_b/actionformer_fine.json \
  --output-dir artifacts/task_b/features_deep_r3d18 \
  --model r3d_18 \
  --weights kinetics400
```
- Stronger ensemble feature set:
```bash
python scripts/extract_task_b_features_ensemble.py \
  --gt-json artifacts/task_b/actionformer_fine.json \
  --output-dir artifacts/task_b/features_deep_ensemble \
  --models r3d_18,mc3_18,r2plus1d_18 \
  --weights kinetics400
```
- Then regenerate config with automatic `input_dim` inference:
```bash
python scripts/prepare_task_b_actionformer.py \
  --feature-dir artifacts/task_b/features_deep_r3d18 \
  --infer-input-dim
```
- Baseline fallback (hand-crafted features):
```bash
python scripts/extract_task_b_features_basic.py \
  --gt-json artifacts/task_b/actionformer_fine.json \
  --output-dir artifacts/task_b/features_basic
```
- Then regenerate config with matching dim:
```bash
python scripts/prepare_task_b_actionformer.py \
  --input-dim 13 \
  --feature-dir artifacts/task_b/features_basic
```

## Runner Quickstart

### VideoMAEv2 (custom class count)

```bash
python runners/train_videomaev2.py \
  --repo external/VideoMAEv2 \
  --auto-patch \
  --data-path artifacts/task_a/repo_formats/videomaev2 \
  --data-root artifacts/task_a \
  --nb-classes <NUM_CLASSES> \
  --output-dir outputs/videomaev2
```

### TSM (`ntnu_sushi` dataset wiring)

First build TSM frame/list artifacts:
```bash
python scripts/prepare_task_a_repo_formats.py --prepare-tsm
```

Then train:
```bash
python runners/train_tsm.py \
  --repo external/temporal-shift-module \
  --dataset ntnu_sushi \
  --auto-patch \
  --root-path artifacts/task_a/repo_formats/tsm/frames \
  --train-list artifacts/task_a/repo_formats/tsm/train_videofolder.txt \
  --val-list artifacts/task_a/repo_formats/tsm/val_videofolder.txt \
  --category-file artifacts/task_a/repo_formats/tsm/category.txt
```

### Multi-Modal Baseline (video + gyro + weight)

```bash
python scripts/train_multimodal_baseline.py \
  --manifest-csv artifacts/task_a/task_a_fine_manifest.csv \
  --master-json artifacts/master/master_annotations.json \
  --output-dir outputs/multimodal_baseline
```

### Multi-Modal Fusion MLP (with missing-modality augmentation)

```bash
python scripts/train_multimodal_fusion.py \
  --manifest-csv artifacts/task_a/task_a_fine_manifest.csv \
  --master-json artifacts/master/master_annotations.json \
  --output-dir outputs/multimodal_fusion \
  --hidden-dim 128 \
  --mod-drop-video 0.05 \
  --mod-drop-weight 0.20 \
  --mod-drop-gyro 0.20
```
Add `--use-reliability-gating` to enable modality reliability gating.
Both fusion trainers now auto-export:
- `ablation_report.json`
- `ablation_report.csv`
with scenarios: `video_only`, `plus_weight`, `plus_gyro`, `plus_gating`.

Video-only (disable sensor modalities):
```bash
python scripts/train_multimodal_fusion.py \
  --manifest-csv artifacts/task_a/task_a_fine_manifest.csv \
  --master-json artifacts/master/master_annotations.json \
  --output-dir outputs/multimodal_fusion_video_only \
  --disable-weight \
  --disable-gyro
```

### Multi-Modal Fusion (Deep Video Backbone + sensors)

```bash
python scripts/train_multimodal_fusion_deepvideo.py \
  --manifest-csv artifacts/task_a/task_a_fine_manifest.csv \
  --master-json artifacts/master/master_annotations.json \
  --output-dir outputs/multimodal_fusion_deepvideo \
  --video-backbone r3d_18 \
  --video-weights kinetics400 \
  --video-device auto
```

Video-only deep backbone:
```bash
python scripts/train_multimodal_fusion_deepvideo.py \
  --manifest-csv artifacts/task_a/task_a_fine_manifest.csv \
  --master-json artifacts/master/master_annotations.json \
  --output-dir outputs/multimodal_fusion_deepvideo_video_only \
  --video-backbone r3d_18 \
  --video-weights none \
  --video-device cpu \
  --disable-weight \
  --disable-gyro
```

## WandB Tracking

Per-training-script tracking:
```bash
python scripts/train_multimodal_fusion.py \
  --manifest-csv artifacts/task_a/task_a_fine_manifest.csv \
  --master-json artifacts/master/master_annotations.json \
  --output-dir outputs/multimodal_fusion \
  --wandb \
  --wandb-project ntnu-sushi \
  --wandb-mode offline
```

k-fold tracking (summary run + per-fold trainer runs):
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_kfold_experiments.ps1 `
  -ModelName multimodal_fusion `
  -RunTaskA `
  -Wandb `
  -WandbMode offline `
  -WandbLogSummaryRun `
  -WandbLogTrainerRuns
```

Multi-seed sweep (same config, multiple seeds + aggregate summary):
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
Outputs:
- `outputs/seed_sweeps/<model>/seed_sweep_summary.json`
- `outputs/seed_sweeps/<model>/seed_sweep_summary.csv`
- `outputs/seed_sweeps/<model>/seed_sweep_aggregate.csv` (mean/std/95% CI + best/worst seed)

Task B perfect-pred smoke for full k-fold eval:
```powershell
# Generate pred_perfect.json for each fold
0..4 | ForEach-Object {
  $fold = ('fold_{0:D2}' -f $_)
  C:\Users\Public\conda-envs\ntnu-sushi\python.exe scripts/generate_task_b_perfect_predictions.py `
    --gt-json artifacts/splits_kfold/$fold/task_b/activitynet_fine.json `
    --output-json artifacts/splits_kfold/$fold/task_b/pred_perfect.json
}

# Run Task B k-fold eval using those predictions
powershell -ExecutionPolicy Bypass -File scripts/run_kfold_experiments.ps1 `
  -ModelName taskb_perfect `
  -RunTaskB `
  -TaskBRawPredRel task_b/pred_perfect.json `
  -TaskBRawFormat json `
  -TaskBSubset all
```

## Develop Without Real Data

You can develop the pipeline before real VIA/video files are ready:

1. Generate mock dataset:
```bash
python scripts/generate_mock_dataset.py
```

2. Run full smoke test (conversion + export + evaluation + tracking):
```bash
python scripts/smoke_test_pipeline.py
```

## Skill-Level Extension (2D / 3D / MANO-compatible)

If you want to evaluate whether hand keypoints can classify `skill_level` (`beginner/intermediate/expert`):

1. Ensure `skill_level` exists in `data/meta/video_meta.json`.
2. Convert annotations to master:
```bash
python scripts/convert_via_to_master.py
```
3. Optional manifest export:
```bash
python scripts/export_skill_level_manifest.py \
  --master-json artifacts/master/master_annotations.json \
  --output-csv artifacts/skill_level/skill_level_manifest.csv
```
4. Batch extract keypoint sequences:
```bash
python scripts/extract_skill_level_keypoints.py \
  --master-json artifacts/master/master_annotations.json \
  --output-dir artifacts/skill_level/keypoints \
  --subset all \
  --drop-unknown-skill
```
5. Run coordinate-mode ablation benchmark:
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

Outputs:
- `outputs/skill_level/skill_level_keypoint_benchmark.json`
- `outputs/skill_level/skill_level_keypoint_benchmark.csv`

## Evaluation Scripts

- Task A classification:
```bash
python scripts/eval_task_a.py --pred-csv <pred.csv>
```

With per-class report + confusion matrix:
```bash
python scripts/eval_task_a.py \
  --pred-csv <pred.csv> \
  --per-class-csv outputs/per_class.csv \
  --confusion-csv outputs/confusion.csv \
  --output-json outputs/metrics_task_a.json
```

- Task B temporal localization:
```bash
python scripts/eval_task_b.py --pred-json <pred.json>
```
`eval_task_b.py` reports both mAP and Recall at each IoU threshold (`Recall_by_IoU`, `average_Recall`).

Prediction formats:
- Task A CSV: `clip_id,pred_label_id` (or `pred_label`)
- Task B JSON:
  - ActivityNet-style `{"results": {"video_id": [{"segment":[s,e],"label":"X","score":0.9}]}}`
  - or flat list with `video_id,label,score,segment`

Prediction conversion helpers:
```bash
python scripts/convert_task_a_predictions.py --input <raw_pred.csv_or_json> --output-csv <pred_eval.csv>
python scripts/convert_task_b_predictions.py --input <raw_pred.csv_or_json_or_pkl> --output-json <pred_eval.json>
# Optional for pkl/numeric labels:
python scripts/convert_task_b_predictions.py --input <eval_results.pkl> --format pkl --label-map <actionformer_label_map.json> --output-json <pred_eval.json>
```

## Experiment Tracking

Log one run:
```bash
python scripts/log_experiment.py \
  --experiment-id exp001 \
  --task task_a \
  --model videomaev2 \
  --split val \
  --metric top1_accuracy=0.81 \
  --metric macro_f1=0.74
```

Summarize runs:
```bash
python scripts/summarize_results.py --task task_a
```

Summarize k-fold metrics (mean/std):
```bash
python scripts/summarize_kfold_metrics.py \
  --split-root artifacts/splits_kfold \
  --task all \
  --output-json artifacts/splits_kfold/kfold_summary.json \
  --output-csv artifacts/splits_kfold/kfold_summary.csv
```

Add `--include-detailed` if you also want per-class metric keys.

Export paper-ready tables (CSV + Markdown):
```bash
python scripts/export_paper_tables.py \
  --experiment-summary outputs/kfold_experiments/multimodal_fusion/experiment_summary.json
```

Export table with both k-fold and seed-sweep stats:
```bash
python scripts/export_paper_tables.py \
  --experiment-summary outputs/kfold_experiments/multimodal_fusion/experiment_summary.json \
  --seed-sweep-summary outputs/seed_sweeps/multimodal_fusion/seed_sweep_summary.json
```

## CI Smoke

GitLab CI smoke pipeline:
- `.gitlab-ci.yml`
- runs `python scripts/ci_smoke.py`

Local run:
```bash
python scripts/ci_smoke.py
```

## Key Principle

Keep one **single source of truth**:
- `master_annotations.json` is canonical
- all task/repo formats are exported from it

This prevents annotation drift and keeps experiments reproducible.

## Current Gaps (What Is Not Finished Yet)

1. VideoMAEv2 / TSM are currently patched by helper scripts; if upstream repo changes, patch scripts may need updates.
2. Deep feature extraction is currently torchvision 3D baseline (`r3d_18/mc3_18/r2plus1d_18`); stronger feature pipelines (e.g., VideoMAEv2/I3D pretrained features) are still future work.
3. Multi-modal has both lightweight (hand-crafted) and deepvideo fusion paths; deepvideo is available, but slower end-to-end training/backbone fine-tuning is still future work.
4. More advanced gyro/weight fusion (sequence encoders, uncertainty-aware missing-modality handling) is still future work.
