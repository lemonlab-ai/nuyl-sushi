param(
  [string]$EnvName = "ntnu-sushi",
  [string]$EnvRoot = "C:\Users\Public\conda-envs",
  [string]$ModelName = "multimodal_baseline",
  [switch]$RunTaskA,
  [switch]$RunTaskB,
  [string]$ActionFormerRepo = "external/actionformer_release",
  [string]$ActionFormerConfigRel = "task_b/actionformer_fine_config.yaml",
  [string]$ActionFormerLabelMapRel = "task_b/actionformer_label_map_fine.json",
  [string]$ActionFormerOutputTag = "actionformer",
  [switch]$ActionFormerSkipTrain = $false,
  [string]$ActionFormerCkptDirRel = "",
  [string]$ActionFormerTrainExtra = "",
  [int]$ActionFormerEvalTopk = -1,
  [int]$ActionFormerEvalPrintFreq = 10,
  [int]$Epochs = 100,
  [double]$LR = 0.1,
  [double]$WeightDecay = 0.0001,
  [double]$TargetFPS = 3.0,
  [int]$MaxFrames = 64,
  [int]$FusionHiddenDim = 128,
  [double]$FusionModDropVideo = 0.05,
  [double]$FusionModDropWeight = 0.20,
  [double]$FusionModDropGyro = 0.20,
  [switch]$UseReliabilityGating = $false,
  [double]$GatingWeightDecay = 0.0001,
  [switch]$DisableWeight = $false,
  [switch]$DisableGyro = $false,
  [ValidateSet("r3d_18", "mc3_18", "r2plus1d_18")]
  [string]$DeepVideoBackbone = "r3d_18",
  [ValidateSet("kinetics400", "none")]
  [string]$DeepVideoWeights = "kinetics400",
  [string]$DeepVideoDevice = "auto",
  [int]$DeepVideoClipLen = 16,
  [int]$DeepVideoClipHop = 8,
  [int]$DeepVideoResizeShort = 128,
  [int]$DeepVideoCropSize = 112,
  [int]$DeepVideoBatchSize = 8,
  [int]$DeepVideoMaxSampledFrames = 0,
  [ValidateSet("train", "val", "test", "all")]
  [string]$TaskASubset = "test",
  [string]$TaskBRawPredRel = "",
  [ValidateSet("auto", "csv", "json")]
  [string]$TaskBRawFormat = "auto",
  [ValidateSet("all", "training", "validation", "testing")]
  [string]$TaskBSubset = "all",
  [string]$TaskBIoUThresholds = "0.3,0.5,0.75",
  [switch]$Wandb = $false,
  [string]$WandbProject = "ntnu-sushi",
  [string]$WandbEntity = "",
  [string]$WandbRunName = "",
  [string]$WandbGroup = "kfold",
  [string]$WandbTags = "",
  [ValidateSet("online", "offline", "disabled")]
  [string]$WandbMode = "online",
  [switch]$WandbLogTrainerRuns = $false,
  [switch]$WandbLogSummaryRun = $false,
  [switch]$ExportPaperTable = $false,
  [string]$PaperTableCsv = "",
  [string]$PaperTableMd = "",
  [ValidateSet("train", "val", "test")]
  [string]$PaperTableAblationSubset = "test",
  [switch]$Resume = $false,
  [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $ScriptDir "run_with_env_python.ps1"

$argsList = @(
  "--model-name", "$ModelName",
  "--multimodal-epochs", "$Epochs",
  "--multimodal-lr", "$LR",
  "--multimodal-weight-decay", "$WeightDecay",
  "--multimodal-target-fps", "$TargetFPS",
  "--multimodal-max-frames", "$MaxFrames",
  "--multimodal-fusion-hidden-dim", "$FusionHiddenDim",
  "--multimodal-fusion-mod-drop-video", "$FusionModDropVideo",
  "--multimodal-fusion-mod-drop-weight", "$FusionModDropWeight",
  "--multimodal-fusion-mod-drop-gyro", "$FusionModDropGyro",
  "--multimodal-gating-weight-decay", "$GatingWeightDecay",
  "--multimodal-deepvideo-backbone", "$DeepVideoBackbone",
  "--multimodal-deepvideo-weights", "$DeepVideoWeights",
  "--multimodal-deepvideo-device", "$DeepVideoDevice",
  "--multimodal-deepvideo-clip-len", "$DeepVideoClipLen",
  "--multimodal-deepvideo-clip-hop", "$DeepVideoClipHop",
  "--multimodal-deepvideo-resize-short", "$DeepVideoResizeShort",
  "--multimodal-deepvideo-crop-size", "$DeepVideoCropSize",
  "--multimodal-deepvideo-batch-size", "$DeepVideoBatchSize",
  "--multimodal-deepvideo-max-sampled-frames", "$DeepVideoMaxSampledFrames",
  "--seed", "$Seed",
  "--actionformer-repo", "$ActionFormerRepo",
  "--actionformer-config-rel", "$ActionFormerConfigRel",
  "--actionformer-label-map-rel", "$ActionFormerLabelMapRel",
  "--actionformer-output-tag", "$ActionFormerOutputTag",
  "--actionformer-eval-topk", "$ActionFormerEvalTopk",
  "--actionformer-eval-print-freq", "$ActionFormerEvalPrintFreq",
  "--task-a-subset", "$TaskASubset",
  "--task-b-subset", "$TaskBSubset",
  "--task-b-raw-format", "$TaskBRawFormat",
  "--task-b-iou-thresholds", "$TaskBIoUThresholds",
  "--paper-table-ablation-subset", "$PaperTableAblationSubset"
)
if ($RunTaskA) { $argsList += "--run-task-a" }
if ($RunTaskB) { $argsList += "--run-task-b" }
if ($DisableWeight) { $argsList += "--multimodal-disable-weight" }
if ($DisableGyro) { $argsList += "--multimodal-disable-gyro" }
if ($UseReliabilityGating) { $argsList += "--multimodal-use-reliability-gating" }
if ($TaskBRawPredRel) { $argsList += @("--task-b-raw-pred-rel", "$TaskBRawPredRel") }
if ($Wandb) { $argsList += "--wandb" }
if ($WandbLogTrainerRuns) { $argsList += "--wandb-log-trainer-runs" }
if ($WandbLogSummaryRun) { $argsList += "--wandb-log-summary-run" }
if ($WandbProject) { $argsList += @("--wandb-project", "$WandbProject") }
if ($WandbEntity) { $argsList += @("--wandb-entity", "$WandbEntity") }
if ($WandbRunName) { $argsList += @("--wandb-run-name", "$WandbRunName") }
if ($WandbGroup) { $argsList += @("--wandb-group", "$WandbGroup") }
if ($WandbTags) { $argsList += @("--wandb-tags", "$WandbTags") }
if ($WandbMode) { $argsList += @("--wandb-mode", "$WandbMode") }
if ($ExportPaperTable) { $argsList += "--export-paper-table" }
if ($PaperTableCsv) { $argsList += @("--paper-table-csv", "$PaperTableCsv") }
if ($PaperTableMd) { $argsList += @("--paper-table-md", "$PaperTableMd") }
if ($Resume) { $argsList += "--resume" }
if ($ActionFormerSkipTrain) { $argsList += "--actionformer-skip-train" }
if ($ActionFormerCkptDirRel) { $argsList += @("--actionformer-ckpt-dir-rel", "$ActionFormerCkptDirRel") }
if ($ActionFormerTrainExtra) { $argsList += @("--actionformer-train-extra", "$ActionFormerTrainExtra") }

& $Runner `
  -EnvName $EnvName `
  -EnvRoot $EnvRoot `
  -Script "scripts/run_kfold_experiments.py" `
  -ScriptArgs $argsList
