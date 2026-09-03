param(
  [string]$EnvName = "ntnu-sushi",
  [string]$EnvRoot = "C:\Users\Public\conda-envs",
  [int]$NumFolds = 5,
  [double]$ValRatio = 0.2,
  [int]$Seed = 42,
  [ValidateSet("none", "coarse", "fine", "both")]
  [string]$KFoldStratifyLevel = "fine",
  [ValidateSet("warn", "fail")]
  [string]$KFoldSubjectLeakagePolicy = "fail",
  [ValidateSet("off", "warn", "fail")]
  [string]$KFoldImbalancePolicy = "warn",
  [double]$KFoldMaxLabelRelativeDeviation = 0.8,
  [int]$KFoldImbalanceMinTotal = 5,
  [int]$KFoldMinTestLabelCountPerFold = 0,
  [ValidateSet("off", "warn", "fail")]
  [string]$KFoldMinCountPolicy = "off",
  [switch]$TrimClips,
  [switch]$PrepareRepoFormats,
  [switch]$PrepareTSM,
  [switch]$PrepareActionFormer,
  [switch]$ExtractActionFormerFeatures,
  [ValidateSet("basic", "deep", "ensemble")]
  [string]$ActionFormerFeatureExtractor = "basic",
  [string]$ActionFormerFeatureDirName = "",
  [switch]$ActionFormerInferInputDim,
  [ValidateSet("r3d_18", "mc3_18", "r2plus1d_18")]
  [string]$DeepFeatureModel = "r3d_18",
  [ValidateSet("kinetics400", "none")]
  [string]$DeepFeatureWeights = "kinetics400",
  [string]$DeepFeatureDevice = "auto",
  [string]$EnsembleFeatureModels = "r3d_18,mc3_18,r2plus1d_18"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$Runner = Join-Path $ScriptDir "run_with_env_python.ps1"

$argsList = @(
  "--num-folds", "$NumFolds",
  "--val-ratio", "$ValRatio",
  "--seed", "$Seed",
  "--kfold-stratify-level", "$KFoldStratifyLevel",
  "--kfold-subject-leakage-policy", "$KFoldSubjectLeakagePolicy",
  "--kfold-imbalance-policy", "$KFoldImbalancePolicy",
  "--kfold-max-label-relative-deviation", "$KFoldMaxLabelRelativeDeviation",
  "--kfold-imbalance-min-total", "$KFoldImbalanceMinTotal",
  "--kfold-min-test-label-count-per-fold", "$KFoldMinTestLabelCountPerFold",
  "--kfold-min-count-policy", "$KFoldMinCountPolicy"
)

if ($TrimClips) { $argsList += "--trim-clips" }
if ($PrepareRepoFormats) { $argsList += "--prepare-repo-formats" }
if ($PrepareTSM) { $argsList += "--prepare-tsm" }
if ($PrepareActionFormer) { $argsList += "--prepare-actionformer" }
if ($ExtractActionFormerFeatures) { $argsList += "--extract-actionformer-features" }
$argsList += @("--actionformer-feature-extractor", $ActionFormerFeatureExtractor)
if ($ActionFormerFeatureDirName) { $argsList += @("--actionformer-feature-dir-name", $ActionFormerFeatureDirName) }
if ($ActionFormerInferInputDim) { $argsList += "--actionformer-infer-input-dim" }
$argsList += @("--deep-feature-model", $DeepFeatureModel)
$argsList += @("--deep-feature-weights", $DeepFeatureWeights)
$argsList += @("--deep-feature-device", $DeepFeatureDevice)
$argsList += @("--ensemble-feature-models", $EnsembleFeatureModels)

Push-Location $ProjectRoot
try {
  & $Runner `
    -EnvName $EnvName `
    -EnvRoot $EnvRoot `
    -Script "scripts/run_kfold_pipeline.py" `
    -ScriptArgs $argsList
}
finally {
  Pop-Location
}
