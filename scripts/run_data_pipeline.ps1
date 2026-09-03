param(
  [string]$EnvName = "ntnu-sushi",
  [string]$EnvRoot = "C:\Users\Public\conda-envs",
  [switch]$TrimClips
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$EnvPrefix = Join-Path $EnvRoot $EnvName
$PythonExe = Join-Path $EnvPrefix "python.exe"

if (-not (Test-Path $PythonExe)) {
  throw "Python not found in env: $EnvPrefix. Run scripts/setup_conda_env.ps1 first."
}

Push-Location $ProjectRoot
try {
  Write-Host "[STEP] Convert VIA -> master_annotations.json" -ForegroundColor Cyan
  & $PythonExe scripts/convert_via_to_master.py
  if ($LASTEXITCODE -ne 0) { throw "convert_via_to_master.py failed." }

  Write-Host "[STEP] Validate master dataset" -ForegroundColor Cyan
  & $PythonExe scripts/validate_master_dataset.py
  if ($LASTEXITCODE -ne 0) { throw "validate_master_dataset.py failed." }

  Write-Host "[STEP] Export Task A artifacts" -ForegroundColor Cyan
  if ($TrimClips) {
    & $PythonExe scripts/export_task_a_clips.py --trim-clips
  } else {
    & $PythonExe scripts/export_task_a_clips.py
  }
  if ($LASTEXITCODE -ne 0) { throw "export_task_a_clips.py failed." }

  Write-Host "[STEP] Export Task B artifacts" -ForegroundColor Cyan
  & $PythonExe scripts/export_task_b_activitynet.py
  if ($LASTEXITCODE -ne 0) { throw "export_task_b_activitynet.py failed." }

  Write-Host "[DONE] Data pipeline completed." -ForegroundColor Green
  Write-Host "Master:   $ProjectRoot\\artifacts\\master"
  Write-Host "Task A:   $ProjectRoot\\artifacts\\task_a"
  Write-Host "Task B:   $ProjectRoot\\artifacts\\task_b"
}
finally {
  Pop-Location
}
