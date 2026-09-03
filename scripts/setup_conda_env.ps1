param(
  [string]$EnvName = "ntnu-sushi",
  [string]$EnvRoot = "C:\Users\Public\conda-envs",
  [string]$PythonVersion = "3.11",
  [switch]$Recreate,
  [switch]$InstallTorch,
  [ValidateSet("cpu", "cu121")]
  [string]$TorchChannel = "cpu",
  [switch]$InstallWandb = $true,
  [switch]$InstallMediapipe = $true
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

function Write-Step {
  param([string]$Message)
  Write-Host "[STEP] $Message" -ForegroundColor Cyan
}

function Ensure-Conda {
  if (Get-Command conda -ErrorAction SilentlyContinue) {
    return
  }
  throw "conda command not found. Please open Anaconda/Miniconda PowerShell or add conda to PATH."
}

function Invoke-Conda {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$CmdArgs)
  & conda @CmdArgs
  if ($LASTEXITCODE -ne 0) {
    throw "conda command failed: conda $($CmdArgs -join ' ')"
  }
}

function Ensure-DirectoryLayout {
  $dirs = @(
    (Join-Path $ProjectRoot "data\\raw\\via_csv"),
    (Join-Path $ProjectRoot "data\\raw\\videos"),
    (Join-Path $ProjectRoot "data\\meta"),
    (Join-Path $ProjectRoot "artifacts\\master"),
    (Join-Path $ProjectRoot "artifacts\\task_a"),
    (Join-Path $ProjectRoot "artifacts\\task_b"),
    (Join-Path $ProjectRoot "external"),
    (Join-Path $ProjectRoot "logs"),
    (Join-Path $ProjectRoot "outputs")
  )

  foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
  }
}

function Ensure-MetaTemplate {
  $target = Join-Path $ProjectRoot "data\\meta\\video_meta.json"
  $template = Join-Path $ProjectRoot "data\\meta\\video_meta.template.json"
  if (-not (Test-Path $target) -and (Test-Path $template)) {
    Copy-Item $template $target
    Write-Host "[INFO] Created $target from template."
  }
}

function Env-Exists {
  param([string]$PrefixPath)
  return (Test-Path (Join-Path $PrefixPath "python.exe"))
}

function Get-EnvPythonVersion {
  param([string]$PrefixPath)
  $pythonExe = Join-Path $PrefixPath "python.exe"
  if (-not (Test-Path $pythonExe)) {
    return ""
  }
  $ver = & $pythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
  if ($LASTEXITCODE -ne 0) {
    return ""
  }
  return "$ver".Trim()
}

function Test-CoreDeps {
  param(
    [string]$PrefixPath,
    [bool]$RequireMediapipe = $true
  )
  if (-not (Env-Exists -PrefixPath $PrefixPath)) {
    return $false
  }
  $pythonExe = Join-Path $PrefixPath "python.exe"
  $imports = "import jsonpickle, moviepy, cv2, yaml, streamlit"
  if ($RequireMediapipe) {
    $imports = "$imports, mediapipe"
  }
  & $pythonExe -c $imports 2>$null
  if ($LASTEXITCODE -ne 0) {
    return $false
  }

  $ffmpegCandidates = @(
    (Join-Path $PrefixPath "Library\\bin\\ffmpeg.exe"),
    (Join-Path $PrefixPath "Scripts\\ffmpeg.exe"),
    (Join-Path $PrefixPath "bin\\ffmpeg.exe")
  )
  foreach ($ff in $ffmpegCandidates) {
    if (Test-Path $ff) {
      return $true
    }
  }
  return $false
}

function Install-TorchStack {
  param(
    [string]$PrefixPath,
    [string]$Channel
  )
  $pythonExe = Join-Path $PrefixPath "python.exe"
  if (-not (Test-Path $pythonExe)) {
    throw "python.exe not found in env: $PrefixPath"
  }

  $pyVer = Get-EnvPythonVersion -PrefixPath $PrefixPath
  if ($pyVer -notin @("3.10", "3.11", "3.12")) {
    throw "Unsupported python version for torch install: $pyVer. Please use Python 3.10/3.11/3.12."
  }

  $indexUrl = if ($Channel -eq "cu121") { "https://download.pytorch.org/whl/cu121" } else { "https://download.pytorch.org/whl/cpu" }
  Write-Step "Installing torch stack (latest stable from channel=$Channel, python=$pyVer)"
  & $pythonExe -m pip install --upgrade pip wheel setuptools
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip/setuptools in $PrefixPath"
  }
  & $pythonExe -m pip install "torch" "torchvision" "torchaudio" --index-url $indexUrl
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to install torch stack in $PrefixPath"
  }
  & $pythonExe -c "import torch; print(torch.__version__)"
  if ($LASTEXITCODE -ne 0) {
    throw "Torch import check failed in $PrefixPath"
  }
}

function Install-PipExtras {
  param(
    [string]$PrefixPath,
    [bool]$InstallWandb = $true,
    [bool]$InstallMediapipe = $true
  )
  $pythonExe = Join-Path $PrefixPath "python.exe"
  if (-not (Test-Path $pythonExe)) {
    throw "python.exe not found in env: $PrefixPath"
  }

  $pkgs = @()
  if ($InstallMediapipe) {
    $pkgs += "mediapipe"
  }
  if ($InstallWandb) {
    $pkgs += "wandb"
  }
  if ($pkgs.Count -le 0) {
    return
  }

  Write-Step "Installing extra pip packages ($($pkgs -join ', '))"
  & $pythonExe -m pip install --upgrade @pkgs
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to install pip extras ($($pkgs -join ', ')) in $PrefixPath"
  }
}

Ensure-Conda
$EnvPrefix = Join-Path $EnvRoot $EnvName

Write-Step "Preparing folder layout"
Ensure-DirectoryLayout
Ensure-MetaTemplate

New-Item -ItemType Directory -Path $EnvRoot -Force | Out-Null

if ($Recreate -and (Env-Exists -PrefixPath $EnvPrefix)) {
  Write-Step "Removing existing conda env: $EnvPrefix"
  Invoke-Conda "env" "remove" "--prefix" $EnvPrefix "-y"
}

if (-not (Env-Exists -PrefixPath $EnvPrefix)) {
  Write-Step "Creating conda env: $EnvPrefix (python=$PythonVersion)"
  Invoke-Conda "create" "--prefix" $EnvPrefix "-y" "python=$PythonVersion" "--solver" "classic"
} else {
  Write-Step "Conda env already exists: $EnvPrefix"
}

if (Test-CoreDeps -PrefixPath $EnvPrefix -RequireMediapipe:$InstallMediapipe) {
  Write-Step "Core dependencies already present in $EnvPrefix (skip install)"
} else {
  Write-Step "Installing core dependencies into $EnvPrefix"
  Invoke-Conda "install" "--prefix" $EnvPrefix "-y" "-c" "conda-forge" "ffmpeg" "opencv" "moviepy" "jsonpickle" "pyyaml" "streamlit" "--solver" "classic"
}

if ($InstallTorch) {
  Install-TorchStack -PrefixPath $EnvPrefix -Channel $TorchChannel
}

Install-PipExtras -PrefixPath $EnvPrefix -InstallWandb:$InstallWandb -InstallMediapipe:$InstallMediapipe

$envsDirs = conda config --show envs_dirs | Out-String
if ($envsDirs -notmatch [regex]::Escape($EnvRoot)) {
  Write-Step "Registering env root for short-name activation"
  Invoke-Conda "config" "--append" "envs_dirs" $EnvRoot
}

Write-Host ""
Write-Host "[DONE] Environment is ready." -ForegroundColor Green
Write-Host "Conda env path: $EnvPrefix"
Write-Host ""
Write-Host "Next commands:"
Write-Host "1) conda activate $EnvName"
Write-Host "   (fallback: conda activate `"$EnvPrefix`")"
Write-Host "2) cd `"$ProjectRoot`""
Write-Host "3) python scripts/convert_via_to_master.py"
Write-Host "4) python scripts/validate_master_dataset.py"
Write-Host "5) python scripts/export_task_a_clips.py --trim-clips"
Write-Host "6) python scripts/export_task_b_activitynet.py"
Write-Host "7) streamlit run apps/streamlit_app.py"
