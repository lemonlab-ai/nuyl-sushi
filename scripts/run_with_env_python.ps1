param(
  [Parameter(Mandatory = $true)]
  [string]$Script,
  [string[]]$ScriptArgs = @(),
  [string]$EnvName = "ntnu-sushi",
  [string]$EnvRoot = "C:\Users\Public\conda-envs"
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
  $scriptPath = $Script
  if (-not [System.IO.Path]::IsPathRooted($scriptPath)) {
    $scriptPath = Join-Path $ProjectRoot $Script
  }
  if (-not (Test-Path $scriptPath)) {
    throw "Script not found: $scriptPath"
  }

  Write-Host "[RUN] $PythonExe $scriptPath $($ScriptArgs -join ' ')" -ForegroundColor Cyan
  & $PythonExe $scriptPath @ScriptArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Script failed with exit code ${LASTEXITCODE}: $scriptPath"
  }
}
finally {
  Pop-Location
}
