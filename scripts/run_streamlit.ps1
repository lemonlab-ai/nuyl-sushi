param(
  [string]$EnvName = "ntnu-sushi",
  [string]$EnvRoot = "C:\Users\Public\conda-envs",
  [int]$Port = 8501,
  [string]$Address = "127.0.0.1"
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
  Write-Host "[RUN] $PythonExe -m streamlit run apps/streamlit_app.py --server.address $Address --server.port $Port" -ForegroundColor Cyan
  & $PythonExe -m streamlit run "apps/streamlit_app.py" "--server.address" "$Address" "--server.port" "$Port"
  if ($LASTEXITCODE -ne 0) {
    throw "Streamlit exited with code $LASTEXITCODE"
  }
}
finally {
  Pop-Location
}
