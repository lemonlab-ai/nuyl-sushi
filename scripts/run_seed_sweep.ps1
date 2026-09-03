param(
  [string]$EnvName = "ntnu-sushi",
  [string]$EnvRoot = "C:\Users\Public\conda-envs",
  [string]$Seeds = "42,43,44",
  [string]$ModelName = "multimodal_baseline",
  [string]$OutputRoot = "outputs/seed_sweeps",
  [string]$AggregateCsvName = "seed_sweep_aggregate.csv",
  [switch]$ExportPaperTable = $false,
  [string]$PaperTableCsv = "",
  [string]$PaperTableMd = "",
  [switch]$ContinueOnError = $false,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$RunArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $ScriptDir "run_with_env_python.ps1"

$argsList = @(
  "--seeds", "$Seeds",
  "--model-name", "$ModelName",
  "--output-root", "$OutputRoot",
  "--aggregate-csv-name", "$AggregateCsvName"
)
if ($ContinueOnError) { $argsList += "--continue-on-error" }
if ($ExportPaperTable) { $argsList += "--export-paper-table" }
if ($PaperTableCsv) { $argsList += @("--paper-table-csv", "$PaperTableCsv") }
if ($PaperTableMd) { $argsList += @("--paper-table-md", "$PaperTableMd") }
if ($RunArgs) { $argsList += $RunArgs }

& $Runner `
  -EnvName $EnvName `
  -EnvRoot $EnvRoot `
  -Script "scripts/run_seed_sweep.py" `
  -ScriptArgs $argsList
