param(
  [string]$BaseDir = "external"
)

$ErrorActionPreference = "Stop"

function Ensure-Repo {
  param(
    [string]$RepoUrl,
    [string]$FolderName
  )

  $target = Join-Path $BaseDir $FolderName
  if (Test-Path $target) {
    Write-Host "[SKIP] $FolderName already exists at $target"
    return
  }

  Write-Host "[CLONE] $RepoUrl -> $target"
  git clone $RepoUrl $target
}

New-Item -ItemType Directory -Path $BaseDir -Force | Out-Null

Ensure-Repo -RepoUrl "https://github.com/mit-han-lab/temporal-shift-module.git" -FolderName "temporal-shift-module"
Ensure-Repo -RepoUrl "https://github.com/facebookresearch/SlowFast.git" -FolderName "SlowFast"
Ensure-Repo -RepoUrl "https://github.com/OpenGVLab/VideoMAEv2.git" -FolderName "VideoMAEv2"
Ensure-Repo -RepoUrl "https://github.com/happyharrycn/actionformer_release.git" -FolderName "actionformer_release"

Write-Host ""
Write-Host "[DONE] Repositories are ready."
Write-Host "Next: create per-repo environments and run wrappers in ntnu-sushi/runners."
