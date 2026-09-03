#!/usr/bin/env bash
set -euo pipefail

# GPU server smoke test for NTNU Sushi training stack.
# Default: dependency/import checks + runner dry-run checks.
# Optional: set RUN_REAL_LAUNCH=1 for short real launch checks.
#
# Usage:
#   conda activate ntnu-sushi-gpu
#   bash scripts/server_smoke_test.sh
#
# Optional env vars:
#   PYTHON_BIN=python
#   RUN_REAL_LAUNCH=0
#   LAUNCH_TIMEOUT_SEC=60

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_REAL_LAUNCH="${RUN_REAL_LAUNCH:-0}"
LAUNCH_TIMEOUT_SEC="${LAUNCH_TIMEOUT_SEC:-60}"

log() {
  echo "[SMOKE] $*"
}

run() {
  log "RUN: $*"
  "$@"
}

run_timeout() {
  local seconds="$1"
  shift
  log "RUN(timeout=${seconds}s): $*"
  set +e
  timeout "${seconds}" "$@"
  local code=$?
  set -e
  # timeout means command started and kept running; treat as pass for launch checks
  if [[ ${code} -eq 124 ]]; then
    log "PASS (timed out as expected): $*"
    return 0
  fi
  return ${code}
}

require_file() {
  local p="$1"
  if [[ ! -f "${p}" ]]; then
    echo "[ERROR] Missing file: ${p}" >&2
    exit 1
  fi
}

cd "${PROJECT_ROOT}"

log "Project root: ${PROJECT_ROOT}"
run "${PYTHON_BIN}" --version

log "Checking required artifacts/templates..."
require_file "${PROJECT_ROOT}/artifacts/task_b/actionformer_fine_config.yaml"
require_file "${PROJECT_ROOT}/artifacts/task_b/actionformer_fine.json"
require_file "${PROJECT_ROOT}/artifacts/task_b/features_basic/feature_meta.json"
require_file "${PROJECT_ROOT}/artifacts/task_a/repo_formats/tsm/train_videofolder.txt"
require_file "${PROJECT_ROOT}/artifacts/task_a/repo_formats/tsm/val_videofolder.txt"
require_file "${PROJECT_ROOT}/artifacts/task_a/repo_formats/tsm/category.txt"
require_file "${PROJECT_ROOT}/artifacts/task_a/repo_formats/videomaev2/train.csv"
require_file "${PROJECT_ROOT}/templates/slowfast/NTNU_SUSHI_SLOWFAST.yaml"

log "Checking base imports..."
"${PYTHON_BIN}" - <<'PY'
import importlib
mods = [
    "torch",
    "torchvision",
    "torchaudio",
    "deepspeed",
    "cv2",
    "yaml",
    "iopath",
    "av",
    "pytorchvideo",
]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, str(e)))
if missing:
    print("[ERROR] Missing imports:", missing)
    raise SystemExit(1)
print("[OK] Base imports all good.")
PY

log "Checking ActionFormer extension import..."
PYTHONPATH="${PROJECT_ROOT}/external/actionformer_release/libs/utils:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" - <<'PY'
import nms_1d_cpu
print("[OK] nms_1d_cpu import good.")
PY

log "Checking SlowFast package import..."
PYTHONPATH="${PROJECT_ROOT}/external/SlowFast:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" - <<'PY'
from slowfast.config.defaults import get_cfg
_ = get_cfg()
print("[OK] SlowFast import good.")
PY

log "Checking runner dry-runs..."
NUM_CLASSES="$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/task_a/class_map_fine.json")
obj = json.loads(p.read_text(encoding="utf-8"))
print(len(obj))
PY
)"

run "${PYTHON_BIN}" runners/train_actionformer.py \
  --repo external/actionformer_release \
  --config artifacts/task_b/actionformer_fine_config.yaml \
  --output-name smoke_ntnu \
  --dry-run

run "${PYTHON_BIN}" runners/train_tsm.py \
  --repo external/temporal-shift-module \
  --dataset ntnu_sushi \
  --root-path artifacts/task_a/repo_formats/tsm/frames \
  --train-list artifacts/task_a/repo_formats/tsm/train_videofolder.txt \
  --val-list artifacts/task_a/repo_formats/tsm/val_videofolder.txt \
  --category-file artifacts/task_a/repo_formats/tsm/category.txt \
  --dry-run

run "${PYTHON_BIN}" runners/train_slowfast.py \
  --repo external/SlowFast \
  --cfg templates/slowfast/NTNU_SUSHI_SLOWFAST.yaml \
  --data-dir artifacts/task_a \
  --output-dir outputs/slowfast_smoke \
  --dry-run

run "${PYTHON_BIN}" runners/train_videomaev2.py \
  --repo external/VideoMAEv2 \
  --data-path artifacts/task_a/repo_formats/videomaev2 \
  --data-root artifacts/task_a \
  --output-dir outputs/videomaev2_smoke \
  --nb-classes "${NUM_CLASSES}" \
  --dry-run

if [[ "${RUN_REAL_LAUNCH}" == "1" ]]; then
  log "RUN_REAL_LAUNCH=1 -> running short real launch checks..."
  run_timeout "${LAUNCH_TIMEOUT_SEC}" "${PYTHON_BIN}" runners/train_actionformer.py \
    --repo external/actionformer_release \
    --config artifacts/task_b/actionformer_fine_config.yaml \
    --output-name smoke_ntnu_real

  run_timeout "${LAUNCH_TIMEOUT_SEC}" "${PYTHON_BIN}" runners/train_tsm.py \
    --repo external/temporal-shift-module \
    --dataset ntnu_sushi \
    --root-path artifacts/task_a/repo_formats/tsm/frames \
    --train-list artifacts/task_a/repo_formats/tsm/train_videofolder.txt \
    --val-list artifacts/task_a/repo_formats/tsm/val_videofolder.txt \
    --category-file artifacts/task_a/repo_formats/tsm/category.txt \
    --epochs 1 \
    --batch-size 2 \
    --workers 0

  run_timeout "${LAUNCH_TIMEOUT_SEC}" "${PYTHON_BIN}" runners/train_slowfast.py \
    --repo external/SlowFast \
    --cfg templates/slowfast/NTNU_SUSHI_SLOWFAST.yaml \
    --data-dir artifacts/task_a \
    --output-dir outputs/slowfast_smoke_real \
    --num-gpus 1

  run_timeout "${LAUNCH_TIMEOUT_SEC}" "${PYTHON_BIN}" runners/train_videomaev2.py \
    --repo external/VideoMAEv2 \
    --data-path artifacts/task_a/repo_formats/videomaev2 \
    --data-root artifacts/task_a \
    --output-dir outputs/videomaev2_smoke_real \
    --nb-classes "${NUM_CLASSES}" \
    --epochs 1 \
    --batch-size 2
fi

log "DONE. Server smoke test passed."

