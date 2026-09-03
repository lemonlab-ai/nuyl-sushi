#!/usr/bin/env bash
set -euo pipefail

# One-shot Linux GPU server setup for NTNU Sushi experiments.
# Usage:
#   bash scripts/setup_gpu_server.sh
#
# Optional environment variables:
#   ENV_NAME=ntnu-sushi-gpu
#   PYTHON_VERSION=3.10
#   INSTALL_SYSTEM_DEPS=1
#   SKIP_ACTIONFORMER_BUILD=0
#   TORCH_INSTALL_CMD="<custom torch install command>"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${ENV_NAME:-ntnu-sushi-gpu}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
SKIP_ACTIONFORMER_BUILD="${SKIP_ACTIONFORMER_BUILD:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

# Repo-compatible default stack (ActionFormer + VideoMAEv2 + SlowFast + TSM):
# - torch 1.12.1 / torchvision 0.13.1 / torchaudio 0.12.1
# - CUDA wheel: cu116 (override by TORCH_INSTALL_CMD if needed)
DEFAULT_TORCH_INSTALL_CMD="pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116"
TORCH_INSTALL_CMD="${TORCH_INSTALL_CMD:-$DEFAULT_TORCH_INSTALL_CMD}"

if [[ "${INSTALL_SYSTEM_DEPS}" == "1" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "[INFO] Installing system dependencies via apt-get..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg build-essential
  else
    echo "[WARN] apt-get not found. Skipping system dependency installation."
  fi
fi

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[INFO] Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi
conda activate "${ENV_NAME}"

echo "[INFO] Installing base Python dependencies..."
python -m pip install --upgrade pip wheel setuptools
python -m pip install \
  numpy==1.23.5 pandas pyyaml jsonpickle moviepy opencv-python av \
  tensorboard tensorboardX scikit-learn tqdm yacs iopath simplejson joblib \
  einops matplotlib decord h5py wandb

echo "[INFO] Installing optional Gesture Coach dependency (mediapipe)..."
if ! python -m pip install mediapipe; then
  echo "[WARN] mediapipe install failed; Gesture Coach keypoint extraction may be unavailable on this server."
fi

echo "[INFO] Installing PyTorch using TORCH_INSTALL_CMD..."
echo "[RUN] ${TORCH_INSTALL_CMD}"
eval "${TORCH_INSTALL_CMD}"

echo "[INFO] Installing model-specific dependencies..."
if ! python -m pip install -r "${PROJECT_ROOT}/external/VideoMAEv2/requirements.txt"; then
  echo "[WARN] Full VideoMAEv2 requirements install failed. Installing fallback minimal deps..."
  python -m pip install \
    deepspeed==0.9.5 timm==0.4.12 einops pandas scipy matplotlib \
    opencv-python tensorboard==2.9.0 tensorboardX==1.8
fi
python -m pip install "deepspeed==0.9.5"
python -m pip install "git+https://github.com/facebookresearch/fvcore.git"
python -m pip install pytorchvideo
if ! python -m pip install fairscale; then
  echo "[WARN] fairscale install failed. SlowFast may still run for basic classification."
fi

if [[ "${SKIP_ACTIONFORMER_BUILD}" != "1" ]]; then
  echo "[INFO] Building ActionFormer NMS extension..."
  pushd "${PROJECT_ROOT}/external/actionformer_release/libs/utils" >/dev/null
  python setup.py install --user
  popd >/dev/null
fi

echo "[INFO] Applying dataset patches..."
python "${PROJECT_ROOT}/scripts/patch_tsm_ntnu_dataset.py" --repo "${PROJECT_ROOT}/external/temporal-shift-module"
python "${PROJECT_ROOT}/scripts/patch_videomaev2_ntnu.py" --repo "${PROJECT_ROOT}/external/VideoMAEv2"

echo "[INFO] Verifying key imports..."
python - <<'PY'
import importlib
import json

mods = [
    "torch",
    "deepspeed",
    "cv2",
    "yaml",
    "iopath",
    "av",
    "pytorchvideo",
    "tensorboardX",
]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, str(e)))

import torch
status = {
    "torch_version": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "missing_modules": missing,
}
print(json.dumps(status, indent=2))
if missing:
    raise SystemExit(1)
PY

echo "[DONE] GPU server environment is ready in conda env: ${ENV_NAME}"
echo "[NEXT] Activate env and run your training wrappers from ${PROJECT_ROOT}"
