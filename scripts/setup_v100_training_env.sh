#!/usr/bin/env bash
set -euo pipefail

# Reproducible NNSLR training environment for NVIDIA Tesla V100 (Volta/sm_70).
# This script never installs/replaces NVIDIA drivers and does not run a GPU
# workload. Run "nnslr env --gpu-smoke" explicitly after installation.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
PYTHON="${NNSLR_GPU_PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: python not found: $PYTHON" >&2
  exit 2
fi

# Triton (inside torch) compiles a small C helper with gcc the first time it runs a kernel of its own: torch 2.14
# sends some eager ops to it (the backward of the Fit box rankers' einsum), and that helper needs this Python's
# headers (Python.h). Debian ships them apart, in python3.X-dev.
include="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
if [ ! -f "$include/Python.h" ]; then
  if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null 2>&1; then
    apt-get install -y "python$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')-dev"
  else
    echo "error: $include/Python.h missing: install the headers of $PYTHON (Debian: apt-get install python3.X-dev)" >&2
    exit 2
  fi
fi

# A venv left half made by a failed run (no pip inside) is made again.
if [ -d "$VENV" ] && [ ! -x "$VENV/bin/pip" ]; then
  echo "rifaccio $VENV: creato a metà da un tentativo fallito"
  rm -rf "$VENV"
fi
if [ ! -d "$VENV" ]; then
  # uv when present: it needs no ensurepip, which Debian ships apart (python3.X-venv, not installed here).
  # Also from ~/.local/bin, which is not on the PATH of a job started by the speedlimits-web service.
  UV="$(command -v uv || true)"
  home="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"
  if [ -z "$UV" ] && [ -x "$home/.local/bin/uv" ]; then UV="$home/.local/bin/uv"; fi
  if [ -n "$UV" ]; then
    "$UV" venv --seed --python "$(command -v "$PYTHON")" "$VENV"
  else
    "$PYTHON" -m venv "$VENV"
  fi
fi
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

# PyTorch 2.14 is the final release line publishing CUDA 12.6 binaries for
# Maxwell/Pascal/Volta. V100 needs Volta/sm_70; CUDA 13 binaries are not valid
# for this target.
"$VENV/bin/python" -m pip install   torch==2.14.0   torchvision==0.29.0   --index-url https://download.pytorch.org/whl/cu126

# OpenCV: the crops of the box refiner of Fit box (nnslr_tools.box_fit.refiner_train), as the review site makes them.
"$VENV/bin/python" -m pip install 'Pillow>=11,<13' onnx onnxscript onnxruntime 'opencv-python-headless>=4.10,<6'
# The "training" extra of pyproject.toml adds wandb (training-run tracking).
"$VENV/bin/python" -m pip install -e "${ROOT_DIR}[training]"

"$VENV/bin/python" - <<'PY'
import torch
import torchvision
import onnx
import onnxruntime
import onnxscript
import wandb

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("onnx:", onnx.__version__)
print("onnxscript:", onnxscript.__version__)
print("onnxruntime:", onnxruntime.__version__)
print("wandb:", wandb.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("compiled architectures:", torch.cuda.get_arch_list() if torch.cuda.is_available() else [])
print()
print("No GPU workload was run.")
print("Next: source .venv-gpu/bin/activate && nnslr env --gpu-smoke --gpu-device 0")
print("wandb: fai una volta 'wandb login' (dentro .venv-gpu); senza login il training va offline.")
PY
