#!/usr/bin/env bash
set -euo pipefail

# The weights of the road users' detector (nnslr road-users, nnslr_tools.road_users.DETECTOR): torchvision's Faster R-CNN
# ResNet-50 FPN v2 on COCO, about 167 MB from download.pytorch.org into torch's hub cache (~/.cache/torch/hub/checkpoints,
# or $TORCH_HOME), checked against the hash in its file name. The first `nnslr road-users` fetches them by itself: this
# only does it ahead, e.g. before a night run. Downloads only; nothing runs on a card.
#
# usage: scripts/fetch_road_users_detector.sh
# Environment: NNSLR_GPU_VENV (.venv-gpu: torch and torchvision).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
"$VENV/bin/python" - <<'PY'
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights as Weights
from torch.hub import get_dir
Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
print(f'road users detector: {Weights.DEFAULT.url} in {get_dir()}/checkpoints')
PY
