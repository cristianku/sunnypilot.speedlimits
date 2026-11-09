#!/usr/bin/env bash
set -euo pipefail

# The review page's Fit box (key C, nnslr_tools.box_fit) trained again on the boxes people drew so far: the colour
# trees and the edge MLPs, on the CPU (numpy and OpenCV, no GPU, no network), about 15 minutes with 12 processes;
# then the box refiner (a small CNN, nnslr_tools.box_fit.refiner_train) on the CPU with the PyTorch of .venv-gpu,
# about an hour, written only when it beats the two rankers on routes left out.
#
# usage: scripts/train_box_fit.sh [--dry-run] [--no-refiner] [any other option of nnslr train-box-fit]
#
# 1. `nnslr train-box-fit --dry-run`: how many boxes, cases and route groups; the script stops there when there are
#    fewer boxes than the training needs ("min_boxes").
# 2. `nnslr train-box-fit`: the measure first (each group of routes fitted by models trained on the others,
#    --folds 4), then the two models trained on every box, written over src/nnslr_tools/box_fit/colour.npz and
#    edges.npz (each replaced in one step). The same boxes and seed give the same files.
# 3. The summary (boxes, measures by fold, kind, size and family) is kept in
#    $DATA_ROOT/evaluations/box_fit/<time>.json (nothing is left there when the training fails).
# 4. The box refiner: trained on the boxes of the frames people confirmed in the review (review.sqlite), the routes
#    of a quarter left out; on those, the rough box, the two rankers and the refiner are measured on the full frames,
#    and src/nnslr_tools/box_fit/refiner.onnx is replaced only when the refiner's mean IoU is higher, in all and on
#    the small signs. Its summary: $DATA_ROOT/evaluations/box_fit/refiner-<time>.json. --no-refiner skips it.
#
# The running site keeps the models it loaded: it uses the new ones after its next restart (helpers/update.sh, or
# the panel's restart). Nothing is imported, committed or published.
#
# Environment: NNSLR_DATA_ROOT, NNSLR_NUM_WORKERS (processes extracting the candidates; default half the cores, 4 to
# 12).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
CORES="$(nproc 2>/dev/null || echo 8)"
WORKERS_DEFAULT=$(( CORES / 2 < 4 ? 4 : (CORES / 2 > 12 ? 12 : CORES / 2) ))
WORKERS="${NNSLR_NUM_WORKERS:-$WORKERS_DEFAULT}"
NNSLR="$ROOT_DIR/.venv/bin/nnslr"
PYTHON="$ROOT_DIR/.venv/bin/python"

GPU_PYTHON="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}/bin/python"
REFINER=1
args=()
for arg in "$@"; do
  if [ "$arg" = "--no-refiner" ]; then REFINER=0; else args+=("$arg"); fi
done
set -- "${args[@]+"${args[@]}"}"
STEPS=$(( REFINER ? 3 : 2 ))

echo "1/$STEPS The boxes drawn by people (dry run)"
dry="$("$NNSLR" train-box-fit --data-root "$DATA_ROOT" --dry-run "$@")"
echo "$dry"
for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then
    if [ "$REFINER" = 1 ]; then
      echo "The boxes of the refiner (dry run)"
      if "$GPU_PYTHON" -c 'import torch, onnx, cv2' 2>/dev/null; then
        "$GPU_PYTHON" -m nnslr_tools.box_fit.refiner_train --data-root "$DATA_ROOT" --dry-run
      else
        echo "The PyTorch environment (.venv-gpu with torch, onnx and opencv) is missing: a real run skips the refiner until scripts/setup_v100_training_env.sh installs it."
      fi
    fi
    exit 0
  fi
done
if ! echo "$dry" | "$PYTHON" -c 'import json, sys; s = json.loads(sys.stdin.read().strip().splitlines()[-1]); sys.exit(s["boxes"] < s["min_boxes"])'; then
  echo "Too few boxes drawn by people to train the box fit: nothing changed." >&2
  exit 1
fi

echo "2/$STEPS Measure on routes left out, then train on every box ($WORKERS processes)"
mkdir -p "$DATA_ROOT/evaluations/box_fit"
summary="$DATA_ROOT/evaluations/box_fit/$(date -u +%Y%m%dT%H%M%SZ).json"
refined="$DATA_ROOT/evaluations/box_fit/refiner-$(date -u +%Y%m%dT%H%M%SZ).json"
trap 'rm -f "$summary.tmp" "$refined.tmp"' EXIT
"$NNSLR" train-box-fit --data-root "$DATA_ROOT" --workers "$WORKERS" "$@" > "$summary.tmp"
mv "$summary.tmp" "$summary"
cat "$summary"
echo "Summary in $summary."

if [ "$REFINER" = 1 ]; then
  echo "3/$STEPS The box refiner: train on the CPU, measure against the rankers on routes left out ($WORKERS threads)"
  if ! "$GPU_PYTHON" -c 'import torch, onnx, cv2' 2>/dev/null; then
    echo "The PyTorch environment (.venv-gpu with torch, onnx and opencv) is missing: run scripts/setup_v100_training_env.sh. The refiner is unchanged." >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES='' "$GPU_PYTHON" -m nnslr_tools.box_fit.refiner_train --data-root "$DATA_ROOT" \
    --workers "$WORKERS" > "$refined.tmp"
  mv "$refined.tmp" "$refined"
  cat "$refined"
  echo "Summary in $refined."
fi
echo "The site uses the new models after its next restart."
