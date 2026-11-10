#!/usr/bin/env bash
set -euo pipefail

# Models › Regression (nnslr_tools.regression): the models he chose along the certified drives, only what is still missing,
# so after a training only the new model runs. On one GPU, one step after the other:
#
#   1. the driving models, drive by drive (helpers/route_speed.sh of openpilot.distill, one decoding of the video for all);
#   2. the signs of the certified drives as the car reads them, with the networks each brain learnt from (nnslr
#      regression-sign-networks, then road-type-signs --certified [--sign-networks]; a drive already read by them is not
#      read again);
#   3. the brains, along the whole drive as on the car (nnslr regression-brain).
#
# usage: NNSLR_REGRESSION_OPENPILOT=stock,teacher NNSLR_REGRESSION_BRAINS=<run>,<run> scripts/run_regression.sh
#
# Writes only into $DATA_ROOT/regression/results/<route>/, the sign readings and the frame cache. It trains nothing and never
# changes a label, the model in use or the car. A drive that fails stops only itself: the others go on and the job ends
# failed; running it again computes what is missing.
#
# Environment: NNSLR_GPU_INDEX (default 0), NNSLR_GPU_VENV (.venv-gpu), NNSLR_SIGN_WORKERS (half the cores),
# NNSLR_DISTILL_ROOT (openpilot.distill next to this checkout), NNSLR_ALLOW_BUSY_GPU=1 to share a GPU that already has compute
# processes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
DISTILL="${NNSLR_DISTILL_ROOT:-$ROOT_DIR/../openpilot.distill}"
OPENPILOT="${NNSLR_REGRESSION_OPENPILOT:-}"
BRAINS="${NNSLR_REGRESSION_BRAINS:-}"
CORES="$(nproc 2>/dev/null || echo 8)"
SIGN_WORKERS_DEFAULT=$(( CORES / 2 > 1 ? CORES / 2 : 1 ))
SIGN_WORKERS="${NNSLR_SIGN_WORKERS:-$SIGN_WORKERS_DEFAULT}"
GPU_LOCK="/run/lock/v100-gpu.lock"

if [[ $# -ne 0 ]]; then
  echo "usage: NNSLR_REGRESSION_OPENPILOT=<specs> NNSLR_REGRESSION_BRAINS=<runs> $0" >&2
  exit 2
fi

if [[ -z "$OPENPILOT" && -z "$BRAINS" ]]; then
  echo "error: no model chosen: set NNSLR_REGRESSION_OPENPILOT or NNSLR_REGRESSION_BRAINS" >&2
  exit 2
fi

NN="$ROOT_DIR/.venv/bin/nnslr"
if [[ ! -x "$NN" ]]; then
  echo "error: environment missing: $NN" >&2
  exit 2
fi
if [[ -n "$BRAINS" && ! -x "$VENV/bin/nnslr" ]]; then
  echo "error: GPU environment missing: $VENV/bin/nnslr" >&2
  echo "run: bash scripts/setup_v100_training_env.sh" >&2
  exit 2
fi
if [[ -n "$OPENPILOT" && ! -x "$DISTILL/helpers/route_speed.sh" ]]; then
  echo "error: helpers/route_speed.sh not found in $DISTILL" >&2
  exit 2
fi

BUSY="$(
  nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "$BUSY" && "${NNSLR_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "error: selected GPU already has compute processes; refusing the regression" >&2
  echo "$BUSY" >&2
  exit 3
fi

# The GPU lock of openpilot.distill, held for the whole run: its helpers see it as theirs (DISTILL_GPU_LOCK_PID, fd 9), and
# no training of that clone starts between two drives.
exec 9>>"$GPU_LOCK"
if ! flock -n 9; then
  echo "error: the GPU is busy with another job (lock $GPU_LOCK taken)" >&2
  exit 3
fi
export DISTILL_GPU_LOCK_PID=$$

export NNSLR_DATA_ROOT="$DATA_ROOT"
# The index is nvidia-smi's, the card checked above: CUDA counts the cards in the same order, by PCI bus (its
# default order, fastest first, leaves two identical V100s in no stated order).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
# The service's PATH has no uv, which runs openpilot.distill's helpers.
export PATH="$HOME/.local/bin:$PATH"

echo "== Regression =="
echo "driving models: ${OPENPILOT:-none}"
echo "brains:         ${BRAINS:-none}"
echo "data root:      $DATA_ROOT"
echo

failed=0

echo "== 1/3 Run the driving models on the certified drives =="
mapfile -t PLAN < <("$NN" regression-plan --openpilot "$OPENPILOT" --data-root "$DATA_ROOT")
position=0
for line in "${PLAN[@]}"; do
  read -r route specs <<< "$line"
  echo "regression item openpilot $position/${#PLAN[@]} $route"
  position=$(( position + 1 ))
  models=()
  for spec in $specs; do
    models+=("--model=$spec")
  done
  "$DISTILL/helpers/route_speed.sh" "--route=$route" "${models[@]}" "--out-dir=$DATA_ROOT/regression/results" || {
    echo "$route: the driving models failed"
    failed=1
  }
done
[[ ${#PLAN[@]} -gt 0 ]] || echo "nothing missing"

if [[ -n "$BRAINS" ]]; then
  echo "== 2/3 Read the signs of the certified drives as the car does today =="
  # Each brain's signs read by the networks it learnt from (#31): '-' is the car's release of the sunnypilot clone.
  NETWORKS_OUT=$("$NN" regression-sign-networks --brains "$BRAINS" --data-root "$DATA_ROOT") || failed=1
  mapfile -t NETWORKS <<< "$NETWORKS_OUT"
  for networks in "${NETWORKS[@]}"; do
    [[ -n "$networks" ]] || continue
    NETWORK_ARGS=()
    [[ "$networks" == "-" ]] || NETWORK_ARGS=(--sign-networks "$networks")
    "$NN" road-type-signs --certified --num-workers "$SIGN_WORKERS" --data-root "$DATA_ROOT" "${NETWORK_ARGS[@]}"
  done
  echo "== 3/3 Run the brains along the certified drives =="
  "$VENV/bin/nnslr" regression-brain --brains "$BRAINS" --data-root "$DATA_ROOT" || failed=1
else
  echo "== 2/3 Read the signs: no brain chosen =="
  echo "== 3/3 Run the brains: no brain chosen =="
fi

if [[ "$failed" != "0" ]]; then
  echo "Regression: some drives failed, the others are done: Details has them, running it again computes what is missing."
  exit 1
fi
echo "Regression complete: open a drive in Models › Regression."
