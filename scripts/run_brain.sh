#!/usr/bin/env bash
set -euo pipefail

# The brain (docs/NEURAL_NETWORK.md): the network that says the speed to hold, four times a second, from the picture and the
# car's own signals, learnt from every drive with its logs read: the driver's own speed, corrected by the blocks of
# Speed review, the roadworks marked there and the next limit of the signs of Sign review. On one GPU, one step after
# the other:
#
#   1. checks the labels, the split and the whole training on the CPU (nnslr train-brain --dry-run);
#   2. a CUDA smoke test;
#   3. reads the signs of every drive it learns from as the car does today (nnslr road-type-signs --labelled; a drive
#      already read by the car's reader of today is not read again: a new vision.py or a new network reads them all
#      again, and the training refuses anything else), then says what the labels give (nnslr brain-targets);
#   4-6. trains seeds 0, 1 and 2, each into a run of its own, runs/NAME-s0, -s1, -s2 (nnslr train-brain, the last epoch
#      kept): three versions in Models › Brain, which differ only by chance. Train › Brain draws them on one chart.
#
# usage: scripts/run_brain.sh NAME
#
# Writes only into $DATA_ROOT/runs/NAME/ (brain.log, dry-run/), $DATA_ROOT/runs/NAME-s{0,1,2}/ (brain-last.pt, result.json),
# the sign readings and the frame cache. It never imports reviews, changes the labels, the model in use or the car, and
# exports nothing: Publish exports the brain chosen in Models and checks it along a real drive. Run it again with the same
# NAME to finish: a finished seed is not repeated, the signs already read are not read again.
#
# Environment: NNSLR_GPU_INDEX (default 0), NNSLR_GPU_VENV (.venv-gpu), NNSLR_BRAIN_EPOCHS (40), NNSLR_SIGN_WORKERS
# (processes reading the signs: half the cores, each with its ffmpeg), NNSLR_BRAIN_WORKS (1: it also learns the
# roadworks; 0: not), NNSLR_BRAIN_DRIVER (0, the usual since 8 Nov 2026: without the driver's inputs; 1: also the gas,
# the brake and the car's speed),
# NNSLR_BRAIN_POOL (grid: the picture's features keep where they are; mean: not), NNSLR_SUNNYPILOT_CLONE
# (/root/sunnypilot, on the speed-limits branch), NNSLR_ALLOW_BUSY_GPU=1 to share a GPU that already has compute processes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
RUN_ID="${1:-}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
EPOCHS="${NNSLR_BRAIN_EPOCHS:-40}"
CORES="$(nproc 2>/dev/null || echo 8)"
SIGN_WORKERS_DEFAULT=$(( CORES / 2 > 1 ? CORES / 2 : 1 ))
SIGN_WORKERS="${NNSLR_SIGN_WORKERS:-$SIGN_WORKERS_DEFAULT}"
WORKS="${NNSLR_BRAIN_WORKS:-1}"
DRIVER="${NNSLR_BRAIN_DRIVER:-0}"
POOL="${NNSLR_BRAIN_POOL:-grid}"
SEEDS=(0 1 2)

if [[ -z "$RUN_ID" || $# -ne 1 ]]; then
  echo "usage: $0 NAME" >&2
  echo "example: NNSLR_GPU_INDEX=0 $0 2026-11-05_1500-brain" >&2
  exit 2
fi

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: NAME contains unsupported characters" >&2
  exit 2
fi

for setting in EPOCHS SIGN_WORKERS; do
  if [[ ! "${!setting}" =~ ^[0-9]+$ ]]; then
    echo "error: $setting must be a whole number: ${!setting}" >&2
    exit 2
  fi
done

if [[ "$WORKS" != "0" && "$WORKS" != "1" ]]; then
  echo "error: NNSLR_BRAIN_WORKS must be 0 or 1: $WORKS" >&2
  exit 2
fi

if [[ "$DRIVER" != "0" && "$DRIVER" != "1" ]]; then
  echo "error: NNSLR_BRAIN_DRIVER must be 0 or 1: $DRIVER" >&2
  exit 2
fi

if [[ "$POOL" != "mean" && "$POOL" != "grid" ]]; then
  echo "error: NNSLR_BRAIN_POOL must be mean or grid: $POOL" >&2
  exit 2
fi

if [[ ! -x "$VENV/bin/nnslr" || ! -x "$VENV/bin/python" ]]; then
  echo "error: GPU environment missing: $VENV/bin/nnslr" >&2
  echo "run: bash scripts/setup_v100_training_env.sh" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi not found" >&2
  exit 2
fi

if ! nvidia-smi -i "$GPU_INDEX" >/dev/null 2>&1; then
  echo "error: invalid/unavailable GPU index: $GPU_INDEX" >&2
  exit 2
fi

BUSY="$(
  nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "$BUSY" && "${NNSLR_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "error: selected GPU already has compute processes; refusing to train the brain" >&2
  echo "$BUSY" >&2
  echo "Set NNSLR_ALLOW_BUSY_GPU=1 only after intentionally accepting contention." >&2
  exit 3
fi

RUN_ROOT="$DATA_ROOT/runs/$RUN_ID"
# A folder of another kind of run is never written into.
for part in detector reader export bundle experiment-b detector-sizes classifier road-type; do
  if [[ -e "$RUN_ROOT/$part" ]]; then
    echo "error: $RUN_ROOT is another kind of run (it has $part/): choose a new NAME" >&2
    exit 3
  fi
done
# SignNet (the cutter) writes its weights at the root of its run, not in a folder.
if [[ -e "$RUN_ROOT/signnet-best.pt" ]]; then
  echo "error: $RUN_ROOT is another kind of run (it has signnet-best.pt, a SignNet): choose a new NAME" >&2
  exit 3
fi

mkdir -p "$RUN_ROOT"
# Everything printed from here on also goes to the run's own log (the panel reads its steps and epochs).
exec > >(tee -a "$RUN_ROOT/brain.log") 2>&1

export NNSLR_DATA_ROOT="$DATA_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
NN="$VENV/bin/nnslr"
SETTINGS=(--epochs "$EPOCHS" --pool "$POOL")
if [[ "$WORKS" == "0" ]]; then
  SETTINGS+=(--no-works)
fi
if [[ "$DRIVER" == "1" ]]; then
  SETTINGS+=(--driver)
fi

echo "== Brain =="
echo "run:          $RUN_ID"
echo "output:       $RUN_ROOT-s{${SEEDS[0]},${SEEDS[1]},${SEEDS[2]}}"
echo "data root:    $DATA_ROOT"
echo "physical GPU: $GPU_INDEX"
echo "settings:     ${SETTINGS[*]}"
echo

echo "== 1/6 Check the labels, the split and the training (CPU-only) =="
CUDA_VISIBLE_DEVICES="" "$NN" train-brain --dry-run --run-dir "$RUN_ROOT/dry-run" "${SETTINGS[@]}"

echo "== 2/6 Explicit CUDA smoke test =="
"$NN" env --gpu-smoke --gpu-device 0

echo "== 3/6 Read the signs of the drives as the car does today =="
# The signs on the GPU: .venv has onnxruntime-gpu, .venv-gpu only the CPU's (torch brings its own CUDA).
SIGNS_NN="$ROOT_DIR/.venv/bin/nnslr"
[ -x "$SIGNS_NN" ] || SIGNS_NN="$NN"
"$SIGNS_NN" road-type-signs --labelled --num-workers "$SIGN_WORKERS" --data-root "$DATA_ROOT"
"$NN" brain-targets --data-root "$DATA_ROOT"

step=4
for seed in "${SEEDS[@]}"; do
  echo "== $step/6 Train seed $seed =="
  step=$(( step + 1 ))
  if [[ -f "$RUN_ROOT-s$seed/result.json" ]]; then
    echo "skip seed $seed: already trained"
    continue
  fi
  "$NN" train-brain --run-dir "$RUN_ROOT-s$seed" --seed "$seed" "${SETTINGS[@]}"
done

echo "Brain complete: choose a seed in Models › Brain, then Publish."
