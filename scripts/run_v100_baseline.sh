#!/usr/bin/env bash
set -euo pipefail

# Execute the complete NNSLR baseline pipeline on one explicitly selected V100.
# This script never kills competing GPU processes and never deploys anything to
# Sunnypilot or a comma device.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
RUN_ID="${1:-}"
MODE="${2:-${NNSLR_TRAIN_MODE:-fresh}}"
DATASET_KIND="${NNSLR_DATASET_KIND:-gold}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"

if [[ -z "$RUN_ID" ]]; then
  echo "usage: $0 RUN_ID [fresh|continue|resume]" >&2
  echo "example: NNSLR_GPU_INDEX=0 $0 baseline-001" >&2
  exit 2
fi

if [[ "$MODE" != "fresh" && "$MODE" != "continue" && "$MODE" != "resume" ]]; then
  echo "error: training mode must be fresh, continue or resume" >&2
  exit 2
fi

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: RUN_ID contains unsupported characters" >&2
  exit 2
fi

if [[ "$DATASET_KIND" != "gold" && "$DATASET_KIND" != "training-candidate" ]]; then
  echo "error: NNSLR_DATASET_KIND must be gold or training-candidate" >&2
  exit 2
fi

if [[ ! -x "$VENV/bin/nnslr" ]]; then
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
  nvidia-smi -i "$GPU_INDEX"     --query-compute-apps=pid,process_name,used_gpu_memory     --format=csv,noheader,nounits 2>/dev/null     | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "$BUSY" && "${NNSLR_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "error: selected GPU already has compute processes; refusing to start training" >&2
  echo "$BUSY" >&2
  echo "Set NNSLR_ALLOW_BUSY_GPU=1 only after intentionally accepting contention." >&2
  exit 3
fi

RUN_ROOT="$DATA_ROOT/runs/$RUN_ID"
if [[ "$MODE" == "resume" ]]; then
  if [[ ! -d "$RUN_ROOT" ]]; then
    echo "error: no existing run to resume: $RUN_ROOT" >&2
    exit 3
  fi
elif [[ -e "$RUN_ROOT" ]]; then
  echo "error: run directory already exists: $RUN_ROOT" >&2
  exit 3
fi

DETECTOR_RUN="$RUN_ROOT/detector"
READER_RUN="$RUN_ROOT/reader"
EVAL_DIR="$RUN_ROOT/evaluation"
EXPORT_DIR="$RUN_ROOT/export"
BUNDLE_DIR="$RUN_ROOT/bundle"
mkdir -p "$EVAL_DIR" "$EXPORT_DIR"

export NNSLR_DATA_ROOT="$DATA_ROOT"
# The index is nvidia-smi's, the card checked above: CUDA counts the cards in the same order, by PCI bus (its
# default order, fastest first, leaves two identical V100s in no stated order).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
# The wandb group of the detector and reader runs (nnslr_tools.tracking).
export NNSLR_RUN_ID="$RUN_ID"

# wandb: "wandb login" saves the key in ~/.netrc; without it, offline.
# WANDB_MODE=disabled turns wandb off completely.
if [ -z "${WANDB_MODE:-}" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  if ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    export WANDB_MODE=offline
    echo "wandb: non loggato, vado offline (i numeri restano nel terminale e in $RUN_ROOT/<task>/wandb)."
  fi
fi

NN="$VENV/bin/nnslr"

echo "== NNSLR baseline pipeline =="
echo "run:          $RUN_ID"
echo "training mode: $MODE"
echo "data root:    $DATA_ROOT"
echo "dataset kind: $DATASET_KIND"
echo "physical GPU: $GPU_INDEX"
echo "wandb:        progetto ${WANDB_PROJECT:-sunnypilot.speedlimits}, gruppo $RUN_ID, modo ${WANDB_MODE:-online}"
echo

echo "== 1/10 Validate exact training populations (CPU-only) =="
"$NN" train --task detector --dataset-kind "$DATASET_KIND" --dry-run
"$NN" train --task reader --dataset-kind "$DATASET_KIND" --dry-run

echo "== 2/10 Validate held-out evaluation populations (CPU-only) =="
"$NN" evaluate --task detector --dataset-kind "$DATASET_KIND" --partition test --dry-run
"$NN" evaluate --task reader --dataset-kind "$DATASET_KIND" --partition test --dry-run

echo "== 3/10 Explicit CUDA smoke test =="
# CUDA_VISIBLE_DEVICES maps the selected physical GPU to logical cuda:0.
"$NN" env --gpu-smoke --gpu-device 0

train_task() {
  local task="$1"
  local output="$2"
  local selected_mode="$MODE"
  if [[ "$MODE" == "resume" ]]; then
    if [[ -f "$output/result.json" ]]; then
      echo "skip completed $task: $output/result.json"
      return 0
    fi
    if [[ -f "$output/training-state.pt" ]]; then
      selected_mode="resume"
    elif [[ ! -e "$output" ]]; then
      # The previous pipeline was interrupted before this task started.
      selected_mode="fresh"
    else
      echo "error: $task output exists without a valid training state: $output" >&2
      return 3
    fi
  fi
  # Where a fresh run starts (the panel's choice, through the environment of helpers/3_train.sh):
  # NNSLR_TRAIN_FROM_RUN=<run> continues from the best weights of a finished run of this server;
  # otherwise MobileNetV3 trained on ImageNet (torchvision, BSD), or random weights with
  # NNSLR_TRAIN_INIT=random. continue/resume start from their own weights.
  local init=()
  if [[ "$selected_mode" == "fresh" ]]; then
    if [[ -n "${NNSLR_TRAIN_FROM_RUN:-}" ]]; then
      selected_mode="continue"
      init=(--init-from "$DATA_ROOT/runs/$NNSLR_TRAIN_FROM_RUN/$task/$task-best.pt")
    elif [[ "${NNSLR_TRAIN_INIT:-imagenet}" != "random" ]]; then
      init=(--pretrained)
    fi
  elif [[ "$selected_mode" == "resume" ]]; then
    # A resumed run trains with exactly the settings it started with (nnslr train refuses others), --pretrained
    # of a start from ImageNet included: they come from its own training state.
    local flags
    flags="$("$VENV/bin/python" -c 'import sys; from nnslr_tools.training import resume_arguments_file; print("\n".join(resume_arguments_file(sys.argv[1])))' "$output/training-state.pt")"
    readarray -t init <<< "$flags"
    echo "resume $task with the settings of its training state: ${init[*]}"
  fi
  "$NN" train --task "$task" --dataset-kind "$DATASET_KIND" --device cuda \
    --mode "$selected_mode" --output "$output" "${init[@]}"
}

echo "== 4/10 Train detector =="
train_task detector "$DETECTOR_RUN"

echo "== 5/10 Train reader =="
train_task reader "$READER_RUN"

echo "== 6/10 Evaluate held-out detector and reader =="
"$NN" evaluate   --task detector   --dataset-kind "$DATASET_KIND"   --partition test   --device cuda   --checkpoint "$DETECTOR_RUN/detector-best.pt"   --output "$EVAL_DIR/detector-test.json"
"$NN" evaluate   --task reader   --dataset-kind "$DATASET_KIND"   --partition test   --device cuda   --checkpoint "$READER_RUN/reader-best.pt"   --output "$EVAL_DIR/reader-test.json"

echo "== 7/10 Mine hard examples =="
"$NN" mine-hard-examples "$EVAL_DIR/detector-test.json" --output "$EVAL_DIR/detector-hard.jsonl"
"$NN" mine-hard-examples "$EVAL_DIR/reader-test.json" --output "$EVAL_DIR/reader-hard.jsonl"

echo "== 8/10 Export and parity-check ONNX =="
"$NN" export-onnx   --task detector   --checkpoint "$DETECTOR_RUN/detector-best.pt"   --output "$EXPORT_DIR/detector.onnx"
"$NN" export-onnx   --task reader   --checkpoint "$READER_RUN/reader-best.pt"   --output "$EXPORT_DIR/reader.onnx"

echo "== 9/10 Package immutable verified bundle =="
if [[ "$MODE" == "resume" && -d "$BUNDLE_DIR" ]]; then
  echo "bundle exists: verifying instead of overwriting"
else
  "$NN" package-model   --detector-onnx "$EXPORT_DIR/detector.onnx"   --detector-contract "$EXPORT_DIR/detector.onnx.contract.json"   --reader-onnx "$EXPORT_DIR/reader.onnx"   --reader-contract "$EXPORT_DIR/reader.onnx.contract.json"   --output "$BUNDLE_DIR"
fi

echo "== 10/10 Verify final bundle =="
"$NN" verify-bundle "$BUNDLE_DIR"

echo
echo "Pipeline complete: $RUN_ROOT"
echo "Nothing was deployed to Sunnypilot or a comma device."
