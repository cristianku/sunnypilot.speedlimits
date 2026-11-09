#!/usr/bin/env bash
set -euo pipefail

# The detector's input size (docs/NEURAL_NETWORK.md, step 3 of the plan): how many signs of each width the
# detector finds when the frame is resized to a larger image inside the model, and what that costs. Trains one
# detector per input size, one after the other on one GPU, the same way and on the same frozen split; evaluates
# each on test and on validation; then prints a comparison, also saved as comparison.txt.
#
# The detector was the cutter of the pipeline (cutter -> classifier -> decider): a sign it misses is lost, a false
# box the classifier can still reject. So the comparison shows recall first: at each score threshold, lowest
# first (0.05, 0.1, the report's 0.25, 0.3, 0.5), recall overall and by the width of the reviewed box, the boxes
# per frame passed on to the classifier and the false ones among them, then precision and F1; on all the frames
# and on the frames a person confirmed. Then the cost of each size: milliseconds per frame of the model alone,
# seconds per training epoch, GPU memory, and the epoch the validation chose.
#
# An old cutter, kept to compare: since 30 Oct 2026 the one cutter is SignNet, the network the comma runs, and the
# portal's "Train the cutter" is scripts/run_signnet.sh. The detectors this script trains stay on the Models tab as
# old versions to compare with; they can no longer be put in use (Models > Use this one refuses them).
#
# usage: scripts/run_detector_sizes.sh NAME [SIZES...]      sizes WIDTHxHEIGHT, default 320x320 640x360 960x544
#
# Writes only into $DATA_ROOT/runs/NAME/detector-sizes/: <size>/ (nnslr train --output: training-plan.json,
# training-state.pt, detector-best.pt, result.json, wandb/), <size>/evaluation/detector-{test,validation}.json,
# <size>.log and comparison.txt. It never imports reviews, builds splits, kills GPU processes, exports or deploys
# anything. Run it again with the same NAME to finish: a size with result.json is not trained again, a size
# interrupted during training resumes from its training-state.pt, a missing or older evaluation is redone.
#
# Every size gets the same settings: ImageNet backbone (NNSLR_TRAIN_INIT=random for random weights), batch 8,
# learning rate 1e-4, up to 20 epochs, the photometric augmentation of the detector, every frame with signs and
# 10% of those without per epoch, the best epoch chosen (and early stopping after 5 epochs measured) by the
# validation recall at score 0.1. The default boxes are small: the smallest is 0.02 of the image side
# (NNSLR_DETECTOR_ANCHOR_MIN_RATIO; torchvision's 0.2 is at least 155 x 88 pixels of a 1344 x 760 frame at any
# input size, since the default boxes are fractions of the image and a larger input only puts them on a finer
# grid). The detector cuts the speed signs and the other traffic signs (NNSLR_CUTTER_FAMILIES=all, the default; speed:
# the speed signs only, as every detector before), so that the classifier sees the other signs and says what they
# are; the comparison shows the recall of the speed signs alone too. A size that resumes keeps the settings it
# started with (they come from its training state).
#
# Environment: NNSLR_GPU_INDEX (default 0), NNSLR_DATASET_KIND (gold), NNSLR_GPU_VENV (.venv-gpu),
# NNSLR_DETECTOR_EPOCHS (20), NNSLR_DETECTOR_PATIENCE (5), NNSLR_DETECTOR_BATCH_SIZE (8), NNSLR_NUM_WORKERS (half the cores, 4 to 16),
# NNSLR_DETECTOR_ANCHOR_MIN_RATIO (0.02), NNSLR_DETECTOR_NEGATIVE_FRACTION (0.1, the share of the frames without
# signs per epoch), NNSLR_CUTTER_FAMILIES (all, or speed), NNSLR_DETECTOR_AUGMENT=0 to train without the
# photometric augmentation,
# NNSLR_TRAIN_INIT=random, NNSLR_ALLOW_BUSY_GPU=1 to share a GPU that already has compute processes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
RUN_ID="${1:-}"
DATASET_KIND="${NNSLR_DATASET_KIND:-gold}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
EPOCHS="${NNSLR_DETECTOR_EPOCHS:-20}"
PATIENCE="${NNSLR_DETECTOR_PATIENCE:-5}"
BATCH_SIZE="${NNSLR_DETECTOR_BATCH_SIZE:-8}"
# The processes reading the frames: half the cores, 4 to 16 (nnslr_tools.training.DEFAULT_WORKERS).
CORES="$(nproc 2>/dev/null || echo 8)"
WORKERS_DEFAULT=$(( CORES / 2 < 4 ? 4 : (CORES / 2 > 16 ? 16 : CORES / 2) ))
NUM_WORKERS="${NNSLR_NUM_WORKERS:-$WORKERS_DEFAULT}"
ANCHOR_MIN_RATIO="${NNSLR_DETECTOR_ANCHOR_MIN_RATIO:-0.02}"
NEGATIVE_FRACTION="${NNSLR_DETECTOR_NEGATIVE_FRACTION:-0.1}"
FAMILIES="${NNSLR_CUTTER_FAMILIES:-all}"
PARTITIONS=(test validation)

if [[ -z "$RUN_ID" ]]; then
  echo "usage: $0 NAME [SIZES...]   (sizes WIDTHxHEIGHT, default 320x320 640x360 960x544)" >&2
  echo "example: NNSLR_GPU_INDEX=0 $0 detector-sizes-001" >&2
  exit 2
fi
shift
SIZES=("$@")
if [[ ${#SIZES[@]} -eq 0 ]]; then
  SIZES=(320x320 640x360 960x544)
fi

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: NAME contains unsupported characters" >&2
  exit 2
fi

for size in "${SIZES[@]}"; do
  if [[ ! "$size" =~ ^[1-9][0-9]*x[1-9][0-9]*$ ]]; then
    echo "error: size must be WIDTHxHEIGHT, e.g. 640x360: $size" >&2
    exit 2
  fi
done

if [[ "$DATASET_KIND" != "gold" && "$DATASET_KIND" != "training-candidate" ]]; then
  echo "error: NNSLR_DATASET_KIND must be gold or training-candidate" >&2
  exit 2
fi

if [[ "$FAMILIES" != "all" && "$FAMILIES" != "speed" ]]; then
  echo "error: NNSLR_CUTTER_FAMILIES must be all or speed" >&2
  exit 2
fi

for setting in EPOCHS PATIENCE BATCH_SIZE NUM_WORKERS; do
  if [[ ! "${!setting}" =~ ^[0-9]+$ ]]; then
    echo "error: $setting must be a whole number: ${!setting}" >&2
    exit 2
  fi
done

for setting in ANCHOR_MIN_RATIO NEGATIVE_FRACTION; do
  if [[ ! "${!setting}" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "error: $setting must be a number: ${!setting}" >&2
    exit 2
  fi
done

if [[ ! -x "$VENV/bin/nnslr" || ! -x "$VENV/bin/python" ]]; then
  echo "error: GPU environment missing: $VENV/bin/nnslr" >&2
  echo "run: bash scripts/setup_v100_training_env.sh" >&2
  exit 2
fi

# The settings the detector accepts: sides from 32 to 4096 pixels, the smallest default box above 0 and below
# 0.95 of the image, a share of frames without signs from 0 to 1 (nnslr_tools.training._check_detector_config).
"$VENV/bin/python" - "$ANCHOR_MIN_RATIO" "$NEGATIVE_FRACTION" "${SIZES[@]}" <<'EOF' || exit 2
import sys

from nnslr_tools.checkpoints import parse_input_size
from nnslr_tools.training import DetectorTrainConfig, _check_detector_config

try:
    for size in sys.argv[3:]:
        _check_detector_config(DetectorTrainConfig(input_size=parse_input_size(size),
                                                   anchor_min_ratio=float(sys.argv[1]),
                                                   negative_frame_fraction=float(sys.argv[2])))
except ValueError as exc:
    sys.exit(f"error: {exc}")
EOF

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
  echo "error: selected GPU already has compute processes; refusing to train the detectors" >&2
  echo "$BUSY" >&2
  echo "Set NNSLR_ALLOW_BUSY_GPU=1 only after intentionally accepting contention." >&2
  exit 3
fi

SIZES_ROOT="$DATA_ROOT/runs/$RUN_ID/detector-sizes"
for size in "${SIZES[@]}"; do
  output="$SIZES_ROOT/$size"
  if [[ -e "$output" && ! -f "$output/result.json" && ! -f "$output/training-state.pt" ]]; then
    echo "error: $output exists without result.json or training-state.pt" >&2
    echo "remove that folder or choose a new NAME" >&2
    exit 3
  fi
done

export NNSLR_DATA_ROOT="$DATA_ROOT"
# The index is nvidia-smi's, the card checked above: CUDA counts the cards in the same order, by PCI bus (its
# default order, fastest first, leaves two identical V100s in no stated order).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

# wandb: "wandb login" saves the key in ~/.netrc; without it, offline.
# WANDB_MODE=disabled turns wandb off completely.
if [ -z "${WANDB_MODE:-}" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  if ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    export WANDB_MODE=offline
    echo "wandb: not logged in, going offline (the numbers stay in the terminal and in $SIZES_ROOT/<size>/wandb)."
  fi
fi

# The same settings for every size that starts; a size that resumes takes its own from its training state.
SETTINGS=(--epochs "$EPOCHS" --patience "$PATIENCE" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
          --anchor-min-ratio "$ANCHOR_MIN_RATIO" --negative-frame-fraction "$NEGATIVE_FRACTION" --select-by recall
          --cutter-families "$FAMILIES")
if [[ "${NNSLR_TRAIN_INIT:-imagenet}" != "random" ]]; then
  SETTINGS+=(--pretrained)
fi
if [[ "${NNSLR_DETECTOR_AUGMENT:-1}" == "0" ]]; then
  SETTINGS+=(--no-augment)
fi
SETTINGS_TEXT="input sizes ${SIZES[*]}; boxes of the families $FAMILIES; smallest default box $ANCHOR_MIN_RATIO of the image side; frames without signs $NEGATIVE_FRACTION per epoch; best epoch by validation recall at score 0.1; ${SETTINGS[*]}"

NN="$VENV/bin/nnslr"
total=$((2 + 2 * ${#SIZES[@]}))

echo "== Detector input sizes =="
echo "run:          $RUN_ID"
echo "sizes:        ${SIZES[*]}"
echo "output:       $SIZES_ROOT"
echo "data root:    $DATA_ROOT"
echo "dataset kind: $DATASET_KIND"
echo "physical GPU: $GPU_INDEX"
echo "default boxes: smallest $ANCHOR_MIN_RATIO of the image side (torchvision's: 0.2)"
echo "sign-free:    $NEGATIVE_FRACTION of the frames without signs per epoch (validation: one fixed draw)"
echo "best epoch:   validation recall at score 0.1, ties to fewer false boxes per frame"
echo "boxes:        $FAMILIES (all: the speed signs and the other traffic signs; speed: the speed signs only)"
echo "settings:     ${SETTINGS[*]}"
echo "wandb:        project ${WANDB_PROJECT:-sunnypilot.speedlimits}, group $RUN_ID-<size>, mode ${WANDB_MODE:-online}"
echo

echo "== 1/$total Validate the training and evaluation populations (CPU-only) =="
"$NN" train --task detector --dataset-kind "$DATASET_KIND" --cutter-families "$FAMILIES" --dry-run
for partition in "${PARTITIONS[@]}"; do
  "$NN" evaluate --task detector --dataset-kind "$DATASET_KIND" --cutter-families "$FAMILIES" --partition "$partition" --dry-run
done
mkdir -p "$SIZES_ROOT"

echo "== 2/$total Explicit CUDA smoke test =="
# CUDA_VISIBLE_DEVICES maps the selected physical GPU to logical cuda:0.
"$NN" env --gpu-smoke --gpu-device 0

step=3
for size in "${SIZES[@]}"; do
  output="$SIZES_ROOT/$size"
  echo "== $step/$total Train the detector at $size =="
  step=$((step + 1))
  if [[ -f "$output/result.json" ]]; then
    echo "skip completed $size: $output/result.json"
  else
    mode=fresh
    flags=("${SETTINGS[@]}")
    if [[ -f "$output/training-state.pt" ]]; then
      mode=resume
      # A resumed run trains with exactly the settings it started with: they come from its training state.
      resume_flags="$("$VENV/bin/python" -c 'import sys; from nnslr_tools.training import resume_arguments_file; print("\n".join(resume_arguments_file(sys.argv[1])))' "$output/training-state.pt")"
      readarray -t flags <<< "$resume_flags"
      echo "resume $size with the settings of its training state: ${flags[*]}"
    fi
    # The epoch lines ("detector epoch N/M ...") and the final JSON go to the terminal and to <size>.log.
    NNSLR_RUN_ID="$RUN_ID-$size" "$NN" train --task detector --dataset-kind "$DATASET_KIND" --device cuda \
      --mode "$mode" --output "$output" --input-size "$size" "${flags[@]}" 2>&1 | tee -a "$SIZES_ROOT/$size.log"
  fi

  echo "== $step/$total Evaluate the detector at $size on ${PARTITIONS[*]} =="
  step=$((step + 1))
  mkdir -p "$output/evaluation"
  for partition in "${PARTITIONS[@]}"; do
    report="$output/evaluation/detector-$partition.json"
    if [[ -f "$report" && "$report" -nt "$output/detector-best.pt" ]]; then
      echo "skip evaluated $size $partition: $report"
      continue
    fi
    # The report (every frame with its boxes) goes to the file only; the comparison below reads it.
    "$NN" evaluate --task detector --dataset-kind "$DATASET_KIND" --partition "$partition" --device cuda \
      --checkpoint "$output/detector-best.pt" --output "$report" > /dev/null
    echo "evaluated $size $partition: $report"
  done
done

echo
echo "== Comparison (the cutter's recall first; validation recall at score 0.1 chose the best epoch of each size) =="
{
  echo "run $RUN_ID: $SETTINGS_TEXT"
  echo "(each size's own settings, from its result.json, are in the cost table: a resumed size keeps those it started with)"
  echo
  "$VENV/bin/python" -c 'import sys; from nnslr_tools.evaluation import detector_sizes_table; print(detector_sizes_table(sys.argv[1], sys.argv[2:]), end="")' \
    "$SIZES_ROOT" "${SIZES[@]}"
} | tee "$SIZES_ROOT/comparison.txt"
echo
echo "Detector sizes complete: $SIZES_ROOT"
echo "Nothing was imported, exported, deployed or published."
