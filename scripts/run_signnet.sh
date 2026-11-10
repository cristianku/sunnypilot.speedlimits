#!/usr/bin/env bash
set -euo pipefail

# The portal's "Train the cutter": one SignNet, the cutter the comma runs (nnslr_tools.signnet), trained with the
# car's setup so the car can load it as it is: the top band of the frame at 896 x 288, a grid of 8 pixels, the
# fuse convolution 1 x 1, channels 16,24,32,48 (the car's vision.py only runs that input and that grid; the run on
# the car, 2026-10-28_1028-signnet-twothirds-s8-f1, was trained so). The best epoch is the one with the best
# validation recall in the 5 best boxes of a frame (the car keeps 5); that epoch is then measured on validation
# and test (result.json).
#
# The Mapillary windows (helpers/mapillary_windows.sh: pieces of the Mapillary Traffic Sign Dataset's photos cut to
# look like our band, other countries' speed signs at our signs' sizes): NNSLR_CUTTER_MAPILLARY=1 adds them to our
# train frames, their speed boxes at most NNSLR_CUTTER_MAPILLARY_RATIO times ours (default 1); =pretrain trains in
# two stages, as scripts/run_classifier.sh: first the windows alone (pretrain/, from random weights or
# NNSLR_CUTTER_INIT_FROM; the epoch chosen on the recall of the speed signs of our validation), then our frames alone
# from its best weights (the root, --init-from pretrain/signnet-best.pt). Validation and test are always ours.
#
# usage: scripts/run_signnet.sh NAME
#
# Writes only into $DATA_ROOT/runs/NAME/: signnet-best.pt, result.json (nnslr train-signnet --run-dir), pretrain/
# (the same for the first stage of NNSLR_CUTTER_MAPILLARY=pretrain) and signnet.log (everything this script prints,
# both stages). It never imports reviews, builds splits, kills GPU processes, exports, deploys or puts the SignNet in
# use (Models > Use this one). A NAME holding another kind of run or a finished SignNet is refused; a NAME with
# signnet-best.pt but no result.json (a training stopped half way: SignNet has no resume) is trained again from
# scratch, and a first stage with its result.json is not trained again ("skip trained"; one trained with other
# settings, pretrain/settings.txt, is refused: remove pretrain/ or use the same settings). The log of a NAME tried
# before is moved to signnet.log.1 (replacing an older one), so the new run's epochs never mix with the old ones.
#
# Environment: NNSLR_GPU_INDEX (default 0), NNSLR_DATASET_KIND (gold), NNSLR_GPU_VENV (.venv-gpu),
# NNSLR_CUTTER_FAMILIES (all: the speed signs and the other traffic signs; speed: the speed signs only),
# NNSLR_SIGNNET_EPOCHS (40), NNSLR_SIGNNET_BATCH (16), NNSLR_SIGNNET_LEARNING_RATE (0.003), NNSLR_NUM_WORKERS (half the
# cores, 4 to 16) (the portal sets those changed in Train › Cutter › Settings, nnslr_tools.train_settings),
# NNSLR_CUTTER_INIT_FROM (a finished SignNet's signnet-best.pt to go on from, instead of random weights: Train › Cutter
# › Start from; with NNSLR_CUTTER_MAPILLARY=pretrain the start of the first stage), NNSLR_CUTTER_MAPILLARY (0, 1 or
# pretrain; default 0), NNSLR_CUTTER_MAPILLARY_REGION (all; or continents and countries, comma-separated: europe or
# CH,DE,IT; needs the positions of helpers/mapillary_locations.sh), NNSLR_CUTTER_MAPILLARY_SHARE (100: the percentage
# of the region's windows, always the same ones for the same share), NNSLR_CUTTER_MAPILLARY_RATIO (1: the Mapillary
# speed boxes per one of ours, NNSLR_CUTTER_MAPILLARY=1 only), NNSLR_CUTTER_PRETRAIN_EPOCHS (the first stage's
# epochs; default NNSLR_SIGNNET_EPOCHS), NNSLR_ALLOW_BUSY_GPU=1 to share a GPU that already has compute processes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
RUN_ID="${1:-}"
DATASET_KIND="${NNSLR_DATASET_KIND:-gold}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
FAMILIES="${NNSLR_CUTTER_FAMILIES:-all}"
EPOCHS="${NNSLR_SIGNNET_EPOCHS:-40}"
PRETRAIN_EPOCHS="${NNSLR_CUTTER_PRETRAIN_EPOCHS:-$EPOCHS}"
BATCH="${NNSLR_SIGNNET_BATCH:-16}"
# The processes reading the frames: half the cores, 4 to 16 (nnslr_tools.training.DEFAULT_WORKERS).
CORES="$(nproc 2>/dev/null || echo 8)"
WORKERS_DEFAULT=$(( CORES / 2 < 4 ? 4 : (CORES / 2 > 16 ? 16 : CORES / 2) ))
NUM_WORKERS="${NNSLR_NUM_WORKERS:-$WORKERS_DEFAULT}"
# The car's setup (nnslr_tools.signnet CAR_BAND, CAR_STRIDE, CAR_FUSE_KERNEL, CAR_WIDTHS; the comma's vision.py).
BAND_WIDTH=896
BAND_HEIGHT=288
STRIDE=8
FUSE_KERNEL=1
WIDTHS=16,24,32,48
LEARNING_RATE="${NNSLR_SIGNNET_LEARNING_RATE:-0.003}"
INIT_FROM="${NNSLR_CUTTER_INIT_FROM:-}"
MAPILLARY_MODE="${NNSLR_CUTTER_MAPILLARY:-0}"
MAPILLARY_REGION="${NNSLR_CUTTER_MAPILLARY_REGION:-all}"
MAPILLARY_SHARE="${NNSLR_CUTTER_MAPILLARY_SHARE:-100}"
MAPILLARY_RATIO="${NNSLR_CUTTER_MAPILLARY_RATIO:-}"
WINDOWS="$DATA_ROOT/external/mapillary/windows.jsonl"

if [[ -z "$RUN_ID" || $# -gt 1 ]]; then
  echo "usage: $0 NAME" >&2
  echo "example: NNSLR_GPU_INDEX=0 $0 2026-10-30_1200-cutter" >&2
  exit 2
fi

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: NAME contains unsupported characters" >&2
  exit 2
fi

if [[ "$DATASET_KIND" != "gold" && "$DATASET_KIND" != "training-candidate" ]]; then
  echo "error: NNSLR_DATASET_KIND must be gold or training-candidate" >&2
  exit 2
fi

if [[ "$FAMILIES" != "all" && "$FAMILIES" != "speed" ]]; then
  echo "error: NNSLR_CUTTER_FAMILIES must be all or speed" >&2
  exit 2
fi

for setting in EPOCHS PRETRAIN_EPOCHS BATCH NUM_WORKERS; do
  if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: $setting must be a whole number above 0: ${!setting}" >&2
    exit 2
  fi
done

# A positive number: 0.003, 3e-3, 1e-05.
if [[ ! "$LEARNING_RATE" =~ ^([0-9]*\.)?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
  echo "error: NNSLR_SIGNNET_LEARNING_RATE must be a number: $LEARNING_RATE" >&2
  exit 2
fi

if [[ -n "$INIT_FROM" && ! -f "$INIT_FROM" ]]; then
  echo "error: NNSLR_CUTTER_INIT_FROM is not a file: $INIT_FROM" >&2
  exit 2
fi

if [[ "$MAPILLARY_MODE" != "0" && "$MAPILLARY_MODE" != "1" && "$MAPILLARY_MODE" != "pretrain" ]]; then
  echo "error: NNSLR_CUTTER_MAPILLARY must be 0, 1 or pretrain: $MAPILLARY_MODE" >&2
  exit 2
fi
if [[ "$MAPILLARY_MODE" == "0" && ( "$MAPILLARY_REGION" != "all" || "$MAPILLARY_SHARE" != "100" ) ]]; then
  echo "error: NNSLR_CUTTER_MAPILLARY_REGION and _SHARE choose Mapillary windows: add NNSLR_CUTTER_MAPILLARY=1" >&2
  exit 2
fi
if [[ -n "$MAPILLARY_RATIO" && "$MAPILLARY_MODE" != "1" ]]; then
  echo "error: NNSLR_CUTTER_MAPILLARY_RATIO caps the Mapillary windows added to ours: only with NNSLR_CUTTER_MAPILLARY=1" >&2
  exit 2
fi
if [[ -n "$MAPILLARY_RATIO" && ! "$MAPILLARY_RATIO" =~ ^([0-9]*\.)?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
  echo "error: NNSLR_CUTTER_MAPILLARY_RATIO must be a number: $MAPILLARY_RATIO" >&2
  exit 2
fi
if [[ "$MAPILLARY_MODE" != "0" && ! -s "$WINDOWS" ]]; then
  echo "error: the Mapillary windows are not cut: run helpers/mapillary_windows.sh first" >&2
  exit 2
fi
if [[ "$MAPILLARY_MODE" != "0" && "$MAPILLARY_REGION" != "all" && ! -s "$DATA_ROOT/external/mapillary/locations.jsonl" ]]; then
  echo "error: the positions of the Mapillary photos are not fetched: run helpers/mapillary_locations.sh first" >&2
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
  echo "error: selected GPU already has compute processes; refusing to train SignNet" >&2
  echo "$BUSY" >&2
  echo "Set NNSLR_ALLOW_BUSY_GPU=1 only after intentionally accepting contention." >&2
  exit 3
fi

RUN_ROOT="$DATA_ROOT/runs/$RUN_ID"
PRETRAIN_ROOT="$RUN_ROOT/pretrain"
# A folder of another kind of run (the training pipeline, experiment B, the detector sizes, a classifier and its
# Mapillary stage, the road type) is never written into. pretrain/ alone is a cutter's own first stage.
for part in detector reader export bundle evaluation experiment-b detector-sizes classifier pretrain/reader-best.pt \
            pretrain/training-plan.json pretrain/training-state.pt negatives.jsonl classifier.log road-type \
            road-type.log; do
  if [[ -e "$RUN_ROOT/$part" ]]; then
    echo "error: $RUN_ROOT is another kind of run (it has $part): choose a new NAME" >&2
    exit 3
  fi
done
if [[ -f "$RUN_ROOT/result.json" ]]; then
  echo "error: $RUN_ROOT already holds a finished SignNet (result.json): choose a new NAME" >&2
  exit 3
fi
if [[ "$MAPILLARY_MODE" != "pretrain" && ( -e "$PRETRAIN_ROOT/signnet-best.pt" || -e "$PRETRAIN_ROOT/result.json" ) ]]; then
  echo "error: $RUN_ROOT has a Mapillary stage (pretrain/): go on with NNSLR_CUTTER_MAPILLARY=pretrain or choose a new NAME" >&2
  exit 3
fi
RESTART=0
if [[ -e "$RUN_ROOT/signnet-best.pt" ]]; then
  RESTART=1
fi
PRETRAIN_RESTART=0
if [[ "$MAPILLARY_MODE" == "pretrain" && -e "$PRETRAIN_ROOT/signnet-best.pt" && ! -f "$PRETRAIN_ROOT/result.json" ]]; then
  PRETRAIN_RESTART=1
fi

SETTINGS=(--families "$FAMILIES" --band-width "$BAND_WIDTH" --stride "$STRIDE" --fuse-kernel "$FUSE_KERNEL"
          --widths "$WIDTHS" --epochs "$EPOCHS" --batch "$BATCH" --learning-rate "$LEARNING_RATE")
# The Mapillary windows: those of the region and share asked (both left out at their defaults).
MAPILLARY=()
if [[ "$MAPILLARY_MODE" != "0" ]]; then
  MAPILLARY=(--mapillary "$WINDOWS")
  if [[ "$MAPILLARY_REGION" != "all" ]]; then
    MAPILLARY+=(--mapillary-region "$MAPILLARY_REGION")
  fi
  if [[ "$MAPILLARY_SHARE" != "100" ]]; then
    MAPILLARY+=(--mapillary-share "$MAPILLARY_SHARE")
  fi
fi
# The two stages (NNSLR_CUTTER_MAPILLARY=pretrain): the first on the windows alone, from random weights or
# NNSLR_CUTTER_INIT_FROM; the second on our frames alone, from the first's best epoch. Else one training, with the
# windows added (=1) or not, from random weights or the weights of a finished SignNet it goes on from.
PRETRAIN_SETTINGS=()
if [[ "$MAPILLARY_MODE" == "pretrain" ]]; then
  PRETRAIN_SETTINGS=(--families "$FAMILIES" --band-width "$BAND_WIDTH" --stride "$STRIDE" --fuse-kernel "$FUSE_KERNEL"
                     --widths "$WIDTHS" --epochs "$PRETRAIN_EPOCHS" --batch "$BATCH" --learning-rate "$LEARNING_RATE"
                     "${MAPILLARY[@]}" --mapillary-only)
  if [[ -n "$INIT_FROM" ]]; then
    PRETRAIN_SETTINGS+=(--init-from "$INIT_FROM")
  fi
  # The check of the second stage's plan cannot name the first stage's weights: they do not exist yet.
  CHECK_SETTINGS=("${SETTINGS[@]}")
  SETTINGS+=(--init-from "$PRETRAIN_ROOT/signnet-best.pt")
else
  SETTINGS+=("${MAPILLARY[@]}")
  if [[ -n "$MAPILLARY_RATIO" ]]; then
    SETTINGS+=(--mapillary-ratio "$MAPILLARY_RATIO")
  fi
  if [[ -n "$INIT_FROM" ]]; then
    SETTINGS+=(--init-from "$INIT_FROM")
  fi
  CHECK_SETTINGS=("${SETTINGS[@]}")
fi
NN="$VENV/bin/nnslr"
# A first stage is only skipped when it was trained with these settings (pretrain/settings.txt, written once it
# completes): one of other settings would start the second stage from other weights than those asked for.
PRETRAIN_RECORD="$PRETRAIN_ROOT/settings.txt"
PRETRAIN_KEY="--dataset-kind $DATASET_KIND ${PRETRAIN_SETTINGS[*]}"
if [[ "$MAPILLARY_MODE" == "pretrain" && -f "$PRETRAIN_ROOT/result.json" && -f "$PRETRAIN_RECORD" ]] \
   && [[ "$(cat "$PRETRAIN_RECORD")" != "$PRETRAIN_KEY" ]]; then
  echo "error: $PRETRAIN_ROOT was trained with other settings ($(cat "$PRETRAIN_RECORD")), not $PRETRAIN_KEY: remove pretrain/ or use the same settings" >&2
  exit 3
fi

mkdir -p "$RUN_ROOT"
# A log of this NAME tried before: kept aside as signnet.log.1, the new one starts empty (the panel and Results
# read the last line of each epoch, so old epochs would show as the new run's).
OLD_LOG=0
if [[ -e "$RUN_ROOT/signnet.log" ]]; then
  mv -f "$RUN_ROOT/signnet.log" "$RUN_ROOT/signnet.log.1"
  OLD_LOG=1
fi
# Everything printed from here on also goes to the run's own log (the Evaluate tab reads its epochs).
exec > >(tee -a "$RUN_ROOT/signnet.log") 2>&1

export NNSLR_DATA_ROOT="$DATA_ROOT"
# The index is nvidia-smi's, the card checked above: CUDA counts the cards in the same order, by PCI bus (its
# default order, fastest first, leaves two identical V100s in no stated order).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
# Python writes each line as it comes, not when its 8 KB buffer fills: the epochs reach signnet.log through tee
# while it trains (nnslr train-signnet flushes them too).
export PYTHONUNBUFFERED=1


echo "== scripts/run_signnet.sh $RUN_ID"
echo "run:          $RUN_ID"
echo "band:         ${BAND_WIDTH}x${BAND_HEIGHT}"
if [[ "$MAPILLARY_MODE" == "pretrain" ]]; then
  echo "pretraining:  ${PRETRAIN_SETTINGS[*]}"
fi
echo "settings:     ${SETTINGS[*]}"
echo "workers:      $NUM_WORKERS"
echo "output:       $RUN_ROOT"
echo "data root:    $DATA_ROOT"
echo "dataset kind: $DATASET_KIND"
echo "physical GPU: $GPU_INDEX"
echo "boxes:        $FAMILIES (all: the speed signs and the other traffic signs; speed: the speed signs only)"
echo "best epoch:   validation recall in the 5 best boxes of a frame (the car keeps 5)"
if [[ "$MAPILLARY_MODE" == "1" ]]; then
  echo "mapillary:    the windows of $MAPILLARY_REGION, ${MAPILLARY_SHARE}%, added to our train frames, at most ${MAPILLARY_RATIO:-1} times our speed boxes"
elif [[ "$MAPILLARY_MODE" == "pretrain" ]]; then
  echo "mapillary:    first the windows of $MAPILLARY_REGION, ${MAPILLARY_SHARE}%, alone ($PRETRAIN_EPOCHS epochs, best epoch by the recall of the speed signs), then our frames alone from them"
fi
echo
if [[ "$PRETRAIN_RESTART" == "1" ]]; then
  echo "restart the first stage from scratch: $PRETRAIN_ROOT/signnet-best.pt has no result.json"
fi
if [[ "$RESTART" == "1" ]]; then
  echo "restart from scratch: $RUN_ROOT/signnet-best.pt has no result.json (a training stopped half way; SignNet has no resume)"
fi
if [[ "$OLD_LOG" == "1" ]]; then
  echo "the log of the earlier try: $RUN_ROOT/signnet.log.1"
fi

echo "== 1/3 Check the training population (CPU-only) =="
if [[ "$MAPILLARY_MODE" == "pretrain" ]]; then
  CUDA_VISIBLE_DEVICES= "$NN" train-signnet --dry-run --run-dir "$PRETRAIN_ROOT" --dataset-kind "$DATASET_KIND" \
    "${PRETRAIN_SETTINGS[@]}"
fi
CUDA_VISIBLE_DEVICES= "$NN" train-signnet --dry-run --run-dir "$RUN_ROOT" --dataset-kind "$DATASET_KIND" "${CHECK_SETTINGS[@]}"

echo "== 2/3 Explicit CUDA smoke test =="
# CUDA_VISIBLE_DEVICES maps the selected physical GPU to logical cuda:0.
"$NN" env --gpu-smoke --gpu-device 0

echo "== 3/3 Train SignNet =="
# The epoch lines ("signnet epoch N/M train_loss ... val_recall ... val_false_per_frame ... 87s"; the first stage's
# "signnet_pretrain epoch N/M ... val_speed_recall ... 87s"), the measure of the best epoch and "SignNet complete:
# <run>" (the first stage's "Mapillary stage complete: <run>/pretrain") go to the terminal and to signnet.log.
if [[ "$MAPILLARY_MODE" == "pretrain" ]]; then
  echo "-- the first stage: the Mapillary windows alone --"
  if [[ -f "$PRETRAIN_ROOT/result.json" ]]; then
    echo "skip trained: $PRETRAIN_ROOT/result.json"
    if [[ ! -f "$PRETRAIN_RECORD" ]]; then
      echo "note: $PRETRAIN_RECORD is missing (a stage of before 3 Nov 2026): its settings are not checked"
    fi
  else
    "$NN" train-signnet --run-dir "$PRETRAIN_ROOT" --dataset-kind "$DATASET_KIND" "${PRETRAIN_SETTINGS[@]}" \
      --workers "$NUM_WORKERS"
    mkdir -p "$PRETRAIN_ROOT"
    printf '%s\n' "$PRETRAIN_KEY" > "$PRETRAIN_RECORD"
  fi
  echo "-- the second stage: our frames alone, from the first --"
fi
"$NN" train-signnet --run-dir "$RUN_ROOT" --dataset-kind "$DATASET_KIND" "${SETTINGS[@]}" --workers "$NUM_WORKERS"

echo
echo "Nothing was imported, exported, deployed or put in use (Models > Use this one)."
