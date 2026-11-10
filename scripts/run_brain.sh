#!/usr/bin/env bash
set -euo pipefail

# The brain (docs/NEURAL_NETWORK.md): the network that says the speed to hold, four times a second, from the picture and the
# car's own signals, learnt from every dashcam drive with its logs read, each whole: the driver's own speed, corrected
# by the blocks of Speed review, the roadworks and the kinds of road marked there and the next limit of the signs our networks read,
# those of Sign review where the frames were confirmed there. On one
# GPU (the readings of step 3 also on the free ones), one step after the other:
#
#   1. checks the labels, the split and the whole training on the CPU (nnslr train-brain --dry-run; its log says the
#      dashcam drives and hours, and it stops while there is none); the drives it takes are the run's, kept in
#      runs/NAME/routes.json and passed with --route to every step after it and to every seed: a drive read meanwhile
#      waits for the next run;
#   2. a CUDA smoke test;
#   3. reads the signs of every drive it learns from (every dashcam drive) as the car does today
#      (nnslr road-type-signs --labelled; a drive already read by the car's reader of today is not read again: a new
#      vision.py or a new network reads them all again, and the training refuses anything else), the signs it is given;
#      then reads them with the networks on the car, cutter, classifier and applicability, on the pictures at 1 a second
#      (nnslr brain-signs, the same: a new network reads them all again), the next limit and its metres it is taught
#      where nobody confirmed the frames in Sign review; then says what the labels give (nnslr brain-targets). Both
#      readings run on the brain's card and on every other card nothing computes on and no other job of the panel
#      holds at that moment, which the panel then counts as the brain's until the step ends (#36: the NVDEC of one
#      V100 held the first back, one drive at a time the second), several drives at once;
#   4-6. trains seeds 0, 1 and 2, each into a run of its own, runs/NAME-s0, -s1, -s2 (nnslr train-brain, the last epoch
#      kept): three versions in Models › Brain, which differ only by chance. Train › Brain draws them on one chart.
#
# usage: scripts/run_brain.sh NAME
#
# Writes only into $DATA_ROOT/runs/NAME/ (brain.log, routes.json, dry-run/), $DATA_ROOT/runs/NAME-s{0,1,2}/ (brain-last.pt, result.json),
# the sign readings (derived/road_type_signs, derived/brain_signs) and the frame cache. It never imports reviews, changes the labels, the model in use or the car, and
# exports nothing: Publish exports the brain chosen in Models and checks it along a real drive. Run it again with the same
# NAME to finish: a finished seed is not repeated, the signs already read are not read again, the drives are the same.
#
# Environment: NNSLR_GPU_INDEX (default 0), NNSLR_GPU_VENV (.venv-gpu), NNSLR_BRAIN_EPOCHS (40), NNSLR_SIGN_WORKERS
# (processes reading the signs as the car does: half the cores, each with its ffmpeg), NNSLR_BRAIN_SIGN_JOBS (drives
# our networks read at once: 0, one per 4 cores), NNSLR_SIGN_GPUS (the cards of step 3, nvidia-smi's numbers: 0,1;
# default the brain's and every other card free then), NNSLR_BRAIN_WORKS (1: it also learns the
# roadworks; 0: not), NNSLR_BRAIN_ROAD (1: it also learns the kinds of road of Speed review; 0: not),
# NNSLR_BRAIN_TYPES (1: it also learns the road type of the map and of Speed review, and only from the seconds that have
# one; 0: not, every second),
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
BRAIN_SIGN_JOBS="${NNSLR_BRAIN_SIGN_JOBS:-0}"
SIGN_GPUS="${NNSLR_SIGN_GPUS:-}"
WORKS="${NNSLR_BRAIN_WORKS:-1}"
ROAD="${NNSLR_BRAIN_ROAD:-1}"
TYPES="${NNSLR_BRAIN_TYPES:-1}"
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

for setting in EPOCHS SIGN_WORKERS BRAIN_SIGN_JOBS; do
  if [[ ! "${!setting}" =~ ^[0-9]+$ ]]; then
    echo "error: $setting must be a whole number: ${!setting}" >&2
    exit 2
  fi
done

if [[ -n "$SIGN_GPUS" && ! "$SIGN_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "error: NNSLR_SIGN_GPUS must be cards by nvidia-smi's numbers, e.g. 0,1: $SIGN_GPUS" >&2
  exit 2
fi

if [[ "$WORKS" != "0" && "$WORKS" != "1" ]]; then
  echo "error: NNSLR_BRAIN_WORKS must be 0 or 1: $WORKS" >&2
  exit 2
fi

if [[ "$ROAD" != "0" && "$ROAD" != "1" ]]; then
  echo "error: NNSLR_BRAIN_ROAD must be 0 or 1: $ROAD" >&2
  exit 2
fi

if [[ "$TYPES" != "0" && "$TYPES" != "1" ]]; then
  echo "error: NNSLR_BRAIN_TYPES must be 0 or 1: $TYPES" >&2
  exit 2
fi

# The driver's inputs are gone since 10 Nov 2026 (brain_targets.LEFT_OUT): with the car's speed in, the brain followed
# his speed on the road. A caller of before that still asks for them stops here.
if [[ "${NNSLR_BRAIN_DRIVER:-0}" != "0" ]]; then
  echo "error: NNSLR_BRAIN_DRIVER is gone: the brain never takes the gas, the brake or the car's speed" >&2
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
# The index is nvidia-smi's, the card checked above: CUDA counts the cards in the same order, by PCI bus (its
# default order, fastest first, leaves two identical V100s in no stated order).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
NN="$VENV/bin/nnslr"
SETTINGS=(--epochs "$EPOCHS" --pool "$POOL")
if [[ "$WORKS" == "0" ]]; then
  SETTINGS+=(--no-works)
fi
if [[ "$ROAD" == "0" ]]; then
  SETTINGS+=(--no-road)
fi
if [[ "$TYPES" == "0" ]]; then
  SETTINGS+=(--no-road-types)
fi

echo "== Brain =="
echo "run:          $RUN_ID"
echo "output:       $RUN_ROOT-s{${SEEDS[0]},${SEEDS[1]},${SEEDS[2]}}"
echo "data root:    $DATA_ROOT"
echo "physical GPU: $GPU_INDEX"
echo "settings:     ${SETTINGS[*]}"
echo

echo "== 1/6 Check the labels, the split and the training (CPU-only) =="
# The drives of the run: those its first dry run took (brain_train.ROUTES), kept for every step and every seed whatever
# drive is read meanwhile; a NAME run again keeps them too.
ROUTES_FILE="$RUN_ROOT/routes.json"
ROUTE_ARGS=()
read_routes() {
  local found route
  found="$(grep -soE '[0-9a-f]{8}--[0-9a-f]{10}' "$1" || true)"
  if [[ -z "$found" ]]; then
    echo "error: no drive in $1" >&2
    exit 3
  fi
  ROUTE_ARGS=()
  while IFS= read -r route; do
    ROUTE_ARGS+=(--route "$route")
  done <<< "$found"
}
if [[ -f "$ROUTES_FILE" ]]; then
  read_routes "$ROUTES_FILE"
fi
CUDA_VISIBLE_DEVICES="" "$NN" train-brain --dry-run --run-dir "$RUN_ROOT/dry-run" "${SETTINGS[@]}" "${ROUTE_ARGS[@]}"
if [[ ! -f "$ROUTES_FILE" ]]; then
  read_routes "$RUN_ROOT/dry-run/routes.json"
  cp "$RUN_ROOT/dry-run/routes.json" "$ROUTES_FILE"
fi
echo "drives:       $(( ${#ROUTE_ARGS[@]} / 2 )) (routes.json)"

echo "== 2/6 Explicit CUDA smoke test =="
"$NN" env --gpu-smoke --gpu-device 0

echo "== 3/6 Read the signs of the drives as the car does today =="
# The signs on the GPU: .venv has onnxruntime-gpu, .venv-gpu only the CPU's (torch brings its own CUDA).
SIGNS_NN="$ROOT_DIR/.venv/bin/nnslr"
[ -x "$SIGNS_NN" ] || SIGNS_NN="$NN"
# The cards that read them (nnslr borrow-cards): the brain's, and every other card no compute process is on and no
# other job of the panel holds; or those of NNSLR_SIGN_GPUS. In a job of the panel the others are written to its record
# (NNSLR_JOB_RECORD): the panel names them as the brain's and starts nothing else on them until this step ends. Chosen
# here, not at the start (the checks above take a minute), and again before the second reading. Without an answer, the
# brain's card alone.
sign_cards() {
  local cards
  if [[ -n "$SIGN_GPUS" ]]; then
    cards="$("$SIGNS_NN" borrow-cards --own "$GPU_INDEX" --cards "$SIGN_GPUS" --data-root "$DATA_ROOT")" ||
      cards="$SIGN_GPUS"
  else
    cards="$("$SIGNS_NN" borrow-cards --own "$GPU_INDEX" --data-root "$DATA_ROOT")" || cards="$GPU_INDEX"
  fi
  [[ "$cards" =~ ^[0-9]+(,[0-9]+)*$ ]] || cards="$GPU_INDEX"
  echo "$cards"
}
CARDS="$(sign_cards)"
echo "sign cards:   $CARDS"
"$SIGNS_NN" road-type-signs --labelled --num-workers "$SIGN_WORKERS" --gpus "$CARDS" --data-root "$DATA_ROOT" \
  "${ROUTE_ARGS[@]}"
# The next limit and its metres it is taught: the car's networks read the pictures (their worker on .venv-gpu's torch).
CARDS="$(sign_cards)"
echo "sign cards:   $CARDS"
"$NN" brain-signs --jobs "$BRAIN_SIGN_JOBS" --gpus "$CARDS" --data-root "$DATA_ROOT" "${ROUTE_ARGS[@]}"
"$SIGNS_NN" borrow-cards --release --data-root "$DATA_ROOT" || true
if [[ "$TYPES" == "0" ]]; then
  "$NN" brain-targets --data-root "$DATA_ROOT" "${ROUTE_ARGS[@]}" --no-road-types
else
  "$NN" brain-targets --data-root "$DATA_ROOT" "${ROUTE_ARGS[@]}"
fi

step=4
for seed in "${SEEDS[@]}"; do
  echo "== $step/6 Train seed $seed =="
  step=$(( step + 1 ))
  if [[ -f "$RUN_ROOT-s$seed/result.json" ]]; then
    echo "skip seed $seed: already trained"
    continue
  fi
  "$NN" train-brain --run-dir "$RUN_ROOT-s$seed" --seed "$seed" "${SETTINGS[@]}" "${ROUTE_ARGS[@]}"
done

echo "Brain complete: choose a seed in Models › Brain, then Publish."
