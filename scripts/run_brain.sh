#!/usr/bin/env bash
set -euo pipefail

# The brain (docs/NEURAL_NETWORK.md): the network that says the speed to hold, four times a second, from the picture and the
# car's own signals, learnt from every dashcam drive with its logs read, each whole: the driver's own speed, corrected
# by the blocks of Speed review, with the roadworks and the kinds of road marked there. On one
# GPU (the reading of step 3 also on the free ones), one step after the other:
#
#   1. checks the labels, the split and the whole training on the CPU (nnslr train-brain --dry-run; its log says the
#      dashcam drives and hours, and it stops while there is none); the drives it takes are the run's, kept in
#      runs/NAME/routes.json and passed with --route to every step after it and to every seed: a drive read meanwhile
#      waits for the next run;
#   2. a CUDA smoke test;
#   3. the networks its signs are read with (nnslr brain-networks): the cutter, the classifier and the applicability
#      network chosen in Train › Brain (NNSLR_BRAIN_CUTTER, _CLASSIFIER, _DECIDER; those in use by default), exported
#      as Publish exports them and written into runs/NAME/networks.json, kept when the run is started again (#31, 10 Nov
#      2026); then reads the signs of every drive it learns from (every dashcam drive) as the car does: its vision.py
#      and applicability.py with those networks, the car's speed from the logs, a sign of another road left out
#      (nnslr road-type-signs --labelled --sign-networks; a drive already read by them is not read again: a new
#      vision.py or a new choice reads them all again, and the training refuses anything else), the signs it is given;
#      then says what the labels give (nnslr brain-targets). The
#      reading runs on the brain's card and on every other card nothing computes on and no other job of the panel
#      holds at that moment, which the panel then counts as the brain's until the step ends (#36: the NVDEC of one
#      V100 held it back). There is no second reading of the signs since 12 Nov 2026: it made the answers of the
#      next limit, which no brain is taught any more;
#   4-6. trains seeds 0, 1 and 2, each into a run of its own, runs/NAME-s0, -s1, -s2 (nnslr train-brain, the last epoch
#      kept): three versions in Models › Brain, which differ only by chance. Train › Brain draws them on one chart. With
#      NNSLR_BRAIN_SEEDS=0 seed 0 alone (step 4), a third of the time.
#
# usage: scripts/run_brain.sh NAME
#
# Writes only into $DATA_ROOT/runs/NAME/ (brain.log, routes.json, dry-run/), $DATA_ROOT/runs/NAME-s{0,1,2}/ (brain-last.pt, result.json),
# the sign readings (derived/road_type_signs) and the frame cache. It never imports reviews, changes the labels, the model in use or the car, and
# exports nothing: Publish exports the brain chosen in Models and checks it along a real drive. Run it again with the same
# NAME to finish: a finished seed is not repeated, the signs already read are not read again, the drives are the same.
#
# Environment: NNSLR_GPU_INDEX (default 0), NNSLR_GPU_VENV (.venv-gpu), NNSLR_BRAIN_EPOCHS (40), NNSLR_SIGN_WORKERS
# (processes reading the signs as the car does: half the cores, each with its ffmpeg),
# NNSLR_SIGN_GPUS (the cards of step 3, nvidia-smi's numbers: 0,1;
# default the brain's and every other card free then), NNSLR_BRAIN_WORKS (1: it also learns the
# roadworks; 0: not), NNSLR_BRAIN_ROAD (1: it also learns the kinds of road of Speed review; 0: not),
# NNSLR_BRAIN_TYPES (1: it also learns the road type of the map and of Speed review, and only from the seconds that have
# one; 0: not, every second),
# NNSLR_BRAIN_POOL (grid: the picture's features keep where they are; mean: not), NNSLR_BRAIN_INIT_FROM (an earlier
# brain's brain-last.pt every seed goes on from, instead of ImageNet and random weights: Train › Brain's "Start from";
# the dry run of step 1 refuses one stepped otherwise or learnt on another split), NNSLR_BRAIN_CUTTER,
# NNSLR_BRAIN_CLASSIFIER, NNSLR_BRAIN_DECIDER (the networks its signs are read with, ids of the Models tab: a SignNet,
# a classifier, an applicability network; empty: the one in use), NNSLR_BRAIN_SEEDS (the seeds it trains: "0 1 2", or
# "0" alone; Train › Brain's Seeds), NNSLR_SUNNYPILOT_CLONE
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
SIGN_GPUS="${NNSLR_SIGN_GPUS:-}"
WORKS="${NNSLR_BRAIN_WORKS:-1}"
ROAD="${NNSLR_BRAIN_ROAD:-1}"
TYPES="${NNSLR_BRAIN_TYPES:-1}"
POOL="${NNSLR_BRAIN_POOL:-grid}"
INIT_FROM="${NNSLR_BRAIN_INIT_FROM:-}"
SIGN_CUTTER="${NNSLR_BRAIN_CUTTER:-}"
SIGN_CLASSIFIER="${NNSLR_BRAIN_CLASSIFIER:-}"
SIGN_DECIDER="${NNSLR_BRAIN_DECIDER:-}"
read -r -a SEEDS <<< "${NNSLR_BRAIN_SEEDS:-0 1 2}"

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

if [[ "${SEEDS[*]}" != "0 1 2" && "${SEEDS[*]}" != "0" ]]; then
  echo "error: NNSLR_BRAIN_SEEDS must be \"0 1 2\" or \"0\": ${NNSLR_BRAIN_SEEDS:-}" >&2
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

for setting in SIGN_CUTTER SIGN_CLASSIFIER SIGN_DECIDER; do
  if [[ -n "${!setting}" && ! "${!setting}" =~ ^(cutter|classifier|decider):[A-Za-z0-9][A-Za-z0-9._/-]*$ ]]; then
    echo "error: NNSLR_BRAIN_${setting#SIGN_} must be an id of the Models tab: ${!setting}" >&2
    exit 2
  fi
done

if [[ -n "$INIT_FROM" && ! -f "$INIT_FROM" ]]; then
  echo "error: NNSLR_BRAIN_INIT_FROM is not a file: $INIT_FROM" >&2
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
if [[ -n "$INIT_FROM" ]]; then
  SETTINGS+=(--init-from "$INIT_FROM")
fi

echo "== Brain =="
echo "run:          $RUN_ID"
echo "seeds:        ${SEEDS[*]}"
echo "output:       $(printf "$RUN_ROOT-s%s " "${SEEDS[@]}")"
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
# The networks they are read with, chosen in Train › Brain (those in use by default), exported on the brain's card into
# the run (networks.json): a run started again keeps its own.
NETWORKS_FILE="$RUN_ROOT/networks.json"
NETWORK_ARGS=()
if [[ -n "$SIGN_CUTTER" ]]; then NETWORK_ARGS+=(--cutter "$SIGN_CUTTER"); fi
if [[ -n "$SIGN_CLASSIFIER" ]]; then NETWORK_ARGS+=(--classifier "$SIGN_CLASSIFIER"); fi
if [[ -n "$SIGN_DECIDER" ]]; then NETWORK_ARGS+=(--decider "$SIGN_DECIDER"); fi
"$NN" brain-networks --run-dir "$RUN_ROOT" --data-root "$DATA_ROOT" "${NETWORK_ARGS[@]}"
SIGN_ARGS=(--sign-networks "$NETWORKS_FILE")
# The signs on the GPU: .venv has onnxruntime-gpu, .venv-gpu only the CPU's (torch brings its own CUDA).
SIGNS_NN="$ROOT_DIR/.venv/bin/nnslr"
[ -x "$SIGNS_NN" ] || SIGNS_NN="$NN"
# The cards that read them (nnslr borrow-cards): the brain's, and every other card no compute process is on and no
# other job of the panel holds; or those of NNSLR_SIGN_GPUS. In a job of the panel the others are written to its record
# (NNSLR_JOB_RECORD): the panel names them as the brain's and starts nothing else on them until this step ends. Chosen
# here, not at the start (the checks above take a minute). Without an answer, the brain's card alone.
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
  "${SIGN_ARGS[@]}" "${ROUTE_ARGS[@]}"
"$SIGNS_NN" borrow-cards --release --data-root "$DATA_ROOT" || true
if [[ "$TYPES" == "0" ]]; then
  "$NN" brain-targets --data-root "$DATA_ROOT" "${SIGN_ARGS[@]}" "${ROUTE_ARGS[@]}" --no-road-types
else
  "$NN" brain-targets --data-root "$DATA_ROOT" "${SIGN_ARGS[@]}" "${ROUTE_ARGS[@]}"
fi

# Each seed its own step, 4 to 6 (job_progress: seed 0 alone is step 4 alone).
for seed in "${SEEDS[@]}"; do
  echo "== $(( 4 + seed ))/6 Train seed $seed =="
  if [[ -f "$RUN_ROOT-s$seed/result.json" ]]; then
    echo "skip seed $seed: already trained"
    continue
  fi
  "$NN" train-brain --run-dir "$RUN_ROOT-s$seed" --seed "$seed" "${SETTINGS[@]}" "${SIGN_ARGS[@]}" "${ROUTE_ARGS[@]}"
done

echo "Brain complete: choose a seed in Models › Brain, then Publish."
