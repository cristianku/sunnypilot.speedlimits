#!/usr/bin/env bash
set -euo pipefail

# The portal's "Train UserNet": UserNet, the road users' detector (nnslr_tools.usernet): person, two-wheeler and
# vehicle in the window 672 x 256 of the road frame (the frame scaled to 672 across, its rows 72 to 328), grid 8,
# fuse 1 x 1, channels 16,24,32,48. Two stages, one a run:
#
#   NNSLR_USERNET_STAGE=bdd     BDD100K's pictures (NNSLR_BDD_DIR, samples.json): one window a picture, read as it
#                               trains (nothing is cut to disk). The weights then carry uses_bdd (their licence).
#   NNSLR_USERNET_STAGE=drives  our reviewed frames: the road-user boxes of Sign review; the frames ticked "road
#                               users complete" teach the negatives too and are validation and test. Usually from a
#                               bdd run (NNSLR_USERNET_INIT_FROM), uses_bdd kept.
#
# The best epoch is the one with the best validation recall at score 0.5 (what the brain will read); it is then
# measured on validation and test (result.json). Validation is our ticked frames; the bdd stage uses BDD's own
# validation pictures only while there are none (result.json's validation_source).
#
# usage: scripts/run_usernet.sh NAME
#
# Writes only into $DATA_ROOT/runs/NAME/: usernet-best.pt, result.json (nnslr train-usernet --run-dir) and
# usernet.log (everything this script prints). It never imports reviews, builds splits, cuts pictures, kills GPU
# processes, exports, deploys or puts the UserNet in use. A NAME holding another kind of run or a finished UserNet is
# refused; a NAME with usernet-best.pt but no result.json (stopped half way: no resume) is trained again from
# scratch. The log of a NAME tried before is moved to usernet.log.1.
#
# Never the CPU: without nvidia-smi or the card it stops (exit 2), a card with compute processes on it is refused
# (exit 3) unless NNSLR_ALLOW_BUSY_GPU=1, and the training itself stops when no card is visible. One training a card.
#
# Environment: NNSLR_USERNET_STAGE (bdd or drives; default bdd), NNSLR_USERNET_INIT_FROM (a finished UserNet's
# usernet-best.pt to go on from: Train › UserNet › Start from), NNSLR_USERNET_EPOCHS (30), NNSLR_USERNET_BATCH (32),
# NNSLR_USERNET_LEARNING_RATE (0.003), NNSLR_NUM_WORKERS (half the cores, 4 to 16) (the portal sets those changed in
# Train › UserNet › Settings, nnslr_tools.train_settings), NNSLR_BDD_DIR (/data/sunnypilot/SPEEDLIMITS/bdd100k),
# NNSLR_GPU_INDEX (0, nvidia-smi's number), NNSLR_DATASET_KIND (gold), NNSLR_GPU_VENV (.venv-gpu),
# NNSLR_ALLOW_BUSY_GPU=1 to share a card that already has compute processes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
RUN_ID="${1:-}"
DATASET_KIND="${NNSLR_DATASET_KIND:-gold}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
STAGE="${NNSLR_USERNET_STAGE:-bdd}"
INIT_FROM="${NNSLR_USERNET_INIT_FROM:-}"
EPOCHS="${NNSLR_USERNET_EPOCHS:-30}"
BATCH="${NNSLR_USERNET_BATCH:-32}"
LEARNING_RATE="${NNSLR_USERNET_LEARNING_RATE:-0.003}"
BDD_DIR="${NNSLR_BDD_DIR:-/data/sunnypilot/SPEEDLIMITS/bdd100k}"
# The processes reading the pictures: half the cores, 4 to 16 (nnslr_tools.training.DEFAULT_WORKERS).
CORES="$(nproc 2>/dev/null || echo 8)"
WORKERS_DEFAULT=$(( CORES / 2 < 4 ? 4 : (CORES / 2 > 16 ? 16 : CORES / 2) ))
NUM_WORKERS="${NNSLR_NUM_WORKERS:-$WORKERS_DEFAULT}"

if [[ -z "$RUN_ID" || $# -gt 1 ]]; then
  echo "usage: $0 NAME" >&2
  echo "example: NNSLR_USERNET_STAGE=bdd NNSLR_GPU_INDEX=0 $0 2026-11-12_1200-users" >&2
  exit 2
fi

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: NAME contains unsupported characters" >&2
  exit 2
fi

if [[ "$STAGE" != "bdd" && "$STAGE" != "drives" ]]; then
  echo "error: NNSLR_USERNET_STAGE must be bdd or drives: $STAGE" >&2
  exit 2
fi

if [[ "$DATASET_KIND" != "gold" && "$DATASET_KIND" != "training-candidate" ]]; then
  echo "error: NNSLR_DATASET_KIND must be gold or training-candidate" >&2
  exit 2
fi

for setting in EPOCHS BATCH NUM_WORKERS; do
  if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: $setting must be a whole number above 0: ${!setting}" >&2
    exit 2
  fi
done

# A positive number: 0.003, 3e-3, 1e-05.
if [[ ! "$LEARNING_RATE" =~ ^([0-9]*\.)?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
  echo "error: NNSLR_USERNET_LEARNING_RATE must be a number: $LEARNING_RATE" >&2
  exit 2
fi

if [[ -n "$INIT_FROM" && ! -f "$INIT_FROM" ]]; then
  echo "error: NNSLR_USERNET_INIT_FROM is not a file: $INIT_FROM" >&2
  exit 2
fi

if [[ "$STAGE" == "bdd" && ! -s "$BDD_DIR/samples.json" ]]; then
  echo "error: BDD100K is not there: $BDD_DIR/samples.json (NNSLR_BDD_DIR)" >&2
  exit 2
fi

if [[ ! -x "$VENV/bin/nnslr" || ! -x "$VENV/bin/python" ]]; then
  echo "error: GPU environment missing: $VENV/bin/nnslr" >&2
  echo "run: bash scripts/setup_v100_training_env.sh" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi not found: UserNet trains on a GPU only, never on the CPU" >&2
  exit 2
fi

if ! nvidia-smi -i "$GPU_INDEX" >/dev/null 2>&1; then
  echo "error: invalid/unavailable GPU index: $GPU_INDEX (UserNet trains on a GPU only, never on the CPU)" >&2
  exit 2
fi

BUSY="$(
  nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "$BUSY" && "${NNSLR_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "error: selected GPU already has compute processes; refusing to train UserNet (one training a card)" >&2
  echo "$BUSY" >&2
  echo "Free the card, or set NNSLR_ALLOW_BUSY_GPU=1 only after intentionally accepting contention." >&2
  exit 3
fi

RUN_ROOT="$DATA_ROOT/runs/$RUN_ID"
# A folder of another kind of run is never written into.
for part in detector reader export bundle evaluation experiment-b detector-sizes classifier pretrain signnet-best.pt \
            signnet.log negatives.jsonl classifier.log road-type road-type.log brain.log routes.json; do
  if [[ -e "$RUN_ROOT/$part" ]]; then
    echo "error: $RUN_ROOT is another kind of run (it has $part): choose a new NAME" >&2
    exit 3
  fi
done
if [[ -f "$RUN_ROOT/result.json" ]]; then
  echo "error: $RUN_ROOT already holds a finished UserNet (result.json): choose a new NAME" >&2
  exit 3
fi
RESTART=0
if [[ -e "$RUN_ROOT/usernet-best.pt" ]]; then
  RESTART=1
fi

SETTINGS=(--stage "$STAGE" --bdd-dir "$BDD_DIR" --epochs "$EPOCHS" --batch "$BATCH" --learning-rate "$LEARNING_RATE")
if [[ -n "$INIT_FROM" ]]; then
  SETTINGS+=(--init-from "$INIT_FROM")
fi
NN="$VENV/bin/nnslr"

mkdir -p "$RUN_ROOT"
# A log of this NAME tried before: kept aside as usernet.log.1, the new one starts empty (the panel reads the last
# line of each epoch, so old epochs would show as the new run's).
OLD_LOG=0
if [[ -e "$RUN_ROOT/usernet.log" ]]; then
  mv -f "$RUN_ROOT/usernet.log" "$RUN_ROOT/usernet.log.1"
  OLD_LOG=1
fi
exec > >(tee -a "$RUN_ROOT/usernet.log") 2>&1

export NNSLR_DATA_ROOT="$DATA_ROOT"
# The index is nvidia-smi's, the card checked above: CUDA counts the cards in the same order, by PCI bus.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONUNBUFFERED=1

echo "== scripts/run_usernet.sh $RUN_ID"
echo "run:          $RUN_ID"
echo "stage:        $STAGE"
echo "window:       672x256 (the frame scaled to 672 across, rows 72-328), grid 8, fuse 1, widths 16,24,32,48"
echo "settings:     ${SETTINGS[*]}"
echo "workers:      $NUM_WORKERS"
echo "output:       $RUN_ROOT"
echo "data root:    $DATA_ROOT"
echo "dataset kind: $DATASET_KIND"
echo "physical GPU: $GPU_INDEX"
echo "best epoch:   validation recall at score 0.5, at most 24 users a frame"
if [[ "$STAGE" == "bdd" || -n "$INIT_FROM" ]]; then
  echo "licence:      learns BDD100K's pictures (bdd stage) or goes on from weights that may have: uses_bdd is kept with the weights"
fi
echo
if [[ "$RESTART" == "1" ]]; then
  echo "restart from scratch: $RUN_ROOT/usernet-best.pt has no result.json (a training stopped half way; UserNet has no resume)"
fi
if [[ "$OLD_LOG" == "1" ]]; then
  echo "the log of the earlier try: $RUN_ROOT/usernet.log.1"
fi

echo "== 1/3 Check the training population (CPU-only) =="
PLANS="$(mktemp -d)"
trap 'rm -rf "$PLANS"' EXIT
"$NN" train-usernet --dry-run --run-dir "$RUN_ROOT" --dataset-kind "$DATASET_KIND" "${SETTINGS[@]}" \
  --plan-out "$PLANS/plan.json"

echo "== 2/3 Explicit CUDA smoke test =="
# CUDA_VISIBLE_DEVICES maps the selected physical GPU to logical cuda:0.
"$NN" env --gpu-smoke --gpu-device 0

echo "== 3/3 Train UserNet =="
# The epoch lines ("usernet epoch N/M train_loss ... val_recall ... val_false_per_frame ... 87s"), the measure of the
# best epoch and "UserNet complete: <run>" go to the terminal and to usernet.log.
"$NN" train-usernet --run-dir "$RUN_ROOT" --dataset-kind "$DATASET_KIND" "${SETTINGS[@]}" --workers "$NUM_WORKERS" \
  --plan "$PLANS/plan.json"

echo
echo "Nothing was imported, exported, deployed or put in use."
