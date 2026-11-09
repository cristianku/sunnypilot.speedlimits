#!/usr/bin/env bash
set -euo pipefail

# The classifier of the pipeline (cutter -> classifier -> decider, docs/NEURAL_NETWORK.md, step 2 of the plan): the
# network that says what each box of the cutter is, "not a sign" included. On one GPU, one step after the other:
#
#   1. checks the dataset, the split and the cutter on the CPU (the cutter in use of the Models tab by default: a
#      SignNet, the network the comma runs, or an old Detector a run started with);
#   2. a CUDA smoke test;
#   3. mines the false boxes of that cutter (nnslr mine-negatives): on every reviewed frame of train, validation and
#      test, its boxes of score >= 0.1 (with a SignNet among the 5 a frame the car passes on) that overlap no
#      reviewed box, at most 3 per frame and 3 x the sign crops of each partition, from the frames a person reviewed
#      first; each partition from its own frames only; in the frames an AI reviewed before the other-sign rule (it
#      left those signs without a box) only boxes under 0.3, and the other-sign proposals it removed there are not
#      "not a sign" either;
#   4. trains the classifier (nnslr train --task reader --classes v2): the reviewed sign crops with their labels and
#      the kinds of other signs, and "not a sign" from the proposals reviewers removed, the boxes marked not a sign
#      and the mined boxes; ImageNet start, darker and lower-contrast crops with jittered boxes, best epoch by the
#      validation balanced accuracy (signs read right, not_a_sign rejected), early stop after 5 epochs without.
#      With NNSLR_CLASSIFIER_MAPILLARY=pretrain, two stages: first the network learns the Mapillary crops alone
#      (pretrain/: every crop of a v2 class, no cap per label, none of ours, from ImageNet; the epoch chosen on our
#      validation signs), then our crops alone from those weights (classifier/, --init-from pretrain/reader-best.pt,
#      the same class list, "not a sign" learnt there); NNSLR_CLASSIFIER_INIT_FROM starts from another v2
#      reader-best.pt instead of ImageNet;
#   5. evaluates it alone on test and validation (every crop, the signs, the not_a_sign crops by source, the crops a
#      person confirmed);
#   6. evaluates the chain on test and validation (nnslr evaluate-chain): the cutter on every frame, each box cut at
#      full resolution and classified; the speed signs found AND read right, overall, by width and on the signs a
#      person confirmed, and the false boxes per frame that survive, at cutter scores 0.1, 0.2 and 0.3 with a SignNet
#      (the car's 5 boxes a frame; 0.3 is the car's), 0.05, 0.1 and 0.25 with an old Detector;
#   7. runs the same chain on test with the old reader (NNSLR_OLD_READER_RUN, default 2026-10-24_1452) and prints a
#      short comparison, also saved as comparison.txt.
#
# usage: scripts/run_classifier.sh NAME
#
# Writes only into $DATA_ROOT/runs/NAME/: classifier.log (everything printed), negatives.jsonl, classifier/
# (nnslr train --output: training-plan.json, training-state.pt, reader-best.pt, result.json, wandb/), pretrain/ (the
# same for the first stage of NNSLR_CLASSIFIER_MAPILLARY=pretrain),
# evaluation/classifier-{test,validation}.json, evaluation/chain-{test,validation}.json,
# evaluation/chain-test-old-reader.json and comparison.txt. It never imports reviews, builds splits, changes the
# model in use, kills GPU processes, exports or deploys anything. Run it again with the same NAME to finish: the
# negatives are mined once, a finished training is not repeated, an interrupted one resumes with the settings it
# started with, an evaluation older than the weights is redone.
#
# Environment: NNSLR_GPU_INDEX (default 0), NNSLR_DATASET_KIND (gold), NNSLR_GPU_VENV (.venv-gpu),
# NNSLR_CUTTER (active: the cutter in use; or a signnet-best.pt, or an old detector-best.pt), NNSLR_CLASSIFIER_EPOCHS (30),
# NNSLR_CLASSIFIER_PATIENCE (5), NNSLR_CLASSIFIER_BATCH_SIZE (64), NNSLR_NUM_WORKERS (half the cores, 4 to 16),
# NNSLR_NEGATIVE_SCORE (0.1), NNSLR_NEGATIVES_PER_FRAME (3), NNSLR_NEGATIVES_FACTOR (3),
# NNSLR_OLD_READER_RUN (2026-10-24_1452; empty: no comparison), NNSLR_CLASSIFIER_AUGMENT=0 to train without the
# darker crops and box jitter, NNSLR_CLASSIFIER_MAPILLARY=1 to add the signs of the Mapillary Traffic Sign Dataset
# to train (the crops of helpers/mapillary.sh, $DATA_ROOT/external/mapillary/crops.jsonl; nnslr_tools.mapillary),
# with NNSLR_CLASSIFIER_MAPILLARY_REGION (all; or continents and countries, comma-separated: europe or CH,DE,IT; needs
# the positions of helpers/mapillary_locations.sh) and NNSLR_CLASSIFIER_MAPILLARY_SHARE (100: the percentage of the
# region's crops, always the same ones for the same share), NNSLR_CLASSIFIER_MAPILLARY=pretrain for the two stages
# (Mapillary alone, then ours from it; region and share choose the crops of the first), NNSLR_CLASSIFIER_PRETRAIN_EPOCHS
# (the epochs of the first stage, default NNSLR_CLASSIFIER_EPOCHS), NNSLR_CLASSIFIER_INIT_FROM (a v2 reader-best.pt
# to start from instead of ImageNet), NNSLR_TRAIN_INIT=random, NNSLR_ALLOW_BUSY_GPU=1 to share a GPU that already has compute processes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
RUN_ID="${1:-}"
DATASET_KIND="${NNSLR_DATASET_KIND:-gold}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
CUTTER_SPEC="${NNSLR_CUTTER:-active}"
EPOCHS="${NNSLR_CLASSIFIER_EPOCHS:-30}"
PATIENCE="${NNSLR_CLASSIFIER_PATIENCE:-5}"
BATCH_SIZE="${NNSLR_CLASSIFIER_BATCH_SIZE:-64}"
PRETRAIN_EPOCHS="${NNSLR_CLASSIFIER_PRETRAIN_EPOCHS:-$EPOCHS}"
MAPILLARY_MODE="${NNSLR_CLASSIFIER_MAPILLARY:-0}"
INIT_FROM="${NNSLR_CLASSIFIER_INIT_FROM:-}"
# The processes reading the frames: half the cores, 4 to 16 (nnslr_tools.training.DEFAULT_WORKERS).
CORES="$(nproc 2>/dev/null || echo 8)"
WORKERS_DEFAULT=$(( CORES / 2 < 4 ? 4 : (CORES / 2 > 16 ? 16 : CORES / 2) ))
NUM_WORKERS="${NNSLR_NUM_WORKERS:-$WORKERS_DEFAULT}"
NEGATIVE_SCORE="${NNSLR_NEGATIVE_SCORE:-0.1}"
PER_FRAME="${NNSLR_NEGATIVES_PER_FRAME:-3}"
FACTOR="${NNSLR_NEGATIVES_FACTOR:-3}"
OLD_READER_RUN="${NNSLR_OLD_READER_RUN-2026-10-24_1452}"
PARTITIONS=(test validation)

if [[ -z "$RUN_ID" || $# -ne 1 ]]; then
  echo "usage: $0 NAME" >&2
  echo "example: NNSLR_GPU_INDEX=0 $0 2026-10-24_1730-classifier" >&2
  exit 2
fi

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: NAME contains unsupported characters" >&2
  exit 2
fi

if [[ -n "$OLD_READER_RUN" && ! "$OLD_READER_RUN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: NNSLR_OLD_READER_RUN contains unsupported characters" >&2
  exit 2
fi

if [[ "$DATASET_KIND" != "gold" && "$DATASET_KIND" != "training-candidate" ]]; then
  echo "error: NNSLR_DATASET_KIND must be gold or training-candidate" >&2
  exit 2
fi

for setting in EPOCHS PRETRAIN_EPOCHS PATIENCE BATCH_SIZE NUM_WORKERS PER_FRAME; do
  if [[ ! "${!setting}" =~ ^[0-9]+$ ]]; then
    echo "error: $setting must be a whole number: ${!setting}" >&2
    exit 2
  fi
done

for setting in NEGATIVE_SCORE FACTOR; do
  if [[ ! "${!setting}" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "error: $setting must be a number: ${!setting}" >&2
    exit 2
  fi
done

if [[ "$MAPILLARY_MODE" != "0" && "$MAPILLARY_MODE" != "1" && "$MAPILLARY_MODE" != "pretrain" ]]; then
  echo "error: NNSLR_CLASSIFIER_MAPILLARY must be 0, 1 or pretrain: $MAPILLARY_MODE" >&2
  exit 2
fi

if [[ -n "$INIT_FROM" && "$MAPILLARY_MODE" == "pretrain" ]]; then
  echo "error: NNSLR_CLASSIFIER_INIT_FROM and NNSLR_CLASSIFIER_MAPILLARY=pretrain are two different starts" >&2
  exit 2
fi

if [[ -n "$INIT_FROM" && ! -f "$INIT_FROM" ]]; then
  echo "error: NNSLR_CLASSIFIER_INIT_FROM is not a file: $INIT_FROM" >&2
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
  echo "error: selected GPU already has compute processes; refusing to train the classifier" >&2
  echo "$BUSY" >&2
  echo "Set NNSLR_ALLOW_BUSY_GPU=1 only after intentionally accepting contention." >&2
  exit 3
fi

RUN_ROOT="$DATA_ROOT/runs/$RUN_ID"
CLASSIFIER_RUN="$RUN_ROOT/classifier"
PRETRAIN_RUN="$RUN_ROOT/pretrain"
EVAL_DIR="$RUN_ROOT/evaluation"
NEGATIVES="$RUN_ROOT/negatives.jsonl"
# A folder of another kind of run (the training pipeline, experiment B, the detector sizes) is never written into,
# nor a SignNet's run (the cutter: its weights at the root of the run).
for part in detector reader export bundle experiment-b detector-sizes; do
  if [[ -e "$RUN_ROOT/$part" ]]; then
    echo "error: $RUN_ROOT is another kind of run (it has $part/): choose a new NAME" >&2
    exit 3
  fi
done
if [[ -e "$RUN_ROOT/signnet-best.pt" ]]; then
  echo "error: $RUN_ROOT is another kind of run (it has signnet-best.pt, a SignNet): choose a new NAME" >&2
  exit 3
fi
for folder in "$CLASSIFIER_RUN" "$PRETRAIN_RUN"; do
  if [[ -e "$folder" && ! -f "$folder/result.json" && ! -f "$folder/training-state.pt" ]]; then
    echo "error: $folder exists without result.json or training-state.pt" >&2
    echo "remove that folder or choose a new NAME" >&2
    exit 3
  fi
done

mkdir -p "$EVAL_DIR"
# Everything printed from here on also goes to the run's own log (the Evaluate tab reads its epochs).
exec > >(tee -a "$RUN_ROOT/classifier.log") 2>&1

export NNSLR_DATA_ROOT="$DATA_ROOT"
# The index is nvidia-smi's, the card checked above: CUDA counts the cards in the same order, by PCI bus (its
# default order, fastest first, leaves two identical V100s in no stated order).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
# The wandb group of the classifier run (nnslr_tools.tracking).
export NNSLR_RUN_ID="$RUN_ID"

# wandb: "wandb login" saves the key in ~/.netrc; without it, offline.
# WANDB_MODE=disabled turns wandb off completely.
if [ -z "${WANDB_MODE:-}" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  if ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    export WANDB_MODE=offline
    echo "wandb: not logged in, going offline (the numbers stay in the terminal and in $CLASSIFIER_RUN/wandb)."
  fi
fi

NN="$VENV/bin/nnslr"
OLD_READER=""
OLD_ROOT=""
if [[ -n "$OLD_READER_RUN" && -f "$DATA_ROOT/runs/$OLD_READER_RUN/reader/reader-best.pt" ]]; then
  OLD_ROOT="$DATA_ROOT/runs/$OLD_READER_RUN"
  OLD_READER="$OLD_ROOT/reader/reader-best.pt"
fi

SETTINGS=(--epochs "$EPOCHS" --patience "$PATIENCE" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS")
IMAGENET=()
if [[ "${NNSLR_TRAIN_INIT:-imagenet}" != "random" ]]; then
  IMAGENET=(--pretrained)
fi
AUGMENT=()
if [[ "${NNSLR_CLASSIFIER_AUGMENT:-1}" == "0" ]]; then
  AUGMENT=(--no-augment)
fi
MAPILLARY=()
if [[ "$MAPILLARY_MODE" != "0" ]]; then
  if [[ ! -s "$DATA_ROOT/external/mapillary/crops.jsonl" ]]; then
    echo "error: Mapillary is not prepared: run helpers/mapillary.sh first" >&2
    exit 2
  fi
  MAPILLARY=(--mapillary "$DATA_ROOT/external/mapillary/crops.jsonl")
  if [[ -n "${NNSLR_CLASSIFIER_MAPILLARY_REGION:-}" && "${NNSLR_CLASSIFIER_MAPILLARY_REGION}" != "all" ]]; then
    if [[ ! -s "$DATA_ROOT/external/mapillary/locations.jsonl" ]]; then
      echo "error: the positions of the Mapillary photos are not fetched: run helpers/mapillary_locations.sh first" >&2
      exit 2
    fi
    MAPILLARY+=(--mapillary-region "$NNSLR_CLASSIFIER_MAPILLARY_REGION")
  fi
  if [[ -n "${NNSLR_CLASSIFIER_MAPILLARY_SHARE:-}" && "${NNSLR_CLASSIFIER_MAPILLARY_SHARE}" != "100" ]]; then
    MAPILLARY+=(--mapillary-share "$NNSLR_CLASSIFIER_MAPILLARY_SHARE")
  fi
elif [[ "${NNSLR_CLASSIFIER_MAPILLARY_REGION:-all}" != "all" || "${NNSLR_CLASSIFIER_MAPILLARY_SHARE:-100}" != "100" ]]; then
  echo "error: NNSLR_CLASSIFIER_MAPILLARY_REGION and _SHARE choose Mapillary crops: add NNSLR_CLASSIFIER_MAPILLARY=1" >&2
  exit 2
fi
# The two stages (NNSLR_CLASSIFIER_MAPILLARY=pretrain): the first on the Mapillary crops alone, from ImageNet; the
# second on ours alone, from the first's best epoch. Otherwise one training, from ImageNet or NNSLR_CLASSIFIER_INIT_FROM.
PRETRAIN_SETTINGS=()
if [[ "$MAPILLARY_MODE" == "pretrain" ]]; then
  PRETRAIN_SETTINGS=(--epochs "$PRETRAIN_EPOCHS" --patience "$PATIENCE" --batch-size "$BATCH_SIZE"
                     --num-workers "$NUM_WORKERS" "${IMAGENET[@]}" "${AUGMENT[@]}" "${MAPILLARY[@]}" --mapillary-only)
  SETTINGS+=("${AUGMENT[@]}" --init-from "$PRETRAIN_RUN/reader-best.pt")
elif [[ -n "$INIT_FROM" ]]; then
  SETTINGS+=("${AUGMENT[@]}" "${MAPILLARY[@]}" --init-from "$INIT_FROM")
else
  SETTINGS+=("${IMAGENET[@]}" "${AUGMENT[@]}" "${MAPILLARY[@]}")
fi

echo "== Classifier (cutter -> classifier -> decider) =="
echo "run:          $RUN_ID"
echo "output:       $RUN_ROOT"
echo "data root:    $DATA_ROOT"
echo "dataset kind: $DATASET_KIND"
echo "physical GPU: $GPU_INDEX"
echo "cutter:       $CUTTER_SPEC"
echo "negatives:    cutter boxes of score >= $NEGATIVE_SCORE overlapping no reviewed box, at most $PER_FRAME per frame and $FACTOR x the sign crops per partition"
if [[ "$MAPILLARY_MODE" == "pretrain" ]]; then
  echo "pretraining:  --classes v2 ${PRETRAIN_SETTINGS[*]}"
fi
echo "settings:     --classes v2 ${SETTINGS[*]}"
echo "old reader:   ${OLD_READER:-none (no comparison)}"
echo "wandb:        project ${WANDB_PROJECT:-sunnypilot.speedlimits}, group $RUN_ID, mode ${WANDB_MODE:-online}"
echo

echo "== 1/7 Check the dataset, the split and the cutter (CPU-only) =="
"$NN" train --task reader --classes v2 --dataset-kind "$DATASET_KIND" "${MAPILLARY[@]}" --dry-run > /dev/null
if [[ "$MAPILLARY_MODE" == "pretrain" ]]; then
  "$NN" train --task reader --classes v2 --dataset-kind "$DATASET_KIND" "${MAPILLARY[@]}" --mapillary-only \
    --dry-run > /dev/null
fi
"$NN" mine-negatives --dataset-kind "$DATASET_KIND" --per-frame "$PER_FRAME" --total-factor "$FACTOR" --dry-run
for partition in "${PARTITIONS[@]}"; do
  "$NN" evaluate-chain --dataset-kind "$DATASET_KIND" --partition "$partition" --dry-run
done
# The cutter is fixed once for the whole run: mining and the chain use the same weights, even if the model in use
# changes meanwhile. A run that resumes keeps the cutter its negatives were mined with.
if [[ -f "$NEGATIVES" ]]; then
  CUTTER="$("$VENV/bin/python" -c 'import json, sys; print(json.loads(open(sys.argv[1]).readline())["cutter"]["checkpoint"])' "$NEGATIVES")"
else
  CUTTER="$("$VENV/bin/python" -c 'import sys; from nnslr_tools.chain import resolve_model; print(resolve_model(sys.argv[1], "cutter", sys.argv[2])["path"])' "$CUTTER_SPEC" "$DATA_ROOT")"
fi
if [[ ! -f "$CUTTER" ]]; then
  echo "error: the cutter's weights are missing: $CUTTER" >&2
  exit 3
fi
echo "cutter weights: $CUTTER"

echo "== 2/7 Explicit CUDA smoke test =="
# CUDA_VISIBLE_DEVICES maps the selected physical GPU to logical cuda:0.
"$NN" env --gpu-smoke --gpu-device 0

echo "== 3/7 Mine the cutter's false boxes =="
if [[ -f "$NEGATIVES" ]]; then
  echo "skip mined: $NEGATIVES"
else
  "$NN" mine-negatives --dataset-kind "$DATASET_KIND" --device cuda --cutter "$CUTTER" --output "$NEGATIVES" \
    --score-threshold "$NEGATIVE_SCORE" --per-frame "$PER_FRAME" --total-factor "$FACTOR" --num-workers "$NUM_WORKERS"
fi

echo "== 4/7 Train the classifier =="
# train OUTPUT WHAT FLAGS...: one training, skipped when finished, resumed with its own settings when interrupted.
train() {
  local output="$1" what="$2" mode=fresh
  shift 2
  local flags=(--classes v2 "$@")
  if [[ -f "$output/result.json" ]]; then
    echo "skip trained: $output/result.json"
    return
  fi
  if [[ -f "$output/training-state.pt" ]]; then
    mode=resume
    # A resumed run trains with exactly the settings it started with: they come from its training state.
    resume_flags="$("$VENV/bin/python" -c 'import sys; from nnslr_tools.training import resume_arguments_file; print("\n".join(resume_arguments_file(sys.argv[1])))' "$output/training-state.pt")"
    readarray -t flags <<< "$resume_flags"
    echo "resume the $what with the settings of its training state: ${flags[*]}"
  fi
  # The epoch lines ("classifier epoch N/M ...", "pretrain epoch N/M ...") and the final JSON go to the terminal
  # and to classifier.log.
  "$NN" train --task reader --dataset-kind "$DATASET_KIND" --device cuda --mode "$mode" --output "$output" \
    --negatives "$NEGATIVES" "${flags[@]}"
}
if [[ "$MAPILLARY_MODE" == "pretrain" ]]; then
  echo "-- the first stage: the Mapillary crops alone --"
  train "$PRETRAIN_RUN" pretraining "${PRETRAIN_SETTINGS[@]}"
  echo "-- the second stage: our crops alone, from the first --"
fi
train "$CLASSIFIER_RUN" classifier "${SETTINGS[@]}"
CHECKPOINT="$CLASSIFIER_RUN/reader-best.pt"

echo "== 5/7 Evaluate the classifier alone =="
for partition in "${PARTITIONS[@]}"; do
  report="$EVAL_DIR/classifier-$partition.json"
  if [[ -f "$report" && "$report" -nt "$CHECKPOINT" ]]; then
    echo "skip evaluated $partition: $report"
    continue
  fi
  # The report (every crop) goes to the file only; the comparison below reads it.
  "$NN" evaluate --task reader --classes v2 --negatives "$NEGATIVES" --dataset-kind "$DATASET_KIND" \
    --partition "$partition" --device cuda --num-workers "$NUM_WORKERS" --checkpoint "$CHECKPOINT" \
    --output "$report" > /dev/null
  echo "evaluated the classifier on $partition: $report"
done

echo "== 6/7 Evaluate the chain =="
for partition in "${PARTITIONS[@]}"; do
  report="$EVAL_DIR/chain-$partition.json"
  if [[ -f "$report" && "$report" -nt "$CHECKPOINT" ]]; then
    echo "skip evaluated chain $partition: $report"
    continue
  fi
  "$NN" evaluate-chain --dataset-kind "$DATASET_KIND" --partition "$partition" --device cuda \
    --num-workers "$NUM_WORKERS" --cutter "$CUTTER" --classifier "$CHECKPOINT" --output "$report" > /dev/null
  echo "evaluated the chain on $partition: $report"
done

echo "== 7/7 Compare with the old reader =="
if [[ -n "$OLD_READER" ]]; then
  report="$EVAL_DIR/chain-test-old-reader.json"
  if [[ -f "$report" ]]; then
    echo "skip evaluated chain with the old reader: $report"
  else
    "$NN" evaluate-chain --dataset-kind "$DATASET_KIND" --partition test --device cuda --num-workers "$NUM_WORKERS" \
      --cutter "$CUTTER" --classifier "$OLD_READER" --output "$report" > /dev/null
    echo "evaluated the chain on test with the old reader: $report"
  fi
else
  echo "no old reader (NNSLR_OLD_READER_RUN=${OLD_READER_RUN:-}): the new classifier alone"
fi
echo
{
  if [[ "$MAPILLARY_MODE" == "pretrain" ]]; then
    echo "run $RUN_ID: pretraining --classes v2 ${PRETRAIN_SETTINGS[*]}"
  fi
  echo "run $RUN_ID: classifier --classes v2 ${SETTINGS[*]}; cutter $CUTTER; old reader ${OLD_READER:-none}"
  echo
  "$VENV/bin/python" -c 'import sys; from pathlib import Path; from nnslr_tools.chain import classifier_comparison; print(classifier_comparison(Path(sys.argv[1]), Path(sys.argv[2]) if sys.argv[2] else None), end="")' \
    "$RUN_ROOT" "$OLD_ROOT"
} | tee "$RUN_ROOT/comparison.txt"
echo
echo "Classifier complete: $RUN_ROOT"
echo "Nothing was imported, exported, deployed or published, and the model in use did not change."
