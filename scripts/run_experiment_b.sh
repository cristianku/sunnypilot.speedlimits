#!/usr/bin/env bash
set -euo pipefail

# Experiment B (docs/NEURAL_NETWORK.md, step 2 of the plan): on the boxes already reviewed, train the
# three variants crop, crop_geom and full one after the other on one GPU, the same way and on the same
# frozen split, then print a comparison: reading accuracy, other_road precision/recall, ownership
# macro-F1 and abstain rate, on test and on the human-reviewed boxes.
#
# Writes only into $DATA_ROOT/runs/<RUN_NAME>/experiment-b/<variant>/ (and the wandb files there).
# It never imports reviews, builds splits, kills GPU processes or deploys anything. Run it again with
# the same RUN_NAME to finish the variants without a result.json; a variant folder left without one
# by an interrupted run must be removed first (or use a new name).
#
# Environment: NNSLR_GPU_INDEX (default 0), NNSLR_DATASET_KIND (gold), NNSLR_GPU_VENV (.venv-gpu),
# NNSLR_EXPERIMENT_EPOCHS (20), NNSLR_TRAIN_INIT=random for random weights instead of ImageNet ones,
# NNSLR_DECIDER_INIT_FROM (the experiment-b folder of an earlier run: each variant goes on from its own there, a
# variant it lacks from ImageNet; Train › Applicability › Start from), NNSLR_ALLOW_BUSY_GPU=1 to share a GPU that
# already has compute processes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
RUN_ID="${1:-}"
DATASET_KIND="${NNSLR_DATASET_KIND:-gold}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
EPOCHS="${NNSLR_EXPERIMENT_EPOCHS:-20}"
START="${NNSLR_DECIDER_INIT_FROM:-}"
VARIANTS=(crop crop_geom full)

if [[ -z "$RUN_ID" ]]; then
  echo "usage: $0 RUN_NAME" >&2
  echo "example: NNSLR_GPU_INDEX=0 $0 experiment-b-001" >&2
  exit 2
fi

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: RUN_NAME contains unsupported characters" >&2
  exit 2
fi

if [[ "$DATASET_KIND" != "gold" && "$DATASET_KIND" != "training-candidate" ]]; then
  echo "error: NNSLR_DATASET_KIND must be gold or training-candidate" >&2
  exit 2
fi

if [[ ! "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: NNSLR_EXPERIMENT_EPOCHS must be a positive integer" >&2
  exit 2
fi

if [[ -n "$START" && ! -d "$START" ]]; then
  echo "error: NNSLR_DECIDER_INIT_FROM is not a folder: $START" >&2
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
  echo "error: selected GPU already has compute processes; refusing to start experiment B" >&2
  echo "$BUSY" >&2
  echo "Set NNSLR_ALLOW_BUSY_GPU=1 only after intentionally accepting contention." >&2
  exit 3
fi

RUN_ROOT="$DATA_ROOT/runs/$RUN_ID/experiment-b"
for variant in "${VARIANTS[@]}"; do
  if [[ -e "$RUN_ROOT/$variant" && ! -f "$RUN_ROOT/$variant/result.json" ]]; then
    echo "error: $RUN_ROOT/$variant exists without result.json (an interrupted run?)" >&2
    echo "remove that folder or choose a new RUN_NAME" >&2
    exit 3
  fi
done
mkdir -p "$RUN_ROOT"

export NNSLR_DATA_ROOT="$DATA_ROOT"
# The index is nvidia-smi's, the card checked above: CUDA counts the cards in the same order, by PCI bus (its
# default order, fastest first, leaves two identical V100s in no stated order).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
# The wandb group of the three variants (nnslr_tools.tracking).
export NNSLR_RUN_ID="$RUN_ID"

# wandb: "wandb login" saves the key in ~/.netrc; without it, offline.
# WANDB_MODE=disabled turns wandb off completely.
if [ -z "${WANDB_MODE:-}" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  if ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    export WANDB_MODE=offline
    echo "wandb: not logged in, going offline (the numbers stay in the terminal and in $RUN_ROOT/<variant>/wandb)."
  fi
fi

INIT=(--pretrained)
if [[ "${NNSLR_TRAIN_INIT:-imagenet}" == "random" ]]; then
  INIT=()
fi
# Where the variants start: ImageNet (or random weights), or each from its own in an earlier run (its run's name).
INIT_TEXT="${NNSLR_TRAIN_INIT:-imagenet}"
if [[ -n "$START" ]]; then
  INIT_TEXT="$(basename "$(dirname "$START")")"
fi

NN="$VENV/bin/nnslr"

echo "== Experiment B =="
echo "run:          $RUN_ID"
echo "output:       $RUN_ROOT"
echo "data root:    $DATA_ROOT"
echo "dataset kind: $DATASET_KIND"
echo "physical GPU: $GPU_INDEX"
echo "epochs:       $EPOCHS"
echo "init:         $INIT_TEXT"
echo "wandb:        project ${WANDB_PROJECT:-sunnypilot.speedlimits}, group $RUN_ID, mode ${WANDB_MODE:-online}"
echo

echo "== 1/5 Validate the experiment population (CPU-only) =="
if ! "$NN" experiment-b --variant full --dataset-kind "$DATASET_KIND" --dry-run > "$RUN_ROOT/dry-run.json"; then
  cat "$RUN_ROOT/dry-run.json"
  echo "error: the experiment population is not trainable (see above)" >&2
  exit 1
fi
"$VENV/bin/python" - "$RUN_ROOT/dry-run.json" <<'EOF'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
counts = plan["counts"]
print(f"dataset {plan['dataset_sha256']}  split {plan['split_sha256']} ({plan['split_policy']})")
for name in plan["partitions"]:
    owned = counts["ownership_by_partition"][name]
    print(f"{name:<10} boxes {counts['boxes_by_partition'][name]:>5}  human {counts['human_review_boxes_by_partition'][name]:>4}  "
          + "  ".join(f"{key} {value}" for key, value in owned.items()))
for warning in plan["warnings"]:
    print("warning:", warning)
EOF

echo "== 2/5 Explicit CUDA smoke test =="
# CUDA_VISIBLE_DEVICES maps the selected physical GPU to logical cuda:0.
"$NN" env --gpu-smoke --gpu-device 0

step=3
for variant in "${VARIANTS[@]}"; do
  echo "== $step/5 Train variant $variant =="
  step=$((step + 1))
  if [[ -f "$RUN_ROOT/$variant/result.json" ]]; then
    echo "skip completed $variant: $RUN_ROOT/$variant/result.json"
    continue
  fi
  # The epoch lines and the final JSON go to the terminal and to <variant>.log; the measures are in
  # <variant>/result.json.
  FROM=()
  if [[ -n "$START" && -f "$START/$variant/$variant-best.pt" && -f "$START/$variant/result.json" ]]; then
    FROM=(--init-from "$START/$variant/$variant-best.pt")
  elif [[ -n "$START" ]]; then
    echo "the start has no finished $variant: it starts from ${NNSLR_TRAIN_INIT:-imagenet}"
  fi
  "$NN" experiment-b --variant "$variant" --dataset-kind "$DATASET_KIND" --device cuda \
    --epochs "$EPOCHS" --output "$RUN_ROOT/$variant" "${INIT[@]}" "${FROM[@]}" 2>&1 | tee "$RUN_ROOT/$variant.log"
done

echo
echo "== Comparison (test; validation chose the best epoch) =="
"$VENV/bin/python" -c 'import sys; from nnslr_tools.context_experiment import comparison_table; print(comparison_table(sys.argv[1]), end="")' \
  "$RUN_ROOT" | tee "$RUN_ROOT/comparison.txt"
echo
echo "Experiment B complete: $RUN_ROOT"
echo "Nothing was imported, deployed or published."
