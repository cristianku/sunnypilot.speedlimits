#!/usr/bin/env bash
set -euo pipefail

# The road-type network with memory (docs/NEURAL_NETWORK.md, "The road type with memory"): the kind of road from the
# camera, one frame a second (town, rural, expressway, motorway, ramp, parking), with a GRU that remembers the seconds
# before and how long ago a motorway sign was seen, so that the 60 of a ramp ends where the ramp merges into the
# motorway (no sign ends it there: the comma's rule, vision.SpeedLimitMemory) and the entry into and exit from a town
# are seen. Its junction head says when the car is inside an intersection and when it has just left one: in Poland
# (and in Switzerland by OSStr art. 16 cpv. 2) a speed sign holds at most to the end of the next intersection, and
# Poland has no end sign. The car's own motion (the turn of every second, the steering wheel, the blinkers, from its
# logs; never the map) tells its memory when the car turned at an intersection.
# On one GPU, one step after the other:
#
#   1. checks the road types, the split and the whole pipeline on the CPU (nnslr train-road-type --dry-run): it stops
#      here while too few frames are confirmed in Data › Roads (the labels are a person's, never the map's proposals);
#   2. a CUDA smoke test;
#   3. finds the intersections every drive passed from its GPS and the map (nnslr junctions --all,
#      data/derived/junctions: the labels of the junction head, a person's corrections over them) and resizes every
#      frame of the drives with road types to 256 x 144 once (data/derived/road_type_frames);
#   4. reads the motorway signs of those drives as the comma does (nnslr road-type-signs: the car's vision.py,
#      SignNet and classifier from the sunnypilot clone, read only, 4 frames a second of fcamera.hevc, decoded and
#      read on the GPU (NVDEC and onnxruntime-gpu of .venv, the CPU when there is none);
#      data/derived/road_type_signs; NNSLR_ROAD_TYPE_CLASSIFIER: another classifier's reader-best.pt instead of the
#      car's, exported once to ONNX): the comma's rule measured with the signs the car confirms today, and the green
#      boards towards a motorway it confirms (the board input; never a limit); and the car's
#      own motion of every second from the drives' rlogs (nnslr car-motion --all: deviceMotion/livePose yaw rate,
#      carState steering angle, blinkers and vEgo; data/derived/car_motion);
#   5. trains the network with memory, the sign input and the car's motion (nnslr train-road-type --memory gru; the
#      sign input is the seconds since a motorway sign, placed at the start of every entry ramp of the map:
#      --sign-source map; the speed input is carState.vEgo; the board input when the classifier read the boards on
#      enough drives, NNSLR_ROAD_TYPE_BOARDS); the road type's epoch is chosen on its own measure, then the junction
#      head gets a stage of its own;
#   6. the same without the car's motion, to compare (--no-motion);
#   7. the same with the sign input but without memory, to compare (--memory none);
#   8. compares them (comparison.txt: the comma's rule on every merge of validation and test, when the ramp's limit
#      ends; F1 per class and the ramp's confusion; the changes of road type; false switches per hour; the first
#      minute after a merge; the junctions: recall of "passed" within 2 and 4 s, false "passed" per hour and per
#      10 km in town and outside, the intersections the car needed; the car's turn alone as a plain rule), with
#      earlier versions too when NNSLR_ROAD_TYPE_BASELINE names their checkpoints (separated by ":"), and exports the
#      network with memory, the sign and the motion as one ONNX step with its contract (nnslr export-road-type).
# Every training has the junction head when step 3 found the intersections (NNSLR_ROAD_TYPE_JUNCTIONS=0: without it,
# as the second version).
#
# usage: scripts/run_road_type.sh NAME
#
# Writes only into $DATA_ROOT/runs/NAME/ (road-type.log, road-type/{dry-run,gru,gru-nomotion,frame,baseline*,export}/,
# road-type/comparison.txt), the junction labels, the frame cache, the sign readings and the car's motion. It never imports reviews, changes the road types,
# the model in use or the car, never publishes. Run it again with the same NAME to finish: a finished training is not
# repeated (one trained before the labels were a person's alone stops it: use another NAME), the signs already read are
# not read again, an export older than its weights is redone.
#
# Environment: NNSLR_GPU_INDEX (default 0), NNSLR_GPU_VENV (.venv-gpu), NNSLR_ROAD_TYPE_EPOCHS (60),
# NNSLR_ROAD_TYPE_PATIENCE (15), NNSLR_ROAD_TYPE_BATCH (12 sequences), NNSLR_NUM_WORKERS (half the cores, 4 to 16),
# NNSLR_SIGN_WORKERS (processes reading the signs and the car's motion: half the cores, each with its ffmpeg), NNSLR_SUNNYPILOT_CLONE (/root/sunnypilot),
# NNSLR_ROAD_TYPE_CLASSIFIER (a classifier's reader-best.pt to read the signs with instead of the car's: one that reads
# the green boards before it is on the car), NNSLR_ROAD_TYPE_BOARDS (1: with the board input, refused when the
# classifier read too few boards; 0: without; empty, the default: with it when it read them on enough drives),
# NNSLR_ROAD_TYPE_BASELINE (earlier road-type-best.pt to compare with, separated by ":"), NNSLR_ROAD_TYPE_INIT_FROM (the
# road-type folder of an earlier run: each network goes on from its own there, one it lacks from ImageNet; Train ›
# Scene › Start from; a run before 4 Nov 2026 brings what it learned from the map's proposals, as the log says), NNSLR_ALLOW_BUSY_GPU=1 to share a GPU that already has compute processes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
RUN_ID="${1:-}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
EPOCHS="${NNSLR_ROAD_TYPE_EPOCHS:-60}"
PATIENCE="${NNSLR_ROAD_TYPE_PATIENCE:-15}"
BATCH="${NNSLR_ROAD_TYPE_BATCH:-12}"
CORES="$(nproc 2>/dev/null || echo 8)"
WORKERS_DEFAULT=$(( CORES / 2 < 4 ? 4 : (CORES / 2 > 16 ? 16 : CORES / 2) ))
NUM_WORKERS="${NNSLR_NUM_WORKERS:-$WORKERS_DEFAULT}"
SIGN_WORKERS_DEFAULT=$(( CORES / 2 > 1 ? CORES / 2 : 1 ))
SIGN_WORKERS="${NNSLR_SIGN_WORKERS:-$SIGN_WORKERS_DEFAULT}"
BASELINE="${NNSLR_ROAD_TYPE_BASELINE:-}"
JUNCTIONS="${NNSLR_ROAD_TYPE_JUNCTIONS:-1}"
START="${NNSLR_ROAD_TYPE_INIT_FROM:-}"
CLASSIFIER="${NNSLR_ROAD_TYPE_CLASSIFIER:-}"
BOARDS="${NNSLR_ROAD_TYPE_BOARDS:-}"

if [[ -z "$RUN_ID" || $# -ne 1 ]]; then
  echo "usage: $0 NAME" >&2
  echo "example: NNSLR_GPU_INDEX=0 $0 2026-10-29_1200-road-type" >&2
  exit 2
fi

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: NAME contains unsupported characters" >&2
  exit 2
fi

BASELINES=()
if [[ -n "$BASELINE" ]]; then
  IFS=':' read -r -a BASELINES <<< "$BASELINE"
fi
for checkpoint in "${BASELINES[@]}"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "error: NNSLR_ROAD_TYPE_BASELINE is not a file: $checkpoint" >&2
    exit 2
  fi
done

if [[ -n "$START" && ! -d "$START" ]]; then
  echo "error: NNSLR_ROAD_TYPE_INIT_FROM is not a folder: $START" >&2
  exit 2
fi

if [[ -n "$CLASSIFIER" && ! -f "$CLASSIFIER" ]]; then
  echo "error: NNSLR_ROAD_TYPE_CLASSIFIER is not a file: $CLASSIFIER" >&2
  exit 2
fi

if [[ -n "$BOARDS" && "$BOARDS" != "0" && "$BOARDS" != "1" ]]; then
  echo "error: NNSLR_ROAD_TYPE_BOARDS must be 0, 1 or empty: $BOARDS" >&2
  exit 2
fi

for setting in EPOCHS PATIENCE BATCH NUM_WORKERS SIGN_WORKERS; do
  if [[ ! "${!setting}" =~ ^[0-9]+$ ]]; then
    echo "error: $setting must be a whole number: ${!setting}" >&2
    exit 2
  fi
done

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
  echo "error: selected GPU already has compute processes; refusing to train the road type" >&2
  echo "$BUSY" >&2
  echo "Set NNSLR_ALLOW_BUSY_GPU=1 only after intentionally accepting contention." >&2
  exit 3
fi

RUN_ROOT="$DATA_ROOT/runs/$RUN_ID"
OUT="$RUN_ROOT/road-type"
# A folder of another kind of run is never written into.
for part in detector reader export bundle experiment-b detector-sizes classifier; do
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

mkdir -p "$OUT"
# Everything printed from here on also goes to the run's own log (the panel reads its steps and epochs).
exec > >(tee -a "$RUN_ROOT/road-type.log") 2>&1

export NNSLR_DATA_ROOT="$DATA_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
NN="$VENV/bin/nnslr"
SETTINGS=(--epochs "$EPOCHS" --patience "$PATIENCE" --batch-size "$BATCH" --num-workers "$NUM_WORKERS")
if [[ "$JUNCTIONS" == "0" ]]; then
  SETTINGS+=(--no-junctions)
fi
if [[ "$BOARDS" == "1" ]]; then
  SETTINGS+=(--boards)
elif [[ "$BOARDS" == "0" ]]; then
  SETTINGS+=(--no-boards)
fi

echo "== Road type with memory =="
echo "run:          $RUN_ID"
echo "output:       $OUT"
echo "data root:    $DATA_ROOT"
echo "physical GPU: $GPU_INDEX"
echo "settings:     ${SETTINGS[*]}"
echo "baseline:     ${BASELINE:-none}"
if [[ -n "$START" ]]; then
  echo "init:         $(basename "$(dirname "$START")")"
fi
echo "junctions:    $([[ "$JUNCTIONS" == "0" ]] && echo "no head" || echo "with the junction head")"
echo "signs read:   ${CLASSIFIER:-by the classifier of the car}"
echo "boards:       $([[ "$BOARDS" == "1" ]] && echo "with the board input" || ([[ "$BOARDS" == "0" ]] && echo "no board input" || echo "the board input when the classifier read the boards"))"
echo

echo "== 1/8 Check the road types, the split and the pipeline (CPU-only) =="
CUDA_VISIBLE_DEVICES="" "$NN" train-road-type --dry-run --run-dir "$OUT/dry-run" --num-workers "$NUM_WORKERS"

echo "== 2/8 Explicit CUDA smoke test =="
"$NN" env --gpu-smoke --gpu-device 0

echo "== 3/8 Find the junctions, resize the frames to 256x144 =="
if [[ "$JUNCTIONS" != "0" ]]; then
  "$NN" junctions --all --data-root "$DATA_ROOT"
fi
"$VENV/bin/python" -c '
import sys
from nnslr_tools import road_type_data as data
dataset = data.load_dataset(sys.argv[1])
paths = data.build_frame_cache(sys.argv[1], dataset["frames"], workers=int(sys.argv[2]))
print(f"frame cache: {len(paths)} drives ready")
' "$DATA_ROOT" "$NUM_WORKERS"

echo "== 4/8 Read the motorway signs as the car does (GPU) and the car's motion (CPU) =="
# The signs on the GPU: .venv has onnxruntime-gpu, .venv-gpu only the CPU's (torch brings its own CUDA).
SIGNS_NN="$ROOT_DIR/.venv/bin/nnslr"
[ -x "$SIGNS_NN" ] || SIGNS_NN="$NN"
SIGN_ARGS=()
if [[ -n "$CLASSIFIER" ]]; then
  # Exported to ONNX once here, with torch: the reading in .venv then finds it.
  "$VENV/bin/python" -c '
import sys
from pathlib import Path
from nnslr_tools.road_type_signs import classifier_onnx
print("classifier as ONNX:", classifier_onnx(Path(sys.argv[1]), Path(sys.argv[2])))
' "$DATA_ROOT" "$CLASSIFIER"
  SIGN_ARGS=(--classifier "$CLASSIFIER")
fi
"$SIGNS_NN" road-type-signs --num-workers "$SIGN_WORKERS" --data-root "$DATA_ROOT" "${SIGN_ARGS[@]}"
"$NN" car-motion --all --num-workers "$SIGN_WORKERS"

train() {
  local memory="$1" folder="$2"
  shift 2
  if [[ -f "$OUT/$folder/result.json" ]]; then
    # Finished before the labels were a person's alone: not a step of this run, beside networks trained after it.
    if ! grep -q '"label_policy": "confirmed-only"' "$OUT/$folder/result.json"; then
      echo "error: $OUT/$folder was trained on the map's proposals: start a new run, with another NAME" >&2
      exit 2
    fi
    echo "skip trained: $OUT/$folder/result.json"
    return
  fi
  # Each network goes on from its own of the start, when the start has it finished.
  local from=()
  if [[ -n "$START" && -f "$START/$folder/road-type-best.pt" && -f "$START/$folder/result.json" ]]; then
    from=(--init-from "$START/$folder/road-type-best.pt")
  elif [[ -n "$START" ]]; then
    echo "the start has no finished $folder: it starts from ImageNet"
  fi
  "$NN" train-road-type --memory "$memory" --run-dir "$OUT/$folder" "${SETTINGS[@]}" "${from[@]}" "$@"
}

echo "== 5/8 Train with memory, the sign and the car's motion (GRU) =="
train gru gru

echo "== 6/8 Train with memory, without the car's motion =="
train gru gru-nomotion --no-motion

echo "== 7/8 Train without memory =="
train none frame

echo "== 8/8 Compare and export the ONNX step =="
for index in "${!BASELINES[@]}"; do
  folder="baseline$([[ "$index" == "0" ]] || echo "-$(( index + 1 ))")"
  if [[ -f "$OUT/$folder/release.json" && "$OUT/$folder/release.json" -nt "${BASELINES[$index]}" ]]; then
    echo "skip measured: $OUT/$folder/release.json"
  else
    "$NN" release-road-type --checkpoint "${BASELINES[$index]}" --output "$OUT/$folder/release.json"
  fi
done
"$NN" compare-road-type --run-root "$RUN_ROOT"
ONNX="$OUT/export/road-type.onnx"
if [[ -f "$ONNX" && -f "$ONNX.contract.json" && "$ONNX" -nt "$OUT/gru/road-type-best.pt" ]]; then
  echo "skip exported: $ONNX"
else
  "$NN" export-road-type --checkpoint "$OUT/gru/road-type-best.pt" --output "$ONNX"
fi
echo
echo "Road type complete: $OUT"
echo "Nothing was imported, deployed or published, and nothing on the car changed."
